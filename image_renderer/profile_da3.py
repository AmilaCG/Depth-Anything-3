import os, glob, time
import torch
from depth_anything_3.api import DepthAnything3

MODEL_DIR = "/home/amila/DA3NESTED-GIANT-LARGE"

dataset_name = "kitti360"
# dataset_name = "matrixcity"
# dataset_name = "physicalai"
resolution = 448
# resolution = 796

test_views = True  # if True, skip every 8th frame (0, 8, 16, ...)

IMG_DIR   = f"/home/amila/datasets/{dataset_name}/images"
N         = 1        # number of images to test
WARMUP    = 3        # warmup runs (exclude from stats)

def mib(x):  # bytes -> MiB
    return x / (1024**2)

def profile_once(model, paths):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # torch.cuda.empty_cache()  # optional; comment out if you want steady-state behavior
    torch.cuda.synchronize()

    # Wall-clock timing (end-to-end)
    t0 = time.perf_counter()
    pred = model.inference(
        image=paths,
        infer_gs=True,
        process_res=resolution,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0

    # VRAM stats
    alloc      = torch.cuda.memory_allocated()
    reserved   = torch.cuda.memory_reserved()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv  = torch.cuda.max_memory_reserved()

    print(f"\n=== Profile ({len(paths)} image(s)) ===")
    print(f"Wall time: {wall_ms:.3f} ms   ({wall_ms/len(paths):.3f} ms/img)")
    print(f"VRAM allocated:      {mib(alloc):.1f} MiB")
    print(f"VRAM reserved:       {mib(reserved):.1f} MiB")
    print(f"VRAM peak allocated: {mib(peak_alloc):.1f} MiB")
    print(f"VRAM peak reserved:  {mib(peak_resv):.1f} MiB")

    return pred

def main():
    device = torch.device("cuda")

    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3NESTED-GIANT-LARGE"
    ).to(device=device)
    model.eval()

    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    # images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not images:
        raise SystemExit(f"No images found in {IMG_DIR}")

    # paths = images[:N]
    # paths = images
    paths = [img for i, img in enumerate(images) if i not in {0, 8, 16, 24, 32, 40, 48, 56}] if test_views else images
    print("Using:", paths)

    # Warmup (ignore results)
    for _ in range(WARMUP):
        _ = model.inference(
        image=paths,
        infer_gs=True,
        process_res=resolution,
    )
    torch.cuda.synchronize()

    # Profile
    _ = profile_once(model, paths)

if __name__ == "__main__":
    main()
