import argparse
import glob
import os

import torch
from depth_anything_3.api import DepthAnything3

def parse_args():
    parser = argparse.ArgumentParser(description="Run DA3 image dump/export job.")
    parser.add_argument(
        "--dump-images",
        action="store_true",
        help="Enable exporting rendered images.",
    )
    parser.add_argument(
        "--dump-ply",
        action="store_true",
        help="Enable exporting Gaussian ply.",
    )
    parser.add_argument(
        "--dump-cameras",
        action="store_true",
        help="Enable exporting camera poses.",
    )
    parser.add_argument(
        "--test-views",
        action="store_true",
        help="Infer using subset of input views (skip 0, 8, 16, ...)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=torch.device("cuda"))

    dataset_name = "kitti360"
    # dataset_name = "matrixcity"
    # dataset_name = "physicalai"

    dataset_path = f"/home/amila/datasets/{dataset_name}/images"
    out_path = f"/home/amila/Depth-Anything-3/outputs/{dataset_name}"
    # images = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
    images = sorted(glob.glob(os.path.join(dataset_path, "*.png")))

    export_args = {
        "gs_video": {
            "vis_depth": None,
            "trj_mode": "original",
            "out_path": out_path,
            "export_images": args.dump_images,
            "export_camera_poses": args.dump_cameras,
        }
    }

    prediction = model.inference(
        image = [img for i, img in enumerate(images) if i not in {0, 8, 16, 24, 32, 40, 48, 56}] if args.test_views else images,
        infer_gs=True,
        export_dir=out_path,
        export_format="gs_ply" if args.dump_ply else "gs_video",
        process_res=448,
        export_kwargs=export_args,
    )

    # prediction.processed_images : [N, H, W, 3] uint8   array
    print(f"Processed images: {prediction.processed_images.shape}")

if __name__ == "__main__":
    main()
