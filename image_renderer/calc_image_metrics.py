import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import lpips
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'lpips'. Install it with: pip install lpips"
    ) from exc

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'scikit-image'. Install it with: pip install scikit-image"
    ) from exc


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS-VGG and LPIPS-Alex for paired images."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="kitti360",
        help="Dataset name.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output CSV path for per-image metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LPIPS (cuda or cpu).",
    )
    return parser.parse_args()


def list_image_names(folder: Path) -> set[str]:
    return {
        p.name
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_lpips_tensor(img_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    img = img_uint8.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)


def main() -> None:
    args = parse_args()
    gt_dir = Path(f"/home/amila/datasets/{args.name}/images/")
    pred_dir = Path(f"/home/amila/Depth-Anything-3/outputs/{args.name}/images/0000/")

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    gt_names = list_image_names(gt_dir)
    pred_names = list_image_names(pred_dir)
    common_names = sorted(gt_names & pred_names)

    if not common_names:
        raise RuntimeError("No matching image filenames found between directories.")

    missing_in_pred = sorted(gt_names - pred_names)
    missing_in_gt = sorted(pred_names - gt_names)
    if missing_in_pred:
        print(f"[warn] {len(missing_in_pred)} GT files missing in pred dir.")
    if missing_in_gt:
        print(f"[warn] {len(missing_in_gt)} pred files missing in GT dir.")

    device = torch.device(args.device)
    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()

    rows: list[dict[str, float | str]] = []
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips_vgg = 0.0
    sum_lpips_alex = 0.0

    for idx, name in enumerate(common_names, start=1):
        gt_img = load_rgb(gt_dir / name)
        pred_img = load_rgb(pred_dir / name)

        if gt_img.shape != pred_img.shape:
            raise ValueError(
                f"Shape mismatch for {name}: gt={gt_img.shape}, pred={pred_img.shape}"
            )

        psnr = float(peak_signal_noise_ratio(gt_img, pred_img, data_range=255))
        ssim = float(
            structural_similarity(
                gt_img, pred_img, data_range=255, channel_axis=2
            )
        )

        with torch.no_grad():
            gt_t = to_lpips_tensor(gt_img, device)
            pred_t = to_lpips_tensor(pred_img, device)
            lp_vgg = float(lpips_vgg(pred_t, gt_t).item())
            lp_alex = float(lpips_alex(pred_t, gt_t).item())

        rows.append(
            {
                "image": name,
                "psnr": psnr,
                "ssim": ssim,
                "lpips_vgg": lp_vgg,
                "lpips_alex": lp_alex,
            }
        )

        sum_psnr += psnr
        sum_ssim += ssim
        sum_lpips_vgg += lp_vgg
        sum_lpips_alex += lp_alex

        print(
            f"[{idx}/{len(common_names)}] {name} | "
            f"PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS-VGG={lp_vgg:.4f} LPIPS-Alex={lp_alex:.4f}"
        )

    n = len(rows)
    mean_psnr = sum_psnr / n
    mean_ssim = sum_ssim / n
    mean_lpips_vgg = sum_lpips_vgg / n
    mean_lpips_alex = sum_lpips_alex / n

    print("\n=== Mean Metrics ===")
    print(f"Pairs evaluated: {n}")
    print(f"PSNR       : {mean_psnr:.6f}")
    print(f"SSIM       : {mean_ssim:.6f}")
    print(f"LPIPS-VGG  : {mean_lpips_vgg:.6f}")
    print(f"LPIPS-Alex : {mean_lpips_alex:.6f}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image", "psnr", "ssim", "lpips_vgg", "lpips_alex"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-image metrics written to: {args.csv_out}")


if __name__ == "__main__":
    main()
