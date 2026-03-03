import argparse
from pathlib import Path

import cv2
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
        "--gt-dir",
        type=Path,
        default=None,
        help="Optional override for GT image directory.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Optional override for prediction image directory.",
    )
    return parser.parse_args()


def list_image_paths(folder: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ],
        key=lambda p: p.name,
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_gt_to_pred(gt_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
    pred_h, pred_w = pred_img.shape[:2]
    # Use area-based downsampling for fair evaluation when GT is higher resolution.
    return cv2.resize(gt_img, (pred_w, pred_h), interpolation=cv2.INTER_AREA)


def to_lpips_tensor(img_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    img = img_uint8.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)


def main() -> None:
    args = parse_args()
    gt_dir = args.gt_dir or Path(f"/home/amila/datasets/{args.name}/images/")
    pred_dir = args.pred_dir or Path(
        f"/home/amila/Depth-Anything-3/outputs/{args.name}/gsplat_renders/"
    )

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    gt_paths = list_image_paths(gt_dir)
    pred_paths = list_image_paths(pred_dir)

    if not gt_paths:
        raise RuntimeError(f"No valid GT images found in: {gt_dir}")
    if not pred_paths:
        raise RuntimeError(f"No valid prediction images found in: {pred_dir}")

    gt_by_name = {p.name: p for p in gt_paths}
    pred_by_name = {p.name: p for p in pred_paths}
    common_names = sorted(set(gt_by_name) & set(pred_by_name))

    if not common_names:
        raise RuntimeError(
            "No matching filenames between GT and predictions. "
            f"gt_dir={gt_dir}, pred_dir={pred_dir}"
        )

    missing_in_pred = sorted(set(gt_by_name) - set(pred_by_name))
    missing_in_gt = sorted(set(pred_by_name) - set(gt_by_name))
    if missing_in_pred:
        print(
            f"[warn] {len(missing_in_pred)} GT file(s) have no matching prediction; "
            "they will be skipped."
        )
    if missing_in_gt:
        print(
            f"[warn] {len(missing_in_gt)} prediction file(s) have no matching GT; "
            "they will be skipped."
        )

    device = torch.device("cuda")
    lpips_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()

    rows: list[dict[str, float | str]] = []
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips_vgg = 0.0
    sum_lpips_alex = 0.0

    pair_count = len(common_names)
    for idx, name in enumerate(common_names, start=1):
        gt_path = gt_by_name[name]
        pred_path = pred_by_name[name]
        gt_img = load_rgb(gt_path)
        pred_img = load_rgb(pred_path)

        if gt_img.shape != pred_img.shape:
            gt_img = resize_gt_to_pred(gt_img, pred_img)

        if gt_img.shape != pred_img.shape:
            raise ValueError(
                "Shape mismatch after GT resize: "
                f"gt_file={gt_path.name}, pred_file={pred_path.name}, "
                f"gt={gt_img.shape}, pred={pred_img.shape}"
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
                "image": f"pair_{idx:06d}",
                "gt_image": gt_path.name,
                "pred_image": pred_path.name,
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
            f"[{idx}/{pair_count}] gt={gt_path.name} pred={pred_path.name} | "
            f"PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS-VGG={lp_vgg:.4f} LPIPS-Alex={lp_alex:.4f}"
        )

    n = len(rows)
    avg_psnr = sum_psnr / n
    avg_ssim = sum_ssim / n
    avg_lpips_vgg = sum_lpips_vgg / n
    avg_lpips_alex = sum_lpips_alex / n

    print("\n=== Average Metrics ===")
    print(f"Pairs evaluated: {n}")
    print(f"PSNR       : {avg_psnr:.6f}")
    print(f"SSIM       : {avg_ssim:.6f}")
    print(f"LPIPS-VGG  : {avg_lpips_vgg:.6f}")
    print(f"LPIPS-Alex : {avg_lpips_alex:.6f}")

if __name__ == "__main__":
    main()
