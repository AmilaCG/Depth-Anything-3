import argparse
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from gsplat import rasterization


def _parse_indices(indices_str: str, max_views: int) -> list[int]:
    vals = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
    if not vals:
        raise ValueError("--indices is empty. Provide comma-separated frame indices.")
    bad = [i for i in vals if i < 0 or i >= max_views]
    if bad:
        raise ValueError(f"Indices out of range [0, {max_views - 1}]: {bad}")
    return vals


def _load_ply_gaussians(ply_path: str, device: torch.device):
    vertex = PlyData.read(ply_path)["vertex"]

    means = torch.from_numpy(
        np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    ).to(device)
    scales = torch.from_numpy(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
    ).to(device)
    quats = torch.from_numpy(
        np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)
    ).to(device)
    opacities = torch.from_numpy(np.asarray(vertex["opacity"], dtype=np.float32)).to(device)

    # Exported .ply stores log-scales and SH DC coefficients (f_dc_*).
    scales = scales.exp()
    # Export path stores inverse-sigmoid opacity for 3DGS compatibility.
    opacities = opacities.sigmoid()
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    sh_dc = torch.from_numpy(
        np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
    ).to(device)
    colors = sh_dc[:, None, :]

    return means, quats, scales, opacities, colors


def _save_images(rgb_frames: torch.Tensor, out_dir: str, frame_indices: Iterable[int]):
    os.makedirs(out_dir, exist_ok=True)
    for frame, idx in zip(rgb_frames, frame_indices):
        frame_u8 = frame.clamp(0, 1).mul(255).byte().cpu().numpy()
        Image.fromarray(frame_u8, mode="RGB").save(os.path.join(out_dir, f"{idx:06d}.png"))


def main():
    parser = argparse.ArgumentParser(description="Render selected views from exported DA3 .ply using gsplat")
    parser.add_argument("--ply", required=True, help="Path to exported .ply (e.g. outputs/.../gs_ply/0000.ply)")
    parser.add_argument(
        "--camera",
        required=True,
        help="Path to exported camera params .npz (e.g. outputs/.../images/0000/camera_params.npz)",
    )
    parser.add_argument(
        "--indices",
        required=True,
        help="Comma-separated frame indices to render, e.g. 0,8,16,24",
    )
    parser.add_argument("--out", required=True, help="Output directory for rendered PNGs")
    parser.add_argument("--device", default="cuda", help="Device for rendering (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    cam = np.load(args.camera)
    viewmats = torch.from_numpy(cam["viewmats"]).float().to(device)
    Ks = torch.from_numpy(cam["Ks"]).float().to(device)
    H = int(cam["height"])
    W = int(cam["width"])

    frame_indices = _parse_indices(args.indices, max_views=viewmats.shape[0])
    sel_viewmats = viewmats[frame_indices]
    sel_Ks = Ks[frame_indices]

    means, quats, scales, opacities, colors = _load_ply_gaussians(args.ply, device)
    backgrounds = torch.zeros((len(frame_indices), 3), device=device)

    render_colors, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=sel_viewmats,
        Ks=sel_Ks,
        backgrounds=backgrounds,
        render_mode="RGB+D",
        width=W,
        height=H,
        packed=False,
        sh_degree=0,
    )

    rgb = render_colors[..., :3]
    _save_images(rgb, args.out, frame_indices)
    print(f"Saved {len(frame_indices)} frame(s) to {args.out}")


if __name__ == "__main__":
    main()
