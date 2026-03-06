import argparse
import os
from math import isqrt
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
    ).to(device)  # [N, 3]

    # Reconstruct SH layout expected by gsplat: [N, K, 3], where K=(degree+1)^2.
    rest_names = [name for name in vertex.data.dtype.names if name.startswith("f_rest_")]
    rest_names = sorted(rest_names, key=lambda x: int(x.split("_")[-1]))
    if rest_names:
        sh_rest_flat = torch.from_numpy(
            np.stack([vertex[name] for name in rest_names], axis=1).astype(np.float32)
        ).to(device)  # [N, 3*(K-1)]
        if sh_rest_flat.shape[1] % 3 != 0:
            raise ValueError(
                f"Invalid f_rest size in {ply_path}: {sh_rest_flat.shape[1]} (must be divisible by 3)."
            )
        sh_rest = sh_rest_flat.view(sh_rest_flat.shape[0], 3, -1)  # [N, 3, K-1]
        sh_full = torch.cat([sh_dc.unsqueeze(-1), sh_rest], dim=-1)  # [N, 3, K]
    else:
        sh_full = sh_dc.unsqueeze(-1)  # [N, 3, 1]

    colors = sh_full.permute(0, 2, 1).contiguous()  # [N, K, 3]
    n_coeff = colors.shape[1]
    degree = isqrt(n_coeff) - 1
    if (degree + 1) ** 2 != n_coeff:
        raise ValueError(
            f"Invalid SH coefficient count in {ply_path}: {n_coeff} (must be a perfect square)."
        )

    return means, quats, scales, opacities, colors, degree


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
    args = parser.parse_args()

    device = torch.device("cuda")

    cam = np.load(args.camera)
    viewmats = torch.from_numpy(cam["viewmats"]).float().to(device)
    Ks = torch.from_numpy(cam["Ks"]).float().to(device)
    H = int(cam["height"])
    W = int(cam["width"])

    frame_indices = _parse_indices(args.indices, max_views=viewmats.shape[0])
    sel_viewmats = viewmats[frame_indices]
    sel_Ks = Ks[frame_indices]

    means, quats, scales, opacities, colors, sh_degree = _load_ply_gaussians(args.ply, device)
    backgrounds = torch.zeros((len(frame_indices), 3), device=device)

    # print(f"means: {means[:4]}, quats: {quats[:4]}, scales: {scales[:4]}")
    print(f"Rendering {means.shape[0]} Gaussians...")
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
        sh_degree=sh_degree,
    )

    rgb = render_colors[..., :3]
    _save_images(rgb, args.out, frame_indices)
    print(f"Saved {len(frame_indices)} frame(s) to {args.out}")


if __name__ == "__main__":
    main()
