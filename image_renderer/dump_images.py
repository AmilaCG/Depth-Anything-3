import argparse
import glob
import os

import torch
from depth_anything_3.api import DepthAnything3
import json
import numpy as np

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

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
    parser.add_argument(
        "--camera-npz",
        type=str,
        default=None,
        help="Path to exported camera poses .npz from gs_renderer.py (keys: viewmats, Ks).",
    )
    parser.add_argument(
        "--camera-npz-native",
        type=str,
        default=None,
        help="Path to exported camera poses .npz from gs_renderer.py (keys: intrinsics, extrinsics).",
    )
    return parser.parse_args()

def load_camera_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    k_mat = np.array(
        [
            [data["fl_x"], 0.0, data["cx"]],
            [0.0, data["fl_y"], data["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    w2c_dict = {}
    for frame in data["frames"]:
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w_cv = c2w_gl @ OPENGL_TO_OPENCV
        w2c = np.linalg.inv(c2w_cv).astype(np.float32)

        file_name = frame["file_path"].split("/")[-1]  # Extract file name from path
        w2c_dict[file_name] = w2c # Example_w2c = w2c_dict['000063.png']

    return k_mat.copy(), w2c_dict

def load_camera_npz(npz_path):
    cam = np.load(npz_path)
    if "viewmats" not in cam or "Ks" not in cam:
        raise KeyError(
            f"{npz_path} must contain 'viewmats' and 'Ks'. "
            f"Found keys: {list(cam.keys())}"
        )

    viewmats = np.asarray(cam["viewmats"], dtype=np.float32)
    intrinsics = np.asarray(cam["Ks"], dtype=np.float32)

    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError(f"Invalid viewmats shape {viewmats.shape}; expected [N, 4, 4].")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Invalid Ks shape {intrinsics.shape}; expected [N, 3, 3].")
    if viewmats.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            f"viewmats/Ks length mismatch: {viewmats.shape[0]} vs {intrinsics.shape[0]}"
        )

    return intrinsics.copy(), viewmats.copy()

def load_camera_npz_native(npz_path):
    cam = np.load(npz_path)

    if "intrinsics" not in cam or "extrinsics" not in cam:
        raise KeyError(
            f"{npz_path} must contain 'intrinsics' and 'extrinsics'. "
            f"Found keys: {list(cam.keys())}"
        )

    intrinsics = np.asarray(cam["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(cam["extrinsics"], dtype=np.float32)

    # Native export may store a batch dimension: [B, N, ...].
    if intrinsics.ndim == 4:
        if intrinsics.shape[0] != 1:
            raise ValueError(
                "Native camera npz intrinsics contain multiple batches. "
                "Please provide a single-batch file for dump_images.py."
            )
        intrinsics = intrinsics[0]
    if extrinsics.ndim == 4:
        if extrinsics.shape[0] != 1:
            raise ValueError(
                "Native camera npz extrinsics contain multiple batches. "
                "Please provide a single-batch file for dump_images.py."
            )
        extrinsics = extrinsics[0]

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(
            f"Invalid intrinsics shape {intrinsics.shape}; expected [N, 3, 3]."
        )

    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(
            f"Invalid extrinsics shape {extrinsics.shape}; expected [N, 3, 4] or [N, 4, 4]."
        )

    # Support native exporters that store 3x4 world-to-camera matrices.
    if extrinsics.shape[1:] == (3, 4):
        bottom_row = np.zeros((extrinsics.shape[0], 1, 4), dtype=extrinsics.dtype)
        bottom_row[:, 0, 3] = 1.0
        extrinsics = np.concatenate([extrinsics, bottom_row], axis=1)

    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            f"intrinsics/extrinsics length mismatch: {intrinsics.shape[0]} vs {extrinsics.shape[0]}"
        )

    return intrinsics.copy(), extrinsics.copy()

def main():
    args = parse_args()

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=torch.device("cuda"))

    dataset_name = "kitti360"
    # dataset_name = "matrixcity"
    # dataset_name = "physicalai"

    dataset_path = f"/home/amila/datasets/{dataset_name}"
    out_path = f"/home/amila/Depth-Anything-3/outputs/{dataset_name}"
    # images = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
    images = sorted(glob.glob(os.path.join(dataset_path, "images", "*.png")))

    export_args = {
        "gs_video": {
            "vis_depth": None,
            "trj_mode": "original",
            "out_path": out_path,
            "export_images": args.dump_images,
            "export_camera_poses": args.dump_cameras,
        }
    }

    # selected_images = images[:5]
    selected_indices = (
        [i for i in range(len(images)) if i not in {0, 8, 16, 24, 32, 40, 48, 56}]
        if args.test_views
        else list(range(len(images)))
    )
    selected_images = [images[i] for i in selected_indices]

    if args.camera_npz:
        intrinsics_all, extrinsics_all = load_camera_npz(args.camera_npz)
        if extrinsics_all.shape[0] == len(images):
            cam_indices = selected_indices
        elif extrinsics_all.shape[0] == len(selected_images):
            cam_indices = list(range(len(selected_images)))
        else:
            raise ValueError(
                "Camera count in npz does not match full or selected image counts: "
                f"{extrinsics_all.shape[0]} vs {len(images)} or {len(selected_images)}."
            )
        extrinsics = extrinsics_all[cam_indices]
        intrinsics = intrinsics_all[cam_indices]
        print(f"Loaded cameras from npz: {args.camera_npz}")
    elif args.camera_npz_native:
        intrinsics_all, extrinsics_all = load_camera_npz_native(args.camera_npz_native)
        if extrinsics_all.shape[0] == len(images):
            cam_indices = selected_indices
        elif extrinsics_all.shape[0] == len(selected_images):
            cam_indices = list(range(len(selected_images)))
        else:
            raise ValueError(
                "Camera count in native npz does not match full or selected image counts: "
                f"{extrinsics_all.shape[0]} vs {len(images)} or {len(selected_images)}."
            )
        extrinsics = extrinsics_all[cam_indices]
        intrinsics = intrinsics_all[cam_indices]
        print(f"Loaded native cameras from npz: {args.camera_npz_native}")
    else:
        # k_mat, w2c_mats = load_camera_json(f"{dataset_path}/transforms.json")
        # selected_names = [os.path.basename(p) for p in selected_images]
        # print(f"Selected names: {selected_names}")
        # extrinsics = np.stack([w2c_mats[name] for name in selected_names], axis=0)
        # intrinsics = np.stack([k_mat.copy() for _ in selected_names], axis=0)
        extrinsics = None
        intrinsics = None

    prediction = model.inference(
        image=selected_images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        infer_gs=True,
        export_dir=out_path,
        export_format="gs_ply" if args.dump_ply else "gs_video",
        process_res=448,
        export_kwargs=export_args,
        align_to_input_ext_scale=False,
    )

    # prediction.processed_images : [N, H, W, 3] uint8   array
    print(f"Processed images: {prediction.processed_images.shape}")

if __name__ == "__main__":
    main()
