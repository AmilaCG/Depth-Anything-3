import argparse
import glob
import os

import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import read_cameras_binary
from depth_anything_3.utils.read_write_model import read_images_binary
from depth_anything_3.utils.pose_align import align_poses_umeyama
import json
import numpy as np

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

def as_homogeneous(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.copy()
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            f"Invalid extrinsics shape {extrinsics.shape}; expected [..., 3, 4] or [..., 4, 4]."
        )

    bottom_row = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=extrinsics.dtype)
    bottom_row[..., 0, 3] = 1.0
    return np.concatenate([extrinsics, bottom_row], axis=-2)

def affine_inverse_np(extrinsics):
    extrinsics = as_homogeneous(extrinsics)
    rot = extrinsics[..., :3, :3]
    trans = extrinsics[..., :3, 3:]
    rot_t = np.swapaxes(rot, -1, -2)
    inv = np.zeros_like(extrinsics)
    inv[..., :3, :3] = rot_t
    inv[..., :3, 3:] = -(rot_t @ trans)
    inv[..., 3, 3] = 1.0
    return inv

def normalize_intrinsics(intrinsics, height, width):
    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    intrinsics[..., 0, :] /= float(width)
    intrinsics[..., 1, :] /= float(height)
    return intrinsics

def export_prediction_cameras_for_align_pose(prediction, image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    export_extrinsics = as_homogeneous(prediction.extrinsics)
    # Nested metric models rescale pose translations after GS creation. Undo that
    # so the exported poses match the exported Gaussian scene frame.
    if prediction.is_metric and prediction.scale_factor is not None:
        export_extrinsics = export_extrinsics.copy()
        export_extrinsics[:, :3, 3] /= float(prediction.scale_factor)

    export_intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    height, width = prediction.processed_images.shape[1:3]
    export_intrinsics = normalize_intrinsics(export_intrinsics, height=height, width=width)
    export_c2w = affine_inverse_np(export_extrinsics)

    frames = []
    for idx, image_path in enumerate(image_paths):
        frames.append(
            {
                "file_name": os.path.basename(image_path),
                "extrinsic_c2w": export_c2w[idx].tolist(),
                "intrinsic_3x3": export_intrinsics[idx].tolist(),
            }
        )

    poses_path = os.path.join(output_dir, "pred_cameras.json")
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, indent=2)

    raw_npz_path = os.path.join(output_dir, "prediction_camera_params.npz")
    np.savez_compressed(
        raw_npz_path,
        extrinsics=prediction.extrinsics,
        intrinsics=prediction.intrinsics,
    )
    print(f"Exported align-pose poses to: {poses_path}")
    print(f"Exported raw predicted camera params to: {raw_npz_path}")

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
    parser.add_argument(
        "--camera-json",
        action="store_true",
        help="Path to exported COLMAP camera poses bin.",
    )
    parser.add_argument(
        "--camera-bin",
        action="store_true",
        help="Path to exported COLMAP camera poses bin.",
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

def load_camera_bin_colmap(filepath, image_paths=None):
    cam_bin_path = f"{filepath}/cameras.bin"
    img_bin_path = f"{filepath}/images.bin"
    cams = read_cameras_binary(cam_bin_path)
    images = read_images_binary(img_bin_path)

    print(f"Loaded COLMAP cameras/images: {len(cams)} / {len(images)}")

    # COLMAP entries keyed by file basename for robust matching.
    colmap_by_name = {os.path.basename(image.name): image for image in images.values()}
    names = (
        [os.path.basename(path) for path in image_paths]
        if image_paths is not None
        else sorted(colmap_by_name.keys())
    )

    extrinsics = []
    intrinsics = []

    for name in names:
        if name not in colmap_by_name:
            raise KeyError(f"Image '{name}' not found in COLMAP images.bin")
        image = colmap_by_name[name]

        # Build extrinsics (world-to-camera)
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec

        # Get camera parameters
        cam_id = image.camera_id
        camera = cams[cam_id]

        # Build intrinsics from camera-model-specific parameter layout.
        params = camera.params
        model = camera.model
        ixt = np.eye(3, dtype=np.float32)

        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
            focal = float(params[0])
            ixt[0, 0] = focal
            ixt[1, 1] = focal
            ixt[0, 2] = float(params[1])
            ixt[1, 2] = float(params[2])
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV"}:
            ixt[0, 0] = float(params[0])
            ixt[1, 1] = float(params[1])
            ixt[0, 2] = float(params[2])
            ixt[1, 2] = float(params[3])
        else:
            raise NotImplementedError(
                f"COLMAP camera model '{model}' is not supported in dump_images.py"
            )

        # COLMAP convention adjustment used across DA3 dataset loaders.
        # ixt[:2, 2] -= 0.5

        extrinsics.append(ext)
        intrinsics.append(ixt)

    if not extrinsics:
        raise ValueError(f"No cameras were resolved from COLMAP model at '{filepath}'")

    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    print(
        f"Prepared COLMAP camera tensors: extrinsics {extrinsics.shape}, "
        f"intrinsics {intrinsics.shape}"
    )
    return intrinsics, extrinsics

def align_render_extrinsics_to_prediction(gt_extrinsics, pred_extrinsics, pred_scale_factor=None):
    pred_render_extrinsics = pred_extrinsics.copy()
    if pred_scale_factor is not None:
        pred_render_extrinsics[:, :3, 3] /= pred_scale_factor

    rot, trans, scale, render_exts = align_poses_umeyama(
        ext_ref=pred_render_extrinsics,
        ext_est=gt_extrinsics,
        return_aligned=True,
        # random_state=42,
    )

    return render_exts.astype(np.float32), rot, trans, scale

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
        gt_extrinsics = extrinsics_all[cam_indices]
        gt_intrinsics = intrinsics_all[cam_indices]
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
        gt_extrinsics = extrinsics_all[cam_indices]
        gt_intrinsics = intrinsics_all[cam_indices]
        print(f"Loaded native cameras from npz: {args.camera_npz_native}")
    elif args.camera_bin:
        gt_intrinsics, gt_extrinsics = load_camera_bin_colmap(
            f"/home/amila/datasets/{dataset_name}/sparse/0/",
            image_paths=selected_images,
        )
        print(f"Loaded cameras from COLMAP bin for {len(selected_images)} selected images.")
    elif args.camera_json:
        k_mat, w2c_mats = load_camera_json(f"{dataset_path}/transforms.json")
        selected_names = [os.path.basename(p) for p in selected_images]
        print(f"Selected names: {selected_names}")
        gt_extrinsics = np.stack([w2c_mats[name] for name in selected_names], axis=0)
        gt_intrinsics = np.stack([k_mat.copy() for _ in selected_names], axis=0)
        print(f"Loaded cameras from json for {len(selected_images)} selected images.")
    else:
        gt_extrinsics = None
        gt_intrinsics = None

    render_exts = None
    if gt_extrinsics is not None:
        pred_init = model.inference(
            image=selected_images,
            process_res=448,
        )
        render_exts, rot, trans, scale = align_render_extrinsics_to_prediction(
            gt_extrinsics=gt_extrinsics,
            pred_extrinsics=pred_init.extrinsics,
            pred_scale_factor=pred_init.scale_factor if pred_init.is_metric else None,
        )
        print("pred -> gt rotation:\n", rot)
        print("pred -> gt translation:\n", trans)
        print("pred -> gt scale:", scale)
        print(f"Aligned render extrinsics to predicted GS frame: {render_exts.shape}")

    prediction = model.inference(
        image=selected_images,
        # extrinsics=extrinsics,
        # intrinsics=intrinsics,
        render_exts=render_exts,
        render_ixts=gt_intrinsics,
        # render_ixts=pred_init.intrinsics,
        infer_gs=True,
        export_dir=out_path,
        export_format="gs_ply" if args.dump_ply else "gs_video",
        process_res=448,
        export_kwargs=export_args,
    )

    if args.dump_cameras:
        export_prediction_cameras_for_align_pose(
            prediction=prediction,
            image_paths=selected_images,
            output_dir=os.path.join(out_path, "cameras"),
        )

    # prediction.processed_images : [N, H, W, 3] uint8   array
    print(f"Processed images: {prediction.processed_images.shape}")

if __name__ == "__main__":
    main()
