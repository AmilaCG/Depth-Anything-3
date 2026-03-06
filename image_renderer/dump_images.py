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
        # c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        # w2c = np.linalg.inv(c2w).astype(np.float32)
        # identity_check = np.matmul(c2w, w2c)
        # print(f"Identity check (should be close to identity matrix):\n{identity_check}")

        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w_cv = c2w_gl @ OPENGL_TO_OPENCV
        w2c = np.linalg.inv(c2w_cv).astype(np.float32)

        file_name = frame["file_path"].split("/")[-1]  # Extract file name from path
        w2c_dict[file_name] = w2c # Example_w2c = w2c_dict['000063.png']

    return k_mat.copy(), w2c_dict

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
    selected_images = [img for i, img in enumerate(images) if i not in {0, 8, 16, 24, 32, 40, 48, 56}] if args.test_views else images

    k_mat, w2c_mats = load_camera_json(f"{dataset_path}/transforms.json")
    selected_names = [os.path.basename(p) for p in selected_images]
    print(f"Selected names: {selected_names}")
    extrinsics = np.stack([w2c_mats[name] for name in selected_names], axis=0)
    # print(f"Selected extrinsics shape: {extrinsics.shape}")
    # print(f"Extrinsics: {extrinsics}")
    intrinsics = np.stack([k_mat.copy() for _ in selected_names], axis=0)
    # print(f"Selected intrinsics shape: {intrinsics.shape}")
    # print(f"Intrinsics: {intrinsics}")

    prediction = model.inference(
        image=selected_images,
        # extrinsics=extrinsics,
        # intrinsics=intrinsics,
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
