import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def run_inference(model, image_paths, selected_indices, out_path, export_args):
    selected_images = [image_paths[i] for i in selected_indices]
    prediction = model.inference(
        image=selected_images,
        process_res=448,
        # infer_gs=True,
        # export_dir=out_path,
        # export_format="gs_video-gs_ply",
        # export_kwargs=export_args,
    )
    return prediction


def get_aligned_camera_values(full_pred, subset_pred, full_indices, subset_indices):
    full_pos = {idx: pos for pos, idx in enumerate(full_indices)}
    subset_pos = {idx: pos for pos, idx in enumerate(subset_indices)}
    common_indices = [idx for idx in full_indices if idx in subset_pos]

    ext_full = np.stack([full_pred.extrinsics[full_pos[idx]] for idx in common_indices], axis=0)
    ext_subset = np.stack([subset_pred.extrinsics[subset_pos[idx]] for idx in common_indices], axis=0)
    ixt_full = np.stack([full_pred.intrinsics[full_pos[idx]] for idx in common_indices], axis=0)
    ixt_subset = np.stack([subset_pred.intrinsics[subset_pos[idx]] for idx in common_indices], axis=0)
    return common_indices, ext_full, ext_subset, ixt_full, ixt_subset


def plot_actual_values(common_indices, ext_full, ext_subset, ixt_full, ixt_subset, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    # Scale width with number of samples so x-axis points/labels have room.
    width = max(18.0, 0.55 * len(common_indices))

    n_ext_r, n_ext_c = ext_full.shape[1], ext_full.shape[2]
    fig, axes = plt.subplots(n_ext_r, n_ext_c, figsize=(width, 2.4 * n_ext_r), sharex=True)
    axes = np.asarray(axes).reshape(n_ext_r, n_ext_c)
    for r in range(n_ext_r):
        for c in range(n_ext_c):
            ax = axes[r, c]
            ax.plot(common_indices, ext_full[:, r, c], marker="o", markersize=3, label="full")
            ax.plot(common_indices, ext_subset[:, r, c], marker="s", markersize=3, label="subset")
            ax.set_title(f"ext[{r},{c}]")
            ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1, 0].set_xlabel("Original image index")
    axes[0, 0].legend()
    fig.suptitle("Extrinsics actual values: full vs subset", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "actual_extrinsics_values.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    n_ixt_r, n_ixt_c = ixt_full.shape[1], ixt_full.shape[2]
    fig, axes = plt.subplots(n_ixt_r, n_ixt_c, figsize=(width, 2.4 * n_ixt_r), sharex=True)
    axes = np.asarray(axes).reshape(n_ixt_r, n_ixt_c)
    for r in range(n_ixt_r):
        for c in range(n_ixt_c):
            ax = axes[r, c]
            ax.plot(common_indices, ixt_full[:, r, c], marker="o", markersize=3, label="full")
            ax.plot(common_indices, ixt_subset[:, r, c], marker="s", markersize=3, label="subset")
            ax.set_title(f"K[{r},{c}]")
            ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1, 0].set_xlabel("Original image index")
    axes[0, 0].legend()
    fig.suptitle("Intrinsics actual values: full vs subset", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "actual_intrinsics_values.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)



def main():
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=torch.device("cuda"))

    dataset_name = "kitti360"
    # dataset_name = "matrixcity"
    # dataset_name = "physicalai"

    dataset_path = f"/home/amila/datasets/{dataset_name}/images"
    out_path = f"/home/amila/Depth-Anything-3/outputs/{dataset_name}"
    image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.png")))
    # image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))

    export_args = {
        "gs_video": {
            "vis_depth": None,
            "trj_mode": "original",
            "dump_images_dir": f"{out_path}/images",
        }
    }

    first_n = 64
    excluded = {0, 8, 16, 24, 32, 40, 48, 56}
    full_indices = list(range(first_n))
    subset_indices = [i for i in full_indices if i not in excluded]

    print(f"Running full inference on images[:{first_n}] ...")
    full_pred = run_inference(
        model=model,
        image_paths=image_paths,
        selected_indices=full_indices,
        out_path=out_path,
        export_args=export_args,
    )

    print(f"Running subset inference excluding indices {sorted(excluded)} ...")
    subset_pred = run_inference(
        model=model,
        image_paths=image_paths,
        selected_indices=subset_indices,
        out_path=out_path,
        export_args=export_args,
    )

    print(f"full_pred.extrinsics.shape: {full_pred.extrinsics.shape}")
    print(f"full_pred.intrinsics.shape: {full_pred.intrinsics.shape}")
    print(f"subset_pred.extrinsics.shape: {subset_pred.extrinsics.shape}")
    print(f"subset_pred.intrinsics.shape: {subset_pred.intrinsics.shape}")

    common_indices, ext_full, ext_subset, ixt_full, ixt_subset = get_aligned_camera_values(
        full_pred=full_pred,
        subset_pred=subset_pred,
        full_indices=full_indices,
        subset_indices=subset_indices,
    )
    print(f"Plotting {len(common_indices)} aligned indices")
    plot_dir = os.path.join(out_path, "camera_compare")
    plot_actual_values(
        common_indices=common_indices,
        ext_full=ext_full,
        ext_subset=ext_subset,
        ixt_full=ixt_full,
        ixt_subset=ixt_subset,
        plot_dir=plot_dir,
    )
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
