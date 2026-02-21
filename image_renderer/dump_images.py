import glob, os, torch
from depth_anything_3.api import DepthAnything3

# device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=torch.device("cuda"))

dataset_path = "/home/amila/datasets/kitti360/images"
images = sorted(glob.glob(os.path.join(dataset_path, "*.png")))

export_args = {
    "gs_video": {
        "vis_depth": None,
        "trj_mode": "original",
        "dump_images_dir": "/home/amila/Depth-Anything-3/outputs/images"
    }
}

prediction = model.inference(
    image=images,
    infer_gs=True,
    export_dir="/home/amila/Depth-Anything-3/outputs",
    export_format="gs_video",
    process_res=448,
    export_kwargs=export_args,
    # align_to_input_ext_scale=True,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)