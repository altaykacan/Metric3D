import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import time
import os
import os.path as osp
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mono.utils.do_test import get_prediction_custom, transform_test_data_scalecano
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_data
from mono.utils.trajectory import read_kitti_trajectory
from mono.utils.unproj_pcd import reconstruct_pcd, transform_pcd_to_world, save_point_cloud, compute_mask
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

def main():
    # Set configuration variables
    file_name = "test.ply"

    weigth_path = Path("./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth")
    trajectory_path = Path("/usr/stud/kaa/thesis/data_temp/deep_scenario/poses_dvso/01.txt")
    image_dir = Path("/usr/stud/kaa/thesis/data_temp/deep_scenario/sequences/01/image_2")
    # Input image resizing (called crop) defined in the config file
    config_path = Path("./mono/configs/HourglassDecoder/convlarge.0.3_150_deepscenario.py")

    trajectory_scale = 100 # scale factor to use for the translation component

    use_every_nth = 50 # every n'th image is used to create point cloud
    start = 0 # index of the starting image
    end = 201 # index of the last image, set None to include all images from start

    H_output = 2988 // 4 # numbers are the original sizes, needs to be integers
    W_output = 5312 // 4

    # Parameters to define max/min depth and region of interest for point cloud backprojection
    min_d = 0
    max_d = 60
    roi = [
        int(H_output * 0.3),
        H_output,
        0,
        W_output
    ] # leave empty to consider the whole image, origin is at top left corner
    dropout = 0.5

    # Construct the Config object for metric3d
    cfg = Config.fromfile(config_path)

    # Use config filename + timestamp as default show_dir if args.show_dir is None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.show_dir = str(Path('./show_dirs',
                            "deep_scenario",
                            timestamp))
    cfg.load_from = str(weigth_path)

    # Load data info as dummy values and other relevant data from config
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info

    normalize_scale = cfg.data_basic.depth_range[-1] # config param for model depth prediction range (?)

    # Update check point info
    reset_ckpt_path(cfg.model, data_info)

    # Create directories to show results
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

    save_image_dir = Path(cfg.show_dir, "raw_images")
    save_pred_dir = Path(cfg.show_dir, "depth")
    save_pcd_dir = Path(cfg.show_dir, "pcd") # point cloud directory

    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_pcd_dir, exist_ok=True)


    # Dump config (not implemented yet)

    # Read in trajectories and image data (including path and intrinsics)
    trajectories = read_kitti_trajectory(trajectory_path)
    image_data = load_data(str(image_dir)) # default implementation expects strings

    assert len(image_data) == len(trajectories), "Expected the length of available trajectories and image data to be the same!"

    if end is None:
        trajectories = trajectories[start:]
        image_data = image_data[start:]
    else:
        trajectories = trajectories[start:end]
        image_data = image_data[start:end]

    print(f"Creating point cloud from {len(image_data)} images using every {use_every_nth}th image...")

    # Standardization values used by the authors, useful to just save image nicely
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # Create a dictionary of lists of np arrays to keep track of the 3D point
    # cloud coordinates (pcd) and colors (rgb), both are (num_points, 3) arrays
    point_cloud = {"pcd": [], "rgb": []}

    # Initialize model and prepare for inference
    model = get_configured_monodepth_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()

    # Setup buffer variables similar to MonoRec's implementation (WIP)
    pose_buffer = []
    mask_buffer = []
    image_buffer = []
    depth_buffer = []

    buffer_length = 5
    key_index = buffer_length // 2

    # Loop over the list of images and the trajectories
    for i, (current_image_data, current_trajectory) in tqdm(enumerate(zip(image_data, trajectories))):
        if i % use_every_nth == 0:
            # Read in image and intrinsics from image_data
            rgb_origin = cv2.imread(current_image_data["rgb"])[:,:,::-1].copy()
            # rgb_origin = cv2.imread(current_image_data["rgb"]).copy() # preprocess step already has color space transformation

            intrinsics = current_image_data["intrinsic"]
            extrinsics = current_trajectory

            # Preprocess input for model and get prediction
            rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsics, cfg.data_basic)

            pred_depth, _, _ = get_prediction_custom(
                model = model,
                input = rgb_input,
                cam_model = cam_models_stacks,
                pad_info = pad,
                scale_info = label_scale_factor,
                gt_depth = None,
                normalize_scale = normalize_scale,
                H_output = H_output,
                W_output = W_output,
                # ori_shape=[rgb_origin.shape[0], rgb_origin.shape[1]]
            )

            # Create point cloud from depth map (in camera coordinates)
            pred_depth = pred_depth.detach().cpu().numpy()
            pcd = reconstruct_pcd(pred_depth, *intrinsics) # unpack the list with *

            # Compute mask to decide which points to show or not (WIP)
            mask = compute_mask(pred_depth, roi, min_d, max_d, dropout)

            # Transform point cloud to world coordinates
            pcd = transform_pcd_to_world(pcd, extrinsics, trajectory_scale)
            # pcd = pcd.reshape(-1,3)

            # Save original image, depth prediction, and transformed point cloud
            image_model_input = rgb_input.squeeze().cpu()
            image_model_input = image_model_input * std + mean
            image_model_input = image_model_input.permute(1,2,0).numpy().astype(np.uint8)

            plt.imsave(
                Path(save_image_dir, current_image_data["filename"]),
                image_model_input
            )

            plt.imsave(
                Path(save_pred_dir, current_image_data["filename"]),
                pred_depth
            )

            point_cloud["pcd"].append(pcd[mask])
            # Need to resize and flatten original rgb
            image_to_backproject = cv2.resize(rgb_origin, dsize=(H_output, W_output))
            image_to_backproject = image_to_backproject.reshape(-1,3)
            point_cloud["rgb"].append(image_to_backproject[mask])

            print("DEBUG: done with one iteration!")

    # Save the reconstructed point cloud
    pcd_total = np.concatenate(point_cloud["pcd"], axis=0)
    rgb_total = np.concatenate(point_cloud["rgb"], axis=0)

    save_point_cloud(pcd_total, rgb_total, filename = Path(save_pcd_dir, "cloud_scale500.ply"))



if __name__ == "__main__":
    main()