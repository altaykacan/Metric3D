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
from mono.utils.custom_data import load_data, load_data_custom
from mono.utils.trajectory import read_kitti_trajectory
from mono.utils.unproj_pcd import reconstruct_pcd, transform_pcd_to_world, save_point_cloud, compute_mask
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

def main():
    # Set configuration variables
    file_name = "demo_metric3d_every_10_maxd_100_smaller_roi.ply"

    weight_path = Path("./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth")
    trajectory_path = Path("/usr/stud/kaa/thesis/data_temp/deep_scenario/poses/CleanedCustomKeyFrameTrajectory.txt") # hacky solution, need to fix ORBSLAM3 function TODO
    image_dir = Path("/usr/stud/kaa/thesis/data_temp/deep_scenario/sequences/01/image_2")
    config_path = Path("./mono/configs/HourglassDecoder/convlarge.0.3_150_deepscenario.py") # input image resizing (called crop) defined in the config file

    trajectory_scale = 393.0966796875 # scale factor to use for the translation component

    use_every_nth = 10 # every n'th image is used to create point cloud
    start = 0 # index of the starting image
    end = None # index of the last image, set None to include all images from start

    # TODO use preprocessing functions to resize, crop, and compute new intrinsics
    H_output = 2988  # numbers are the original sizes, needs to be integers,
    W_output = 5312

    # Backprojection mask parameters
    min_d = 0
    max_d = 100
    roi = [
        int(H_output * 0.45), # top border
        int(H_output * 0.95), # bottom border
        int(W_output * 0.3), # left border
        int(W_output * 0.7) # right border
    ] # leave empty to consider the whole image, origin is at top left corner
    dropout = 0.95

    # Construct the Config object for metric3d
    cfg = Config.fromfile(config_path)

    # Use config filename + timestamp as default show_dir if args.show_dir is None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.show_dir = str(Path('./show_dirs',
                            "scale_alignment_pcd",
                            timestamp))
    cfg.load_from = str(weight_path)

    # Load data info as dummy values and other relevant data from config
    #(taken from Metric3d do_scalecano_test_with_custom_data())
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info

    normalize_scale = cfg.data_basic.depth_range[-1] # config param for model depth prediction range? TODO figure out

    # Update check point info
    reset_ckpt_path(cfg.model, data_info)

    # Create directories to show results
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

    save_image_dir = Path(cfg.show_dir, "images")
    save_pred_dir = Path(cfg.show_dir, "depth") # predicted depth directory
    save_pcd_dir = Path(cfg.show_dir, "pcd") # point cloud directory

    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_pcd_dir, exist_ok=True)

    # TODO dump config

    # Read in trajectories and image data (including path and intrinsics)
    trajectories, paths = read_kitti_trajectory(trajectory_path)
    image_data = load_data_custom(image_dir, paths)

    assert len(image_data) == len(trajectories), "Expected the length of available trajectories and image data to be the same!"

    if end is None:
        trajectories = trajectories[start:]
        image_data = image_data[start:]
    else:
        trajectories = trajectories[start:end]
        image_data = image_data[start:end]

    print(f"Number of images: {len(image_data)}")
    print(f"Using every n'th image, n: {use_every_nth}")

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

    # TODO implement object detection here and figure out why exactly we need this buffer
    # Setup buffer variables similar to MonoRec's implementation
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
            #  TODO figure out what this label_scale_factor is
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

            # Compute mask to decide which points to show or not, set None to ignore
            pred_depth = pred_depth.detach().cpu().numpy() # need change device and type
            mask = compute_mask(pred_depth, roi, min_d, max_d, dropout) # [num_points,]

            # Append to buffers
            pose_buffer.append(extrinsics)
            mask_buffer.append(mask)
            image_buffer.append(rgb_origin)
            depth_buffer.append(pred_depth)

            if len(pose_buffer) >= buffer_length:
                extrinsics = pose_buffer[key_index]
                mask = mask_buffer[key_index]
                rgb_origin = image_buffer[key_index]
                pred_depth = depth_buffer[key_index]

                # Create point cloud from depth map (in camera coordinates)
                pcd = reconstruct_pcd(pred_depth, *intrinsics) # unpack the list with *

                # Transform point cloud to world coordinates, just flatten to igno
                pcd = transform_pcd_to_world(pcd, extrinsics, trajectory_scale)

                # Save point cloud
                if mask is not None:
                    pcd = pcd[mask]
                point_cloud["pcd"].append(pcd)

                # Need to resize and flatten original rgb
                # image_to_backproject = cv2.resize(rgb_origin, dsize=(H_output, W_output))
                image_to_backproject = rgb_origin
                image_to_backproject = image_to_backproject.reshape(-1,3)
                if mask is not None:
                    image_to_backproject = image_to_backproject[mask]
                point_cloud["rgb"].append(image_to_backproject)

                # Save used images/masks for debugging
                image_model_input = rgb_input.squeeze().cpu()
                image_model_input = image_model_input * std + mean
                image_model_input = image_model_input.permute(1,2,0).numpy().astype(np.uint8)

                plt.imsave(
                    Path(save_image_dir, current_image_data["filename"].replace(".png","") + "_input.png"),
                    image_model_input
                )

                plt.imsave(
                    Path(save_pred_dir, current_image_data["filename"]),
                    pred_depth
                )

                if mask is not None:
                    # Need to reshape mask again to save it and do element-wise
                    # multiplication with the original image
                    mask_reshaped = mask.reshape(pred_depth.shape[0], pred_depth.shape[1])
                    plt.imsave(
                        Path(save_image_dir, current_image_data["filename"].replace(".png","") + "_mask.png"),
                        mask_reshaped
                    )
                    plt.imsave(
                        Path(save_image_dir, current_image_data["filename"].replace(".png","") + "_for_pcd.png"),
                        rgb_origin * mask_reshaped[:,:,None] # element-wise multiplication with broadcasting
                    )

                # Move along buffer
                del pose_buffer[0]
                del mask_buffer[0]
                del image_buffer[0]
                del depth_buffer[0]


    # Save the full reconstructed point cloud
    pcd_total = np.concatenate(point_cloud["pcd"], axis=0)
    rgb_total = np.concatenate(point_cloud["rgb"], axis=0)

    save_point_cloud(pcd_total, rgb_total, filename = Path(save_pcd_dir, file_name))



if __name__ == "__main__":
    main()