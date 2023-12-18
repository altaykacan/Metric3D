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

from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs
from mono.utils.do_test import get_prediction_custom, transform_test_data_scalecano
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
from mono.utils.trajectory import read_kitti_trajectory
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
    config_path = Path("./mono/configs/HourglassDecoder/convlarge.0.3_150_deepscenario.py")

    use_every_nth = 1 # Every n'th image is used to create point cloud
    start = 0 # index of the starting image
    end = 4 # index of the last image, set None to include all images from start

    H_output = 747 # 1/4th of our custom data, keeping the aspect ratio the same
    W_output = 1328



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

    normalize_scale = cfg.data_basic.depth_range[-1]

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
            # rgb_origin = cv2.imread(current_image_data["rgb"])[:,:,::-1].copy()
            rgb_origin = cv2.imread(current_image_data["rgb"]).copy() # preprocess step already has color space transformation

            intrinsics = current_image_data["intrinsic"]

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
            )

            # Compute mask to decide which points to show or not

            # Create point cloud from depth map (in camera coordinates)

            # Transform point cloud to world coordinates

            # Save original image, depth prediction
            image_to_save = rgb_input.squeeze().cpu()
            image_to_save = image_to_save * std + mean
            image_to_save = image_to_save.permute(1,2,0).numpy().astype(int)
            plt.imsave(
                Path(save_image_dir, current_image_data["filename"]),
                image_to_save
            )

            plt.imsave(
                Path(save_pred_dir, current_image_data["filename"]),
                # TODO, add the depth here
            )

            print("DEBUG: done with one iteration!")


    # Save the reconstructed point cloud

if __name__ == "__main__":
    main()