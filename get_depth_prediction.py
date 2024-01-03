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
from mono.utils.unproj_pcd import compute_mask
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

def main():
    # Set configuration variables

    weigth_path = Path("./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth")
    image_dir = Path("./deepscenario/data/scale_alignment")
    config_path = Path("./mono/configs/HourglassDecoder/convlarge.0.3_150_deepscenario.py") # input image resizing (called crop) defined in the config file

    H_output = 2988  # numbers are the original sizes, needs to be integers,
    W_output = 5312

    # Backprojection mask parameters
    min_d = 0
    max_d = 100
    roi = [
        int(H_output * 0.4), # top border
        int(H_output * 0.95), # bottom border
        int(W_output * 0.2), # left border
        int(W_output * 0.8) # right border
    ] # leave empty to consider the whole image, origin is at top left corner
    dropout = 0.9

    # Construct the Config object for metric3d
    cfg = Config.fromfile(config_path)

    # Use config filename + timestamp as default show_dir if args.show_dir is None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.show_dir = str(Path('./show_dirs',
                            "scale_alignment",
                            timestamp))
    cfg.load_from = str(weigth_path)

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

    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)

    # TODO dump config

    # Read in image_data
    image_data = load_data(str(image_dir)) # default implementation expects strings

    # Standardization values used by the authors, useful to just save image nicely
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # Initialize model and prepare for inference
    model = get_configured_monodepth_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()

    # Loop over the list of images and the trajectories
    for i, current_image_data in tqdm(enumerate(image_data)):
        # Read in image and intrinsics from image_data
        rgb_origin = cv2.imread(current_image_data["rgb"])[:,:,::-1].copy()
        # rgb_origin = cv2.imread(current_image_data["rgb"]).copy() # preprocess step already has color space transformation

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
            # ori_shape=[rgb_origin.shape[0], rgb_origin.shape[1]]
        )

        # Compute mask to decide which points to show or not, set None to ignore
        pred_depth = pred_depth.detach().cpu().numpy() # need change device and type
        mask = compute_mask(pred_depth, roi, min_d, max_d, dropout) # [num_points,]

        # Need to resize and flatten original rgb
        # image_to_backproject = cv2.resize(rgb_origin, dsize=(H_output, W_output))
        image_to_backproject = rgb_origin
        image_to_backproject = image_to_backproject.reshape(-1,3)
        if mask is not None:
            image_to_backproject = image_to_backproject[mask]

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

        np.save(Path(save_pred_dir, current_image_data["filename"].replace(".png","") + "_array"), pred_depth)

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



if __name__ == "__main__":
    main()