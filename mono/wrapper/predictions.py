from typing import Union

import numpy as np
from PIL import Image

from mono.utils.do_test import get_prediction_custom, transform_test_data_scalecano

def predict_depth(model, cfg, image: Union[Image.Image, np.ndarray], intrinsics: tuple, predict_normals=False):
    """Predict depth (and optionally surface normals) given a Metric3D model, it's config and an input image"""

    normalize_scale = cfg.data_basic.depth_range[-1]

    # Ugly way to make sure we are working with np arrays
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    # We get the intrinsics as the input, no need to use the mono/utils/custom_data.py like the authors suggest
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(image, intrinsics, cfg.data_basic)

    pred_depth, _, _, output_dict = get_prediction_custom(
        model = model,
        input = rgb_input,
        cam_model = cam_models_stacks,
        pad_info = pad,
        scale_info = label_scale_factor,
        gt_depth = None,
        normalize_scale = normalize_scale,
        ori_shape=[image.shape[0], image.shape[1]]
    )

    pred_depth = pred_depth.detach().cpu()

    if predict_normals:
        pred_normal = output_dict['normal_out_list'][0][:, :3, :, :]
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

        pred_normal = pred_normal.squeeze()
        if pred_normal.size(0) == 3:
            pred_normal = pred_normal.permute(1,2,0)

        return pred_depth, pred_normal

    else:
        return pred_depth