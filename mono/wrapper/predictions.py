from typing import Union

import numpy as np
from PIL import Image

from mono.utils.do_test import get_prediction_custom, transform_test_data_scalecano

def predict_depth(model, cfg, image: Image.Image, intrinsics: tuple):
    """Predict depth given a Metric3D model, it's config and an input image"""

    normalize_scale = cfg.data_basic.depth_range[-1]

    image = np.asarray(image)

    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(image, intrinsics, cfg.data_basic)

    pred_depth, _, _ = get_prediction_custom(
        model = model,
        input = rgb_input,
        cam_model = cam_models_stacks,
        pad_info = pad,
        scale_info = label_scale_factor,
        gt_depth = None,
        normalize_scale = normalize_scale,
        ori_shape=[image.shape[0], image.shape[1]]
    )

    pred_depth = pred_depth.detach().cpu().numpy()

    return pred_depth