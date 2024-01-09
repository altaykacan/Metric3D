from pathlib import Path

import torch
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.mldb import load_data_info, reset_ckpt_path

DEFAULT_WEIGHT_PATH = Path("./modules/depth/Metric3D/weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth")
DEFAULT_CONFIG_PATH = Path("./modules/depth/Metric3D/mono/configs/HourglassDecoder/convlarge.0.3_150_deepscenario.py") # input image resizing (called crop) defined in the config file

def get_config(cfg_path: Path = DEFAULT_CONFIG_PATH, weight_path: Path = DEFAULT_WEIGHT_PATH):
    """Returns the config object specific for Metric3D"""
    cfg = Config.fromfile(cfg_path)
    cfg.load_from = str(weight_path)

    # Loads basic dummy data information (in-the-wild-case)
    data_info = {}
    load_data_info("data_info", data_info=data_info)
    cfg.mldb_info = data_info

    # Update checkpoint information
    reset_ckpt_path(cfg.model, data_info)

    return cfg

def get_model(cfg):
    """
    Return model in the format specific for Metric3D to be used for
    predict method of the wrapper
    """
    model = get_configured_monodepth_model(cfg)

    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()

    return model