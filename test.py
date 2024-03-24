from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

from mono.wrapper.loading import get_config, get_model
from mono.wrapper.predictions import predict_depth

cfg = get_config()
model = get_model(cfg)

fx = 2779.9523856929486
fy = 2779.9523856929486
cx = 2655.5
cy = 1493.5

intrinsics = (fx, fy, cx, cy)

img = Image.open(Path("deepscenario", "data", "demo","1403637130538319105.png"))

preds = predict_depth(model, cfg, img, intrinsics)

# plt.imsave("debug.png", preds)

