import glob
import os
import json
from pathlib import Path
from typing import List, Dict, Union

import cv2

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
        }
        datas.append(data_i)
    return datas

def load_data(path: str):
    """
    Load custom data from the specified directory.

    Args:
        path (str): The path to the directory containing the data.

    Returns:
        list: A list of dictionaries representing the loaded data.
              Each dictionary contains the following keys:
              - 'rgb': The path to the RGB image file.
              - 'intrinsic': The intrinsic parameters of the camera in the format [fx, fy, cx, cy].
              - 'filename': The filename of the image file.
              - 'folder': The folder name containing the image file.
    """
    rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    rgbs = sorted(rgbs) # hacky solution to get file names, glob doesn't sort on it's own
    data = [{'rgb':i, 'depth':None,
             'intrinsic': [2779.9523856929486, 2779.9523856929486,  2655.5, 1493.5],
             'filename':os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    return data

def load_data_custom(root: Union[Path, str], paths: List[Union[Path, str]]):
    """Load the image data specified from the paths list"""
    # Convert to strings if inputs are Path objects
    if isinstance(paths[0], Path):
        paths = [str(element) for element in paths]
    if isinstance(root, Path):
        root = str(root)

    # Appending the trailing slash to the root path is necessary
    if root[-1] != "/":
        root += "/"

    # Keeping this part the same as the original repo, by no means ideal
    rgbs = [root + path for path in paths]
    data = [{'rgb':i, 'depth':None,
             'intrinsic': [2779.9523856929486, 2779.9523856929486,  2655.5, 1493.5],
             'filename':os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    return data