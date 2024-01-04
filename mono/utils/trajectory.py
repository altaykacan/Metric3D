from typing import Union, List
import numpy as np
from pathlib import Path

def read_kitti_trajectory(trajectory_path: Union[str, Path]) -> List[np.ndarray]:
    """
    Reads the provided trajectory file and returns a list of the extrinsic
    matrices (T_w_cam, homogeneous transformation matrices from
    camera coordinates to world coordinates) as numpy arrays, also returns a list
    of image names for the trajectories if they are present in the input file.

    Args:
        trajectory_path (Union[str, Path]): Path to the trajectory file.

    Returns:
        List[np.ndarray]: List of extrinsic matrices.
        List[Path|str]: List of image names that correspond to the trajectories

    Raises:
        AssertionError: If the trajectory file cannot be split up correctly.

    Assumes the trajectory file is a txt with each line containing
    the rows of the extrinsic matrix (KITTI format)

    i.e.
    r11 r12 r13 tx
    r21 r22 r23 ty
    r31 r32 r33 tz
    0   0   0   1

    is saved as

    (/path/to/image.png) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

    where the image path is optionally provided.
    """
    trajectories = []
    paths = []

    with open(trajectory_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        entries = line.split(" ")

        if len(entries) == 13: # Means image path is included in the trajectory
            image_path = entries.pop(0)
            paths.append(Path(image_path))

        entries = np.array([float(entry) for entry in entries])

        assert len(entries) == 12, "Your trajectory file cannot be split up correctly, expected 12 columns corresponding to the first three rows of the 4x4 extrinsic matrix (KITTI format)"

        extrinsic = np.eye(4)
        extrinsic[:3, :] = entries.reshape(3,4)

        trajectories.append(extrinsic)

    return trajectories, paths



if __name__ == "__main__":
    print("Testing functionality...")
    trajectory_path = Path("/usr/stud/kaa/thesis/data_temp/deep_scenario_old/poses_dvso/01.txt")
    read_kitti_trajectory(trajectory_path)


