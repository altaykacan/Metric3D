import numpy as np
import typing
import torch
from plyfile import PlyData, PlyElement
import cv2

from typing import Union


def get_pcd_base(H, W, u0, v0, fx, fy):
    """
    Calculate the base point cloud coordinates for a given camera intrinsic parameters.

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        u0 (float): Principal point x-coordinate.
        v0 (float): Principal point y-coordinate.
        fx (float): Focal length in the x-direction.
        fy (float): Focal length in the y-direction.

    Returns:
        np.ndarray: Array of shape (H, W, 3) representing the base point cloud coordinates.
                   Each element in the array is a 3D point (x, y, z) in camera coordinates.
    """
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32) # image plane coordinates for the width direction
    u_m_u0 = x - u0 # Center the coordinates to the camera principal point in the camera coordinate frame

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T # Transpose to make work as the columns
    y = y.astype(np.float32) # image plane coordinates for the height direction
    v_m_v0 = y - v0 # Center the coordinates to the camera principal point

    x = u_m_u0 / fx # revert the scaling from the standard image plane with f=1 to the actual camera image plane
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [H, W, c], c is the number of coords which is usually 3
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    """
    Reconstructs a point cloud from a single depth map.

    Args:
        depth (numpy.ndarray or torch.Tensor): The depth map with shape (H,W)
        fx (float): The focal length in the x-direction.
        fy (float): The focal length in the y-direction.
        u0 (float): The x-coordinate of the principal point.
        v0 (float): The y-coordinate of the principal point.
        pcd_base (numpy.ndarray, optional): The base point cloud. If not provided, it will be computed internally.
        mask (numpy.ndarray, optional): A mask indicating which points to exclude from the point cloud.

    Returns:
        numpy.ndarray: The reconstructed point cloud.

    """
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask:
        pcd[mask] = 0
    return pcd


def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8),
                              (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack[x, y, z, r, g, b], fmt='%f %f %f %d %d %d', header=ply_head, comments='')


def transform_pcd_to_world(pcd:  np.ndarray, extrinsics: np.ndarray, scale_factor: float=1.0):
    """
    Helper function to transform a given 3d point cloud from camera coordinates to
    world coordinates.

    The transform is done by flattening the point cloud to be of shape
    [num_points, 3] adding and extra dimension with 1 for homogenous coordinates,
    transposing the matrix and multiplying with the extrinsic transformation
    matrix. The flattened point cloud without the extra dimension is returned.

    Args:
        pcd (np.ndarray): The point cloud in camera coordinates
        with shape [H,W,3] where the last dimension stores the x-y-z coordinates

        extrinsics (np.ndarray): The extrinsic matrix, 4x4
        homogeneous transform to convert camera coordinates to world coordinates

        scale_factor (float): The factor to multiply the translation components
        of the extrinsic matrix, used to counter the scale ambiguity of monocular
        trajectory estimation methods.

    Returns:
        transformed_pcd (np.ndarray): Flattened and transformed
        point cloud, has shape [num_points,3]
    """
    pcd = pcd.reshape(-1, 3)
    row_of_ones = np.ones((pcd.shape[0], 1)) # shape is [num_points, 1]
    pcd = np.concatenate((pcd, row_of_ones), axis=1)

    extrinsics[:3,3] = scale_factor * extrinsics[:3,3]

    pcd_transformed = extrinsics @ pcd.T # [4,4] @ [4, num_points]
    pcd_transformed = pcd_transformed.T
    pcd_transformed = pcd_transformed[:, :3] # [num_points, 3]

    return pcd_transformed

def compute_mask(depth: np.ndarray, roi: typing.List, min_d: float, max_d: float, dropout: float=0):
    """
    Helper function to compute mask to decide which points to include in the
    main point cloud.

    Args:
        depth (np.ndarray): The depth map.
        roi (List): The region of interest [top, bottom, left, right].
        min_d (float): The minimum depth value to include in the mask.
        max_d (float): The maximum depth value to include in the mask.
        dropout (float, optional): The dropout probability to randomly exclude points from the mask.

    Returns:
        torch.Tensor: The computed mask as a flattened array.
    """
    mask = (depth > min_d) & (depth < max_d)
    if roi != []:
        mask[:roi[0], :] = False
        mask[roi[1]:, :] = False
        mask[:, roi[2]] = False
        mask[:, roi[3]:] = False
    if dropout > 0:
        mask = mask & (np.random.rand(*depth.shape) > dropout) # need to unpack shape which is a tuple

    mask = mask.reshape(-1)

    return mask