import numpy as np
import torch
from plyfile import PlyData, PlyElement
import cv2


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
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    """
    Reconstructs a point cloud from a depth map.

    Args:
        depth (numpy.ndarray or torch.Tensor): The depth map.
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