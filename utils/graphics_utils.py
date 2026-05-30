#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

@torch.no_grad()
def world_to_ndc(xyz, viewmatrix, intrinsics, width, height):
    """Project 3D points to NDC coordinates and determine visibility
    
    Args:
        xyz: torch.Tensor of shape [N, 3] - 3D points in world space
        viewmatrix: torch.Tensor of shape [3, 4] or [4, 4] - camera view matrix
        intrinsics: torch.Tensor of shape [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        
    Returns:
        ndc_xy: torch.Tensor of shape [N, 2] - NDC coordinates in range [-1, 1]
        visible: torch.BoolTensor of shape [N] - visibility mask (z>0 and in image bounds)
    """
    if not torch.is_tensor(xyz):
        xyz = torch.tensor(xyz, dtype=torch.float32)
    
    if not torch.is_tensor(viewmatrix):
        viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32)
        
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    
    # Ensure all tensors are on the same device and dtype
    device = viewmatrix.device
    dtype = viewmatrix.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    
    # Extract 3x4 if 4x4 is provided
    if viewmatrix.shape[0] == 4:
        viewmatrix = viewmatrix[:3, :]
    
    # Transform to camera space: [N, 3] @ [3, 3]^T + [3]
    R = viewmatrix[:3, :3]  # [3, 3]
    t = viewmatrix[:3, 3]   # [3]
    xyz_cam = xyz @ R.T + t  # [N, 3]
    
    # Extract z coordinate for depth check
    z = xyz_cam[:, 2]  # [N]
    
    # Project to image plane using intrinsics: K @ xyz_cam^T
    # intrinsics is [3, 3], xyz_cam is [N, 3]
    # Result: [3, N] -> transpose to [N, 3]
    xyz_proj = (intrinsics @ xyz_cam.T).T  # [N, 3]
    
    # Normalize by z (homogeneous coordinates)
    z_proj = xyz_proj[:, 2:3].clamp(min=1e-6)  # [N, 1]
    x_img = xyz_proj[:, 0:1] / z_proj  # [N, 1]
    y_img = xyz_proj[:, 1:2] / z_proj  # [N, 1]
    
    # Check visibility: z > 0 and within image bounds [0, W-1] x [0, H-1]
    visible = (z > 1e-6) & (x_img.squeeze() >= 0) & (x_img.squeeze() <= width - 1) & \
              (y_img.squeeze() >= 0) & (y_img.squeeze() <= height - 1)
    
    # Normalize to NDC [-1, 1] using (W-1, H-1)
    ndc_x = 2.0 * x_img / (width - 1) - 1.0   # [N, 1]
    ndc_y = 2.0 * y_img / (height - 1) - 1.0  # [N, 1]
    ndc_xy = torch.cat([ndc_x, ndc_y], dim=1)  # [N, 2]
    
    return ndc_xy, visible