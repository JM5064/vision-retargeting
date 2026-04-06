import torch
import numpy as np


def reproject_xyZ2XYZ(xyz, K):
    # Decompose intrinsics
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]

    # Decompose xyZ
    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    Z = xyz[:, :, 2]

    # Reproject xyz to XYZ
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    XYZ = torch.stack([X, Y, Z], dim=-1)

    return XYZ


def xyZ2XYZ(keypoints, image_size, Ks, wrist_depths, scales):
    """
    args:
        keypoints: [batch, num_keypoints, 3]
        wrist_depths: [batch]
        scales: [batch]
        Ks: [batch, 3, 3]

    """
    xyz = keypoints.clone()

    # Unnormalize xy
    xyz[:, :, :2] *= image_size

    # Unnormalize depth with wrist depths
    xyz[:, :, 2] += wrist_depths.view(-1, 1)

    XYZ = reproject_xyZ2XYZ(xyz, Ks)

    # Undo scaling
    XYZ *= scales.view(-1, 1, 1)

    return XYZ


def get_positions(fk_result):
    matrices = [fk_result[key].get_matrix() for key in sorted(fk_result.keys())]
    stacked_matrices = torch.stack(matrices)
    stacked_matrices = stacked_matrices.transpose(0, 1)
    positions = stacked_matrices[:, :, :3, 3]

    return positions


def get_positions(fk_result):
    link_names = [
        "base_link", "link_0.0", "link_1.0", "link_2.0", "link_3.0", "link_3.0_tip", 
        "link_4.0", "link_5.0", "link_6.0", "link_7.0", "link_7.0_tip", 
        "link_8.0", "link_9.0", "link_10.0", "link_11.0", "link_11.0_tip", 
        "link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"
    ]

    matrices = [fk_result[name].get_matrix() for name in link_names]
    stacked_matrices = torch.stack(matrices) # [num_links, batch, 4, 4]
    stacked_matrices = stacked_matrices.transpose(0, 1) # [batch, num_links, 4, 4]
    
    return stacked_matrices[:, :, :3, 3] # [batch, num_links, 3]


def rotation_scale_normalize(xyz, scale_factor, eps=1e-8):
    """
    Scales, root-normalizes, and rotation aligns a batch of FreiHAND coordinates.
    
    args:
        xyz (torch.Tensor): (batch_size, num_keypoints, 3)
        scale_factor (torch.Tensor): Shape (batch_size)
        eps (float): Small value to avoid division by zero.
        
    returns:
        torch.Tensor: Transformed coordinates (batch_size, num_keypoints, 3)
    """
    # Scale coordinates
    xyz = xyz / scale_factor.view(-1, 1, 1)

    # Multiply by scale factor to match with GeoRT
    xyz = xyz * 0.028
    
    # Root normalize
    wrist = xyz[:, 0:1, :] 
    xyz = xyz - wrist
    
    # Define the Z-axis
    # Vector from Wrist (0) to Middle Finger MCP (9)
    z_vec = xyz[:, 9, :] - xyz[:, 0, :]
    z_axis = z_vec / (torch.linalg.norm(z_vec, dim=1, keepdim=True) + eps)
    
    # Define the Palm Normal (X direction)
    v_index = xyz[:, 5, :] - xyz[:, 0, :]
    v_pinky = xyz[:, 17, :] - xyz[:, 0, :]
    
    # Cross product to find raw palm normal
    palm_normal_raw = torch.linalg.cross(v_index, v_pinky)
    
    # Build Orthonormal Basis
    # Y is perpendicular to both the Up direction (Z) and the Palm Normal.
    y_vec = torch.linalg.cross(z_axis, palm_normal_raw)
    y_axis = y_vec / (torch.linalg.norm(y_vec, dim=1, keepdim=True) + eps)
    
    # X-axis is now calculated to be perfectly orthogonal to Y and Z
    x_axis = torch.linalg.cross(y_axis, z_axis)
    
    # Construct Rotation Matrix
    rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)
    
    # Apply Rotation
    transformed_coords = torch.bmm(xyz, rotation_matrix.transpose(1, 2))
    
    return transformed_coords
