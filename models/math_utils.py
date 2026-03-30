import torch


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

    # Unnormalize depth
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
