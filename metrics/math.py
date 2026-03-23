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