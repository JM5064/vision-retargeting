import numpy as np
import torch
import torch.nn.functional as F
from models.utils import DEVICE


def get_heatmap_keypoints(heatmap_preds):
    """
    heatmap_preds: (batch, num_keypoints*3, heatmap_size, heatmap_size)

    returns:
        keypoints: (batch, num_keypoints, 2)
    """

    batch_size, N, heatmap_size, _ = heatmap_preds.shape
    num_keypoints = N // 3

    # Extract heatmaps and offset maps
    heatmaps = heatmap_preds[:, :num_keypoints]
    x_offsets = heatmap_preds[:, num_keypoints:2*num_keypoints]
    y_offsets = heatmap_preds[:, 2*num_keypoints:3*num_keypoints]

    # Flatten heatmaps and get argmax location
    argmaxes = np.argmax(heatmaps.reshape(batch_size, num_keypoints, -1), axis=-1)

    # Get xs and ys of argmax locations
    x = argmaxes % heatmap_size
    y = argmaxes // heatmap_size

    # Broadcast stuff...
    b = np.arange(batch_size)[:, None]
    k = np.arange(num_keypoints)[None, :]

    dx = x_offsets[b, k, y, x]
    dy = y_offsets[b, k, y, x]

    # Add offsets
    keypoint_x = x + dx
    keypoint_y = y + dy

    # Combine, convert to tensor, and normalize
    keypoints = np.stack([keypoint_x, keypoint_y], axis=-1)
    keypoints = torch.tensor(keypoints) / heatmap_size

    return keypoints


def heatmap_inference(heatmap_preds):
    """
    heatmap_preds: (batch, num_keypoints, heatmap_size, heatmap_size)

    returns:
        keypoints: (batch, num_keypoints, 2)
    """

    batch_size, num_keypoints, heatmap_size, _ = heatmap_preds.shape

    # Flatten heatmaps and get argmax location
    flat_heatmaps = heatmap_preds.view(batch_size, num_keypoints, -1)
    argmaxes = torch.argmax(flat_heatmaps, dim=-1)  # [batch, num_keypoints]

    # Get xs and ys of argmax locations
    x = argmaxes % heatmap_size
    y = argmaxes // heatmap_size

    # Combine, convert to tensor, and normalize
    keypoints = torch.stack([x, y], dim=-1)
    keypoints = keypoints / (heatmap_size - 1)

    return keypoints


def heatmap_inference_testing(heatmap_preds, heatmap_flipped_preds):
    """
    heatmap_preds: (batch, num_keypoints, heatmap_size, heatmap_size)
    heatmap_flipped_preds: (batch, num_keypoints, heatmap_size, heatmap_size), predictions made on a flipped image

    returns:
        keypoints: (batch, num_keypoints, 2)
    """
    # Get keypoint locations from heatmaps
    keypoints = heatmap_inference(heatmap_preds)
    flipped_keypoints = heatmap_inference(heatmap_flipped_preds)

    # Flip back the flipped heatmap
    corrected_keypoints = flipped_keypoints.clone()
    corrected_keypoints[:, :, 0] = 1.0 - corrected_keypoints[:, :, 0]

    corrected_keypoints[:, [0, 5]] = corrected_keypoints[:, [5, 0]]
    corrected_keypoints[:, [1, 4]] = corrected_keypoints[:, [4, 1]]
    corrected_keypoints[:, [2, 3]] = corrected_keypoints[:, [3, 2]]
    corrected_keypoints[:, [10, 15]] = corrected_keypoints[:, [15, 10]]
    corrected_keypoints[:, [11, 14]] = corrected_keypoints[:, [14, 11]]
    corrected_keypoints[:, [12, 13]] = corrected_keypoints[:, [13, 12]]

    # Average the two sets of keypoints
    averaged_keypoints = (keypoints + corrected_keypoints) / 2.0

    return averaged_keypoints


def marginal_heatmap_inference(heatmap_preds, z_min=-9.0, z_max=9.0):
    """
    heatmap_preds: (batch, num_keypoints*3, heatmap_size, heatmap_size)
    
    returns:
        keypoints: (batch, num_keypoints, 3)
    """
    
    batch_size, C, heatmap_size, _ = heatmap_preds.shape
    num_keypoints = C // 3
    z_range = z_max - z_min

    # Split heatmaps into xy, xz, zy
    xy_heatmaps = heatmap_preds[:, 0:num_keypoints]
    xz_heatmaps = heatmap_preds[:, num_keypoints:2*num_keypoints]
    zy_heatmaps = heatmap_preds[:, 2*num_keypoints:3*num_keypoints]

    # 2. Get x, y coordinate from xy heatmap
    flat_xy = xy_heatmaps.view(batch_size, num_keypoints, -1)
    argmax_xy = torch.argmax(flat_xy, dim=-1) 
    
    x_idx = argmax_xy % heatmap_size
    y_idx = argmax_xy // heatmap_size

    # Get z from xy and zy heatmaps using predicted x and y
    # Slice xz/zy planes and predicted x and y
    batch_indices = torch.arange(batch_size, device=heatmap_preds.device).view(batch_size, 1).expand(batch_size, num_keypoints)
    joint_indices = torch.arange(num_keypoints, device=heatmap_preds.device).view(1, num_keypoints).expand(batch_size, num_keypoints)

    # Slice xz heatmap using our predicted x and get z value
    z_slice_xz = xz_heatmaps[batch_indices, joint_indices, :, x_idx]
    z_idx_xz = torch.argmax(z_slice_xz, dim=-1)

    # Slice zy heatmap using our predicted y and get z value
    z_slice_zy = zy_heatmaps[batch_indices, joint_indices, y_idx, :]
    z_idx_zy = torch.argmax(z_slice_zy, dim=-1)

    # Average the z from xz and zy heatmaps
    z_idx_avg = (z_idx_xz.float() + z_idx_zy.float()) / 2.0
    
    # Normalize x and y outputs
    x_norm = x_idx.float() / (heatmap_size - 1)
    y_norm = y_idx.float() / (heatmap_size - 1)
    
    # Normalize z output
    z_val = (z_idx_avg / (heatmap_size - 1)) * z_range + z_min

    keypoints = torch.stack([x_norm, y_norm, z_val], dim=-1)

    return keypoints


def marginal_soft_argmax(heatmaps, temperature=10.0, device=DEVICE):
    """
    args:
        heatmaps: [batch_size, num_keypoints * 3, H, W] 

    returns:
        predicted xyZ keypoints [batch_size, num_keypoints, 3]
    """
    B, C, H, W = heatmaps.shape
    num_keypoints = C // 3
    
    # Create normalized coordinate grids (0 to 1)
    # y is vertical (dim -2), x is horizontal (dim -1)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )

    def get_2d_expectations(hm_slice):
        # Softmax over the spatial dimensions to get a probability distribution
        # Higher temperature = sharper peak
        probs = F.softmax(hm_slice.reshape(B, num_keypoints, -1) * temperature, dim=-1)
        probs = probs.reshape(B, num_keypoints, H, W)
        
        # Expected values
        mu_vert = torch.sum(probs * grid_y, dim=(-2, -1)) # Vertical axis
        mu_horiz = torch.sum(probs * grid_x, dim=(-2, -1)) # Horizontal axis
        return mu_horiz, mu_vert

    # Get
    xy_heatmap = heatmaps[:, :num_keypoints, :, :]
    xz_heatmap = heatmaps[:, num_keypoints:2*num_keypoints, :, :]
    zy_heatmap = heatmaps[:, 2*num_keypoints:, :, :]

    # Get x and y coordinates from xy heatmap
    x_xy, y_xy = get_2d_expectations(xy_heatmap)
    
    # Get z coordinate from xz and zy heatmaps
    x_xz, z_xz = get_2d_expectations(xz_heatmap)
    z_zy, y_zy = get_2d_expectations(zy_heatmap)
    
    # Get z from averaging z prediction from xz and zy heatmaps
    final_z = (z_xz + z_zy) / 2.0

    return torch.stack([x_xy, y_xy, final_z], dim=-1)
