import numpy as np
import torch


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

