import torch
from config.allegro import Allegro


def calculate_pinch_distance(pred_pos, labels_XYZ, threshold_human=0.03, threshold_robot=0.05):
    """Calculates average pinch distance for pinching frames
    args:
        pred_pos: predicted positions after FK
        labels_XYZ: freihand XYZ labels
        threshold: pinch distance threshold in meters
    """

    HUMAN_THUMB_TIP = 4
    HUMAN_FINGERTIPS = [8, 12, 16]

    # Get human thumb and fingertips
    human_thumb = labels_XYZ[:, HUMAN_THUMB_TIP, :].unsqueeze(1)
    human_fingertips = labels_XYZ[:, HUMAN_FINGERTIPS, :]

    # Calculate distance between human fingertips and thumb
    human_dist = torch.norm(human_fingertips - human_thumb, dim=-1)

    # Create mask for fingers that are pinching
    pinch_mask = (human_dist < threshold_human).float()

    # Get robot thumb and fingertips
    robot_thumb = pred_pos[:, Allegro.THUMB, :].unsqueeze(1)
    robot_fingertips = pred_pos[:, Allegro.FINGERTIPS, :]

    # Calculate distance between robot fingertips and thumb
    robot_dist = torch.norm(robot_fingertips - robot_thumb, dim=-1)

    masked_robot_dist = robot_dist * pinch_mask

    # Calculate number of actual pinches
    num_pinches = pinch_mask.sum()
    if num_pinches == 0: return 0, 0, 0

    total_pinch_distance = masked_robot_dist.sum()
    
    success_count = ((robot_dist < threshold_robot).float() * pinch_mask).sum()

    return total_pinch_distance.item(), success_count.item(), num_pinches.item()

