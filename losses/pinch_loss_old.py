import torch
import torch.nn as nn

from models.utils import freihand_to_allegro


class PinchLoss(nn.Module):

    def __init__(self, threshold=0.015):
        super().__init__()

        # Pinching distance threshold (in meters)
        self.threshold = threshold


    def forward(self, pred_positions, labels):
        """
        args:
            pred_positions: predicted robot link positions after FK
            labels: FreiHAND 3D labels, [batch, num_keypoints, 3]
        """
        batch_size = labels.shape[0]
        
        pinch_loss = 0

        fingertips = [8, 12, 16] # FreiHAND indices for index, middle, and ring fingertips
        thumb_labels = labels[:, 4, :]
        thumb_positions = pred_positions[:, freihand_to_allegro(4), :]

        for fingertip in fingertips:
            # Calculate distance between GT thumb and fingertip
            fingertip_labels = labels[:, fingertip, :]
            label_distance = thumb_labels - fingertip_labels

            # Create mask for GT fingers which are pinching
            mask = (torch.norm(label_distance, dim=-1) < self.threshold).float()

            # Calculate euclidian distance between predicted positions of thumb and fingertip
            fingertip_positions = pred_positions[:, freihand_to_allegro(fingertip), :]
            # position_distance = ((thumb_positions - fingertip_positions) ** 2).sum(dim=-1)
            position_distance = torch.norm(thumb_positions - fingertip_positions, dim=-1)

            pinch_loss += (mask * position_distance).mean() / (mask.sum() + 1e-7) * batch_size
            # Investigate normalizing by batch:
            # pinch_loss += (mask * position_distance).sum() / (batch_size + 1e-7)
            
        return pinch_loss
