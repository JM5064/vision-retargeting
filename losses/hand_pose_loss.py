import torch
import torch.nn as nn

from models.utils import freihand_to_allegro


class HandPoseLoss(nn.Module):

    def __init__(self, beta):
        super().__init__()

        self.beta = beta


    def forward(self, pred_positions, gt_positions):
        # Lhand pose = p h thumb − p r thumb 2 2

        thumb_gt_positions = gt_positions[:, freihand_to_allegro(4), :]
        thumb_pred_positions = pred_positions[:, freihand_to_allegro(4), :]

        # Calculate thumb distances
        thumb_distances = torch.sum((thumb_pred_positions - thumb_gt_positions)**2, dim=-1).mean()

        return thumb_distances
