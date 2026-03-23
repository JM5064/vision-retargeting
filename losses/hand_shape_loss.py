import torch
import torch.nn as nn


class HandPoseLoss(nn.Module):

    def __init__(self, beta):
        super().__init__()

        self.beta = beta



    def forward(self, pred_positions, gt_positions, pred_qpos, gt_qpos):
        # Lhand pose = p h thumb − p r thumb 2 2 + βrot angle(q h wrist, q r wrist)

        # Calculate thumb distances
        thumb_distances = 0

        # Calculate angle difference between wrists
        wrist_angle = 0

        return thumb_distances + self.beta * thumb_distances


