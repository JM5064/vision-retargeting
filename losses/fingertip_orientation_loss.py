import torch
import torch.nn as nn

from config.allegro import Allegro


class FingertipOrientationLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, pred_positions, gt_positions):
        # Vector between fingertips and distal interphalangeal joints
        r_gt = gt_positions[:, Allegro.FINGERTIPS, :] - gt_positions[:, Allegro.DIP_JOINTS, :]
        r_pred = pred_positions[:, Allegro.FINGERTIPS, :] - pred_positions[:, Allegro.DIP_JOINTS, :]

        # Sum distances between pred and gt fingertip-dip_joint vectors
        dist = torch.sum((r_pred - r_gt)**2)

        return dist
