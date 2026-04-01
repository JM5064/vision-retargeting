import torch
import torch.nn as nn

from models.utils import freihand_to_allegro


class FingertipOrientationLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, pred_positions, gt_positions):
        fingertips = [5, 10, 15] # AllegroHand indices for index, middle, and ring fingertips
        dip_joints = [4, 9, 14] # AllegroHand indices for distal interphalangeal joints for index, middle, and ring fingers

        # Vector between fingertips and wrist
        r_gt = gt_positions[:, fingertips, :] - gt_positions[:, dip_joints, :]
        r_pred = pred_positions[:, fingertips, :] - pred_positions[:, dip_joints, :]

        # Sum distances between pred and gt fingertip-dip_joint vectors
        dist = torch.sum((r_pred - r_gt)**2)

        return dist
