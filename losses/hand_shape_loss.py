import torch
import torch.nn as nn

from models.utils import freihand_to_allegro


class HandShapeLoss(nn.Module):

    def __init__(self, epsilon=0.1):
        super().__init__()

        self.epsilon = epsilon


    def forward(self, pred_positions, gt_positions):
        fingertips = [5, 10, 15] # AllegroHand indices for index, middle, and ring fingertips

        # Vector between fingertips and wrist
        v_gt = gt_positions[:, fingertips, :]
        v_pred = pred_positions[:, fingertips, :]

        # Sum distances between pred and gt fingertip-wrist vectors
        dist = torch.sum((v_pred - v_gt)**2, dim=-1)

        sdi = self.calculate_sigmoid(gt_positions, fingertips)

        loss = (sdi * dist).sum(dim=-1).mean()

        return loss

    
    def calculate_sigmoid(self, gt_positions, fingertips):
        # Calculate distance from each fingertip to thumb
        gt_thumb = gt_positions[:, freihand_to_allegro(4), :].unsqueeze(1)
        gt_tips = gt_positions[:, fingertips, :]
        d_i = torch.norm(gt_tips - gt_thumb, dim=-1)

        # Pass through sigmoid
        sdi = torch.sigmoid(-10 * (d_i - self.epsilon))

        return sdi

