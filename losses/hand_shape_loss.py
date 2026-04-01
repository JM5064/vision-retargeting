import torch
import torch.nn as nn

from config.allegro import Allegro


class HandShapeLoss(nn.Module):

    def __init__(self, epsilon1=0.1):
        super().__init__()

        self.epsilon1 = epsilon1


    def forward(self, pred_positions, gt_positions):
        # Vector between fingertips and wrist
        v_gt = gt_positions[:, Allegro.FINGERTIPS, :]
        v_pred = pred_positions[:, Allegro.FINGERTIPS, :]

        # Sum distances between pred and gt fingertip-wrist vectors
        dist = torch.sum((v_pred - v_gt)**2, dim=-1)

        # Calculate distance from each fingertip to thumb
        gt_thumb = gt_positions[:, Allegro.THUMB, :].unsqueeze(1)
        gt_tips = gt_positions[:, Allegro.FINGERTIPS, :]
        d_i = torch.norm(gt_tips - gt_thumb, dim=-1)

        sdi = self.calculate_sigmoid(d_i)

        loss = (sdi * dist).sum(dim=-1).mean()

        return loss

    
    def calculate_sigmoid(self, d_i):
        # Pass through sigmoid
        sdi = torch.sigmoid(10 * (d_i - self.epsilon1))

        return sdi

