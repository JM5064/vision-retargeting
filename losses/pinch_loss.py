import torch
import torch.nn as nn

from config.allegro import Allegro


class PinchLoss(nn.Module):

    def __init__(self, epsilon1=0.1, epsilon2=0.01):
        super().__init__()

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2


    def forward(self, pred_positions, gt_positions):
        # Vector between fingertips and thumb
        gamma_gt = gt_positions[:, Allegro.FINGERTIPS, :] - gt_positions[:, Allegro.THUMB, :].unsqueeze(1)
        gamma_pred = pred_positions[:, Allegro.FINGERTIPS, :] - pred_positions[:, Allegro.THUMB, :].unsqueeze(1)

        # Fingertips-thumb length of gt
        d_i = torch.sqrt(torch.sum(gamma_gt**2, dim=-1) + 1e-8)

        # Normalize gamma_gt
        diff = torch.sum((gamma_pred - gamma_gt)**2, dim=-1)

        # For using rescaling function: 
        # gamma_gt_hat = gamma_gt / (d_i.unsqueeze(-1) + 1e-8)
        # rescaled = self.rescale(d_i)

        # sdi = self.calculate_sigmoid(d_i)

        # diff = torch.sum((gamma_pred - rescaled.unsqueeze(-1) * gamma_gt_hat)**2, dim=-1)

        sdi = self.calculate_sigmoid(d_i)
        
        # Calculate weighted sum
        loss = (sdi * diff).sum(dim=-1).mean()

        return loss


    def calculate_sigmoid(self, d_i):
        # Pass through sigmoid
        sdi = torch.sigmoid(-10 * (d_i - self.epsilon1))

        return sdi
    

    def rescale(self, d_i):
        # e2 <= d_o <= e1 term
        scaled_d = (self.epsilon1 / (self.epsilon1 - self.epsilon2)) * (d_i - self.epsilon2)
        
        # If d_i < e2, then 0, else, scaled_d
        result = torch.where(d_i < self.epsilon2, torch.zeros_like(d_i), scaled_d)
        
        # If d > e1, then d_i, else, result
        result = torch.where(d_i > self.epsilon1, d_i, result)
        
        return result
