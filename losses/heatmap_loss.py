import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, heatmap_preds, heatmap_labels):
        _, C, _, _ = heatmap_preds.shape
        num_keypoints = C // 3

        # Split into xy, xz, zy
        xy_preds = heatmap_preds[:, :num_keypoints]
        xz_preds = heatmap_preds[:, num_keypoints:2*num_keypoints]
        zy_preds = heatmap_preds[:, 2*num_keypoints:]

        xy_labels = heatmap_labels[:, :num_keypoints]
        xz_labels = heatmap_labels[:, num_keypoints:2*num_keypoints]
        zy_labels = heatmap_labels[:, 2*num_keypoints:]

        # Calculate Jensen–Shannon divergence for each heatmap
        loss_xy = self.jensen_shannon_loss(xy_preds, xy_labels)
        loss_xz = self.jensen_shannon_loss(xz_preds, xz_labels)
        loss_zy = self.jensen_shannon_loss(zy_preds, zy_labels)
        
        return (loss_xy + loss_xz + loss_zy) / 3.0
    

    def jensen_shannon_loss(self, heatmap_preds, heatmap_labels, eps=1e-8):
        # Softmax to get P
        p = F.softmax(heatmap_preds.view(heatmap_preds.size(0), heatmap_preds.size(1), -1), dim=-1)
        
        # Normalize GT to get Q
        q = heatmap_labels.view(heatmap_labels.size(0), heatmap_labels.size(1), -1)
        q = q / (q.sum(dim=-1, keepdim=True) + eps)
        
        # Compute M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # Clamping M ensures that we never take log(0)
        # We use log_m to pass into kl_div
        log_m = torch.clamp(m, min=eps).log()
        
        # F.kl_div(input, target) expects input to be in log-space
        kl_p_m = F.kl_div(log_m, p, reduction='batchmean')
        kl_q_m = F.kl_div(log_m, q, reduction='batchmean')
        
        return 0.5 * (kl_p_m + kl_q_m)
