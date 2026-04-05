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

    def jensen_shannon_loss(self, heatmap_preds, heatmap_labels):
        """
        pred_logits: Raw CNN output [B, K, H, W]
        gt_heatmap: Your existing Gaussian GT [B, K, H, W]
        """
        # 1. Turn predicted logits into a probability distribution (sum to 1)
        p = F.softmax(heatmap_preds.view(heatmap_preds.size(0), heatmap_preds.size(1), -1), dim=-1)
        
        # 2. Ensure GT is also a valid probability distribution
        # Even if your GT is a Gaussian, it must sum to 1 for JSD
        q = heatmap_labels.view(heatmap_labels.size(0), heatmap_labels.size(1), -1)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. Calculate M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # 4. JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        # kl_div expects log-space for the first argument
        kl_p_m = F.kl_div(m.log(), p, reduction='batchmean')
        kl_q_m = F.kl_div(m.log(), q, reduction='batchmean')
        
        return 0.5 * (kl_p_m + kl_q_m)
