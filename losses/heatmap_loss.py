import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, heatmap_preds, heatmap_labels):
        _, C, _, _ = heatmap_preds.shape
        num_keypoints = C // 3

        if torch.isnan(heatmap_preds).any() or torch.isinf(heatmap_preds).any():
            raise Exception("Heatmap predictions contain NaN/Inf before Loss")

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
    

    def jensen_shannon_loss(self, heatmap_preds, heatmap_labels, eps=1e-6):
        B, K, _, _ = heatmap_preds.shape
        
        # Softmax to get P
        p = F.softmax(heatmap_preds.view(B, K, -1), dim=-1)
        
        # Calculate the sum of each joint's GT heatmap
        # If sum is 0, the joint is off-screen
        q_raw = heatmap_labels.view(B, K, -1)
        joint_sums = q_raw.sum(dim=-1, keepdim=True) 
        mask = (joint_sums > 0).float()
        
        # Normalize GT to get Q
        q = q_raw / (joint_sums + eps)
        
        # Compute M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # Clamping M ensures that we never take log(0)
        # We use log_m to pass into kl_div
        m = torch.clamp(m, min=eps)
        
        # Calculate JSD per joint
        kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
        kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
        jsd_per_joint = 0.5 * (kl_pm + kl_qm) # Shape: [B, K]

        # Apply mask for visible joints
        masked_loss = jsd_per_joint * mask.squeeze(-1)
        
        return masked_loss.sum() / (mask.sum() + eps)
