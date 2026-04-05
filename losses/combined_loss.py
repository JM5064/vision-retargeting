import torch.nn as nn

from losses.keypoint_loss import KeypointLoss
from losses.heatmap_loss import HeatmapLoss

class CombinedLoss(nn.Module):

    def __init__(self, a=2, b=1):
        super().__init__()

        self.keypoint_loss_func = KeypointLoss()
        self.heatmap_loss_func = HeatmapLoss()

        self.a = a
        self.b = b


    def forward(self, keypoint_preds, keypoint_labels, heatmap_preds, heatmap_labels):
        # Weight each loss function
        keypoint_loss = self.a * self.keypoint_loss_func(keypoint_preds, keypoint_labels)
        heatmap_loss = self.b * self.heatmap_loss_func(heatmap_preds, heatmap_labels)

        # Combine losses
        combined_loss = (
            keypoint_loss + 
            heatmap_loss
        )

        return combined_loss, keypoint_loss, heatmap_loss
