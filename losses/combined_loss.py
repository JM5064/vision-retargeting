import torch.nn as nn

from losses.depth_loss import DepthLoss
from losses.heatmap_loss import HeatmapLoss


class CombinedLoss(nn.Module):

    def __init__(self, a=10, b=1):
        super().__init__()

        self.depth_loss = DepthLoss()
        self.heatmap_loss_func = HeatmapLoss()

        self.a = a
        self.b = b


    def forward(self, heatmap_preds, heatmap_labels, depth_preds, depth_labels):
        heatmap_loss = self.heatmap_loss_func(heatmap_preds, heatmap_labels)
        depth_loss = self.depth_loss(depth_preds, depth_labels)

        # Weight each loss function
        combined_loss = self.a * heatmap_loss + self.b * depth_loss

        return combined_loss, self.a * heatmap_loss, self.b * depth_loss
