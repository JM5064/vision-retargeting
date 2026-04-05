import torch.nn as nn

from losses.heatmap_loss import HeatmapLoss
from losses.hand_pose_loss import HandPoseLoss
from losses.hand_shape_loss import HandShapeLoss
from losses.fingertip_orientation_loss import FingertipOrientationLoss
from losses.pinch_loss import PinchLoss


class CombinedLoss(nn.Module):

    def __init__(self, a=1, b=10*10, c=1*10, d=1*10, e=10*10):
        super().__init__()

        self.heatmap_loss_func = HeatmapLoss()
        self.pose_loss_func = HandPoseLoss()
        self.shape_loss_func = HandShapeLoss()
        self.orientation_loss_func = FingertipOrientationLoss()
        self.pinch_loss_func = PinchLoss()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e


    def forward(self, heatmap_preds, heatmap_labels, pred_positions, gt_positions):
        heatmap_loss = self.a * self.heatmap_loss_func(heatmap_preds, heatmap_labels)
        pose_loss = self.b * self.pose_loss_func(pred_positions, gt_positions)
        shape_loss = self.c * self.shape_loss_func(pred_positions, gt_positions)
        orientation_loss = self.d * self.orientation_loss_func(pred_positions, gt_positions)
        pinch_loss = self.e * self.pinch_loss_func(pred_positions, gt_positions)

        # Weight each loss function
        combined_loss = (
            heatmap_loss + 
            pose_loss + 
            shape_loss + 
            pinch_loss + 
            orientation_loss
        )

        return combined_loss, heatmap_loss, pose_loss, shape_loss, pinch_loss, orientation_loss
