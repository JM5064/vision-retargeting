import torch.nn as nn

from losses.keypoint_loss import KeypointLoss
from losses.heatmap_loss import HeatmapLoss
from losses.hand_pose_loss import HandPoseLoss
from losses.hand_shape_loss import HandShapeLoss
from losses.fingertip_orientation_loss import FingertipOrientationLoss
from losses.pinch_loss import PinchLoss


class CombinedLoss(nn.Module):

    # Epochs 1 - 30: 0 weight for retargeting losses
    # Epochs 31 - 55:  weight retargeting losses by 1x
    # Epochs 56 - 75: weight retargeting losses by 50x
    def __init__(self, a=1, b=1, c=50*1, d=50*0.1, e=50*0.1, f=50*1):
        super().__init__()

        self.keypoint_loss_func = KeypointLoss()
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
        self.f = f


    def forward(self, keypoint_preds, keypoint_labels, heatmap_preds, heatmap_labels, pred_positions, gt_positions):
        keypoint_loss = self.a * self.keypoint_loss_func(keypoint_preds, keypoint_labels)
        heatmap_loss = self.b * self.heatmap_loss_func(heatmap_preds, heatmap_labels)

        pose_loss = self.c * self.pose_loss_func(pred_positions, gt_positions)
        shape_loss = self.d * self.shape_loss_func(pred_positions, gt_positions)
        orientation_loss = self.e * self.orientation_loss_func(pred_positions, gt_positions)
        pinch_loss = self.f * self.pinch_loss_func(pred_positions, gt_positions)

        # Weight each loss function
        combined_loss = (
            keypoint_loss + 
            heatmap_loss + 
            pose_loss + 
            shape_loss + 
            pinch_loss + 
            orientation_loss
        )

        return combined_loss, keypoint_loss, heatmap_loss, pose_loss, shape_loss, pinch_loss, orientation_loss
