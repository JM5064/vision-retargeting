import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, keypoint_preds, keypoint_labels):
        # Calculate MSE between keypoint preds and labels
        mse = F.mse_loss(keypoint_preds, keypoint_labels)

        return mse
    
