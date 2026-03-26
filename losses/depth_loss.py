import numpy as np

import torch
import torch.nn as nn


class DepthLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, preds, labels):
        if preds.shape != labels.shape:
            print("Uh oh, predictions and labels have differing shapes", preds.shape, labels.shape)

        # preds and labels have shape [batch_size, num_keypoints]

        # Calculate MSE between keypoints
        squared_errors = (preds - labels) ** 2
        mse = torch.mean(squared_errors, dim=(0, 1))

        return mse
