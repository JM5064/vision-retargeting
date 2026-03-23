import numpy as np

import torch
import torch.nn as nn


class RegressionLoss(nn.Module):

    def __init__(self, alpha=2.0):
        super().__init__()

        self.alpha = alpha


    def forward(self, preds, labels):
        if preds.shape != labels.shape:
            print("Uh oh, predictions and labels have differing shapes", preds.shape, labels.shape)

        # preds and labels have shape [batch_size, num_keypoints, 3]

        # Calculate MSE between keypoints
        squared_errors = (preds - labels) ** 2
        mean_squared_errors = torch.mean(squared_errors, dim=(0, 1))

        # Split MSEs into xy and z
        xy_loss = mean_squared_errors[0] + mean_squared_errors[1]
        z_loss = mean_squared_errors[2]

        # Calculate total loss
        total_loss = xy_loss + z_loss * self.alpha

        return total_loss


if __name__ == "__main__":
    # Batch of 2 images, 5 keypoints each
    outputs = torch.tensor([
        [[10, 20, 0], [30, 40, 1], [50, 60, 0], [70, 80, 0], [90, 100, 0]],
        [[11, 21, 0], [31, 41, 1], [51, 61, 0], [71, 81, 0.9], [91, 101, 0.5]]
    ])

    labels = torch.tensor([
        [[10, 20, 0], [29, 41, -1], [49, 59, 0], [70, 79, 0], [88, 102, 0]],
        [[12, 22, 1], [29, 39, 1], [52, 62, -1], [70, 82, 0], [89, 99, 0]]
    ])
    outputs = torch.tensor([
        [[1,1,1], [2,2,2]],
        [[3,3,3], [4,4,4]]
    ], dtype=torch.float32)
    labels = torch.tensor([
        [[2,2,2], [2,2,2]],
        [[3,3,3], [4,4,4]]
    ], dtype=torch.float32)

    criterion = RegressionLoss()

    loss = criterion(outputs, labels)
    print("Total loss:", loss)



