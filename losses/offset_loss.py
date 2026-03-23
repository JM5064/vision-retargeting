import torch.nn as nn


class OffsetLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, preds, heatmap_offset_labels, offset_masks):
        num_keypoints = heatmap_offset_labels.shape[1] // 3

        # Extract x offset and y offset maps
        x_offset_labels = heatmap_offset_labels[:, num_keypoints:2*num_keypoints, :, :]
        y_offset_labels = heatmap_offset_labels[:, 2*num_keypoints:3*num_keypoints, :, :]

        x_offset_preds = preds[:, num_keypoints:2*num_keypoints, :, :]
        y_offset_preds = preds[:, 2*num_keypoints:3*num_keypoints, :, :]

        # Get losses for x and y offsets
        x_offset_loss = self.get_offset_loss(x_offset_preds, x_offset_labels, offset_masks)
        y_offset_loss = self.get_offset_loss(y_offset_preds, y_offset_labels, offset_masks)

        return x_offset_loss + y_offset_loss


    def get_offset_loss(self, offset_preds, offset_labels, offset_masks):
        # Calculate squared errors between each pixel
        squared_errors = (offset_preds - offset_labels) ** 2

        # Mask out the parts of the offset map that don't matter
        masked_squared_errors = squared_errors * offset_masks

        loss = masked_squared_errors.sum() / (offset_masks.sum() + 1e-8)

        return loss
    