"""
Visualizes keypoints and heatmaps from dataloader
"""

import cv2
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import albumentations as A


def visualize(dataset):
    for image, keypoints, heatmaps, _, _, _ in dataset:
        image = np.array(image)
        image = image.transpose(1, 2, 0) # transpose from 3 x 224 x 224 -> 224 x 224 x 3
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Unnormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        image = image * std + mean
        
        # Create keypoint image
        keypoints_image = add_keypoints(image, keypoints)

        # Create heatmap and offset images
        heatmap_image = add_heatmap(heatmaps)

        cv2.imshow("Keypoints", keypoints_image)
        cv2.imshow("Heatmaps", heatmap_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def add_keypoints(image, keypoints, joint_names=None):
    image = image.copy()
    keypoints = keypoints.clone()
    h, w, _ = image.shape
        
    # Unnormalize keypoints
    keypoints[:, 0] *= w
    keypoints[:, 1] *= h

    for i in range(len(keypoints)):
        keypoint = keypoints[i]
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 1, (0, 0, 255), -1)

        if joint_names:
            cv2.putText(image, joint_names[int(i)], (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    return image
    

def add_heatmap(heatmaps):
    """Create heatmap for each image """
    num_keypoints = heatmaps.shape[0]

    # Combine heatmaps
    combined_heatmap = np.zeros(heatmaps[0].shape)
    for i in range(num_keypoints):
        # if i == 1:
        combined_heatmap += np.array(heatmaps[i])

    return combined_heatmap


def main():
    from freihand_dataset import FreiHAND

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'

    train_transform = A.Compose([
        A.Rotate(limit=[-45, 45]),
        A.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.05, 0.05]),
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=train_transform)

    visualize(dataset)


if __name__ == "__main__":
    main()
    