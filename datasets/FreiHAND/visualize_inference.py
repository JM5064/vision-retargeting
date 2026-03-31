"""Usage:
python -m datasets.FreiHAND.visualize_inference
"""

import numpy as np
import cv2
from PIL import Image
import time

import torch
import albumentations as A

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models.model import SimpleBaselines
from models.utils import DEVICE
from datasets.FreiHAND.freihand_dataset import FreiHAND
from datasets.FreiHAND.visualize_dataloader import add_keypoints, add_heatmap
from datasets.FreiHAND.heatmap_inference import marginal_heatmap_inference
from models.math_utils import xyZ2XYZ


def load_model(model_path, num_keypoints=21):

    model = SimpleBaselines(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def denormalize_color(tensor, mean, std):
    """Convert normalized color back into regular rgb"""
    return tensor * std + mean


def inference(model, dataset):
    """Run model inference and visualize keypoint, heatmap, and offset results"""
    plt.ion() # Interactive mode for matplotlib
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for image, gt_keypoints, _, K, wrist_depth, scale in dataset:
        gt_keypoints = torch.tensor(gt_keypoints).unsqueeze(dim=0)
        K = torch.tensor(K).unsqueeze(dim=0)
        wrist_depth = torch.tensor(wrist_depth).unsqueeze(0)
        scale = torch.tensor(scale).unsqueeze(0)

        # Evaluate model
        with torch.no_grad():
            input_tensor = torch.tensor(image).unsqueeze(0)

            heatmap_predictions = model(input_tensor)

            # Convert heatmap preds to keypoints
            keypoint_predictions = marginal_heatmap_inference(heatmap_predictions).squeeze()

        # Convert PIL image to numpy
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        # Recolor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = denormalize_color(image, mean=[0.472, 0.450, 0.413], std=[0.277, 0.272, 0.273])

        # Create keypoint image
        keypoints_image = add_keypoints(image, keypoint_predictions[:, :2])

        # Create heatmap image
        heatmap_image = add_heatmap(heatmap_predictions.squeeze())

        create_3d_visualization(keypoint_predictions.unsqueeze(dim=0), gt_keypoints, K, wrist_depth, scale, ax)
        plt.draw()

        cv2.imshow("Keypoints", keypoints_image)
        cv2.imshow("Heatmaps", heatmap_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


def create_3d_visualization(pred_keypoints, gt_keypoints, K, wrist_depth, scale, ax):
    pred_XYZ = xyZ2XYZ(pred_keypoints, 224, K, wrist_depth, scale)
    gt_XYZ = xyZ2XYZ(gt_keypoints, 224, K, wrist_depth, scale)

    plot_hand_3d(gt_XYZ, pred_XYZ, ax)


def get_skeleton_lines():
    """Returns the pairs of joint indices that form the hand skeleton."""
    # Connections: (start_joint, end_joint)
    return [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]


def plot_hand_3d(gt_3d: np.ndarray, pred_3d: np.ndarray, ax):
    gt_3d = gt_3d.squeeze()
    pred_3d = pred_3d.squeeze()

    ax.cla()  # 🔑 clear previous frame

    lines = get_skeleton_lines()

    # Joints
    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c="blue", s=30, label="GT")
    ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c="red", s=30, label="Pred")

    # Skeleton lines
    for i, j in lines:
        ax.plot(
            [gt_3d[i, 0], gt_3d[j, 0]],
            [gt_3d[i, 1], gt_3d[j, 1]],
            [gt_3d[i, 2], gt_3d[j, 2]],
            c="blue",
            linewidth=2,
        )
        ax.plot(
            [pred_3d[i, 0], pred_3d[j, 0]],
            [pred_3d[i, 1], pred_3d[j, 1]],
            [pred_3d[i, 2], pred_3d[j, 2]],
            c="red",
            linewidth=2,
        )

    ax.set_title("GT (blue) vs Pred (red)")
    ax.legend()

    # Equal scaling
    all_pts = np.vstack([gt_3d, pred_3d])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    max_range = (maxs - mins).max() / 2.0

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)


if __name__ == "__main__":
    model_path = "runs/2026.3.31-marginal-hm/last.pt"

    model = load_model(model_path)

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'

    transform = A.Compose([
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=transform)
    
    inference(model, dataset)