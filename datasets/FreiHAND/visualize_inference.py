"""Usage:
python -m datasets.FreiHAND.visualize_inference
"""

import numpy as np
import cv2
from PIL import Image
import time

import torch
from torchvision.transforms import v2

from models.BlazePoseFreiHAND.blazepose import BlazePose
from datasets.FreiHAND.freihand_dataset import FreiHAND
from datasets.FreiHAND.visualize_dataloader import add_keypoints, add_heatmap_offsets


def load_model(model_path, device, num_keypoints=21):
    print("Using device", device)

    model = BlazePose(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def denormalize_color(tensor, mean, std):
    """Convert normalized color back into regular rgb"""
    return tensor * std + mean


def inference(model, dataset):
    """Run model inference and visualize keypoint, heatmap, and offset results"""
    for image, _, _, _, _, _ in dataset:
        # Evaluate model
        with torch.no_grad():
            input_tensor = torch.tensor(image).unsqueeze(0)

            keypoint_predictions, heatmap_offset_predictions = model(input_tensor)

            keypoint_predictions = np.array(keypoint_predictions.squeeze())
            heatmap_offset_predictions = np.array(heatmap_offset_predictions.squeeze())

        # Convert PIL image to numpy
        image = np.array(image)
        image = np.transpose(image, (1, 2, 0))

        # Recolor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = denormalize_color(image, mean=[0.472, 0.450, 0.413], std=[0.277, 0.272, 0.273])

        # Create keypoint image
        keypoints_image = add_keypoints(image, keypoint_predictions)

        # Create heatmap and offset images
        heatmap_image, x_offset_image, y_offset_image = add_heatmap_offsets(heatmap_offset_predictions)

        cv2.imshow("Keypoints", keypoints_image)
        cv2.imshow("Heatmaps", heatmap_image)
        cv2.imshow("x offsets", x_offset_image)
        cv2.imshow("y offsets", y_offset_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "models/BlazePoseFreiHAND/runs/1epoch/best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(model_path, device)

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    vertices_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_verts.json'

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
    ])

    dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=transform)
    
    inference(model, dataset)