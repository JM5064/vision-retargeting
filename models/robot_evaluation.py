import numpy as np
import time

import torch
import albumentations as A

from models.model import SimpleBaselines
from models.utils import DEVICE
from datasets.FreiHAND.freihand_dataset import FreiHAND
from datasets.FreiHAND.heatmap_inference import marginal_soft_argmax
from models.math_utils import xyZ2XYZ, rotation_scale_normalize

from GeoRT.geort.export import load_model
import pytorch_kinematics as pk

import time
import numpy as np

import pybullet as p


def load_keypoint_model(model_path, num_keypoints=21):
    model = SimpleBaselines(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def denormalize_color(tensor, mean, std):
    """Convert normalized color back into regular rgb"""
    return tensor * std + mean


def inference(model, ik_model, dataset, index, image_size=224):
    """Run model inference and visualize keypoint, heatmap, and offset results"""

    image, gt_keypoints, _, K, wrist_depth, scale = dataset[index]

    gt_keypoints = torch.tensor(gt_keypoints).unsqueeze(dim=0)
    K = torch.tensor(K).unsqueeze(dim=0)
    wrist_depth = torch.tensor(wrist_depth).unsqueeze(0)
    scale = torch.tensor(scale).unsqueeze(0)

    # Evaluate model
    with torch.no_grad():
        input_tensor = torch.tensor(image).unsqueeze(0)

        heatmap_predictions = model(input_tensor)

        # Convert heatmap preds to keypoints
        keypoint_predictions = marginal_soft_argmax(heatmap_predictions, device='cpu')

    # Convert xyZ back to XYZ
    pred_XYZ = xyZ2XYZ(keypoint_predictions, image_size, K, wrist_depth, scale)
    labels_XYZ = xyZ2XYZ(gt_keypoints, image_size, K, wrist_depth, scale)

    # Root + rotate + scale normalize
    pred_XYZ = torch.tensor(rotation_scale_normalize(pred_XYZ, scale))
    labels_XYZ = torch.tensor(rotation_scale_normalize(labels_XYZ, scale))

    print(pred_XYZ.shape, "lol")

    # Calculate predicted and GT qpos
    pred_qpos = ik_model.forward_batch(pred_XYZ)
    gt_qpos = ik_model.forward_batch(labels_XYZ)

    pred_qpos = pred_qpos.cpu().detach().numpy().squeeze()
    gt_qpos = gt_qpos.cpu().detach().numpy().squeeze()

    return pred_qpos, gt_qpos


def render_allegro_hand(urdf_path, qpos):
    p.connect(p.GUI)
    
    p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=70, cameraPitch=-20, cameraTargetPosition=[0,0,0])

    # Load the hand
    hand_id = p.loadURDF(urdf_path, useFixedBase=True)

    # Identify only the active joints
    active_joint_ids = []
    for i in range(p.getNumJoints(hand_id)):
        joint_info = p.getJointInfo(hand_id, i)
        joint_type = joint_info[2]
        # JOINT_REVOLUTE is type 0
        if joint_type == p.JOINT_REVOLUTE:
            active_joint_ids.append(i)

    # Set joint states using the filtered IDs
    for i, joint_id in enumerate(active_joint_ids):
        if i < len(qpos):
            p.resetJointState(hand_id, joint_id, qpos[i])

    while True:
        p.stepSimulation()
        time.sleep(1/240)


if __name__ == "__main__":
    # --- INITIALIZE ROBOT MODELS ---

    ik_model = load_model('allegro_right_freihand', epoch=30)
    ik_model.model.to(DEVICE)
    chain = pk.build_chain_from_urdf(open("config/allegro_hand_description_right.urdf").read()).to(device=DEVICE)


    # --- INITIALIZE KEYPOINT MODEL ---

    model_path = "runs_robot/2026.4.13-freihand-ik-ablation/last.pt"
    # model_path = "runs/2026.4.6-z-normalized/last.pt"
    model = load_keypoint_model(model_path)

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
    
    pred_qpos, gt_qpos = inference(model, ik_model, dataset, index=0)

    urdf_file = "config/allegro_hand_description_right.urdf"

    render_allegro_hand(urdf_file, pred_qpos)
    