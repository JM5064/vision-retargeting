"""Usage:
python -m models.test
"""

import torch
from torch.utils.data import DataLoader

from models.model import SimpleBaselines
from models.utils import DEVICE
from datasets.FreiHAND.freihand_dataset import FreiHAND
from losses.combined_loss_robot import CombinedLoss

from models.train_robot import validate
import albumentations as A

from GeoRT.geort.export import load_model
import pytorch_kinematics as pk


def test(model, ik_model, fk_model, test_loader, loss_func):
    print("Testing Model")
    metrics = validate(model, ik_model, fk_model, test_loader, loss_func, image_size=224)
    print("Testing Results")
    for metric in metrics:
        print(f'{metric}: {metrics[metric]}')

def load_keypoint_model(model_path, num_keypoints=21):
    model = SimpleBaselines(num_keypoints)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(DEVICE)

    model.eval()

    return model


if __name__ == "__main__":
    # --- INITIALIZE ROBOT MODELS ---

    ik_model = load_model('allegro_right_freihand', epoch=30)
    ik_model.model.to(DEVICE)

    # --- INITIALIZE KEYPOINT MODEL ---

    model_path = "runs_robot/2026.4.9-freihand-ik-55-larger-weights/epoch75.pt"
    model = load_keypoint_model(model_path)
    chain = pk.build_chain_from_urdf(open("config/allegro_hand_description_right.urdf").read()).to(device=DEVICE)

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'

    transform = A.Compose([
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    test_dataset = FreiHAND(
        images_dir=images_dir, 
        keypoints_json=keypoints_path, 
        intrinsics_json=intrinsics_path,
        scale_json=scale_path,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    
    test(model, ik_model, chain, test_loader, CombinedLoss())
