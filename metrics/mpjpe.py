import torch
from models.math_utils import reproject_xyZ2XYZ


def mpjpe_3D(preds_XYZ, labels_XYZ):
    """Calculates MPJPE (mean per joint position error)
    args:
        preds_XYZ: torch tensor [batch, num_keypoints, 3] predicted 3D keypoints
        labels_XYZ: torch tensor [batch, num_keypoints, 3] GT 3D keypoints

    returns:
        mpjpe in millimeters
    """

    # Calculate MPJPE
    distances = torch.norm(preds_XYZ - labels_XYZ, dim=-1)

    mpjpe = distances.mean() * 1000 # multiply by 1000 for millimeters

    return mpjpe


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets.FreiHAND.freihand_dataset import FreiHAND
    from models.math_utils import xyZ2XYZ

    images_dir =      'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/rgb'
    keypoints_json =  'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_xyz.json'
    intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_K.json'
    scale_json =      'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_scale.json'

    freihand = FreiHAND(images_dir, keypoints_json, intrinsics_json, scale_json, transform=None)

    dl = DataLoader(freihand, batch_size=1, shuffle=False)

    for np_image, tensor_keypoints, heatmaps, K, wrist_depth, scale in dl:
        preds_kp = tensor_keypoints.clone()
        labels_kp = tensor_keypoints.clone()

        preds_kp[:, 0, 2] = 0.2
        print(preds_kp)
        print(labels_kp)

        predsXYZ = xyZ2XYZ(preds_kp, 224, K, wrist_depth, scale)
        labelsXYZ = xyZ2XYZ(labels_kp, 224, K, wrist_depth, scale)

        mpjpe = mpjpe_3D(predsXYZ, labelsXYZ)
        print("mpjpe:", mpjpe)

        break

