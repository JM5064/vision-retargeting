import torch
from metrics.math import reproject_xyZ2XYZ


def mpjpe_3D(preds_kp, labels_kp, K, root_depths, image_size):
    """Calculates MPJPE (mean per joint position error)
    args:
        preds_kp: torch tensor [batch, num_keypoints, 3] predicted keypoints
        labels_kp: torch tensor [batch, num_keypoints, 3] GT keypoints
        K: [batch, instrinsics matrix
        root_depths: [batch, depth normalization scalar
        image_size: int

    returns:
        mpjpe in millimeters
    """

    preds_kp = preds_kp.clone()
    labels_kp = labels_kp.clone()

    # Unnormalize x, y
    preds_kp[:, :, :2] *= image_size
    labels_kp[:, :, :2] *= image_size

    # Unnormalize z
    preds_kp[:, :, 2] += root_depths[:, None]
    labels_kp[:, :, 2] += root_depths[:, None]

    # Reproject coordinates
    preds_XYZ = reproject_xyZ2XYZ(preds_kp, K)
    labels_XYZ = reproject_xyZ2XYZ(labels_kp, K)

    # Calculate MPJPE
    distances = torch.norm(preds_XYZ - labels_XYZ, dim=-1)

    mpjpe = distances.mean() * 1000 # multiply by 1000 for millimeters

    return mpjpe


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from datasets.FreiHAND.freihand_dataset import FreiHAND

    images_dir =      'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/rgb'
    keypoints_json =  'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_xyz.json'
    intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_K.json'
    scale_json =      'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2/FreiHAND64/training_scale.json'

    freihand = FreiHAND(images_dir, keypoints_json, intrinsics_json, scale_json, transform=None)

    dl = DataLoader(freihand, batch_size=1, shuffle=False)

    for np_image, tensor_keypoints, heatmaps, offset_masks, K, wrist_depth in dl:
        preds_kp = tensor_keypoints.clone()
        labels_kp = tensor_keypoints.clone()

        preds_kp[:, 0, 0] = 0

        print("preds_kp", preds_kp)
        print("labels_kp", labels_kp)
        print("K", K)
        print("wrist_depth", wrist_depth)

        mpjpe = mpjpe_3D(preds_kp, labels_kp, K, wrist_depth, 224)
        print("mpjpe:", mpjpe)

        break

