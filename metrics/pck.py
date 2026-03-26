import torch
from models.math_utils import reproject_xyZ2XYZ


def pck_2D(preds_kp, labels_kp, percent, norm_p1, norm_p2):
    """Calculate pck for 2D keypoints. A keypoint is considered correct if it's within percent% of the normalization size
    args:
        preds_kp: [batch, num_keypoints, 2]
        labels_kp: [batch, num_keypoints, 2]
        percent: number, percentage for pck threshold
        norm_p1: index of 1st normalization point
        norm_p2: index of 2nd normalization point
            -> PCK is normalized by calculating the % of points within ||norm_p2 - norm_p1|| * percent

    returns:
        pck: % correct keypoints according to threshold    
    """

    # eg for MPII: p1 = 3 (left hip), p2 = 12 (right shoulder) -> torso size
    p1_labels = labels_kp[:, norm_p1, :]
    p2_labels = labels_kp[:, norm_p2, :]

    # Calculate the normalization size for each label
    # eg for MPII: torso size
    normalization_size_labels = torch.norm(p1_labels - p2_labels, dim=1)

    # Calculate distances between predicted keypoints and labels
    distances = torch.norm(preds_kp - labels_kp, dim=2)

    # Normalize distances wrt normalization size instead of image size
    # [:, None] reshapes from 1 * N to N * 1
    norm_distances = distances / normalization_size_labels[:, None]

    # Count as correct if the distance is within pck% of the normalization size
    correct = (norm_distances < percent).float()

    # Calculate pck as the number of correct keypoints over the total number of keypoints (aka mean)
    pck = correct.mean()

    return pck


def pck_2D_visibile(preds_kp, labels_kp, percent, norm_p1, norm_p2):
    """Calculate pck for 2D keypoints. A keypoint is considered correct if it's within percent% of the normalization size
    A keypoint of value [-1, -1] is considered not visible, and is thus masked in PCK calculation
    args:
        preds_kp: [batch, num_keypoints, 2]
        labels_kp: [batch, num_keypoints, 2]
        percent: number, percentage for pck threshold
        norm_p1: index of 1st normalization point
        norm_p2: index of 2nd normalization point
            -> PCK is normalized by calculating the % of points within ||norm_p2 - norm_p1|| * percent

    returns:
        pck: % correct keypoints according to threshold    
    """

    # eg for MPII: p1 = 3 (left hip), p2 = 12 (right shoulder) -> torso size
    p1_labels = labels_kp[:, norm_p1, :]
    p2_labels = labels_kp[:, norm_p2, :]

    # Make sure that not labeled points (-1) are not included in the correctness calculation (use x for the visibility)
    visibilities = labels_kp[:, :, 0]
    visibility_mask = (visibilities != -1).float()

    # Calculate the normalization size for each label
    # eg for MPII: torso size
    normalization_size_labels = torch.norm(p1_labels - p2_labels, dim=1)

    # Mask out images where normalization points aren't visible or size is too small
    p1_visibilities = (p1_labels[:, 0] != -1).float()
    p2_visibilities = (p2_labels[:, 0] != -1).float()
    valid_normalizations_mask = p1_visibilities * p2_visibilities * (normalization_size_labels > 0.01).float()

    # Calculate distances between predicted keypoints and labels
    distances = torch.norm(preds_kp - labels_kp, dim=2)
    distances = distances * visibility_mask

    # Normalize distances wrt normalization size instead of image size
    # [:, None] reshapes from 1 * N to N * 1
    norm_distances = distances / normalization_size_labels[:, None]
    norm_distances = norm_distances * valid_normalizations_mask[:, None]

    # Count as correct if the distance is within pck% of the normalization size
    mask = visibility_mask * valid_normalizations_mask[:, None]
    correct = (norm_distances < percent).float() * mask

    # Calculate pck as the number of correct keypoints over the total number of valid keypoints
    pck = correct.sum() / mask.sum()

    return pck


def pck_3D(preds_kp, labels_kp, threshold, K, root_depths, image_size):
    """Calculate pck for 3D keypoints. A keypoint is considered correct if it's within a millimeter threshold
    args:
        preds_kp: [batch, num_keypoints, 2]
        labels_kp: [batch, num_keypoints, 2]
        threshold: number, in millimeters
        K: [batch, instrinsics matrix
        root_depths: [batch, depth normalization scalar
        image_size: int

    returns:
        pck: % correct keypoints according to threshold    
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

    # Calculate distances between preds and GT in millimeters
    distances = torch.norm(preds_XYZ - labels_XYZ, dim=2) * 1000

    # Count as correct if distance is within threshold
    correct = (distances < threshold).float()

    # Calculate pck
    pck = correct.mean()

    return pck
