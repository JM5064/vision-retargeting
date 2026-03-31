"""
FreiHAND dataset with keypoints and heatmaps
"""

import os
import math
import numpy as np
import random
import cv2
import jsonyx as json
import time

import torch
from PIL import Image
from torch.utils.data import Dataset


class FreiHAND(Dataset):

    def __init__(
        self,
        images_dir, 
        keypoints_json, 
        intrinsics_json,
        scale_json,
        transform=None, 
        image_size=224, 
        heatmap_size=56,
        percent=1
    ):
        self.images_dir = images_dir
        self.image_names = sorted(os.listdir(images_dir))

        self.keypoints_json = json.load(open(keypoints_json, "r"))
        self.intrinsics_json = json.load(open(intrinsics_json, "r"))
        self.scale_json = json.load(open(scale_json, "r"))

        # Training images are duplicated 4 times with differing backgrounds
        num_unique_images = len(self.keypoints_json)

        # Calculate how many images to use
        num_to_take = int(num_unique_images * abs(percent))

        if percent > 0:
            # Grab the first X percent
            subset_range = range(0, num_to_take)
        elif percent < 0:
            # Grab the last X percent
            subset_range = range(num_unique_images - num_to_take, num_unique_images)

        # Data contains (image name, xyz keypoints, intrinsics matrix K) for each image
        self.data = []
        for i in range(len(self.image_names)):
            if i % num_unique_images not in subset_range:
                continue

            self.data.append(
                (
                    self.image_names[i], 
                    np.array(self.keypoints_json[i % num_unique_images], dtype=np.float32),
                    np.array(self.intrinsics_json[i % num_unique_images], dtype=np.float32),
                    np.array(self.scale_json[i % num_unique_images], dtype=np.float32)
                )
            )

        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        """Get a sample from the dataset by index
        args:
            index: number

        returns:
            np_image: ndarray of the input image
            tensor_keypoints: [[x1, y1, z1], ...] keypoints
            heatmaps: ndarray of the heatmaps
        """
        image_name, keypoints, K, scale = self.data[index]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path)

        # Scale normalize 3D keypoints
        keypoints_scaled = (1 / scale) * keypoints

        # Project keypoints
        projected_keypoints = self.project_keypoints(keypoints_scaled, K)

        # Normalize Z wrt root
        wrist_depth = projected_keypoints[0][2]
        projected_keypoints[:, 2] -= wrist_depth

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), keypoints=projected_keypoints[:, :2])
            image = transformed['image']
            projected_keypoints[:, :2] = np.array(transformed['keypoints'])

        # Normalize xy between 0-1
        normalized_keypoints = self.normalize_keypoints(projected_keypoints)
        
        # Convert keypoints to tensor
        tensor_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float32)

        # Create heatmaps
        # heatmaps = self.create_heatmaps(projected_keypoints)
        heatmaps = self.create_marginal_heatmaps(tensor_keypoints)

        # Convert image to numpy
        np_image = np.array(image)

        return np_image, tensor_keypoints, heatmaps, K, wrist_depth, scale


    def project_keypoints(self, keypoints, K):
        """Project 3D keypoints onto 2D image space, leaving Z coordinate the same
        args:
            keypoints: np.array([[X1, Y1, Z1], ...])
            K: np.array([3D intrinsics matrix])
        
        returns:
            keypoints_xyZ: np.array([[x1, y1, Z1], ...])
        """

        # Project keypoints
        keypoints_xyZ = (keypoints @ K.T) / keypoints[:, 2:3]
        keypoints_xyZ[:, 2] = keypoints[:, 2]

        return keypoints_xyZ

        
    def normalize_keypoints(self, keypoints):
        """Normalizes keypoints between 0-1
        args:
            keypoints: np.array([[x1, y1, Z1], ...])

        returns:
            normalized_keypoints: [[x1', y1', z1'], ...] normalized keypoints
            wrist_depth: depth normalization scalar
        """
        
        normalized_keypoints = keypoints.copy()

        # Normalize xy
        normalized_keypoints[:, :2] /= self.image_size

        return normalized_keypoints
    

    def create_heatmaps(self, keypoints, sigma=1.0):
        """Creates xy heatmaps for each joint
        args:
            keypoints: np.array([[x1, y1, Z1], ...])
            sigma: standard deviation for Gaussian heatmap

        returns:
            heatmaps: np.array of (num_keypoints, heatmap_size, heatmap_size) heatmaps
        """
        num_keypoints = len(keypoints)

        # Create coordinate grid
        x_grid = np.arange(self.heatmap_size, dtype=np.float32)
        y_grid = np.arange(self.heatmap_size, dtype=np.float32)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Scale keypoints down to heatmap size
        scale = self.heatmap_size / self.image_size     
        scaled_keypoints = keypoints[:, :2] * scale

        # Reshape keypoints for broadcasting
        xs = scaled_keypoints[:, 0].reshape(num_keypoints, 1, 1)
        ys = scaled_keypoints[:, 1].reshape(num_keypoints, 1, 1)

        # Calculate Offsets
        x_offsets = xs - xx
        y_offsets = ys - yy

        # Calculate squared distances
        dist_sq = x_offsets ** 2 + y_offsets ** 2

        # Create heatmaps
        heatmaps = np.exp(-dist_sq / (2 * sigma ** 2))

        return heatmaps


    def create_marginal_heatmaps(self, keypoints, sigma=1.0, z_min=-9.0, z_max=9.0):
        """Creates xy, xz, yz heatmaps for each joint
        args:
            keypoints: np.array([[x1, y1, Z1], ...])
            sigma: standard deviation for Gaussian heatmap
            z_min, z_max: min/max values of scale normalized Z

        returns:
            heatmaps: np.array of (num_keypoints*3, heatmap_size, heatmap_size) heatmaps
        """

        # Define the range of z values
        z_range = z_max - z_min
        
        # Create coordinate grid
        grid = np.arange(self.heatmap_size, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid)

        # Scale x, y to [0, heatmap_size]
        scaled_x = keypoints[:, 0] * (self.heatmap_size - 1)
        scaled_y = keypoints[:, 1] * (self.heatmap_size - 1)

        # Scale Z: shift by z_min, normalize to [0, 1], scale to [0, heatmap_size]
        scaled_z = ((keypoints[:, 2] - z_min) / z_range) * (self.heatmap_size - 1)

        # Reshape for broadcasting
        xs, ys, zs = [c.reshape(-1, 1, 1) for c in [scaled_x, scaled_y, scaled_z]]

        # Create xy heatmap
        xy_heatmap = np.exp(-((xs - xx)**2 + (ys - yy)**2) / (2 * sigma**2))
        
        # Create xz heatmap
        xz_heatmap = np.exp(-((xs - xx)**2 + (zs - yy)**2) / (2 * sigma**2))
        
        # Create zy heatmap
        zy_heatmap = np.exp(-((zs - xx)**2 + (ys - yy)**2) / (2 * sigma**2))
        
        return np.concatenate([xy_heatmap, xz_heatmap, zy_heatmap], axis=0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import albumentations as A
    # from models.math_utils import reproject_xyZ2XYZ

    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    intrinsics_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    scale_json = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'

    transform = A.Compose([
        # A.Rotate(limit=[-45, 45]),
        # A.ColorJitter(brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.05, 0.05]),
        A.Normalize(mean=[0.472, 0.450, 0.413],
                            std=[0.277, 0.272, 0.273]),
        A.ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    freihand = FreiHAND(images_dir, keypoints_json, intrinsics_json, scale_json, transform=transform)

    dl = DataLoader(freihand, batch_size=1, shuffle=False)

    for item in dl:
        # Orig:
        # [-0.06876980513334274, 0.0244811549782753, 0.9476402997970581], 
        # [-0.05068895220756531, -0.008157476782798767, 0.9384765625]

        for i in range(len(item)):
            # print(item[i])
            # print("--------------------------------")



            if i == 1: # keypoints
                keypoints = item[i]
                # Unnormalize xy
                # keypoints[:, :, :2] *= 224

                # Unnormalize depth
                # keypoints[:, :, 2] += item[5]
                # xn = keypoints[0][0][0]
                # yn = keypoints[0][0][1]
                # Zn = keypoints[0][0][2]
                # xm = keypoints[0][5][0]
                # ym = keypoints[0][5][1]
                # Zm = keypoints[0][5][2]

                # a = (xn - xm) ** 2 + (yn - ym) ** 2
                # b = Zn*(xn**2 + yn**2 - xn*xm - yn*ym) + Zm*(xm**2 + ym**2 - xn*xm - yn*ym)
                # c = (xn*Zn - xm*Zm) ** 2 + (yn*Zn - ym*Zm) ** 2 + (Zn - Zm) ** 2 - 1

                # root = 0.5 * (-b + (b**2 - 4*a*c) ** 0.5) / a
                # print(a, b, c)
                # print((b**2 - 4*a*c))
                # print("Calculated:", root, " Actual:", item[5])

                # Reproject to XYZ
                # keypoints = reproject_xyZ2XYZ(keypoints, item[4])

                # Undo scaling
                # keypoints *= item[6]

                print(torch.max(torch.abs(keypoints[:, :, 2])))
                print("--------------------------------")

        break