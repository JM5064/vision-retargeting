import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
import json
import time


def visualize_keypoints(images_dir, keypoints_path, scale_path, intrinsics_path):
    image_paths = sorted(os.listdir(images_dir))
    xyz_json = json.load(open(keypoints_path, 'r'))
    scale_json = json.load(open(scale_path, 'r'))
    intrinsics_json = json.load(open(intrinsics_path, 'r'))

    for i in range(0, 100):
        image_path = os.path.join(images_dir, image_paths[i])
        image = cv2.imread(image_path)

        h, w, _ = image.shape

        scale = scale_json[i]
        K = np.array(intrinsics_json[i])

        # Plot keypoints
        keypoints = np.array(xyz_json[i])

        keypoints = (keypoints @ np.transpose(K)) / keypoints[:, 2:3]

        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 1, (0, 0, 255), -1)

        cv2.imshow("Image", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def visualize_keypoints_3d(keypoints_path):
    xyz_json = json.load(open(keypoints_path, 'r'))

    for i in range(0, 100):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        keypoints = np.array(xyz_json[i])

        ax.scatter(keypoints[:,0], keypoints[:,1], keypoints[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_box_aspect([1,1,1])
        plt.show()


def visualize_vertices(vertices_path):
    start = time.time()
    print(start)
    vertices_json = json.load(open(vertices_path, 'r'))
    end = time.time()
    print(end-start)

    # v = np.array(vertices_json[0])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(v)

    # o3d.visualization.draw_geometries([pcd])



def main():
    images_dir = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation/rgb'
    keypoints_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_xyz.json'
    scale_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_scale.json'
    intrinsics_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_K.json'
    vertices_path = 'datasets/FreiHAND/FreiHAND/FreiHAND_pub_v2_eval/evaluation_verts.json'

    visualize_keypoints(images_dir, keypoints_path, scale_path, intrinsics_path)
    # visualize_keypoints_3d(keypoints_path)
    # visualize_vertices(vertices_path)


if __name__ == "__main__":
    main()