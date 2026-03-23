import torch
import torch.nn as nn
import pytorch_kinematics as pk


class DFKLayer(nn.Module):

    def __init__(self, joint_angles) -> None:
        super().__init__()

        self.joint_angles = joint_angles
        self.chain = pk.build_chain_from_urdf(open("models/GeoRT/allegro_hand_description_right.urdf").read())



    def scale_to_limits(self, q_raw):
        q_min = torch.tensor([
            -0.47, -0.196, -0.174, -0.227,  # Index
            -0.47, -0.196, -0.174, -0.227,  # Middle
            -0.47, -0.196, -0.174, -0.227,  # Ring
            0.263, -0.105, -0.189, -0.162   # Thumb
        ])

        q_max = torch.tensor([
            0.47,  1.61,   1.709,  1.618,  # Index
            0.47,  1.61,   1.709,  1.618,  # Middle
            0.47,  1.61,   1.709,  1.618,  # Ring
            1.396, 1.163,  1.644,  1.719   # Thumb
        ])

        # Scale predicted q_raw between the min and max angles of each joint
        return q_min + torch.sigmoid(q_raw) * (q_max - q_min)
    

    def forward_kinematics(self, q):
        # Calculate forward kinematics for joints
        transforms = self.chain.forward_kinematics(q)

        return transforms
    


if __name__ == "__main__":
    pred_joint_angles = torch.tensor([[0] * 16])
    dfklayer = DFKLayer(joint_angles=pred_joint_angles)

    scaled = dfklayer.scale_to_limits(pred_joint_angles)
    transforms = dfklayer.forward_kinematics(scaled)

    print(len(transforms))
    for link in transforms:
        print(link)
        # print(transforms[link])
        print("Position(s): ", transforms[link].get_matrix()[:, :3, 3])
        print()





