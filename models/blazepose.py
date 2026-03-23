import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class BlazePose(nn.Module):

    def __init__(self, num_keypoints):
        super().__init__()

        self.num_keypoints = num_keypoints

        self.bb1 = BlazeBlock(3, 16)        # 224 x 224 x 3 -> 112 x 112 x 16
        self.bb2 = BlazeBlock(16, 32)       # 112 x 112 x 16 -> 56 x 56 x 32
        self.bb3 = BlazeBlock(32, 64)       # 56 x 56 x 32 -> 28 x 28 x 64
        self.bb4 = BlazeBlock(64, 128)      # 28 x 28 x 64 -> 14 x 14 x 128
        self.bb5 = BlazeBlock(128, 192)     # 14 x 14 x 128 -> 7 x 7 x 192

        self.hb1 = HeatmapBlock(192, 32)    # 7 x 7 x 192 -> 7 x 7 x 32
        self.hb2 = HeatmapBlock(128, 32)    # 14 x 14 x 128 + (7 x 7 x 32)+ -> 14 x 14 x 32
        self.hb3 = HeatmapBlock(64, 32)    # 28 x 28 x 64 + (14 x 14 x 32)+ -> 28 x 28 x 32
        # TODO: this one doesnt actually need a conv since it's 28 -> 28
        self.hb4 = HeatmapBlock(32, 32)     # 56 x 56 x 32 + (28 x 28 x 32)+ -> 56 x 56 x 32

        # Heatmap output (each keypoint has its own heatmap, x, and y offset maps)
        # num_keypoints heatmaps
        # num_keypoints x-offset maps
        # num_keypoints y-offset maps   -> 64 x 64 x num_keypoints * 3
        self.outH = nn.Conv2d(in_channels=32, out_channels=num_keypoints * 3, kernel_size=3, padding='same')


        self.pb1 = PoseBlock(32, 64)        # 56 x 56 x 28 -> 28 x 32 x 64
        self.pb2 = PoseBlock(64, 128)       # 28 x 28 x 64 -> 14 x 14 x 128
        self.pb3 = PoseBlock(128, 192)      # 14 x 14 x 128 -> 7 x 7 x 192
        self.pb4 = PoseBlock(192, 192)      # 7 x 7 x 192 -> 4 x 4 x 192
        self.pb5 = PoseBlock(192, 192)      # 4 x 4 x 192 -> 2 x 2 x 192

        # Regression output
        # 2 x 2 x 192 -> 1 x 1 x num_keypoints * 3 (x, y, visibility)
        self.outR = nn.Conv2d(in_channels=192, out_channels=num_keypoints * 3, kernel_size=2, padding=0)
    

    def forward(self, x):
        """
        x -> input tensor, and eventually regression path
        y -> center path
        z -> heatmap path

        Returns:
            x: Regression output
            z5: Heatmap + offset map output
        """

        x = self.bb1(x)

        # Center BlazeBlock path
        y1 = self.bb2(x)
        y2 = self.bb3(y1)
        y3 = self.bb4(y2)
        y4 = self.bb5(y3) # last shared layer between regression and heatmap paths

        # Heatmap path
        z1 = self.hb1(y4, None)
        z2 = self.hb2(y3, z1)
        z3 = self.hb3(y2, z2)
        z4 = self.hb4(y1, z3)

        z5 = self.outH(z4)

        # Regression path
        # Detach to stop gradients from flowing back
        x = self.pb1(y1.detach() + z4.detach())
        x = self.pb2(x + y2.detach())
        x = self.pb3(x + y3.detach())
        x = self.pb4(x + y4.detach())
        x = self.pb5(x)

        x = self.outR(x)
        x = torch.flatten(x, start_dim=1)       # [batch, num_keypoints * 3, 1, 1] -> [batch, num_keypoints * 3]
        x = x.view(-1, self.num_keypoints, 3)   # [batch, num_keypoints * 3] -> [batch, num_keypoints, 3]

        return x, z5


class BlazeBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Downsample
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Regular convolution for good measure
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same")
        # self.bn22 = nn.BatchNorm2d(out_channels)

        # Depthwise seperable convolution
        # Add batch norm after pointwise? (convnext doesnt do this)
        self.dw = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, groups=out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pw1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, padding='same')
        self.pw2 = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1, padding='same')


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu6(x)

        x_resid = x

        # x = self.conv2(x)
        # x = self.bn22(x)
        # x = F.relu6(x)

        x = self.dw(x)
        x = self.bn2(x)

        x = F.relu6(x)
        x = self.pw1(x)
        x = self.pw2(x)

        x = x + x_resid

        return x
    

class PoseBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Depthwise seperable convolution
        self.dw = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, groups=out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pw1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, padding='same')
        self.pw2 = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1, padding='same')


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu6(x)

        x_resid = x

        x = self.dw(x)
        x = self.bn2(x)

        x = self.pw1(x)
        x = F.relu6(x)
        x = self.pw2(x)

        x = x + x_resid

        return x


class HeatmapBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, midpath_x, prev_x):
        x = midpath_x

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu6(x)

        if prev_x is not None:
            prev_x = self.upsample(prev_x)

            x = x + prev_x

        return x


if __name__ == "__main__":
    model = BlazePose(21)
    profile = summary(model, input_size=(3, 224, 224))