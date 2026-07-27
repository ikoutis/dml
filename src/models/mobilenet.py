"""MobileNet v1 (Howard et al., 2017) adapted for 32x32 inputs.

The DML paper's compact net (~3.3M parameters on CIFAR-100). Standard CIFAR
adaptation: the stem convolution uses stride 1 (instead of 2), keeping four
downsampling stages for a final 2x2 feature map at 32x32 input.
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mobilenet"]


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3,
                               stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)))


class MobileNet(nn.Module):
    # (out_planes, stride) per depthwise-separable block.
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2),
           (1024, 1)]

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers = []
        in_planes = 32
        for out_planes, stride in self.cfg:
            layers.append(DepthwiseSeparable(in_planes, out_planes, stride))
            in_planes = out_planes
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def mobilenet(num_classes=100):
    return MobileNet(num_classes=num_classes)
