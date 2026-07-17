"""Wide ResNet (Zagoruyko & Komodakis, 2016) for 32x32 inputs.

WRN-28-10 is the DML paper's large ("teacher-size") network: depth 28,
widening factor 10, ~36.5M parameters at 100 classes. Standard pre-activation
wide basic blocks with dropout (0.3 by default, per the WRN paper's CIFAR
setting, which the DML paper says it follows).
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ["wrn28x10"]


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                          bias=False))

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3,
                 num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(widths[1], n, stride=1,
                                       dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[2], n, stride=2,
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[3], n, stride=2,
                                       dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.zeros_(m.bias)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def wrn28x10(num_classes=100):
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3,
                      num_classes=num_classes)
