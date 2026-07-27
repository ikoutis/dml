"""LeNet-5 adapted for 32x32 RGB inputs — the weak-learner probe (M7-L).

Same wiring as the classic LeNet-5 (two 5x5 conv blocks 6->16, then FC
120 -> 84 -> classes) and the knowledge-diffusion repo's LeNet, but with
ReLU + max-pool in place of the 1998 sigmoid + avg-pool so it trains
under the suite's shared recipe (SGD 0.1, momentum, weight decay, step
decay). The original sigmoid variant needs plain SGD at lr ~0.9 and
would confound the capacity question with an optimization artifact;
capacity class (~0.09M params, ~5x below ResNet-20) is what the probe
varies, not the training recipe.
"""

import torch.nn as nn

__all__ = ["lenet"]


class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),   # 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 16x16
            nn.Conv2d(6, 16, kernel_size=5),             # 12x12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 6x6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def lenet(num_classes=100):
    return LeNet(num_classes=num_classes)
