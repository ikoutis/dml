"""Model zoo for the mutual-learning cohort.

Architectures follow the DML paper's CIFAR-100 table of networks
(Zhang et al., CVPR 2018, Table 1):

    name        class            params (100 classes)
    resnet20    ResNet-20        ~0.28M
    resnet32    ResNet-32        ~0.47M   (the paper's small net)
    resnet56    ResNet-56        ~0.86M
    mobilenet   MobileNet(v1)    ~3.2M    (the paper's compact net)
    wrn28x10    WRN-28-10        ~36.5M   (the paper's large net)

`build_model(arch, num_classes)` is the single entry point used by the
cohort builder.
"""

from .resnet_cifar import resnet20, resnet32, resnet56, resnet110
from .wrn import wrn28x10
from .mobilenet import mobilenet

_REGISTRY = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenet": mobilenet,
    "wrn28x10": wrn28x10,
}


def available_archs():
    return sorted(_REGISTRY)


def build_model(arch: str, num_classes: int = 100):
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {available_archs()}")
    return _REGISTRY[arch](num_classes=num_classes)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
