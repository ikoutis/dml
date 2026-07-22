"""Architectures must match the DML paper's Table 1 parameter counts
(CIFAR-100): ResNet-32 ~0.5M, MobileNet ~3.3M, WRN-28-10 ~36.5M."""

import pytest
import torch

from src.models import available_archs, build_model, count_params


@pytest.mark.parametrize("arch,lo,hi", [
    ("lenet", 0.08e6, 0.1e6),
    ("resnet32", 0.4e6, 0.55e6),
    ("mobilenet", 3.0e6, 3.5e6),
    ("wrn28x10", 36.0e6, 37.0e6),
])
def test_param_counts(arch, lo, hi):
    n = count_params(build_model(arch, num_classes=100))
    assert lo <= n <= hi, f"{arch}: {n:,} params outside [{lo:,}, {hi:,}]"


@pytest.mark.parametrize("arch", ["lenet", "resnet32", "mobilenet", "wrn28x10"])
def test_forward_shape(arch):
    m = build_model(arch, num_classes=100)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 100)


def test_registry():
    assert {"lenet", "resnet32", "mobilenet", "wrn28x10"} <= set(available_archs())
    with pytest.raises(ValueError):
        build_model("nope")
