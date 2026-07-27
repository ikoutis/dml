"""Cohort assembly: parse a cohort spec into K model slots.

Spec syntax: comma-separated ``arch:count`` groups, e.g.
    "resnet32:8"                 homogeneous small-net cohort
    "wrn28x10:4,resnet32:4"      the heterogeneous (large+small) cohort
    "wrn28x10:1,resnet32:1"      the paper's Table-2 WRN/ResNet pair
Slot order follows the spec left to right; slot names are m00, m01, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .models import build_model, count_params


@dataclass
class Slot:
    index: int
    name: str
    arch: str
    model: torch.nn.Module
    n_params: int


def parse_cohort_spec(spec: str) -> List[str]:
    """'wrn28x10:4,resnet32:4' -> ['wrn28x10']*4 + ['resnet32']*4."""
    archs: List[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            arch, cnt = part.rsplit(":", 1)
            archs.extend([arch.strip()] * int(cnt))
        else:
            archs.append(part)
    if not archs:
        raise ValueError(f"Empty cohort spec '{spec}'.")
    return archs


def cohort_slug(spec: str) -> str:
    """Filename-safe slug: 'wrn28x10:4,resnet32:4' -> 'wrn28x10x4-resnet32x4'."""
    parts = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            arch, cnt = part.rsplit(":", 1)
            parts.append(f"{arch.strip()}x{int(cnt)}")
        else:
            parts.append(part)
    return "-".join(parts)


def build_cohort(spec: str, num_classes: int, device) -> List[Slot]:
    archs = parse_cohort_spec(spec)
    slots = []
    for i, arch in enumerate(archs):
        model = build_model(arch, num_classes=num_classes).to(device)
        slots.append(Slot(index=i, name=f"m{i:02d}", arch=arch, model=model,
                          n_params=count_params(model)))
    return slots
