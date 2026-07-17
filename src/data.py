"""Data pipeline: CIFAR-100 / CIFAR-10 with a deterministic validation split
and optional symmetric label noise.

Protocol (dev-communication/experiments.md §2):

* The torchvision train set (50k) is split deterministically (``split_seed``,
  shared across ALL runs and arms) into a training subset (default 45k) and a
  held-out validation/probe set (default 5k). The matcher's inputs (hard
  predictions) and the policy-facing metrics are computed on the validation
  set; the 10k test set is reporting-only.
* Augmentation follows the DML paper (which follows the WRN paper): random
  crop from a 4-pixel reflection-padded image + horizontal flip.
* Symmetric label noise (``--label_noise_rate``) flips each TRAINING label
  independently, with the given probability, to a uniformly random WRONG
  class. Validation and test labels are always clean. The noise mask is drawn
  from ``noise_seed`` (independent of the run seed), so the same corruption
  is shared by every seed of a cell.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel statistics of the respective training sets.
_STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

_DATASETS = {
    "cifar100": (datasets.CIFAR100, 100),
    "cifar10": (datasets.CIFAR10, 10),
}


def num_classes(dataset: str) -> int:
    return _DATASETS[dataset][1]


def _transforms(dataset: str):
    mean, std = _STATS[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def apply_symmetric_noise(targets, indices, rate: float, n_cls: int,
                          noise_seed: int) -> int:
    """Flip ``targets[i]`` for i in ``indices`` with probability ``rate`` to a
    uniformly random wrong class, in place. Returns the number flipped."""
    if rate <= 0:
        return 0
    rng = np.random.RandomState(noise_seed)
    flipped = 0
    for i in indices:
        if rng.rand() < rate:
            wrong = rng.randint(n_cls - 1)
            if wrong >= targets[i]:
                wrong += 1
            targets[i] = int(wrong)
            flipped += 1
    return flipped


def load_data(dataset: str, data_dir: str, n_valid: int = 5000,
              split_seed: int = 0, label_noise_rate: float = 0.0,
              noise_seed: int = 42, download: bool = False):
    """Returns (train_set, valid_set, test_set, info dict).

    train_set carries (possibly noisy) labels and the train transform;
    valid/test carry clean labels and the eval transform.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    cls, n_cls = _DATASETS[dataset]
    train_tf, eval_tf = _transforms(dataset)

    # Two views of the torchvision train split: one with train-time
    # augmentation (for the training subset), one with eval transforms (for
    # the validation subset). Separate objects so noise touches only train.
    full_train = cls(root=data_dir, train=True, transform=train_tf,
                     download=download)
    full_eval = cls(root=data_dir, train=True, transform=eval_tf,
                    download=False)
    test_set = cls(root=data_dir, train=False, transform=eval_tf,
                   download=False)

    n_total = len(full_train)
    if not (0 < n_valid < n_total):
        raise ValueError(f"n_valid={n_valid} out of range (1, {n_total}) — "
                         "the matcher and metrics need a validation set.")
    perm = np.random.RandomState(split_seed).permutation(n_total)
    valid_idx = np.sort(perm[:n_valid])
    train_idx = np.sort(perm[n_valid:])

    n_flipped = apply_symmetric_noise(full_train.targets, train_idx,
                                     label_noise_rate, n_cls, noise_seed)

    train_set = Subset(full_train, train_idx.tolist())
    valid_set = Subset(full_eval, valid_idx.tolist())

    info = {
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_set),
        "num_classes": n_cls,
        "actual_noise_rate": n_flipped / max(len(train_idx), 1),
    }
    return train_set, valid_set, test_set, info


def make_loaders(train_set, valid_set, test_set, batch_size: int,
                 eval_batch_size: int, seed: int, num_workers: int = 4,
                 pin_memory: bool = True):
    """Train loader shuffles with a torch.Generator seeded from ``seed`` so
    the batch order is reproducible and shared across arms at equal seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=gen, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=eval_batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
