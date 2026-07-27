"""Stage datasets on a login node (compute nodes have no internet).

    python tools/stage_data.py cifar100
    python tools/stage_data.py cifar10 --data_dir ./data
"""

import argparse

from torchvision import datasets

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=["cifar100", "cifar10"])
ap.add_argument("--data_dir", default="./data")
args = ap.parse_args()

cls = {"cifar100": datasets.CIFAR100, "cifar10": datasets.CIFAR10}[args.dataset]
cls(root=args.data_dir, train=True, download=True)
cls(root=args.data_dir, train=False, download=True)
print(f"[*] {args.dataset} staged under {args.data_dir}")
