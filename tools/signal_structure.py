"""Measure how much cross-pair STRUCTURE each candidate matching signal has,
from a completed cohort checkpoint — the cheap diagnostic that decides whether
a richer (multidimensional) edge weight can rescue MWM where scalar
disagreement could not.

Motivation (dev-communication/log.md, the M1 verdict + MWM-correctness check):
scalar disagreement is near-uniform in the homogeneous clean cohort late in
training, so max-weight matching ties random. A per-class or per-example
distance can be structured even when the scalar is flat. This tool loads one
run's final checkpoint, rebuilds every model's validation predictions, and for
each signal reports:
  * CV   — coefficient of variation of the off-diagonal edge weights (spread /
           mean). CV ~ 0 => uniform => matching cannot beat random; larger CV
           => real structure to exploit.
  * gain — realized max-weight matching weight vs the EXPECTED weight of a
           uniformly random perfect matching, as a percentage. This is exactly
           how much accuracy-relevant signal MWM can extract over random.

Usage (on the cluster, where checkpoints live):
    python tools/signal_structure.py \
        results/suite/m1_headline/wrn28x10x4-resnet32x4_cifar100_K8_mwmd1_seed0001_d001_ckpt.pt \
        --cohort wrn28x10:4,resnet32:4 --dataset cifar100 --data_dir data

Run it on one completed clean, one hetero, and one noise checkpoint. If the
per-class / error-field CV and gain are ~scalar in every regime, the MWM null
is robust to signal choice (a clean paper result). If they are materially
larger in hetero/noise, add a matched arm with --match_weight perclass (or
errorfield) in that regime — the modes are implemented and CLI-exposed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Allow `python tools/signal_structure.py ...` from the repo root: put the repo
# root (this file's parent's parent) on the path so `src` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import matching as mt  # noqa: E402
from src.cohort import build_cohort, parse_cohort_spec
from src.data import load_data, make_loaders, num_classes
from src.metrics import evaluate_cohort


def random_matching_expected_weight(W: np.ndarray, rng, trials: int = 2000
                                    ) -> float:
    """Monte-Carlo expected weight of a uniform random perfect matching."""
    K = W.shape[0]
    tot = 0.0
    for _ in range(trials):
        order = rng.permutation(K)
        tot += sum(W[order[2 * i], order[2 * i + 1]] for i in range(K // 2))
    return tot / trials


def report(name: str, W: np.ndarray, rng) -> None:
    K = W.shape[0]
    off = W[~np.eye(K, dtype=bool)]
    cv = off.std() / off.mean() if off.mean() > 1e-12 else float("nan")
    pairs = mt.max_weight_perfect_matching(W)
    mwm_w = sum(W[i, j] for i, j in pairs)
    rand_w = random_matching_expected_weight(W, rng)
    gain = (mwm_w - rand_w) / rand_w * 100 if rand_w > 1e-12 else float("nan")
    print(f"  {name:14s}  mean={off.mean():.4f}  CV={cv:.3f}  "
          f"MWM_weight={mwm_w:.4f}  rand_exp={rand_w:.4f}  "
          f"MWM_gain_over_random={gain:+.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n_valid", type=int, default=5000)
    ap.add_argument("--label_noise_rate", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu"
                             or torch.cuda.is_available()) else "cpu"
    n_cls = num_classes(args.dataset)
    _, valid_set, test_set, info = load_data(
        args.dataset, args.data_dir, n_valid=args.n_valid,
        label_noise_rate=args.label_noise_rate)
    _, valid_loader, _ = make_loaders(
        valid_set, valid_set, test_set, 64, 500, seed=0, num_workers=2,
        pin_memory=(device != "cpu"))

    slots = build_cohort(args.cohort, n_cls, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    for s, sd in zip(slots, state["models"]):
        s.model.load_state_dict(sd)

    ev = evaluate_cohort([s.model for s in slots], valid_loader, device)
    preds, y = ev["preds"], ev["y"]
    print(f"checkpoint epoch {state['epoch']}, K={len(slots)}, "
          f"individual val accs = {np.round(ev['accs'] * 100, 1)}")
    print("signal structure (higher CV / gain => more for MWM to exploit):")
    rng = np.random.default_rng(0)
    report("disagreement", mt.pairwise_disagreement(preds), rng)
    report("errorfield", mt.errorfield_distance_weights(preds, y), rng)
    report("perclass", mt.perclass_distance_weights(preds, y, n_cls), rng)
    report("teachable", mt.teachable_weights(preds, y, 1.0), rng)
    report("accgap", mt.accgap_weights(preds, y), rng)


if __name__ == "__main__":
    main()
