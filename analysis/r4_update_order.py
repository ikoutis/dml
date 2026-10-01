"""R4 ([D-019]): sequential vs simultaneous update order, paired by seed.

Each sequential run in results/suite/r4_update_order is paired with the
simultaneous run of the same cohort, arm and seed (run_tag d001) in r1_pairs
or m1_headline. Pairs share initialisation and batch order, so the seed is the
unit of replication and the CI is a t-interval on the per-seed differences
(sequential - simultaneous), as in the paper's other paired comparisons.

    python analysis/r4_update_order.py [--results results/suite]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

# (label, sequential file stem, comparator dir, comparator file stem)
CELLS = [
    ("K=2 dense", "resnet32x2_cifar100_K2_dml-seq", "r1_pairs",
     "resnet32x2_cifar100_K2_dml"),
    ("K=8 dense", "resnet32x8_cifar100_K8_dml-seq", "m1_headline",
     "resnet32x8_cifar100_K8_dml"),
    ("K=8 degree 1 (random)", "resnet32x8_cifar100_K8_rand1-seq",
     "m1_headline", "resnet32x8_cifar100_K8_rand1"),
]
METRICS = ["avg_test_acc", "ensemble_test_acc"]


def final_row(path):
    df = pd.read_csv(path)
    return df.loc[df["epoch"].idxmax()]


def load(directory, stem):
    out = {}
    for path in sorted(glob.glob(os.path.join(
            directory, f"{stem}_seed*_d001_metrics.csv"))):
        seed = int(path.split("_seed")[1][:4])
        out[seed] = final_row(path)
    return out


def ci(diffs):
    d = np.asarray(diffs, dtype=float)
    m = d.mean()
    if len(d) < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/suite")
    args = ap.parse_args()
    seq_dir = os.path.join(args.results, "r4_update_order")

    for metric in METRICS:
        print(f"\n== {metric} (final epoch, %) ==")
        print(f"{'cell':<24}{'n':>3}{'simult.':>10}{'sequent.':>10}"
              f"{'diff':>9}   95% CI")
        for label, seq_stem, cmp_dir, cmp_stem in CELLS:
            seq = load(seq_dir, seq_stem)
            sim = load(os.path.join(args.results, cmp_dir), cmp_stem)
            seeds = sorted(set(seq) & set(sim))
            short = [s for s in seeds if seq[s]["epoch"] < sim[s]["epoch"]]
            if short:
                print(f"  [warn] {label}: seeds {short} did not finish all "
                      f"epochs; excluded")
                seeds = [s for s in seeds if s not in short]
            missing = sorted(set(sim) - set(seq))
            if missing:
                print(f"  [warn] {label}: no sequential run for seeds "
                      f"{missing}")
            if not seeds:
                print(f"{label:<24}  no paired runs found")
                continue
            a = [100 * sim[s][metric] for s in seeds]
            b = [100 * seq[s][metric] for s in seeds]
            m, lo, hi = ci([y - x for x, y in zip(a, b)])
            print(f"{label:<24}{len(seeds):>3}{np.mean(a):>10.2f}"
                  f"{np.mean(b):>10.2f}{m:>+9.2f}   [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
