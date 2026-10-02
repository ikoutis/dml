"""R7: localized supervision. Only slots 0,1,2 of a K=12 cohort see labels.

Per arm (ring, per-epoch random matching-2, dense): final-epoch test accuracy
of the labeled and unlabeled groups; per-hop accuracy of unlabeled models by
their ring distance to the nearest labeled slot (a control for the non-ring
arms, where that distance is not a graph distance); the paired ring - random
difference on the unlabeled group; and, on the ring, the slope of accuracy
in hop distance (OLS over models x seeds, with a seed-level CI: one slope
per seed, t-interval).

    python analysis/r7_partial.py [--dir results/suite/r7_partial_supervision]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

K = 12
LABELED = (0, 1, 2)
ARMS = {"ring": "topo-ring", "random-2": "rand2", "dense": "dml"}


def hops(slot):
    return min(min(abs(slot - l), K - abs(slot - l)) for l in LABELED)


def load(directory, stem):
    out = {}
    for f in glob.glob(os.path.join(
            directory, f"resnet32x12_cifar100_K12_{stem}-lab0-1-2_seed*_metrics.csv")):
        df = pd.read_csv(f)
        if df["epoch"].max() < 199:
            continue
        r = df.loc[df["epoch"].idxmax()]
        out[int(f.split("_seed")[1][:4])] = np.array(
            [100 * r[f"model_{i:02d}_test_acc"] for i in range(K)])
    return out


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/suite/r7_partial_supervision")
    args = ap.parse_args()
    unl = [s for s in range(K) if s not in LABELED]
    dist = {s: hops(s) for s in unl}
    data = {a: load(args.dir, stem) for a, stem in ARMS.items()}

    print("Labeled slots 0,1,2; final-epoch test accuracy (%)\n")
    print(f"{'arm':<10}{'seeds':>6}{'labeled':>9}{'unlabeled':>11}   "
          + "  ".join(f"hop{h}" for h in range(1, 6)))
    for a, runs in data.items():
        if not runs:
            print(f"{a:<10}  no finished runs")
            continue
        A = np.stack(list(runs.values()))
        by_hop = [A[:, [s for s in unl if dist[s] == h]].mean() for h in range(1, 6)]
        print(f"{a:<10}{len(runs):>6}{A[:, list(LABELED)].mean():>9.2f}"
              f"{A[:, unl].mean():>11.2f}   "
              + "  ".join(f"{v:5.2f}" for v in by_hop))

    r, q = data["ring"], data["random-2"]
    seeds = sorted(set(r) & set(q))
    if seeds:
        m, lo, hi = tci([r[s][unl].mean() - q[s][unl].mean() for s in seeds])
        print(f"\nring - random-2, unlabeled group (n={len(seeds)}): "
              f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]")
        m, lo, hi = tci([r[s][list(LABELED)].mean() - q[s][list(LABELED)].mean()
                         for s in seeds])
        print(f"ring - random-2, labeled group   (n={len(seeds)}): "
              f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    for a in ("ring", "random-2"):
        runs = data[a]
        if len(runs) < 1:
            continue
        xs = np.array([dist[s] for s in unl])
        per_seed = [stats.linregress(xs, runs[k][unl]).slope for k in runs]
        m, lo, hi = tci(per_seed)
        print(f"{a}: accuracy slope in hop distance {m:+.3f} pp/hop "
              f"[{lo:+.3f},{hi:+.3f}] (seed-level, n={len(runs)})")


if __name__ == "__main__":
    main()
