"""R5/R5b: the uniform-sampling degree curve at K=12 and its comparison with
the suite's peeled matchings.

Final-epoch mean individual test accuracy for uniform per-update sampling at
d in {1,2,4,8} (results/suite/r5_uniform_sampling, run_tag d020), against
independent, dense (d=11) and peeled MWM matchings from m7_topology (d011),
paired by seed (equal seeds share initializations and batch order). Runs
that did not reach the final epoch are ignored.

    python analysis/r5_degree_curve.py [--root results/suite]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R12 = "resnet32x12_cifar100_K12"


def final(root, directory, stem, col="avg_test_acc"):
    out = {}
    for f in glob.glob(os.path.join(root, directory,
                                    f"{R12}_{stem}_seed*_metrics.csv")):
        df = pd.read_csv(f)
        if df["epoch"].max() < 199:
            continue
        out[int(f.split("_seed")[1][:4])] = \
            100 * df.loc[df["epoch"].idxmax(), col]
    return out


def paired(a, b):
    """mean and 95% t-interval of a - b over shared seeds."""
    s = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in s])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return len(s), d.mean(), d.mean() - h, d.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/suite")
    args = ap.parse_args()
    r = args.root
    indep = final(r, "m7_topology", "indep")
    dense = final(r, "m7_topology", "dml")
    unif = {d: final(r, "r5_uniform_sampling", f"unif{d}-step")
            for d in (1, 2, 4, 8)}
    peeled = {1: final(r, "m7_topology", "mwmd1"),
              2: final(r, "m7_topology", "mwmd2")}
    I, D = np.mean(list(indep.values())), np.mean(list(dense.values()))

    print(f"independent {I:.2f}   dense {D:.2f}\n")
    print(f"{'uniform d':<10}{'n':>3}{'acc':>8}{'share':>7}   dense - d")
    for d, u in unif.items():
        m = np.mean(list(u.values()))
        n, mu, lo, hi = paired(dense, u)
        print(f"{d:<10}{len(u):>3}{m:>8.2f}{100 * (m - I) / (D - I):>6.0f}%"
              f"   {mu:+.2f} [{lo:+.2f},{hi:+.2f}]")
    print()
    for a, b in ((1, 2), (2, 4), (4, 8)):
        n, mu, lo, hi = paired(unif[b], unif[a])
        print(f"d={b} - d={a} (n={n}): {mu:+.2f} [{lo:+.2f},{hi:+.2f}]")
    for d, p in peeled.items():
        n, mu, lo, hi = paired(unif[d], p)
        print(f"uniform d={d} - peeled MWM d={d} (n={n}): "
              f"{mu:+.2f} [{lo:+.2f},{hi:+.2f}]")
    xs, ys = [], []
    for d, u in list(unif.items()) + [(11, dense)]:
        xs += [np.log2(d)] * len(u)
        ys += list(u.values())
    fit = stats.linregress(xs, ys)
    print(f"\nlog2-degree trend (uniform d=1..8 and dense): "
          f"{fit.slope:+.3f} pp per doubling, p={fit.pvalue:.1e}")


if __name__ == "__main__":
    main()
