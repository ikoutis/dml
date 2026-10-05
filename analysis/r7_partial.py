"""R7/R8: localized supervision. Only slots 0,1,2 of a K=12 cohort see labels.

Arms (all degree 2 except dense):
  ring             fixed ring i ~ i+1                       (R7, run_tag d022)
  random-2         random matching-2 redrawn every epoch    (R7, d022)
  random-2 fixed   random matching-2 drawn once and frozen  (R8, d023): a
                   fixed random 2-regular graph; with the same seed it is the
                   epoch-0 graph of the rotating run
  dense            all peers                                (R7, d022)

Reports, from final-epoch test accuracy: labeled / unlabeled group means; the
paired comparisons that separate graph structure from rotation
(fixed random - ring, fixed random - rotating random); and, for the two FIXED
graphs, unlabeled accuracy by true graph distance to the nearest labeled slot
(read from each run's logged matching; a fixed random graph is a union of
cycles, and models on a cycle with no labeled slot are 'unreachable').
Seed-level 95% t-intervals.

    python analysis/r7_partial.py [--dir results/suite/r7_partial_supervision]
"""

import argparse
import glob
import os
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats

K = 12
LABELED = (0, 1, 2)
ARMS = {"ring": "topo-ring", "random-2": "rand2",
        "random-2 fixed": "rand2-static", "dense": "dml"}
UNL = [s for s in range(K) if s not in LABELED]


def run_files(directory, stem):
    return glob.glob(os.path.join(
        directory, f"resnet32x12_cifar100_K12_{stem}-lab0-1-2_seed*_metrics.csv"))


def load(directory, stem):
    out = {}
    for f in run_files(directory, stem):
        df = pd.read_csv(f)
        if df["epoch"].max() < 199:
            continue
        r = df.loc[df["epoch"].idxmax()]
        out[int(f.split("_seed")[1][:4])] = np.array(
            [100 * r[f"model_{i:02d}_test_acc"] for i in range(K)])
    return out


def distances(adj):
    """BFS distance from the labeled set; None if unreachable."""
    dist = {s: 0 for s in LABELED}
    q = deque(LABELED)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return {s: dist.get(s) for s in UNL}


def ring_distances():
    return distances({i: {(i - 1) % K, (i + 1) % K} for i in range(K)})


def fixed_graph_distances(directory, stem):
    """{seed: {slot: distance or None}} from each run's epoch-0 matching."""
    out = {}
    for f in run_files(directory, stem):
        m = f.replace("_metrics.csv", "_matches.csv")
        if not os.path.exists(m):
            continue
        e = pd.read_csv(m)
        e = e[e["epoch"] == e["epoch"].min()]
        adj = {i: set() for i in range(K)}
        for i, j in zip(e["i"], e["j"]):
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
        out[int(f.split("_seed")[1][:4])] = distances(adj)
    return out


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def paired(a, b, slots, label):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return
    m, lo, hi = tci([a[s][slots].mean() - b[s][slots].mean() for s in seeds])
    print(f"  {label:<46}(n={len(seeds)}) {m:+.2f} [{lo:+.2f},{hi:+.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/suite/r7_partial_supervision")
    args = ap.parse_args()
    data = {a: load(args.dir, stem) for a, stem in ARMS.items()}
    ring_d = ring_distances()

    print("Labeled slots 0,1,2; final-epoch test accuracy (%)\n")
    print(f"{'arm':<16}{'seeds':>6}{'labeled':>9}{'unlabeled':>11}   "
          "unlabeled by ring position hop 1..5")
    for a, runs in data.items():
        if not runs:
            print(f"{a:<16}  no finished runs")
            continue
        A = np.stack(list(runs.values()))
        by_hop = [A[:, [s for s in UNL if ring_d[s] == h]].mean()
                  for h in range(1, 6)]
        print(f"{a:<16}{len(runs):>6}{A[:, list(LABELED)].mean():>9.2f}"
              f"{A[:, UNL].mean():>11.2f}   "
              + "  ".join(f"{v:5.2f}" for v in by_hop))

    print("\nPaired differences, unlabeled models (labeled models in brackets):")
    for a, b, lab in (("ring", "random-2", "ring - rotating random"),
                      ("random-2 fixed", "ring", "fixed random - ring  [structure]"),
                      ("random-2 fixed", "random-2",
                       "fixed random - rotating random  [rotation]"),
                      ("dense", "random-2", "dense - rotating random")):
        paired(data[a], data[b], UNL, lab)
        paired(data[a], data[b], list(LABELED), "  [labeled]")

    # True graph distance for the two fixed graphs.
    fr_d = fixed_graph_distances(args.dir, ARMS["random-2 fixed"])
    print("\nFixed random graphs (distance of each unlabeled slot to the "
          "labeled set; '-' = unreachable):")
    for s in sorted(fr_d):
        row = " ".join("-" if fr_d[s][u] is None else str(fr_d[s][u]) for u in UNL)
        print(f"  seed {s}: slots 3..11 -> {row}")

    pts = []   # (arm, seed, distance or None, accuracy)
    for s, acc in data["ring"].items():
        pts += [("ring", s, ring_d[u], acc[u]) for u in UNL]
    for s, acc in data["random-2 fixed"].items():
        if s in fr_d:
            pts += [("fixed random", s, fr_d[s][u], acc[u]) for u in UNL]
    if pts:
        df = pd.DataFrame(pts, columns=["arm", "seed", "dist", "acc"])
        print("\nUnlabeled accuracy by true graph distance (fixed graphs):")
        reach = df[df["dist"].notna()]
        tab = reach.groupby(["arm", "dist"])["acc"].agg(["mean", "count"])
        print(tab.round(2).to_string())
        unreach = df[df["dist"].isna()]
        if len(unreach):
            print(f"  unreachable: {len(unreach)} models, mean "
                  f"{unreach['acc'].mean():.2f}%")
        for arm in ("ring", "fixed random"):
            per_seed = []
            for s, g in reach[reach["arm"] == arm].groupby("seed"):
                if g["dist"].nunique() > 1:
                    per_seed.append(stats.linregress(g["dist"], g["acc"]).slope)
            if per_seed:
                m, lo, hi = tci(per_seed)
                print(f"  {arm}: slope {m:+.2f} [{lo:+.2f},{hi:+.2f}] pp/hop "
                      f"(seed-level, n={len(per_seed)})")
        fit = stats.linregress(reach["dist"].astype(float), reach["acc"])
        print(f"  pooled over both fixed graphs: {fit.slope:+.2f} pp/hop, "
              f"r^2={fit.rvalue ** 2:.2f}")

    rot = data["random-2"]
    for arm in ("ring", "random-2"):
        runs = data[arm]
        if runs:
            xs = np.array([ring_d[u] for u in UNL])
            m, lo, hi = tci([stats.linregress(xs, runs[k][UNL]).slope
                             for k in runs])
            print(f"{arm}: slope in ring position {m:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"pp/hop (seed-level, n={len(runs)})")
    del rot


if __name__ == "__main__":
    main()
