"""Statistics ledger for the paper's paired comparisons (Table tab:nulls).

For every comparison: per-seed paired differences of final-epoch mean
individual test accuracy (refinement - baseline, percentage points), the
95% t-interval over seeds (df = n-1), the two-sided paired t-test p-value,
its Holm-adjusted value across the whole table, and a TOST equivalence test
against a declared practical margin (default +/-0.5 pp): equivalence is
concluded when the 90% t-interval lies inside (-margin, +margin).

    python analysis/stats_ledger.py [--roots results/suite ...] [--margin 0.5]

--roots may list several result trees; each cell is looked up in all of them
(the R3 temporal-dense runs live on the r2-results-drop branch).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R8, H8 = "resnet32x8_cifar100_K8", "wrn28x10x4-resnet32x4_cifar100_K8"
R12 = "resnet32x12_cifar100_K12"

# (block, comparison, setting, refinement (dir, stem), baseline (dir, stem))
ROWS = [
    ("same-degree communication patterns", "prism vs. random 3-reg.",
     "K=12 clean", ("m7_topology", f"{R12}_topo-prism_seed*_d011"),
     ("m7_topology", f"{R12}_topo-rregular3_seed*_d011")),
    ("same-degree communication patterns", "ring vs. matched-2",
     "K=12 clean", ("m7_topology", f"{R12}_topo-ring_seed*_d011"),
     ("m7_topology", f"{R12}_mwmd2_seed*_d011")),
    ("same-degree communication patterns", "4-cliques vs. prism",
     "K=12 clean", ("m7_topology", f"{R12}_topo-clusters4_seed*_d011"),
     ("m7_topology", f"{R12}_topo-prism_seed*_d011")),
    ("target composition and teacher weighting", "DML_e vs. DML",
     "K=8 clean", ("m5_target_structure", f"{R8}_dmle_seed*_d001"),
     ("m1_headline", f"{R8}_dml_seed*_d001")),
    ("target composition and teacher weighting", "weight- vs. uniform alpha",
     "K=8 clean", ("m2_degree_dial", f"{R8}_mwmd2_seed*_d001"),
     ("m2_degree_dial", f"{R8}_mwmd2-unif_seed*_d001")),
    ("partner selection", "MWM(disagree.) vs. rand.", "K=8 clean",
     ("m1_headline", f"{R8}_mwmd1_seed*_d001"),
     ("m1_headline", f"{R8}_rand1_seed*_d001")),
    ("partner selection", "MWM(disagree.) vs. rand.", "K=8 noise",
     ("m1_headline", f"{R8}_mwmd1_noise40_seed*_d001"),
     ("m1_headline", f"{R8}_rand1_noise40_seed*_d001")),
    ("partner selection", "MWM(per-class) vs. rand.", "K=8 clean",
     ("m1_headline", f"{R8}_mwmpc1_seed*_d001"),
     ("m1_headline", f"{R8}_rand1_seed*_d001")),
    ("partner selection", "MWM(per-class) vs. rand.", "K=8 noise",
     ("m1_headline", f"{R8}_mwmpc1_noise40_seed*_d001"),
     ("m1_headline", f"{R8}_rand1_noise40_seed*_d001")),
    ("budget allocation", "temporal-dense (dose-m.) vs. deg-1",
     "K=12 clean",
     ("r3_temporal_dense", f"{R12}_dmle-t6of11-dose_seed*_d018"),
     ("m7_topology", f"{R12}_mwmd1_seed*_d011")),
    ("intervals excluding zero", "MWM(disagree.) vs. rand.", "K=8 hetero",
     ("m1_headline", f"{H8}_mwmd1_seed*_d001"),
     ("m1_headline", f"{H8}_rand1_seed*_d001")),
    ("intervals excluding zero", "frozen vs. rotating pairs", "K=8 hetero",
     ("m4_rotation", f"{H8}_mwmd1-static_seed*_d001"),
     ("m1_headline", f"{H8}_mwmd1_seed*_d001")),
    ("intervals excluding zero", "isolated pairs vs. matched-1",
     "K=12 clean", ("m7_topology", f"{R12}_topo-clusters2_seed*_d011"),
     ("m7_topology", f"{R12}_mwmd1_seed*_d011")),
    ("intervals excluding zero", "temporal-dense (strict) vs. deg-1",
     "K=12 clean", ("r3_temporal_dense", f"{R12}_dmle-t6of11_seed*_d018"),
     ("m7_topology", f"{R12}_mwmd1_seed*_d011")),
]


def load(roots, cell):
    directory, pattern = cell
    out = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, directory,
                                           pattern + "_metrics.csv")):
            seed = int(path.split("_seed")[1][:4])
            df = pd.read_csv(path)
            out[seed] = 100 * df.loc[df["epoch"].idxmax(), "avg_test_acc"]
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["results/suite"])
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--csv", default="", help="also write the ledger here")
    args = ap.parse_args()

    recs = []
    for block, name, setting, ref, base in ROWS:
        a, b = load(args.roots, ref), load(args.roots, base)
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 2:
            print(f"[skip] {name} ({setting}): {len(seeds)} paired seeds")
            continue
        d = np.array([a[s] - b[s] for s in seeds])
        n, m, sd = len(d), d.mean(), d.std(ddof=1)
        se = sd / np.sqrt(n)
        t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
        p = 2 * stats.t.sf(abs(m / se), n - 1)
        # TOST: the larger of the two one-sided p-values.
        p_tost = max(stats.t.sf((m + args.margin) / se, n - 1),
                     stats.t.cdf((m - args.margin) / se, n - 1))
        recs.append({
            "block": block, "comparison": name, "setting": setting, "n": n,
            "diffs": " ".join(f"{x:+.2f}" for x in d),
            "mean": m, "sd": sd, "lo95": m - t95 * se, "hi95": m + t95 * se,
            "lo90": m - t90 * se, "hi90": m + t90 * se,
            "p": p, "p_tost": p_tost,
            "equivalent": (m - t90 * se > -args.margin
                           and m + t90 * se < args.margin),
            # Smallest margin at which TOST (alpha = .05) concludes
            # equivalence: margin-free, so readers can apply their own.
            "delta_min": max(abs(m - t90 * se), abs(m + t90 * se)),
        })
    df = pd.DataFrame(recs)
    df["p_holm"] = holm(df["p"])

    print(f"Equivalence margin +/-{args.margin} pp; Holm over "
          f"{len(df)} comparisons\n")
    block = None
    for r in df.itertuples():
        if r.block != block:
            block = r.block
            print(f"-- {block}")
        eq = "equiv." if r.equivalent else "  --  "
        print(f"  {r.comparison:<36}{r.setting:<12}{r.mean:+.2f} "
              f"[{r.lo95:+.2f},{r.hi95:+.2f}]  p={r.p:.3f} "
              f"holm={r.p_holm:.3f}  tost={r.p_tost:.3f} {eq} "
              f"dmin={r.delta_min:.2f}")
        print(f"  {'':<36}seeds: {r.diffs}")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
