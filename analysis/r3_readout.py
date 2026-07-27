"""Read out R3 ([D-018]): the byte-matched temporally sparse dense baseline.

Reports the comparison the experiment was designed to make, as SEED-LEVEL
PAIRED differences with 95% confidence intervals. Paired, not Welch: R3's arms
share cohort, seeds, initialisations and batch order with the d011 comparators,
and this project has already been misled once by reading unpaired cell means as
a tie (the frozen-vs-rotating row, which looked like 75.60 vs 75.60 and is in
fact +0.27 [+0.16, +0.38] when paired). `analysis/aggregate.py` runs unpaired
Welch tests across all arms; use this for the R3 verdict.

Usage, from the repo root:

    python analysis/r3_readout.py

Prints final and best accuracy per arm, the accounted communication each arm
spent, and the paired differences against degree-1 that decide the experiment.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st

# Two-sided t critical values at 95%, indexed by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}

R3_DIR = "results/suite/r3_temporal_dense"
M7_DIR = "results/suite/m7_topology"

# label -> (directory, run_id glob) for every arm in the comparison
ARMS = [
    ("independent",          M7_DIR, "*K12_indep_seed*_metrics.csv"),
    ("degree-1 (every step)", M7_DIR, "*K12_mwmd1_seed*_metrics.csv"),
    ("dense (every step)",   M7_DIR, "*K12_dml_seed*_metrics.csv"),
    ("temporal-dense 6/11",  R3_DIR, "*K12_dmle-t6of11_seed*_metrics.csv"),
    ("  + dose-matched",     R3_DIR, "*K12_dmle-t6of11-dose_seed*_metrics.csv"),
]

BASELINE = "degree-1 (every step)"


def load(directory: str, pattern: str):
    """{seed: dict of final/best/bytes/duty} for each completed run."""
    out = {}
    for path in glob.glob(os.path.join(directory, pattern)):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        accs = [100 * float(r["avg_test_acc"]) for r in rows]
        out[seed] = {
            "epochs": len(rows),
            "final": accs[-1],
            "best": max(accs),
            "bytes": float(rows[-1]["comm_bytes_cum"]),
            "duty": float(rows[-1].get("comm_duty", 1.0) or 1.0),
        }
    return out


def paired(a: dict, b: dict):
    """(mean difference a-b, ci half-width, n) over shared seeds."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = [a[s]["final"] - b[s]["final"] for s in seeds]
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    return m, T975.get(n - 1, 2.776) * sd / math.sqrt(n), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200,
                    help="expected epoch count; runs short of it are flagged")
    args = ap.parse_args()

    data = {}
    print(f"{'arm':<24} {'n':>2}  {'final':>7} {'best':>7}  {'GB/model':>9}  duty")
    print("-" * 68)
    for label, directory, pattern in ARMS:
        runs = load(directory, pattern)
        data[label] = runs
        if not runs:
            print(f"{label:<24} {'--':>2}  (no runs yet)")
            continue
        short = [s for s, r in runs.items() if r["epochs"] < args.epochs]
        finals = [r["final"] for r in runs.values()]
        bests = [r["best"] for r in runs.values()]
        gb = st.mean(r["bytes"] for r in runs.values()) / 1e9
        duty = st.mean(r["duty"] for r in runs.values())
        flag = f"   [{len(short)} run(s) short of {args.epochs} epochs]" if short else ""
        print(f"{label:<24} {len(runs):>2}  {st.mean(finals):7.2f} "
              f"{st.mean(bests):7.2f}  {gb:9.4f}  {duty:.4f}{flag}")

    base = data.get(BASELINE) or {}
    if not base:
        print("\nNo degree-1 comparator found; nothing to pair against.")
        return

    print(f"\nPaired differences vs. {BASELINE} (final accuracy, seed-level):")
    print("positive = the arm beats degree-1 at the same accounted budget\n")
    for label, _, _ in ARMS:
        if label == BASELINE or not data.get(label):
            continue
        res = paired(data[label], base)
        if res is None:
            print(f"  {label:<24} too few shared seeds")
            continue
        m, hw, n = res
        verdict = "excludes 0" if abs(m) > hw else "includes 0"
        wide = "  <-- widen to 10 seeds" if hw > 0.3 else ""
        print(f"  {label:<24} {m:+6.2f}  [{m-hw:+.2f}, {m+hw:+.2f}]  "
              f"n={n}  {verdict}{wide}")

    print("\nNotes on the byte column:")
    print("  * The dense-every-step row is billed under the NAIVE point-to-point")
    print("    ledger it was run with (K-1 streams). Its all-reduce equivalent is")
    print("    2(K-1)/K = 1.83 streams, i.e. 6.60 GB - that figure is modeled, not")
    print("    measured, and is the baseline the R3 arms are matched against.")
    print("  * The R3 arms are billed at 6/11 x 11/6 = 1 stream per update. The")
    print("    degree-1 comparator additionally pays ~2 MB of matcher probes")
    print("    (0.06%), so temporal-dense sits marginally UNDER budget, not at it.")


if __name__ == "__main__":
    main()
