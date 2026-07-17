"""Aggregate run CSVs into summary statistics.

Usage (from the repo root, after cluster runs are committed):
    python analysis/aggregate.py --results_dir results/ --output_dir analysis/output/

Produces in --output_dir:
    summary_by_epoch.csv   mean +/- 95% CI per (cell, arm_label, epoch)
    summary_final.csv      final-epoch stats per (cell, arm_label), incl.
                           communication totals (the Pareto table)
    pairwise_tests_avg_test_acc.csv  Welch t-tests, Bonferroni-corrected,
                                     all arm pairs within each cell
    pairwise_tests_ensemble_test_acc.csv  same for ensemble accuracy

A "cell" is (cohort, dataset, K, label_noise_rate, graph) — the experimental
context an arm comparison is valid within.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

CELL = ["cohort", "dataset", "K", "label_noise_rate", "graph"]
METRICS = ["avg_test_acc", "std_test_acc", "min_test_acc", "max_test_acc",
           "ensemble_test_acc", "ens_minus_avg_test", "avg_val_acc",
           "ensemble_val_acc", "disagreement_val", "rho_val",
           "disagreement_test", "rho_test", "comm_bytes_cum",
           "comm_bytes_cum_cohort"]


def load_all(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "**",
                                          "*_metrics.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No *_metrics.csv under {results_dir}")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:  # noqa: BLE001 — skip malformed partials
            print(f"[warn] skipping {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[*] loaded {len(files)} runs, {len(df)} rows")
    return df


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1)
                 / np.sqrt(len(x)))


def summarize(df: pd.DataFrame, by_epoch: bool) -> pd.DataFrame:
    keys = CELL + ["arm_label"] + (["epoch"] if by_epoch else [])
    metrics = [m for m in METRICS if m in df.columns]
    rows = []
    for key_vals, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key_vals if isinstance(key_vals, tuple)
                       else (key_vals,)))
        row["n_seeds"] = g["seed"].nunique()
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_ci95"] = ci95(g[m].values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def final_epoch_rows(df: pd.DataFrame, keep_partial: bool = False
                     ) -> pd.DataFrame:
    """One row per run: its last logged epoch. Unless ``keep_partial``,
    runs that stopped short of their cell's target (the max final epoch
    seen among same-cell same-arm runs) are DROPPED with a warning —
    CSVs are flushed every epoch, so a killed job otherwise leaks a
    half-trained run into the final-round statistics."""
    idx = df.groupby("run_id")["epoch"].idxmax()
    final = df.loc[idx].reset_index(drop=True)
    if keep_partial:
        return final
    target = final.groupby(CELL + ["arm_label"], dropna=False)["epoch"] \
                  .transform("max")
    partial = final["epoch"] < target
    if partial.any():
        for rid in final.loc[partial, "run_id"]:
            print(f"[warn] dropping partial run {rid} "
                  f"(use --keep_partial to keep)")
    return final[~partial].reset_index(drop=True)


def pairwise_tests(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for cell_vals, g in final.groupby(CELL, dropna=False):
        arms = sorted(g["arm_label"].unique())
        cell_rows = []
        for a, b in itertools.combinations(arms, 2):
            xa = g.loc[g["arm_label"] == a, metric].astype(float).values
            xb = g.loc[g["arm_label"] == b, metric].astype(float).values
            if len(xa) < 2 or len(xb) < 2:
                continue
            t, p = stats.ttest_ind(xa, xb, equal_var=False)
            cell_rows.append({**dict(zip(CELL, cell_vals)),
                              "arm_a": a, "arm_b": b,
                              "n_a": len(xa), "n_b": len(xb),
                              f"{metric}_a": xa.mean(),
                              f"{metric}_b": xb.mean(),
                              "diff": xa.mean() - xb.mean(),
                              "t": t, "p": p})
        # Bonferroni over the tests actually performed in this cell.
        for r in cell_rows:
            r["p_bonferroni"] = min(r["p"] * len(cell_rows), 1.0)
        rows.extend(cell_rows)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--output_dir", default="analysis/output/")
    ap.add_argument("--keep_partial", action="store_true",
                    help="Keep runs that stopped before their cell's final "
                         "epoch in the final-round statistics")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all(args.results_dir)
    summarize(df, by_epoch=True).to_csv(
        os.path.join(args.output_dir, "summary_by_epoch.csv"), index=False)
    final = final_epoch_rows(df, keep_partial=args.keep_partial)
    summarize(final, by_epoch=False).to_csv(
        os.path.join(args.output_dir, "summary_final.csv"), index=False)
    for metric in ("avg_test_acc", "ensemble_test_acc"):
        pairwise_tests(final, metric).to_csv(
            os.path.join(args.output_dir, f"pairwise_tests_{metric}.csv"),
            index=False)
    print(f"[*] summaries written to {args.output_dir}")


if __name__ == "__main__":
    main()
