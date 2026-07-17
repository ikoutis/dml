"""Figures from the aggregated summaries.

Usage:
    python analysis/plot.py --input_dir analysis/output/ \
        --output_dir analysis/figures/ --fmt pdf

Figures per cell (cohort, dataset, K, noise, graph):
    curves_<cell>.<fmt>       avg test acc vs epoch, one line per arm, CI band
    diversity_<cell>.<fmt>    ensemble-minus-avg gap and rho_test vs epoch
    pareto_<cell>.<fmt>       THE money plot: final avg test acc vs cumulative
                              per-model communication (log-x), one point per
                              arm with CI bars
    heatmap_final.<fmt>       arm x cell heatmap of final avg test acc
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CELL = ["cohort", "dataset", "K", "label_noise_rate", "graph"]


def cell_slug(vals) -> str:
    cohort, dataset, K, noise, graph = vals
    s = f"{str(cohort).replace(':', 'x').replace(',', '-')}_{dataset}_K{K}"
    if float(noise) > 0:
        s += f"_noise{int(round(float(noise) * 100))}"
    if isinstance(graph, str) and graph not in ("", "complete"):
        s += f"_{graph.replace(':', '')}"
    return s


def plot_curves(by_epoch: pd.DataFrame, outdir: str, fmt: str) -> None:
    for vals, g in by_epoch.groupby(CELL, dropna=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm, ga in g.groupby("arm_label"):
            ga = ga.sort_values("epoch")
            m = ga["avg_test_acc_mean"].astype(float)
            ci = ga["avg_test_acc_ci95"].astype(float)
            ax.plot(ga["epoch"], m, label=arm, lw=1.5)
            ax.fill_between(ga["epoch"], m - ci, m + ci, alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel("avg individual test acc")
        ax.set_title(cell_slug(vals))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"curves_{cell_slug(vals)}.{fmt}"))
        plt.close(fig)


def plot_diversity(by_epoch: pd.DataFrame, outdir: str, fmt: str) -> None:
    for vals, g in by_epoch.groupby(CELL, dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for arm, ga in g.groupby("arm_label"):
            ga = ga.sort_values("epoch")
            axes[0].plot(ga["epoch"], ga["ens_minus_avg_test_mean"],
                         label=arm, lw=1.5)
            axes[1].plot(ga["epoch"], ga["rho_test_mean"], label=arm, lw=1.5)
        axes[0].set_ylabel("ensemble - avg test acc")
        axes[1].set_ylabel("mean pairwise error correlation rho")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(cell_slug(vals))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir,
                                 f"diversity_{cell_slug(vals)}.{fmt}"))
        plt.close(fig)


def plot_pareto(final: pd.DataFrame, outdir: str, fmt: str) -> None:
    for vals, g in final.groupby(CELL, dropna=False):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for _, row in g.iterrows():
            x = max(float(row["comm_bytes_cum_mean"]), 1.0)
            y = float(row["avg_test_acc_mean"])
            yerr = row.get("avg_test_acc_ci95", np.nan)
            ax.errorbar(x, y, yerr=None if pd.isna(yerr) else float(yerr),
                        marker="o", capsize=3)
            ax.annotate(row["arm_label"], (x, y), fontsize=8,
                        textcoords="offset points", xytext=(5, 3))
        ax.set_xscale("symlog", linthresh=10.0)
        ax.set_xlabel("cumulative bytes sent per model (log)")
        ax.set_ylabel("final avg individual test acc")
        ax.set_title(f"accuracy vs communication - {cell_slug(vals)}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"pareto_{cell_slug(vals)}.{fmt}"))
        plt.close(fig)


def plot_heatmap(final: pd.DataFrame, outdir: str, fmt: str) -> None:
    final = final.copy()
    final["cell"] = [cell_slug(t) for t in
                     final[CELL].itertuples(index=False, name=None)]
    pivot = final.pivot_table(index="arm_label", columns="cell",
                              values="avg_test_acc_mean", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(pivot.columns),
                                    1 + 0.5 * len(pivot.index)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="final avg test acc")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heatmap_final.{fmt}"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="analysis/output/")
    ap.add_argument("--output_dir", default="analysis/figures/")
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    by_epoch = pd.read_csv(os.path.join(args.input_dir,
                                        "summary_by_epoch.csv"))
    final = pd.read_csv(os.path.join(args.input_dir, "summary_final.csv"))
    plot_curves(by_epoch, args.output_dir, args.fmt)
    plot_diversity(by_epoch, args.output_dir, args.fmt)
    plot_pareto(final, args.output_dir, args.fmt)
    plot_heatmap(final, args.output_dir, args.fmt)
    print(f"[*] figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
