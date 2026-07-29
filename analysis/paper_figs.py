"""Generate the paper's figures from the suite CSVs -> paper/figures/*.pdf.

Run from the repo root:  python analysis/paper_figs.py
"""

import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.matching import build_graph_mask  # noqa: E402

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

C = {"indep": "#767676", "sparse": "#0072B2", "dense": "#D55E00",
     "aux": "#009E73"}


def curves(pattern, cols=("avg_test_acc", "comm_bytes_cum")):
    out = []
    for f in glob.glob(pattern):
        ep, vals = [], {c: [] for c in cols}
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ep.append(int(r["epoch"]))
                for c in cols:
                    vals[c].append(float(r[c]))
        out.append((np.array(ep), {c: np.array(v) for c, v in vals.items()}))
    return out


def finals(pattern, col="avg_test_acc", min_ep=199, scale=100):
    accs = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            last, v = -1, None
            for r in csv.DictReader(fh):
                last, v = int(r["epoch"]), float(r[col]) * scale
        if last >= min_ep:
            accs.append(v)
    return np.array(accs)


# ---------------------------------------------------------------- Fig 1: Pareto
def fig_pareto():
    M1 = "results/suite/m1_headline/"
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for label, pat, color in [
            ("dense DML", "resnet32x8_cifar100_K8_dml_seed*_d001", C["dense"]),
            ("matched deg-1", "resnet32x8_cifar100_K8_mwmd1_seed*_d001",
             C["sparse"])]:
        runs = curves(M1 + pat + "_metrics.csv")
        # mean curve over seeds on the common epoch grid
        n = min(len(ep) for ep, _ in runs)
        acc = np.mean([v["avg_test_acc"][:n] * 100 for _, v in runs], axis=0)
        comm = runs[0][1]["comm_bytes_cum"][:n] / 1e9
        ax.plot(comm, acc, color=color, label=label, lw=1.4)
    ind = finals(M1 + "resnet32x8_cifar100_K8_indep_seed*_d001_metrics.csv")
    ax.axhline(ind.mean(), color=C["indep"], ls=":", lw=1,
               label="independent (final)")
    ax.axvline(3.6, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.annotate("equal budget:\n71.7 vs 40.6", xy=(3.6, 45), fontsize=7,
                xytext=(6.5, 42), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("cumulative communication per model (GB)")
    ax.set_ylabel("mean test accuracy (%)")
    ax.set_xlim(0, 26)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pareto.pdf")


# ------------------------------------------------- Fig 2: conversion under noise
def fig_conversion():
    M7 = "results/suite/m7_topology/resnet32x12_cifar100_K12_"
    arms = [("indep", "independent", C["indep"]),
            ("mwmd1", "matched deg-1", C["sparse"]),
            ("dml", "dense", C["dense"])]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    w, x0 = 0.35, np.arange(3)
    for off, (regime, alpha) in enumerate([("", 1.0), ("_noise40", 0.55)]):
        ind_v, ens_v = [], []
        for arm, _, _ in arms:
            ind_v.append(finals(f"{M7}{arm}{regime}_seed*_d011_metrics.csv").mean())
            ens_v.append(finals(f"{M7}{arm}{regime}_seed*_d011_metrics.csv",
                                col="ensemble_test_acc").mean())
        bars = ax.bar(x0 + off * w, ind_v, w * 0.9,
                      color=[c for _, _, c in arms], alpha=alpha,
                      label="clean" if off == 0 else "40% label noise")
        for x, e in zip(x0 + off * w, ens_v):
            ax.plot([x - w * 0.4, x + w * 0.4], [e, e], color="k", lw=1)
    ax.set_xticks(x0 + w / 2)
    ax.set_xticklabels([lab for _, lab, _ in arms])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(45, 80)
    ax.annotate("bars: individual mean\nticks: ensemble", xy=(0.02, 0.97),
                xycoords="axes fraction", va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/conversion.pdf")


# ------------------------------------------------------- Fig 3: M9 damage field
def fig_damage():
    M7 = "results/suite/m7_topology/"
    M9 = "results/suite/m9_zombie/"

    def per_seed_models(pattern):
        out = {}
        for f in glob.glob(pattern):
            seed = int(f.split("seed")[1][:4])
            with open(f) as fh:
                for last in csv.DictReader(fh):
                    pass
            if int(last["epoch"]) < 199:
                continue
            out[seed] = np.array([float(last[f"model_{i:02d}_test_acc"]) * 100
                                  for i in range(12)])
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3))

    # left: ring spatial profile
    z = per_seed_models(M9 + "resnet32x12_cifar100_K12_topo-ring-zomb0_seed*_d015_metrics.csv")
    a = per_seed_models(M7 + "resnet32x12_cifar100_K12_topo-ring_seed*_d011_metrics.csv")
    by_d = {}
    for s in set(z) & set(a):
        for i in range(1, 12):
            by_d.setdefault(min(i, 12 - i), []).append(z[s][i] - a[s][i])
    ds = sorted(by_d)
    means = [np.mean(by_d[d]) for d in ds]
    errs = [np.std(by_d[d]) / np.sqrt(len(by_d[d])) for d in ds]
    ax1.errorbar(ds, means, yerr=errs, fmt="o-", color=C["sparse"], lw=1.2,
                 capsize=2, ms=3.5)
    ax1.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax1.set_xlabel("ring distance from dead model")
    ax1.set_ylabel("paired damage (pp)")
    ax1.set_title("damage is one-hop local", fontsize=8)

    # right: dose-response points (chronic) + pulsed
    pts = []  # (dose, damage, sem, label)
    for zpat, apat, vic_fn, alpha, lab in [
        ("topo-ring-zomb0", "topo-ring", lambda s: {1, 11}, 1 / 2, "ring"),
        ("topo-prism-zomb0", "topo-prism", lambda s: {1, 5, 6}, 1 / 3, "prism"),
        ("topo-rregular3-zomb0", "topo-rregular3",
         lambda s: set(np.nonzero(build_graph_mask("rregular:3", 12, seed=s)[0])[0]),
         1 / 3, "expander"),
    ]:
        zz = per_seed_models(M9 + f"resnet32x12_cifar100_K12_{zpat}_seed*_d015_metrics.csv")
        aa = per_seed_models(M7 + f"resnet32x12_cifar100_K12_{apat}_seed*_d011_metrics.csv")
        d = [zz[s][i] - aa[s][i] for s in set(zz) & set(aa)
             for i in range(1, 12) if i in vic_fn(s)]
        pts.append((alpha, np.mean(d), np.std(d) / np.sqrt(len(d)), lab))
    zz = per_seed_models(M9 + "resnet32x12_cifar100_K12_dml-zomb0_seed*_d015_metrics.csv")
    aa = per_seed_models(M7 + "resnet32x12_cifar100_K12_dml_seed*_d011_metrics.csv")
    d = [zz[s][i] - aa[s][i] for s in set(zz) & set(aa) for i in range(1, 12)]
    pts.append((1 / 11, np.mean(d), np.std(d) / np.sqrt(len(d)), "dense"))
    for alpha, m, e, lab in pts:
        ax2.errorbar(alpha, m, yerr=e, fmt="o", color=C["dense"], ms=4,
                     capsize=2)
        ax2.annotate(lab, xy=(alpha, m), xytext=(4, -3),
                     textcoords="offset points", fontsize=7)
    ax2.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax2.set_xlabel(r"dead-teacher weight $\alpha$ in victim's KD mix")
    ax2.set_ylabel("victim damage (pp)")
    ax2.set_title("dose response consistent with saturation", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/damage.pdf")


# ------------------------------------------------ Fig 4: LeNet collapse contagion
def fig_contagion():
    arms = [("indep", "independent", 0), ("mwmd1", "matched-1", 1),
            ("topo-ring", "ring", 1), ("topo-prism", "prism", 1),
            ("mwmd2", "matched-2", 3), ("topo-rregular3", "expander", 3),
            ("dml", "dense", 3), ("topo-clusters2", "isolated pairs", 0),
            ("topo-clusters4", "isolated 4-cliques", 0)]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    labels = [lab for _, lab, _ in arms]
    rates = [n / 5 for _, _, n in arms]
    colors = [C["indep"] if r == 0 else C["dense"] for r in rates]
    colors[0] = C["indep"]
    colors[-2] = colors[-1] = C["aux"]
    ax.barh(range(len(arms)), rates, color=colors, height=0.65)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("P(full-cohort collapse), LeNet K=12")
    ax.set_xlim(0, 0.7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/contagion.pdf")


if __name__ == "__main__":
    fig_pareto()
    fig_conversion()
    fig_damage()
    fig_contagion()
    print("figures written to", OUT)
