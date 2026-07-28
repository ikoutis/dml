"""Seed-clustered readout for the failure-injection campaign (M9, [D-015]).

Every inferential claim in the paper's failure section, recomputed with the
COHORT SEED as the replication unit, printed next to the pooled per-model
number the paper currently reports. Motivation: models inside one cohort run
share the seed, the batch order, the augmentations and the implanted teacher,
so the ten direct neighbors pooled across five ring runs are not ten
independent observations. Per-seed averaging over the relevant shell, followed
by a t-interval over n=5 seeds, is the conservative fix; for the
recency--damage correlation, where per-seed averaging would destroy the
signal, a cluster bootstrap resamples seeds and keeps all models within each
resampled seed.

Usage, from the repo root:

    python analysis/failure_seed_readout.py

Nothing here changes any point estimate; only the uncertainty statements move.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.matching import build_graph_mask  # noqa: E402

M7 = "results/suite/m7_topology"
M9 = "results/suite/m9_zombie"
K = 12
ZOMBIE = 0
FINAL_EPOCH = 199

# Two-sided t critical values at 95%, indexed by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def t_sf(t: float, df: int) -> float:
    """P(T_df > t) by numerical integration of the t density (no scipy)."""
    if t < 0:
        return 1.0 - t_sf(-t, df)
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    xs = np.linspace(t, t + 60.0, 200001)
    pdf = c * (1.0 + xs ** 2 / df) ** (-(df + 1) / 2)
    return float(np.trapezoid(pdf, xs))


def p_two_sided(values) -> float:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2 or np.std(v, ddof=1) == 0:
        return float("nan")
    t = np.mean(v) / (np.std(v, ddof=1) / math.sqrt(n))
    return 2.0 * t_sf(abs(t), n - 1)


def ci_half(values) -> float:
    v = np.asarray(values, dtype=float)
    n = len(v)
    return T975[n - 1] * np.std(v, ddof=1) / math.sqrt(n)


def per_seed_models(pattern):
    """{seed: np.array of 12 final per-model test accuracies (pp)}."""
    out = {}
    for f in glob.glob(pattern):
        seed = int(f.split("seed")[1][:4])
        with open(f) as fh:
            last = None
            for last in csv.DictReader(fh):
                pass
        if last is None or int(last["epoch"]) < FINAL_EPOCH:
            continue
        out[seed] = np.array([float(last[f"model_{i:02d}_test_acc"]) * 100
                              for i in range(K)])
    return out


def damage(zpat: str, apat: str):
    """{seed: np.array of per-model paired damage, implant minus anchor}."""
    z = per_seed_models(f"{M9}/{zpat}")
    a = per_seed_models(f"{M7}/{apat}")
    return {s: z[s] - a[s] for s in sorted(set(z) & set(a))}


def shells(kind: str, seed: int = 0):
    """{graph distance from the zombie: sorted list of healthy slots}."""
    mask = build_graph_mask(kind, K, seed=seed)
    dist = {ZOMBIE: 0}
    frontier = [ZOMBIE]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.nonzero(mask[u])[0]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    out = {}
    for v, d in dist.items():
        if v != ZOMBIE:
            out.setdefault(d, []).append(v)
    return {d: sorted(vs) for d, vs in out.items()}


def report(label, dmg, victims_fn):
    """One line: pooled per-model stats next to seed-clustered stats.

    victims_fn(seed) -> iterable of slots to average within that seed.
    """
    pooled, seed_means = [], []
    for s, vec in dmg.items():
        vic = [i for i in victims_fn(s) if i != ZOMBIE]
        vals = [vec[i] for i in vic]
        pooled.extend(vals)
        seed_means.append(float(np.mean(vals)))
    pm, psd = np.mean(pooled), np.std(pooled, ddof=1)
    pp = p_two_sided(pooled)
    m = np.mean(seed_means)
    hw, sp = ci_half(seed_means), p_two_sided(seed_means)
    flag = "  <-- crosses 0.05" if (pp < 0.05) != (sp < 0.05) else ""
    print(f"  {label:<34} pooled {pm:+6.2f}+-{psd:4.2f} (n={len(pooled):2d}, "
          f"p={pp:.2g})   seed-level {m:+6.2f} [{m-hw:+.2f},{m+hw:+.2f}] "
          f"(n={len(seed_means)}, p={sp:.2g}){flag}")
    return seed_means


def last_contacts(pattern):
    """{seed: {model: last epoch with an edge to the zombie}}."""
    out = {}
    for f in glob.glob(pattern):
        seed = int(f.split("seed")[1][:4])
        last = {}
        with open(f) as fh:
            for row in csv.DictReader(fh):
                i, j, ep = int(row["i"]), int(row["j"]), int(row["epoch"])
                if i == ZOMBIE:
                    last[j] = max(last.get(j, -1), ep)
                elif j == ZOMBIE:
                    last[i] = max(last.get(i, -1), ep)
        out[seed] = last
    return out


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.sum(xc * yc) /
                 math.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))


def main() -> None:
    print("Seed-clustered failure-section readout (replication unit: seed, "
          "n per cell = shared seeds)\n")

    # ---- ring spatial profile -------------------------------------------
    ring = damage("*_topo-ring-zomb0_seed*_d015_metrics.csv",
                  "*_topo-ring_seed*_d011_metrics.csv")
    print("Ring damage by graph distance:")
    rs = shells("ring")
    for d in sorted(rs):
        report(f"distance {d} (slots {rs[d]})", ring,
               lambda s, d=d: rs[d])

    # ---- chronic dose--response, victims per arm ------------------------
    print("\nChronic victim damage per arm (first shell; alpha = dead-teacher "
          "weight):")
    prism = damage("*_topo-prism-zomb0_seed*_d015_metrics.csv",
                   "*_topo-prism_seed*_d011_metrics.csv")
    rreg = damage("*_topo-rregular3-zomb0_seed*_d015_metrics.csv",
                  "*_topo-rregular3_seed*_d011_metrics.csv")
    dense = damage("*_dml-zomb0_seed*_d015_metrics.csv",
                   "*_dml_seed*_d011_metrics.csv")
    clus = damage("*_topo-clusters4-zomb0_seed*_d015_metrics.csv",
                  "*_topo-clusters4_seed*_d011_metrics.csv")
    report("ring, alpha=1/2", ring, lambda s: shells("ring")[1])
    report("prism, alpha=1/3", prism, lambda s: shells("prism")[1])
    report("expander, alpha=1/3", rreg,
           lambda s: shells("rregular:3", seed=s)[1])
    report("dense, alpha=1/11 (all 11)", dense, lambda s: range(1, K))
    cs = shells("clusters:4")
    report("sealed clique, alpha=1/3", clus, lambda s: cs[1])
    outsiders = [i for i in range(1, K) if i not in cs.get(1, [])]
    report("clique outsiders", clus, lambda s: outsiders)

    # ---- second shells ---------------------------------------------------
    print("\nSecond-shell damage (where a second shell exists):")
    report("prism, distance 2", prism, lambda s: shells("prism")[2])
    report("expander, distance 2", rreg,
           lambda s: shells("rregular:3", seed=s).get(2, []))

    # ---- pulsed arms and recency ----------------------------------------
    print("\nPulsed (rotating) arms, all healthy models:")
    r1 = damage("*_rand1-zomb0_seed*_d015_metrics.csv",
                "*_mwmd1_seed*_d011_metrics.csv")
    r2 = damage("*_rand2-zomb0_seed*_d015_metrics.csv",
                "*_mwmd2_seed*_d011_metrics.csv")
    if r1:
        report("matched-1 pulsed", r1, lambda s: range(1, K))
    if r2:
        report("matched-2 pulsed", r2, lambda s: range(1, K))

    lc = last_contacts(f"{M9}/*_rand1-zomb0_seed*_d015_matches.csv")
    shared = sorted(set(r1) & set(lc))
    if shared:
        print("\nRecency in the matched-1 pulsed arm "
              "(recency = final epoch minus last zombie contact):")
        report("last contact within 10 epochs", r1,
               lambda s: [i for i in range(1, K)
                          if FINAL_EPOCH - lc[s].get(i, -10**6) <= 10])
        report("last contact 11-30 epochs ago", r1,
               lambda s: [i for i in range(1, K)
                          if 11 <= FINAL_EPOCH - lc[s].get(i, -10**6) <= 30])

        # Pooled correlation + cluster bootstrap over seeds.
        xy = {s: ([FINAL_EPOCH - lc[s].get(i, FINAL_EPOCH)
                   for i in range(1, K)],
                  [r1[s][i] for i in range(1, K)]) for s in shared}
        x0 = [v for s in shared for v in xy[s][0]]
        y0 = [v for s in shared for v in xy[s][1]]
        r_pooled = pearson(x0, y0)
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(20000):
            pick = rng.choice(shared, size=len(shared), replace=True)
            bx = [v for s in pick for v in xy[s][0]]
            by = [v for s in pick for v in xy[s][1]]
            sd_x, sd_y = np.std(bx), np.std(by)
            if sd_x > 0 and sd_y > 0:
                boots.append(pearson(bx, by))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        frac = min(np.mean(np.asarray(boots) <= 0),
                   np.mean(np.asarray(boots) >= 0))
        print(f"\n  recency--damage correlation: pooled r={r_pooled:+.2f}; "
              f"seed-cluster bootstrap 95% CI [{lo:+.2f},{hi:+.2f}], "
              f"two-sided p~={2*frac:.2g} (20k resamples, fixed rng)")

    print("\nNotes:")
    print("  * 'pooled' reproduces the paper's current per-model inference,")
    print("    which treats models within one run as independent; the")
    print("    seed-level column is the corrected, clustered inference.")
    print("  * Point estimates are unchanged by construction; only the")
    print("    uncertainty statements move.")
    print("  * The clusters:4 second shell does not exist (the zombie's")
    print("    component is a 4-clique); containment is read from the")
    print("    'clique outsiders' row instead.")


if __name__ == "__main__":
    main()
