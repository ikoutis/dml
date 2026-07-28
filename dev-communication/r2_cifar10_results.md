# R2 — CIFAR-10 transfer results (paper-facing summary)

**Audience:** the paper-writing session (other repo). Self-contained: all
numbers, provenance, caveats, and the update protocol are here. Cross-refs:
design + registered expectations in `log.md` [D-017]; partial readout in the
2026-07-28 reply to [D-017].

**Status: PARTIAL (updated 2026-07-28).**
- Clean tier: **complete** (19/20 at ep199; mwmd1 s3 at ep195, included).
- Noise tier: **in flight** — indep complete (5/5); dml/rand1/mwmd1 pending.
- ⚠ Numbers marked **TBD** below get filled when the noise tier drains.
  Everything else is final and safe to typeset.

## What R2 is

Dataset-transfer grid for the paper's two load-bearing claims, at the
headline cell. CIFAR-10, ResNet-32 × 8, DML-paper recipe (SGD 0.1, momentum
0.9, wd 5e-4, 200 epochs, step decay), clean + 40% symmetric label noise,
5 seeds, run_tag d017. Arms: `indep` (no communication), `dml` (dense,
degree 7), `rand1` (one random partner per epoch), `mwmd1` (one MWM-on-
disagreement partner per epoch). 40 tasks = 4 arms × 2 regimes × 5 seeds.

- Claim 1 (degree law): indep vs rand1 vs dml.
- Claim 2 (selection null): mwmd1 vs rand1.

## Clean tier — FINAL

Mean individual test accuracy ± std over seeds (final epoch); ensemble =
mean-posterior cohort ensemble. CIFAR-100 comparison column = the same cell
from `results/suite/m1_headline` (tag d001).

| arm | CIFAR-10 acc | CIFAR-10 ens | CIFAR-100 acc (ref) |
|---|---|---|---|
| indep | 92.61 ± 0.06 | 94.63 | 69.39 ± 0.14 |
| rand1 | 92.86 ± 0.09 | 94.31 | 71.44 ± 0.25 |
| mwmd1 | 92.98 ± 0.06 | 94.34 | 71.66 ± 0.15 |
| dense | 93.16 ± 0.09 | 94.47 | 72.04 ± 0.08 |

Derived (clean):
- Coupling benefit over indep: dense **+0.55**, rand1 **+0.25**, mwmd1 +0.37.
- Degree-1 share of dense benefit: **~46%** (rand1) / ~67% (mwmd1). NOTE:
  effects compress ~5× near the 93% ceiling (CIFAR-100 dense benefit was
  +2.65), so shares are noisy at ±0.09-scale effects. See phrasing guidance.
- Selection null: mwmd1 − rand1 = **+0.12** (same negligible scale as
  CIFAR-100's +0.21, p=.08). Ordering identical to CIFAR-100:
  indep < rand1 ≈ mwmd1 < dense.
- Ensemble trade-off transfers: indep ensemble (94.63) > all coupled
  ensembles (94.31–94.47), as on CIFAR-100 ([D-010]).

## Noise tier (40% symmetric) — PARTIAL

| arm | CIFAR-10 acc | CIFAR-10 ens | CIFAR-100 acc (ref, K=12 tier) |
|---|---|---|---|
| indep | **72.38 ± 0.71** (final) | **86.79** | 50.63 |
| rand1 | TBD (tracking 82.6–85.7 at ep137–199) | TBD | 58.34 (mwmd1) |
| mwmd1 | TBD (runs at ~ep50) | TBD | — |
| dense | TBD (runs at ep172–194; see anomaly) | TBD | 59.22 |

Already safe to state: the conversion-fuel gap transfers — indep
ensemble−individual gap is **14.4 pp** on CIFAR-10 (86.79 vs 72.38) vs 14.2
on CIFAR-100 — and in-flight coupled runs are tracking a conversion of
**≥ +10 pp**, larger than CIFAR-100's +8. Do not typeset the exact coupled
numbers until final; the shape is locked, the digits are not.

## Anomaly protocol (MUST respect in aggregation)

`dml_noise40_seed0005`: models 0, 1, 3 were **born dead** (never above 15%
at any epoch — spontaneous birth-collapse, third natural-zombie event of the
program; cf. [D-011] prism noise s2 and [D-014]). The five healthy members
finished ~81.8 with **no contagion and no further deaths** — a −1.2 pp
deficit at combined dead-teacher weight 3/7, consistent with the [D-015]
dose–response saturation; quotable as independent confirmation of R0 = 0 for
robust hosts (second dataset, triple dose).
**Aggregation rule:** for this seed use the healthy-model mean
(`mean of model_XX_test_acc over models {2,4,5,6,7}`), and footnote it, as
done for prism noise s2 in the CIFAR-100 tables. If other noise-tier seeds
show the same signature when they finish (check per-model finals < 15%),
apply the same rule and update this section.

## Recommended paper phrasing (agreed in discussion)

- Clean: "On CIFAR-10 the ordering and both nulls are preserved
  (indep < sparse ≈ sparse < dense; selection ties random; independent
  ensembles stay on top); absolute magnitudes compress with the reduced
  headroom near the 93% ceiling."
- Noise: carries the magnitude story — conversion of ≥ +10 pp with the same
  ensemble-gap mechanism. (Fill exact numbers when final.)
- Abstract clause (add when noise tier is final): "consistent across
  CIFAR-100 and CIFAR-10."
- Do NOT claim "84% of dense benefit" for CIFAR-10 clean; that share is a
  CIFAR-100 statement. The transfer claim is ordering + nulls + conversion.

## Provenance & regeneration

- CSVs: `results/suite/r2_cifar10/resnet32x8_cifar10_K8_{arm}[_noise40]_seed000{s}_d017_metrics.csv`
  on branch `claude/dml-variants-reduced-comm-l9uiz8` (repo ikoutis/dml).
  Per-epoch rows; final row = last epoch reached; complete ⇔ epoch ≥ 199.
  Per-model columns `model_XX_test_acc` (fractions; ×100 for %).
- CIFAR-100 reference cells: `results/suite/m1_headline/…_d001_metrics.csv`
  (K=8) and `results/suite/m7_topology/…_d011_metrics.csv` (K=12 noise tier).
- One-shot aggregation (run at repo root; prints both tiers + flags
  incomplete runs and dead models):

```python
import csv, glob, re, numpy as np
for f in sorted(glob.glob('results/suite/r2_cifar10/*_metrics.csv')):
    rows = list(csv.DictReader(open(f)))
    last = rows[-1]
    accs = [float(last[f'model_{i:02d}_test_acc'])*100 for i in range(8)]
    dead = [i for i, a in enumerate(accs) if a < 15]
    healthy = [a for a in accs if a >= 15]
    print(f.split('/')[-1], 'ep', last['epoch'],
          'mean %.2f' % np.mean(accs),
          'healthy %.2f' % np.mean(healthy),
          'ens %.2f' % (float(last['ensemble_test_acc'])*100),
          'DEAD', dead if dead else '-')
```

## Update protocol

When the noise tier finishes: (1) rerun the snippet; (2) fill the TBD cells
(healthy-mean where flagged); (3) recompute conversion (+X pp) and degree-1
share of it; (4) flip the Status header to FINAL with the date; (5) the
paper then adds the transfer table (draft slot: after Table 1 in
`paper/main.tex`, one table, both tiers) and the abstract clause.
