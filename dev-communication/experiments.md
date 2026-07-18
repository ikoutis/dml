# Reduced-communication deep mutual learning — the experiment suite

*Created 2026-07-17 ([D-001]). Status: designed and harness-implemented; nothing launched yet.*

This document is the full design of the experimental program for **matched mutual
learning (MML)**: variants of Deep Mutual Learning (Zhang et al., CVPR 2018 — "the
paper" below) that replace dense all-to-all mimicry with per-round **maximum-weight
matchings** on measured disagreement. The design goal has two axes:

1. **Reduce communication.** Dense DML has every model exchange per-batch logits with
   all K−1 peers; a matching exchanges with exactly one (or k) partners, making
   per-model communication **constant in cohort size** instead of linear.
2. **Use the MWM idea to pump performance in specific dimensions.** Not a uniform
   accuracy win — the theory predicts *where* the matching choice matters: cohorts that
   are architecturally **heterogeneous** (the paper's flagship use case: big + small
   nets), cohorts under **label noise**, and the **diversity / ensemble-utility** axis
   that dense mimicry is known to drain.

The ideas originate in the knowledge-diffusion (KD) repo:
`dev-communication/ideas.md`, entry *"2026-07-16 — Matched & sparse mutual learning
(DML × MWM-D × topology)"* (Ideas 1, 2b, 2c, the literature sweep, and the gap
analysis), plus its 2026-07-17 implementation notes. Per those notes, the KD side runs
the round-based approximation; **this repo carries the faithful per-batch DML trainer**
— the suite below is organized at the level of the original paper's experiments
(its §3), with the paper's own tables as calibration anchors.

The suite is structured as the paper's §3 was: a settings section, a two-network
table, a cohort-scaling study, and mechanism experiments — with our arms added.

---

## 0. Headline claims under test

| ID | Claim | Confirmed if | Falsified if |
|----|-------|--------------|--------------|
| H1 | **Matching is (almost) free.** k=1 matched mimicry retains dense DML's individual-accuracy gain over Independent, at 1/(K−1) of the per-model communication. | matched within CI of dense in every regime (M1), incl. cohort growth (M3) | dense beats matched by a consistent margin (> ~0.5 pp across regimes/seeds) |
| H2 | **The MWM gain is selective.** MWM-matched ≥ randomly-matched, with the gap concentrated in the heterogeneous and noisy regimes; ≈ tie in the homogeneous clean regime (first-order modularity: the landscape is flat exactly there). | M1: mwmd1 > rand1 in hetero+noise cells, ≈ in clean homog | mwmd1 < rand1 anywhere robustly; or mwmd1 > rand1 *uniformly* (would contradict modularity, itself informative) |
| H3 | **Matched coupling conserves diversity.** Dense DML homogenizes fastest (highest pairwise error correlation ρ, smallest ensemble−individual gap late in training); matched arms retain more of both at equal individual accuracy. | M1/M2 diversity trajectories order dense < matched | no ordering, or reversed |
| H4 | **Accuracy-gap pairing is a shadow.** Matching on \|acc_i − acc_j\| performs like random matching — the pairing signal lives in second-order disagreement structure, not first-order accuracy. | M6: gap1 ≈ rand1 everywhere | gap1 robustly > rand1 |
| H5 | **The degree dial has an interior sweet spot.** Peeled k-matchings interpolate matched (k=1) → dense (k=K−1); early mean accuracy is nearly k-independent, consensus speed rises with k, final ensemble−individual gap falls with k; expected k* ≈ 2–3 with degree-annealing ≥ any fixed k. | M2 trend + anneal arm | accuracy strictly monotone in k (dense strictly best) |
| H6 | **Rotation is load-bearing.** A static (frozen) matching underperforms per-epoch re-matching, and its frozen pairs show a per-edge ρ climb (partners memorize each other's errors); a recency penalty ≥ plain re-matching. | M4 ranking recency ≥ plain > static, per-edge ρ climb in static pairs | static ≈ re-matched (persistence harmless in supervised DML) |

Secondary open question (Idea 2b's sharper claim): because mutual sessions cancel at
first order *pair-wise*, disagreement weighting operates entirely at second order and
**may help even in the homogeneous clean cell**. H2 states the conservative selective
prediction; a homogeneous-cell win for mwmd1 would upgrade, not break, the story.

One honest scoping note, carried over from the KD ideas entry: in the paper's native
one-box setting the communication argument is *weak* (logits are tiny next to K
forward passes). The communication claim is aimed at the distributed/decentralized
deployment — constant-degree bandwidth, pairwise sync instead of lockstep all-to-all,
no aggregation server, and applicability to architecturally heterogeneous cohorts
where parameter averaging (D-SGD) is impossible. We therefore *measure* communication
exactly (§1.4) and lead with the learning-dynamics results.

---

## 1. Methods

### 1.1 Arms

All arms train every model on the hard-label CE anchor; they differ only in the
mimicry term and in who talks to whom. **Total mimicry mass is 1 in every coupled
arm** (dense averages its K−1 KLs; matched arms normalize their k teacher weights to
sum to 1), so arms differ in the *structure* of the mimicry signal and in
communication — not in loss magnitude.

| Label | Loss for model i (T=1) | Partners / refresh | CLI |
|-------|------------------------|--------------------|-----|
| `indep` | CE(z_i, y) | none | `--arm indep` |
| `dml` | CE + (1/(K−1)) Σ_{j≠i} KL(p_j ‖ p_i) | all, every batch | `--arm dml` |
| `dmle` | CE + KL(p̄_{−i} ‖ p_i), p̄_{−i} = mean posterior of others | all, every batch | `--arm dml --target ensemble` |
| `rand1` | CE + KL(p_{M(i)} ‖ p_i) | random perfect matching, re-drawn per epoch | `--arm matched --match_weight random` |
| `mwmd1` | CE + KL(p_{M(i)} ‖ p_i) | max-weight matching on disagreement d_ij, per epoch | `--arm matched --match_weight disagreement` |
| `mwmd{k}` | CE + Σ_ℓ α_ℓ KL(p_{M_ℓ(i)} ‖ p_i), Σα=1 | k peeled max-weight matchings | `--k_matchings k` |
| `mwmt1` | as mwmd1, weight w=m_ij−κ·u_ij (labeled val) | per epoch | `--match_weight teachable --kappa κ` |
| `gap1` | as mwmd1, weight \|acc_i−acc_j\| | per epoch | `--match_weight accgap` |
| `…-static` | matching computed once (epoch 0), frozen | never re-matched | `--rematch_every_epochs 0` |
| `…-rec{λ}` | recency-penalized selection w′=w−λ·r_ij·w̄ | per epoch | `--recency_lambda λ` |
| `…-anneal` | degree schedule k: 3→2→1 at the LR drops | per epoch | `--k_anneal 0:3,60:2,120:1` |

KL is torch's `KLDivLoss(batchmean)` with the **teacher as target** —
KL(p_teacher ‖ p_student) — exactly the paper's Eq. (2)–(4); temperature T=1 as in
the paper (a `--kd_T` knob exists for a future annealed-softness arm; when T≠1 the KL
is scaled by T²). The k peeled KL terms are kept **separate, never averaged** (the
DML_e lesson, §3.6 of the paper; tested directly in M5); their weights α_ℓ are
proportional to the positive part of each matched edge's weight (`--peel_weighting
weight`, Idea 2c design point (i)) with a `uniform` ablation.

### 1.2 Update rule — declared design decision

**Simultaneous.** For every batch, all K models' logits are computed (train mode, one
forward each) *before* any model steps; mimicry targets are detached pre-step logits;
each model then takes its own SGD step. No model ever trains on a peer's already-
updated weights within a batch, in any arm. (The paper describes alternating updates;
the legacy root-level trainer here is effectively simultaneous; the KD repo's C-002
episode showed this grade of mechanical choice can change outcomes — so it is fixed,
uniform across arms, and enforced by a parity unit test, `tests/test_trainer.py`.)

### 1.3 The matcher

At each refresh (default: every epoch, before training), each model's hard
predictions on the held-out validation set are collected, an edge-weight matrix W is
built from the chosen signal, and a **maximum-weight perfect matching** is solved —
exactly (bitmask DP) for K ≤ 16, greedy (the Preis/Hoepman ½-approximation order)
above. For k > 1 the matching is peeled: solve, delete its edges, solve again — the
weight-greedy k-matching, already edge-colored by weight rank (Idea 2c).

Weight signals (all symmetric, computed on the 5k validation set):

* `disagreement` (default, **label-free**): d_ij = P(pred_i ≠ pred_j). The benign-
  cohort v1 weight of Idea 2b — the matcher needs no ground truth, so in a
  distributed deployment the probe set can be unlabeled public data.
* `teachable` (labeled): w_ij = m_ij − κ·u_ij with mentor = the higher-val-acc member
  of the pair; the KD repo's MWM-D weight (its `src/policies/mwm.py`).
* `accgap` (labeled): \|acc_i − acc_j\| — deliberately included as the falsification
  arm for first-order modularity (H4).
* `random`: i.i.d. uniform weights ⇒ a fresh uniformly-random perfect matching every
  refresh (the Def-KT-style unselective control).

Epoch-0 matchings are computed from randomly-initialized models' predictions and are
effectively random in every arm — declared and harmless.

The recency penalty (rotation protection, Idea 1 point 5): r_ij decays by γ per
refresh (default 0.5) and increments for matched pairs; selection maximizes
w′_ij = w_ij − λ·r_ij·w̄ with w̄ the mean \|w\| over allowed edges (λ unitless).
A `--graph {complete,ring,rregular:d}` mask restricts matchings to a fixed
communication graph — implemented and unit-tested, scheduled only in deferred M7.

### 1.4 Communication accounting

Logged per epoch into every metrics row, computed analytically:

* **Per-batch logit traffic**: a model that teaches d partners sends
  d × (examples seen) × (num classes) floats per epoch. Dense/DML_e: d = K−1.
  Matched: d = k. Independent: 0.
* **Matcher overhead**: each model ships its n_val hard predictions (ints) per
  refresh that uses a measured weight (`disagreement`/`teachable`/`accgap`); random
  matching and static epochs cost 0.
* `comm_bytes_cum` = 4·floats + 2·ints, cumulative per model;
  `comm_bytes_cum_cohort` = ×K.

Concretely, at K=8 / CIFAR-100 / 45k train / 200 epochs: dense ≈ 25.2 GB sent per
model; k=1 matched ≈ 3.6 GB + 2 MB matcher overhead — a **7× reduction** (K−1 in
general), with the ledger in the CSVs rather than asserted.

---

## 2. Common protocol (paper §3.1 analogue)

| Item | Value | Note |
|------|-------|------|
| Dataset | CIFAR-100 (50k train / 10k test), top-1 accuracy | the paper's classification benchmark; CIFAR-10 supported |
| Validation/probe split | 5k carved from train, `split_seed=0`, shared by ALL runs | needed by the matcher + policy-facing metrics; the paper trains on 50k — our Independent/DML numbers are expected ~0.3–0.8 pp below Table 2, and all comparisons are internal (R1 checks the *gains*, not absolute numbers) |
| Networks | ResNet-32 (0.47M), MobileNet (3.2M), WRN-28-10 (36.5M) | the paper's Table 1 trio (InceptionV1 omitted; Market-1501 out of scope v1) |
| Optimization | SGD, Nesterov momentum 0.9, lr 0.1, ×0.1 every 60 epochs, 200 epochs, weight decay 5e-4, batch 64 | the paper's CIFAR-100 settings, identical across arms and architectures |
| Augmentation | 4-px reflection-pad random crop + horizontal flip; per-channel CIFAR-100 normalization | the paper's "standard augmentation" |
| Label noise (noisy regime) | 40% symmetric flips on TRAIN labels only, `noise_seed=42` shared across seeds; val/test clean | the KD repo's E5 convention |
| Seeds | 5 per cell (WRN-heavy cells may gate at 3 — §6); seed fixes init AND batch order, shared across arms | the paper reports single runs; KD standard is 20 (cheaper models) |
| Ensemble metric | argmax of mean posterior over the cohort | reporting only, never a training signal (except in the DML_e arm, where it is the point) |
| Workhorse cohort | K=8 | at K=2 all coupled arms coincide; K ≥ 6 is where matching bites (KD ideas.md) |
| Reproducibility & resilience | one process per run; `--run_tag` (= log entry id) stamped into run_id and CSVs; checkpoint/resume every 10 epochs (incl. all RNG streams and the shuffle generator) | KD C-001/C-002 conventions |
| Wulver 72 h wall | every script requests the 72 h max + `--signal=B:USR1@1800`: 30 min before the wall the trainer checkpoints at the end of the epoch in flight and the script requeues its own array task, which resumes bit-identically — runs longer than one window complete unattended. Preemption (`qos=low`) and node failures requeue automatically and resume from the last periodic checkpoint. Anything else: `sbatch --array=$(python tools/incomplete.py <exp>) slurm/<script>` resubmits exactly the unfinished indices | [D-002] |
| Allocation | `--account=ikoutis`, `--qos=low` on the gpu partition ([D-005]; the dept_dms allocation of [D-003] was tried and reverted). `--qos=debug` for smokes via command-line override | [D-005] |
| Environment | `bash tools/setup_env.sh` on a login node creates `/project/ikoutis/conda_env/dml-torch` (python 3.11 + requirements.txt; PyPI torch wheels bundle the CUDA 12 runtime); the sbatch scripts activate that path, `DML_CONDA_ENV` overrides | [D-004] |

Runner: `python -m src.run_experiment` (see `--help`); one run = one (cohort, arm,
seed) cell producing `{run_id}_metrics.csv` (one row/epoch) and
`{run_id}_matches.csv` (one row per matched pair per refresh).

---

## 3. The experiments

### R1 — Two-network cohorts: replication gate (paper Table 2)

**Purpose.** Validate the harness against the paper before interpreting anything
else. At K=2 every coupled arm coincides (dense = matched-k1), so this is purely
DML-vs-Independent — the paper's Table 2, minus InceptionV1.

**Cells.** {ResNet-32/ResNet-32, MobileNet/ResNet-32, MobileNet/MobileNet,
WRN-28-10/ResNet-32, WRN-28-10/WRN-28-10, WRN-28-10/MobileNet} × {indep, dml} × 5
seeds = 60 runs. Script: `slurm/r1_pairs.sbatch`.

**Paper's Table 2 (CIFAR-100 top-1, for calibration):**

| Net 1 | Net 2 | Ind 1 | Ind 2 | DML 1 | DML 2 | Δ1 | Δ2 |
|-------|-------|-------|-------|-------|-------|-----|-----|
| ResNet-32 | ResNet-32 | 68.99 | 68.99 | 71.19 | 70.75 | +2.20 | +1.76 |
| WRN-28-10 | ResNet-32 | 78.69 | 68.99 | 78.96 | 70.73 | +0.27 | +1.74 |
| MobileNet | ResNet-32 | 73.65 | 68.99 | 76.13 | 71.10 | +2.48 | +2.11 |
| MobileNet | MobileNet | 73.65 | 73.65 | 76.21 | 76.10 | +2.56 | +2.45 |
| WRN-28-10 | MobileNet | 78.69 | 73.65 | 80.28 | 77.39 | +1.59 | +3.74 |
| WRN-28-10 | WRN-28-10 | 78.69 | 78.69 | 80.28 | 80.08 | +1.59 | +1.39 |

**Gate.** The DML−Independent *gains* (Δ columns) reproduce in sign and roughly in
magnitude (within ~1 pp per cell at 5 seeds); absolute accuracies may sit ~0.3–0.8 pp
low (45k train). If the gate fails, stop and debug before M1.

### M1 — Matched vs dense at K=8: the headline table (Table 2's successor)

**Purpose.** The core 4×3: does matching preserve dense DML's gain at a fraction of
the communication (H1), and is the MWM-over-random gain selective (H2)? Plus the
diversity readouts (H3).

**Cells.** Arms {indep, dml, rand1, mwmd1} × regimes {resnet32:8 clean;
wrn28x10:4,resnet32:4 clean (heterogeneous — the paper's flagship big+small case at
cohort scale); resnet32:8 + 40% noise} × 5 seeds = 60 runs.
Script: `slurm/m1_headline_k8.sbatch`.

**Predictions.**
- P1 (H1): rand1 and mwmd1 within CI of dml on avg individual accuracy in all three
  regimes, > indep everywhere, at 1/7 the logit traffic.
- P2 (H2): mwmd1 − rand1 > 0 with CI clearing 0 in the hetero and noise cells; ≈ 0
  in clean homog. In the hetero cell, report per-architecture means (the CSVs carry
  `avg_test_acc_wrn28x10` / `avg_test_acc_resnet32`): the prediction localizes the
  gain in the *small* nets (the paper's own observation that small nets benefit most).
- P3 (H3): late-training ordering ρ(dml) > ρ(matched arms); ens−avg gap ordering
  reversed; dml's ensemble advantage over matched smaller than its ρ excess suggests
  (diversity, not individual accuracy, is what dense coupling spends).

**Headline figure.** The Pareto plot (analysis/plot.py `pareto_*`): final avg
individual accuracy vs cumulative bytes/model, one point per arm per regime.

### M2 — The degree dial: peeled k-matchings (Fig. 2's successor on the k-axis)

**Purpose.** k interpolates matched → dense at fixed cohort size: where on the
communication axis does dense DML's remaining advantage (if any) live (H5)? Tests
Idea 2c's predictions and its two design points (weight-proportional α; degree
annealing = "spend degree early, conserve diversity late").

**Cells.** resnet32:8 clean; new arms {mwmd2, mwmd3, anneal(3→2→1 at LR drops),
mwmd2-unif} × 5 seeds = 20 runs; anchors mwmd1, dml, (dmle from M5) shared from M1.
Script: `slurm/m2_degree_dial.sbatch`.

**Predictions.** Early avg accuracy ≈ k-independent; disagreement decays faster with
larger k; final ens−avg gap monotone ↓ in k; avg accuracy flat or gently peaked at
k* ≈ 2–3; anneal ≥ every fixed k on the (accuracy, final diversity) pair; mwmd2 ≥
mwmd2-unif (the weaker teacher should not get equal pull).

### M3 — Cohort scaling: communication constant in K (paper §3.5 / Fig. 2)

**Purpose.** The paper showed individual accuracy rises with cohort size under dense
coupling — with per-model communication growing as K−1. Does degree-1 MWM matching
track those gains at constant communication? This is the cleanest deployment-facing
claim: the cost of joining a cohort stops depending on its size.

**Cells.** resnet32:K for K ∈ {2, 4, 12} × {dml, mwmd1} × 5 seeds = 25 runs (K=8
from M1; the K=2 dml anchor comes from R1's identical resnet32:2 cells — re-running
it would duplicate run_ids — so K=2 adds only mwmd1, whose coincidence with dml at
K=2 is the sanity check).
Script: `slurm/m3_cohort_scaling.sbatch`.

**Predictions.** Both arms' avg accuracy rises in K (replicating Fig. 2's trend on
CIFAR); the dml−mwmd1 gap does not grow with K (H1 at scale); bytes/model: flat in K
for mwmd1, linear for dml — the money curve annotated with measured `comm_bytes_cum`.

### M4 — Rotation & persistence (the KD persistence dichotomy, ported)

**Purpose.** Is per-round re-matching load-bearing (H6)? KD theory: persistent
partners mutually memorize errors — per-edge ρ climbs, diversity drains. Supervised
DML's CE anchor may protect against this; measuring which is a result either way.

**Cells.** wrn28x10:4,resnet32:4 (hetero — where a frozen matching is most tempting,
pairing every WRN with "its" ResNet permanently) × {mwmd1-static, mwmd1-rec0.5,
mwmd1-rec1} × 5 seeds = 15 runs; plain mwmd1 from M1.
Script: `slurm/m4_rotation.sbatch`.

**Predictions.** Ranking recency ≥ plain re-match > static on avg accuracy and on
final diversity; in static runs, the ρ of *matched* pairs rises above the ρ of
unmatched pairs over training (computable from matches.csv + per-model predictions);
λ has an interior optimum (λ too large overrides the weight signal → drifts toward
rand1 behavior).

### M5 — Target structure: the DML_e lesson (paper §3.6)

**Purpose.** The paper found distilling from the averaged peer posterior (DML_e)
*worse* than averaging separate KLs — the finding that motivates matching in the
first place (individual teachers' fluctuations are functional; averaging destroys
them) and our never-average peel design. Replicate it in-harness so the design axiom
rests on our own data.

**Cells.** resnet32:8 clean; new arm {dmle} × 5 seeds = 5 runs; comparators dml,
mwmd1, mwmd2, mwmd2-unif from M1/M2. Script: `slurm/m5_target_structure.sbatch`.

**Predictions.** dml > dmle on individual accuracy (paper's finding, at K=8);
diversity ordering mwmd1 > mwmd2 > dml > dmle (target averaging is the strongest
homogenizer); mwmd2 (separate KLs) > a hypothetical averaged-pair target — proxied
by mwmd2-unif vs mwmd2 comparison from M2.

### M6 — The matching signal: what should the matcher maximize?

**Purpose.** Justify "who teaches whom = max-weight matching on *disagreement*"
against its cheaper shadows (H4), and check the damage term κ.

**Cells.** Regimes {hetero clean, resnet32:8 noise40} (where signals decouple) ×
{mwmt1 (teachable, κ=1), mwmt1k0 (κ=0), gap1 (accgap)} × 5 seeds = 30 runs;
label-free mwmd1 and rand1 anchors from M1. Script: `slurm/m6_weight_signal.sbatch`.

**Predictions.** gap1 ≈ rand1 (modularity, H4); mwmt1 ≈ mwmd1 ≥ rand1 (in benign
cohorts raw disagreement ≈ mutual teachable mass — Idea 2b's identity
d_ij = (m_ij + m_ji) + both-wrong-differently); κ matters little here (no
adversaries; deferred to the v2 anti-Oracle setting). A mwmt1-over-mwmd1 gap would
mean labels buy something the label-free weight misses — worth knowing either way,
since label-free matching is the deployable one.

### M7 (deferred) — Fixed communication graphs

Matching within a fixed graph G (ring / random 3-regular / complete): Idea 2b's
regime, the expander-vs-ring spectral story, and the edge-coloring round-robin
baseline. **Harness ready** (`--graph`, masked solvers, unit-tested); scheduled
after M1–M6 read out, as a second campaign — each arm is a full-length run and the
interesting cohort sizes are larger (K ≥ 16, where the greedy solver takes over).

---

## 4. Metrics and file schema

`{run_id}_metrics.csv`, one row per epoch — run-identity columns (run_id, run_tag,
cohort, dataset, K, arm, arm_label, target, seed, all matcher/optimizer knobs, noise
config, split sizes, cohort_params), then:

| Column | Meaning |
|--------|---------|
| `avg/std/min/max_test_acc` | individual test accuracy across the cohort |
| `avg_test_acc_{arch}` | per-architecture means (hetero cells) |
| `ensemble_test_acc`, `ens_minus_avg_test` | mean-posterior ensemble; the diversity dividend |
| `avg_val_acc`, `ensemble_val_acc` | policy-facing (validation) counterparts |
| `disagreement_val/test` | mean pairwise P(pred_i ≠ pred_j) |
| `rho_val/test` | mean pairwise error correlation |
| `model_{i}_arch/test_acc/val_acc/train_loss/ce_loss/kd_loss` | per-slot blocks |
| `k_current`, `lr`, `epoch_seconds` | schedule state |
| `comm_logit_floats_epoch`, `comm_matcher_ints_epoch`, `comm_bytes_epoch`, `comm_bytes_cum`, `comm_bytes_cum_cohort` | the communication ledger (§1.4) |

`{run_id}_matches.csv`, one row per matched pair per refresh: epoch, layer (peel
rank), i, j, arch_i, arch_j, raw and recency-penalized weight, weight mode, solver,
k. Together with per-model prediction columns this supports the per-edge ρ analysis
(M4) and "who got matched with whom" forensics.

Aggregation: `analysis/aggregate.py` (mean ± 95% CI per cell×arm×epoch; final-epoch
Welch t-tests, Bonferroni-corrected within cell) → `analysis/plot.py` (accuracy
curves, diversity trajectories, the Pareto plot, final heatmap).

## 5. Statistics

5 seeds per cell (3-seed gate for WRN-heavy cells, §6). Report mean ± 95% CI
(t-distribution); the headline comparisons (H1: dml vs mwmd1; H2: mwmd1 vs rand1 per
regime; H4: gap1 vs rand1; H6: static vs plain) get Welch t-tests with Bonferroni
correction within each cell. The paper reports single runs; we treat ±CI at 5 seeds
as the minimum for claiming a pp-scale effect. Seed pairing (same seed ⇒ same init
and batch order across arms) enables paired comparisons as a secondary analysis.

## 6. Cost and launch order

Estimates for A100-class GPUs, 200 epochs, batch 64 (±2×; R1's first completions
calibrate — each run logs `epoch_seconds`):

| Exp | Runs | GPU-h/run (est.) | Total (est.) | Priority |
|-----|------|------------------|--------------|----------|
| R1 pairs | 60 | 1–6 (arch-dependent) | ~200 | **1 — the gate** |
| M1 headline | 60 | 7 (r32:8) / 15–25 (hetero) | ~600 | **2 — the paper's core** |
| M2 degree dial | 20 | 7 | ~140 | 3 |
| M3 cohort scaling | 25 | 1–12 | ~145 | 3 |
| M5 target structure | 5 | 7 | ~35 | 3 (cheap, run with M2) |
| M4 rotation | 15 | 15–25 | ~250 | 4 |
| M6 weight signal | 30 | 7–25 | ~330 | 4 |
| **Total** | **215** | | **~1700 GPU-h** | |

At 3 seeds everything scales by 0.6 (~1000 GPU-h). Recommended sequence:

1. **Smoke + timing** (single R1 resnet32:2 dml run to completion; a 5-epoch hetero
   run for memory/timing on the full-A100 partition).
2. **R1** in full. Check the gate against Table 2 before proceeding.
3. **M1** — every headline claim lives here; M2/M5 alongside (same cell, cheap).
4. **M3**, then **M4/M6** once M1's hetero cells look sane.
5. M7 as campaign 2, informed by everything above.

Completion check per experiment: `python tools/incomplete.py <exp> --list` shows
every array index's last logged epoch; the bare command prints the `--array`
expression of unfinished indices for resubmission (empty when done).

## 7. Mapping to the original paper

| Paper element | Here |
|---------------|------|
| Table 1 (network sizes) | same trio; `tests/test_models.py` pins param counts |
| Table 2 (CIFAR-100 two-net cohorts) | R1 (replication gate) → M1 (its K=8 successor with matched arms) |
| Table 3 (Market-1501) | out of scope v1 (re-ID pipeline, not a harness limitation) |
| Table 4 (vs distillation) | not replicated; the Independent anchor and DML arm carry the comparison we need |
| §3.5 / Fig. 2 (cohort size) | M3, plus the communication axis the paper didn't have |
| §3.6 (how & why; DML_e) | M5 (DML_e), M1/M2 diversity trajectories (ρ, ens−avg) |
| — (no analogue) | M4 rotation, M6 matching signal, M7 topology: the new axes this program adds |
