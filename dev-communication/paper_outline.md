# Paper outline — "How Much Communication Does Deep Mutual Learning Need?"

Target: AAAI technical track (7 pages + references). Shape A: communication
efficiency as the spine, failure propagation as the second pillar. Every
section below names the log entries / result directories that feed it, so the
writing is largely assembly.

Abstract: the [D-017]-era draft (quadratic→linear; recovers almost the entire
benefit; iso-communication gap stark; ablation as secondary claim; add one
clause when R2 lands: "consistent across CIFAR-100 and CIFAR-10").

## 1. Introduction (0.75 pp)
- DML recap; the quadratic coupling cost Θ(K²) and the implicit assumption
  that partner choice should matter.
- Thesis: both are wrong — dose (degree) is the operative axis; and the same
  channel that carries the benefit carries failure, which is where structure
  DOES matter.
- Contributions list (5): (i) degree law + iso-comm quantification; (ii)
  selection/rotation/topology/connectivity nulls with the pool-saturation
  refinement; (iii) noise-conversion mechanism; (iv) failure propagation:
  contagion in fragile cohorts, dose-response/healing/quarantine law in
  robust ones; (v) replication-gated open harness + 400-run evidence base.

## 2. Related work (0.75 pp)
- DML lineage (Zhang et al. 2018); codistillation (Anil et al. 2018) — the
  closest prior; position: they establish distributed online distillation
  works; we ablate its communication structure systematically and add the
  failure channel.
- Decentralized/gossip SGD & topology (D-PSGD etc.): parameter averaging, not
  prediction exchange; our "structure inert" result + the gradient-coupling
  explanation (structure has nothing to transport when data is shared).
- Byzantine-robust FL: attack/defense framing vs our epidemiology of benign
  degradation. Confidence penalty / label smoothing: the KL-to-uniform
  mechanism behind zombie damage (and why strong hosts nearly shrug it off).
- Peer-learning societies: the ECAI-2023 nKDiff line (rationed oracle,
  localized knowledge — the regime where structure/selection DO matter);
  frames our setting as the delocalized endpoint.

## 3. Setup (0.75 pp)
- Cohort, arms (indep / dense / matched-k / fixed-graph), simultaneous
  updates, communication ledger (bytes; matcher probes), pairing discipline
  (same seed ⇒ same inits & batch order ⇒ per-model paired stats).
- Replication gate: R1 reproduces DML Table 2 (table in appendix).
  [results/suite/r1_pairs; log D-002..D-005]

## 4. The degree law (1.5 pp)  — the spine
- Table 1: K=12 clean ladder + K-sweep (K=2,4,8,12): matched-1 captures
  82–90%; dense premium +0.3–0.5 at 7–11× traffic.
  [m1_headline, m3_cohort_scaling, m7_topology clean; log D-006, D-011 reply]
- Fig. 1: accuracy-vs-communication Pareto; iso-comm readout (71.7% vs 40.6%
  at 3.6 GB; dense below the independent floor at matched's budget).
  [log D-009/D-011-era analysis; recompute script in analysis/]
- Noise conversion: Fig. 2 — indep 50.6 / coupled 58.3–59.2 at K=12 (+8.6
  dense, degree-1 = 90%); ensemble−individual gap closing 14→6.
  [m1 noise, m7 noise tier; log D-016 reply to D-011]
- R2 transfer table (CIFAR-10, when it lands). [r2_cifar10; D-017]

## 5. Nothing else matters (constructive channel) (1.25 pp)
- Selection null: MWM on 4 signals ≈ random (table; all cells).
  [m1, m1b, m6; log D-007/D-009]
- Rotation/staticness null (M4), target-structure null (DML_e, M5),
  peel-weighting null (M2).
- Topology-shape null at fixed degree (ring/prism/expander), incl. the
  noise-tier non-amplification. [m7; D-011 reply, D-016 reply]
- Connectivity = pool access only: clusters point-predictions hit within
  0.06 pp; −0.46 at pool-1 (p=.008), null by pool-3. Closes the transport
  question. [m7 clusters + D-012, D-016 reply]
- One honest caveat box: the ensemble trade-off (indep ensembles beat all
  coupled ensembles; coupling converts collective→individual at a diversity
  price). [D-010]

## 6. The failure channel (1.5 pp)  — the second pillar
- 6.1 Contagion in fragile cohorts: LeNet spontaneous deaths → graph-borne
  cascades; collapse rates by arm (0/5 indep … 3/5 dense; 0/10 disconnected);
  wave speeds (ring 10 ep, expander 3, rotation 2); quarantine.
  [m7l_lenet; D-013/D-014]
- 6.2 Dose-response in robust cohorts (M9): the table (ring −0.96 @α=1/2;
  deg-3 −0.86/−0.93; dense −0.13 @1/11; sealed −1.22 = dose + pool), one-hop
  locality, exact quarantine, ZERO deaths (R0=0), natural-zombie preview.
  [m9_zombie; D-015 + final reply]
- 6.3 Healing: recency forensics (≤10 ep −0.69 → 11–30 ep ≈ 0; r=.29 p=.03);
  rotation heals robust cohorts / kills fragile ones — the susceptibility
  threshold and the design inversion.
- Design table: safest protocol by host robustness × failure model.

## 7. Discussion & limitations (0.5 pp)
- When communication is actually the bottleneck (edge/decentralized, large K);
  bytes are small in absolute terms — honesty paragraph.
- Scope: one recipe; LeNet fragility is recipe-dependent (warmup would mute
  patient zero — but not the propagation law); capacity confound discussion.
- The localization conjecture (structure couples to gradients; nKDiff as the
  localized regime) — one paragraph, flagged as interpretation, seeds the
  follow-up.

## Appendix (no page limit)
- R1 replication table; full per-cell tables; spectral gaps; M9 per-arm
  details; harness/reproducibility (incomplete.py, tags, seeds); the M7-L
  bimodality forensics.

## Figures to produce (analysis/ scripts exist or are one-liners)
1. Pareto: accuracy vs cumulative bytes, all arms.  2. Noise conversion bars
(indep/deg1/dense × clean/noise, individual+ensemble).  3. Degree ladder with
K-sweep inset.  4. LeNet collapse: per-arm collapse rates + one death-wave
timeline.  5. M9: spatial damage profile (ring) + dose-response points +
healing scatter.

## Timeline (deadline ≈ +7 days)
D0: stage cifar10, sbatch r2 (40 tasks, slices). D0–D4: write §1–§6 from log.
D4–5: R2 lands → transfer table + abstract clause. D6: limitations, appendix,
polish. D7: submit.
