# Communication log

Reverse-chronological task/reply log for the DML (matched mutual learning) project —
**newest entry at the top**. Each entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: …`,
where `<kind>` is `Task`, `Note`, or a `<name>` reply. See [`README.md`](README.md) for
the format and the `[D-00N]` ID convention. To add an entry or reply, insert a new dated
section at the top; for a reply, cite the entry you are answering.

<!-- Add your next entry or reply here, above the older ones. -->

---

## 2026-07-20 — Note [D-008]: signal-structure diagnostic run — per-class is the only richer signal with structure; launching a per-class arm (M1b) to test it

`tools/signal_structure.py` was run on completed checkpoints (results/analysis/).
For each edge-weight signal it reports the cross-pair coefficient of variation
and the max-weight matching's realized-weight gain over a random matching —
i.e. how much accuracy-relevant signal MWM can actually extract.

| Signal | clean (indep) CV / gain | noise (indep) CV / gain | clean (mwmd1) gain |
|---|---|---|---|
| disagreement (scalar) | 0.015 / +1.4% | 0.013 / +0.7% | +1.4% |
| errorfield (per-example) | 0.020 / +1.9% | 0.019 / +2.2% | +2.5% |
| **perclass (per-class)** | **0.088 / +6.8%** | **0.092 / +8.2%** | +6.0% |
| teachable, accgap | (near-zero-mean artifact — ignore) | | |

Findings. (1) **Per-class accuracy-vector distance carries ~5x the exploitable
structure of scalar disagreement** — consistently across regimes and on both
independent (max-structure) and coupled cohorts, so coupling does not wash it
out. This vindicates the granularity intuition ([D-007]): a scalar can be
near-uniform (CV ~0.015 => matching ~ random) while the per-class profile is
structured (CV ~0.09). (2) **The per-example error field is NOT worth pursuing**
— barely above scalar (+2%). (3) The teachable/accgap "+20-30%" is a
divide-by-near-zero artifact (mean weight ~0.005 because all models have
near-equal accuracy in these homogeneous cohorts); it is not real structure,
and it foreshadows M6's mwmt/gap arms also being null here. (4) **Hetero could
not be measured** — the ~600 MB WRN checkpoints were cleaned from disk, so no
completed hetero cohort remained for any arm. Hetero is where per-class
structure should be LARGEST (architectural difference => genuinely different
per-class strengths), so that number is still open.

Decision. Rather than reconstruct a hetero checkpoint for a proxy measurement,
test the real thing: `slurm/m1b_perclass.sbatch` runs `--match_weight perclass`
(label `mwmpc1`) in the two slice-friendly regimes (resnet32:8 clean and +40%
noise), 5 seeds each = 10 tasks. The rand1 and mwmd1 anchors already exist in
results/suite/m1_headline at the same cohort and seeds, so aggregate.py compares
directly. The measured +6-8% per-class structure is ~5x the scalar's +1.4% that
produced the +0.2 pp mwmd1-over-rand1 effect, so if the accuracy effect scales
with weight-advantage, mwmpc1 could land ~+0.5-1 pp over random — worth the cheap
test. Falsifiable both ways: a clear margin (> ~0.4 pp paired) => the
multidimensional signal matters (a real positive for who-teaches-whom); a tie
(~+0.2 pp) => selection is flat even at per-class granularity (a robust null).
Hetero perclass (full-A100) is deferred pending this. Tracking: `tools/incomplete.py m1b`.

---

## 2026-07-20 — Note [D-007]: MWM verified correct; the null is real; adding a multidimensional (per-class / per-example) edge weight to attack it where it should live

Two things in this entry: (a) the MWM implementation is verified correct end to
end, so the H2 null is a finding not a bug; (b) a richer edge-weight family is
now implemented, motivated by the KD repo's error-field idea, to give H2 a
second shot in the regimes where scalar disagreement underperformed.

**(a) MWM is correct — the null is real.** Verified against the production
`matches.csv` and a controlled harness: the solver realizes the exact
brute-force optimum on every matrix; the pipeline beats a random matching by
+61.5% (structured) / +24.4% (moderate) in realized weight when structure
exists; each model provably mimics exactly its matched partner, symmetric,
alpha=1; weight_mode=disagreement, exact solver, 200 refreshes, full rotation.
The reason MWM ties random in the clean cohort is measured, not mysterious: the
real late-training disagreement matrix is near-uniform (the four max-weight
chosen pairs at epoch 199 span only 0.222–0.241), and on a matrix matching
those stats (0.23±0.01) MWM can beat random by only +3.1% in weight → the
observed ~0.2 pp. This is the first-order modularity prediction confirmed at the
mechanism level: in a homogeneous cohort the models are exchangeable, so
disagreement is uniform and "who teaches whom" is a flat choice.

**(b) The granularity ladder — why a richer signal might rescue H2 where the
scalar could not.** A scalar `P(pred_i != pred_j)` can be uniform across pairs
while multidimensional structure is rich: two models can disagree the same TOTAL
amount on DIFFERENT classes or examples. That decoupling is exactly what should
appear in the HETERO and NOISE cohorts (different architectures strong on
different classes; different models memorizing different noisy examples) — the
two regimes where H2 was predicted to win and the scalar signal disappointed.
The KD repo already has the deep version of this (`plan/theory_plan.md` D4: the
per-example error field `E_i` and error-field overlap `rho_ij`, used as the
diversity term `w - beta*rho`); the per-CLASS accuracy-vector distance is its
coarser, interpretable sibling. Both are now implemented as edge weights:

| mode | weight | granularity |
|---|---|---|
| `disagreement` (done) | P(pred_i != pred_j) | scalar |
| `perclass` (new) | ‖acc_i − acc_j‖ over classes | per-class competence profile |
| `errorfield` (new) | P(exactly one errs) = mean(e_i XOR e_j) | per-example error field |

`src/matching.py` (`perclass_distance_weights`, `errorfield_distance_weights`),
CLI `--match_weight {perclass,errorfield}` (arm labels `mwmpc1`, `mwmef1`), 83
tests pass. A synthetic check confirms the discriminating power: on models with
equal accuracy but complementary per-class strengths, per-class distance shows
CV 0.275 and +29% MWM-gain-over-random vs the scalar's 0.159 / +16% — the richer
signal sees structure the scalar half-misses.

**Disciplined next step (cheap, before any GPU-hours): measure, then maybe
run.** `tools/signal_structure.py` loads one completed checkpoint, rebuilds val
predictions, and reports each signal's cross-pair CV and MWM-vs-random weight
gain, per regime. Run it on one clean, one hetero, one noise checkpoint (they
live on the cluster; checkpoints are gitignored):

    python tools/signal_structure.py <ckpt.pt> --cohort <spec> --dataset cifar100 --data_dir data

Decision rule: if per-class/error-field CV and gain are ~scalar in every regime,
the MWM null is robust to signal choice — a clean, strong paper result
("selection is flat at every granularity"). If they are materially larger in
hetero/noise, launch a matched arm there with `--match_weight perclass` (or
`errorfield`) — a targeted second shot at H2, aimed where the theory says the
structure is. Either outcome is publishable; the diagnostic decides which
without spending a GPU-hour on a guess.

---

## 2026-07-20 — Claude: THE M1 VERDICT — R1 closes at 60/60; dense keeps a real half-point; matched keeps 80–95% of the benefit at 1/7 the bandwidth; and the noise cell produces the largest effect of the study (re: [D-001])

R1 is complete (60/60; the retrained wrn:2 row lands at indep 78.76±0.28 vs dml
80.04±0.22, Δ +1.28 against the paper's +1.39/+1.59 — the gate file is closed).
M1's clean cell is at full seeds on three of four arms, and the paired statistics
now speak clearly. The headline table (final epoch, mean±std):

| Regime | indep | dml (25.2 GB) | mwmd1 (3.6 GB) | rand1 (3.6 GB) |
|---|---|---|---|---|
| clean r32:8 | 69.39±0.15 (5) | **72.07±0.07 (3)** | 71.66±0.17 (5) | 71.44±0.25 (5) |
| hetero 4W+4R | 74.13±0.13 (5) | **75.92±0.11 (4)** | 75.42±0.22 (3) | 75.56±0.08 (3) |
| noise40 r32:8 | 50.81±0.28 (3) | **59.21±0.21 (2)** | 58.70±0.61 (2) | 58.98 (1) |

**Verdict 1 — H1: strict matched-equals-dense is REJECTED; the honest number is
"80–95% of the benefit at 14% of the communication."** Dense holds a small but
statistically real premium in every regime: paired dml−mwmd1 = +0.52±0.13 pp
(p=0.020) clean; dml−rand1 = +0.34±0.09 (p=0.024) hetero; ~+0.4 noise. Framed as
fractions of the coupling benefit over independent: matched captures 84% of
dense's gain in the clean cell (2.26 vs 2.68 pp), 72–80% hetero, 86–97% noise —
at 1/7 the logit traffic. The iso-communication view (earlier entry) is
unchanged and remains lopsided: at equal bytes matched is 25–30 pp ahead,
because dense's last half-point costs 7× the bandwidth. That asymmetry — a
half-point premium priced at sevenfold communication — IS the paper's central
quantitative claim now, and it is arguably cleaner than a tie would have been.

**Verdict 2 — the noise cell delivers the study's largest effect, and it is a
conversion phenomenon.** Under 40% label noise, independent training collapses
individuals to 50.81 while their ENSEMBLE holds 64.79 — a 14 pp ensemble−
individual gap: the noisy models are individually wrecked but collectively
knowledgeable. Every coupled arm converts that collective knowledge into
individual competence: +8.48±0.13 pp for dense (paired p=0.007), +7.3–8.2 for
matched, gap compressed from 14 to ~6 pp. Peer mimicry under label noise is not
a small regularizer here — it recovers most of the ensemble's buried signal
into every single model, and degree-1 matching does nearly all of what dense
does. This connects directly to the KD repo's noise-filter narrative (its E5)
and deserves its own figure.

**Verdict 3 — H2: the weighting signal exists, is small, and lives in the WRONG
regime.** Clean cell at full 5v5: mwmd1 − rand1 = +0.21±0.21 pp, positive in
5/5 seeds, paired p=0.082 — a weak but sign-consistent effect, concentrated in
the late training phase (epoch-70 paired read +0.27±0.09; the mechanism note:
early disagreement is uniformly ~1.0, so MWM IS random early by construction;
structure only differentiates late). Hetero currently reads −0.29 on one paired
seed (open; seeds in flight), noise unresolved. This is the OPPOSITE selectivity
from the design's prediction (win in hetero/noise, tie clean). Honest posture:
report the clean-cell effect with its marginal p, complete the hetero/noise
pairs, and let M6's teachable-weight arms say whether a labeled signal does
better — but the "who teaches whom" dial is, on this evidence, worth at most a
fifth of a point, not the paper's centerpiece.

**Verdict 4 — H3 (diversity conservation): NULL, uniformly.** All coupled arms
sit at ρ ≈ 0.65 and ens−avg ≈ 4.1–4.2 in every regime — matched pays exactly
dense's diversity price for its slightly smaller gain. The interesting residual:
INDEPENDENT ensembles still beat every coupled ensemble (76.93 vs 75.6–76.2
clean; 81.21 vs 79.6–80.1 hetero) — coupling of any structure trades ensemble
ceiling for individual floor. M5's dmle arm (now running) completes this axis.

**Paper shape after today:** (1) the 80–95%-at-1/7 quantification + iso-comm
Pareto; (2) the noise-conversion result; (3) the drag-dilution mechanism (K=2
extreme-gap drag, falsified at K=8 — both documented in earlier entries); (4)
the honest small-and-misplaced weighting effect. Remaining data that can still
move the story: dml clean seeds 2/5 and the noise-cell seed completions (top-up
statistics), hetero mwmd1/rand1 completions (H2-hetero), M2's degree-anneal arm
(can annealing capture dense's early +5 pp advantage at matched's late cost?),
and M5's DML_e. M4/M6 remain gated on whether there is a weighting effect worth
mechanising — current evidence says spend those ~350 GPU-h carefully.

---

## 2026-07-19 — Claude: first multi-seed hetero read — the drag prediction is falsified, H1/H2/H3 lean conservative (re: [D-001])

The heterogeneous cohort (4×WRN-28-10 + 4×ResNet-32) now has real seed counts on
three arms, and it forces three honest updates. Final-epoch means:

| Arm | n | cohort avg | WRN side | ResNet side | ens | ens−avg | ρ | bytes/model |
|---|---|---|---|---|---|---|---|---|
| indep | 4 | 74.11±0.14 | 78.81 | 69.41 | 81.20 | 7.09 | 0.567 | 0 |
| dml (dense) | 2 | **75.91±0.08** | 79.85 | 71.97 | 80.18 | 4.27 | 0.657 | 25.2 GB |
| rand1 | 2 | 75.54±0.10 | 79.31 | 71.76 | 79.72 | 4.18 | 0.657 | 3.6 GB |
| mwmd1 | 1 | 75.31 | 79.19 | 71.43 | 79.41 | 4.10 | 0.654 | 3.6 GB |

**1. My drag prediction is falsified — cleanly.** I predicted dense coupling would
drag the WRNs below their 78.81 independent baseline in this cohort (as it did in
R1's wrn+mobilenet at K=2). The opposite happens: *every* coupled arm LIFTS the
WRNs — dense +1.04, random +0.50, MWM +0.38 — and lifts the ResNets ~+2 to +2.6.
This is now robust (multiple arms, n=2). The mechanism is the dilution argument
from the previous entry: at K=8 with a ~9 pp competence gap, each WRN's mimicry
target is dominated by strong peers, so there is no weak-partner drag. The K=2
wrn+mobilenet drag needed BOTH an extreme gap (14 pp) AND an undiluted single
partner. So the honest paper claim is not "dense drags big models" but "drag is a
function of competence-gap × coupling-concentration; it appears at K=2 with an
extreme-gap partner and vanishes at K=8 under dilution." That is a more careful and
more defensible statement than the one I registered.

**2. H1 (matched vs dense) — dense noses ahead here, ~0.4 pp.** dml 75.91 vs rand1
75.54 (Δ +0.37) vs mwmd1 75.31 (Δ +0.60), with tight ±0.08–0.10 variance. This is
NOT the clean tie H1 hoped for — dense wins a small but apparently real margin in
the hetero cohort. It does not dent the communication story: +0.4 pp of accuracy
for 7× the bandwidth is a trade almost any distributed deployment takes, and
matched still crushes independent (+1.4 pp). But "matched equals dense" needs
qualification — at K=8 it is "matched within ~0.4 pp of dense at 1/7 the
communication." The clean-cohort H1 verdict is imminent (dml seeds 4/5 finally
advancing, epochs 105/93) and will say whether the ~0.4 pp dense edge is
hetero-specific or general.

**3. H3 (matched conserves diversity) is NOT appearing in the hetero cohort.** All
three coupled arms sit at ρ ≈ 0.655 and ens−avg gap ≈ 4.1–4.3 — dense and matched
pay the *same* diversity price for their gains. The prediction that dense
homogenizes more than matched does not show here. (Whether it appears in the clean
cohort awaits clean dense.)

**4. H2 (MWM > random) — within noise everywhere so far, leaning modularity-tie.**
Clean: mwmd1 71.66 (n=5) vs rand1 71.33 (n=1). Noise: mwmd1 58.70 (n=2) vs rand1
58.98 (n=1). Hetero: mwmd1 75.31 (n=1) vs rand1 75.54 (n=2). MWM is not beating
random in any regime yet — consistent with H2's *conservative* prediction (the
first-order modularity tie), and against the sharper second-order hope. BUT rand1
is badly undersampled (n=1 in two of three regimes), so this is provisional; the
pending rand1 seeds are now the highest-value runs in the queue for the
who-teaches-whom question.

**Net:** the communication/scaling story (the publishable floor) is intact and
strong. The two "full-win" storylines — MWM-beats-random, and matched-conserves-
diversity — are both currently reading NULL, which would land us at the "solid
win" rung (communication result + the drag-dilution mechanism) rather than the
top. Not decided: clean dense H1, and rand1 at full seeds. Those two batches
determine which paper this is.

---

## 2026-07-19 — Claude: R1 at 52/60 — the gate is PASSED (re: [D-001])

With the mobilenet cells and most WRN cells complete, the Table-2 comparison is
final in all but one row, and the verdict no longer needs hedging:

| Cell | Arch | Indep (ours) | DML (ours) | Δ ours | Δ paper |
|---|---|---|---|---|---|
| resnet32:2 | resnet32 | 69.56±0.33 (5) | 70.95±0.26 (5) | +1.39 | +2.20/+1.76 |
| mob+r32 | mobilenet | 64.55±0.29 (5) | 66.18±1.17 (5) | +1.63 | +2.48 |
| mob+r32 | resnet32 | 69.55±0.46 (5) | 70.73±0.64 (5) | +1.18 | +2.11 |
| mob:2 | mobilenet | 64.60±0.30 (5) | 66.74±0.19 (5) | +2.14 | +2.56/+2.45 |
| wrn+r32 | resnet32 | 69.41±0.33 (5) | 71.19±0.26 (2) | **+1.77** | **+1.74** |
| wrn+r32 | wrn28x10 | 78.93±0.37 (5) | 79.12±0.02 (2) | **+0.20** | **+0.27** |
| wrn:2 | wrn28x10 | 78.98 (1; recovery seeds queued) | 80.04±0.22 (5) | +1.06 | +1.59/+1.39 |
| wrn+mob | mobilenet | 64.44±0.40 (5) | 67.24±0.09 (4) | +2.80 | +3.74 |
| wrn+mob | wrn28x10 | 78.87±0.15 (5) | 76.61±0.33 (4) | **−2.27** | +1.59 |

Every row reproduces the paper's sign and approximate magnitude — the wrn+r32
deltas land within 0.03–0.07 pp of the published values, which is as close as
replication gets — except the documented wrn+mob WRN drag, now confirmed at
essentially full seed count (−2.27±0.33) and standing as our motivating exhibit
for selective coupling rather than as a defect. The harness is validated;
everything M1 and beyond reports can be taken at face value. Remaining R1
housekeeping: the four wrn:2 indep recovery tasks and a few wrn dml seed
top-ups, none of which can change the verdict.

M1 status alongside: first noise-regime completion (rand1 at 40% noise: 58.98
avg vs 71.33 clean — no comparators yet); clean K=8 dml seeds at epochs ~40–50;
the hetero coupled cells are queued behind them. The H1 and drag verdicts remain
the next milestones.

---

## 2026-07-19 — Note [D-006]: incident — "completed" tasks with frozen CSVs; two mechanisms found and fixed

The wrn:2 independent tasks (R1 indices 40–44) showed sacct COMPLETED with ~3 h
elapsed each, while their metrics CSVs sat frozen at epochs ~32–67. Diagnosis
found two independent mechanisms compounding, both now fixed in code:

**1. Orphaned CSV writers (data loss).** The trainer held one open append handle
per CSV for the whole run. During the git autostash-conflict episode on the
cluster, git rewrote the dirty results CSVs (git replaces files by inode), so
every running job's handle pointed at the orphaned old inode from that moment
on — the visible files froze at the stash-time snapshot while the jobs kept
appending into the void. Rows written after the rewrite are unrecoverable, but
the CHECKPOINTS are unaffected (written via atomic tmp+rename — a fresh inode
each time), so no model state was lost anywhere. *Fix:* CsvWriter now reopens
the file by path on every write (open-append-close, once per epoch — zero cost);
any future file replacement self-heals on the next row. *Practice note:* plain
`git pull` (merge) never touched results files and was always safe; it was the
stash apply that rewrote them. With the fix, even that is safe.

**2. The requeue race (tasks silently leaving the queue).** SLURM's preemption
with GraceTime resets a job's end time to now+grace — which makes our
`--signal=B:USR1@1800` fire immediately, not just at the 72 h wall. The trainer
then checkpoints and exits 85 (by design), but the script's next moves were
`scontrol requeue; exit 0` — and the clean exit races the requeue: SLURM records
COMPLETED and the task vanishes from the queue mid-run. That is exactly the
simultaneous 19:14 "completions" of tasks 40/41/42/44 (a preemption wave) and
43's second instance dying at epoch ~192. *Fix:* after `scontrol requeue` the
script now WAITS to be killed (the kill is the proof the requeue took) and, if
no kill arrives in 5 minutes, exits 75 — a loud FAILED that
`tools/incomplete.py` surfaces. It never exits 0 on this path again. A
side-benefit of understanding this: USR1-at-preemption is actually a feature —
the trainer gets a graceful checkpoint at the epoch boundary even under
preemption, when the grace window allows it.

**Recovery (one-time, on the cluster):**
1. `git pull` (picks up both fixes; running jobs keep old code until restarted).
2. Resubmit the affected R1 tasks: `sbatch --array=40-44 slurm/r1_pairs.sbatch`
   — their checkpoints are at epoch ~190+, so each finishes in minutes. Their
   CSVs will carry a gap (frozen rows → final rows); final-epoch gate numbers
   are unaffected, and if we want gap-free curves for these five seeds later,
   a `--resume`-less rerun is one array line.
3. Restart everything currently RUNNING so it picks up fixed code and fresh
   file handles (each resumes from its ≤10-epoch-old checkpoint):
   `scontrol requeue $(squeue --me -h -r -n dml_r1,dml_m1 -t R -o "%i" | paste -sd' ')`
4. Sweep for any other task that left the queue unfinished:
   compare `python tools/incomplete.py <exp>` against `squeue --me -r` per the
   MISSING one-liner in the 2026-07-19 exchange, and resubmit the difference.

Audit note for analysis: runs whose CSVs contain an epoch gap (the five wrn:2
indep seeds, possibly a few M1 cells caught in the same stash window) are
identifiable by non-contiguous epoch columns; `analysis/aggregate.py` uses
final-epoch rows and per-epoch means, both of which tolerate gaps, but curve
figures for those seeds will show a hole over the frozen interval.

---

## 2026-07-19 — Claude: first M1 completions — matched mutual learning works, +2.4 pp over Independent at degree 1, and the matcher's dynamics are textbook (re: [D-001])

The first matched-arm runs finished (resnet32:8 clean: mwmd1 seeds 2/4/5, rand1
seed 5, indep seeds 2/3/4; dense dml still at epochs ≤ 25). Final-epoch numbers:

| Arm | Avg individual | Ensemble | Ens−avg gap | ρ | Bytes/model |
|---|---|---|---|---|---|
| indep (n=3) | 69.31±0.12 | 76.97 | 7.66 | 0.555 | 0 |
| mwmd1 (n=3) | **71.75±0.13** | 75.87 | 4.12 | 0.657 | 3.6 GB |
| rand1 (n=1) | 71.34 | 75.47 | 4.14 | 0.657 | 3.6 GB |
| dml (n=0) | — pending — | | | | (will be 25.2 GB) |

**What this already establishes.** Degree-1 matched mimicry at K=8 lifts average
individual accuracy by **+2.44 pp over Independent** — noticeably more than dense
DML bought at K=2 (+1.39 in R1) — at one-seventh of dense coupling's
communication (the ledger columns confirm 3.6 GB vs the 25.2 GB dense will log).
The cohort-size benefit at constant per-model communication, which is M3's
thesis, is already visible here. What it does NOT yet establish is H1 proper —
whether matched *equals dense* at K=8 — because the dml cells are the slowest to
schedule. That comparison is the single number to watch next.

**mwmd1 vs rand1: direction interesting, evidence insufficient.** 71.75 vs 71.34
(+0.42), and +0.51 on the one paired seed. H2 predicted a *tie* here (clean
homogeneous = the modularity-flat regime); Idea 2b's sharper secondary claim —
that in mutual sessions the first-order terms cancel pair-wise, so
disagreement-weighting works at second order even in homogeneous cohorts — would
predict exactly a small mwmd1 edge. At n=1 rand1 this is a hypothesis, not a
finding; the running rand1 seeds decide it.

**Matcher diagnostics (the [D-001] epoch-0 check): all clean.**
- Epoch 0: every pairwise disagreement is 1.000 (random-init models agree
  nowhere), so all matchings tie and the solver's deterministic tie-break picks
  (0,1)(2,3)(4,5)(6,7) — effectively arbitrary, as declared. One honest nuance:
  at epoch 0 the matching is deterministic-under-ties rather than sampled.
- The mean matched-edge weight then *depletes* over training — 0.92 (ep 1) →
  0.54 (ep 10) → 0.30 (ep 120) → 0.22 (ep 199) — the disagreement-consumption
  dynamic Idea 2b predicts, measured directly in matches.csv.
- Rotation is healthy: every model partners with all 7 peers across the run;
  consecutive-epoch partner-repeat rate 0.21 vs ~0.14 for uniform random — mild
  persistence (max-weight re-selects high-disagreement pairs), which is exactly
  the dial M4's recency arms modulate.

**Diversity ledger so far:** both coupled arms trade ensemble for individuals
(ensemble 75.5–75.9 vs indep's 76.97; ρ 0.657 vs 0.555) — same currency exchange
R1 showed at K=2. The H3 question is whether dense dml pays MORE for the same
individual gain; pending its completions.

R1 meanwhile: 37/60, wrn:2 now indep 78.98 (1 seed) vs dml 79.98±0.20 (4) →
Δ +1.00 heading toward the paper's +1.39/+1.59. No change to the gate verdict.

---

## 2026-07-18 — Claude: R1 at 33/60 — gate holds, WRN:2 matches the paper to 0.2 pp, and one qualitative deviation worth having (re: [D-001])

Extended read after today's pushes. The table (complete runs only, mean±std):

| Cell | Arch | Indep | DML | Δ ours | Δ paper |
|---|---|---|---|---|---|
| resnet32:2 | resnet32 | 69.56±0.33 (5) | 70.95±0.26 (5) | **+1.39** | +2.20/+1.76 |
| mob+r32 | mobilenet | 64.55±0.29 (5) | 66.18±1.17 (5) | **+1.63** | +2.48 |
| mob+r32 | resnet32 | 69.55±0.46 (5) | 70.73±0.64 (5) | **+1.18** | +2.11 |
| mob:2 | mobilenet | 64.70 (1) | 66.72±0.10 (3) | **+2.02** | +2.56/+2.45 |
| wrn+r32 | resnet32 | 69.16 (1) | 71.37 (1) | **+2.21** | +1.74 |
| wrn+r32 | wrn28x10 | 79.14 (1) | 79.14 (1) | **+0.00** | +0.27 |
| wrn:2 | wrn28x10 | (in flight) | **80.07±0.12 (3)** | — | 80.28/80.08 |
| **wrn+mob** | mobilenet | 64.60±0.31 (2) | 67.25±0.15 (2) | **+2.64** | +3.74 |
| **wrn+mob** | wrn28x10 | 78.82±0.24 (2) | **76.36±0.28 (2)** | **−2.46** | **+1.59** |

Everything is the paper's story — every small-net delta positive at sensible
magnitudes, the wrn+r32 resnet gain (+2.21) right on the paper's +1.74, the WRN
barely moved by a resnet partner (+0.00 vs +0.27), and the WRN:2 DML cell landing
at 80.07 against the paper's 80.28/80.08 — except one row, and that row is
informative rather than embarrassing.

**The deviation: our WRN LOSES 2.46 pp when densely coupled to MobileNet** (paper:
gains 1.59). The trajectory makes the mechanism plain: the coupled WRN tracks its
independent twin's shape but plateaus ~2.3 pp lower from the first LR drop onward,
with its KL term still ≈ 0.26 at epoch 199 — it spends the whole run being pulled
toward its partner's posterior. The root cause is the already-flagged MobileNet
calibration gap: the paper's MobileNet sat ~5 pp below its WRN; ours sits ~14 pp
below (64.6 vs 78.8). At that competence gap, dense mimicry of the weak partner
stops being regularization and becomes drag. Same code, same settings, produces
+1.2 to +2.6 everywhere else and reproduces WRN:2 to 0.2 pp — this is a property
of the pairing, not a bug.

Two consequences. First, for the writeup: the wrn+mob row is our cleanest
motivating exhibit for *selective* coupling — dense DML has no mechanism to notice
that a partner is too weak to imitate; the teachable weight w = m − κ·u (M6's
mwmt arms) measures exactly the damage term this row realizes. Second, a sharpened
prediction for M1's heterogeneous cell (4×WRN + 4×resnet32, a ~9 pp competence
gap): dense dml should show WRN-side drag relative to indep, and the matched arms
— which concentrate mimicry on the highest-disagreement partner rather than
averaging over all peers — should show less of it. If mwmd1 protects the WRNs
while keeping the resnet gains, that is a headline figure, not a footnote.

**Status:** 33/60 R1 complete; wrn:2 indep (5 seeds, epochs ~30–70) and the
remaining wrn+mob/wrn+r32 seeds close the table. M1 is running — its resnet32:8
indep/dml cells are at epochs ~0–70; rand1/mwmd1 cells not yet started, so the
epoch-0 matching-randomness check is still pending.

---

## 2026-07-18 — Claude: first R1 outcomes — the gate is provisionally PASSING (re: [D-001], partial batch)

Read from the 20/60 completed R1 runs pushed today (all resnet32:2 and
mobilenet+resnet32 cells, one mobilenet:2 seed; the rest mid-training or queued —
`tools/incomplete.py r1` currently returns `19-28,30-59`). Three findings, then the
recalibrated costs.

**1. The gate table so far.** Final-epoch test accuracy, mean±std over completed
seeds, against the paper's Table 2:

| Cell | Arch | Ours indep | Ours DML | Ours Δ | Paper indep | Paper Δ |
|---|---|---|---|---|---|---|
| resnet32:2 | resnet32 | 69.56±0.33 | 70.95±0.26 | **+1.39** | 68.99 | +2.20/+1.76 |
| mob+r32 | mobilenet | 64.55±0.29 | 65.86±1.09 (n=4) | **+1.31** | 73.65 | +2.48 |
| mob+r32 | resnet32 | 69.55±0.46 | 70.56±0.60 (n=4) | **+1.01** | 68.99 | +2.11 |
| wrn+r32 (in flight, ep 133) | wrn28x10 | 78.76 (1 seed) | — | — | 78.69 | +0.27 |

Every DML−Independent delta is positive with tight seed variance, at roughly half
to two-thirds of the paper's magnitudes — same sign, same ballpark, and note the
paper's deltas are single-run numbers while ours carry error bars (the unofficial
implementation this repo started from reported +1.20 for the resnet pair; we get
+1.39±0.4-ish, right between it and the paper). Absolute calibration is
remarkable where it matters: ResNet-32 independent lands 0.6 pp ABOVE the paper
despite our 45k train split, and the one near-final WRN-28-10 independent reads
78.76 vs the paper's 78.69. I call the gate provisionally passed; the WRN cells
will finish the table.

**2. The one discrepancy, flagged honestly: MobileNet absolutes.** Our MobileNet
sits ~9 pp below the paper (64.6 vs 73.65). The paper never specifies its CIFAR
MobileNet adaptation, and 73.65 is high for a standard MobileNet-v1/32×32 port
(ours is the usual stride-1-stem variant at the paper's own 3.3M param count,
learning-rate recipe identical to the other nets). Every internal comparison is
unaffected — all arms share the architecture — but Table-2-style absolute
comparisons for MobileNet rows should carry this caveat in any writeup. Not a
blocker; worth one line in the paper.

**3. A free mechanism observation, already on-message for H3.** At K=2, dense
coupling raises individuals but drains the ensemble: resnet32:2 DML individuals
beat Independent by +1.39 while the DML *ensemble* trails the Independent
ensemble (72.85 vs 73.15), with exactly the diversity signature the theory
expects — pairwise error correlation 0.68 vs 0.55, ensemble−individual gap 1.90
vs 3.59, disagreement 0.22 vs 0.31. The mimicry benefit is real and it is paid
for out of diversity. This is the currency the matched arms are designed to
spend more carefully; M1/M2 will show whether they do.

**4. Costs recalibrated from measured `epoch_seconds`** (median): resnet32:2
18.9 s/ep (~1.0 h/run), mobilenet pairs ~15 s/ep (~0.8 h), wrn+r32 34.1 s/ep
(~1.9 h — my estimate said 6–10 h, so 3–5× cheaper). Implied per-model rates
(r32 ≈ 9.5, mobilenet ≈ 7, wrn ≈ 24.5 s/ep) project: resnet32:8 ≈ 4.2 h/run,
hetero 4W+4R ≈ 7.6 h/run, K=12 ≈ 6.3 h/run. Suite totals drop roughly in half
vs experiments.md §6: R1 ≈ 80, M1 ≈ 320, M2 ≈ 85, M3 ≈ 90, M4 ≈ 115, M5 ≈ 20,
M6 ≈ 180 — **≈ 890 GPU-h for everything at 5 seeds**. No task is anywhere near
the 72 h wall; the self-requeue machinery is insurance, not a dependency.

**Verdict / actions:** M1 (already submitted) proceeds with confidence; nothing
to cancel, nothing to change. When R1 completes: `tools/incomplete.py r1` should
print nothing, then the full six-row gate table goes in a follow-up reply. The
epoch-0 matches.csv randomness check from the [D-001] list applies to M1's
matched arms once those start landing.

---

## 2026-07-17 — Note [D-005]: allocation reverted to ikoutis / qos=low

The switch to the department allocation ([D-003]) is rolled back per Ioannis:
all seven suite scripts submit with `--account=ikoutis` and `--qos=low` again.
With `low` being preemptable, [D-002]'s original preemption story applies as
written — preempted tasks are requeued by SLURM automatically (`--requeue`)
and resume from their last periodic checkpoint (every 10 epochs), while the
USR1 self-requeue continues to carry runs across the 72 h wall. Nothing else
changes: env, data staging, recovery via `tools/incomplete.py`, and the debug
QOS for smoke runs (via `srun/sbatch --qos=debug` command-line overrides) all
work as before. If the department allocation becomes usable later, no code
needs to change — `sbatch --account=dept_dms --qos=<qos>` on the command line
overrides the in-script values per submission.

---

## 2026-07-17 — Note [D-004]: the conda environment is now a one-command setup

[D-001]'s checklist step 1 ("create/point `DML_CONDA_ENV` at a torch ≥ 2.0
env") is now concrete: run `bash tools/setup_env.sh` once on a Wulver login
node. It loads Miniforge3, creates a prefix env at
`/project/ikoutis/conda_env/dml-torch` *(updated same day, per Ioannis: the env
lives on project storage — off the `$HOME` quota, shareable within the group —
not under `$HOME/envs` as this entry first said)*, python 3.11, and
pip-installs `requirements.txt` — the PyPI torch/torchvision wheels bundle the
CUDA 12 runtime, so no cluster CUDA module is involved and the GPU nodes'
driver is all that's needed. All seven sbatch scripts activate
that same path by default; exporting `DML_CONDA_ENV=<other path>` before both
setup and submission points everything at a different env (e.g. to reuse an
existing torch env from the KD project).
Two expectations to save head-scratching: on the login node the script's sanity
check prints `cuda available: False` — login nodes have no GPU, that is normal —
and the script's header gives a one-line `srun --qos=debug` command that
verifies CUDA on an actual A100 before the first real submission. After setup:
`python tools/stage_data.py cifar100`, optionally `pytest tests/` (~2 min,
CPU-only), then submit R1.

---

## 2026-07-17 — Note [D-003]: suite runs under the department allocation (dept_dms / high_dept_dms+)

All seven suite scripts now submit with `--account=dept_dms` and
`--qos=high_dept_dms+` (the department's priority QOS) instead of the personal
account with `qos=low`. Two consequences worth recording. First, scheduling:
the priority QOS should start the big arrays (R1's and M1's 60 tasks) much
faster than `low` would. Second, a correction of scope to [D-002]: its remarks
about preemption applied to the preemptable `low` QOS; under `high_dept_dms+`
jobs are not preempted, so the `--requeue` + periodic-checkpoint machinery now
covers only node failures and admin drains, while the USR1 self-requeue remains
the mechanism that carries runs across the 72 h wall. Nothing about recovery
changes: `tools/incomplete.py` works the same, and `sbatch --account/--qos`
flags on the command line can still override the in-script values if a batch
ever needs to fall back to another allocation.

---

## 2026-07-17 — Note [D-002]: 72-hour-wall resilience — self-requeueing arrays and one-line recovery

*(Supersedes the resilience paragraph of [D-001]; no reply needed.)*

Wulver caps every job at 72 hours, and our cost estimates carry ±2× uncertainty —
so a WRN-heavy M1/M4/M6 task could plausibly outlive its window. The suite now
survives that without anyone watching:

1. **Self-requeue across the wall.** Every script requests the full 72 h and
   `--signal=B:USR1@1800`. Thirty minutes before the wall, SLURM signals the batch
   shell; `slurm/requeue_lib.sh` forwards it to the trainer, which finishes the
   epoch in flight, writes a checkpoint (models, optimizers, schedulers, every RNG
   stream including the data-loader shuffle generator, matcher recency state,
   communication counters), and exits with code 85; the script then
   `scontrol requeue`s its own array task. The requeued task re-enters the same
   script with `--resume` and continues bit-identically. A run needing 100 GPU-hours
   completes across two windows with zero manual action. The mechanism is
   unit-tested end-to-end (`tests/test_trainer.py::TestPreemption`: signal mid-run
   → checkpoint at epoch boundary → resumed run completes with no duplicate or
   missing CSV rows).
2. **Tighter periodic checkpoints.** `--checkpoint_every` dropped from 25 to 10
   epochs, so a hard preemption (`qos=low` + `--requeue`, which SLURM handles by
   itself) loses at most ~1 h of WRN-cohort work.
3. **Recovery is an array expression, not a hunt.** `tools/incomplete.py` mirrors
   each script's array map, recomputes every index's run_id, and checks its metrics
   CSV against the target epoch. Recovery after anything the mechanisms above
   didn't catch — cancelled array, node failure, a requeue that never ran — is:

       IDS=$(python tools/incomplete.py m1)
       [ -n "$IDS" ] && sbatch --array=$IDS slurm/m1_headline_k8.sbatch

   Only the unfinished indices rerun, and each resumes from its checkpoint rather
   than restarting. `--list` prints a per-index status table (run_id + last logged
   epoch). A new test (`tests/test_suite_grids.py`) pins the tool's grids to the
   sbatch array sizes and asserts run_id uniqueness across all 215 runs of the
   suite, so the grids cannot silently drift from the scripts.

One habit this enables: after submitting a batch, `tools/incomplete.py <exp>` is
also the completion check — an empty output means the experiment is done and ready
for `analysis/aggregate.py`.

---

## 2026-07-17 — Task [D-001]: the reduced-communication DML program — design and harness are in; smoke-test, calibrate, then launch R1

**To:** Ioannis (and whoever picks up the cluster work). This entry announces the
whole program in one piece: the experiment design, the harness that implements it,
and the first concrete checklist. Everything referenced below is on this branch.

### 1 · What this is

This repo now carries the DML side of the program sketched in the knowledge-diffusion
repo's `dev-communication/ideas.md` (entry *2026-07-16, "Matched & sparse mutual
learning"*, plus its 2026-07-17 implementation notes): replace dense all-to-all
mutual distillation with a **per-epoch maximum-weight matching** on measured
disagreement, so that each model exchanges logits with exactly one (or k) partners
instead of all K−1. Two things are being bought at once, and it matters to keep them
separate in our heads: a **communication reduction** that is mechanical and exact
(degree 1 versus degree K−1 — the ledger is computed into every CSV row, not
asserted), and a **selective performance claim** — that *who* is matched matters
precisely where the theory says second-order structure decouples from accuracy:
heterogeneous cohorts (the original paper's flagship big+small setting), label
noise, and the diversity/ensemble axis that dense mimicry is known to drain. The
full design — six headline claims H1–H6, experiments R1 + M1–M6 with predictions and
falsification conditions, a deferred topology campaign M7, costs, and a table-by-table
mapping onto the original paper — is in [`experiments.md`](experiments.md). It is
organized at the level of the original paper's §3 on purpose: R1 *is* Table 2, M3
*is* Fig. 2 with a communication axis, M5 *is* the DML_e finding, and M1 is what
Table 2 becomes at K=8 with matched arms.

### 2 · What is implemented (and how it is checked)

The harness is `src/` (the original chxy95 single-file trainer stays untouched at
the repo root as the legacy reference; it is not used). One process trains a whole
cohort: `python -m src.run_experiment --cohort wrn28x10:4,resnet32:4 --arm matched
--match_weight disagreement --seed 1 …`. Models: ResNet-32, MobileNet, WRN-28-10
(param counts pinned to the paper's Table 1 by a unit test). Arms: `indep`, `dml`
(dense), `dmle` (averaged-posterior target), and the `matched` family — random /
MWM-on-disagreement / teachable (m−κu, the KD MWM-D weight) / accuracy-gap weights,
k peeled matchings with weight-proportional KL alphas, degree annealing, static
matching, recency penalty, and graph-restricted matching (ring / random-regular; for
the deferred M7). The exact solver is a bitmask DP (exact to K=16, verified against
brute force), greedy (Preis/Hoepman order) beyond.

Two design decisions are fixed and worth knowing by heart, both inherited from the
KD side's lessons. First, updates are **simultaneous** in every arm — all K forwards
precede any step, targets are detached pre-step logits — and a parity test
(`tests/test_trainer.py`) locks the trainer to a manual reference implementation of
exactly that rule, so a sequential-update or leaky-gradient regression fails CI.
Second, all runs (including baselines) train on 45k and hold out the same 5k
validation split for the matcher and policy-facing metrics; absolute accuracies will
sit a few tenths below the paper's Table 2, the *gains* are what R1 gates on. The
test suite is 75 tests, all passing on CPU; `pytest tests/` needs no GPU and no
dataset download.

### 3 · The checklist (in order; nothing is launched yet)

1. **Environment + data staging** on Wulver: create/point `DML_CONDA_ENV` at a
   torch ≥ 2.0 env, then `python tools/stage_data.py cifar100` on a login node.
2. **Smoke + timing** (one GPU, ~1 h): a short hetero run to verify memory on a
   full A100 and calibrate `epoch_seconds` —
   `python -m src.run_experiment --cohort wrn28x10:4,resnet32:4 --arm matched
   --epochs 3 --output_dir results/smoke --verbose`
   plus the same for `--cohort resnet32:8 --arm dml`. Report the two per-epoch
   times in a reply to this entry; the cost table in experiments.md §6 is ±2× until
   then. Check `nvidia-smi` peak memory in the hetero case — if 4 simultaneous WRN
   graphs don't fit the 40 GB card, we halve the batch and note it.
3. **R1** (`sbatch slurm/r1_pairs.sbatch`, 60 tasks): the replication gate. The
   reply should paste the R1 gains table against the paper's Table 2 (the
   aggregation one-liner is in the repo README). **If the gate fails we stop and
   debug — nothing downstream is interpretable without it.**
4. **M1** (`sbatch slurm/m1_headline_k8.sbatch`), then M2+M5 (cheap, same cell),
   then M3, M4, M6 per the launch order in experiments.md §6.

All scripts carry `--run_tag d001`, checkpoint/resume every 25 epochs, and
`--requeue`; a preempted or timed-out array element is re-`sbatch`-able and resumes
from its last checkpoint (SLURM does not requeue on TIMEOUT — resubmit the index
manually, same as the KD convention).

### 4 · What to report back

A reply citing [D-001] with: (a) the two smoke timings + peak memory, (b) the R1
gains table next to the paper's, (c) anything that smells wrong — especially epoch-0
weirdness in `*_matches.csv` (epoch-0 matchings are computed from random-init
predictions and should look random; if they don't, that's a bug, not a finding).
