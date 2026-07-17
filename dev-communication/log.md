# Communication log

Reverse-chronological task/reply log for the DML (matched mutual learning) project —
**newest entry at the top**. Each entry is headed `## YYYY-MM-DD [HH:MM TZ] — <kind>: …`,
where `<kind>` is `Task`, `Note`, or a `<name>` reply. See [`README.md`](README.md) for
the format and the `[D-00N]` ID convention. To add an entry or reply, insert a new dated
section at the top; for a reply, cite the entry you are answering.

<!-- Add your next entry or reply here, above the older ones. -->

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
