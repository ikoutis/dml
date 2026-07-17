# Deep Mutual Learning — reduced-communication variants (matched mutual learning)

Deep Mutual Learning (DML, Zhang et al., CVPR 2018) trains a cohort of K networks
that all mimic each other: every model, on every batch, distills from all K−1 peers.
This repo extends a DML baseline with **matched mutual learning (MML)**: each epoch,
a **maximum-weight matching** on measured pairwise disagreement decides who teaches
whom, and each model mimics exactly its matched partner(s) — reducing per-model
communication from O(K) to O(1) while targeting the mimicry where it is predicted to
matter (heterogeneous cohorts, label noise, diversity retention).

The research program originates in the companion **knowledge-diffusion** repo
(`dev-communication/ideas.md`, entry 2026-07-16 — matched & sparse mutual learning);
this repo carries the faithful per-batch DML trainer and the full experiment suite.

**Start here:**
- [`dev-communication/experiments.md`](dev-communication/experiments.md) — the full
  experiment design: arms, headline claims H1–H6, experiments R1 + M1–M7,
  predictions, costs, and the mapping onto the original paper's tables.
- [`dev-communication/log.md`](dev-communication/log.md) — the running task/reply
  log (entry [D-001] is the program announcement + launch checklist).

## Repository layout

```
src/                    The cohort harness (new; the deliverable)
  ├── models/           ResNet-32 / MobileNet / WRN-28-10 (paper Table 1 trio)
  ├── data.py           CIFAR-100/10, deterministic 45k/5k split, label noise
  ├── matching.py       Edge weights, exact/greedy MWM, peeling, recency, graphs
  ├── cohort.py         Cohort specs ("wrn28x10:4,resnet32:4")
  ├── mutual_trainer.py Simultaneous-update trainer, all arms, comm accounting,
  │                     checkpoint/resume
  ├── metrics.py        Ensemble, disagreement, error-correlation, CSV streaming
  └── run_experiment.py CLI entry point
slurm/                  SLURM arrays for the suite (R1, M1–M6) — Wulver conventions
analysis/               aggregate.py (CI + Welch/Bonferroni) and plot.py
                        (curves, diversity, the accuracy-vs-bytes Pareto plot)
tests/                  75 CPU-only unit tests (solver vs brute force, trainer
                        parity vs the declared update rule, comm accounting)
tools/stage_data.py     Dataset staging for login nodes
dev-communication/      Design doc + dated task/reply log ([D-00N] IDs)
main.py, trainer.py,    The original unofficial DML implementation (chxy95),
config.py, resnet.py,   kept untouched at the root as the legacy reference —
data_loader.py, utils.py  not used by src/
```

## Install

Local:

```bash
pip install -r requirements.txt      # torch >= 2.0, torchvision, numpy, scipy, ...
pytest tests/                        # no GPU or dataset needed
```

Wulver (once, on a login node — creates `/project/ikoutis/conda_env/dml-torch`
and installs everything; the sbatch scripts activate that path by default,
`DML_CONDA_ENV` overrides it):

```bash
bash tools/setup_env.sh
python tools/stage_data.py cifar100
```

## Running a single experiment

```bash
# Dense DML pair (the paper's Table 2 ResNet-32/ResNet-32 cell):
python -m src.run_experiment --cohort resnet32:2 --arm dml --seed 1 \
    --data_dir data --download --output_dir results/dev --verbose

# MWM-matched mutual learning, K=8 heterogeneous (the headline cell):
python -m src.run_experiment --cohort wrn28x10:4,resnet32:4 --arm matched \
    --match_weight disagreement --seed 1 --output_dir results/dev --verbose

# Reduced-degree dial: k=2 peeled matchings, weight-proportional KL terms:
python -m src.run_experiment --cohort resnet32:8 --arm matched \
    --k_matchings 2 --seed 1 --output_dir results/dev

# CPU smoke test (~2 min):
python -m src.run_experiment --cohort resnet20:4 --arm matched --epochs 2 \
    --batch_size 32 --n_valid 500 --limit_train 512 --device cpu \
    --data_dir data --download --output_dir results/smoke --verbose
```

Each run writes `{run_id}_metrics.csv` (one row/epoch: accuracies, ensemble,
diversity, the communication ledger) and `{run_id}_matches.csv` (who was matched
with whom, at what weight). `--resume` continues from the last checkpoint.

## Cluster (SLURM)

```bash
python tools/stage_data.py cifar100          # once, on a login node
sbatch slurm/r1_pairs.sbatch                 # R1: replication gate (Table 2)
sbatch slurm/m1_headline_k8.sbatch           # M1: matched vs dense at K=8
# then m2_degree_dial, m3_cohort_scaling, m5_target_structure,
#      m4_rotation, m6_weight_signal — see experiments.md §6 for the order
```

All scripts run as SLURM arrays with the 72 h Wulver maximum and survive it:
30 min before the wall the trainer checkpoints and the task requeues itself,
resuming bit-identically (`slurm/requeue_lib.sh`); preempted `qos=low` tasks
resume from their last periodic checkpoint (every 10 epochs). To check
completion or recover from anything else, resubmit exactly the unfinished
array indices:

```bash
python tools/incomplete.py m1 --list                      # per-index status
IDS=$(python tools/incomplete.py m1)                      # e.g. "7,13,40-44"
[ -n "$IDS" ] && sbatch --array=$IDS slurm/m1_headline_k8.sbatch
```

## Analysis

```bash
python analysis/aggregate.py --results_dir results/ --output_dir analysis/output/
python analysis/plot.py --input_dir analysis/output/ \
    --output_dir analysis/figures/ --fmt pdf
```

## Legacy baseline

The original unofficial DML implementation this repo started from (chxy95's
Deep-Mutual-Learning, ResNet-32 on CIFAR-100) remains at the repo root:
`python main.py --model_num 2`. Its reported result: ResNet-32 independent 69.83%,
DML 71.03% top-1. The new harness supersedes it for all experiments.
