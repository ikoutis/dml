#!/bin/bash -l
# =============================================================================
# r1_pairs.sbatch — R1: two-network cohorts (harness validation).
# Design: dev-communication/experiments.md §3.R1. Batch entry: [D-001].
#
# Replicates the DML paper's Table 2 (CIFAR-100, K=2): every cohort
# combination of {ResNet-32, MobileNet, WRN-28-10}, Independent vs DML.
# At K=2 every coupled arm coincides (dense == matched, k=1), so this is
# the calibration gate for the whole suite: our DML-minus-Independent gains
# must reproduce the paper's column (+1.7 to +3.7 pp for the small nets)
# before any matched-arm result is interpreted.
#
# Array map: 6 cohorts x 2 arms {indep, dml} x 5 seeds = 60 tasks.
#   idx = (cohort * 2 + arm) * 5 + (seed - 1)
# Cost/task: resnet32 pairs ~1-2 GPU-h; wrn28x10 pairs ~6-10 GPU-h.
# GPU: full A100 — the WRN-bearing cohorts (indices 30-59) hold two
# simultaneous WRN-28-10 graphs, too big for a 10 GB MIG slice.
#
# Submit (after staging CIFAR-100 once on a login node):
#     python tools/stage_data.py cifar100
#     sbatch slurm/r1_pairs.sbatch
# =============================================================================

#SBATCH --job-name=dml_r1
#SBATCH --output=results/logs/r1_%A_%a.log
#SBATCH --error=results/logs/r1_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-59

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-/project/ikoutis/conda_env/dml-torch}"
set -u
cd "$SLURM_SUBMIT_DIR"
source slurm/requeue_lib.sh

test -d data/cifar-100-python || {
    echo "STAGE ERROR: CIFAR-100 not found under data/." >&2
    echo "Run 'python tools/stage_data.py cifar100' on a LOGIN node first." >&2
    exit 1
}

COHORTS=("resnet32:2" "mobilenet:1,resnet32:1" "mobilenet:2" \
         "wrn28x10:1,resnet32:1" "wrn28x10:2" "wrn28x10:1,mobilenet:1")
ARMS=(indep dml)

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5 + 1))
ARM=${ARMS[$(((i / 5) % 2))]}
COHORT=${COHORTS[$((i / 10))]}

OUT=results/suite/r1_pairs
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== R1 | cohort=$COHORT arm=$ARM seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "$COHORT" --arm "$ARM" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
