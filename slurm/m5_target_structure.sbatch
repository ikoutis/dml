#!/bin/bash -l
# =============================================================================
# m5_target_structure.sbatch — M5: target structure (the DML_e lesson).
# Design: dev-communication/experiments.md §3.M5. Batch entry: [D-001].
#
# The DML paper (§3.6) found that distilling from the AVERAGED peer
# posterior (DML_e) is WORSE than averaging separate KLs to individual
# peers — the fluctuations of individual teachers are functional. This is
# the design axiom behind keeping peeled KL terms separate (never averaged)
# in the matched arms. This batch adds the one missing arm in our harness:
#   dmle   dense coupling, single KL to the mean posterior of the others
# Comparators from M1/M2 (same cell resnet32:8, same seeds): dml, mwmd1,
# mwmd2 (separate KLs), mwmd2-unif.
# Prediction: dml > dmle on individual accuracy and diversity retention.
#
# Array map: 5 seeds = 5 tasks. Cost/task ~6-9 GPU-h.
# =============================================================================

#SBATCH --job-name=dml_m5
#SBATCH --output=results/logs/m5_%A_%a.log
#SBATCH --error=results/logs/m5_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms+
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-4

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

SEED=$((SLURM_ARRAY_TASK_ID + 1))
OUT=results/suite/m5_target_structure
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M5 | dmle seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:8 --arm dml --target ensemble \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
