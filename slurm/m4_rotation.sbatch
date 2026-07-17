#!/bin/bash -l
# =============================================================================
# m4_rotation.sbatch — M4: rotation & persistence ablation.
# Design: dev-communication/experiments.md §3.M4. Batch entry: [D-001].
#
# Does the matching need to MOVE? Persistent partners are predicted to
# memorize each other's errors (per-edge rho climbs, diversity drains).
# Arms (heterogeneous cohort wrn28x10:4,resnet32:4 — where matching bites):
#   mwmd1-static   match ONCE at epoch 0, freeze all 200 epochs
#   mwmd1-rec0.5   re-match per epoch + recency penalty lambda=0.5
#   mwmd1-rec1     lambda=1.0
# The plain per-epoch mwmd1 arm comes from M1 (same cell, same seeds).
# Predicted ranking: recency >= plain re-match >> static; static pairs
# show a pairwise-rho climb between frozen partners (matches.csv +
# per-model preds give the per-edge readout).
#
# Array map: 3 arms x 5 seeds = 15 tasks. Cost/task ~15-25 GPU-h (WRN).
# Needs a full A100 (4 simultaneous WRN graphs).
# =============================================================================

#SBATCH --job-name=dml_m4
#SBATCH --output=results/logs/m4_%A_%a.log
#SBATCH --error=results/logs/m4_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-14

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-$HOME/envs/torch-cuda}"
set -u
cd "$SLURM_SUBMIT_DIR"
source slurm/requeue_lib.sh

test -d data/cifar-100-python || {
    echo "STAGE ERROR: CIFAR-100 not found under data/." >&2
    echo "Run 'python tools/stage_data.py cifar100' on a LOGIN node first." >&2
    exit 1
}

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5 + 1))
ARM_IDX=$((i / 5))

case $ARM_IDX in
    0) ARMARGS="--arm matched --match_weight disagreement --rematch_every_epochs 0" ;;
    1) ARMARGS="--arm matched --match_weight disagreement --recency_lambda 0.5" ;;
    2) ARMARGS="--arm matched --match_weight disagreement --recency_lambda 1.0" ;;
esac

OUT=results/suite/m4_rotation
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M4 | arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "wrn28x10:4,resnet32:4" $ARMARGS \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
