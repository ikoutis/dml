#!/bin/bash -l
# =============================================================================
# m1_headline_k8.sbatch — M1: matched vs dense at K=8 (the headline 4x3).
# Design: dev-communication/experiments.md §3.M1. Batch entry: [D-001].
#
# Arms x regimes:
#   arms   : indep | dml (dense) | rand1 (random matching, k=1)
#            | mwmd1 (MWM on disagreement, k=1)
#   regimes: resnet32:8 clean | wrn28x10:4,resnet32:4 clean (heterogeneous)
#            | resnet32:8 + 40% symmetric label noise
#
# Selective predictions (ideas origin: knowledge-diffusion ideas.md
# 2026-07-16): matched arms match dense DML's accuracy everywhere at
# 1/(K-1) = 1/7 of the communication; MWM > random ONLY in the
# heterogeneous and noisy regimes; matched arms retain more diversity
# (larger ensemble-minus-avg gap, lower pairwise rho) late in training.
#
# Array map: 3 regimes x 4 arms x 5 seeds = 60 tasks.
#   idx = (regime * 4 + arm) * 5 + (seed - 1)
# Cost/task: resnet32:8 ~6-9 GPU-h; hetero (4 WRN + 4 R32) ~15-25 GPU-h.
# The hetero cells need a FULL A100 (4 simultaneous WRN-28-10 graphs);
# adjust --gres to the site's full-GPU name if a100_10g is a MIG slice.
# =============================================================================

#SBATCH --job-name=dml_m1
#SBATCH --output=results/logs/m1_%A_%a.log
#SBATCH --error=results/logs/m1_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms+
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

i=$SLURM_ARRAY_TASK_ID
SEED=$((i % 5 + 1))
ARM_IDX=$(((i / 5) % 4))
REGIME=$((i / 20))

case $REGIME in
    0) COHORT="resnet32:8";              NOISE="0.0" ;;
    1) COHORT="wrn28x10:4,resnet32:4";   NOISE="0.0" ;;
    2) COHORT="resnet32:8";              NOISE="0.4" ;;
esac
case $ARM_IDX in
    0) ARMARGS="--arm indep" ;;
    1) ARMARGS="--arm dml" ;;
    2) ARMARGS="--arm matched --match_weight random" ;;
    3) ARMARGS="--arm matched --match_weight disagreement" ;;
esac

OUT=results/suite/m1_headline
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M1 | cohort=$COHORT noise=$NOISE arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "$COHORT" $ARMARGS \
    --label_noise_rate "$NOISE" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
