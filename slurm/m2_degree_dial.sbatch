#!/bin/bash -l
# =============================================================================
# m2_degree_dial.sbatch — M2: peeled k-matchings (the degree dial).
# Design: dev-communication/experiments.md §3.M2. Batch entry: [D-001].
#
# k interpolates matched-DML (k=1) -> dense DML (k=K-1). New arms here:
#   mwmd2       k=2 peeled MWM, weight-proportional KL alphas
#   mwmd3       k=3
#   anneal      degree annealing 3 -> 2 -> 1 at the LR drops (0/60/120)
#   mwmd2-unif  k=2 with UNIFORM alphas (ablates design point (i) of
#               Idea 2c: weight-proportional vs equal teacher weighting)
# The k=1 (mwmd1) and dense (dml) anchors come from M1 — same cell, same
# seeds, comparable by run_tag. Cohort: resnet32:8 clean.
#
# Array map: 4 arms x 5 seeds = 20 tasks. Cost/task ~6-9 GPU-h.
# =============================================================================

#SBATCH --job-name=dml_m2
#SBATCH --output=results/logs/m2_%A_%a.log
#SBATCH --error=results/logs/m2_%A_%a.err
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
#SBATCH --array=0-19

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-$HOME/envs/dml-torch}"
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
    0) ARMARGS="--arm matched --match_weight disagreement --k_matchings 2" ;;
    1) ARMARGS="--arm matched --match_weight disagreement --k_matchings 3" ;;
    2) ARMARGS="--arm matched --match_weight disagreement --k_anneal 0:3,60:2,120:1" ;;
    3) ARMARGS="--arm matched --match_weight disagreement --k_matchings 2 --peel_weighting uniform" ;;
esac

OUT=results/suite/m2_degree_dial
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M2 | arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:8 $ARMARGS \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
