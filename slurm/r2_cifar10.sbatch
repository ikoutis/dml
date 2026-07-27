#!/bin/bash -l
# =============================================================================
# r2_cifar10.sbatch — R2: dataset-transfer grid for the paper. Entry: [D-017].
#
# Reviewers cannot accept an empirical law from one dataset. R2 transfers the
# paper's two load-bearing claims to CIFAR-10 at the headline cell
# (resnet32:8, clean + 40% noise, 5 seeds):
#   1. the DEGREE law     — indep vs rand1 (one random partner) vs dense
#   2. the SELECTION null — mwmd1 (MWM on disagreement) vs rand1
#
# Array map: 4 arms x 2 noise x 5 seeds = 40 tasks.
#   idx = noise*20 + arm*5 + (seed-1)
#   0-19 : clean          20-39 : +40% label noise
#   arms: 0 indep   1 dml   2 rand1   3 mwmd1
#
# Stage the data on a LOGIN node first:  python tools/stage_data.py cifar10
# =============================================================================

#SBATCH --job-name=dml_r2
#SBATCH --output=results/logs/r2_%A_%a.log
#SBATCH --error=results/logs/r2_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=36:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-39

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-/project/ikoutis/conda_env/dml-torch}"
set -u
cd "$SLURM_SUBMIT_DIR"
source slurm/requeue_lib.sh

test -d data/cifar-10-batches-py || {
    echo "STAGE ERROR: CIFAR-10 not found under data/." >&2
    echo "Run 'python tools/stage_data.py cifar10' on a LOGIN node first." >&2
    exit 1
}

i=$SLURM_ARRAY_TASK_ID
NOISE_IDX=$((i / 20))
J=$((i % 20))
ARM_IDX=$((J / 5))
SEED=$((J % 5 + 1))

case $NOISE_IDX in
    0) NOISE=0.0 ;;
    1) NOISE=0.4 ;;
esac

case $ARM_IDX in
    0) ARMARGS="--arm indep" ;;
    1) ARMARGS="--arm dml" ;;
    2) ARMARGS="--arm matched --match_weight random" ;;
    3) ARMARGS="--arm matched --match_weight disagreement" ;;
esac

OUT=results/suite/r2_cifar10
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d017 --checkpoint_every 10 --resume"

echo "=== R2 | arm_idx=$ARM_IDX noise=$NOISE seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar10 --data_dir data \
    --cohort resnet32:8 $ARMARGS \
    --label_noise_rate "$NOISE" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
