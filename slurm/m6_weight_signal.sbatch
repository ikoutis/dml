#!/bin/bash -l
# =============================================================================
# m6_weight_signal.sbatch — M6: what signal should the matcher maximize?
# Design: dev-communication/experiments.md §3.M6. Batch entry: [D-001].
#
# First-order modularity says accuracy-gap pairing should NOT beat random —
# the "who pairs with whom" signal lives in second-order structure
# (disagreement), and only decouples from accuracy under heterogeneity or
# label noise. New arms (regimes where the decoupling exists):
#   mwmt1     teachable weights m_ij - kappa*u_ij, kappa=1 (labeled val)
#   mwmt1k0   kappa=0 (pure teachable mass, no damage term)
#   gap1      |acc_i - acc_j| — the first-order shadow (falsification arm)
# Comparators from M1 (same cells, same seeds): rand1, mwmd1 (label-free).
# Predictions: gap1 ~ rand1 everywhere; mwmt1 ~ mwmd1 >= rand1 in these
# regimes; if gap1 > rand1 robustly, first-order modularity is falsified.
#
# Array map: 2 regimes x 3 arms x 5 seeds = 30 tasks.
#   idx = (regime * 3 + arm) * 5 + (seed - 1)
#   regime 0: wrn28x10:4,resnet32:4 clean (full A100 needed)
#   regime 1: resnet32:8 + 40% label noise
# =============================================================================

#SBATCH --job-name=dml_m6
#SBATCH --output=results/logs/m6_%A_%a.log
#SBATCH --error=results/logs/m6_%A_%a.err
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
#SBATCH --array=0-29

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
ARM_IDX=$(((i / 5) % 3))
REGIME=$((i / 15))

case $REGIME in
    0) COHORT="wrn28x10:4,resnet32:4"; NOISE="0.0" ;;
    1) COHORT="resnet32:8";            NOISE="0.4" ;;
esac
case $ARM_IDX in
    0) ARMARGS="--arm matched --match_weight teachable --kappa 1.0" ;;
    1) ARMARGS="--arm matched --match_weight teachable --kappa 0.0 --arm_label mwmt1k0" ;;
    2) ARMARGS="--arm matched --match_weight accgap" ;;
esac

OUT=results/suite/m6_weight_signal
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M6 | cohort=$COHORT noise=$NOISE arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "$COHORT" $ARMARGS \
    --label_noise_rate "$NOISE" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
