#!/bin/bash -l
# =============================================================================
# r7_partial_supervision.sbatch - R7: localized supervision. Batch entry:
# [D-022].
#
# Every suite run trains all models on the same labeled data, the regime
# least favorable to topology: there is nothing local for a graph to carry.
# Here only slots 0,1,2 of a K=12 ResNet-32 cohort receive the labeled loss;
# the other nine learn purely from their teachers. On the ring (i ~ i+1) the
# unlabeled models sit 1 to 5 hops from the nearest labeled one, so if
# knowledge has to travel through the graph, accuracy should fall with hop
# distance on the ring and not under rotating random matchings.
#
#   arm 0  ring (fixed, degree 2)
#   arm 1  random matching-2, redrawn every epoch (degree 2, mixes fast)
#   arm 2  dense (degree 11)
#
# Array map: 3 arms x 3 seeds = 9 tasks, idx = arm*3 + (seed-1). The ring vs
# random comparison (tasks 0-5) is the one that matters: run those on the
# department QOS (6 slices) and dense on qos=low, e.g.
#   sbatch --array=0-5 --account=dept_dms --qos=high_dept_dms \
#          --gres=gpu:a100_20g:1 slurm/r7_partial_supervision.sbatch
#   sbatch --array=6-8 slurm/r7_partial_supervision.sbatch
# ~8 h per run; checkpoint every 2 epochs.
# =============================================================================

#SBATCH --job-name=dml_r7
#SBATCH --output=results/logs/r7_%A_%a.log
#SBATCH --error=results/logs/r7_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-8

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
    exit 1
}

i=$SLURM_ARRAY_TASK_ID
ARM_IDX=$((i / 3))
SEED=$((i % 3 + 1))
case $ARM_IDX in
    0) ARMARGS="--arm topology --graph ring" ;;
    1) ARMARGS="--arm matched --match_weight random --k_matchings 2" ;;
    2) ARMARGS="--arm dml" ;;
    *) echo "bad task $i" >&2; exit 1 ;;
esac

OUT=results/suite/r7_partial_supervision
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d022 --checkpoint_every 2 --resume"

echo "=== R7 | task=$i | $ARMARGS | seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 $ARMARGS --labeled_slots 0,1,2 \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
