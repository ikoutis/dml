#!/bin/bash -l
# =============================================================================
# r4_update_order.sbatch — R4: the original sequential update order. [D-019]
#
# Every suite result uses SIMULTANEOUS updates (all forwards precede all
# steps). The original DML paper's Algorithm 1 is SEQUENTIAL: models step in
# turn and later models distil from earlier models' post-step predictions.
# Reviewers noted the communication conclusions (one all-reduce per update)
# are stated for the simultaneous form only. This array reruns the headline
# cells under --update_order sequential so the paper can report both orders.
#
# Arms, 5 seeds each, paired against comparators already on disk (same
# cohort, seeds, initialisations and batch order, run_tag d001):
#
#   0 resnet32:2  dml-seq    vs r1_pairs     resnet32x2 ... K2_dml
#   1 resnet32:8  dml-seq    vs m1_headline  resnet32x8 ... K8_dml
#   2 resnet32:8  rand1-seq  vs m1_headline  resnet32x8 ... K8_rand1
#
# Sequential costs one extra no-grad forward per model per update (~+15-20%
# wall time); the 48 h limit is ample for ResNet-32 cohorts.
#
# Array map: 3 arms x 5 seeds = 15 tasks. idx = arm*5 + (seed-1).
# =============================================================================

#SBATCH --job-name=dml_r4
#SBATCH --output=results/logs/r4_%A_%a.log
#SBATCH --error=results/logs/r4_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-14

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

SEEDS=5
i=$SLURM_ARRAY_TASK_ID
ARM_IDX=$((i / SEEDS))
SEED=$((i % SEEDS + 1))

case $ARM_IDX in
    0) COHORT="resnet32:2"; ARMARGS="--arm dml" ;;
    1) COHORT="resnet32:8"; ARMARGS="--arm dml" ;;
    2) COHORT="resnet32:8"; ARMARGS="--arm matched --match_weight random" ;;
    *) echo "bad ARM_IDX=$ARM_IDX" >&2; exit 1 ;;
esac

OUT=results/suite/r4_update_order
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== R4 | cohort=$COHORT arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "$COHORT" $ARMARGS --update_order sequential \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
