#!/bin/bash -l
# =============================================================================
# m3_cohort_scaling.sbatch — M3: cohort-size scaling (paper §3.5 analogue).
# Design: dev-communication/experiments.md §3.M3. Batch entry: [D-001].
#
# The DML paper's Fig. 2 shows individual accuracy rising with cohort size
# under dense coupling — where per-model communication grows as K-1. The
# question here: does MWM-matched (degree 1, communication CONSTANT in K)
# track dense DML's cohort-size gains? Cohorts resnet32 x K, K in {2,4,12};
# the K=8 points come from M1 (same tag, same seeds), and the K=2 dml
# anchor comes from R1's resnet32:2 cells (identical config and seeds —
# re-running it here would duplicate run_ids and double-count in the
# aggregation, so K=2 only runs mwmd1: the coincidence of the two arms at
# K=2 is the sanity check).
#
# Array map: 25 tasks.
#   0-4   : K=2  mwmd1, seeds 1-5
#   5-14  : K=4  {dml, mwmd1} x seeds 1-5
#   15-24 : K=12 {dml, mwmd1} x seeds 1-5
# Cost/task: ~1 GPU-h (K=2) to ~12 GPU-h (K=12).
# =============================================================================

#SBATCH --job-name=dml_m3
#SBATCH --output=results/logs/m3_%A_%a.log
#SBATCH --error=results/logs/m3_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms+
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-24

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
if [ "$i" -lt 5 ]; then
    K=2
    ARM_IDX=1                       # mwmd1 only (dml K=2 comes from R1)
else
    j=$((i - 5))
    K=$([ $((j / 10)) -eq 0 ] && echo 4 || echo 12)
    ARM_IDX=$(((j / 5) % 2))
fi

case $ARM_IDX in
    0) ARMARGS="--arm dml" ;;
    1) ARMARGS="--arm matched --match_weight disagreement" ;;
esac

OUT=results/suite/m3_cohort_scaling
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M3 | K=$K arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort "resnet32:$K" $ARMARGS \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
