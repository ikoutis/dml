#!/bin/bash -l
# =============================================================================
# r5b_fast.sbatch - R5b: the deadline-trimmed R5. Batch entry: [D-020b].
#
# R5 (r5_uniform_sampling.sbatch, 65 x 200-epoch runs) would take days. This
# keeps what the reviewers' questions need:
#
#   tasks 0-8   200-epoch uniform per-update sampling, d in {2,4,8} x seeds
#               1-3 (idx = d_idx*3 + seed-1). With R5 tasks 0-4 (d=1, five
#               seeds, already running) and the on-disk d=11 dense runs, these
#               fill in the degree curve at K=12. Checkpoint every 2 epochs so
#               a preemption on qos=low costs minutes, not hours.
#               -> results/suite/r5_uniform_sampling (same run_ids as R5)
#
#   tasks 9-21  15-epoch gradient-probe runs, seed 1, one per R5 arm:
#               uniform per-update d in {1,2,3,4,6,8,10}, uniform per-epoch
#               d in {1,2,3}, peeled random matchings d in {1,2,3}. They log
#               the sampled-vs-dense gradient discrepancy through the epoch;
#               final accuracy is not needed for that question.
#               -> results/suite/r5_probe (own run_tag, never collides)
# =============================================================================

#SBATCH --job-name=dml_r5b
#SBATCH --output=results/logs/r5b_%A_%a.log
#SBATCH --error=results/logs/r5b_%A_%a.err
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
#SBATCH --array=0-21

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
if [ "$i" -lt 9 ]; then
    DEGREES=(2 4 8)
    D=${DEGREES[$((i / 3))]}
    SEED=$((i % 3 + 1))
    ARMARGS="--arm sampled --resample step --k_matchings $D"
    EPOCHS=200
    OUT=results/suite/r5_uniform_sampling
    HARDEN="--run_tag d020 --checkpoint_every 2 --resume"
else
    j=$((i - 9))
    STEP_DEGREES=(1 2 3 4 6 8 10)
    if [ "$j" -lt 7 ]; then
        ARMARGS="--arm sampled --resample step --k_matchings ${STEP_DEGREES[$j]}"
    elif [ "$j" -lt 10 ]; then
        ARMARGS="--arm sampled --resample epoch --k_matchings $((j - 6))"
    elif [ "$j" -lt 13 ]; then
        ARMARGS="--arm matched --match_weight random --k_matchings $((j - 9))"
    else
        echo "bad task $i" >&2; exit 1
    fi
    SEED=1
    EPOCHS=15
    OUT=results/suite/r5_probe
    HARDEN="--run_tag d020p --checkpoint_every 5 --resume"
fi
mkdir -p results/logs "$OUT"

echo "=== R5b | task=$i | $ARMARGS | seed=$SEED epochs=$EPOCHS | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 $ARMARGS --epochs "$EPOCHS" \
    --grad_probe_every 50 --grad_probe_model 0 \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
