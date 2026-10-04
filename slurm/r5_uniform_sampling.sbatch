#!/bin/bash -l
# =============================================================================
# r5_uniform_sampling.sbatch - R5: uniform teacher sampling and the degree
# grid at K=12. Batch entry: [D-020].
#
# Reviewers noted that the sampled-peers proposition assumes a uniformly
# random d-subset drawn at the current parameters, while the suite uses
# peeled perfect matchings (not uniform d-subsets for d > 1) redrawn once
# per epoch, and that only degrees 1, 2, 3 and 11 were run at K=12.
# This array separates the two gaps and fills in the degree grid:
#
#   arms 0-6   unif{1,2,3,4,6,8,10}-step  uniform d-subset redrawn before
#              EVERY update: exactly the proposition's setting
#   arms 7-9   unif{1,2,3}                uniform d-subset, redrawn once per
#              epoch: isolates staleness from non-uniformity
#   arms 10-12 rand{1,2,3}                peeled random matchings, redrawn
#              once per epoch: the suite's construction, with probes
#
# Every run logs the gradient probe (--grad_probe_every 50, model 0): at
# ~7 points per epoch it computes all 11 per-peer KD gradients and records
# the sampled-vs-dense gradient error, its cosine, the proposition's
# predicted variance, and the updates since the last redraw. The probe does
# not change training. Comparators on disk at K=12 (m7_topology, run_tag
# d011): dml (= degree 11), indep, mwmd1, mwmd2.
#
# Array map: 13 arms x 5 seeds = 65 tasks. idx = arm*5 + (seed-1).
# Cost: ~9 A100-hours per run, ~600 A100-hours in total.
# =============================================================================

#SBATCH --job-name=dml_r5
#SBATCH --output=results/logs/r5_%A_%a.log
#SBATCH --error=results/logs/r5_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-64

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

STEP_DEGREES=(1 2 3 4 6 8 10)
if [ "$ARM_IDX" -lt 7 ]; then
    ARMARGS="--arm sampled --resample step --k_matchings ${STEP_DEGREES[$ARM_IDX]}"
elif [ "$ARM_IDX" -lt 10 ]; then
    ARMARGS="--arm sampled --resample epoch --k_matchings $((ARM_IDX - 6))"
elif [ "$ARM_IDX" -lt 13 ]; then
    ARMARGS="--arm matched --match_weight random --k_matchings $((ARM_IDX - 9))"
else
    echo "bad ARM_IDX=$ARM_IDX" >&2; exit 1
fi

OUT=results/suite/r5_uniform_sampling
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d020 --checkpoint_every 10 --resume"

echo "=== R5 | arm_idx=$ARM_IDX seed=$SEED | $ARMARGS | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 $ARMARGS \
    --grad_probe_every 50 --grad_probe_model 0 \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
