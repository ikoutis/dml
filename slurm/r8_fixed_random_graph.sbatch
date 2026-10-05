#!/bin/bash -l
# =============================================================================
# r8_fixed_random_graph.sbatch - R8: fixed random degree-2 graph under
# localized supervision. Batch entry: [D-023].
#
# R7 compared a FIXED ring with ROTATING random matching-2, so its six-point
# gap mixes graph structure with rotation. This adds the missing arm: random
# matching-2 drawn once and frozen (--rematch_every_epochs 0), i.e. a fixed
# random 2-regular graph. With the same seed it is exactly the epoch-0 graph
# of the rotating R7 run, so
#   fixed random  vs ring            -> graph structure at fixed edges
#   fixed random  vs rotating random -> rotation at the same initial graph
# Labeled slots 0,1,2 as in R7. Seeds 1-3 pair with R7 (run_tag d022);
# seeds 4-6 add fixed graphs for the by-distance comparison. Seed 3's graph
# leaves 6 of the 9 unlabeled models on a cycle with no labeled slot.
#
# A fixed random 2-regular graph is a union of cycles; a cycle without a
# labeled slot never receives supervision. analysis/r7_partial.py reads each
# run's matching from its _matches.csv and reports accuracy by true graph
# distance to the labeled set (unreachable models separately).
#
# Department QOS (6 A100 slices): all six run at once, ~7.5 h.
# =============================================================================

#SBATCH --job-name=dml_r8
#SBATCH --output=results/logs/r8_%A_%a.log
#SBATCH --error=results/logs/r8_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_20g:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-5

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

SEED=$((SLURM_ARRAY_TASK_ID + 1))
OUT=results/suite/r7_partial_supervision
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d023 --checkpoint_every 2 --resume"

echo "=== R8 | seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 --arm matched --match_weight random \
    --k_matchings 2 --rematch_every_epochs 0 --labeled_slots 0,1,2 \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
