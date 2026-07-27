#!/bin/bash -l
# =============================================================================
# m1b_perclass.sbatch — the per-class-distance matched arm ([D-007] follow-up).
#
# The signal-structure diagnostic (tools/signal_structure.py, results/analysis/)
# found that the per-CLASS accuracy-vector distance carries ~5x the exploitable
# structure of scalar disagreement (MWM-gain-over-random ~+6-8% vs ~+1.4%),
# consistently across regimes and on both coupled and independent cohorts —
# whereas the per-example error field was barely above scalar. This is the one
# multidimensional signal worth an accuracy test.
#
# This runs the matched arm with --match_weight perclass (label mwmpc1) in the
# two SLICE-friendly regimes (resnet32:8 clean and +40% noise); the rand1 and
# mwmd1 anchors already exist in results/suite/m1_headline/ at the same cohort
# and seeds, so aggregate.py compares them directly. Hetero perclass (where
# per-class structure should be largest but needs a full A100) is deferred to a
# separate script pending this result.
#
# Decision: if mwmpc1 beats rand1/mwmd1 by a clear margin (> ~0.4 pp, paired),
# the multidimensional signal matters — a real positive for the who-teaches-whom
# question. If it ties (~+0.2 pp like scalar), selection is flat even at
# per-class granularity — a robust null. Both are publishable.
#
# Array map: 2 regimes x 5 seeds = 10 tasks.  idx = regime*5 + (seed-1)
#   0-4 : resnet32:8 clean         5-9 : resnet32:8 + 40% noise
# Slice pool, ~4-9 GPU-h/run. Submit from the repo root: sbatch slurm/m1b_perclass.sbatch
# =============================================================================

#SBATCH --job-name=dml_pc
#SBATCH --output=results/logs/m1b_%A_%a.log
#SBATCH --error=results/logs/m1b_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-9

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
NOISE=$([ "$((i / 5))" -eq 0 ] && echo 0.0 || echo 0.4)

OUT=results/suite/m1_headline           # sits with the M1 rand1/mwmd1 anchors
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d001 --checkpoint_every 10 --resume"

echo "=== M1b perclass | noise=$NOISE seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:8 --arm matched --match_weight perclass \
    --label_noise_rate "$NOISE" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
