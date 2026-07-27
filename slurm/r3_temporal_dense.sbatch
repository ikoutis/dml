#!/bin/bash -l
# =============================================================================
# r3_temporal_dense.sbatch — R3: the byte-matched temporally sparse dense
# baseline. Batch entry: [D-018].
#
# The paper's degree law rests on a comparison that is not yet fair. Exact
# dense DML does not need pairwise messages: the uniform dense objective
# depends on the peers only through their MEAN posterior, so a ring all-reduce
# computes it in 2(K-1)/K = 11/6 posterior streams per node at K=12. Degree-1
# point-to-point costs one stream per update. Dense therefore fits inside the
# degree-1 budget by distilling on 6 of every 11 updates:
#
#       (6/11) x (11/6) = 1 stream per update, exactly.
#
# The schedule selects updates 1, 3, 5, 7, 9, 11 of each 11-block (the rule
# (t*6) % 11 < 6). With 45k training images at batch 128 there are 352
# batches per epoch and 352 = 32 x 11, so the duty cycle is exactly 6/11 with
# no partial block; `comm_duty` in the CSV records the realized fraction.
#
# Two arms, both dense-via-aggregate (--target ensemble: what all-reduce
# actually computes, and gradient-identical to per-peer KLs):
#
#   0 tdense       6/11 updates, KD weight 1     — strict equal-byte baseline
#   1 tdense-dose  6/11 updates, KD weight 11/6  — same bytes AND the same
#                  time-integrated distillation dose, (6/11)x(11/6) = 1, so
#                  it separates "fewer KD updates" from "weaker total KD"
#
# Comparators already on disk at run_tag d011 (results/suite/m7_topology),
# same cohort, seeds, initialisations and batch order: mwmd1 (degree 1, every
# update), dml (dense, every update), indep.
#
# Registered predictions ([D-018]). We genuinely do not know the sign, and
# every outcome is publishable:
#   - tdense >= mwmd1  => one-peer sampling is not uniquely efficient;
#                         temporal subsampling of the exact mean does as well,
#                         and the paper becomes a comparison of two stochastic
#                         implementations of the same dense objective.
#   - tdense <  mwmd1  => at equal bytes, frequent access to one fresh peer
#                         beats less frequent access to the exact cohort mean.
#   - dose closes a gap that tdense does not => the effect is KD dose/count,
#                         not teacher sampling.
#
# Array map: 2 arms x 5 seeds = 10 tasks. idx = arm*5 + (seed-1).
# Extend to 10 seeds (--array=0-19 with SEEDS=10 below) if the paired
# difference against mwmd1 has a CI crossing about +/-0.3 pp.
# =============================================================================

#SBATCH --job-name=dml_r3
#SBATCH --output=results/logs/r3_%A_%a.log
#SBATCH --error=results/logs/r3_%A_%a.err
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

SEEDS=5
i=$SLURM_ARRAY_TASK_ID
ARM_IDX=$((i / SEEDS))
SEED=$((i % SEEDS + 1))

# 11/6 as a decimal for the dose-matched arm; the trainer only ever multiplies
# the KD term by it, so the 1e-9 truncation is far below any measurable effect.
DOSE=1.833333333

case $ARM_IDX in
    0) ARMARGS="--kd_scale 1.0" ;;
    1) ARMARGS="--kd_scale $DOSE" ;;
    *) echo "bad ARM_IDX=$ARM_IDX" >&2; exit 1 ;;
esac

OUT=results/suite/r3_temporal_dense
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d018 --checkpoint_every 10 --resume"

echo "=== R3 | arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 \
    --arm dml --target ensemble \
    --comm_on 6 --comm_block 11 --comm_accounting allreduce \
    $ARMARGS \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
