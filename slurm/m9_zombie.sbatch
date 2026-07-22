#!/bin/bash -l
# =============================================================================
# m9_zombie.sbatch — M9: controlled epidemiology. Batch entry: [D-015].
#
# [D-014] found that training collapse spreads through the communication
# graph (LeNet: spontaneous deaths -> contagion; quarantine in disconnected
# cliques; wave speed tracks mixing). This campaign turns the accident into
# a dose-response measurement: implant ONE dead model (slot 0: frozen,
# exactly-uniform logits — what a collapsed model converges to) in an
# otherwise-healthy ResNet-32 K=12 cohort and measure the damage field as a
# function of communication structure.
#
# Healthy slots keep bit-identical inits with the M7 d011 no-zombie anchors
# at equal seed, so every comparison is per-model paired. Matched arms use
# RANDOM weights (MWM would glue itself to the zombie and confound exposure).
#
#   0 topo ring        zombie touches 2 fixed neighbours (alpha 1/2 each)
#   1 topo prism       3 fixed neighbours (alpha 1/3)
#   2 topo rregular:3  3 fixed neighbours, short paths (graph_seed=seed)
#   3 topo clusters:4  zombie sealed in clique 0 -> cliques 1-2 are an
#                      INTERNAL no-exposure control in the same run
#   4 matched rand k=1 rotating: full-dose victim (alpha 1), ~1/11 of rounds
#   5 matched rand k=2 rotating, two partners
#   6 dml              dense: everyone, every round, alpha 1/11 (dilution)
#
# Registered predictions ([D-015]): ResNet hosts do NOT die (R0 < 1 —
# damage without transmission); per-victim damage tracks the zombie's
# weight in the victim's KD mix (ring 1/2 > prism/rreg3 1/3 >> dense 1/11);
# damage decays with graph distance; clusters cliques 1-2 match anchors
# exactly. COUNTER-hypothesis, registered with equal standing: KL-to-uniform
# is a confidence penalty (entropy regularizer), so victims may be UNHARMED
# or slightly HELPED on clean data — in which case [D-014]'s contagion
# needed fragile hosts, i.e. the epidemic has a susceptibility threshold.
#
# Array map: 7 arms x 5 seeds = 35 tasks. idx = arm*5 + (seed-1).
# =============================================================================

#SBATCH --job-name=dml_m9
#SBATCH --output=results/logs/m9_%A_%a.log
#SBATCH --error=results/logs/m9_%A_%a.err
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
#SBATCH --array=0-34

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
ARM_IDX=$((i / 5))
SEED=$((i % 5 + 1))

case $ARM_IDX in
    0) ARMARGS="--arm topology --graph ring" ;;
    1) ARMARGS="--arm topology --graph prism" ;;
    2) ARMARGS="--arm topology --graph rregular:3 --graph_seed $SEED" ;;
    3) ARMARGS="--arm topology --graph clusters:4" ;;
    4) ARMARGS="--arm matched --match_weight random --k_matchings 1" ;;
    5) ARMARGS="--arm matched --match_weight random --k_matchings 2" ;;
    6) ARMARGS="--arm dml" ;;
esac

OUT=results/suite/m9_zombie
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d015 --checkpoint_every 10 --resume"

echo "=== M9 | arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 $ARMARGS \
    --zombie_slot 0 \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
