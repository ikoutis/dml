#!/bin/bash -l
# =============================================================================
# m7_topology.sbatch — M7 Tier 1: fixed communication topologies at N=12.
# Design: dev-communication/experiments.md §3.M7. Batch entry: [D-011].
#
# Question: does a FIXED sparse communication graph recover the dense DML
# benefit, and does the SHAPE of the graph matter at equal degree?
# Homogeneous ResNet-32, K=12, CIFAR-100, clean + 40% label noise, 5 seeds.
#
# Seven arms along the degree axis (each distills from its neighbours):
#   0 indep        deg 0   floor
#   1 matched k=1  deg 1   rotating partners (MWM, refreshed each epoch)
#   2 cycle        deg 2   FIXED ring
#   3 matched k=2  deg 2   rotating partners (MWM x2)   <- headline vs arm 2
#   4 prism        deg 3   fixed, structured (circular ladder)
#   5 rregular:3   deg 3   fixed, expander draw (graph_seed = seed)
#   6 dml          deg 11  dense (ceiling)
# Arms 2 and 3 are iso-degree (2): the only difference is fixed ring
# neighbours vs partners that rotate every epoch -> isolates temporal mixing.
#
# Array map: 7 arms x 2 noise x 5 seeds = 70 tasks.
#   idx = noise*35 + arm*5 + (seed-1)
#   0-34  : clean         35-69 : +40% label noise
#
# [D-012] connectivity probe (clean only), appended as indices 70-79:
#   70-74 clusters:2  six DISCONNECTED frozen pairs   (deg 1; vs matched-k1)
#   75-79 clusters:4  three DISCONNECTED 4-cliques    (deg 3; vs prism/rreg3)
# If degree alone drives the benefit, the isolated cliques match their
# connected same-degree counterparts; a deficit = the connectivity premium.
# Launch just these:  sbatch --array=70-79 slurm/m7_topology.sbatch
#
# Full A100 (not a MIG slice): at K=12 the trainer holds all 12 computation
# graphs live simultaneously (~1.5x the K=8 activation memory the slices ran),
# so a 10G slice risks OOM. Cost/task ~6-8 GPU-h.
# =============================================================================

#SBATCH --job-name=dml_m7
#SBATCH --output=results/logs/m7_%A_%a.log
#SBATCH --error=results/logs/m7_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-79

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
if [ "$i" -ge 70 ]; then
    # [D-012] connectivity probe: clean, disconnected cliques
    ARM_IDX=$((i / 5))    # 14/15; must be set (set -u, echoed below)
    NOISE=0.0
    SEED=$((i % 5 + 1))
    if [ "$i" -lt 75 ]; then
        ARMARGS="--arm topology --graph clusters:2"
    else
        ARMARGS="--arm topology --graph clusters:4"
    fi
else
    NOISE_IDX=$((i / 35))
    J=$((i % 35))
    ARM_IDX=$((J / 5))
    SEED=$((J % 5 + 1))

    case $NOISE_IDX in
        0) NOISE=0.0 ;;
        1) NOISE=0.4 ;;
    esac

    case $ARM_IDX in
        0) ARMARGS="--arm indep" ;;
        1) ARMARGS="--arm matched --match_weight disagreement --k_matchings 1" ;;
        2) ARMARGS="--arm topology --graph ring" ;;
        3) ARMARGS="--arm matched --match_weight disagreement --k_matchings 2" ;;
        4) ARMARGS="--arm topology --graph prism" ;;
        5) ARMARGS="--arm topology --graph rregular:3 --graph_seed $SEED" ;;
        6) ARMARGS="--arm dml" ;;
    esac
fi

OUT=results/suite/m7_topology
mkdir -p results/logs "$OUT"
# run_tag d011 (not d001): keeps the K=12 dml/mwmd1 cells from colliding with
# M3's cohort-scaling cells; this is a self-contained campaign.
HARDEN="--run_tag d011 --checkpoint_every 10 --resume"

echo "=== M7 | arm_idx=$ARM_IDX noise=$NOISE seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort resnet32:12 $ARMARGS \
    --label_noise_rate "$NOISE" \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
