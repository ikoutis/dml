#!/bin/bash -l
# =============================================================================
# m7l_lenet.sbatch — M7-L: the weak-learner probe. Batch entry: [D-013].
#
# Question: does the communication law (degree ~ everything) survive at LeNet
# capacity, or does collective structure start to bind when learners are weak?
# LeNet ~0.09M params on CIFAR-100 is architecturally what 40% label noise
# makes ResNets behaviorally: individually weak, high-diversity, with a large
# ensemble-individual gap for coupling to convert. If topology/connectivity/
# mixing matter anywhere, this regime amplifies them.
#
# Cohort lenet:12, CIFAR-100 CLEAN, 5 seeds. The full ladder plus both
# disconnected-clusters cells in one array:
#   0 indep         deg 0    floor
#   1 matched k=1   deg 1    rotating
#   2 cycle/ring    deg 2    fixed, slow mixer      <- ring-dent replication
#   3 matched k=2   deg 2    rotating
#   4 prism         deg 3    fixed, structured
#   5 rregular:3    deg 3    fixed, expander draw (graph_seed = seed)
#   6 clusters:2    deg 1    six DISCONNECTED pairs   [D-012]
#   7 clusters:4    deg 3    three DISCONNECTED 4-cliques
#   8 dml           deg 11   dense (ceiling)
#
# Registered predictions ([D-013]): conversion story => indep->dense gain
# >= 2x the ResNet-32 gain, ring dent and clusters deficits scale up;
# capacity-cap story => compressed ladder, coupled arms bunched. Kill
# criterion for the structure program: same-degree spreads < 0.5 pp here AND
# in the m7 noise tier AND in the clusters probe => "degree is everything"
# is capacity- and connectivity-invariant; fold into paper 1 and stop.
#
# Array map: 9 arms x 5 seeds = 45 tasks. idx = arm*5 + (seed-1).
# LeNet K=12 is tiny (~1.1M params total): MIG slice, short wall.
# =============================================================================

#SBATCH --job-name=dml_m7l
#SBATCH --output=results/logs/m7l_%A_%a.log
#SBATCH --error=results/logs/m7l_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-44

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
    0) ARMARGS="--arm indep" ;;
    1) ARMARGS="--arm matched --match_weight disagreement --k_matchings 1" ;;
    2) ARMARGS="--arm topology --graph ring" ;;
    3) ARMARGS="--arm matched --match_weight disagreement --k_matchings 2" ;;
    4) ARMARGS="--arm topology --graph prism" ;;
    5) ARMARGS="--arm topology --graph rregular:3 --graph_seed $SEED" ;;
    6) ARMARGS="--arm topology --graph clusters:2" ;;
    7) ARMARGS="--arm topology --graph clusters:4" ;;
    8) ARMARGS="--arm dml" ;;
esac

OUT=results/suite/m7l_lenet
mkdir -p results/logs "$OUT"
HARDEN="--run_tag d013 --checkpoint_every 10 --resume"

echo "=== M7-L | arm_idx=$ARM_IDX seed=$SEED | job=$SLURM_JOB_ID | $(date) ==="
# shellcheck disable=SC2086
run_with_requeue python -m src.run_experiment \
    --dataset cifar100 --data_dir data \
    --cohort lenet:12 $ARMARGS \
    --seed "$SEED" --output_dir "$OUT" \
    $HARDEN --verbose
echo "=== done $(date) ==="
