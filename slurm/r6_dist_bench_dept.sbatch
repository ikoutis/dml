#!/bin/bash -l
# =============================================================================
# r6_dist_bench_dept.sbatch - R6 on the department QOS.
# Same tasks as r6_dist_bench.sbatch (read its header); Batch entry: [D-021].
#
# Reviewers noted that every communication result is accounted (bytes counted
# or modeled), never measured. This array runs the K=12 ResNet-32 cohort with
# ONE MODEL PER GPU across nodes (src/dist_trainer.py), exchanging posteriors
# over NCCL, and logs per-update compute time, communication time (including
# waits for slower peers) and test accuracy every epoch.
#
#   task 0      indep          200 epochs   compute-only baseline
#   task 1      dense          200 epochs   exact DML via one all-reduce/update
#   task 2      p2p degree 1   200 epochs   random matching, redrawn per epoch
#   tasks 3-10  straggler sweep, 3 epochs each: rank 5 sleeps S ms per update
#               before exchanging; S in {0, 5, 15, 50}
#                 3-6  dense  S = 0, 5, 15, 50
#                 7-10 p2p1   S = 0, 5, 15, 50
#   tasks 11-12 dense / p2p1 with InfiniBand disabled (NCCL over TCP), 3 epochs
#
# Tasks 0-2 give wall-clock time to target accuracy; 3-10 test whether a slow
# member stalls the whole cohort (dense: global collective) or only its
# partner (p2p); 11-12 give a slower-network condition.
#
# Layout: 3 nodes x 4 GPUs = 12 ranks. If GPU nodes on the partition hold a
# different number of GPUs, override on the command line, e.g. for 2-GPU
# nodes:  sbatch --nodes=6 --ntasks-per-node=2 --gres=gpu:a100:2 <this file>
# Runtime: the 200-epoch tasks take roughly an hour; the rest a few minutes.
# Smaller cohort when 3 free nodes are hard to get (K=8 on 2 nodes):
#   sbatch --nodes=2 --export=ALL,COHORT=resnet32:8 <this file>
# The straggler stays on rank 5, which exists for any K >= 6.
#
# Department QOS variant: at most 6 a100_20g MIG slices, so K=6, one slice
# per task (CUDA exposes one MIG instance per process, which the trainer
# then addresses as cuda:0). Slices may sit on different nodes. Only the
# short tasks (3-12) are enabled; with all 6 slices per task they run one
# at a time, a few minutes each. Expected p2p straggler wait: S/5 per rank.
# =============================================================================

#SBATCH --job-name=dml_r6d
#SBATCH --output=results/logs/r6d_%A_%a.log
#SBATCH --error=results/logs/r6d_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=high_dept_dms
#SBATCH --account=dept_dms
#SBATCH --ntasks=6
#SBATCH --gpus-per-task=a100_20g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --array=3-12

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-/project/ikoutis/conda_env/dml-torch}"
set -u
cd "$SLURM_SUBMIT_DIR"

test -d data/cifar-100-python || {
    echo "STAGE ERROR: CIFAR-100 not found under data/." >&2
    exit 1
}

i=$SLURM_ARRAY_TASK_ID
EPOCHS=3
STRAG=""
case $i in
    0) MODE="--mode indep"; EPOCHS=200 ;;
    1) MODE="--mode dense"; EPOCHS=200 ;;
    2) MODE="--mode p2p --degree 1"; EPOCHS=200 ;;
    3|4|5|6|7|8|9|10)
        DELAYS=(0 5 15 50)
        S=${DELAYS[$(((i - 3) % 4))]}
        if [ "$i" -le 6 ]; then MODE="--mode dense"; else MODE="--mode p2p --degree 1"; fi
        if [ "$S" -gt 0 ]; then STRAG="--straggler_rank 5 --straggler_ms $S"; fi ;;
    11) MODE="--mode dense"; export NCCL_IB_DISABLE=1 ;;
    12) MODE="--mode p2p --degree 1"; export NCCL_IB_DISABLE=1 ;;
    *) echo "bad task $i" >&2; exit 1 ;;
esac
# Short runs get their own tag so they never overwrite the 200-epoch CSVs.
TAG="d021"
[ "$EPOCHS" -lt 200 ] && TAG="d021-short"
[ "${NCCL_IB_DISABLE:-0}" = "1" ] && TAG="d021-tcp"

# Two ranks can land on MIG slices of the same physical A100. MIG instances
# support neither CUDA P2P nor IPC, so NCCL's intra-node paths fail at init
# ("invalid device ordinal"); route same-node traffic through the network
# transport instead. NCCL_DEBUG=WARN surfaces any remaining transport error.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + i))
OUT=results/suite/r6_dist_bench
mkdir -p results/logs "$OUT"

echo "=== R6 | task=$i | $MODE $STRAG | epochs=$EPOCHS | nodes=$SLURM_JOB_NODELIST | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
# shellcheck disable=SC2086
srun --kill-on-bad-exit=1 python -m src.dist_trainer \
    --cohort "${COHORT:-resnet32:6}" --dataset cifar100 --data_dir data \
    $MODE $STRAG --epochs "$EPOCHS" --seed 1 --num_workers 3 \
    --run_tag "$TAG" --output_dir "$OUT"
echo "=== done $(date) ==="
