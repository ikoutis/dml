# requeue_lib.sh — wall-clock self-requeue for the suite's array tasks.
#
# Wulver caps jobs at 72 h. Every suite script therefore:
#   * requests --time=72:00:00 and --signal=B:USR1@1800, so SLURM delivers
#     SIGUSR1 to the batch shell 30 min before the wall;
#   * runs the trainer through run_with_requeue below, which forwards the
#     signal. The trainer (src/mutual_trainer.py, trap_usr1) finishes the
#     epoch in flight, writes a checkpoint, and exits with code 85;
#   * on 85, run_with_requeue calls `scontrol requeue` on THIS array task.
#     The requeued task re-enters the same script and `--resume` continues
#     from the checkpoint — bit-identical state, including RNG streams and
#     the shuffle generator.
# A run longer than one 72 h window therefore completes across as many
# windows as it needs, with no manual action. (Preemption under qos=low and
# other scheduler-initiated requeues — node failure, admin drain — are
# separate: SLURM requeues automatically via --requeue, and the run resumes
# from its last periodic checkpoint — every 10 epochs.)
#
# Manual recovery after anything else (node failure, cancelled array, ...):
#   IDS=$(python tools/incomplete.py <exp>)          # e.g. m1
#   [ -n "$IDS" ] && sbatch --array=$IDS slurm/<exp script>.sbatch
#
# Usage in a script (after `cd "$SLURM_SUBMIT_DIR"`):
#   source slurm/requeue_lib.sh
#   run_with_requeue python -m src.run_experiment ... $HARDEN --verbose

run_with_requeue() {
    local pid rc
    trap 'kill -USR1 "$pid" 2>/dev/null' USR1
    "$@" &
    pid=$!
    set +e
    wait "$pid"; rc=$?
    # A trapped signal interrupts `wait` (rc > 128); wait again until the
    # child has actually exited.
    while [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; do
        wait "$pid"; rc=$?
    done
    set -e
    trap - USR1
    if [ "$rc" -eq 85 ]; then
        echo "=== 72 h wall approaching: checkpointed; requeueing ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(date) ==="
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        exit 0
    fi
    return "$rc"
}
