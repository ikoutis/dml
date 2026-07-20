#!/bin/bash -l
# =============================================================================
# signal_structure.sbatch — the "measure before you run" diagnostic ([D-007]).
#
# For each M1 regime (clean / hetero / noise) it finds a COMPLETED mwmd1
# checkpoint and runs tools/signal_structure.py, which rebuilds the cohort's
# validation predictions and reports, for every candidate edge-weight signal
# (scalar disagreement, per-class distance, per-example error field, teachable,
# accgap), its cross-pair coefficient of variation and the max-weight
# matching's realized weight gain over a random matching. Decision rule
# ([D-007]): if per-class / error-field structure is ~ scalar in every regime,
# the MWM null is robust to signal granularity; if it is materially larger in
# hetero/noise, launch a matched arm there with --match_weight perclass
# (or errorfield).
#
# Inference only (no training): runs on a 10 GB MIG slice so it never competes
# with the full-A100 M1 runs. Output is written to results/analysis/ AND the
# SLURM .log; commit results/analysis/ afterwards to share the numbers.
#
# Submit from the repo root:  sbatch slurm/signal_structure.sbatch
# =============================================================================

#SBATCH --job-name=dml_sig
#SBATCH --output=results/logs/sig_%j.log
#SBATCH --error=results/logs/sig_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=01:00:00

set -euo pipefail
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${DML_CONDA_ENV:-/project/ikoutis/conda_env/dml-torch}"
set -u
cd "$SLURM_SUBMIT_DIR"

CK=results/suite/m1_headline
OUT=results/analysis
mkdir -p "$OUT" results/logs

# Pick the first checkpoint in a glob whose run reached epoch 199 (a COMPLETED
# run — we want the final-training signal structure, not a mid-run snapshot).
pick_complete() {
    local f m last
    for f in $1; do
        [ -f "$f" ] || continue
        m="${f%_ckpt.pt}_metrics.csv"
        [ -f "$m" ] || continue
        last=$(tail -n 1 "$m" | cut -d, -f"$(head -1 "$m" | tr ',' '\n' | grep -nx epoch | cut -d: -f1)")
        if [ "$last" = "199" ]; then echo "$f"; return; fi
    done
}

run_regime() {
    local name="$1" cohort="$2" glob="$3"
    echo "=================================================================="
    echo "REGIME: $name | cohort=$cohort | $(date)"
    local ckpt; ckpt=$(pick_complete "$glob" || true)
    if [ -z "${ckpt:-}" ]; then
        echo "  NO completed (epoch-199) checkpoint found — skipping."
        echo "  (glob: $glob)"; return
    fi
    echo "  checkpoint: $ckpt"
    # Don't let one regime's failure abort the whole job (pipefail + set -e).
    set +e
    python tools/signal_structure.py "$ckpt" --cohort "$cohort" \
        --dataset cifar100 --data_dir data --device cuda \
        > "$OUT/signal_structure_${name}.txt" 2>&1
    set -e
    cat "$OUT/signal_structure_${name}.txt"
}

# Any completed cohort of the right architecture works for measuring signal
# structure; try arms in priority order (indep shows the MOST structure —
# an upper bound before coupling homogenizes the cohort — then the coupled
# arms) so a missing/cleaned checkpoint for one arm doesn't blank the regime.
# Any completed cohort of the right architecture works for measuring signal
# structure; try arms in priority order (indep shows the MOST structure — an
# upper bound before coupling homogenizes the cohort — then the coupled arms)
# so a missing/cleaned checkpoint for one arm doesn't blank the regime. The
# winning arm's report is copied to the canonical regime filename.
try_regime() {  # $1 regime, $2 cohort, $3 checkpoint-glob template with @ARM@
    local regime="$1" cohort="$2" tmpl="$3" arm glob
    for arm in indep dml rand1 mwmd1; do
        glob="${tmpl//@ARM@/$arm}"
        run_regime "${regime}_${arm}" "$cohort" "$glob"
        if grep -q MWM_gain "$OUT/signal_structure_${regime}_${arm}.txt" 2>/dev/null; then
            cp "$OUT/signal_structure_${regime}_${arm}.txt" \
               "$OUT/signal_structure_${regime}.txt"
            echo "  [$regime] used arm=$arm"; return
        fi
    done
    echo "  [$regime] no usable checkpoint for any arm"
}

try_regime clean  "resnet32:8"            "$CK/resnet32x8_cifar100_K8_@ARM@_seed000*_d001_ckpt.pt"
try_regime hetero "wrn28x10:4,resnet32:4" "$CK/wrn28x10x4-resnet32x4_cifar100_K8_@ARM@_seed000*_d001_ckpt.pt"
try_regime noise  "resnet32:8"            "$CK/resnet32x8_cifar100_K8_@ARM@_noise40_seed000*_d001_ckpt.pt"

echo "=== done $(date) ==="
