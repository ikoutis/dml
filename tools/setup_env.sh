#!/bin/bash
# =============================================================================
# tools/setup_env.sh — create the suite's conda environment on Wulver.
#
# Run ONCE on a LOGIN node, from the repo root:
#     bash tools/setup_env.sh
#
# Creates a prefix env at $HOME/envs/dml-torch (override by exporting
# DML_CONDA_ENV first) with python 3.11 and installs requirements.txt.
# The torch/torchvision wheels on PyPI bundle the CUDA 12 runtime, so no
# cluster CUDA module is needed — the driver on the GPU nodes is enough.
#
# The sbatch scripts activate the same path by default; if you point
# DML_CONDA_ENV somewhere else here, export it in the shell you submit
# from too (sbatch exports your environment by default).
#
# Login nodes have no GPU, so torch.cuda.is_available() is False here —
# that is expected. Verify on a GPU node with the debug QOS:
#     srun --account=dept_dms --qos=debug --partition=gpu \
#          --gres=gpu:a100_10g:1 --time=00:05:00 \
#          bash -lc 'module load Miniforge3 && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${DML_CONDA_ENV:-$HOME/envs/dml-torch}" && python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
# =============================================================================
set -euo pipefail

ENV_PREFIX="${DML_CONDA_ENV:-$HOME/envs/dml-torch}"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "$ENV_PREFIX" ]; then
    echo "[*] env already exists at $ENV_PREFIX — updating packages in place"
else
    echo "[*] creating conda env at $ENV_PREFIX"
    conda create -y --prefix "$ENV_PREFIX" python=3.11
fi

set +u
conda activate "$ENV_PREFIX"
set -u

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "[*] sanity check (login nodes have no GPU — cuda=False is expected here):"
python - <<'EOF'
import torch, torchvision, numpy, scipy, pandas
print(f"    torch {torch.__version__} | torchvision {torchvision.__version__} "
      f"| cuda available: {torch.cuda.is_available()}")
EOF

echo
echo "[*] done. Next steps (login node):"
echo "      python tools/stage_data.py cifar100"
echo "      pytest tests/                      # ~2 min, CPU-only"
echo "    then submit, e.g.: sbatch slurm/r1_pairs.sbatch"
if [ "$ENV_PREFIX" != "$HOME/envs/dml-torch" ]; then
    echo "    NOTE: non-default env path — export DML_CONDA_ENV=$ENV_PREFIX"
    echo "    in the shell you sbatch from."
fi
