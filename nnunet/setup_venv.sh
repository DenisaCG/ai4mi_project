#!/bin/bash
# Build the nnU-Net virtualenv. Run on a Snellius login node (no SLURM job needed):
#   bash nnunet/setup_venv.sh
set -euo pipefail

VENV="$HOME/.venv_nnunet_ai4mi"
NNUNET_VERSION="2.8.1"

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# Keep pip's build and cache files out of the home quota's way.
export PIP_NO_CACHE_DIR=1
export TMPDIR="$HOME/.tmp_pip_build"
mkdir -p "$TMPDIR"

python -m venv "$VENV"
source "$VENV/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel setuptools
python -m pip install "nnunetv2==$NNUNET_VERSION"

python - <<'PY'
import importlib.metadata

import torch

print("torch    :", torch.__version__)
print("cuda     :", torch.version.cuda)
print("nnunetv2 :", importlib.metadata.version("nnunetv2"))
PY

rm -rf "$TMPDIR"
echo "venv ready: $VENV"
