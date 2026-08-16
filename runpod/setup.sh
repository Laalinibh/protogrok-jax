#!/usr/bin/env bash
# Provision a RunPod box for Protogrok-JAX.
#
# Two modes, because preprocessing needs no GPU at all (it is network + CPU
# bound) and running it on an H100 wastes rental:
#
#   bash repo/runpod/setup.sh          # GPU box for training (default)
#   MODE=cpu bash repo/runpod/setup.sh # cheap box for preprocessing only
#
# Training box:
#   Image  : runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#            (any CUDA 12.x; Hopper/H100 needs CUDA >= 11.8)
#   GPU    : H100 80GB SXM preferred, then H100 PCIe, then A100 80GB
#   Volume : network volume mounted at /workspace, >= 50 GB
#   RAM    : >= 32 GB (one 2.5M-row Arrow row group is ~3 GB)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/repo}"
VENV="${VENV:-$WORKSPACE/venv}"
MODE="${MODE:-gpu}"

echo "=== Protogrok-JAX RunPod setup (mode=$MODE) ==="

if [ "$MODE" = "gpu" ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
        echo "!! no GPU visible -- attach one, or use MODE=cpu for preprocessing"
        exit 1; }
fi

apt-get update -qq && apt-get install -y -qq python3-venv tmux >/dev/null

python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -q -U pip wheel

if [ "$MODE" = "gpu" ]; then
    echo "=== installing JAX with CUDA 12 ==="
    pip install -q -U "jax[cuda12]"
else
    # models/__init__ imports flax/jax, so even the preprocessing path needs
    # them -- but the CPU build is enough and installs in seconds.
    echo "=== installing JAX (CPU build; preprocessing only) ==="
    pip install -q -U jax
fi
pip install -q -U flax optax orbax-checkpoint chex numpy pyyaml \
                  pyarrow fsspec aiohttp huggingface_hub scapy

MODE="$MODE" python - <<'PY'
import os, jax
devs = jax.devices()
print("jax", jax.__version__, "| backend:", jax.default_backend(), "|", devs)
if os.environ.get("MODE", "gpu") == "gpu":
    assert devs and devs[0].platform == "gpu", \
        "JAX is not seeing the GPU -- stop here, do not train on CPU"
    print("GPU confirmed.")
else:
    print("CPU mode: fine for preprocessing, do NOT train on this box.")
PY

echo
echo "=== setup complete ==="
echo "  source $VENV/bin/activate"
if [ "$MODE" = "gpu" ]; then
    echo
    echo "!! RUN THE PREFLIGHT BEFORE ANYTHING ELSE -- it compiles the training"
    echo "   step on this GPU and reports real memory use in about a minute."
    echo "   Skipping it risks an OOM hours into a paid run."
    echo "     cd $REPO && python tools/preflight.py configs/xl_500m.yaml --batch 256"
    echo
    echo "  then: bash $REPO/runpod/train_xl.sh    # ~3 h for 500M x 10 epochs on H100"
else
    echo "  bash $REPO/runpod/preprocess_all.sh    # streams 47.9 GB -> ~3.2 GB shards"
fi
