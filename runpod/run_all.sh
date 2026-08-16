#!/usr/bin/env bash
# One-shot: environment -> preprocessing -> full training run.
#
#     cd /workspace && bash repo/runpod/run_all.sh
#
# Safe to re-run. Existing shards are reused, so an interrupted run resumes
# rather than re-downloading 47.9 GB.
#
# Environment (all optional):
#     MODEL=configs/large_300m.yaml    model config (default)
#     TRAIN_CFG=configs/train_300m.yaml
#     CKPT=/workspace/checkpoints/large300_full
#     JOBS=4                           parallel preprocessing workers
#     VENV=/venv                       venv location (default: container disk)
#
# The venv defaults to /venv (container disk) NOT the network volume: it is
# 5.9 GB, rebuilds in ~3 minutes, and putting it on a small paid volume is what
# repeatedly filled the disk and killed training at epoch 1.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/repo}"
OUT="${OUT:-$WORKSPACE/shards}"
LOGS="${LOGS:-$WORKSPACE/logs}"
VENV="${VENV:-/venv}"
MODEL="${MODEL:-configs/large_300m.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train_300m.yaml}"
CKPT="${CKPT:-$WORKSPACE/checkpoints/large300_full}"
JOBS="${JOBS:-4}"
NEED_GB="${NEED_GB:-6}"

export VENV OUT LOGS

banner() { echo; echo "=============== $* ==============="; date; }

# ---------------------------------------------------------------- disk guard
banner "0/4  disk check"
avail=$(df -BG --output=avail "$WORKSPACE" 2>/dev/null | tail -1 | tr -dc '0-9')
avail="${avail:-0}"
echo "free on $WORKSPACE: ${avail} GB (need >= ${NEED_GB} GB)"
if [ "$avail" -lt "$NEED_GB" ]; then
    echo
    echo "!! Not enough space. Shards need 3.2 GB and each checkpoint 1.2 GB."
    echo "   Biggest win -- move the venv off the volume (it is ~5.9 GB):"
    echo "       rm -rf $WORKSPACE/venv"
    echo "   Also removable once the model is copied off:"
    echo "       rm -rf $WORKSPACE/checkpoints"
    df -h "$WORKSPACE"
    du -sh "$WORKSPACE"/* 2>/dev/null | sort -rh | head
    exit 1
fi
mkdir -p "$LOGS" "$(dirname "$CKPT")"

# ---------------------------------------------------------------- environment
banner "1/4  environment"
if [ -f "$VENV/bin/activate" ] && "$VENV/bin/python" -c "import jax" 2>/dev/null; then
    echo "reusing venv at $VENV"
else
    echo "building venv at $VENV"
    VENV="$VENV" bash "$REPO/runpod/setup.sh"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python - <<'PY'
import jax, sys
d = jax.devices()
print("jax", jax.__version__, "|", jax.default_backend(), "|", d)
if d[0].platform != "gpu":
    sys.exit("!! JAX cannot see the GPU -- refusing to run on CPU")
PY

# ---------------------------------------------------------------- preprocess
banner "2/4  preprocessing"
have=$(ls -d "$OUT"/File_* 2>/dev/null | wc -l | tr -d ' ')
if [ "$have" -eq 18 ]; then
    echo "all 18 shards already present, skipping"
else
    echo "$have/18 shards present -- running (resumable)"
    JOBS="$JOBS" OUT="$OUT" bash "$REPO/runpod/preprocess_all.sh" 2>&1 | tee "$LOGS/preprocess.log"
fi
n=$(ls -d "$OUT"/File_* 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -eq 18 ] || { echo "!! only $n/18 shards -- see $LOGS/preprocess.log"; exit 1; }

# ---------------------------------------------------------------- preflight
banner "3/4  preflight"
cd "$REPO"
python tools/preflight.py "$MODEL" --batch 256 2>&1 | tee "$LOGS/preflight.log"
grep -q "  yes" "$LOGS/preflight.log" || {
    echo "!! no configuration fits in device memory -- stopping before a paid run"; exit 1; }

# ---------------------------------------------------------------- train
banner "4/4  training"
echo "model=$MODEL  cfg=$TRAIN_CFG  ckpt=$CKPT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SHARDS=$(ls -d "$OUT"/File_* | sort -V | tr '\n' ' ')
# shellcheck disable=SC2086
python -u train.py --model-config "$MODEL" --train-config "$TRAIN_CFG" \
    --packet-shard $SHARDS --out "$CKPT" 2>&1 | tee "$LOGS/train.log"

banner "done"
ls -lh "$CKPT"
echo
echo "Checkpoint: $CKPT"
echo "Push to HuggingFace:"
echo "    huggingface-cli login"
echo "    REPO_ID=you/protogrok-jax-306m CKPT=$CKPT bash $WORKSPACE/hfsend/hf_push.sh"
