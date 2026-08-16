#!/usr/bin/env bash
# Train Protogrok-JAX on the full UNSW-NB15 packet-byte corpus.
#
#     bash repo/runpod/train_xl.sh                    # 500M, 10 epochs
#     MODEL=configs/base_124m.yaml bash .../train_xl.sh   # cheap sanity run first
#
# Run it under tmux so a dropped SSH session does not kill the job:
#     tmux new -s train 'bash repo/runpod/train_xl.sh 2>&1 | tee /workspace/logs/train.log'
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/repo}"
OUT="${OUT:-$WORKSPACE/shards}"
CKPT="${CKPT:-$WORKSPACE/checkpoints/xl500_anomaly}"
MODEL="${MODEL:-configs/xl_500m.yaml}"
TRAIN_CFG="${TRAIN_CFG:-configs/train_xl.yaml}"
VENV="${VENV:-$WORKSPACE/venv}"

# Activate the venv ourselves: `tmux new -s train '<cmd>'` starts a fresh shell
# that has not sourced it, and the system python has no jax -- which would kill
# a multi-hour run seconds after launch.
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi
python -c "import jax" 2>/dev/null || {
    echo "!! python at $(command -v python) has no jax. Run: bash $REPO/runpod/setup.sh"
    exit 1; }
python -c "import jax,sys; sys.exit(0 if jax.devices()[0].platform=='gpu' else 1)" || {
    echo "!! JAX sees no GPU -- refusing to start a multi-hour CPU run."
    echo "   This box may have the CPU-only jax from 'MODE=cpu'. Re-run setup.sh."
    exit 1; }

cd "$REPO"
mkdir -p "$(dirname "$CKPT")" "$WORKSPACE/logs"

SHARDS=$(ls -d "$OUT"/File_* 2>/dev/null | sort -V | tr '\n' ' ')
[ -z "$SHARDS" ] && { echo "!! no shards in $OUT -- run preprocess_all.sh first"; exit 1; }
echo "=== shards: $(echo "$SHARDS" | wc -w) ==="

python tools/count_params.py "$MODEL"

# Let JAX grow its allocation instead of preallocating 90% of VRAM up front,
# so an OOM surfaces as a clear error rather than at an arbitrary later step.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.92

echo "=== training: $MODEL / $TRAIN_CFG ==="
date
# -u: unbuffered, so `tee`d logs show progress live rather than in blocks
# shellcheck disable=SC2086
python -u train.py \
    --model-config "$MODEL" \
    --train-config "$TRAIN_CFG" \
    --packet-shard $SHARDS \
    --out "$CKPT"
date

echo
echo "=== checkpoint: $CKPT ==="
ls -lh "$CKPT"
echo "Copy it off the pod before terminating:"
echo "  runpodctl send $CKPT   (or scp / upload to HF)"
