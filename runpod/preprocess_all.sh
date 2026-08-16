#!/usr/bin/env bash
# Stream all 18 UNSW-NB15 Payload-Bytes files and write per-flow shards.
#
# The source files total 47.9 GB, but only ~130 of 1485 columns are read and
# nothing is staged on disk -- output is ~3.2 GB of shards. Resumable: a file
# whose shard already exists is skipped, so a dropped connection costs one
# file, not the whole run.
#
#     bash repo/runpod/preprocess_all.sh          # all 18
#     FILES="1 2 3" bash repo/runpod/preprocess_all.sh
#     JOBS=3 bash repo/runpod/preprocess_all.sh   # 3 at a time (~3 GB RAM each)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-$WORKSPACE/repo}"
OUT="${OUT:-$WORKSPACE/shards}"
VENV="${VENV:-$WORKSPACE/venv}"

# Activate the venv ourselves rather than trusting the caller's shell: xargs
# subshells and `tmux new '<cmd>'` both start fresh shells that have not run
# `source .../activate`, and the fallback system python has no jax.
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi
python -c "import jax, pyarrow, fsspec" 2>/dev/null || {
    echo "!! python at \$(command -v python) is missing deps (jax/pyarrow/fsspec)."
    echo "   Run: bash $REPO/runpod/setup.sh   (MODE=cpu for a preprocessing box)"
    exit 1; }
FILES="${FILES:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18}"
JOBS="${JOBS:-2}"

mkdir -p "$OUT" "$WORKSPACE/logs"
cd "$REPO"

one() {
    local n="$1"
    if [ -f "$OUT/File_$n/payload.npy" ]; then
        echo "  File_$n already done, skipping"; return 0
    fi
    echo "  -> File_$n starting"
    if python tools/prepare_packets.py --files "$n" --out "$OUT" \
            >"$WORKSPACE/logs/prep_$n.log" 2>&1; then
        echo "  -> File_$n done: $(grep -c . "$WORKSPACE/logs/prep_$n.log") log lines"
    else
        echo "  !! File_$n FAILED -- see $WORKSPACE/logs/prep_$n.log"; return 1
    fi
}
export -f one; export OUT WORKSPACE

echo "=== preprocessing files: $FILES (parallel $JOBS) ==="
date
printf '%s\n' $FILES | xargs -P "$JOBS" -I{} bash -c 'one {}'
date

echo
echo "=== shards written ==="
du -sh "$OUT"/* | sort -k2
python - <<PY
import glob, numpy as np, os
tot = 0
for d in sorted(glob.glob(os.path.join("$OUT", "File_*"))):
    n = len(np.load(os.path.join(d, "label_bin.npy"), mmap_mode="r"))
    tot += n
print(f"total flows: {tot:,}")
PY
