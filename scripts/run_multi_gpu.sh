#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional positional arguments: config path, output directory. Further CLI args
# (e.g. --regimes B --seeds 0 1) are forwarded to the experiment launcher.
META_CONFIG="${1:-configs/meta/default.json}"
META_OUTPUT="${2:-runs/full_study}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec bash scripts/meta.sh launch --require-gpu --config "$META_CONFIG" --out "$META_OUTPUT" \
    --workers-per-gpu "${WORKERS_PER_GPU:-2}" --frozen-batch-size "${FROZEN_BATCH_SIZE:-32}" "$@"
