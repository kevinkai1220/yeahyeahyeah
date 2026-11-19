#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .deps ]]; then
    export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "${META_PYTHON:-}" ]]; then
    META_INTERPRETER="$META_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    META_INTERPRETER="$PWD/.venv/bin/python"
else
    META_INTERPRETER=python3
fi
if ! command -v "$META_INTERPRETER" >/dev/null 2>&1; then
    echo "Python not found: $META_INTERPRETER. Run bash scripts/setup_gpu.sh first." >&2
    exit 1
fi
exec "$META_INTERPRETER" -m meta_exploration "$@"
