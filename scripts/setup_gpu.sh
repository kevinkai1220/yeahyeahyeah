#!/usr/bin/env bash
# Linux GPU server setup. Reuse an existing CUDA PyTorch installation when possible.
set -euo pipefail
cd "$(dirname "$0")/.."
META_BASE_PYTHON="${PYTHON:-python3}"
"$META_BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required; 3.11+ recommended"'
if [[ ! -x .venv/bin/python ]]; then
    "$META_BASE_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools
if [[ -n "${TORCH_INDEX_URL:-}" ]] || ! .venv/bin/python -c 'import torch; assert torch.version.cuda is not None; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2,2)' >/dev/null 2>&1; then
    # Override TORCH_INDEX_URL if another CUDA wheel family suits the server.
    .venv/bin/python -m pip install --upgrade --force-reinstall torch \
        --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
fi
.venv/bin/python -m pip install -e '.[meta]'
.venv/bin/python - <<'PY'
import json
from meta_exploration.hardware import hardware_report
report = hardware_report()
print(json.dumps(report, indent=2))
if not report['usable_devices']:
    raise SystemExit('No usable CUDA GPU. Check driver, CUDA wheel compatibility and GPU allocation; see https://pytorch.org/get-started/locally/')
print('Setup complete. Healthy CUDA devices:', report['usable_devices'])
PY
