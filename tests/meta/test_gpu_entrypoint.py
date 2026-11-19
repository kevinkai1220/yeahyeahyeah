import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from meta_exploration.config import load_config
from meta_exploration.hardware import launch, hardware_report


def test_required_gpu_rejects_cpu_fallback(tmp_path, monkeypatch):
    import meta_exploration.hardware as hardware
    monkeypatch.setattr(hardware, 'hardware_report', lambda: {'usable_devices': []})
    with pytest.raises(RuntimeError, match='No usable CUDA GPU'):
        launch(load_config('configs/meta/smoke.json'), 'configs/meta/smoke.json', tmp_path, ['B'], [0], require_gpu=True)
    assert json.loads((tmp_path/'hardware.json').read_text())['usable_devices'] == []
    assert not (tmp_path/'B').exists()


def test_portable_wrapper_respects_explicit_interpreter():
    env = dict(os.environ, META_PYTHON=sys.executable)
    result = subprocess.run(['bash', 'scripts/meta.sh', '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and '--require-gpu' in result.stdout


def test_probe_synchronizes_each_selected_device(monkeypatch):
    import types
    import meta_exploration.hardware as hardware
    calls = []
    monkeypatch.setattr(hardware.torch.cuda, 'device_count', lambda: 2)
    def fake_run(command, **kwargs):
        if command[0] != 'nvidia-smi':
            calls.append(command[-1])
            assert 'synchronize(d)' in command[-1]
            return types.SimpleNamespace(returncode=0, stdout='{}', stderr='')
        return types.SimpleNamespace(returncode=0, stdout='GPU list', stderr='')
    monkeypatch.setattr(hardware.subprocess, 'run', fake_run)
    assert hardware.hardware_report()['usable_devices'] == [0, 1]
    assert len(calls) == 2
