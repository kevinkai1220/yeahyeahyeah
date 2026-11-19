"""Build a standalone Colab notebook containing a snapshot of v2 source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ('marl', 'configs', 'tests'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.json'):
                archive.write(path, path.relative_to(ROOT))
    archive.write(ROOT/'pyproject.toml', 'pyproject.toml')
payload = buffer.getvalue()
cells = []
def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), outputs=[], execution_count=None))

md('''# MARL Strategic Exploration v2 — Colab GPU

이 노트북에 v2 소스가 포함되어 있어 별도 ZIP 업로드나 GitHub 저장소가 필요 없습니다.

1. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
2. 위에서 아래로 셀을 실행하고 Google Drive 연결을 허용합니다.
3. 기본 `pilot`은 동작/처리량 확인용입니다. 정식 예산은 설정 셀에서 `PROFILE = "full"`로 바꾸세요.

학습 순서: opponent → single → K별 combo history 기반 exploit 사전학습 → frozen exploit 기반 exploration → 평가.
주입 행동은 실제 환경에서 실행되며 prefix에는 PPO loss를 적용하지 않습니다.

**런타임 중단:** Drive에 저장된 checkpoint는 평가에 사용할 수 있지만 현재 구현은 optimizer/RNG resume를 지원하지 않습니다. 중단된 학습을 이어서 실행할 수는 없습니다. 재학습은 새 RUN_ID로 시작하세요. 완료된 학습의 평가는 저장 경로로 다시 실행할 수 있습니다.
Colab GPU 종류·할당 시간은 가변적이며 full 실험이 한 세션에 끝난다는 보장은 없습니다.
[공식 Colab FAQ](https://research.google.com/colaboratory/faq.html)
''')
code('''import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, json, subprocess, uuid
from pathlib import Path
from datetime import datetime, timezone
import torch
assert torch.cuda.is_available(), 'GPU 런타임을 선택하세요. CPU 전용 PyTorch를 설치하지 마세요.'
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
''')
md('## 포함된 소스 설치\nColab의 CUDA PyTorch를 그대로 사용합니다. 소스를 변경하면 `scripts/build_colab_notebook.py`로 노트북을 다시 생성하세요.')
code(f'''import base64, hashlib, io, zipfile
SOURCE = Path('/content/marl_v2_' + uuid.uuid4().hex[:8])
SOURCE.mkdir(parents=True, exist_ok=False)
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(SOURCE)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(SOURCE)], check=True)
os.chdir(SOURCE)
print('Source:', SOURCE)
''')
md('## Google Drive에 결과 보존\n코드는 로컬 디스크에서 실행하며 로그·모델·평가 결과는 Drive에 직접 저장합니다. 기록 I/O로 인한 지연이 있을 수 있습니다.')
code('''from google.colab import drive
drive.mount('/content/drive')
RESULTS = Path('/content/drive/MyDrive/marl_strategic_exploration_v2/runs')
RESULTS.mkdir(parents=True, exist_ok=True)

def run_cli(*args):
    subprocess.run([sys.executable, '-u', '-m', 'marl.cli', *map(str, args)], cwd=SOURCE, check=True)
''')
md('## 테스트와 GPU smoke\n별도 작은 설정으로 전체 pipeline을 검증합니다. smoke 보상은 학습 성능을 뜻하지 않습니다.')
code('''subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], cwd=SOURCE, check=True)
smoke = json.loads((SOURCE/'configs/v2_smoke.json').read_text())
smoke['device'] = 'cuda'
smoke_path = SOURCE/'configs/colab_smoke.json'
smoke_path.write_text(json.dumps(smoke, indent=2))
SMOKE_OUT = RESULTS / ('smoke_' + uuid.uuid4().hex[:8])
run_cli('run', '--config', smoke_path, '--output', SMOKE_OUT)
''')
md('''## 학습 예산 설정
`pilot`: 256 games, K=10, opponent 50 / exploit 100 / exploration 100 updates. 성능 결론용이 아닙니다.
`full`: 8192 games, K=10/25/50, opponent 5000 / exploit 10000 / exploration 10000 updates.

VRAM 부족 시 `tasks_per_update`, `rollouts_per_task`, PPO `minibatch`를 줄인 뒤 새 run으로 시작하세요. 배치 변경은 학습 조건도 바꿉니다.
게임 규칙 다양성은 기존 R1/R2 family 내 reward/combo 샘플링입니다.
''')
code('''PROFILE = 'pilot'  # 'pilot' 또는 'full'
SEED = 0
assert PROFILE in ('pilot', 'full')
config = json.loads((SOURCE/'configs/default.json').read_text())
config.update(device='cuda', seed=SEED)
if PROFILE == 'pilot':
    config.update(training_tasks=256, evaluation_tasks=16, budgets=[10],
                  tasks_per_update=16, rollouts_per_task=8, opponent_updates=50,
                  updates=100, exploit_pretrain_updates=100, evaluation_rollouts=2,
                  checkpoint_every=25)
    config['ppo']['minibatch'] = 64
# 추가 설정은 여기서 수정합니다.
config['history_injection_probability'] = 0.75
RUN_ID = f"{PROFILE}_seed{SEED}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
OUT = RESULTS / RUN_ID
CONFIG = SOURCE/'configs/colab_run.json'
CONFIG.write_text(json.dumps(config, indent=2))
print('Output:', OUT)
print(json.dumps(config, indent=2))
''')
md('## 1. Task split 준비\n동일 output 재실행은 덮어쓰지 않고 실패합니다.')
code("run_cli('prepare', '--config', CONFIG, '--output', OUT)\n")
md('## 2. Opponent 사전학습')
code("run_cli('pretrain', '--output', OUT)\n")
md('## 3. Single 및 dual 학습\n각 K의 exploit 사전학습을 완료한 후 exploration 학습을 시작합니다. 학습 예산은 수렴 보장 조건이 아닙니다.')
code("run_cli('train', '--output', OUT)\n")
md('## 4. Held-out 평가\n기존 evaluator는 CPU에서 실행합니다. full 설정은 평가에도 시간이 걸립니다.')
code("run_cli('evaluate', '--output', OUT)\n")
md('## 학습 곡선 및 결과 확인\nExploit 사전학습 곡선은 주입 history 분포에서의 지표입니다. 최종 평가와 구분하세요.')
code('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
for path in sorted(OUT.glob('*.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if 'mean_exploitation_reward' in r]
    if rows:
        ax.plot([r['update'] for r in rows], [r['mean_exploitation_reward'] for r in rows], label=path.stem)
ax.set(xlabel='Update (within stage)', ylabel='Mean exploitation reward')
ax.legend()
plt.show()
summary = OUT/'evaluation/summary.json'
if summary.exists():
    print(summary.read_text()[:20000])
print('모든 결과:', OUT)
''')
nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(colab=dict(name='marl_v2_colab.ipynb'), accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.11')), cells=cells)
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:03}'
path = ROOT/'notebooks/marl_v2_colab.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')
print(path)
