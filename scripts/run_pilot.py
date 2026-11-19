"""Run the 100-update CPU pilot and leave a report even if training fails."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/pilot_seed0'
STATUS = ROOT / 'runs/pilot_seed0.status.json'
REPORT = ROOT / 'runs/pilot_seed0_report.md'


def status(**fields):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(fields, indent=2) + '\n')
    temporary.replace(STATUS)


def report(returncode, elapsed):
    lines = ['# CPU 100-update pilot', '',
             f'- Exit code: {returncode}', f'- Elapsed minutes: {elapsed / 60:.1f}',
             '- Seed: 0; T=100; K=10/25/50; 100 updates per model.',
             '- Training/evaluation tasks: 256/128; 8 evaluation rollouts per task/opponent.',
             '- A single-seed pilot does not establish statistical superiority.', '',
             '## Training trends', '',
             '| Model | Metric | First 20 updates | Last 20 updates |',
             '|---|---|---:|---:|']
    for name in ('opponent', 'single', 'dual_k10', 'dual_k25', 'dual_k50'):
        path = OUTPUT / f'{name}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        metrics = ['mean_total_reward', 'mean_exploitation_reward']
        metrics += [key for key in rows[-1] if key.endswith('_entropy')]
        for metric in metrics:
            if metric in rows[-1]:
                first = statistics.mean(row[metric] for row in rows[:20])
                last = statistics.mean(row[metric] for row in rows[-20:])
                lines.append(f'| {name} | {metric} | {first:.4f} | {last:.4f} |')
    issues = []
    for path in OUTPUT.glob('*_issues.jsonl'):
        issues += [f'{path.stem}: {line}' for line in path.read_text().splitlines()]
    lines += ['', '## Stability events', '', f'Recorded events: {len(issues)}']
    lines += ['- ' + issue for issue in issues]
    summary_path = OUTPUT / 'evaluation/summary.json'
    if returncode == 0 and summary_path.exists():
        lines += ['', '## Held-out evaluation', '',
                  '| K | Opponent | Agent | Exploit mean | AUC | Discovery rate | Total reward |',
                  '|---:|---|---|---:|---:|---:|---:|']
        for row in json.loads(summary_path.read_text()):
            lines.append(f'| {row["k"]} | {row["opponent"]} | {row["agent"]} | '
                         f'{row["exploitation_mean"]:.4f} | {row["adaptation_auc"]:.4f} | '
                         f'{row["combo_discovery_rate"]:.4f} | {row["total_reward"]:.4f} |')
    else:
        lines += ['', 'The pipeline did not finish successfully. Inspect the process log; no algorithm changes were made.']
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    started = time.time()
    status(state='running', pid=os.getpid(), started_unix=started, output=str(OUTPUT))
    code = 1
    try:
        result = subprocess.run([sys.executable, '-u', '-m', 'marl.cli', 'run',
                                 '--config', 'configs/pilot.json', '--output', str(OUTPUT)], cwd=ROOT)
        code = result.returncode
        report(code, time.time() - started)
    finally:
        status(state='completed' if code == 0 else 'failed', pid=os.getpid(),
               exit_code=code, started_unix=started, elapsed_seconds=time.time()-started,
               output=str(OUTPUT), report=str(REPORT))
    return code


if __name__ == '__main__':
    sys.exit(main())
