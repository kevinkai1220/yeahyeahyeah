"""GPU arithmetic probes and dynamically scheduled independent experiment workers."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import torch
from .training import write_json, append_json


def hardware_report():
    report = dict(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  devices=[], usable_devices=[])
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu',
                            '--format=csv'], capture_output=True, text=True, timeout=15)
        report['nvidia_smi'] = p.stdout or p.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        report['nvidia_smi'] = str(e)
    for i in range(torch.cuda.device_count()):
        # Probe in a child: a bad CUDA device must not poison the launcher process.
        code = ('import torch,json; d="cuda:'+str(i)+'"; '
                'x=torch.ones((32,32),device=d); y=x@x; torch.cuda.synchronize(d); '
                'assert y[0,0].item()==32; '
                'print(json.dumps({"name":torch.cuda.get_device_name('+str(i)+'),'
                '"memory":torch.cuda.mem_get_info('+str(i)+')}))')
        try:
            p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=45)
            report['devices'].append(dict(index=i, usable=p.returncode==0, details=p.stdout, error=p.stderr))
            if p.returncode == 0:
                report['usable_devices'].append(i)
        except subprocess.TimeoutExpired:
            report['devices'].append(dict(index=i, usable=False, error='arithmetic probe timeout'))
    return report


def job_path(root, regime, seed):
    return Path(root) / regime / f'seed_{seed}'


def launch(config, config_path, root, regimes, seeds, require_gpu=False,
           workers_per_gpu=1, frozen_batch_size=32):
    if workers_per_gpu < 1 or frozen_batch_size < 1:
        raise ValueError('Worker count and batch size must be positive')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    report = hardware_report(); write_json(root / 'hardware.json', report)
    hc = config['hardware']
    if hc['device'] == 'cpu':
        devices = []
    elif hc['device'].startswith('cuda:'):
        i = int(hc['device'].split(':')[1]); devices = [i] if i in report['usable_devices'] else []
    else:
        devices = report['usable_devices']
    if not hc['use_all_available_gpus']:
        devices = devices[:1]
    if require_gpu and not devices:
        raise RuntimeError('No usable CUDA GPU: check the NVIDIA driver, CUDA-enabled PyTorch, '
                           'and CUDA_VISIBLE_DEVICES. See ' + str(root / 'hardware.json'))
    slots = [f'cuda:{i}' for _ in range(workers_per_gpu) for i in devices] or ['cpu'] * hc['cpu_workers']
    write_json(root / 'scheduling.json', dict(workers_per_gpu=workers_per_gpu,
               frozen_batch_size=frozen_batch_size, slots=slots,
               note='Runtime-only controls; scientific config/checkpoint compatibility is unchanged'))
    jobs = queue.Queue()
    for regime in regimes:
        for seed in seeds:
            jobs.put((regime, seed))
    lock = threading.Lock(); failures = []; excluded = set()
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=5)
                append_json(root / 'gpu_utilization.jsonl', dict(time=time.time(), sample=p.stdout, error=p.stderr))
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return
            stop.wait(5)
    monitor_thread = threading.Thread(target=monitor, daemon=True); monitor_thread.start()
    def worker(slot, device):
        while True:
            try:
                regime, seed = jobs.get_nowait()
            except queue.Empty:
                return
            out = job_path(root, regime, seed); out.mkdir(parents=True, exist_ok=True)
            completed = out / 'complete.json'
            if completed.exists():
                if json.loads((out / 'config.json').read_text()) != config:
                    with lock:
                        failures.append(dict(regime=regime, seed=seed, reason='completed job config mismatch'))
                continue
            for attempt in range(hc['max_retries']+1):
                with lock:
                    if device in excluded:
                        device = 'cpu'
                start = time.monotonic()
                cmd = [sys.executable, '-m', 'meta_exploration', 'run', '--config', str(Path(config_path).resolve()),
                       '--regime', regime, '--seed', str(seed), '--out', str(out.resolve()), '--device', device,
                       '--frozen-batch-size', str(frozen_batch_size)]
                print(f'Start {regime}/seed_{seed} on {device} (worker {slot}, attempt {attempt})', flush=True)
                with (out / f'worker_{slot}_attempt_{attempt}.log').open('w') as log:
                    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                event = dict(regime=regime, seed=seed, device=device, worker=slot, attempt=attempt,
                             returncode=p.returncode, runtime_seconds=time.monotonic()-start)
                with lock:
                    append_json(root / 'launcher.jsonl', event)
                print(f'Finished {regime}/seed_{seed}: exit={p.returncode}, {event["runtime_seconds"]:.1f}s', flush=True)
                if p.returncode == 0:
                    break
                error_text = (out / f'worker_{slot}_attempt_{attempt}.log').read_text().lower()
                if device.startswith('cuda') and any(token in error_text for token in ('cuda error', 'cuda out of memory', 'cublas', 'device-side assert')):
                    with lock:
                        excluded.add(device)
                        append_json(root / 'excluded_devices.jsonl', dict(device=device, reason='runtime CUDA failure', job=str(out)))
                    device = 'cpu'  # quarantine this slot; retry and future jobs use CPU
                if attempt == hc['max_retries']:
                    with lock:
                        failures.append(event)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, i, d) for i, d in enumerate(slots)]
        for future in futures:
            future.result()
    stop.set(); monitor_thread.join(timeout=6)
    write_json(root / 'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed; see {root}/failures.json and worker logs')
