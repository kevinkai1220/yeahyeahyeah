import argparse
import fcntl
import json
import time
from pathlib import Path
import torch
from .config import load_config, REGIMES
from .training import train, write_json
from .evaluation import evaluate
from .hardware import hardware_report, launch


def main():
    parser = argparse.ArgumentParser(description='Reproducible three-action meta-exploration')
    parser.add_argument('command', choices=['run', 'pretrain', 'train', 'evaluate', 'launch', 'plot', 'inspect', 'hardware'])
    parser.add_argument('--config'); parser.add_argument('--out', default='runs/meta')
    parser.add_argument('--regime', choices=list(REGIMES), default='B')
    parser.add_argument('--regimes', nargs='+', choices=list(REGIMES), default=list(REGIMES))
    parser.add_argument('--seed', type=int, default=0); parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--device'); parser.add_argument('--history', nargs='+')
    parser.add_argument('--require-gpu', action='store_true', help='Fail launch if no usable CUDA GPU is found')
    args = parser.parse_args(); config = load_config(args.config)
    torch.set_num_threads(config['hardware']['threads_per_worker'])
    if args.command == 'hardware':
        print(json.dumps(hardware_report(), indent=2)); return
    if args.command == 'plot':
        from .plotting import plot_results
        plot_results(args.out); return
    if args.command == 'inspect':
        from .plotting import inspect_history
        if not args.history:
            parser.error('--history paths required')
        inspect_history(args.history, args.out); return
    if args.command == 'launch':
        if not args.config:
            parser.error('--config required for launcher')
        launch(config, args.config, args.out, args.regimes, args.seeds or config['evaluation']['seeds'],
               require_gpu=args.require_gpu)
        if config['logging']['save_plots']:
            from .plotting import plot_results
            plot_results(args.out)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # OS advisory lock releases on crash; no stale PID files or conflicting writes.
    with (out / '.job.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another worker owns {out}')
        identity = dict(regime=args.regime, seed=args.seed)
        identity_path = out / 'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Output directory belongs to another regime/seed')
        write_json(identity_path, identity)
        cp = out / 'config.json'
        if cp.exists() and json.loads(cp.read_text()) != config:
            raise ValueError('Output directory has a different config')
        device = args.device or config['hardware']['device']
        if device == 'auto':
            usable = hardware_report()['usable_devices']
            device = f'cuda:{usable[0]}' if usable else 'cpu'
        start = time.monotonic()
        if args.command in ('run', 'pretrain', 'train'):
            train(config, args.regime, args.seed, out, device, pretrain_only=args.command=='pretrain')
        if args.command in ('run', 'evaluate'):
            evaluate(config, args.regime, args.seed, out, device)
        write_json(out / 'runtime.json', dict(device=device, torch_version=torch.__version__,
                    runtime_seconds=time.monotonic()-start,
                    peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0))
        if args.command == 'run':
            write_json(out / 'complete.json', dict(regime=args.regime, seed=args.seed, status='complete'))
        print(f'{args.command} complete: {out}', flush=True)


if __name__ == '__main__':
    main()
