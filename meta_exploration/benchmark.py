"""Measure frozen evaluation throughput without changing worlds or model weights.

python -m meta_exploration.benchmark --device cuda:0 --worlds 32 --out runs/benchmark.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from .config import load_config
from .evaluation import evaluate_worlds
from .learning import make_model
from .training import world_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', default='runs/throughput_benchmark.json')
    args = parser.parse_args()
    if min(args.worlds, args.batch_size, args.repeats) < 1:
        parser.error('worlds, batch-size and repeats must be positive')
    torch.set_num_threads(1); torch.manual_seed(0)
    config = load_config()
    config['evaluation']['num_test_worlds'] = args.worlds
    worlds = world_set(config, 'B', 0, 'test')
    models = {key: make_model(config, args.device) for key in ('single_policy','dual_neural','explorer')}
    records = []
    def synchronize():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
    for batch_size in sorted({1,args.batch_size}):
        list(evaluate_worlds(config, models, worlds[:min(4,len(worlds))], 0, 'frozen', frozen_batch_size=batch_size))
        durations = []
        for _ in range(args.repeats):
            synchronize(); start = time.perf_counter()
            result = list(evaluate_worlds(config, models, worlds, 0, 'frozen', frozen_batch_size=batch_size))
            synchronize(); durations.append(time.perf_counter()-start)
        steps = args.worlds * 4 * (config['training']['exploration_steps']+config['training']['exploitation_steps'])
        records.append(dict(batch_size=batch_size, seconds=durations, median_seconds=statistics.median(durations),
                            logical_method_steps_per_second=steps/statistics.median(durations)))
    payload = dict(device=args.device, torch_version=torch.__version__, worlds=args.worlds,
                   evaluation='frozen', hidden=config['training']['hidden'], timings=records,
                   speedup=records[0]['median_seconds']/records[-1]['median_seconds'],
                   note='Includes all four methods, Q replay, history reference and diagnostics; excludes disk logging/training. GPU utilization is not inferred from CPU timings.')
    path = Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
