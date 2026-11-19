"""Join learning, discovery and event-aligned reward curves by checkpoint update."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    curves=[];discovery=[];aligned=[];lines=['# Learning / combo timing tracking','',
        'Two time axes: training update and episode round. Diagnostic tasks/seeds stay fixed across checkpoints. Checkpoints include update 0. Diagnostic tasks are independent of training and final evaluation.',
        'Restricted first-discovery time includes misses censored at T. Lower is faster only when read together with discovery rate. Conditional mean alone can hide failures.',
        'Largest return slope below means the largest observed adjacent-checkpoint difference / updates, not a statistically detected change point. Lag 0 is the first successful combo by definition; only sustained positive-lag rewards/reuse suggest subsequent exploitation. Before/after observations are not a causal intervention.',
        '', '| Case | K | Discovery by K: initial → final | Restricted discovery time: initial → final | Largest return slope interval | Return change/update | Discovery time change/update in same interval |',
        '|---|---:|---|---|---|---:|---:|']
    for path in sorted(args.output.glob('*/tracking/*/learning_curve.csv')):
        case=path.parents[2].name
        with path.open() as stream:
            records=list(csv.DictReader(stream))
        curves.extend([dict(case=case,**r) for r in records])
        learned=[r for r in records if r['agent']=='dual_learned']
        if learned:
            first,last=learned[0],learned[-1]
            candidates=[r for r in learned if r['return_change_per_update']]
            peak=max(candidates,key=lambda r:float(r['return_change_per_update'])) if candidates else None
            if peak:
                pos=learned.index(peak);previous=learned[pos-1]
                interval=f"{previous['update']} → {peak['update']}"
                lines.append(f"| {case} | {last['k']} | {float(first['discovery_rate_by_k']):.1%} → {float(last['discovery_rate_by_k']):.1%} | {float(first['restricted_mean_first_combo_round']):.2f} → {float(last['restricted_mean_first_combo_round']):.2f} | {interval} | {float(peak['return_change_per_update']):+.4f} | {float(peak['discovery_time_change_per_update']):+.4f} |")
        for summary in sorted(path.parent.glob('update_*/summary.json')):
            update=int(summary.parent.name.split('_')[1])
            for row in json.loads(summary.read_text()):
                identity=dict(case=case,k=row['k'],agent=row['agent'],update=update)
                discovery.extend([dict(**identity,**r) for r in row['discovery_cdf']])
                aligned.extend([dict(**identity,**r) for r in row['event_aligned_pair_rewards']])
    write_csv(args.output/'learning_curves.csv',curves)
    write_csv(args.output/'discovery_curves.csv',discovery)
    write_csv(args.output/'event_aligned_rewards.csv',aligned)
    lines+=['','Raw per-round traces: CASE/tracking/dual_kK/update_NNNNNN/AGENT.steps.jsonl.gz. Each includes actions, behavior/exploration/target probabilities, rewards, cumulative reward, rolling reward, phase and first-discovery flag.',
        'Model snapshots: CASE/checkpoints/dual_kK/update_NNNNNN.pt. Training updates after a checkpoint do not overwrite earlier snapshots. Optimizer resume is not supported.']
    (args.output/'tracking_report.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'tracking_report.md')

if __name__=='__main__':main()
