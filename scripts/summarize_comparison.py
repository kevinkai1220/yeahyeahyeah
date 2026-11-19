"""Aggregate per-task comparisons; bootstrap tasks, never count rollouts as seeds."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output
    summaries=[]; task_values={}; stability={}
    for case in sorted(root.glob('*_lambda*')):
        if not (case/'evaluation/summary.json').exists(): continue
        config=json.loads((case/'config.json').read_text())
        summary=json.loads((case/'evaluation/summary.json').read_text())
        for k in config['budgets']:
            for agent in ('dual_learned','dual_random','dual_target','single_gru','q_learning'):
                rr=[r for r in summary if r['k']==k and r['agent']==agent]
                if not rr:continue
                row=dict(case=case.name,algorithm=config['exploit_learning'],exploration_lambda=config['exploration_lambda'],k=k,agent=agent)
                for key in ('exploitation_mean','total_reward','adaptation_auc','combo_discovery_rate','combo_discovery_rate_by_k','safe_action_fraction'):
                    vals=[r[key] for r in rr if r.get(key) is not None]
                    row[key]=float(np.mean(vals)) if vals else None
                for label,numerator,denominator in [('combo_reuse_rate','reuse_completions','reuse_opportunities'),('completion_given_prefix','reuse_completions','reuse_prefixes')]:
                    den=sum(r.get(denominator,0) for r in rr)
                    row[label]=sum(r.get(numerator,0) for r in rr)/den if den else None
                summaries.append(row)
        groups=defaultdict(list)
        for line in (case/'evaluation/episodes.jsonl').open():
            row=json.loads(line)
            if row['agent']=='dual_learned':
                groups[row['k'],row['task_index']].append(row['exploitation_mean'])
        for k in config['budgets']:
            indices=sorted(t for budget,t in groups if budget==k)
            task_values[case.name,k]=np.array([np.mean(groups[k,t]) for t in indices])
        stability[case.name]=sum(len(p.read_text().splitlines()) for p in case.glob('*_issues.jsonl'))
    if not summaries:raise RuntimeError('No completed evaluations')
    (root/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# Lambda × off-policy 비교 실험','',
        '학습 seed=0, 각 모델 100 updates, T=100, K=10/25/50. 같은 train/eval tasks 256/128 및 고정 opponent를 사용했다. 평가 task·opponent별 8회 반복. 표는 상대 3종 평균이다.',
        'PPO λ=0도 동일한 배치 수집/RNG 규약으로 새로 학습했다. Single GRU/Q-learning은 기존 파일럿의 동일 체크포인트·동일 평가 결과를 공통 기준으로 재사용했다.',
        '', 'Exploration target: λ × sum(r[t:K]) + sum(r[K:T]). Exploitation PPO target은 기존 total-to-go다.',
        'Off-policy 조건: exploration head는 fresh data PPO, target head는 전체 시퀀스(t<K 포함)를 uniform FIFO replay에서 V-trace로 학습. GRU는 매 replay에서 처음부터 재계산하며 shared encoder/GRU를 유지한다. Capacity=1,024 sequences, 16 replay minibatches × 16 sequences/update, rho/c clip=1, gamma=1, terminal bootstrap=0. 평가 중 가중치는 고정된다.',
        '데이터 수집량은 같다. PPO는 update당 joint gradient step 16회, off-policy는 exploration PPO 16회 + V-trace 16회다. 따라서 replay만의 효과를 분리한 compute-matched ablation은 아니다. Off-policy는 현재 target head가 모든 구간을 제어한다고 가정한 return을 학습한다.',
        '', '| 학습 | λ | K | 활용 평균 보상 | 전체 보상 | K까지 발견률 | 발견 후 콤보 재활용률 | 정답 prefix 이후 완성률 | 안전 행동률 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    pct=lambda value: 'NA' if value is None else f'{value:.1%}'
    for r in summaries:
        if r['agent']!='dual_learned':continue
        lines.append(f"| {r['algorithm']} | {r['exploration_lambda']:g} | {r['k']} | {r['exploitation_mean']:.4f} | {r['total_reward']:.2f} | {pct(r['combo_discovery_rate_by_k'])} | {pct(r['combo_reuse_rate'])} | {pct(r['completion_given_prefix'])} | {pct(r['safe_action_fraction'])} |")
    lines+=['','재활용률은 최초 발견 후이면서 exploitation 구간에 온전히 포함된 두 라운드 쌍에서 계산했다. 발견하지 못한 episode는 해당 분모에 들어가지 않는다. 조건별 발견 집단이 다르므로 인과적 효과로 해석하지 않는다.',
        '', '## Task 단위 paired bootstrap','',
        '각 task 내 opponent·반복을 먼저 평균하고, 대응 task 차이를 10,000회 bootstrap했다. 95% 구간은 task 변동만 반영하고 training-seed 변동은 반영하지 않는다. 여러 비교에 대한 다중검정 보정은 하지 않았다.',
        '', '| 비교 (앞−뒤) | K | 활용 보상 차이 | 95% 구간 |','|---|---:|---:|---:|']
    rng=np.random.default_rng(2026);contrasts=[]
    pairs=[(f'{a}_lambda{w}',f'{a}_lambda0p0') for a in ('ppo','vtrace') for w in ('0p5','1p0')]
    pairs += [(f'vtrace_lambda{w}',f'ppo_lambda{w}') for w in ('0p0','0p5','1p0')]
    for first,second in pairs:
        for k in (10,25,50):
            if (first,k) not in task_values or (second,k) not in task_values:continue
            d=task_values[first,k]-task_values[second,k]
            samples=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
            low,high=np.quantile(samples,[.025,.975])
            contrasts.append(dict(first=first,second=second,k=k,mean_difference=float(d.mean()),ci_low=float(low),ci_high=float(high),tasks=len(d)))
            lines.append(f'| {first} − {second} | {k} | {d.mean():+.4f} | [{low:+.4f}, {high:+.4f}] |')
    lines+=['','## 학습 보상 추이','', '| 조건 | K | 첫 20 update 전체 보상 | 마지막 20 update 전체 보상 | 마지막 target entropy |','|---|---:|---:|---:|---:|']
    for case in sorted(root.glob('*_lambda*')):
        for k in (10,25,50):
            path=case/f'dual_k{k}.jsonl'
            if not path.exists(): continue
            records=[json.loads(line) for line in path.read_text().splitlines()]
            if not records: continue
            first=np.mean([r['mean_total_reward'] for r in records[:20]])
            last=np.mean([r['mean_total_reward'] for r in records[-20:]])
            entropy=np.mean([r['target_entropy'] for r in records[-20:]])
            lines.append(f'| {case.name} | {k} | {first:.3f} | {last:.3f} | {entropy:.3f} |')
    lines+=['','## 학습 안정성','']+[f'- {name}: recorded issues {count}' for name,count in stability.items()]
    lines+=['','전체 조건·상대 평균은 comparison.csv, 상대별 상세는 각 case/evaluation/summary.json, 원시 결과는 episodes.jsonl에 있다. 단일 seed·100 updates 파일럿이므로 수렴이나 일반적 우위를 주장할 수 없다.']
    (root/'contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('correction'):
            lines[2:2]=['비활성 head의 Adam momentum 업데이트를 차단한 수정 버전이다. Off-policy 9개 모델을 재학습·재평가했고, 영향을 받지 않는 PPO 9개 모델은 최초 실행에서 재사용했다.', '']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(root/'report.md')

if __name__=='__main__':main()
