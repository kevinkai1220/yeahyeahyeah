import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze import paired_summary, REGIMES
from meta_exploration.environment import generate_world

ROOT=Path('examples/full_study');OUT=Path('analysis/full_study')
rows=[r for p in ROOT.glob('*/seed_*/evaluation/metrics.json') for r in json.loads(p.read_text()) if r['evaluation']=='online']
lookup={(r['regime'],r['seed'],r['world_id'],r['B'],r['method']):r for r in rows}
result={}
budgets=[]
for regime in REGIMES:
    for B in [8,16,32,64]:
        selected=[r for r in rows if r['regime']==regime and r['B']==B and r['method']=='dual_q_table']
        for metric in ['exploitation_mean','total_return','auc_128']:
            pairs=[(r['seed'],r[metric]-lookup[(regime,r['seed'],r['world_id'],B,'q_learning')][metric]) for r in selected]
            budgets.append(dict(regime=regime,B=B,metric=metric,**paired_summary(pairs)))
result['dual_q_minus_q_by_budget']=budgets
seed_bias=[]
for seed in range(5):
    selected=[r for r in rows if r['regime']=='A2' and r['seed']==seed and r['B']==32 and r['method']=='dual_neural']
    counts=defaultdict(int);by_combo=defaultdict(list)
    for r in selected:
        for k,v in r['combo_attempt_counts'].items():counts[k]+=v
        w=generate_world(int(r['world_id'].rsplit('-',1)[1]),'type2','test')
        pair=np.unravel_index(w.bonus.mean(axis=(1,3)).argmax(),(3,3))
        by_combo['ABC'[pair[0]]+'ABC'[pair[1]]].append(r['optimal_combo_utilization'])
    found=[r for r in selected if r['rewarding_combo_attempted']]
    seed_bias.append(dict(seed=seed,exploration_pair_frequencies={k:v/sum(counts.values()) for k,v in counts.items()},
         optimal_utilization_by_true_pair={k:float(np.mean(v)) for k,v in by_combo.items()},
         discovery=float(np.mean([r['rewarding_combo_attempted'] for r in selected])),
         discovered_neural_utilization=float(np.mean([r['discovered_combo_utilization'] for r in found])),
         discovered_q_utilization=float(np.mean([lookup[('A2',seed,r['world_id'],32,'dual_q_table')]['discovered_combo_utilization'] for r in found]))))
result['type2_seed_bias']=seed_bias
(OUT/'deep_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print('B64 Type3:',[r for r in budgets if r['regime']=='A3' and r['B']==64])
print('TYPE2 SEED BIAS',json.dumps(seed_bias,indent=2))
