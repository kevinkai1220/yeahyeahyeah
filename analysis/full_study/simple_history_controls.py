"""Exploratory diagnostic controls: hidden-rule-free prefix collectors, same Dual-Q suffix."""
import json
import time
from pathlib import Path
import numpy as np
from analyze import paired_summary, REGIMES
from meta_exploration.environment import generate_world
from meta_exploration.learning import QLearner, collect
from meta_exploration.training import make_game, HistoryCollector

ROOT=Path('examples/full_study');OUT=Path('analysis/full_study')
config=json.loads((ROOT/'A1/seed_0/config.json').read_text())
selected=[r for p in ROOT.glob('*/seed_*/evaluation/metrics.json') for r in json.loads(p.read_text())
          if r['evaluation']=='online' and r['B']==32 and r['K']==128 and r['method']=='dual_q_table']
cache={};rows=[];start=time.perf_counter()
for r in selected:
    world_id=r['world_id']
    if world_id not in cache:
        seed=int(world_id.rsplit('-',1)[1]);s=r['seed']*100000+seed
        w=generate_world(seed,r['world_type'],'test')
        outcomes={}
        for strategy in ['random','coverage','repeat']:
            game=make_game(w,160,s,config)
            prefix=collect(game,HistoryCollector(s,strategy),32)
            q=QLearner(config['q_learning'],s+10000);q.replay(prefix)
            suffix=collect(game,q,128,online_q=True)
            outcomes[strategy]=dict(exploitation_mean=float(np.mean([t.reward for t in suffix])),
                                   total_return=float(sum(t.reward for t in prefix+suffix)))
        cache[world_id]=outcomes
    rows.append(dict(regime=r['regime'],seed=r['seed'],world_id=world_id,
                     learned_exploitation_mean=r['exploitation_mean'],learned_total_return=r['total_return'],**cache[world_id]))
summary=[]
for regime in REGIMES:
    selected=[r for r in rows if r['regime']==regime]
    for strategy in ['random','coverage','repeat']:
        for metric,baseline in [('exploitation_mean','learned_exploitation_mean'),('total_return','learned_total_return')]:
            summary.append(dict(regime=regime,strategy=strategy,metric=metric,
                 control_mean=float(np.mean([r[strategy][metric] for r in selected])),
                 learned_mean=float(np.mean([r[baseline] for r in selected])),
                 **paired_summary([(r['seed'],r[strategy][metric]-r[baseline]) for r in selected])))
payload=dict(protocol='New exploratory CPU controls, separate from uploaded results. Random actions; coverage scheduling all nine ordered pairs; one random action repeated throughout B. Each uses identical fresh Q replay, suffix seed, gamma/epsilon and B32/K128 budget as learned Dual-Q. Collectors never see world parameters/type.',
             unique_worlds=len(cache),runtime_seconds=time.perf_counter()-start,
             summary=summary,world_results=rows)
(OUT/'simple_history_controls.json').write_text(json.dumps(payload,indent=2)+'\n')
for r in summary:
    if r['metric']=='exploitation_mean':print(r)
