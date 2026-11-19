"""Additional observed-only control, clearly separate from uploaded experiment.

Q epsilon-greedy collects B with online updates. A FRESH Q table then receives
exactly the same replay/exploitation setup as the published Dual-Q method.
No hidden parameters enter either learner. Regenerated worlds are environments.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
from analyze import paired_summary, REGIMES
from meta_exploration.environment import generate_world
from meta_exploration.learning import QLearner, collect
from meta_exploration.training import make_game

ROOT=Path('examples/full_study'); OUT=Path('analysis/full_study')
config=json.loads((ROOT/'A1/seed_0/config.json').read_text())
selected=[r for p in ROOT.glob('*/seed_*/evaluation/metrics.json') for r in json.loads(p.read_text())
          if r['evaluation']=='online' and r['B']==32 and r['K']==128 and r['method']=='q_learning']
all_dual={}
for p in ROOT.glob('*/seed_*/evaluation/metrics.json'):
    for r in json.loads(p.read_text()):
        if r['evaluation']=='online' and r['B']==32 and r['method']=='dual_q_table':
            all_dual[(r['regime'],r['seed'],r['world_id'])]=r
cache={};result=[];start=time.perf_counter()
for r in selected:
    key=r['world_id']
    if key not in cache:
        seed=int(key.rsplit('-',1)[1]); s=r['seed']*100000+seed
        w=generate_world(seed,r['world_type'],'test')
        game=make_game(w,160,s,config)
        q=QLearner(config['q_learning'],s)
        prefix=collect(game,q,32,online_q=True)
        assert abs(sum(t.reward for t in prefix)-r['auc_32'])<1e-9
        # Reproduce the uploaded Q-only K-window before evaluating the control.
        baseline=collect(game.snapshot(),copy.deepcopy(q),128,online_q=True)
        assert abs(np.mean([t.reward for t in baseline])-r['exploitation_mean'])<1e-10
        fresh=QLearner(config['q_learning'],s+10000); fresh.replay(prefix)
        suffix=collect(game.snapshot(),fresh,128,online_q=True)
        cache[key]=dict(exploitation_mean=float(np.mean([t.reward for t in suffix])),
                        total_return=float(sum(t.reward for t in prefix+suffix)))
    result.append(dict(regime=r['regime'],seed=r['seed'],world_id=key,world_type=r['world_type'],
                       uploaded_q_only=r['exploitation_mean'],
                       uploaded_dual_q=all_dual[(r['regime'],r['seed'],key)]['exploitation_mean'],**cache[key]))
summary=[]
for regime in REGIMES:
    group=[r for r in result if r['regime']==regime]
    for baseline in ['uploaded_q_only','uploaded_dual_q']:
        summary.append(dict(regime=regime,contrast='Q-history fresh replay minus '+baseline,
                            control_mean=float(np.mean([r['exploitation_mean'] for r in group])),
                            baseline_mean=float(np.mean([r[baseline] for r in group])),
                            **paired_summary([(r['seed'],r['exploitation_mean']-r[baseline]) for r in group])))
payload=dict(protocol='New CPU control; B=32 K=128; Q epsilon-greedy prefix with online TD; fresh-Q replay and suffix exactly match Dual-Q settings, including suffix action seed; not part of uploaded main results.',
             unique_worlds=len(cache),q_only_exact_reproduction_checks=len(cache),
             runtime_seconds=time.perf_counter()-start,
             extra_table_updates_during_prefix=32,
             summary=summary,world_results=result)
(OUT/'q_history_control.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:v for k,v in payload.items() if k!='world_results'},indent=2))
