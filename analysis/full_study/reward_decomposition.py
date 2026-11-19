"""Exact additive mean-reward decomposition for medium, opponent-independent bonus."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from meta_exploration.environment import generate_world
ROOT=Path('examples/full_study'); OUT=Path('analysis/full_study')
cache={};groups=defaultdict(list)
for p in ROOT.glob('*/seed_*/evaluation/metrics.json'):
    for r in json.loads(p.read_text()):
        if r['evaluation']!='online' or r['B']!=32 or not r['combo_applicable']: continue
        w_id=r['world_id']
        if w_id not in cache:
            w=generate_world(int(w_id.rsplit('-',1)[1]),r['world_type'],'test')
            assert np.count_nonzero(w.bonus.mean(axis=(1,3)))==1
            cache[w_id]=float(w.bonus.max())
        expected=(r['oracle_exploitation_return_from_agent_state']-r['oracle_exploitation_regret'])/r['K']
        bonus=.5*r['optimal_combo_utilization']*cache[w_id]
        groups[(r['regime'],r['world_type'],r['method'])].append((expected,expected-bonus,bonus))
result=[]
for (regime,typ,method),values in groups.items():
    v=np.mean(values,axis=0)
    result.append(dict(regime=regime,world_type=typ,method=method,expected_reward_mean=float(v[0]),base_component=float(v[1]),bonus_component=float(v[2])))
(OUT/'reward_decomposition.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result:
    if r['regime']=='A3':print(r)
