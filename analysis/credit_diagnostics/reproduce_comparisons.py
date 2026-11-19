"""Recompute report contrasts from compact actual per-world records; prints JSON.

Run from repository root:
  python analysis/credit_diagnostics/reproduce_comparisons.py > /tmp/contrasts.json
"""
import json
from pathlib import Path
from collections import defaultdict
from meta_exploration.plotting import bootstrap


def estimate(groups, keys):
    results=[]
    for key, by_seed in groups.items():
        m,l,h=bootstrap(by_seed,1000)
        results.append(dict(zip(keys,key),mean=float(m),ci_low=float(l),ci_high=float(h),n_pairs=sum(map(len,by_seed.values()))))
    return results


def main():
    root=Path(__file__).parent
    history=[]
    for name in ['easy_history_interventions.json','staged_history_interventions.json']:
        history.extend(json.loads((root/name).read_text()))
    lookup={(r['stage'],r['variant'],r['seed'],r['world_id'],r['intervention']):r for r in history}
    groups=defaultdict(lambda:defaultdict(list))
    for r in history:
        if r['intervention']=='real':continue
        base=lookup[r['stage'],r['variant'],r['seed'],r['world_id'],'real']
        groups[r['stage'],r['variant'],'real minus '+r['intervention']][r['seed']].append(base['exploitation_mean']-r['exploitation_mean'])
    result={'history':estimate(groups,['stage','variant','contrast'])}
    speed=json.loads((root/'world_speed_K128.json').read_text())
    lookup={(r['method'],r['seed'],r['world_id'],r['B'],r['evaluation']):r for r in speed}
    groups=defaultdict(lambda:defaultdict(list))
    for r in speed:
        if r['B']==8 or r['evaluation']!='online':continue
        base=lookup[r['method'],r['seed'],r['world_id'],8,'online']
        if not r['absolute']['defined'] or not base['absolute']['defined']:continue
        a,b=r['absolute']['events']['0.9'],base['absolute']['events']['0.9']
        for metric,value in [('rmst_90',a['restricted_time']-b['restricted_time']),
                             ('total_interaction_rmst_90',r['B']-8+a['restricted_time']-b['restricted_time']),
                             ('reach_90',float(a['event'])-float(b['event']))]:
            groups[r['method'],r['B'],8,metric][r['seed']].append(value)
    result['budget']=estimate(groups,['method','B','reference_B','metric'])
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
