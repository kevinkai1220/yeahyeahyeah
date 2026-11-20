"""Additional observed-policy diagnostic: initial greedy action held constant.

No reward/hidden parameter selects the action. Each seed's Single policy has the
same initial observation on every world; argmax of its logged first probabilities
is repeated for all 160 steps. This is a NEW CPU evaluation, not an uploaded run.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from meta_exploration.training import world_set,make_game
from meta_exploration.plotting import bootstrap
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;DATA=ROOT/'examples/full_study_corrected_diagnostics'
records=json.loads((OUT/'world_metrics.json').read_text())
trajectories=json.loads((OUT/'trajectory_stats.json').read_text())
reference=[];groups=[];training=[];sample_checks=[]
for seed in range(5):
 run=DATA/f'A2/seed_{seed}';cfg=json.loads((run/'config.json').read_text())
 samples=[]
 for p in sorted((run/'evaluation/online_B32_K128').glob('*single_policy.jsonl')):
  h=[json.loads(x) for x in p.read_text().splitlines()];samples.append(h[0]['action_probs'])
 assert all(np.allclose(x,samples[0]) for x in samples)
 action=int(np.argmax(samples[0]));worlds=world_set(cfg,'A2',seed,'test')
 for w in worlds:
  game=make_game(w,160,seed*100000+w.seed,cfg)
  rewards=[];means=[]
  for t in range(160):
   o=game.observation();_,reward,_,b=game.step(action)
   if t>=32:
    rewards.append(reward);means.append(w.mean(o.mode,(o.previous_own,o.previous_other),action,b))
  reference.append(dict(seed=seed,world_id=w.world_id,action='ABC'[action],reward=float(np.mean(rewards)),expected_conditional_reward=float(np.mean(means))))
 for method in ['single_policy','dual_neural','dual_q_table']:
  for good in range(3):
   ids={w.world_id for w in worlds if np.unravel_index(w.bonus.mean(axis=(1,3)).argmax(),(3,3))[0]==good}
   rs=[r for r in records if r['regime']=='A2' and r['seed']==seed and r['method']==method and r['world_id'] in ids]
   groups.append(dict(seed=seed,method=method,preferred='ABC'[action],true_good='ABC'[good],n=len(rs),
    reward=float(np.mean([r['exploitation_mean'] for r in rs])),
    optimal_utilization=float(np.mean([r['optimal_combo_utilization'] for r in rs])),
    discovery=float(np.mean([r['discovery_rate'] for r in rs]))))
# Link recorded normalized episode-level explorer credit to observed downstream.
credit=[]
for run in sorted(DATA.glob('*/seed_*')):
 reg=run.parent.name;seed=int(run.name[5:]);logs=[json.loads(x) for x in (run/'training.jsonl').read_text().splitlines()]
 episodes={r['episode']:r for r in logs if r['stage']=='explorer'}
 cs=[]
 for r in logs:
  if r['stage']!='explorer_update':continue
  a=r['normalized_episode_advantage_means'];ret=[episodes[i]['downstream_return'] for i in r['rollout_episode_ids']]
  cs.append(float(np.corrcoef(a,ret)[0,1]))
 credit.append(dict(regime=reg,seed=seed,median_credit_return_correlation=float(np.median(cs))))
 for stage in ['pretraining','single_policy','dual_neural','explorer']:
  rs=[r for r in logs if r['stage']==stage]
  training.append(dict(regime=reg,seed=seed,stage=stage,first120=float(np.mean([r['return_mean'] for r in rs[:120]])),last120=float(np.mean([r['return_mean'] for r in rs[-120:]]))))
 for typ in ['type1','type2','type3']:
  full=[r for r in records if r['regime']==reg and r['seed']==seed and r['world_type']==typ and r['method']=='dual_neural']
  selected=[r for r in trajectories if r['regime']==reg and r['seed']==seed and r['world_type']==typ and r['method']=='dual_neural' and r['phase']=='suffix']
  if not selected:continue
  sample_checks.append(dict(regime=reg,seed=seed,world_type=typ,n_sample=len(selected),n_population=len(full),sample_discovery=float(np.mean([r['good_in_prefix'] for r in selected])),population_discovery=float(np.mean([r['rewarding_combo_attempted'] for r in full]))))
lookup={(r['seed'],r['world_id']):r for r in reference};comparisons=[]
for method in ['single_policy','dual_neural']:
 rs=[r for r in records if r['regime']=='A2' and r['method']==method]
 by_seed=defaultdict(list)
 for r in rs:by_seed[r['seed']].append(r['exploitation_mean']-lookup[r['seed'],r['world_id']]['reward'])
 m,l,h=bootstrap(by_seed,5000)
 comparisons.append(dict(method=method,contrast='neural minus fixed initial-greedy action',mean=float(m),ci_low=float(l),ci_high=float(h)))
summary=dict(reference_mean=float(np.mean([r['reward'] for r in reference])),reference_expected_conditional_mean=float(np.mean([r['expected_conditional_reward'] for r in reference])),comparisons=comparisons,
 reference_protocol='New CPU evaluation on identical worlds/noise/opponent seeds. Fixed action selected ONLY by Single policy initial-observation probability argmax. No history adaptation or oracle selection. Not a new trained policy.')
for name,data in [('fixed_action_reference',reference),('fixed_action_comparison',summary),('target_action_groups',groups),('credit_correlation',credit),('training_endpoints',training),('sample_selection_checks',sample_checks)]:
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
for method in ['single_policy','dual_neural','dual_q_table']:
 for matches in [True,False]:
  rs=[r for r in groups if r['method']==method and (r['preferred']==r['true_good'])==matches]
  print(method,'good==preferred',matches,'n',sum(r['n'] for r in rs),'optimal use',np.average([r['optimal_utilization'] for r in rs],weights=[r['n'] for r in rs]),'discovery',np.average([r['discovery'] for r in rs],weights=[r['n'] for r in rs]))
