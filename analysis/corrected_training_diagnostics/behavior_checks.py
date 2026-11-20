import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from meta_exploration.training import world_set,make_game
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;DATA=ROOT/'examples/full_study_corrected_diagnostics'
records=json.loads((OUT/'world_metrics.json').read_text());refs=json.loads((OUT/'fixed_action_reference.json').read_text())
ref_index={(r['seed'],r['world_id']):r for r in refs}
decomp=[];behavior=[];replayed=0
for run in sorted(DATA.glob('*/seed_*')):
 reg=run.parent.name;seed=int(run.name[5:]);cfg=json.loads((run/'config.json').read_text());worlds={w.world_id:w for w in world_set(cfg,reg,seed,'test')}
 folder=run/'evaluation/online_B32_K128'
 for p in folder.glob('*dual_neural.jsonl'):
  history=[json.loads(x) for x in p.read_text().splitlines()];wid=history[0]['world_id'];w=worlds[wid]
  game=make_game(w,160,seed*100000+w.seed,cfg)
  for t in history:
   _,reward,done,b=game.step('ABC'.index(t['agent_action']))
   assert np.isclose(reward,t['reward'],atol=1e-10) and 'ABC'[b]==t['opponent_action'] and done==t['done']
   replayed+=1
 for r in [r for r in records if r['regime']==reg and r['seed']==seed]:
  w=worlds[r['world_id']]
  expected=(r['oracle_exploitation_return_from_agent_state']-r['oracle_exploitation_regret'])/128
  strength=float(w.bonus.max())
  # Medium worlds have exactly one positive pair, opponent-independent bonus.
  assert cfg['environment']['difficulty']=='medium' and cfg['environment']['opponent_dependence']==0
  bonus=0. if r['optimal_combo_utilization'] is None else r['optimal_combo_utilization']*strength/2
  decomp.append(dict(regime=reg,seed=seed,world_id=r['world_id'],world_type=r['world_type'],method=r['method'],expected_reward=expected,base=expected-bonus,bonus=bonus))
 if reg=='A2':
  metrics=json.loads((folder/'metrics.json').read_text())
  preferred=ref_index[seed,next(iter(worlds))]['action']
  for r in metrics:
   w=worlds[r['world_id']];good='ABC'[np.unravel_index(w.bonus.mean(axis=(1,3)).argmax(),(3,3))[0]]
   behavior.append(dict(seed=seed,world_id=r['world_id'],method=r['method'],preferred=preferred,good=good,
    matches=preferred==good,discovered=r['rewarding_combo_attempted'],supported=r['rewarding_combo_supported'],
    optimal_use=r['optimal_combo_utilization'],reward=r['exploitation_mean'],reach90=r['adaptation_speed']['absolute']['events']['0.9']['event']))
  for wid,w in worlds.items():
   ref=ref_index[seed,wid];a='ABC'.index(ref['action']);bonus=float(w.bonus[a,0,a,0])/2
   decomp.append(dict(regime='A2',seed=seed,world_id=wid,world_type='type2',method='fixed_initial_action',expected_reward=ref['expected_conditional_reward'],base=ref['expected_conditional_reward']-bonus,bonus=bonus))
summary=[]
for reg in ['A1','A2','A3','B']:
 for method in sorted({r['method'] for r in decomp if r['regime']==reg}):
  rs=[r for r in decomp if r['regime']==reg and r['method']==method]
  summary.append(dict(regime=reg,method=method,**{k:float(np.mean([r[k] for r in rs])) for k in ['expected_reward','base','bonus']}))
conditional=[]
for method in ['single_policy','dual_neural','dual_q_table']:
 for match in [True,False]:
  for filt in ['all','discovered','supported']:
   rs=[r for r in behavior if r['method']==method and r['matches']==match and (filt=='all' or r[filt])]
   conditional.append(dict(method=method,preferred_matches_optimal=match,condition=filt,n=len(rs),optimal_use=float(np.mean([r['optimal_use'] for r in rs])),reward=float(np.mean([r['reward'] for r in rs])),reach90=float(np.mean([r['reach90'] for r in rs]))))
for name,data in [('reward_decomposition',summary),('A2_conditional_behavior',conditional),('A2_world_behavior',behavior),('replay_audit',dict(verified_transitions=replayed,scope='Replay observed actions on regenerated world/opponent/reward RNG; diagnostic only, not new neural inference'))]:
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
print('REPLAY',replayed)
for r in summary:
 if r['regime'] in ['A2','A3']:print(r)
print('CONDITIONAL')
for r in conditional:
 if r['method']=='dual_neural':print(r)
