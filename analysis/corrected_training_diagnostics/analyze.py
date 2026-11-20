"""Audit added run logs and observed trajectories, without retraining models."""
import json, csv, math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from meta_exploration.plotting import bootstrap, pyplot, save, COLORS
from meta_exploration.training import world_set

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
DATA=ROOT/'examples/full_study_corrected_diagnostics'
aggregates=json.loads((ROOT/'examples/full_study_corrected_v1/figures/aggregated.json').read_text())
records=[];train_stats=[];trajectories=[];configs=[];costs=[];update_records=[];episode_curves=[]
audit=Counter();sample_selection=[]
for run in sorted(DATA.glob('*/seed_*')):
 reg=run.parent.name;seed=int(run.name[5:]);cfg=json.loads((run/'config.json').read_text());configs.append(cfg)
 cost=json.loads((run/'costs.json').read_text());costs.append(dict(regime=reg,seed=seed,**cost))
 assert json.loads((run/'complete.json').read_text())['status']=='complete';audit['complete_runs']+=1
 logs=[json.loads(x) for x in (run/'training.jsonl').read_text().splitlines()]
 assert len(logs)==6788
 assert len({(r['stage'],r['episode']) for r in logs})==len(logs)
 for stage in ['pretraining','single_policy','dual_neural','explorer','q_learning','dual_q_table']:
  rs=[r for r in logs if r['stage']==stage];expected=300 if stage=='pretraining' else 1200
  assert [r['episode'] for r in rs]==list(range(expected))
  episode_curves.append(dict(regime=reg,seed=seed,stage=stage,returns=[r['return_mean'] for r in rs]))
 for stage in ['pretraining_update','single_policy_update','dual_neural_update','explorer_update']:
  rs=[r for r in logs if r['stage']==stage];expected=38 if stage=='pretraining_update' else 150
  assert len(rs)==expected
  start=38 if stage in ['single_policy_update','dual_neural_update'] else 0
  ids=[]
  for i,r in enumerate(rs):
   assert r['rollout_policy_version']==start+i and r['policy_version']==start+i+1
   assert r['normalization']=='batch'
   assert abs(r['normalized_advantage_mean'])<1e-5
   assert abs(r['normalized_advantage_std']-1)<1e-5
   assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
   assert r['parameter_delta_norm']>0
   assert len(r['rollout_episode_ids'])==r['episodes']
   ids.extend(r['rollout_episode_ids']);audit['valid_updates']+=1
   update_records.append(dict(regime=reg,seed=seed,**r))
  assert ids==list(range(300 if stage=='pretraining_update' else 1200))
  costkey={'pretraining_update':'pretraining_optimizer_steps','single_policy_update':'single_optimizer_steps','dual_neural_update':'dual_exploiter_optimizer_steps','explorer_update':'extra_explorer_optimizer_steps'}[stage]
  assert sum(r['updates'] for r in rs)==cost[costkey]
  for label,section in [('all',rs),('first20',rs[:20]),('last20',rs[-20:])]:
   values={k:float(np.median([r[k] for r in section])) for k in ['entropy','actor_gradient_norm','critic_gradient_norm','gradient_norm','approximate_kl','clip_fraction','value_explained_variance','critic_loss','parameter_delta_norm']}
   values.update(critic_actor_ratio_median=float(np.median([r['critic_gradient_norm']/max(r['actor_gradient_norm'],1e-12) for r in section])),
    fraction_batch_mean_gradient_above_clip=float(np.mean([r['gradient_norm']>.5 for r in section])),
    fraction_batch_mean_KL_above_002=float(np.mean([r['approximate_kl']>.02 for r in section])),
    episode_advantage_mean_std=float(np.median([np.std(r['normalized_episode_advantage_means']) for r in section])))
   train_stats.append(dict(regime=reg,seed=seed,stage=stage,window=label,**values))
 metric_file=run/'evaluation/online_B32_K128/metrics.json'
 metrics=json.loads(metric_file.read_text());assert len(metrics)==400
 worlds={w.world_id:w for w in world_set(cfg,reg,seed,'test')}
 metric_index={(r['world_id'],r['method']):r for r in metrics}
 for r in metrics:
  d={k:v for k,v in r.items() if k not in ('adaptation_speed','adaptation_compute')}
  for norm in ['absolute','baseline_normalized']:
   sp=r['adaptation_speed'][norm]
   d[norm+'_auc_mean']=sp.get('normalized_adaptation_auc_mean')
  records.append(d)
  assert r['world_id'] in worlds
 for wid in worlds:
  a,b=[metric_index[wid,m] for m in ['dual_neural','dual_q_table']]
  assert a['combo_attempt_counts']==b['combo_attempt_counts']
  assert a['auc_32']==b['auc_32'];audit['paired_prefix_metric_matches']+=1
 folder=metric_file.parent
 sample_ids=set()
 for file in sorted(folder.glob('*.jsonl')):
  h=[json.loads(x) for x in file.read_text().splitlines()];r=h[0];method=r['method'];wid=r['world_id'];sample_ids.add(wid)
  w=worlds[wid];metric=metric_index[wid,method]
  assert len(h)==160 and [t['step'] for t in h]==list(range(160))
  assert np.isclose(np.mean([t['reward'] for t in h[32:]]),metric['exploitation_mean'])
  for p,t in zip(h,h[1:]):
   assert p['agent_action']==t['previous_agent_action'] and p['opponent_action']==t['previous_opponent_action']
   assert p['mode']!=t['mode'] and not p['done']
  assert h[-1]['done'];audit['raw_histories_checked']+=1
  if method=='dual_neural':
   twin=[json.loads(x) for x in (folder/f'{wid}_dual_q_table.jsonl').read_text().splitlines()]
   assert [{k:v for k,v in t.items() if k!='method'} for t in h[:32]]==[{k:v for k,v in t.items() if k!='method'} for t in twin[:32]];audit['raw_dual_prefix_matches']+=1
  for phase,part in [('prefix',h[:32]),('suffix',h[32:])]:
   actions=[t['agent_action'] for t in part];pairs=[t['observed_combo'] for t in part if t['mode']==2]
   counts=Counter(actions);pair_counts=Counter(pairs)
   entropies=[-sum(p*np.log(max(p,1e-30)) for p in t['action_probs']) for t in part]
   probabilities=np.array([t['action_probs'] for t in part])
   means=[];bonuses=[];bases=[];good_uses=[]
   for t in part:
    a,b='ABC'.index(t['agent_action']),'ABC'.index(t['opponent_action'])
    prev=(3,3) if t['previous_agent_action'] is None else ('ABC'.index(t['previous_agent_action']),'ABC'.index(t['previous_opponent_action']))
    base=w.base[a,b];bonus=w.bonus[*prev,a,b] if t['mode']==2 else 0.
    bases.append(float(base));bonuses.append(float(bonus));means.append(float(base+bonus))
   same=sum(p[0]==p[1] for p in pairs)/len(pairs)
   good=np.unravel_index(w.bonus.mean(axis=(1,3)).argmax(),(3,3)) if w.world_type!='type1' else None
   dominant=counts.most_common(1)[0][0]
   trajectories.append(dict(regime=reg,seed=seed,world_id=wid,world_type=w.world_type,method=method,phase=phase,
    dominant_action=dominant,dominant_fraction=counts[dominant]/len(part),action_counts=dict(counts),pair_counts=dict(pair_counts),
    same_action_pair_fraction=same,unique_pairs=len(pair_counts),entropy=float(np.mean(entropies)),
    max_action_probability=float(np.mean(probabilities.max(axis=1))),first_action_probs=probabilities[0].tolist(),
    mean_base=float(np.mean(bases)),mean_bonus=float(np.mean(bonuses)),mean_reward=float(np.mean([t['reward'] for t in part])),
    good_combo=''.join('ABC'[i] for i in good) if good else None,
    good_in_prefix=metric['rewarding_combo_attempted'],optimal_utilization=metric['optimal_combo_utilization']))
 for wid in sample_ids:
  row=metric_index[wid,'dual_neural'];sample_selection.append(dict(regime=reg,seed=seed,world_id=wid,discovered=row['rewarding_combo_attempted'],reward=row['exploitation_mean']))
 del metrics,logs
assert all(c==configs[0] for c in configs)
audit['all_configs_equal']=True;audit['world_metric_rows']=len(records)
# Raw-to-aggregate mean and standard-deviation verification, main online window.
for a in aggregates:
 if a['B']!=32 or a['evaluation']!='online':continue
 rs=[r for r in records if r['regime']==a['regime'] and r['method']==a['method'] and (a['world_type']=='all' or r['world_type']==a['world_type']) and r.get(a['metric']) is not None]
 if not rs:continue
 by_seed=defaultdict(list)
 for r in rs:by_seed[r['seed']].append(float(r[a['metric']]))
 assert np.isclose(np.mean([np.mean(v) for v in by_seed.values()]),a['mean'])
 assert np.isclose(np.std([r[a['metric']] for r in rs]),a['std']);audit['aggregate_checks']+=1
# Paired contrasts, on matching worlds; legacy side has raw main records.
legacy={}
for p in (ROOT/'examples/full_study').glob('*/seed_*/evaluation/metrics.json'):
 for r in json.loads(p.read_text()):
  if r['B']==32 and r['K']==128 and r['evaluation']=='online':legacy[r['regime'],r['seed'],r['world_id'],r['method']]=r
lookup={(r['regime'],r['seed'],r['world_id'],r['method']):r for r in records}
contrasts=[]
def contrast(name,reg,left,right,metric='exploitation_mean'):
 by_seed=defaultdict(list)
 for r in left:
  key=r['seed'],r['world_id']
  if key not in right or r.get(metric) is None or right[key].get(metric) is None:continue
  by_seed[r['seed']].append(r[metric]-right[key][metric])
 if not by_seed:return
 m,l,h=bootstrap(by_seed,5000)
 contrasts.append(dict(regime=reg,contrast=name,metric=metric,mean=float(m),ci_low=float(l),ci_high=float(h),n_pairs=sum(map(len,by_seed.values()))))
for reg in ['A1','A2','A3','B']:
 for method in ['single_policy','dual_neural','q_learning','dual_q_table']:
  rs=[r for r in records if r['regime']==reg and r['method']==method]
  old={(s,w):v for (g,s,w,m),v in legacy.items() if g==reg and m==method}
  contrast(method+': corrected - original',reg,rs,old)
 for left,right in [('dual_neural','single_policy'),('dual_neural','dual_q_table'),('dual_q_table','q_learning')]:
  for metric in ['exploitation_mean','total_return']:
   a=[r for r in records if r['regime']==reg and r['method']==left]
   b={(r['seed'],r['world_id']):r for r in records if r['regime']==reg and r['method']==right}
   contrast(left+' - '+right,reg,a,b,metric)
# Mixed versus specialized matched subset.
for typ,reg in [('type1','A1'),('type2','A2'),('type3','A3')]:
 for method in ['single_policy','dual_neural','q_learning','dual_q_table']:
  a=[r for r in records if r['regime']=='B' and r['world_type']==typ and r['method']==method]
  b={(r['seed'],r['world_id']):r for r in records if r['regime']==reg and r['method']==method}
  contrast('mixed - specialized: '+method,typ,a,b)
seeds=[]
for reg in ['A1','A2','A3','B']:
 for seed in range(5):
  for method in ['single_policy','dual_neural','q_learning','dual_q_table']:
   rs=[r for r in records if r['regime']==reg and r['seed']==seed and r['method']==method]
   seeds.append(dict(regime=reg,seed=seed,method=method,reward=float(np.mean([r['exploitation_mean'] for r in rs])),
    optimal_utilization=float(np.mean([r['optimal_combo_utilization'] for r in rs if r['optimal_combo_utilization'] is not None])) if any(r['optimal_combo_utilization'] is not None for r in rs) else None))
for name,values in [('audit',dict(audit)),('training_stats',train_stats),('seed_results',seeds),('paired_contrasts',contrasts),('trajectory_stats',trajectories),('sample_selection',sample_selection),('costs',costs),('episode_curves',episode_curves)]:
 (OUT/(name+'.json')).write_text(json.dumps(values,indent=2,allow_nan=False))
# Compact all-world records support further diagnostics without giant curves.
(OUT/'world_metrics.json').write_text(json.dumps(records,allow_nan=False))
# Per-policy training curves and gradient diagnostics. Color=policy, line=seed.
plt=pyplot()
for reg in ['A1','A2','A3','B']:
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for stage,color in [('single_policy',COLORS['single_policy']),('dual_neural',COLORS['dual_neural']),('explorer','#882255')]:
  for seed in range(5):
   curve=next(r for r in episode_curves if r['regime']==reg and r['seed']==seed and r['stage']==stage)
   y=np.asarray(curve['returns']).reshape(-1,60).mean(axis=1)
   axes[0,0].plot(np.arange(60,1201,60),y,color=color,alpha=.6,label=f'{stage}/s{seed}')
   updates=[r for r in update_records if r['regime']==reg and r['seed']==seed and r['stage']==stage+'_update']
   for ax,metric in zip(list(axes.flat)[1:],['entropy','critic_gradient_norm','actor_gradient_norm','value_explained_variance','approximate_kl']):
    ax.plot([r['episode'] for r in updates],[r[metric] for r in updates],color=color,alpha=.4)
 axes[0,0].set_ylabel('Downstream reward (60-episode blocks)')
 for ax,metric in zip(list(axes.flat)[1:],['entropy','critic_gradient_norm','actor_gradient_norm','value_explained_variance','approximate_kl']):ax.set_ylabel(metric)
 axes[0,2].set_yscale('log');axes[1,0].set_yscale('log')
 for ax in axes.flat:ax.set_xlabel('Training episode');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=5,ncol=3);fig.suptitle(reg);fig.tight_layout();save(fig,OUT/(reg+'_training'))
print(json.dumps(dict(audit),indent=2))
print('CONFIG verified and log stages/versions/normalization/optimizer budgets checked.')
