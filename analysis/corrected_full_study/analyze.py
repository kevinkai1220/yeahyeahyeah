"""Analyze uploaded aggregate artifacts only; never reconstruct missing raw data."""
import json, math, csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
old=json.loads((ROOT/'examples/full_study/figures/aggregated.json').read_text())
new=json.loads((ROOT/'examples/full_study_corrected_v1/figures/aggregated.json').read_text())
speed=json.loads((ROOT/'examples/full_study_corrected_speed_v1/aggregated_speed.json').read_text())
paired=json.loads((ROOT/'examples/full_study_corrected_speed_v1/paired_speed.json').read_text())
keys=('regime','evaluation','world_type','B','K','method','metric')
def index(data, extra=()):
 out={tuple(r[k] for k in keys+extra):r for r in data}
 assert len(out)==len(data)
 return out
old_idx,new_idx=index(old),index(new)
speed_idx=index(speed,('normalization',))
def get(data,reg,method,metric,B=32,mode='online',typ='all',norm=None):
 return next(r for r in data if r['regime']==reg and r['method']==method and r['metric']==metric and r['B']==B and r['K']==128 and r['evaluation']==mode and r['world_type']==typ and (norm is None or r['normalization']==norm))
# Unchanged baseline is a consistency check, NOT raw-world identity proof.
qchecks=[]
for key,r in new_idx.items():
 if r['method']=='q_learning' and key in old_idx:
  qchecks.append(all(r[k]==old_idx[key][k] for k in ['mean','std','worst','n_worlds','n_seeds']))
assert all(qchecks)
# Verify reported speed estimates against survival summaries.
survival_checks=0
for p in (ROOT/'examples/full_study_corrected_speed_v1').glob('survival_*.json'):
 parts=p.stem[len('survival_'):].split('_')
 reg,mode,typ,B,K=parts[:5];norm='_'.join(parts[5:]);B=int(B);K=int(K)
 for r in json.loads(p.read_text()):
  curve=np.asarray(r['reach_curve']);surv=np.asarray(r['survival'])
  assert len(curve)==K and np.all(np.diff(curve)>=-1e-12)
  assert np.allclose(curve+surv,1)
  q=f"{r['threshold']:g}"
  assert np.isclose(curve[-1],get(speed,reg,r['method'],'reach_at_K_'+q,B,mode,typ,norm)['mean'])
  assert np.isclose(1+surv[:-1].sum(),get(speed,reg,r['method'],'rmst_'+q,B,mode,typ,norm)['mean'])
  survival_checks+=1
for r in speed:
 assert math.isfinite(r['mean']) and r['ci_low']<=r['ci_high']
 if r['metric']=='normalized_adaptation_auc':
  mean=get(speed,r['regime'],r['method'],'normalized_adaptation_auc_mean',r['B'],r['evaluation'],r['world_type'],r['normalization'])
  assert np.isclose(r['mean'],113*mean['mean'])
# Prefix discovery is shared between dual branches.
prefix_checks=0
for r in new:
 if r['method']=='dual_neural' and r['metric']=='discovery_rate':
  q=get(new,r['regime'],'dual_q_table','discovery_rate',r['B'],r['evaluation'],r['world_type'])
  assert r['mean']==q['mean'];prefix_checks+=1
comparison=[]
for reg in ['A1','A2','A3','B']:
 for method in ['single_policy','q_learning','dual_neural','dual_q_table']:
  a,b=get(old,reg,method,'exploitation_mean'),get(new,reg,method,'exploitation_mean')
  total=get(new,reg,method,'total_return')['mean']
  comparison.append(dict(regime=reg,method=method,old=a['mean'],new=b['mean'],difference=b['mean']-a['mean'],ci_low=b['ci_low'],ci_high=b['ci_high'],prefix_mean=(total-128*b['mean'])/32))
with (OUT/'reward_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');w.writeheader();w.writerows(comparison)
audit=dict(aggregate_rows=len(new),speed_rows=len(speed),paired_rows=len(paired),
 unchanged_q_summary_checks=len(qchecks),survival_consistency_checks=survival_checks,dual_prefix_summary_checks=prefix_checks,
 nominal_worlds_per_regime=500,nominal_seeds=5,
 limitation='Uploaded aggregates only; no corrected config, per-world metrics, trajectories, checkpoints, training/runtime logs. No new paired old/new test or seed reconstruction.',
 csv_literal_escape=bool(b'\\r\\n' in (ROOT/'examples/full_study_corrected_v1/figures/aggregated.csv').read_bytes()))
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
(OUT/'main_speed.json').write_text(json.dumps([r for r in speed if r['B']==32 and r['world_type']=='all' and r['evaluation']=='online'],indent=2))
(OUT/'paired_main.json').write_text(json.dumps([r for r in paired if r['B']==32 and r['world_type']=='all' and r['evaluation']=='online'],indent=2))
# Readable overview from genuine uploaded summaries.
from meta_exploration.plotting import pyplot,save,COLORS
plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(12,8))
methods=['single_policy','dual_neural','q_learning','dual_q_table']
for ax,reg in zip(axes.flat,['A1','A2','A3','B']):
 for i,m in enumerate(methods):
  a,b=get(old,reg,m,'exploitation_mean'),get(new,reg,m,'exploitation_mean')
  ax.bar(i-.17,a['mean'],width=.32,color=COLORS[m],alpha=.3)
  ax.bar(i+.17,b['mean'],width=.32,color=COLORS[m])
  ax.errorbar(i+.17,b['mean'],yerr=[[max(0,b['mean']-b['ci_low'])],[max(0,b['ci_high']-b['mean'])]],fmt='none',color='black')
 ax.set_xticks(range(4),['Single','Dual NN','Q-only','Dual Q']);ax.set(title=reg+' — pale: original; solid: corrected',ylabel='Exploitation mean reward')
fig.tight_layout();save(fig,OUT/'reward_overview')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for reg,style in [('A2','-'),('A3','--')]:
 for m in ['dual_neural','dual_q_table']:
  Bs=[8,16,32,64]
  axes[0].plot(Bs,[get(new,reg,m,'exploitation_mean',B)['mean'] for B in Bs],style,marker='o',color=COLORS[m],label=reg+'/'+m)
  axes[1].plot(Bs,[get(speed,reg,m,'total_interaction_rmst_0.9',B,norm='absolute')['mean'] for B in Bs],style,marker='o',color=COLORS[m],label=reg+'/'+m)
axes[0].set(xlabel='B',ylabel='Exploitation mean reward');axes[1].set(xlabel='B',ylabel='B + RMST to 90% oracle (lower is better)')
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();save(fig,OUT/'budget_tradeoff')
print(json.dumps(audit,indent=2))
print('A2 prefix means',[(r['method'],r['prefix_mean']) for r in comparison if r['regime']=='A2'])
