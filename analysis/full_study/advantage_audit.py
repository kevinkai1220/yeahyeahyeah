"""Counterexample using the actual targets() and ppo_update normalization formula.
This is an implementation diagnostic, not a new environment performance result.
"""
import json
from pathlib import Path
import numpy as np
import torch
from meta_exploration.config import load_config
from meta_exploration.environment import Observation
from meta_exploration.learning import Transition, targets

c=load_config();B=c['training']['exploration_steps'];K=c['training']['exploitation_steps']
def advantages(suffix_reward, lam):
    c['training']['gae_lambda']=lam
    h=[]
    previous=0.
    for t in range(B+K):
        reward=0. if t<B else suffix_reward
        o=Observation(t%2,0,0,previous)
        h.append(Transition(o,0,reward,Observation((t+1)%2,0,0,reward),t==B+K-1,0,{'value':0.}))
        previous=reward
    return torch.tensor(targets(h,c)[0][:B])
def normalize(x):return (x-x.mean())/(x.std(unbiased=False)+1e-8)
rows=[]
for lam in [1.,.95]:
    low,high=advantages(1.,lam),advantages(10.,lam)
    joint=normalize(torch.cat((low,high)))
    row=dict(gae_lambda=lam,low_downstream_return=float(K),high_downstream_return=float(10*K),
        raw_advantage_means=[float(low.mean()),float(high.mean())],
        per_episode_normalized_means=[float(normalize(low).mean()),float(normalize(high).mean())],
        max_difference_after_per_episode_normalization=float((normalize(low)-normalize(high)).abs().max()),
        per_episode_normalized_norms=[float(normalize(low).norm()),float(normalize(high).norm())],
        pooled_batch_normalized_episode_means=[float(joint[:B].mean()),float(joint[B:].mean())])
    if lam==1.:assert row['per_episode_normalized_norms']==[0.,0.]
    else:assert row['max_difference_after_per_episode_normalization']<1e-5
    assert row['pooled_batch_normalized_episode_means'][0]<0<row['pooled_batch_normalized_episode_means'][1]
    rows.append(row)
result=dict(note='Diagnostic: zero prefix rewards, zero critic values; suffix reward 1 vs 10. Uses production targets and exact per-segment normalization formula. Main GAE=.95 retains some temporal/critic-dependent signal in general; this counterexample does not prove all downstream credit vanishes in real runs.',results=rows)
Path('analysis/full_study/advantage_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
