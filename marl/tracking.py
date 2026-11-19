"""Read-only episode tracking. Combo labels are evaluator-only, never policy inputs."""
import gzip
import json
from pathlib import Path
import numpy as np
from .batched import collect_batch


def describe_episode(task, trajectory, rewards, opponent_actions, trace, index, k):
    if task.combos is None:
        raise ValueError('Combo tracking requires evaluator metadata')
    u,v=map(int,task.combos[0]);actions=trajectory.actions;horizon=len(actions)
    hits=[t for t in range(1,horizon,2) if actions[t-1]==u and actions[t]==v]
    first=hits[0] if hits else None
    repeat=hits[1] if len(hits)>1 else None
    steps=[]
    for t,a in enumerate(actions):
        probs=trace[t]['policy_probabilities'][index]
        behavior=trace[t]['behavior_probabilities'][index]
        steps.append(dict(round=t+1,mode='R1' if t%2==0 else 'R2',
            phase='explore' if t<k else 'exploit',active_head=trace[t]['head'],
            cumulative_reward=float(rewards[:t+1,0].sum()),
            rolling_10_round_reward=float(rewards[max(0,t-9):t+1,0].mean()),
            pair_reward=float(rewards[t-1:t+1,0].sum()) if t%2 else None,
            action=int(a),opponent_action=int(opponent_actions[t]),reward=float(rewards[t,0]),
            behavior_probabilities=behavior.tolist(),target_probabilities=probs[0].tolist(),
            exploration_probabilities=probs[1].tolist() if len(probs)>1 else None,
            behavior_entropy=float(-(behavior*np.log(np.maximum(behavior,1e-30))).sum()),
            first_discovery=t==first,combo_completed=t in hits,
            previously_discovered=first is not None and t>first,
            target_correct_action_probability=float(probs[0,u if t%2==0 else v]),
            completion_context=bool(t%2 and actions[t-1]==u),
            completed_pair=[int(actions[t-1]),int(a)] if t%2 else None))
    subsequent=[t for t in range(1,horizon,2) if first is not None and t-1>first]
    exploit_pairs=[t for t in subsequent if t-1>=k]
    # Compare target's R1 prefix probability before first success with the next R1.
    # The change is descriptive: conditioning on first success is selection-biased.
    shift=None
    if first is not None and first+1<horizon:
        shift=float(trace[first+1]['policy_probabilities'][index,0,u]-trace[first-1]['policy_probabilities'][index,0,u])
    episode=dict(combo=[u,v],first_combo_round=None if first is None else first+1,
        first_combo_censored=first is None,first_repeat_round=None if repeat is None else repeat+1,
        repeat_delay_rounds=None if repeat is None else repeat-first,
        remaining_rounds_after_discovery=None if first is None else horizon-first-1,
        discovered_by_k=first is not None and first+1<=k,
        post_discovery_pairs=len(subsequent),post_discovery_hits=sum(t in hits for t in subsequent),
        exploit_reuse_pairs=len(exploit_pairs),exploit_reuse_hits=sum(t in hits for t in exploit_pairs),
        next_pair_reused=None if not subsequent else subsequent[0] in hits,
        target_prefix_probability_shift=shift,total_reward=float(rewards[:,0].sum()),
        exploitation_mean=float(rewards[k:,0].mean()),
        unique_pairs_by_k=len({(int(actions[t-1]),int(actions[t])) for t in range(1,k,2)}))
    return episode,steps


def aggregate(episodes,step_sets,horizon,k):
    first=[e['first_combo_round'] for e in episodes];n=len(first)
    cdf=[dict(round=t,discovery_rate=sum(x is not None and x<=t for x in first)/n,
              at_risk_before_round=sum(x is None or x>=t for x in first),
              new_discoveries=sum(x==t for x in first)) for t in range(1,horizon+1)]
    lags=[]
    for lag in range(1,min(10,horizon//2)+1):
        pool=[]
        for e,steps in zip(episodes,step_sets):
            if e['first_combo_round'] is None:continue
            # First complete pair strictly after the discovery is lag 1.
            end=e['first_combo_round']-1+2*lag
            if end>=horizon:continue
            pool.append((steps[end-1],steps[end]))
        eligible=[p for p in pool if p[0]['phase']=='exploit']
        lags.append(dict(pair_lag=lag,episodes=len(pool),
            combo_reuse_rate=float(np.mean([b['combo_completed'] for a,b in pool])) if pool else None,
            pair_reward_mean=float(np.mean([b['pair_reward'] for a,b in pool])) if pool else None,
            target_prefix_probability=float(np.mean([a['target_correct_action_probability'] for a,b in pool])) if pool else None,
            exploit_episodes=len(eligible),
            exploit_combo_reuse_rate=float(np.mean([b['combo_completed'] for a,b in eligible])) if eligible else None))
    event_rewards=[]
    for lag in range(-5,11):
        pool=[]
        for e,steps in zip(episodes,step_sets):
            if e['first_combo_round'] is None: continue
            end=e['first_combo_round']-1+2*lag
            if 1<=end<horizon:pool.append(steps[end])
        event_rewards.append(dict(pair_lag=lag,episodes=len(pool),
            pair_reward_mean=float(np.mean([r['pair_reward'] for r in pool])) if pool else None,
            combo_completion_rate=float(np.mean([r['combo_completed'] for r in pool])) if pool else None))
    shifts=[e['target_prefix_probability_shift'] for e in episodes if e['target_prefix_probability_shift'] is not None]
    hits=[x for x in first if x is not None]
    return dict(episodes=n,discovery_rate=len(hits)/n,discovery_rate_by_k=sum(e['discovered_by_k'] for e in episodes)/n,
        first_combo_conditional_mean=float(np.mean(hits)) if hits else None,
        restricted_mean_first_combo_round=float(np.mean([horizon if x is None else x for x in first])),
        undiscovered=sum(x is None for x in first),
        target_prefix_probability_shift_mean=float(np.mean(shifts)) if shifts else None,
        target_prefix_probability_shift_count=len(shifts),
        total_reward_mean=float(np.mean([e['total_reward'] for e in episodes])),
        exploitation_mean=float(np.mean([e['exploitation_mean'] for e in episodes])),
        post_discovery_reuse_rate=(sum(e['post_discovery_hits'] for e in episodes)/sum(e['post_discovery_pairs'] for e in episodes)) if sum(e['post_discovery_pairs'] for e in episodes) else None,
        discovery_cdf=cdf,after_discovery_by_pair_lag=lags,event_aligned_pair_rewards=event_rewards,
        note='Fixed-horizon censoring: misses remain censored, never treated as discoveries. Event-aligned curves condition on discovery and remaining horizon; they are descriptive, not causal.')


def track_model(tasks,model,pretrained,config,k,output,rollouts=2,
                conditions=('dual_learned','dual_random','dual_target'),batch_size=64):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    summaries=[];horizon=config['horizon']
    for kind in conditions:
        episodes=[];step_sets=[]
        specs=[(ti,oi,rep) for ti in range(len(tasks)) for oi in range(3) for rep in range(rollouts)]
        with gzip.open(output/f'{kind}.steps.jsonl.gz','wt') as stream:
            for start in range(0,len(specs),batch_size):
                selected=specs[start:start+batch_size]
                seeds=[np.random.SeedSequence([config['seed'],400,ti,oi,rep]).spawn(2) for ti,oi,rep in selected]
                kinds=[('random','fixed','ppo')[oi] for ti,oi,rep in selected];trace=[]
                trajectories,rewards,other=collect_batch([tasks[ti] for ti,_,_ in selected],model,pretrained,kinds,horizon,k,
                    [np.random.default_rng(s[0]) for s in seeds],[np.random.default_rng(s[1]) for s in seeds],kind,trace)
                for j,(ti,oi,rep) in enumerate(selected):
                    ep,steps=describe_episode(tasks[ti],trajectories[j],rewards[j],other[j],trace,j,k)
                    identity=dict(task_index=ti,task_id=tasks[ti].fingerprint,opponent=kinds[j],repetition=rep,k=k,agent=kind)
                    ep.update(identity);episodes.append(ep);step_sets.append(steps)
                    for row in steps:stream.write(json.dumps(dict(**identity,**row),allow_nan=False)+'\n')
        with (output/f'{kind}.episodes.jsonl').open('w') as stream:
            for ep in episodes:stream.write(json.dumps(ep,allow_nan=False)+'\n')
        summaries.append(dict(agent=kind,k=k,**aggregate(episodes,step_sets,horizon,k)))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    return summaries


class TrainingTracker:
    """Fixed monitoring split/seeds, separate from training and final evaluation.

    Observation-only hooks consume no training RNG and never feed labels to the model.
    """
    def __init__(self,tasks,config,output,name,k,pretrained):
        from .generator import generate_task,save_tasks,load_tasks
        self.config=config;self.options=config['tracking'];self.k=k;self.pretrained=pretrained
        self.root=Path(output)/'tracking'/name
        self.root.mkdir(parents=True,exist_ok=True)
        path=Path(output)/'diagnostic_tasks.npz'
        if path.exists():self.tasks=load_tasks(path)
        else:
            rng=np.random.default_rng(np.random.SeedSequence([config['seed'],987654]))
            self.tasks=[generate_task(rng,**config['generator']) for _ in range(self.options.get('tasks',16))]
            save_tasks(path,self.tasks)
        excluded={t.fingerprint for t in tasks}
        final_tasks=Path(output)/'evaluation_tasks.npz'
        if final_tasks.exists():excluded.update(t.fingerprint for t in load_tasks(final_tasks))
        if excluded & {t.fingerprint for t in self.tasks}:raise ValueError('Diagnostic task overlap')
        self.previous={}
        self.checkpoints=Path(output)/'checkpoints'/name
        self.checkpoints.mkdir(parents=True,exist_ok=True)

    def capture(self,model,update):
        import csv
        interval=self.options.get('every',5)
        if update and update%interval and update!=self.config['updates']:return
        was_training=model.training
        try:
            model.eval()
            model.save(self.checkpoints/f'update_{update:06d}.pt',update=update,k=self.k,
                horizon=self.config['horizon'],seed=self.config['seed'],
                exploration_lambda=self.config.get('exploration_lambda',0.),
                exploit_learning=self.config.get('exploit_learning','ppo'))
            summaries=track_model(self.tasks,model,self.pretrained,self.config,self.k,
                self.root/f'update_{update:06d}',rollouts=self.options.get('rollouts',2))
        finally:model.train(was_training)
        rows=[]
        for summary in summaries:
            row={key:summary[key] for key in ('agent','k','episodes','total_reward_mean','exploitation_mean',
                'discovery_rate','discovery_rate_by_k','first_combo_conditional_mean',
                'restricted_mean_first_combo_round','undiscovered','post_discovery_reuse_rate',
                'target_prefix_probability_shift_mean')}
            row['update']=update
            prev=self.previous.get(row['agent'])
            row['return_change_per_update']=None if prev is None else (row['total_reward_mean']-prev['total_reward_mean'])/(update-prev['update'])
            row['discovery_time_change_per_update']=None if prev is None else (row['restricted_mean_first_combo_round']-prev['restricted_mean_first_combo_round'])/(update-prev['update'])
            self.previous[row['agent']]=row;rows.append(row)
        path=self.root/'learning_curve.csv';exists=path.exists()
        with path.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if not exists:writer.writeheader()
            writer.writerows(rows)
        print(f'tracking {self.root.name}: update={update} saved',flush=True)
