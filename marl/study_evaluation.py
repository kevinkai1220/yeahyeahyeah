"""Batched held-out evaluation with combo reuse diagnostics and unchanged seeds."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .batched import collect_batch
from .oracle import oracle_values


def episode_metrics(task, trajectory, rewards, other_actions, k, oracle):
    own = rewards[:,0]
    exploit = own[k:]
    actions = trajectory.actions
    hits=[]
    if task.combos is not None:
        u,v=task.combos[0]
        hits=[t for t in range(1,len(own),2) if actions[t-1]==u and actions[t]==v]
    first=hits[0] if hits else None
    eligible=[t for t in range(1,len(own),2) if first is not None and t-1>first and t-1>=k]
    prefixes=sum(int(actions[t-1]==u) for t in eligible) if task.combos is not None else 0
    reused=sum(int(t in hits) for t in eligible)
    # Safe action is evaluator-only tensor-derived metadata, never an input.
    safe=int(task.r1[0].mean(axis=1).argmax())
    oracle_start=None if oracle is None else float(oracle[k,actions[k-1],other_actions[k-1]])
    return dict(total_reward=float(own.sum()),mean_reward=float(own.mean()),
        exploitation_return=float(exploit.sum()),exploitation_mean=float(exploit.mean()),
        adaptation_auc=float(np.mean(exploit.cumsum()/np.arange(1,len(exploit)+1))),
        opponent_total_reward=float(rewards[:,1].sum()),reward_difference=float((own-rewards[:,1]).sum()),
        combo_metadata_available=task.combos is not None,
        combo_discovered=bool(hits) if task.combos is not None else None,
        first_combo_round=None if first is None else first+1,
        combo_discovered_by_k=None if task.combos is None else first is not None and first+1<=k,
        oracle_expected_return=oracle_start,oracle_regret=None if oracle_start is None else oracle_start-float(exploit.sum()),
        reuse_opportunities=len(eligible),reuse_prefixes=prefixes,reuse_completions=reused,
        total_combo_completions=len(hits),safe_action_fraction=float(np.mean(actions==safe)))


def evaluate_study(tasks,duals,pretrained,config,output,baseline_path=None,batch_size=64):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    oracle_cache={(i,op):oracle_values(task,config['horizon'],op) for i,task in enumerate(tasks) for op in ('random','fixed')}
    groups=defaultdict(list)
    with (output/'episodes.jsonl').open('w') as stream:
        if baseline_path is not None:
            for line in Path(baseline_path).open():
                row=json.loads(line)
                if row['agent'] in ('single_gru','q_learning') and row['k'] in duals:
                    stream.write(line);groups[row['k'],row['agent'],row['opponent']].append(row)
        for k,model in duals.items():
            for kind in ('dual_learned','dual_random','dual_target'):
                specs=[(ti,oi,rep) for ti in range(len(tasks)) for oi in range(3) for rep in range(config['evaluation_rollouts'])]
                for start in range(0,len(specs),batch_size):
                    selected=specs[start:start+batch_size]
                    kinds=[('random','fixed','ppo')[oi] for ti,oi,rep in selected]
                    seeds=[np.random.SeedSequence([config['seed'],400,ti,oi,rep]).spawn(2) for ti,oi,rep in selected]
                    ts,rewards,other=collect_batch([tasks[ti] for ti,_,_ in selected],model,pretrained,kinds,
                        config['horizon'],k,[np.random.default_rng(s[0]) for s in seeds],
                        [np.random.default_rng(s[1]) for s in seeds],kind)
                    for j,(ti,oi,rep) in enumerate(selected):
                        op=kinds[j]
                        row=dict(task_index=ti,task_id=tasks[ti].fingerprint,k=k,agent=kind,opponent=op,repetition=rep,
                            **episode_metrics(tasks[ti],ts[j],rewards[j],other[j],k,oracle_cache.get((ti,op))))
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                        groups[k,kind,op].append(row)
                stream.flush()
                print(f'evaluate K={k} {kind}: {len(specs)} episodes',flush=True)
    summaries=[]
    for (k,agent,op),rows in groups.items():
        summary=dict(k=k,agent=agent,opponent=op,rollouts=len(rows))
        for metric in ('total_reward','mean_reward','exploitation_return','exploitation_mean','adaptation_auc','reward_difference','oracle_regret','safe_action_fraction','total_combo_completions'):
            vals=[r[metric] for r in rows if r.get(metric) is not None]
            summary[metric]=float(np.mean(vals)) if vals else None
        summary['combo_discovery_rate']=float(np.mean([r['combo_discovered'] for r in rows]))
        summary['combo_discovery_rate_by_k']=float(np.mean([r['combo_discovered_by_k'] for r in rows]))
        opportunities=sum(r.get('reuse_opportunities',0) for r in rows)
        prefixes=sum(r.get('reuse_prefixes',0) for r in rows)
        completions=sum(r.get('reuse_completions',0) for r in rows)
        summary.update(reuse_opportunities=opportunities,reuse_prefixes=prefixes,reuse_completions=completions,
            combo_reuse_rate=completions/opportunities if opportunities else None,
            completion_given_prefix=completions/prefixes if prefixes else None)
        summaries.append(summary)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    return summaries
