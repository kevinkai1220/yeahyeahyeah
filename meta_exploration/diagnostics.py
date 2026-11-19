"""Observed-only history interventions and small staged pilot runner.

Synthetic histories are context interventions, never executed in the environment
or inserted into a Q replay buffer. Every suffix starts from the same real state.
"""
import copy
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .environment import Observation
from .learning import Controller, QLearner, collect, clone_model, make_model, adapt
from .training import train, make_game, world_set, HistoryCollector, write_json
from .evaluation import evaluate, metrics


def intervene(history, kind, seed):
    h=copy.deepcopy(history)
    if kind=='real': return h
    if kind=='reset': return []
    rng=np.random.default_rng(seed)
    # Leave final complete pair unchanged: the actual current observation at
    # handoff must agree with the synthetic history's final action/reward.
    stop=max(0,len(h)-2)
    if kind=='reward_shuffled':
        rewards=[t.reward for t in h[:stop]];rng.shuffle(rewards)
        for t,r in zip(h[:stop],rewards): t.reward=float(r)
    elif kind=='action_relabelled':
        # Nonidentity relabelling, preserving mode, chronology and final pair.
        mapping=np.roll(np.arange(3),int(rng.integers(1,3)))
        for t in h[:stop]:
            t.action=int(mapping[t.action]);t.opponent_action=int(mapping[t.opponent_action])
    else: raise ValueError(kind)
    for i,t in enumerate(h):
        if i:
            prev=h[i-1]
            t.observation=Observation(t.observation.mode,prev.action,prev.opponent_action,prev.reward)
        t.next_observation=Observation(t.next_observation.mode,t.action,t.opponent_action,t.reward)
    # These are coherent observed-token chains, not guaranteed likely under the
    # true reward law. Old action probabilities are invalid and never optimized.
    return h


def history_interventions(config, models, world, seed, mode='frozen', strategy='learned'):
    B,K=[config['training'][k] for k in ('exploration_steps','exploitation_steps')]
    game=make_game(world,B+K,seed,config)
    if strategy=='learned': actor=Controller(models['explorer'],seed)
    elif strategy=='q_learning': actor=QLearner(config['q_learning'],seed)
    else: actor=HistoryCollector(seed,strategy)
    prefix=collect(game,actor,B,online_q=strategy=='q_learning')
    snapshot=game.snapshot()
    results=[]
    for kind in ['real','reward_shuffled','action_relabelled','reset']:
        context=intervene(prefix,kind,seed)
        ctrl=Controller(clone_model(models['dual_neural']),seed+10000)
        ctrl.ingest(context)
        optimizer=torch.optim.Adam(ctrl.model.parameters(),lr=config['training']['learning_rate'])
        before=time.monotonic()
        suffix,updates=adapt(snapshot.snapshot(),ctrl,K,optimizer,config,online=mode=='online')
        results.append(dict(intervention=kind,history_strategy=strategy,evaluation=mode,
            world_id=world.world_id,world_type=world.world_type,
            exploitation_mean=float(np.mean([t.reward for t in suffix])),
            optimizer_steps=sum(u['updates'] for u in updates),
            runtime_seconds=time.monotonic()-before,
            post_B_observation=vars(snapshot.observation()),
            action_sequence=[t.action for t in suffix]))
    return results


def load_models(config,path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    models={}
    for key,state in checkpoint['models'].items():
        m=make_model(config,device);m.load_state_dict(state);m.eval();models[key]=m
    return models


def pilot(config, root, stages, variants, seeds, device='cpu'):
    """Fresh paths only. No full-study launch, no old-checkpoint objective changes."""
    from .plotting import bootstrap, pyplot, save
    root=Path(root)
    if root.exists(): raise ValueError('Pilot output must be a new directory; choose a fresh --out')
    root.mkdir(parents=True)
    all_rows=[]; interventions=[]; training_rows=[]; costs=[]
    specs={'easy':('A1',1,0.,['constant']),
           'type1':('A1',12,0.,['constant']),
           'type2':('A2',12,0.,['constant']),
           'type3':('A3',12,0.,['constant']),
           'mixed_noise':('B',12,1.,['random','fixed'])}
    variants_spec={'legacy':(1,'legacy_episode','legacy_cross_critic'),
                   'legacy_batch':(8,'legacy_episode','legacy_cross_critic'),
                   'batch_cross_critic':(8,'batch','legacy_cross_critic'),
                   'corrected':(8,'batch','observed_suffix')}
    for stage in stages:
        regime,n,sigma,opponents=specs[stage]
        for seed in seeds:
            shared_states=[]
            for variant in variants:
                c=copy.deepcopy(config)
                batch,norm,credit=variants_spec[variant]
                c['environment'].update(reward_sigma=sigma,opponents=opponents)
                c['training'].update(train_worlds=n,rollout_batch_size=batch,advantage_normalization=norm,dual_credit=credit,
                    pretraining_ppo=dict(rollout_batch_size=8,advantage_normalization='batch'))
                out=root/stage/variant/f'seed_{seed}'
                print(f'Pilot {stage}/{variant}/seed_{seed}',flush=True)
                before=time.monotonic()
                train(c,regime,seed,out,device)
                write_json(out/'identity.json',dict(regime=regime,seed=seed))
                shared=torch.load(out/'checkpoints/shared_exploitation_pretrain.pt',map_location='cpu',weights_only=False)['model']
                if shared_states:
                    assert all(torch.equal(shared[k],shared_states[0][k]) for k in shared)
                else: shared_states.append(shared)
                rows=evaluate(c,regime,seed,out,device)
                for row in rows: row.update(stage=stage,variant=variant,split='test')
                all_rows.extend(rows)
                models=load_models(c,out/'checkpoints/trained.pt',device)
                for split in ['train','test']:
                    for world in world_set(c,regime,seed,split)[:c['evaluation']['num_test_worlds']]:
                        s=seed*100000+world.seed
                        # Frozen train/test comparison and analytic random reference.
                        from .evaluation import evaluate_world
                        _,summaries=evaluate_world(c,models,world,s,'frozen')
                        for method,m in summaries.items():
                            training_rows.append(dict(stage=stage,variant=variant,seed=seed,split=split,
                                world_id=world.world_id,method=method,exploitation_mean=m['exploitation_mean'],
                                random_expected_reward=m['adaptation_speed'].get('random_expected_reward')))
                        if split=='test':
                            for row in history_interventions(c,models,world,s):
                                interventions.append(dict(stage=stage,variant=variant,seed=seed,**row))
                cost=json.loads((out/'costs.json').read_text())
                costs.append(dict(stage=stage,variant=variant,seed=seed,wall_seconds=time.monotonic()-before,**cost))
                write_json(root/'pilot_metrics.json',all_rows)
                write_json(root/'history_interventions.json',interventions)
                write_json(root/'train_test_diagnostics.json',training_rows)
                write_json(root/'budgets.json',costs)
    summaries=[]
    groups=defaultdict(list)
    for r in training_rows:groups[r['stage'],r['variant'],r['split'],r['method']].append(r)
    for key,rs in groups.items():
        by_seed=defaultdict(list)
        for r in rs:by_seed[r['seed']].append(r['exploitation_mean'])
        mean,lo,hi=bootstrap(by_seed,config['evaluation']['bootstrap_replicates'])
        summaries.append(dict(zip(('stage','variant','split','method'),key),mean=float(mean),ci_low=float(lo),ci_high=float(hi),
            seed_means={str(s):float(np.mean(v)) for s,v in by_seed.items()},
            random_expected_reward=float(np.mean([r['random_expected_reward'] for r in rs]))))
    write_json(root/'summary.json',summaries)
    # Paired contrasts retain actual world identity, seed, and split.
    lookup={(r['stage'],r['variant'],r['seed'],r['split'],r['world_id'],r['method']):r for r in training_rows}
    diffs=defaultdict(lambda:defaultdict(list))
    for r in training_rows:
        if r['variant']!='corrected':continue
        for other in variants:
            if other=='corrected':continue
            b=lookup.get((r['stage'],other,r['seed'],r['split'],r['world_id'],r['method']))
            if b:diffs[r['stage'],r['split'],r['method'],other][r['seed']].append(r['exploitation_mean']-b['exploitation_mean'])
    comparisons=[]
    for key,by_seed in diffs.items():
        m,l,h=bootstrap(by_seed,config['evaluation']['bootstrap_replicates'])
        comparisons.append(dict(zip(('stage','split','method','reference_variant'),key),contrast='corrected minus reference',mean=float(m),ci_low=float(l),ci_high=float(h),n_pairs=sum(map(len,by_seed.values()))))
    write_json(root/'paired_comparisons.json',comparisons)
    # Policy-specific training and optimization diagnostics; actual logged values.
    for stage in stages:
        fig,axes=pyplot().subplots(2,3,figsize=(15,8))
        for variant in variants:
            for seed in seeds:
                log=root/stage/variant/f'seed_{seed}'/'training.jsonl'
                records=[json.loads(x) for x in log.read_text().splitlines()]
                for ax,metric in zip(axes.flat,['return_mean','actor_loss','critic_loss','actor_gradient_norm','entropy','parameter_delta_norm']):
                    for policy in ['single_policy','explorer','dual_neural']:
                        rs=[r for r in records if r['stage']==policy+('_update' if metric!='return_mean' else '') and metric in r]
                        if rs:ax.plot([r['episode'] for r in rs],[r[metric] for r in rs],label=f'{variant}/{policy}/s{seed}',alpha=.5)
                    ax.set(xlabel='Episode',ylabel=metric)
        axes[0,0].legend(fontsize=4,ncol=2);save(fig,root/(stage+'_learning'))
    return summaries


def evaluate_budgets(checkpoint_root, output, device='cpu', budgets=(8,16,32,64), K=128):
    """Evaluation-only copies; retain training config and checkpoint provenance."""
    import hashlib
    import shutil
    checkpoint_root,output=Path(checkpoint_root),Path(output)
    if output.exists(): raise ValueError('Evaluation output must be a fresh directory')
    sources=sorted(checkpoint_root.glob('seed_*/checkpoints/trained.pt'))
    if not sources: raise ValueError('Expected seed_*/checkpoints/trained.pt under checkpoint root')
    output.mkdir(parents=True)
    for source in sources:
        run=source.parent.parent
        c=json.loads((run/'config.json').read_text())
        original=copy.deepcopy(c)
        identity=json.loads((run/'identity.json').read_text())
        c['training']['exploration_steps']=budgets[0]
        c['training']['exploitation_steps']=K
        c['evaluation'].update(exploration_budgets=list(budgets),exploitation_budgets=[K])
        target=output/run.name;(target/'checkpoints').mkdir(parents=True)
        shutil.copyfile(source,target/'checkpoints/trained.pt')
        write_json(target/'config.json',c);write_json(target/'identity.json',identity)
        write_json(target/'training_source_config.json',original)
        write_json(target/'evaluation_source.json',dict(checkpoint=str(source.resolve()),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_performed=False))
        before=time.monotonic()
        evaluate(c,identity['regime'],identity['seed'],target,device)
        write_json(target/'runtime.json',dict(evaluation_seconds=time.monotonic()-before,device=device))


def main():
    import argparse
    from .config import load_config
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/meta/diagnostic_pilot.json');p.add_argument('--out',required=True)
    p.add_argument('--checkpoint-root',help='Evaluation-only B=8/16/32/64, K=128 study from existing seed directories')
    p.add_argument('--stages',nargs='+',default=['easy'],choices=['easy','type1','type2','type3','mixed_noise'])
    p.add_argument('--variants',nargs='+',default=['legacy','corrected'],choices=['legacy','legacy_batch','batch_cross_critic','corrected'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--device',default='cpu')
    a=p.parse_args();c=load_config(a.config);torch.set_num_threads(c['hardware']['threads_per_worker'])
    if a.checkpoint_root:
        evaluate_budgets(a.checkpoint_root,a.out,a.device)
    else:
        pilot(c,a.out,a.stages,a.variants,a.seeds,a.device)


if __name__=='__main__':main()
