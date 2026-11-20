"""Small fresh-training A3 diagnostics; explicit stages/variants, no full study."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .config import load_config
from .training import train,world_set,make_game,write_json,HistoryCollector
from .learning import collect
from .diagnostics import load_models
from .evaluation import evaluate_world,metrics
from .environment import oracle_values

STAGES={
    'deterministic':(1,0.,['constant']),
    'random_opponent':(1,0.,['random']),
    'noisy':(1,1.,['random']),
    'multi_constant':(12,0.,['constant']),
    'multi_mixed':(12,1.,['random','fixed']),
}
VARIANTS={'baseline':{},'cadence1':{'sequence_minibatch_size':1},
          'value_coef':{'value_coefficient':.05},'value_scale':{'critic_loss_scale':10.},'group_clip':{'gradient_clipping':'parameter_groups'}}


def oracle_rollout(game,world):
    """Diagnostic-only exact DP controller for current opponent/mode, no training."""
    values=oracle_values(world,game.horizon,game.opponent_kind);total=0.;pairs=[]
    while game.t<game.horizon:
        t=game.t
        p=np.full(3,1/3) if game.opponent_kind=='random' else np.eye(3)[0 if game.opponent_kind=='constant' else t%3]
        means=world.base.copy()
        if game.mode:means=means+world.bonus[game.previous[0],game.previous[1]]
        action=int(((means+values[t+1,:3,:3])*p).sum(-1).argmax())
        if game.mode:pairs.append([game.previous[0],action])
        _,reward,_,_=game.step(action);total+=reward
    return dict(realized_return=total,oracle_expected_return=float(values[0,3,3]),pairs=pairs)


def run(config,out,stages,variants,seeds,device):
    out=Path(out)
    if out.exists():raise ValueError('Fresh pilot output required')
    out.mkdir(parents=True);summary=[]
    for stage in stages:
        n,sigma,opponents=STAGES[stage]
        for seed in seeds:
            shared=None
            for variant in variants:
                c=copy.deepcopy(config);c['environment'].update(reward_sigma=sigma,opponents=opponents)
                c['training'].update(train_worlds=n,rollout_batch_size=8,advantage_normalization='batch',
                    dual_credit='observed_suffix',gradient_diagnostics=True,sequence_minibatch_size=0,critic_loss_scale=1.)
                c['training'].update(VARIANTS[variant])
                # Fix pretraining across variants: same RNG/init/history/optimization.
                c['training']['pretraining_ppo']=dict(sequence_minibatch_size=0,critic_loss_scale=1.,value_coefficient=.5,
                    advantage_normalization='batch',rollout_batch_size=8,gradient_clipping='global')
                target=out/stage/variant/f'seed_{seed}';start=time.monotonic()
                print(f'Train {stage}/{variant}/seed_{seed}',flush=True)
                train(c,'A3',seed,target,device)
                state=torch.load(target/'checkpoints/shared_exploitation_pretrain.pt',map_location='cpu',weights_only=False)['model']
                if shared is not None:assert all(torch.equal(v,shared[k]) for k,v in state.items())
                shared=state
                models=load_models(c,target/'checkpoints/trained.pt',device)
                rows=[];B,K=c['training']['exploration_steps'],c['training']['exploitation_steps']
                for split in ['train','test']:
                    for world in world_set(c,'A3',seed,split):
                        s=seed*100000+world.seed;initial=make_game(world,B+K,s,c)
                        _,scores=evaluate_world(c,models,world,s,'frozen')
                        for strategy in ['random','repeat']:
                            h=collect(initial.snapshot(),HistoryCollector(s,strategy),B+K)
                            scores[strategy]=metrics(h,world,B,initial.opponent_kind,c)
                        for method,score in scores.items():
                            rows.append(dict(stage=stage,variant=variant,seed=seed,split=split,world_id=world.world_id,method=method,**score))
                    w=world_set(c,'A3',seed,split)[0]
                    write_json(target/f'{split}_oracle_check.json',oracle_rollout(make_game(w,B+K,seed,c),w))
                write_json(target/'pilot_evaluation.json',rows)
                log=[json.loads(l) for l in (target/'training.jsonl').read_text().splitlines()]
                for method in ['single_policy','dual_neural']:
                    learning=[r for r in log if r['stage']==method]
                    updates=[r for r in log if r['stage']==method+'_update']
                    for split in ['train','test']:
                        rs=[r for r in rows if r['split']==split and r['method']==method]
                        summary.append(dict(stage=stage,variant=variant,seed=seed,method=method,split=split,
                            early_training_reward=float(np.mean([r['return_mean'] for r in learning[:16]])),
                            late_training_reward=float(np.mean([r['return_mean'] for r in learning[-16:]])),
                            evaluation_reward=float(np.mean([r['exploitation_mean'] for r in rs])),
                            optimal_combo_utilization=float(np.mean([r['optimal_combo_utilization'] for r in rs])),
                            optimizer_steps=sum(r['updates'] for r in updates),
                            interactions=c['training']['episodes']*(B+K),
                            last_update=updates[-1],runtime_seconds=time.monotonic()-start))
                write_json(out/'summary.json',summary)
                from .plotting import pyplot,save
                fig,axes=pyplot().subplots(1,2,figsize=(10,4))
                for method in ['single_policy','dual_neural']:
                    rs=[r for r in log if r['stage']==method]
                    vals=np.array([r['return_mean'] for r in rs]);window=min(16,len(vals))
                    axes[0].plot(np.arange(window,len(vals)+1),np.convolve(vals,np.ones(window)/window,'valid'),label=method)
                    us=[r for r in log if r['stage']==method+'_update']
                    axes[1].plot([r['episode'] for r in us],[r['shared_gradient_cosine'] for r in us],label=method)
                axes[0].set(xlabel='Training episode',ylabel='Trailing 16-episode K reward')
                axes[1].set(xlabel='Training episode',ylabel='Shared actor/critic gradient cosine');axes[0].legend()
                save(fig,target/'learning')
    write_json(out/'complete.json',dict(training_runs=len(stages)*len(variants)*len(seeds)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/meta/history_optimization_pilot.json');p.add_argument('--out',required=True)
    p.add_argument('--stages',nargs='+',choices=STAGES,default=['deterministic'])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--device',default='cpu')
    a=p.parse_args();c=load_config(a.config);torch.set_num_threads(c['hardware']['threads_per_worker'])
    run(c,a.out,a.stages,a.variants,a.seeds,a.device)

if __name__=='__main__':main()
