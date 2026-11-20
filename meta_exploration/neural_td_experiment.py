"""Independent Neural TD experiments; existing PPO/Tabular paths are unchanged."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .config import load_config,merge,REGIMES
from .neural_td import QNetwork,MetaExplorer,NeuralTD,episode,explorer_targets,leave_one_out
from .learning import Controller,Transition,collect,ppo_batch_update
from .training import (world_set,make_game,HistoryCollector,write_json,append_json,atomic_save,
    provenance,rng_state,restore_rng)
from .evaluation import metrics,public_rows,diagnostic_rows,evaluate_world

TD_DEFAULT=dict(hidden=64,reward_input_scale=10.,learning_rate=.001,gamma=.99,
    capacity=2048,batch_size=32,learning_starts=1,prefix_updates=4,k_updates=1,
    prefix_timing='online',target_tau=.05,target_interval=0,max_grad_norm=10.,
    epsilon_start=.3,epsilon_end=.05,epsilon_decay_steps=160,
    pretrain_interactions=96000,pretrain_updates=1,outer_k_online=True,
    same_world_rollouts=1,loo_baseline=False,
    evaluation_timings=['online'],evaluation_budgets=[8,16,32,64])
METHODS={'random':'neural_td_random','coverage':'neural_td_coverage','repeat':'neural_td_repeat',
         'existing':'neural_td_existing','meta':'neural_td_meta'}


def config(path=None):
    c=load_config(path);c['neural_td']=merge(copy.deepcopy(TD_DEFAULT),c.get('neural_td',{}));d=c['neural_td']
    if c['training']['advantage_normalization']!='batch' or c['training']['gamma']!=1:
        raise ValueError('Neural TD outer optimization requires batch advantage normalization and gamma=1')
    if not 0<=d['gamma']<=1 or d['prefix_timing'] not in ['online','batch']:raise ValueError('Invalid TD settings')
    if min(d['capacity'],d['batch_size'],d['learning_starts'],d['pretrain_interactions'])<1:raise ValueError('Invalid replay/pretrain size')
    if d['learning_starts']>d['capacity']:raise ValueError('learning_starts exceeds capacity')
    if not 0<d['target_tau']<=1 or d['target_interval']<0:raise ValueError('Invalid target update')
    group=d['same_world_rollouts'];batch=c['training']['rollout_batch_size']
    if group<1 or batch%group or c['training']['episodes']%group:raise ValueError('Batches/episodes must contain complete world groups')
    if d['loo_baseline'] and group<2:raise ValueError('LOO requires multiple same-world rollouts')
    if 'ppo' in c['environment']['opponents']:raise ValueError('Use constant/random/fixed opponent for first Neural TD study')
    return c


def model_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def pretrain(c,regime,seed,out,device):
    """Observed-only cumulative train-world TD; replay/optimizer reset per world.

    Pretraining parameters persist across train worlds. No pooled test replay,
    no privileged combo selection, no candidate filtering, no meta joint update.
    """
    d=c['neural_td'];worlds=world_set(c,regime,seed,'train')
    initial=QNetwork(d['hidden'],d['reward_input_scale']).to(device)
    horizon=c['training']['exploration_steps']+c['training']['exploitation_steps']
    used=updates=sampled=0;index=0;seconds=0.
    while used<d['pretrain_interactions']:
        w=worlds[index%len(worlds)];steps=min(horizon,d['pretrain_interactions']-used)
        # Last partial collection is a truncation, never a fabricated termination.
        game=make_game(w,horizon,seed*100000+index,c)
        learner=NeuralTD(initial,d,w.world_id,seed+index)
        actor=HistoryCollector(seed+index,['coverage','random','repeat'][index%3])
        rewards=[]
        for _ in range(steps):
            o=game.observation();a,info=actor.act(o);n,r,done,b=game.step(a)
            learner.observe(Transition(o,a,r,n,done,b,info),w.world_id)
            learner.update(d['pretrain_updates']);rewards.append(r)
        initial.load_state_dict(learner.model.state_dict())
        used+=steps;updates+=learner.updates;sampled+=learner.replay.sampled;seconds+=learner.seconds
        append_json(out/'pretraining.jsonl',dict(world_id=w.world_id,split='train',interactions=used,
            reward_mean=float(np.mean(rewards)),**learner.diagnostics()))
        index+=1
    atomic_save(dict(model=initial.state_dict(),config=c,train_fingerprints=[w.fingerprint for w in worlds]),out/'checkpoints/td_initial.pt')
    return initial,dict(interactions=used,td_updates=updates,sampled_transitions=sampled,optimizer_seconds=seconds)


def train(c,regime,seed,out,device='cpu',resume=False):
    out=Path(out)
    if out.exists() and not resume:raise ValueError('Existing output; use a fresh path or explicit --resume')
    out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=c:raise ValueError('Resume config mismatch')
    write_json(out/'config.json',c);write_json(out/'identity.json',dict(regime=regime,seed=seed))
    if not (out/'provenance.json').exists():write_json(out/'provenance.json',provenance(c,device,seed))
    torch.manual_seed(seed);t=c['training'];d=c['neural_td'];B,K=t['exploration_steps'],t['exploitation_steps']
    worlds=world_set(c,regime,seed,'train');test=world_set(c,regime,seed,'test')
    assert not {w.fingerprint for w in worlds}&{w.fingerprint for w in test}
    write_json(out/'worlds_diagnostic.json',[w.diagnostic() for w in worlds+test])
    if (out/'checkpoints/td_initial.pt').exists():
        initial=QNetwork(d['hidden'],d['reward_input_scale']).to(device)
        saved=torch.load(out/'checkpoints/td_initial.pt',map_location=device,weights_only=False)
        initial.load_state_dict(saved['model']);cost=json.loads((out/'pretraining_cost.json').read_text())
    else:
        if (out/'pretraining.jsonl').exists():raise ValueError('Incomplete pretraining preserved; start a fresh output')
        initial,cost=pretrain(c,regime,seed,out,device);write_json(out/'pretraining_cost.json',cost)
    # Meta initialization independent of pretraining runtime/RNG consumption/resume.
    torch.manual_seed(seed+700001);explorer=MetaExplorer(t['hidden']).to(device)
    optimizer=torch.optim.Adam(explorer.parameters(),lr=t['learning_rate'])
    before=model_digest(initial);start=0;total_td=total_sampled=outer_updates=0
    checkpoint=out/'checkpoints/meta_resume.pt'
    if resume and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        explorer.load_state_dict(saved['explorer']);optimizer.load_state_dict(saved['optimizer'])
        explorer.policy_version=saved['version'];restore_rng(saved);start=saved['episode']
        total_td=saved['total_td'];total_sampled=saved['total_sampled'];outer_updates=saved['outer_updates']
    if resume and (out/'training.jsonl').exists():
        logged=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
        if any(r['stage']=='meta_episode' and r['episode']>=start for r in logged):
            raise ValueError('Uncommitted partial batch preserved; use a fresh output, no silent replay of interactions')
    for batch_start in range(start,t['episodes'],t['rollout_batch_size']):
        episodes=[];ids=[]
        for i in range(batch_start,min(t['episodes'],batch_start+t['rollout_batch_size'])):
            w=worlds[(i//d['same_world_rollouts'])%len(worlds)];s=seed*100000+i
            history,prefix,learner,diag=episode(make_game(w,B+K,s,c),initial,d,B,K,'meta',s,explorer,
                                              online_k=d['outer_k_online'])
            a,r=explorer_targets(prefix,history[B:],c);episodes.append((prefix,a,r,[]));ids.append(w.world_id)
            total_td+=learner.updates;total_sampled+=learner.replay.sampled
            append_json(out/'training.jsonl',dict(stage='meta_episode',episode=i,world_id=w.world_id,
                exploration_return=sum(x.reward for x in history[:B]),downstream_return=sum(x.reward for x in history[B:]),
                exploitation_mean=float(np.mean([x.reward for x in history[B:]])),objective=float(r[0]),
                interactions=(i+1)*(B+K),rollout_version=getattr(explorer,'policy_version',0),**learner.diagnostics()))
        if d['loo_baseline']:episodes=leave_one_out(episodes,ids)
        u=ppo_batch_update(Controller(explorer,0),optimizer,episodes,c);outer_updates+=u['updates']
        end=min(t['episodes'],batch_start+t['rollout_batch_size'])
        append_json(out/'training.jsonl',dict(stage='meta_update',episode=end,world_ids=ids,**u))
        atomic_save(dict(explorer=explorer.state_dict(),optimizer=optimizer.state_dict(),version=explorer.policy_version,
            episode=end,total_td=total_td,total_sampled=total_sampled,outer_updates=outer_updates,**rng_state()),checkpoint)
    assert model_digest(initial)==before
    atomic_save(dict(explorer=explorer.state_dict(),td_initial=initial.state_dict(),config=c,td_initial_sha256=before),out/'checkpoints/neural_td.pt')
    write_json(out/'costs.json',dict(pretraining=cost,main_interactions=t['episodes']*(B+K),
        inner_td_updates=total_td,inner_sampled_transitions=total_sampled,outer_optimizer_steps=outer_updates,
        outer_transitions=t['episodes']*B,outer_token_passes=t['episodes']*B*t['ppo_epochs'],
        td_parameters=sum(p.numel() for p in initial.parameters()),explorer_parameters=sum(p.numel() for p in explorer.parameters()),
        initialization_fixed_during_meta=True,same_world_rollouts=d['same_world_rollouts'],
        q_gamma=d['gamma'],outer_gamma=1.,outer_credit='complete MC return-to-go; no intrinsic reward',
        target_parameters=sum(p.numel() for p in initial.parameters())))
    write_json(out/'training_complete.json',dict(regime=regime,seed=seed))
    return initial,explorer


def diagnostic_scores(history,world,B,c,opponent):
    score=metrics(history,world,B,opponent,c)
    ds=list(diagnostic_rows(history[B:],world))
    score.update(base_reward_component=float(np.mean([r['true_mean']-r['true_bonus'] for r in ds])),
                 combo_bonus_component=float(np.mean([r['true_bonus'] for r in ds])),
                 unique_pair_coverage=len(score['combo_attempt_counts']))
    pairs=world.base.mean(1)[:,None]+world.base.mean(1)[None,:]+world.bonus.mean((1,3))
    good=tuple(np.unravel_index(pairs.argmax(),(3,3)))
    evidence=[t.reward for t in history[:B] if t.observation.mode and (t.observation.previous_own,t.action)==good]
    supported=len(evidence)>=2 and np.mean(evidence)-1.96*max(world.sigma,np.std(evidence))/np.sqrt(len(evidence))>0
    score.update(optimal_pair_attempts=len(evidence),optimal_pair_positive_rewards=sum(r>0 for r in evidence),
        optimal_pair_supported=bool(supported),evidence_group='not_applicable' if world.world_type=='type1' else
        'not_attempted' if not evidence else 'strong_evidence' if supported else 'weak_evidence',
        discovered_but_not_utilized=bool(world.world_type!='type1' and supported and (score.get('optimal_combo_utilization') or 0)<.1),
        type3_ordered_completion=score['optimal_combo_utilization'] if world.world_type=='type3' else None)
    # Sampled discounted return calibration at actual action time. Not oracle Q*.
    tail=0.;errors=[]
    for step in reversed(range(len(history))):
        t=history[step];tail=t.reward+c['neural_td']['gamma']*(not t.done)*tail
        if step>=B and 'q_values' in t.info:errors.append((t.info['q_values'][t.action]-tail)**2)
    score['q_sampled_return_rmse']=float(np.sqrt(np.mean(errors))) if errors else None
    score['q_accuracy_reference']='one sampled finite-horizon discounted behavior return; not stationary optimal Q'
    return score


def evaluate(c,regime,seed,source,out,device='cpu',existing_root=None):
    from .diagnostics import load_models
    source,out=Path(source),Path(out)
    if out.exists():raise ValueError('Evaluation output must be fresh')
    saved=torch.load(source/'checkpoints/neural_td.pt',map_location=device,weights_only=False)
    if saved['config']!=c:raise ValueError('Source config mismatch; evaluation budgets/timing belong in training config')
    d=c['neural_td'];initial=QNetwork(d['hidden'],d['reward_input_scale']).to(device);initial.load_state_dict(saved['td_initial'])
    explorer=MetaExplorer(c['training']['hidden']).to(device);explorer.load_state_dict(saved['explorer']);explorer.eval()
    existing=None;reference=None
    if existing_root:
        reference=Path(existing_root)/regime/f'seed_{seed}'
        ec=json.loads((reference/'config.json').read_text())
        existing=load_models(ec,reference/'checkpoints/trained.pt',device)
    out.mkdir(parents=True)
    write_json(out/'provenance.json',dict(execution=provenance(c,device,seed),checkpoint=str(source.resolve()),
        checkpoint_sha256=hashlib.sha256((source/'checkpoints/neural_td.pt').read_bytes()).hexdigest(),
        td_initial_sha256=model_digest(initial),existing_checkpoint=str(reference) if reference else None,
        existing_training_config=ec if reference else None,
        missing_methods=[] if existing else ['single_policy','dual_neural','dual_q_table','neural_td_existing']))
    results=[];B_default=c['training']['exploration_steps'];K=c['training']['exploitation_steps']
    for B in d['evaluation_budgets']:
        cc=copy.deepcopy(c);cc['training']['exploration_steps']=B
        for timing in d['evaluation_timings']:
            td=dict(d,prefix_timing=timing)
            for mode in ['frozen_k','online_k']:
                for w in world_set(cc,regime,seed,'test'):
                    s=seed*100000+w.seed;initial_game=make_game(w,B+K,s,cc)
                    for strategy,method in METHODS.items():
                        if strategy=='existing' and existing is None:continue
                        start=time.perf_counter()
                        h,_,learner,diag=episode(initial_game.snapshot(),initial,td,B,K,strategy,s,explorer,
                            existing['explorer'] if existing else None,online_k=mode=='online_k')
                        score=diagnostic_scores(h,w,B,cc,initial_game.opponent_kind)
                        row=dict(regime=regime,seed=seed,world_id=w.world_id,world_type=w.world_type,method=method,
                            evaluation=mode,B=B,K=K,prefix_timing=timing,initial_sha256=model_digest(initial),
                            training_B=B_default,training_prefix_timing=d['prefix_timing'],
                            compute=dict(prefix=diag['prefix'],total=diag['total'],k_updates=diag['k_updates'],
                                b_parameter_delta=diag['b_parameter_delta'],k_parameter_delta=diag['k_parameter_delta'],runtime_seconds=time.perf_counter()-start),
                            interaction_count=B+K,**score)
                        results.append(row)
                        folder=out/f'{timing}_{mode}_B{B}';folder.mkdir(exist_ok=True)
                        stem=f'{w.world_id}_{method}'
                        if c['logging']['save_trajectories']:
                            for r in public_rows(h,w,method,B):append_json(folder/(stem+'.jsonl'),r)
                            for r in learner.update_log:append_json(folder/(stem+'_td.jsonl'),r)
                    if not existing:
                        from .learning import QLearner
                        q=QLearner(c['q_learning'],s)
                        h=collect(initial_game.snapshot(),q,B+K,online_q=True)
                        results.append(dict(regime=regime,seed=seed,world_id=w.world_id,world_type=w.world_type,
                            method='q_learning',evaluation=mode,B=B,K=K,prefix_timing=timing,interaction_count=B+K,
                            legacy_semantics='Tabular updates in B and K in both modes',
                            **diagnostic_scores(h,w,B,cc,initial_game.opponent_kind)))
                    if existing:
                        _,summaries=evaluate_world(cc,existing,w,s,'online' if mode=='online_k' else 'frozen')
                        for method,score in summaries.items():
                            if method not in ['single_policy','dual_neural','q_learning','dual_q_table']:continue
                            results.append(dict(regime=regime,seed=seed,world_id=w.world_id,world_type=w.world_type,method=method,
                                evaluation=mode,B=B,K=K,prefix_timing=timing,interaction_count=B+K,
                                legacy_semantics='Tabular always updates; neural frozen means no B or K gradient updates',**score))
                print(f'Evaluated {regime}/seed{seed}/B{B}/{timing}/{mode}',flush=True)
    assert model_digest(initial)==saved['td_initial_sha256']
    write_json(out/'metrics.json',results);write_json(out/'complete.json',dict(rows=len(results)))
    return results


def gate(c,out,device):
    """Observed coverage only; no true pair enters learner or data selection."""
    from .environment import generate_world,Game
    out=Path(out)
    if out.exists():raise ValueError('Gate output must be fresh')
    out.mkdir(parents=True);rows=[]
    for kind in ['type2','type3']:
        for seed in [0,1,2]:
            torch.manual_seed(seed);d=dict(c['neural_td'],epsilon_start=0.,epsilon_end=0.,prefix_updates=c['neural_td'].get('gate_updates',64),prefix_timing=c['neural_td'].get('gate_timing','online'))
            B,K=c['neural_td'].get('gate_B',36),128;w=generate_world(seed,kind,split='train',sigma=0.)
            initial=QNetwork(d['hidden'],d['reward_input_scale']).to(device)
            h,_,learner,diag=episode(Game(w,B+K,seed,'constant'),initial,d,B,K,'coverage',seed,online_k=False)
            cc=copy.deepcopy(c);cc['training'].update(exploration_steps=B,exploitation_steps=K)
            row=dict(world_type=kind,seed=seed,B=B,K=K,prefix_updates_per_step=d['prefix_updates'],prefix_timing=d['prefix_timing'],
                **diagnostic_scores(h,w,B,cc,'constant'),compute=learner.diagnostics())
            rows.append(row)
            for r in learner.update_log:append_json(out/f'{kind}_seed{seed}_td.jsonl',r)
            print(kind,seed,'utilization',row['optimal_combo_utilization'],'reward',row['exploitation_mean'],flush=True)
    passed=all(r['optimal_combo_utilization']>=.9 for r in rows)
    write_json(out/'results.json',rows);write_json(out/'gate.json',dict(passed=passed,
        criterion='>=90% frozen exploitation optimal combo use in all 6 deterministic coverage cases',
        note='Explicit gate_B/gate_updates in config; mechanism test, not default-budget sample efficiency proof',
        execution=provenance(c,device,0)))
    return passed


def paired_td_speed(rows,replicates=1000):
    """New explorer contrasts, same seed/world/B/K/timing; censoring is preserved."""
    from .plotting import bootstrap
    keys=('regime','seed','world_id','prefix_timing','evaluation','B','K')
    lookup={tuple(r[k] for k in keys)+(r['method'],):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate paired evaluation identity')
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['method']!='neural_td_meta':continue
        for other in ['neural_td_random','neural_td_coverage','neural_td_repeat','neural_td_existing']:
            ref=lookup.get(tuple(r[k] for k in keys)+(other,))
            if ref is None:continue
            for norm in ['absolute','baseline_normalized']:
                a,b=r['adaptation_speed'][norm],ref['adaptation_speed'][norm]
                if not a['defined'] or not b['defined']:continue
                diffs={}
                if a['normalized_adaptation_auc'] is not None and b['normalized_adaptation_auc'] is not None:
                    diffs['normalized_adaptation_auc']=a['normalized_adaptation_auc']-b['normalized_adaptation_auc']
                for q,ea in a['events'].items():
                    eb=b['events'][q]
                    diffs['reach_'+q]=float(ea['event'])-float(eb['event'])
                    diffs['rmst_'+q]=ea['restricted_time']-eb['restricted_time']
                for metric,value in diffs.items():
                    groups[r['regime'],r['world_type'],r['prefix_timing'],r['evaluation'],r['B'],r['K'],other,norm,metric][r['seed']].append(value)
    result=[]
    for key,by in groups.items():
        m,l,h=bootstrap(by,replicates)
        result.append(dict(zip(['regime','world_type','prefix_timing','evaluation','B','K','reference','normalization','metric'],key),
            contrast='meta minus reference',mean=float(m),ci_low=float(l),ci_high=float(h),seeds=len(by),pairs=sum(map(len,by.values()))))
    return result


def summarize(inputs,out):
    from .plotting import summarize as aggregate,pyplot,save,bootstrap
    out=Path(out)
    if out.exists():raise ValueError('Fresh summary output required')
    out.mkdir(parents=True);rows=[]
    for source in map(Path,inputs):
        paths=[source/'metrics.json'] if (source/'metrics.json').exists() else list(source.glob('**/evaluation/metrics.json'))
        for p in paths:rows.extend(json.loads(p.read_text()))
    if not rows:raise ValueError('No executed evaluations')
    from .plotting import COLORS
    COLORS.update(neural_td_random='#17becf',neural_td_coverage='#e377c2',neural_td_repeat='#7f7f7f',
                  neural_td_existing='#8c564b',neural_td_meta='#d62728')
    paired=[]
    diagnostic_groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        fields={k:r.get(k) for k in ['unique_pair_coverage','optimal_pair_attempts','optimal_pair_positive_rewards',
            'base_reward_component','combo_bonus_component','q_sampled_return_rmse','type3_ordered_completion']}
        fields.update(r.get('compute',{}).get('total',{}))
        for metric,value in fields.items():
            if value is not None and isinstance(value,(int,float)):
                diagnostic_groups[r['regime'],r['world_type'],r['prefix_timing'],r['evaluation'],r['B'],r['method'],metric][r['seed']].append(value)
    diagnostic_summary=[]
    for key,values in diagnostic_groups.items():
        m,l,h=bootstrap(values,1000)
        diagnostic_summary.append(dict(zip(['regime','world_type','prefix_timing','evaluation','B','method','metric'],key),
            mean=float(m),ci_low=float(l),ci_high=float(h),seeds=len(values)))
    write_json(out/'diagnostic_aggregate.json',diagnostic_summary)
    # Conditional groups are descriptive, not cross-method causal effects.
    conditional=[]
    groups=defaultdict(list)
    for r in rows:
        if 'evidence_group' in r:groups[r['regime'],r['prefix_timing'],r['evaluation'],r['B'],r['method'],r['evidence_group']].append(r)
    for key,rs in groups.items():
        by=defaultdict(list)
        for r in rs:by[r['seed']].append(r['exploitation_mean'])
        m,l,h=bootstrap(by,1000)
        conditional.append(dict(zip(['regime','prefix_timing','evaluation','B','method','evidence_group'],key),
            worlds=len(rs),seeds=len(by),reward_mean=float(m),ci_low=float(l),ci_high=float(h),
            utilization=float(np.mean([r['optimal_combo_utilization'] for r in rs if r['optimal_combo_utilization'] is not None])) if rs[0]['world_type']!='type1' else None,
            evidence_but_failed=sum(r['discovered_but_not_utilized'] for r in rs)))
    write_json(out/'evidence_groups.json',conditional)
    # Training logs are actual sequential curves; show individual seeds, not iid-episode CIs.
    learning=defaultdict(list)
    costs=[]
    for source in map(Path,inputs):
        for path in source.glob('**/training.jsonl'):
            identity_path=path.parent/'identity.json'
            if not identity_path.exists():continue
            identity=json.loads(identity_path.read_text())
            records=[json.loads(l) for l in path.read_text().splitlines()]
            learning[identity['regime']].append((identity['seed'],records))
            costs.append(dict(**identity,**json.loads((path.parent/'costs.json').read_text())))
    write_json(out/'training_costs.json',costs)
    for regime,logs in learning.items():
        fig,axes=pyplot().subplots(1,2,figsize=(11,4))
        for seed,records in logs:
            es=[r for r in records if r['stage']=='meta_episode'];us=[r for r in records if r['stage']=='meta_update']
            axes[0].plot([r['episode'] for r in es],[r['exploitation_mean'] for r in es],label=f'seed {seed}')
            axes[1].plot([r['episode'] for r in us],[r['actor_loss'] for r in us],label=f'seed {seed}')
        axes[0].set(xlabel='Training episode',ylabel='K-window reward');axes[1].set(xlabel='Training episode',ylabel='Outer PPO actor loss')
        axes[0].legend();fig.tight_layout();save(fig,out/(regime+'_learning'))
    for timing in sorted({r['prefix_timing'] for r in rows}):
        rs=[r for r in rows if r['prefix_timing']==timing];folder=out/timing;folder.mkdir()
        aggregate(rs,folder,1000)
        from .adaptation import aggregate_speed
        aggregate_speed(rs,folder/'adaptation_speed',1000)
        lookup={(r['regime'],r['seed'],r['world_id'],r['evaluation'],r['B'],r['method']):r for r in rs}
        groups=defaultdict(lambda:defaultdict(list))
        for r in rs:
            if r['method']!='neural_td_meta':continue
            for other in ['neural_td_random','neural_td_coverage','neural_td_repeat','neural_td_existing']:
                b=lookup.get((r['regime'],r['seed'],r['world_id'],r['evaluation'],r['B'],other))
                if not b:continue
                for metric in ['exploitation_mean','total_return','optimal_combo_utilization']:
                    if r[metric] is not None:
                        groups[r['regime'],r['world_type'],r['evaluation'],r['B'],other,metric][r['seed']].append(r[metric]-b[metric])
        for key,values in groups.items():
            m,l,h=bootstrap(values,1000)
            paired.append(dict(zip(['regime','world_type','evaluation','B','reference','metric'],key),
                prefix_timing=timing,contrast='meta minus reference',mean=float(m),ci_low=float(l),ci_high=float(h),
                seeds=len(values),pairs=sum(map(len,values.values()))))
        for regime in sorted({r['regime'] for r in rs}):
            fig,axes=pyplot().subplots(1,2,figsize=(12,4))
            for method in sorted({r['method'] for r in rs}):
                subset=[r for r in rs if r['regime']==regime and r['method']==method and r['evaluation']=='online_k']
                budgets=sorted({r['B'] for r in subset})
                for ax,metric in zip(axes,['exploitation_mean','optimal_combo_utilization']):
                    ys=[];lo=[];hi=[]
                    for B in budgets:
                        by=defaultdict(list)
                        for r in subset:
                            if r['B']==B and r[metric] is not None:by[r['seed']].append(r[metric])
                        if not by:continue
                        m,l,h=bootstrap(by,1000);ys.append(m);lo.append(l);hi.append(h)
                    if len(ys)==len(budgets):ax.errorbar(budgets,ys,yerr=[np.maximum(0,np.array(ys)-lo),np.maximum(0,np.array(hi)-ys)],label=method,marker='o',color=COLORS.get(method))
                    ax.set(xlabel='Exploration B',ylabel=metric)
            axes[0].legend(fontsize=6);fig.tight_layout();save(fig,folder/(regime+'_budget'))
    write_json(out/'paired_comparison.json',paired)
    write_json(out/'paired_adaptation.json',paired_td_speed(rows))
    write_json(out/'complete.json',dict(rows=len(rows),inputs=list(map(str,inputs))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','evaluate','gate','summarize'])
    p.add_argument('--config',default='configs/meta/neural_td.json');p.add_argument('--regime',choices=REGIMES,default='A3')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--source');p.add_argument('--existing-root');p.add_argument('--inputs',nargs='+');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=config(a.config);torch.set_num_threads(c['hardware']['threads_per_worker'])
    if a.command=='gate':
        if not gate(c,a.out,a.device):raise SystemExit('Deterministic mechanism gate failed; do not launch full study')
    elif a.command=='summarize':summarize(a.inputs or [],a.out)
    elif a.command=='evaluate':evaluate(c,a.regime,a.seed,a.source,a.out,a.device,a.existing_root)
    else:
        start=time.monotonic();train(c,a.regime,a.seed,a.out,a.device,a.resume)
        eval_out=Path(a.out)/'evaluation'
        if not (eval_out/'complete.json').exists():evaluate(c,a.regime,a.seed,a.out,eval_out,a.device,a.existing_root)
        write_json(Path(a.out)/'runtime.json',dict(total_seconds=time.monotonic()-start,device=a.device))
        write_json(Path(a.out)/'complete.json',dict(regime=a.regime,seed=a.seed))

if __name__=='__main__':main()
