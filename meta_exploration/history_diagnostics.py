"""Checkpoint-only history interventions. Never modifies source checkpoints.

Alternative observed histories are collected in independent sandbox rollouts;
attaching them to the common anchor is an explicitly counterfactual context test.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .diagnostics import load_models, intervene
from .learning import Controller, QLearner, collect, adapt, clone_model, assert_equal
from .training import world_set, make_game, HistoryCollector, write_json, provenance
from .evaluation import metrics, public_rows, diagnostic_rows, combo_metrics


def digest_state(game):
    """State and random streams, not mutable policy objects."""
    data = dict(t=game.t, mode=game.mode, previous=game.previous, reward=game.reward,
                other_reward=game.other_reward, opponent=game.opponent_kind,
                noise=game.noise_rng.bit_generator.state, opponent_rng=game.opponent_rng.bit_generator.state,
                hidden=None if game.opponent_hidden is None else game.opponent_hidden.cpu().tolist())
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def observed_stats(history):
    """Targets from the exact observation tokens encoded by ingest, no last-token leak.

    Each token describes the preceding transition. Thus H[-1].reward is excluded
    here and consumed only when the actual post-B observation is processed.
    """
    counts = np.zeros(12); sums = counts.copy(); squares = counts.copy()
    opponents = np.zeros(3)
    for i,t in enumerate(history):
        o = t.observation
        if o.previous_own >= 3:
            continue
        a = o.previous_own
        key = a if o.mode == 1 or i == 0 or history[i-1].observation.previous_own >= 3 else 3+3*history[i-1].observation.previous_own+a
        counts[key] += 1; sums[key] += o.reward; squares[key] += o.reward**2
        if o.previous_other < 3: opponents[o.previous_other] += 1
    mean = sums/np.maximum(counts,1)
    variance = np.maximum(0,squares/np.maximum(counts,1)-mean**2)
    last = history[-1].observation.vector()[:10] if history else np.zeros(10)
    return np.concatenate((counts,mean,variance,(counts==0).astype(float),opponents,last))


@torch.no_grad()
def probabilities(model, history, observation):
    ctrl = Controller(model,0); ctrl.ingest(history)
    logits,_,hidden = model(torch.as_tensor(observation.vector(),device=ctrl.device)[None,None],ctrl.hidden)
    return logits[0,0,0].softmax(-1).cpu().numpy(), hidden


@torch.no_grad()
def ordered_probability(model, history, snapshot, pair):
    """One actual-RNG conditional branch, not an unconditional product estimate.

    At R1, force first action in a private snapshot then evaluate R2. The same
    opponent/noise response is used across interventions. Odd B is not reset.
    """
    p,h = probabilities(model,history,snapshot.observation())
    a,c = pair
    if snapshot.mode:
        return dict(first_action_probability=None, second_action_probability=float(p[c]) if snapshot.previous[0]==a else None,
                    ordered_product_probability=None, ordered_probe='R2 handoff; no forced mode reset')
    branch = snapshot.snapshot(); obs,_,_,_ = branch.step(a)
    device=next(model.parameters()).device
    logits,_,_ = model(torch.as_tensor(obs.vector(),device=device)[None,None],h)
    pc=float(logits[0,0,0].softmax(-1)[c])
    return dict(first_action_probability=float(p[a]), second_action_probability=pc,
                ordered_product_probability=float(p[a])*pc,
                ordered_probe='product conditional on one shared snapshot RNG response; not marginal over reward/opponent')


def action_shuffle(history, seed):
    h=copy.deepcopy(history);rng=np.random.default_rng(seed)
    # Shuffle joint action tokens together; preserve the final pair and modes.
    stop=max(0,len(h)-2); order=rng.permutation(stop)
    old=[(t.action,t.opponent_action) for t in h[:stop]]
    for i,j in enumerate(order): h[i].action,h[i].opponent_action=old[j]
    from .environment import Observation
    for i,t in enumerate(h):
        if i:
            p=h[i-1];t.observation=Observation(t.observation.mode,p.action,p.opponent_action,p.reward)
        t.next_observation=Observation(t.next_observation.mode,t.action,t.opponent_action,t.reward)
    return h


def evidence_score(history):
    buckets=defaultdict(list)
    for t in history:
        if t.observation.mode: buckets[t.observation.previous_own,t.action].append(t.reward)
    # Observed-only selection. No requirement that the selected candidate is true.
    scores=[np.mean(v)-1.96*np.std(v)/np.sqrt(len(v)) for v in buckets.values() if len(v)>=2]
    return float(max(scores)) if scores else -1e30


def observed_evidence_candidate(initial, B, seed):
    """Cover pairs, then repeat best OBSERVED pair to obtain repeated evidence."""
    game=initial.snapshot()
    n=min(B,max(2,2*(B//4)))
    history=collect(game,HistoryCollector(seed,'coverage'),n)
    buckets=defaultdict(list)
    for t in history:
        if t.observation.mode:buckets[t.observation.previous_own,t.action].append(t.reward)
    if n<B:
        actor=HistoryCollector(seed,'coverage')
        if buckets:
            a,c=max(buckets,key=lambda p:np.mean(buckets[p]))
            actor.pairs=[a*3+c]*9
        history+=collect(game,actor,B-n)
    return history


def contexts(initial, models, config, seed, candidates=4):
    B=config['training']['exploration_steps']
    anchor=initial.snapshot(); real=collect(anchor,Controller(models['explorer'],seed),B)
    result={'real':real,'reward_shuffled':intervene(real,'reward_shuffled',seed),
            'action_shuffled':action_shuffle(real,seed),'reset':[]}
    costs={'real':B,'reward_shuffled':0,'action_shuffled':0,'reset':0}
    for strategy in ['coverage','random','repeat','q_learning']:
        actor=QLearner(config['q_learning'],seed) if strategy=='q_learning' else HistoryCollector(seed,strategy)
        result[strategy]=collect(initial.snapshot(),actor,B,online_q=strategy=='q_learning');costs[strategy]=B
    pool=[observed_evidence_candidate(initial,B,seed+7919*(i+1)) for i in range(candidates)]
    result['successful_observed']=max(pool,key=evidence_score);costs['successful_observed']=B*candidates
    return result,anchor,costs


def evidence_curve(model, history, pair):
    """Diagnostic probability at actual historical states; weights remain fixed."""
    rows=[]; positive=0; attempts=0
    for i in range(len(history)+1):
        obs=history[i].observation if i<len(history) else history[-1].next_observation
        p,_=probabilities(model,history[:i],obs)
        rows.append(dict(observed_steps=i,mode=obs.mode,good_pair_attempts=attempts,
                         positive_good_pair_rewards=positive,action_probs=p.tolist()))
        if i<len(history):
            t=history[i]
            if t.observation.mode and (t.observation.previous_own,t.action)==pair:
                attempts+=1;positive+=int(t.reward>0)
    return rows


def run_world(config, models, world, seed, mode, candidates=4):
    B,K=[config['training'][k] for k in ['exploration_steps','exploitation_steps']]
    initial=make_game(world,B+K,seed,config)
    histories,anchor,costs=contexts(initial,models,config,seed,candidates)
    state_hash=digest_state(anchor);rows=[];probes=[]
    # Hidden labels are used ONLY below for diagnostic grouping and forced probes.
    pair=tuple(int(v) for v in np.unravel_index((world.base.mean(1)[:,None]+world.base.mean(1)[None,:]+world.bonus.mean((1,3))).argmax(),(3,3)))
    for model_key in ['single_policy','dual_neural']:
        model=models[model_key]
        initial_p,_=probabilities(model,[],initial.observation());preferred=int(initial_p.argmax())
        baseline_p,_=probabilities(model,[],anchor.observation())
        for kind,history in histories.items():
            ctrl=Controller(clone_model(model),seed+10000);ctrl.ingest(history)
            if kind=='real':
                probes.append(dict(model=model_key,world_id=world.world_id,
                    hidden=ctrl.hidden.detach().cpu().flatten().tolist() if ctrl.hidden is not None else [0.]*model.hidden_size,
                    targets=observed_stats(history).tolist()))
            p,_=probabilities(model,history,anchor.observation())
            ordered=ordered_probability(model,history,anchor,pair)
            optimizer=torch.optim.Adam(ctrl.model.parameters(),lr=config['training']['learning_rate'])
            start=time.monotonic();suffix,updates=adapt(anchor.snapshot(),ctrl,K,optimizer,config,mode=='online')
            if mode=='frozen':assert_equal(ctrl.model,model)
            # Actual environment prefix always used for metrics / total cost.
            score=metrics(histories['real']+suffix,world,B,anchor.opponent_kind,config)
            diagnostic=list(diagnostic_rows(suffix,world))
            score['mean_combo_component']=float(np.mean([r['true_bonus'] for r in diagnostic]))
            score['mean_base_component']=float(np.mean([r['true_mean']-r['true_bonus'] for r in diagnostic]))
            matched=preferred==pair[0] if world.world_type=='type2' else None
            starts=[t for t in suffix if t.observation.mode==0]
            seconds=[t for t in suffix if t.observation.mode==1]
            empirical=sum((t.observation.previous_own,t.action)==pair for t in seconds)/max(1,len(seconds))
            row=dict(method=model_key,intervention=kind,evaluation=mode,world_id=world.world_id,world_type=world.world_type,
                snapshot_hash=state_hash,initial_preferred_action=preferred,optimal_pair=list(pair),preferred_matches=matched,
                action_probs=p.tolist(),optimal_action_probability=float(p[pair[0]]),
                optimal_probability_change=float(p[pair[0]]-baseline_p[pair[0]]),
                preferred_probability_change=float(p[preferred]-baseline_p[preferred]),
                switching_success=bool(p.argmax()==pair[0]),
                kl_from_reset=float(np.sum(p*np.log(np.maximum(p,1e-12)/np.maximum(baseline_p,1e-12)))),
                tv_from_reset=float(np.abs(p-baseline_p).sum()/2),
                empirical_ordered_completion=empirical, empirical_first_action=sum(t.action==pair[0] for t in starts)/max(1,len(starts)),
                empirical_second_given_first=(sum((t.observation.previous_own,t.action)==pair for t in seconds)/sum(t.observation.previous_own==pair[0] for t in seconds)) if any(t.observation.previous_own==pair[0] for t in seconds) else None,
                mode_action_probabilities={str(m):np.mean([t.info['action_probs'] for t in suffix if t.observation.mode==m],axis=0).tolist() for m in (0,1) if any(t.observation.mode==m for t in suffix)},
                context_combo_diagnostics=combo_metrics(history+suffix,world,len(history)),
                context_evidence_score=evidence_score(history), observed_repeated_positive_evidence=evidence_score(history)>0,
                context_generation_interactions=costs[kind],
                intervention_semantics='actual prefix' if kind=='real' else 'reset memory; actual boundary observation retained' if kind=='reset' else 'counterfactual hidden-state context; environment never replayed',
                source_history='synthetic' if 'shuffled' in kind else 'observed',
                actual_rollout_interactions=B+K,optimizer_steps=sum(u['updates'] for u in updates),
                optimizer_seconds=sum(u['update_seconds'] for u in updates),runtime_seconds=time.monotonic()-start,
                ordered_diagnostic=ordered, metrics=score,
                evidence_curve=evidence_curve(model,history,pair) if history else [],
                suffix=list(public_rows(suffix,world,model_key,0)))
            row['speed_summary']=speed_scalars(score['adaptation_speed'],B)
            rows.append(row)
        assert_equal(model,models[model_key])
    # Q counterpart only for actual observed histories: no fabricated TD transitions.
    for kind,history in histories.items():
        if 'shuffled' in kind:continue
        q=QLearner(config['q_learning'],seed+10000);q.replay(history)
        suffix=collect(anchor.snapshot(),q,K,online_q=True)
        rows.append(dict(method='q_table',intervention=kind,evaluation='online_q',world_id=world.world_id,
            world_type=world.world_type,snapshot_hash=state_hash,source_history='observed',
            context_generation_interactions=costs[kind],actual_rollout_interactions=B+K,
            td_updates=q.update_count,td_update_seconds=q.update_seconds,
            context_combo_diagnostics=combo_metrics(history+suffix,world,len(history)),
            metrics=metrics(histories['real']+suffix,world,B,anchor.opponent_kind,config)))
    for r in rows:
        if 'speed_summary' not in r:r['speed_summary']=speed_scalars(r['metrics']['adaptation_speed'],B)
    assert digest_state(anchor)==state_hash
    return rows,probes


def linear_probe(rows):
    """Leave one training seed out; world identity also disjoint across folds.

    Standardization and ridge fit use training folds only. Ridge penalty is fixed.
    """
    output=[]
    for model in sorted({r['model'] for r in rows}):
        source=[r for r in rows if r['model']==model]
        seeds=sorted({r['seed'] for r in source})
        if len(seeds)<2:
            output.append(dict(model=model,status='undefined: at least two checkpoint seeds required'));continue
        for seed in seeds:
            train=[r for r in source if r['seed']!=seed];test=[r for r in source if r['seed']==seed]
            assert not {r['world_id'] for r in train}&{r['world_id'] for r in test}
            x,y=np.array([r['hidden'] for r in train]),np.array([r['targets'] for r in train])
            tx,ty=np.array([r['hidden'] for r in test]),np.array([r['targets'] for r in test])
            xm,xs=x.mean(0),x.std(0)+1e-6;ym,ys=y.mean(0),y.std(0)+1e-6
            x=np.column_stack(((x-xm)/xs,np.ones(len(x))))
            tx=np.column_stack(((tx-xm)/xs,np.ones(len(tx))))
            penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
            w=np.linalg.solve(x.T@x+penalty,x.T@((y-ym)/ys))
            pred=(tx@w)*ys+ym;mse=((pred-ty)**2).mean(0);baseline=((ym-ty)**2).mean(0)
            output.append(dict(model=model,heldout_seed=seed,n_train=len(train),n_test=len(test),
                mse=mse.tolist(),train_mean_baseline_mse=baseline.tolist(),
                skill_vs_train_mean=[float(1-a/b) if b>1e-10 else None for a,b in zip(mse,baseline)],
                groups=dict(counts=[0,12],reward_mean=[12,24],reward_variance=[24,36],unobserved_mask=[36,48],opponent_counts=[48,51],last_mode_action=[51,61]),
                caveat='Across-checkpoint latent coordinates need not align; failure is not evidence of no decodable information.'))
    return output


def within_checkpoint_probes(rows):
    """World-disjoint linear readout in each fixed checkpoint coordinate system."""
    result=[]
    for model,seed in sorted({(r['model'],r['seed']) for r in rows}):
        rs=sorted([r for r in rows if r['model']==model and r['seed']==seed],key=lambda r:r['world_id'])
        if len(rs)<4:
            result.append(dict(model=model,seed=seed,status='undefined: fewer than four independent worlds'));continue
        train=[r for i,r in enumerate(rs) if i%3];test=[r for i,r in enumerate(rs) if not i%3]
        x=np.array([r['hidden'] for r in train]);y=np.array([r['targets'] for r in train])
        tx=np.array([r['hidden'] for r in test]);ty=np.array([r['targets'] for r in test])
        xm,xs=x.mean(0),x.std(0)+1e-6;ym,ys=y.mean(0),y.std(0)+1e-6
        x=np.column_stack(((x-xm)/xs,np.ones(len(x))));tx=np.column_stack(((tx-xm)/xs,np.ones(len(tx))))
        penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
        w=np.linalg.solve(x.T@x+penalty,x.T@((y-ym)/ys))
        mse=(((tx@w)*ys+ym-ty)**2).mean(0);baseline=((ym-ty)**2).mean(0)
        result.append(dict(model=model,seed=seed,n_train=len(train),n_test=len(test),
            train_worlds=[r['world_id'] for r in train],test_worlds=[r['world_id'] for r in test],
            mse=mse.tolist(),train_mean_baseline_mse=baseline.tolist(),
            skill_vs_train_mean=[float(1-a/b) if b>1e-10 else None for a,b in zip(mse,baseline)]))
    return result


def preference_summary(rows):
    groups=defaultdict(list)
    for r in rows:
        if 'switching_success' not in r:continue
        groups[r['regime'],r['method'],r['evaluation'],r['intervention'],r.get('preferred_matches')].append(r)
    result=[]
    for key,rs in groups.items():
        result.append(dict(zip(['regime','method','evaluation','intervention','preferred_matches'],key),
            worlds=len(rs),seeds=len({r['seed'] for r in rs}),
            switching_rate=float(np.mean([r['switching_success'] for r in rs])),
            reward=float(np.mean([r['metrics']['exploitation_mean'] for r in rs])),
            optimal_utilization=float(np.mean([r['metrics']['optimal_combo_utilization'] for r in rs if r['metrics']['optimal_combo_utilization'] is not None])) if rs[0]['world_type']!='type1' else None,
            ordered={k:float(np.mean([r['ordered_diagnostic'][k] for r in rs if r['ordered_diagnostic'][k] is not None])) if any(r['ordered_diagnostic'][k] is not None for r in rs) else None for k in ['first_action_probability','second_action_probability','ordered_product_probability']}))
    return result


def speed_scalars(speed,B):
    result={}
    for name in ['absolute','baseline_normalized']:
        block=speed[name]
        result[name]={'defined':block['defined']}
        if not block['defined']:continue
        result[name].update(normalized_adaptation_auc=block['normalized_adaptation_auc'],
                            final_window_oracle_ratio=block['final_window_oracle_ratio'])
        for q,e in block['events'].items():
            result[name]['reach_'+q]=int(e['event'])
            result[name]['restricted_time_'+q]=e['restricted_time']
            result[name]['total_restricted_time_'+q]=B+e['restricted_time']
    return result


def paired_summary(rows, replicates=1000):
    from .plotting import bootstrap
    keys=['regime','method','evaluation','seed','world_id']
    lookup={tuple(r[k] for k in keys)+(r['intervention'],):r for r in rows}
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        ref=lookup.get(tuple(row[k] for k in keys)+('real',))
        if ref is None or row['intervention']=='real':continue
        for metric in ['exploitation_mean','optimal_combo_utilization']:
            a,b=row['metrics'].get(metric),ref['metrics'].get(metric)
            if a is not None and b is not None:
                for subset in ['all']+(['preference_mismatch'] if row.get('preferred_matches') is False else []):
                    groups[row['regime'],row['method'],row['evaluation'],row['intervention'],subset,metric][row['seed']].append(a-b)
        for normalization,block in row.get('speed_summary',{}).items():
            reference=ref.get('speed_summary',{}).get(normalization,{})
            if block.get('defined') and reference.get('defined'):
                for metric,a in block.items():
                    b=reference.get(metric)
                    if metric!='defined' and a is not None and b is not None:
                        groups[row['regime'],row['method'],row['evaluation'],row['intervention'],'all',normalization+'/'+metric][row['seed']].append(a-b)
        if 'optimal_action_probability' in row:
            groups[row['regime'],row['method'],row['evaluation'],row['intervention'],'all','optimal_action_probability'][row['seed']].append(row['optimal_action_probability']-ref['optimal_action_probability'])
    result=[]
    for key,values in groups.items():
        mean,lo,hi=bootstrap(values,replicates)
        result.append(dict(zip(['regime','method','evaluation','intervention','subset','metric'],key),
            contrast='intervention minus real',mean=float(mean),ci_low=float(lo),ci_high=float(hi),
            seeds=len(values),world_pairs=sum(map(len,values.values())),seed_means={s:float(np.mean(v)) for s,v in values.items()}))
    return result


def run(root,out,regimes,seeds,worlds,modes,device,candidates=4):
    root,out=Path(root),Path(out)
    if candidates<1:raise ValueError('At least one evidence candidate required')
    sources=[]
    for regime in regimes:
        for seed in seeds:
            source=root/regime/f'seed_{seed}'
            if not (source/'checkpoints/trained.pt').exists():
                raise FileNotFoundError(f'Missing checkpoint {source}; no random model substituted')
            sources.append((regime,seed,source))
    if out.exists():raise ValueError('Output already exists; choose a new directory')
    out.mkdir(parents=True);all_rows=[];all_probes=[];manifest=[]
    for regime,seed,source in sources:
        c=json.loads((source/'config.json').read_text());torch.set_num_threads(c['hardware']['threads_per_worker'])
        if c['environment']['opponents']==['ppo'] or 'ppo' in c['environment']['opponents']:
            raise ValueError('This diagnostic runner currently supports constant/random/fixed opponents only')
        c=copy.deepcopy(c)
        if worlds is not None:c['evaluation']['num_test_worlds']=worlds
        checkpoint=source/'checkpoints/trained.pt'
        checkpoint_meta=torch.load(checkpoint,map_location='cpu',weights_only=False)
        original_config=json.loads((source/'config.json').read_text())
        if checkpoint_meta.get('config')!=original_config:
            raise ValueError(f'Checkpoint/config mismatch: {checkpoint}')
        initialization_audit={'status':'unavailable: initialization/pretrain checkpoints missing'}
        init_path=source/'checkpoints/identical_initialization.pt'
        pretrain_path=source/'checkpoints/shared_exploitation_pretrain.pt'
        if init_path.exists() and pretrain_path.exists():
            initial_states=torch.load(init_path,map_location='cpu',weights_only=False)
            pretrain_state=torch.load(pretrain_path,map_location='cpu',weights_only=False)['model']
            equal=all(torch.equal(v,initial_states['dual_neural'][k]) and torch.equal(v,pretrain_state[k]) for k,v in initial_states['single_policy'].items())
            if not equal:raise ValueError('Shared initialization equality failed')
            initialization_audit={'status':'verified Single = Dual exploiter = shared pretrain',
                'initialization_sha256':hashlib.sha256(init_path.read_bytes()).hexdigest(),
                'pretrain_sha256':hashlib.sha256(pretrain_path.read_bytes()).hexdigest()}
        models=load_models(c,checkpoint,device)
        manifest.append(dict(regime=regime,seed=seed,checkpoint=str(checkpoint.resolve()),initialization_audit=initialization_audit,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source_config=json.loads((source/'config.json').read_text()),evaluation_config=c,
            execution=provenance(c,device,seed)))
        path=out/f'{regime}_seed_{seed}.jsonl'
        for world in world_set(c,regime,seed,'test'):
            for mode in modes:
                rows,probes=run_world(c,models,world,seed*100000+world.seed,mode,candidates)
                if mode!=modes[0]:rows=[r for r in rows if r['method']!='q_table']
                for row in rows:row.update(regime=regime,seed=seed)
                with path.open('a') as f:
                    for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
                # Keep full trajectories on disk; aggregate compact rows in RAM.
                for r in rows:
                    compact={k:v for k,v in r.items() if k not in ('suffix','evidence_curve','metrics')}
                    compact['metrics']={k:v for k,v in r['metrics'].items() if k!='adaptation_speed'}
                    all_rows.append(compact)
                if mode==modes[0]:
                    for row in probes:row.update(regime=regime,seed=seed)
                    all_probes.extend(probes)
            print(f'{regime}/seed_{seed}/{world.world_id}',flush=True)
        write_json(out/'provenance.json',manifest)
    write_json(out/'paired_summary.json',paired_summary(all_rows))
    write_json(out/'preference_summary.json',preference_summary(all_rows))
    write_json(out/'within_checkpoint_probes.json',{r:within_checkpoint_probes([p for p in all_probes if p['regime']==r]) for r in regimes})
    write_json(out/'probe_samples.json',all_probes)
    # Probe separate regimes: different model training distributions not pooled.
    write_json(out/'probes.json',{r:linear_probe([p for p in all_probes if p['regime']==r]) for r in regimes})
    write_json(out/'complete.json',dict(rows=len(all_rows),checkpoint_runs=len(sources),training_performed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--regimes',nargs='+',default=['A2','A3']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--worlds',type=int);p.add_argument('--modes',nargs='+',choices=['frozen','online'],default=['frozen'])
    p.add_argument('--device',default='cpu');p.add_argument('--evidence-candidates',type=int,default=4)
    a=p.parse_args();run(a.checkpoint_root,a.out,a.regimes,a.seeds,a.worlds,a.modes,a.device,a.evidence_candidates)

if __name__=='__main__':main()
