"""Read-only analysis of uploaded full-study results; writes derived artifacts here."""
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from meta_exploration.environment import generate_world, oracle_values

ROOT=Path('examples/full_study'); OUT=Path('analysis/full_study')
METHODS=['single_policy','q_learning','dual_neural','dual_q_table']
REGIMES=['A1','A2','A3','B']


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),std=float(np.std(values)),
                min=float(np.min(values)),p10=float(np.quantile(values,.1)),median=float(np.median(values))) if values else None


def paired_summary(pairs, reps=5000):
    by_seed=defaultdict(list)
    for seed,delta in pairs: by_seed[seed].append(delta)
    arrays=[np.array(by_seed[s],dtype=float) for s in sorted(by_seed)]
    assert len(set(map(len,arrays)))==1
    data=np.stack(arrays);rng=np.random.default_rng(2026)
    selected=rng.integers(len(arrays),size=(reps,len(arrays)))
    selected_world=rng.integers(data.shape[1],size=(reps,len(arrays),data.shape[1]))
    estimates=data[selected[:,:,None],selected_world].mean(axis=(1,2))
    lo,hi=np.quantile(estimates,[.025,.975])
    return dict(mean=float(data.mean()),ci_low=float(lo),ci_high=float(hi),n=len(pairs),
                seed_means={str(s):float(np.mean(by_seed[s])) for s in sorted(by_seed)},
                world_win_fraction=float((data>0).mean()))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    rows=[]; configs=[]; costs=[]; file_hashes={}
    for path in sorted(ROOT.glob('*/seed_*/evaluation/metrics.json')):
        content=path.read_bytes();file_hashes[str(path)]=hashlib.sha256(content).hexdigest()
        job=path.parent.parent
        c=json.loads((job/'config.json').read_text());configs.append(c)
        costs.append(json.loads((job/'costs.json').read_text()))
        complete=json.loads((job/'complete.json').read_text());assert complete['status']=='complete'
        records=json.loads(content)
        assert len(records)==3200
        assert {r['regime'] for r in records}=={complete['regime']}
        assert {r['seed'] for r in records}=={complete['seed']}
        rows.extend(records)
    assert len(configs)==20 and all(c==configs[0] for c in configs)
    lookup={}
    for r in rows:
        key=(r['regime'],r['seed'],r['world_id'],r['evaluation'],r['B'],r['K'],r['method'])
        assert key not in lookup
        lookup[key]=r
        assert abs(r['exploitation_return']/r['K']-r['exploitation_mean'])<1e-10
    # Confirm the uploaded aggregate matches raw metrics, including sample counts.
    groups=defaultdict(list)
    for r in rows:
        for typ in [r['world_type'],'all']:
            groups[(r['regime'],r['evaluation'],r['B'],r['K'],r['method'],typ)].append(r)
    aggregate=json.loads((ROOT/'figures/aggregated.json').read_text())
    for r in aggregate:
        g=groups[tuple(r[k] for k in ['regime','evaluation','B','K','method','world_type'])]
        values=[x[r['metric']] for x in g if x.get(r['metric']) is not None]
        assert len(values)==r['n_worlds']
        seed_values=defaultdict(list)
        for x in g:
            if x.get(r['metric']) is not None: seed_values[x['seed']].append(x[r['metric']])
        assert abs(float(np.mean([np.mean(v) for v in seed_values.values()]))-r['mean'])<1e-8
    paired_checks=0; frozen_q_checks=0
    for r in rows:
        if r['method']=='dual_neural':
            q=lookup[(r['regime'],r['seed'],r['world_id'],r['evaluation'],r['B'],r['K'],'dual_q_table')]
            for key in ['combo_attempt_counts','rewarding_combo_attempted','rewarding_combo_supported','auc_32']:
                if key=='auc_32' and r['B']<32: continue
                assert r[key]==q[key],(key,r['regime'],r['world_id'])
            paired_checks+=1
        if r['method'] in ['q_learning','dual_q_table'] and r['evaluation']=='online':
            q=lookup[(r['regime'],r['seed'],r['world_id'],'frozen',r['B'],r['K'],r['method'])]
            assert r['exploitation_mean']==q['exploitation_mean']
            frozen_q_checks+=1
    main_rows=[r for r in rows if r['B']==32 and r['K']==128]
    main_aggregate=[r for r in aggregate if r['B']==32 and r['K']==128 and r['world_type']=='all' and r['metric']=='exploitation_mean']
    contrasts=[]
    for regime in REGIMES:
        for mode in ['online','frozen']:
            for a,b in [('dual_neural','single_policy'),('dual_q_table','q_learning'),('dual_q_table','dual_neural')]:
                selected=[r for r in main_rows if r['regime']==regime and r['evaluation']==mode and r['method']==a]
                pairs=[(r['seed'],r['exploitation_mean']-lookup[(regime,r['seed'],r['world_id'],mode,32,128,b)]['exploitation_mean']) for r in selected]
                contrasts.append(dict(regime=regime,evaluation=mode,contrast=a+' minus '+b,**paired_summary(pairs)))
        for method in ['single_policy','dual_neural']:
            selected=[r for r in main_rows if r['regime']==regime and r['evaluation']=='online' and r['method']==method]
            pairs=[(r['seed'],r['exploitation_mean']-lookup[(regime,r['seed'],r['world_id'],'frozen',32,128,method)]['exploitation_mean']) for r in selected]
            contrasts.append(dict(regime=regime,evaluation='online minus frozen',contrast=method,**paired_summary(pairs)))
    generalization=[]
    for typ,specialized in [('type1','A1'),('type2','A2'),('type3','A3')]:
        for method in METHODS:
            selected=[r for r in main_rows if r['regime']=='B' and r['world_type']==typ and r['evaluation']=='online' and r['method']==method]
            comparisons=[lookup[(specialized,r['seed'],r['world_id'],'online',32,128,method)] for r in selected]
            pairs=[(r['seed'],r['exploitation_mean']-v['exploitation_mean']) for r,v in zip(selected,comparisons)]
            generalization.append(dict(world_type=typ,method=method,mixed_mean=float(np.mean([r['exploitation_mean'] for r in selected])),
                 specialized_matched_mean=float(np.mean([r['exploitation_mean'] for r in comparisons])),**paired_summary(pairs)))
    diagnostics=[]; windows=[]; budgets=[]
    for regime in REGIMES:
        for method in METHODS:
            selected=[r for r in main_rows if r['regime']==regime and r['evaluation']=='online' and r['method']==method]
            combo=[r for r in selected if r['combo_applicable']]
            discovered=[r for r in combo if r['rewarding_combo_attempted']]
            attempts=[r['combo_attempt_counts'] for r in selected]
            diag=dict(regime=regime,method=method,n=len(selected),combo_n=len(combo),
                exploitation=stats([r['exploitation_mean'] for r in selected]),
                oracle_exploitation_mean=float(np.mean([r['oracle_exploitation_return_from_agent_state']/r['K'] for r in selected])),
                optimal_utilization=float(np.mean([r['optimal_combo_utilization'] for r in combo])) if combo else None,
                discovery=float(np.mean([r['rewarding_combo_attempted'] for r in combo])) if combo else None,
                supported=float(np.mean([r['rewarding_combo_supported'] for r in combo])) if combo else None,
                conditional_utilization=float(np.mean([r['discovered_combo_utilization'] for r in discovered])) if discovered else None,
                conditional_exploitation=float(np.mean([r['exploitation_mean'] for r in discovered])) if discovered else None,
                discovered_n=len(discovered),unique_pairs=float(np.mean([len(x) for x in attempts])),
                same_action_attempt_fraction=sum(sum(v for k,v in x.items() if k[0]==k[1]) for x in attempts)/sum(sum(x.values()) for x in attempts),
                history_transfer_gap_per_step=float(np.mean([r['history_transfer_gap']/r['K'] for r in selected])) if 'history_transfer_gap' in selected[0] else None)
            diagnostics.append(diag)
            windows.append(dict(regime=regime,method=method,means=[float(np.mean([f(r) for r in selected])) for f in [
                lambda r:r['auc_32']/32, lambda r:(r['auc_64']-r['auc_32'])/32,
                lambda r:(r['auc_128']-r['auc_64'])/64, lambda r:(r['total_return']-r['auc_128'])/32]]))
            for B in [8,16,32,64]:
                selected=[r for r in rows if r['regime']==regime and r['evaluation']=='online' and r['method']==method and r['B']==B]
                budgets.append(dict(regime=regime,method=method,B=B,exploitation_mean=float(np.mean([r['exploitation_mean'] for r in selected])),
                                    total_mean=float(np.mean([r['total_return']/(r['B']+r['K']) for r in selected])),
                                    auc128_mean=float(np.mean([r['auc_128']/128 for r in selected]))))
    # Regenerate immutable worlds to check oracle consistency and analytical controls.
    cache={}; analytic=[]
    env=configs[0]['environment']
    for regime in REGIMES:
        selected=[r for r in main_rows if r['regime']==regime and r['evaluation']=='online' and r['method']=='single_policy']
        vals=[];greedies=[]
        for r in selected:
            key=(r['world_id'],r['opponent'])
            if key not in cache:
                w=generate_world(int(r['world_id'].rsplit('-',1)[1]),r['world_type'],'test',env['difficulty'],env['reward_sigma'],env['opponent_dependence'])
                value=oracle_values(w,160,r['opponent'])[0,3,3]
                assert abs(value-r['oracle_expected_return'])<1e-8
                uni=[];greedy=[];a=int(w.base.mean(axis=1).argmax())
                for t in range(32,160):
                    if r['opponent']=='random':
                        uni.append(float(w.base.mean()+(w.bonus.mean() if t%2 else 0)))
                        greedy.append(float(w.base[a].mean()+(w.bonus[a,:,a,:].mean() if t%2 else 0)))
                    else:
                        b=t%3;bp=(t-1)%3
                        uni.append(float(w.base[:,b].mean()+(w.bonus[:,bp,:,b].mean() if t%2 else 0)))
                        greedy.append(float(w.base[a,b]+(w.bonus[a,bp,a,b] if t%2 else 0)))
                cache[key]=(float(np.mean(uni)),float(np.mean(greedy)))
            u,g=cache[key];vals.append(u);greedies.append(g)
        analytic.append(dict(regime=regime,uniform_random_expected_exploitation_mean=float(np.mean(vals)),
                             best_immediate_fixed_action_expected_mean=float(np.mean(greedies)),
                             note='Analytical hidden-parameter diagnostic; not a trained/evaluated agent. Exact finite-horizon expectations with iid/cyclic opponent.'))
    by_opponent=[]
    for regime in REGIMES:
        for method in METHODS:
            for opp in ['random','fixed']:
                v=[r['exploitation_mean'] for r in main_rows if r['regime']==regime and r['method']==method and r['opponent']==opp and r['evaluation']=='online']
                by_opponent.append(dict(regime=regime,method=method,opponent=opp,**stats(v)))
    payload=dict(audit=dict(jobs=len(configs),raw_rows=len(rows),aggregate_rows_checked=len(aggregate),
        duplicate_rows=0,paired_prefix_checks=paired_checks,frozen_online_q_equal_checks=frozen_q_checks,
        regenerated_oracle_world_checks=len(cache),config=configs[0],costs=costs[0],source_files_sha256=file_hashes,
        unavailable=['training.jsonl','per-step JSONL histories','checkpoints','runtime.json','hardware/utilization logs','execution source commit']),
        main=main_aggregate,contrasts=contrasts,generalization_matched=generalization,
        diagnostics=diagnostics,adaptation_windows=windows,budgets=budgets,analytical_controls=analytic,by_opponent=by_opponent)
    (OUT/'analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    for name in ['contrasts','generalization_matched','diagnostics','adaptation_windows','budgets','analytical_controls','by_opponent']:
        data=payload[name]
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader()
            writer.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data])
    print('AUDIT', {k:v for k,v in payload['audit'].items() if k not in ['config','costs','source_files_sha256']})
    for name in ['analytical_controls','contrasts','generalization_matched','diagnostics','adaptation_windows','budgets']:
        print('\n'+name)
        for r in payload[name]:
            if name=='contrasts' and r['evaluation']=='frozen':continue
            print(json.dumps(r))


if __name__=='__main__':main()
