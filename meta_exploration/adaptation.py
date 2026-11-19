"""Evaluation-only finite-horizon references and censored adaptation speed.

Full trailing windows only: first eligible observation is W, earliest confirmed
crossing W+hold-1. Reference is K-step optimal expected return / K from actual
post-B state, NOT a stationary average or an oracle action supplied to an agent.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from .environment import oracle_values


def reference_curves(world, B, K, previous, opponent, random_agent=False):
    values = oracle_values(world, B+K, opponent, random_agent=random_agent)
    occupancy = {tuple(previous): 1.}
    rewards = []
    for t in range(B, B+K):
        bp = np.full(3, 1/3) if opponent == 'random' else np.eye(3)[0 if opponent == 'constant' else t % 3]
        nxt, expected = defaultdict(float), 0.
        for (pa, pb), mass in occupancy.items():
            means = world.base + (world.bonus[pa, pb] if t % 2 else 0.)
            q = ((means + values[t+1,:3,:3])*bp).sum(-1)
            ap = np.full(3, 1/3) if random_agent else np.eye(3)[q.argmax()]
            for a in range(3):
                for b in range(3):
                    prob = mass*ap[a]*bp[b]
                    expected += prob*means[a,b]
                    nxt[a,b] += prob
        occupancy = nxt
        rewards.append(float(expected))
    return np.asarray(rewards)


def speed_metrics(rewards, oracle_curve, random_curve, settings=None, expected_rewards=None):
    c = dict(window=16, hold=8, thresholds=[.25,.5,.75,.9], denominator_epsilon=1e-8)
    c.update(settings or {})
    y = np.asarray(rewards, dtype=float); K = len(y)
    if not K or c['window'] < 1 or c['hold'] < 1 or not np.isfinite(y).all():
        raise ValueError('Finite nonempty rewards and positive window/hold required')
    W = c['window']
    smooth = np.full(K, np.nan)
    if K >= W:
        smooth[W-1:] = np.convolve(y, np.ones(W)/W, mode='valid')
    result = dict(K=K, window=W, hold=c['hold'], thresholds=c['thresholds'],
                  first_eligible_step=W, reference='finite-horizon K-step mean from actual post-B state',
                  smoothed_rewards=[float(v) if np.isfinite(v) else None for v in smooth])
    for name in ('absolute', 'baseline_normalized'):
        result[name] = dict(defined=False, reason='oracle_unavailable')
    if oracle_curve is None or random_curve is None:
        return result
    oracle_curve, random_curve = np.asarray(oracle_curve), np.asarray(random_curve)
    if len(oracle_curve) != K or len(random_curve) != K:
        raise ValueError('Reference length mismatch')
    oracle, random = float(oracle_curve.mean()), float(random_curve.mean())
    result.update(oracle_expected_reward=oracle, random_expected_reward=random,
                  oracle_expected_curve=oracle_curve.tolist(), random_expected_curve=random_curve.tolist(),
                  cumulative_oracle_regret=np.cumsum(oracle_curve-y).tolist())
    if expected_rewards is not None:
        result['cumulative_expected_oracle_regret'] = np.cumsum(oracle_curve-np.asarray(expected_rewards)).tolist()
    for name, offset, denominator in [('absolute',0.,oracle), ('baseline_normalized',random,oracle-random)]:
        if denominator <= c['denominator_epsilon']:
            result[name] = dict(defined=False, reason='nonpositive_or_near_zero_denominator', denominator=denominator)
            continue
        curve = (smooth-offset)/denominator
        valid = curve[np.isfinite(curve)]
        events = {}
        for threshold in c['thresholds']:
            run, reached = 0, None
            for step, value in enumerate(curve, 1):
                run = run+1 if np.isfinite(value) and value >= threshold else 0
                if run >= c['hold']:
                    reached = step
                    break
            events[f'{threshold:g}'] = dict(time=reached, event=reached is not None,
                censored_at=K, restricted_time=reached if reached is not None else K)
        result[name] = dict(defined=True, denominator=denominator, curve=[float(v) if np.isfinite(v) else None for v in curve],
            events=events, normalized_adaptation_auc=float(valid.sum()) if len(valid) else None,
            normalized_adaptation_auc_mean=float(valid.mean()) if len(valid) else None,
            auc_observed_steps=len(valid), final_window_oracle_ratio=float(curve[-1]) if len(valid) else None,
            **{f'time_to_{int(round(q*100))}pct':events[f'{q:g}']['time'] for q in c['thresholds']})
    return result


def history_speed(history, world, B, opponent, config):
    K = len(history)-B
    o = history[B].observation
    refs = [None,None]
    if opponent in ('random','fixed','constant'):
        refs = [reference_curves(world,B,K,(o.previous_own,o.previous_other),opponent,r) for r in (False,True)]
    expected = [world.mean(t.observation.mode,(t.observation.previous_own,t.observation.previous_other),t.action,t.opponent_action) for t in history[B:]]
    return speed_metrics([t.reward for t in history[B:]], *refs,
                         config['evaluation'].get('adaptation_speed', {}), expected)


def aggregate_speed(rows, output, replicates=500):
    """Equal-seed estimates; paired bootstrap uses same seed/world differences.

    All censoring is administrative at K. Thus KM is empirical survival and
    RMST[0,K]=mean(min(T,K)), including every eligible nonreacher.
    Undefined references are excluded with explicit counts, never nonreachers.
    """
    from .plotting import bootstrap, pyplot, save, COLORS
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    keys = ('regime','evaluation','world_type','B','K')
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        if 'adaptation_speed' not in r:
            continue
        identity = tuple(r[k] for k in keys)+ (r['method'],r['seed'],r['world_id'])
        if identity in seen:
            raise ValueError('Duplicate world/method identity; aggregate distinct training variants separately')
        seen.add(identity)
        groups[tuple(r[k] for k in keys)].append(r)
        groups[tuple(r[k] if k!='world_type' else 'all' for k in keys)].append(r)
    if not groups:
        raise ValueError('No step-derived speed data. Aggregate rewards cannot reconstruct crossing times; use raw JSONL backfill or re-evaluate.')
    aggregates, paired = [], []
    plt = pyplot()
    for group, items in groups.items():
        for normalization in ('absolute','baseline_normalized'):
            thresholds = items[0]['adaptation_speed']['thresholds']
            curves, arrivals, scalar_panels = {}, defaultdict(dict), defaultdict(dict)
            for method in sorted({r['method'] for r in items}):
                all_method = [r for r in items if r['method']==method]
                valid = [r for r in all_method if r['adaptation_speed'][normalization]['defined']]
                base = dict(zip(keys,group), method=method, normalization=normalization,
                            n_worlds=len(valid), n_undefined=len(all_method)-len(valid), n_seeds=len({r['seed'] for r in valid}))
                if not valid:
                    aggregates.append(dict(base, metric='eligibility', mean=None, ci_low=None, ci_high=None)); continue
                measures = {}
                for r in valid:
                    s = r['adaptation_speed'][normalization]
                    d = {k:s[k] for k in ('normalized_adaptation_auc','normalized_adaptation_auc_mean','final_window_oracle_ratio')}
                    d['cumulative_oracle_regret'] = r['adaptation_speed']['cumulative_oracle_regret'][-1]
                    for q, event in s['events'].items():
                        d[f'reach_at_K_{q}'] = float(event['event'])
                        d[f'rmst_{q}'] = event['restricted_time']
                        d[f'total_interaction_rmst_{q}'] = r['B']+event['restricted_time']
                    measures[r['seed'],r['world_id']] = d
                for metric in next(iter(measures.values())):
                    by_seed = defaultdict(list)
                    for (seed,_), d in measures.items():
                        if d[metric] is not None:
                            by_seed[seed].append(d[metric])
                    if not by_seed: continue
                    mean, lo, hi = bootstrap(by_seed,replicates)
                    aggregates.append(dict(base, metric=metric,mean=float(mean),ci_low=float(lo),ci_high=float(hi)))
                    scalar_panels[metric][method] = (float(mean),float(lo),float(hi))
                by_seed = defaultdict(list)
                first = valid[0]['adaptation_speed']['window']-1
                if first < group[-1]:
                    for r in valid: by_seed[r['seed']].append(r['adaptation_speed'][normalization]['curve'][first:])
                    curves[method] = (first,bootstrap(by_seed,replicates))
                for threshold in thresholds:
                    by_seed = defaultdict(list)
                    for r in valid:
                        event = r['adaptation_speed'][normalization]['events'][f'{threshold:g}']
                        by_seed[r['seed']].append([float(event['event'] and event['time']<=t) for t in range(1,r['K']+1)])
                    arrivals[threshold][method] = bootstrap(by_seed,replicates)
            stem = '_'.join(map(str,group))+'_'+normalization
            fig, ax = plt.subplots(figsize=(8,4))
            for method,(first,(mean,lo,hi)) in curves.items():
                x=np.arange(first+1,group[-1]+1); color=COLORS.get(method)
                ax.plot(x,mean,label=method,color=color); ax.fill_between(x,lo,hi,alpha=.15,color=color)
            for q in thresholds: ax.axhline(q,color='gray',ls=':',lw=.7)
            ax.set(xlabel='Exploitation step',ylabel=normalization,title=str(group));
            if curves: ax.legend(fontsize=7)
            save(fig,output/('A_curve_'+stem))
            fig, axes = plt.subplots(1,len(thresholds),figsize=(4*len(thresholds),3),squeeze=False)
            survival_rows = []
            for ax,q in zip(axes.flat,thresholds):
                for method,(mean,lo,hi) in arrivals[q].items():
                    x=np.arange(1,group[-1]+1); color=COLORS.get(method)
                    ax.step(x,mean,where='post',label=method,color=color); ax.fill_between(x,lo,hi,step='post',alpha=.15,color=color)
                    survival_rows.append(dict(method=method,threshold=q,reach_curve=mean.tolist(),survival=(1-mean).tolist(),ci_low=lo.tolist(),ci_high=hi.tolist()))
                ax.set(title=f'{q:.0%}',xlabel='Exploitation step',ylabel='Cumulative reach fraction',ylim=(0,1))
            if arrivals: axes[0,0].legend(fontsize=6)
            save(fig,output/('B_reach_'+stem))
            (output/('survival_'+stem+'.json')).write_text(json.dumps(survival_rows,allow_nan=False))
            fig, axes=plt.subplots(1,3,figsize=(13,3))
            for ax,metric in zip(axes,['normalized_adaptation_auc_mean',f'reach_at_K_{thresholds[-1]:g}',f'rmst_{thresholds[-1]:g}']):
                for i,(method,(mean,lo,hi)) in enumerate(scalar_panels[metric].items()):
                    ax.errorbar(i,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color=COLORS.get(method))
                ax.set_xticks(range(len(scalar_panels[metric])),list(scalar_panels[metric]),rotation=30,fontsize=7); ax.set_title(metric)
            save(fig,output/('C_speed_'+stem))
            # Pair methods on matching seed/world, including administrative censoring.
            lookup={(r['method'],r['seed'],r['world_id']):r for r in items}
            for left,right in [('dual_neural','single_policy'),('dual_q_table','q_learning')]:
                diffs=defaultdict(lambda:defaultdict(list))
                for r in items:
                    if r['method']!=left: continue
                    other=lookup.get((right,r['seed'],r['world_id']))
                    if other is None: continue
                    a,b=[x['adaptation_speed'][normalization] for x in (r,other)]
                    if not a['defined'] or not b['defined']: continue
                    if a['normalized_adaptation_auc'] is not None and b['normalized_adaptation_auc'] is not None:
                        diffs['normalized_adaptation_auc'][r['seed']].append(a['normalized_adaptation_auc']-b['normalized_adaptation_auc'])
                    for q in thresholds:
                        ea,eb=[x['events'][f'{q:g}'] for x in (a,b)]
                        diffs[f'rmst_{q:g}'][r['seed']].append(ea['restricted_time']-eb['restricted_time'])
                        diffs[f'reach_at_K_{q:g}'][r['seed']].append(float(ea['event'])-float(eb['event']))
                for metric,by_seed in diffs.items():
                    m,l,h=bootstrap(by_seed,replicates)
                    paired.append(dict(zip(keys,group),normalization=normalization,contrast=left+' - '+right,metric=metric,mean=float(m),ci_low=float(l),ci_high=float(h),n_pairs=sum(map(len,by_seed.values()))))
    # Graph D: include every nonreacher at its full administrative budget.
    budget_groups=defaultdict(list)
    for r in aggregates:
        if r['metric'].startswith('rmst_') or r['metric'].startswith('total_interaction_rmst_'):
            key=tuple(r[k] for k in ('regime','evaluation','world_type','K','normalization'))
            budget_groups[key].append(r)
    for key,rs in budget_groups.items():
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,prefix in zip(axes,('rmst_','total_interaction_rmst_')):
            for method in sorted({r['method'] for r in rs}):
                for metric in sorted({r['metric'] for r in rs if r['metric'].startswith(prefix)}):
                    points=sorted([r for r in rs if r['method']==method and r['metric']==metric],key=lambda r:r['B'])
                    ax.errorbar([r['B'] for r in points],[r['mean'] for r in points],yerr=[[max(0,r['mean']-r['ci_low']) for r in points],[max(0,r['ci_high']-r['mean']) for r in points]],marker='o',label=method+'/'+metric,color=COLORS.get(method),alpha=.65)
            ax.set(xlabel='Exploration budget B',ylabel=prefix,title=str(key))
        axes[0].legend(fontsize=5); save(fig,output/('D_budget_'+'_'.join(map(str,key))))
    for name,data in [('aggregated_speed',aggregates),('paired_speed',paired)]:
        (output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
        if data:
            with (output/(name+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return aggregates


def backfill_raw(root, output):
    """Read existing logs only; write augmented metrics to a separate directory."""
    from .environment import Observation, World
    from .learning import Transition
    from .training import world_set
    root,output=Path(root).resolve(),Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('Backfill output must be outside the source result tree')
    if output.exists():
        raise ValueError('Use a fresh output directory; existing results are never overwritten')
    output.mkdir(parents=True)
    rows=[];missing=[]
    for metric_file in sorted(root.glob('**/evaluation/metrics.json')):
        run=metric_file.parent.parent
        config=json.loads((run/'config.json').read_text())
        metadata=json.loads(metric_file.read_text())
        if not metadata:continue
        diag=run/'diagnostics/worlds.json'
        if diag.exists():
            worlds={w['world_id']:World.from_dict(w) for w in json.loads(diag.read_text())}
        else:
            # Deterministic public seed/config regeneration, diagnostics only.
            worlds={w.world_id:w for w in world_set(config,metadata[0]['regime'],metadata[0]['seed'],'test')}
        for row in metadata:
            file=metric_file.parent/f"{row['evaluation']}_B{row['B']}_K{row['K']}"/f"{row['world_id']}_{row['method']}.jsonl"
            if not file.exists():missing.append(str(file));continue
            records=[json.loads(line) for line in file.read_text().splitlines()]
            if len(records)!=row['B']+row['K'] or [r['step'] for r in records]!=list(range(len(records))):
                raise ValueError(f'Incomplete or unordered raw trajectory: {file}')
            history=[]
            action=lambda x:3 if x is None else 'ABC'.index(x)
            for i,r in enumerate(records):
                if r['phase']!=('exploration' if i<row['B'] else 'exploitation'):
                    raise ValueError(f'Phase mismatch: {file}')
                # Current observation reward is the preceding recorded reward.
                o=Observation(r['mode']-1,action(r['previous_agent_action']),action(r['previous_opponent_action']),records[i-1]['reward'] if i else 0.)
                a,b=action(r['agent_action']),action(r['opponent_action'])
                nxt=Observation(1-o.mode,a,b,r['reward'])
                history.append(Transition(o,a,r['reward'],nxt,r.get('done',i==len(records)-1),b,{}))
            augmented=dict(row)
            augmented['adaptation_speed']=history_speed(history,worlds[row['world_id']],row['B'],row['opponent'],config)
            rows.append(augmented)
    (output/'backfill_audit.json').write_text(json.dumps(dict(source=str(root),n_reconstructed=len(rows),n_missing=len(missing),missing_raw_files=missing),indent=2))
    (output/'world_metrics.json').write_text(json.dumps(rows,allow_nan=False))
    return rows


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replicates',type=int,default=500)
    p.add_argument('--backfill-raw',action='store_true',help='Read original trajectory JSONL; never infer speed from aggregate returns')
    a=p.parse_args()
    if a.backfill_raw:
        rows=backfill_raw(a.root,a.out)
        if not rows:
            print('No raw trajectories available. Audit saved; no time-to-threshold inferred.');return
        target=Path(a.out)/'figures'
    else:
        files=list(Path(a.root).glob('**/evaluation/metrics.json'))
        rows=[r for f in files for r in json.loads(f.read_text())]
        target=a.out
    aggregate_speed(rows,target,a.replicates)


if __name__=='__main__': main()
