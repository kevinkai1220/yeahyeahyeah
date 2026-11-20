"""Merge disjoint checkpoint-evaluation outputs and plot observed results only."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .history_diagnostics import (paired_summary,preference_summary,linear_probe,
    within_checkpoint_probes,speed_scalars)
from .training import write_json
from .plotting import bootstrap,pyplot,save


def summarize(inputs,out):
    out=Path(out)
    if out.exists():raise ValueError('Choose a new summary directory')
    rows=[];probes=[];seen={};manifests=[]
    for source in map(Path,inputs):
        if not (source/'complete.json').exists():raise ValueError(f'Incomplete evaluation: {source}')
        manifests.extend(json.loads((source/'provenance.json').read_text()))
        probes.extend(json.loads((source/'probe_samples.json').read_text()))
        for path in sorted(source.glob('*_seed_*.jsonl')):
            with path.open() as f:
                for line in f:
                    r=json.loads(line);key=tuple(r[k] for k in ['regime','seed','world_id','method','intervention','evaluation'])
                    if key in seen:
                        # Early local smoke recorded identical Q run for both neural modes.
                        if r['method']=='q_table' and {k:v for k,v in r['metrics'].items() if k!='adaptation_speed'}==seen[key]['metrics']:continue
                        raise ValueError(f'Duplicate world/evaluation identity: {key}')
                    # Actual B is recovered from source config, not guessed from K.
                    manifest=next(m for m in manifests if m['regime']==r['regime'] and m['seed']==r['seed'])
                    B=manifest['evaluation_config']['training']['exploration_steps']
                    r['speed_summary']=speed_scalars(r['metrics']['adaptation_speed'],B)
                    r={k:v for k,v in r.items() if k not in ['suffix','evidence_curve']}
                    r['metrics']={k:v for k,v in r['metrics'].items() if k!='adaptation_speed'}
                    seen[key]=r;rows.append(r)
    out.mkdir(parents=True)
    paired=paired_summary(rows);preferences=preference_summary(rows)
    write_json(out/'paired_summary.json',paired);write_json(out/'preference_summary.json',preferences)
    write_json(out/'provenance.json',manifests)
    write_json(out/'within_checkpoint_probes.json',{reg:within_checkpoint_probes([p for p in probes if p['regime']==reg]) for reg in {r['regime'] for r in rows}})
    write_json(out/'probes.json',{reg:linear_probe([p for p in probes if p['regime']==reg]) for reg in {r['regime'] for r in rows}})
    aggregates=[]
    for regime in sorted({r['regime'] for r in rows}):
        for mode in ['frozen','online']:
            fig,axes=pyplot().subplots(1,2,figsize=(12,4))
            for method in ['single_policy','dual_neural','q_table']:
                rs=[r for r in rows if r['regime']==regime and r['method']==method and r['evaluation']==('online_q' if method=='q_table' else mode)]
                kinds=sorted({r['intervention'] for r in rs})
                for ax,metric in zip(axes,['exploitation_mean','optimal_combo_utilization']):
                    means=[];low=[];high=[]
                    for kind in kinds:
                        by_seed=defaultdict(list)
                        for r in rs:
                            if r['intervention']==kind and r['metrics'][metric] is not None:by_seed[r['seed']].append(r['metrics'][metric])
                        if not by_seed:continue
                        m,l,h=bootstrap(by_seed,1000);means.append(m);low.append(l);high.append(h)
                        aggregates.append(dict(regime=regime,evaluation=mode,method=method,intervention=kind,metric=metric,mean=float(m),ci_low=float(l),ci_high=float(h),seeds=len(by_seed)))
                    if len(means)==len(kinds):
                        ax.errorbar(kinds,means,yerr=[np.maximum(0,np.array(means)-low),np.maximum(0,np.array(high)-means)],marker='o',label=method)
                    ax.set_ylabel(metric);ax.tick_params(axis='x',rotation=70)
            axes[0].legend(fontsize=7);fig.suptitle(f'{regime} / {mode} (Q always updates)');fig.tight_layout();save(fig,out/f'{regime}_{mode}')
    write_json(out/'aggregated.json',aggregates)
    write_json(out/'complete.json',dict(world_condition_rows=len(rows),inputs=list(map(str,inputs))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();summarize(a.inputs,a.out)

if __name__=='__main__':main()
