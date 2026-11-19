import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('analysis/full_study')
p=json.loads((OUT/'analysis.json').read_text())
c=json.loads((OUT/'simple_history_controls.json').read_text())
COLORS={'single_policy':'#0072B2','q_learning':'#E69F00','dual_neural':'#009E73','dual_q_table':'#CC79A7'}
LABELS={'single_policy':'Single NN','q_learning':'Q-only','dual_neural':'Dual NN','dual_q_table':'Dual Q'}
regimes=['A1','A2','A3','B'];methods=list(COLORS)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axes=plt.subplots(2,2,figsize=(14,10))
ax=axes[0,0];x=np.arange(4)
for i,method in enumerate(methods):
    rs=[next(r for r in p['main'] if r['evaluation']=='online' and r['regime']==regime and r['method']==method) for regime in regimes]
    m=np.array([r['mean'] for r in rs]);lo=np.array([r['ci_low'] for r in rs]);hi=np.array([r['ci_high'] for r in rs])
    ax.bar(x+(i-1.5)*.19,m,.18,color=COLORS[method],label=LABELS[method],yerr=[m-lo,hi-m],capsize=2)
for i,r in enumerate(p['analytical_controls']):
    ax.plot([i-.4,i+.4],[r['uniform_random_expected_exploitation_mean']]*2,'--',color='gray',linewidth=1.4)
ax.set_xticks(x,['Type 1 only','Type 2 only','Type 3 only','Mixed'])
ax.set(ylabel='K-window mean reward',title='Uploaded results: online, B=32 / K=128')
ax.legend(fontsize=8,ncol=2)
ax.text(.02,.98,'Gray dashed: analytical uniform-action expectation',transform=ax.transAxes,va='top',fontsize=8)
ax=axes[0,1]
for i,method in enumerate(['dual_neural','dual_q_table']):
    rs=[next(r for r in p['diagnostics'] if r['regime']==regime and r['method']==method) for regime in ['A2','A3']]
    ax.bar(np.arange(2)+(i-.5)*.3,[r['conditional_utilization']*100 for r in rs],.28,color=COLORS[method],label=LABELS[method])
ax.set_xticks([0,1],['Type 2: discovered in 181/500','Type 3: discovered in 405/500'])
ax.set(ylabel='Reuse of discovered optimal combo (%)',ylim=(0,100),title='Same exploration history; discovered worlds only')
ax.legend()
ax=axes[1,0]
for i,method in enumerate(methods):
    rs=[next(r for r in p['generalization_matched'] if r['world_type']==typ and r['method']==method) for typ in ['type1','type2','type3']]
    m=np.array([r['mean'] for r in rs]);lo=np.array([r['ci_low'] for r in rs]);hi=np.array([r['ci_high'] for r in rs])
    ax.errorbar(np.arange(3)+(i-1.5)*.12,m,yerr=[m-lo,hi-m],fmt='o',color=COLORS[method],label=LABELS[method],capsize=3)
ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(3),['Type 1 (170)','Type 2 (165)','Type 3 (165)'])
ax.set(ylabel='Mixed minus single-type reward',title='Generalization on identical held-out worlds')
ax=axes[1,1]
for i,(strategy,color) in enumerate([('random','#666666'),('coverage','#56B4E9'),('repeat','#D55E00')]):
    rs=[next(r for r in c['summary'] if r['regime']==regime and r['strategy']==strategy and r['metric']=='exploitation_mean') for regime in regimes]
    m=np.array([r['mean'] for r in rs]);lo=np.array([r['ci_low'] for r in rs]);hi=np.array([r['ci_high'] for r in rs])
    ax.errorbar(np.arange(4)+(i-1)*.15,m,yerr=[m-lo,hi-m],fmt='o',color=color,label=strategy,capsize=3)
ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(4),['Type 1 only','Type 2 only','Type 3 only','Mixed'])
ax.set(ylabel='Control minus learned-explorer reward',title='NEW CPU controls: identical fresh-Q exploiter')
ax.legend(fontsize=8,ncol=3)
for ax in axes.flat:ax.grid(axis='y',alpha=.18)
fig.suptitle('Meta-exploration full study: diagnosis, not just final reward',fontsize=14)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'overview.{ext}',dpi=160)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for method in methods:
    rs=[r for r in p['budgets'] if r['regime']=='A3' and r['method']==method]
    axes[0].plot([r['B'] for r in rs],[r['exploitation_mean'] for r in rs],'o-',color=COLORS[method],label=LABELS[method])
    axes[1].plot([r['B'] for r in rs],[r['total_mean'] for r in rs],'o-',color=COLORS[method],label=LABELS[method])
axes[0].set(title='Type 3: downstream K-window',ylabel='Mean reward during K steps')
axes[1].set(title='Type 3: full B+K budget',ylabel='Mean reward over all interactions')
for ax in axes:ax.set_xlabel('Exploration B (K fixed at 128)');ax.set_xticks([8,16,32,64]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'type3_budget_tradeoff.{ext}',dpi=160)
