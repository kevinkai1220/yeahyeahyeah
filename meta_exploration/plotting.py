"""Figures and seed/world hierarchical bootstrap; exclusively reads actual logs."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

COLORS = {'single_policy': '#0072B2', 'q_learning': '#E69F00', 'dual_neural': '#009E73',
          'dual_q_table': '#CC79A7', 'dual_q_aware': '#D55E00'}


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'figure.dpi': 120})
    return plt


def save(fig, path):
    for suffix in ('png', 'pdf'):
        fig.savefig(str(path) + '.' + suffix)
    pyplot().close(fig)


def bootstrap(values_by_seed, replicates=500):
    """Seed is outer sampling unit, world is inner; supports scalars or curves."""
    arrays = [np.asarray(x, dtype=float) for x in values_by_seed.values()]
    mean = np.mean([a.mean(axis=0) for a in arrays], axis=0)
    rng = np.random.default_rng(194)
    if len({len(a) for a in arrays}) == 1:
        data = np.stack(arrays)
        seeds = rng.integers(len(arrays), size=(replicates, len(arrays)))
        worlds = rng.integers(data.shape[1], size=(replicates, len(arrays), data.shape[1]))
        samples = data[seeds[:, :, None], worlds].mean(axis=2).mean(axis=1)
        lo, hi = np.quantile(samples, [.025, .975], axis=0)
        return mean, lo, hi
    samples = []
    for _ in range(replicates):
        selected = rng.integers(len(arrays), size=len(arrays))
        samples.append(np.mean([arrays[i][rng.integers(len(arrays[i]), size=len(arrays[i]))].mean(axis=0)
                                for i in selected], axis=0))
    lo, hi = np.quantile(samples, [.025, .975], axis=0)
    return mean, lo, hi


def summarize(rows, output, replicates=500):
    keys = ('regime', 'evaluation', 'B', 'K', 'method', 'world_type')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
        groups[tuple(row[k] if k != 'world_type' else 'all' for k in keys)].append(row)
    result = []
    metrics = ['exploitation_mean', 'total_return', 'oracle_regret', 'discovery_rate',
               'discovered_combo_utilization', 'optimal_combo_utilization', 'history_transfer_gap']
    metrics += sorted({k for r in rows for k in r if k.startswith('auc_')})
    for group, items in groups.items():
        for metric in metrics:
            valid = [r for r in items if r.get(metric) is not None]
            if not valid:
                continue
            by_seed = defaultdict(list)
            for r in valid:
                by_seed[r['seed']].append(r[metric])
            mean, lo, hi = bootstrap(by_seed, replicates)
            vals = [r[metric] for r in valid]
            result.append(dict(zip(keys, group), metric=metric, mean=float(mean), ci_low=float(lo), ci_high=float(hi),
                               std=float(np.std(vals)), worst=float(min(vals)), n_worlds=len(valid), n_seeds=len(by_seed)))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'aggregated.json').write_text(json.dumps(result, indent=2))
    if result:
        with (output / 'aggregated.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(result[0])); w.writeheader(); w.writerows(result)
    return result


def inspect_history(paths, output):
    plt = pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for path in paths:
        rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
        method = rows[0]['method']; color = COLORS.get(method, 'gray')
        reward = np.array([r['reward'] for r in rows]); steps = np.arange(len(rows))
        actions = np.array(['ABC'.index(r['agent_action']) for r in rows])
        B = sum(r['phase'] == 'exploration' for r in rows)
        diagnostic_path = Path(path).parent / 'diagnostics' / Path(path).name
        diagnostics = [json.loads(x) for x in diagnostic_path.read_text().splitlines()] if diagnostic_path.exists() else None
        axes[0, 0].plot(steps, actions, '.-', alpha=.65, color=color, label=method)
        axes[0, 1].plot(steps, reward, alpha=.6, color=color, label=method)
        axes[1, 0].plot(steps, reward.cumsum(), color=color, label=method)
        pairs = [a+b for a in 'ABC' for b in 'ABC']
        counts = [sum(r['observed_combo'] == p for r in rows[:B]) for p in pairs]
        axes[1, 1].plot(pairs, counts, 'o-', color=color, label=method)
        for phase, segment in [('B', rows[:B]), ('K', rows[B:])]:
            combos = [r for r in segment if r['observed_combo'] is not None]
            positive = sum(r['reward'] > 0 for r in combos)
            axes[2, 0].scatter([phase+' positive', phase+' nonpositive'],
                               [positive, len(combos)-positive], color=color)
            if diagnostics is not None:
                indices = range(B) if phase=='B' else range(B, len(rows))
                valid = [i for i in indices if rows[i]['observed_combo'] is not None]
                actual = sum(diagnostics[i]['true_bonus'] > 0 for i in valid)
                axes[2, 0].scatter([phase+' positive', phase+' nonpositive'], [actual, len(valid)-actual], color=color, marker='x', s=65)
        # Rolling action frequencies, each action differentiated by line style.
        for a, style in enumerate(('-', '--', ':')):
            freq = [np.mean(actions[max(0, i-7):i+1] == a) for i in steps]
            axes[2, 1].plot(steps, freq, style, color=color, alpha=.65, label=f'{method}: {"ABC"[a]}')
    titles = ['Action timeline', 'Observed rewards', 'Cumulative return', 'Exploration combo attempts',
              'Reward sign (o) vs bonus sign (x)', 'Action fraction: A solid, B dashed, C dotted (8 steps)']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.grid(alpha=.2)
    axes[0, 0].set_yticks([0, 1, 2], ['A', 'B', 'C'])
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[2, 1]):
        ax.axvline(B-.5, color='gray', linestyle='--'); ax.set_xlabel('Interaction step')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); save(fig, output)


def plot_results(root, output=None):
    root = Path(root); output = Path(output) if output else root / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(root.glob('**/evaluation/metrics.json'))
    if (root / 'evaluation/metrics.json').exists() and root / 'evaluation/metrics.json' not in metric_files:
        metric_files.append(root / 'evaluation/metrics.json')
    rows = [r for p in metric_files for r in json.loads(p.read_text())]
    if not rows:
        raise ValueError('No executed evaluation results found')
    config_path = metric_files[0].parent.parent / 'config.json'
    replicates = json.loads(config_path.read_text())['evaluation']['bootstrap_replicates'] if config_path.exists() else 500
    summary = summarize(rows, output, replicates)
    plt = pyplot()
    # Reward curves grouped by regime, mode, budget and type; include aggregate.
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in metric_files:
        for folder in p.parent.iterdir():
            if not folder.is_dir():
                continue
            meta_path = folder / 'metrics.json'
            if not meta_path.exists():
                continue
            for r in json.loads(meta_path.read_text()):
                history = folder / f"{r['world_id']}_{r['method']}.jsonl"
                if not history.exists():
                    continue
                rewards = np.array([json.loads(x)['reward'] for x in history.read_text().splitlines()])
                smooth = np.array([rewards[max(0, i-7):i+1].mean() for i in range(len(rewards))])
                for typ in (r['world_type'], 'all'):
                    key = r['regime'], r['evaluation'], r['B'], r['K'], typ
                    curves[key][r['method']][r['seed']].append(smooth)
    for key, methods in curves.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, by_seed in methods.items():
            mean, lo, hi = bootstrap(by_seed, replicates)
            ax.plot(mean, color=COLORS[method], label=method)
            ax.fill_between(np.arange(len(mean)), lo, hi, color=COLORS[method], alpha=.15)
        ax.axvline(key[2]-.5, color='gray', linestyle='--')
        ax.set(xlabel='Test interaction step', ylabel='Reward (8-step moving average)', title=str(key))
        ax.legend(fontsize=8); save(fig, output / ('adaptation_'+'_'.join(map(str, key))))
    # Metric comparisons, exploration budget, discovery/utilization and generalization.
    panels = sorted({(r['evaluation'], r['K']) for r in rows})
    for mode, K in panels:
        selected = [r for r in summary if r['evaluation'] == mode and r['K'] == K and r['world_type'] == 'all']
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for regime in sorted({r['regime'] for r in selected}):
            for method in COLORS:
                points = sorted([r for r in selected if r['regime'] == regime and r['method'] == method and r['metric'] == 'exploitation_mean'], key=lambda r:r['B'])
                if not points:
                    continue
                axes[0].errorbar([r['B'] for r in points], [r['mean'] for r in points],
                                yerr=[[max(0,r['mean']-r['ci_low']) for r in points], [max(0,r['ci_high']-r['mean']) for r in points]],
                                label=f'{regime}/{method}', color=COLORS[method], marker='o', alpha=.7)
                for metric, marker in [('discovery_rate', 'o'), ('discovered_combo_utilization', 'x')]:
                    ps = [r for r in selected if r['regime'] == regime and r['method'] == method and r['metric'] == metric]
                    axes[1].scatter([r['B'] for r in ps], [r['mean'] for r in ps], marker=marker, color=COLORS[method], alpha=.65)
        axes[0].set(xlabel='Exploration budget B', ylabel='Exploitation mean reward', title='Budget ablation')
        axes[1].set(xlabel='Exploration budget B', ylabel='Rate', title='Discovery (o) vs discovered utilization (x)')
        axes[0].legend(fontsize=6, ncol=2); fig.tight_layout(); save(fig, output / f'budget_discovery_{mode}_K{K}')
        # Separate budget panels prevent mixing unlike interaction budgets.
        for B in sorted({r['B'] for r in selected}):
            points = [r for r in selected if r['B'] == B and r['metric'].startswith('auc_') and not r['metric'].startswith('auc_mean')]
            if points:
                fig, ax = plt.subplots(figsize=(max(8,len(points)*.17), 4))
                x = np.arange(len(points))
                ax.bar(x, [r['mean'] for r in points], color=[COLORS[r['method']] for r in points])
                ax.errorbar(x, [r['mean'] for r in points], yerr=[[max(0,r['mean']-r['ci_low']) for r in points], [max(0,r['ci_high']-r['mean']) for r in points]], fmt='none', color='black', linewidth=.6)
                ax.set_xticks(x, [f"{r['regime']} {r['method']} {r['metric']}" for r in points], rotation=90, fontsize=6)
                ax.set_ylabel('Discrete reward AUC'); save(fig, output / f'auc_{mode}_B{B}_K{K}')
            fig, ax = plt.subplots(figsize=(9, 4))
            ps = [r for r in summary if r['evaluation']==mode and r['B']==B and r['K']==K and r['world_type']!='all' and r['metric']=='exploitation_mean']
            for i, r in enumerate(ps):
                ax.errorbar(i, r['mean'], yerr=[[max(0,r['mean']-r['ci_low'])], [max(0,r['ci_high']-r['mean'])]], color=COLORS[r['method']], marker='o')
            ax.set_xticks(range(len(ps)), [f"{r['regime']}/{r['world_type']}/{r['method']}" for r in ps], rotation=90, fontsize=6)
            ax.set_ylabel('Exploitation mean reward'); save(fig, output / f'generalization_{mode}_B{B}_K{K}')
    training = defaultdict(dict)
    for p in root.glob('**/training.jsonl'):
        identity = json.loads((p.parent / 'identity.json').read_text()) if (p.parent / 'identity.json').exists() else {'regime':p.parent.name, 'seed':0}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = identity['regime'], r['stage'], identity['seed']
            training[key][r['episode']] = r
    if training:
        regimes = sorted({k[0] for k in training})
        fig, axes = plt.subplots(len(regimes), 1, figsize=(9, 3*len(regimes)), squeeze=False)
        for ax, regime in zip(axes.flat, regimes):
            for stage in sorted({k[1] for k in training if k[0]==regime}):
                values = {seed: sorted(v.values(), key=lambda r:r['episode']) for (r,s,seed),v in training.items() if r==regime and s==stage}
                common = min(map(len, values.values()))
                by_seed = {seed:[[r['return_mean'] for r in v[:common]]] for seed,v in values.items()}
                mean, lo, hi = bootstrap(by_seed, replicates)
                x = [r['interactions'] for r in next(iter(values.values()))[:common]]
                color = COLORS.get(stage, '#444444' if stage=='pretraining' else '#882255')
                ax.plot(x, mean, label=stage, color=color)
                ax.fill_between(x, lo, hi, color=color, alpha=.15)
            ax.set(title=regime, xlabel='Cumulative environment interactions', ylabel='Training exploitation reward')
            ax.legend(fontsize=7, ncol=3)
        fig.tight_layout(); save(fig, output / 'training_progress')
    # Representative paired histories for each world type from first matching run.
    chosen = set()
    for p in metric_files:
        for r in json.loads(p.read_text()):
            if r['method'] != 'dual_neural' or r['world_type'] in chosen:
                continue
            folder = p.parent / f"{r['evaluation']}_B{r['B']}_K{r['K']}"
            paths = [folder / f"{r['world_id']}_{m}.jsonl" for m in ('dual_neural', 'dual_q_table')]
            if all(x.exists() for x in paths):
                inspect_history(paths, output / f"history_{r['world_type']}"); chosen.add(r['world_type'])
    return summary
