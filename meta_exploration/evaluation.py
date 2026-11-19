"""Held-out adaptation and paired exploitation. Diagnostics never feed controllers."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .environment import ACTIONS, oracle_values
from .learning import Controller, QLearner, collect, collect_batched, adapt, clone_model, make_model, assert_equal
from .training import world_set, make_game, load_opponent, write_json, append_json


def public_rows(history, world, method, B):
    for step, t in enumerate(history):
        o = t.observation
        yield dict(world_id=world.world_id, method=method,
                   phase='exploration' if step < B else 'exploitation', step=step,
                   mode=o.mode + 1, agent_action=ACTIONS[t.action], opponent_action=ACTIONS[t.opponent_action],
                   previous_agent_action=ACTIONS[o.previous_own] if o.previous_own < 3 else None,
                   previous_opponent_action=ACTIONS[o.previous_other] if o.previous_other < 3 else None,
                   reward=t.reward, done=t.done,
                   observed_combo=ACTIONS[o.previous_own] + ACTIONS[t.action] if o.mode else None,
                   **t.info)


def diagnostic_rows(history, world):
    for step, t in enumerate(history):
        o = t.observation
        bonus = float(world.bonus[o.previous_own, o.previous_other, t.action, t.opponent_action]) if o.mode else 0.
        yield dict(step=step, world_type=world.world_type,
                   true_mean=world.mean(o.mode, (o.previous_own, o.previous_other), t.action, t.opponent_action),
                   true_bonus=bonus)


def combo_metrics(history, world, B):
    pairs = world.bonus.mean(axis=(1, 3))
    good = {tuple(k) for k in np.argwhere(pairs > 0.)}
    # Best two-step pair under iid opponent; labelled separately from the exact
    # opponent-dependent DP oracle. Includes base cost of initiating a combo.
    scores = world.base.mean(-1)[:, None] + world.base.mean(-1)[None, :] + pairs
    optimal = {tuple(k) for k in np.argwhere(scores >= scores.max() - 1e-9)} if world.world_type != 'type1' else set()
    attempted, observed = {}, {}
    for t in history[:B]:
        if t.observation.mode:
            pair = (t.observation.previous_own, t.action)
            attempted[pair] = attempted.get(pair, 0) + 1
            observed.setdefault(pair, []).append(t.reward)
    # Observable evidence heuristic, NOT hidden combo recognition by an agent.
    # Positive lower confidence bound and >=2 samples; actual good membership is
    # evaluated independently. sigma is experimental metadata used only here.
    confident = {p for p, rewards in observed.items() if len(rewards) >= 2 and
                 np.mean(rewards) - 1.96 * max(world.sigma, np.std(rewards)) / np.sqrt(len(rewards)) > 0}
    tried_good = good & attempted.keys()
    exploited = [(t.observation.previous_own, t.action) for t in history[B:] if t.observation.mode]
    denominator = len(exploited)
    def rate(target):
        return sum(p in target for p in exploited) / denominator if denominator else None
    return dict(combo_applicable=world.world_type != 'type1',
                rewarding_combo_attempted=bool(tried_good),
                rewarding_combo_positive_observation=any(max(observed[p]) > 0 for p in tried_good),
                rewarding_combo_supported=bool(good & confident),
                observed_success_false_positives=len(confident - good),
                discovery_rate=len(tried_good) / len(good) if good else None,
                discovered_combo_utilization=rate(tried_good) if tried_good else None,
                optimal_combo_utilization=rate(optimal) if optimal else None,
                combo_attempt_counts={''.join(ACTIONS[a] for a in p): n for p, n in attempted.items()})


def metrics(history, world, B, opponent, config, reference=None):
    rewards = np.array([t.reward for t in history])
    means = np.array([r['true_mean'] for r in diagnostic_rows(history, world)])
    m = dict(total_return=float(rewards.sum()), exploitation_return=float(rewards[B:].sum()),
             exploitation_mean=float(rewards[B:].mean()), expected_total_return=float(means.sum()),
             **combo_metrics(history, world, B))
    for k in config['evaluation']['auc_windows']:
        # Discrete area, not trapezoidal endpoints. No extrapolation beyond budget.
        if k <= len(rewards):
            m[f'auc_{k}'] = float(rewards[:k].sum())
            m[f'auc_mean_{k}'] = float(rewards[:k].mean())
    if opponent in ('random', 'fixed'):
        values = oracle_values(world, len(history), opponent)
        m['oracle_expected_return'] = float(values[0, 3, 3])
        m['oracle_regret'] = float(values[0, 3, 3] - means.sum())
        o = history[B].observation
        m['oracle_exploitation_return_from_agent_state'] = float(values[B, o.previous_own, o.previous_other])
        m['oracle_exploitation_regret'] = m['oracle_exploitation_return_from_agent_state'] - float(means[B:].sum())
        m['oracle_method'] = 'exact finite-horizon DP; random iid or known cyclic phase'
    else:
        m.update(oracle_expected_return=None, oracle_regret=None,
                 oracle_method='unavailable: recurrent opponent latent state; no iid proxy substituted')
    if reference is not None:
        m['history_only_reference_expected_return'] = reference
        m['history_transfer_gap'] = reference - float(means[B:].sum())
    return m


def evaluate_world(config, models, world, seed, mode, opponent_model=None,
                   exploration=None, frozen_single=None, frozen_suffix=None):
    B, K = config['training']['exploration_steps'], config['training']['exploitation_steps']
    online = mode == 'online'
    initial = make_game(world, B + K, seed, config, opponent_model)
    results, update_counts = {}, {}
    if frozen_single is not None:
        assert not online
        results['single_policy'], updates = frozen_single, []
    else:
        single = Controller(clone_model(models['single_policy']), seed)
        optimizer = torch.optim.Adam(single.model.parameters(), lr=config['training']['learning_rate'])
        results['single_policy'], updates = adapt(initial.snapshot(), single, B + K, optimizer, config, online)
        if not online:
            assert_equal(single.model, models['single_policy'])
    update_counts['single_policy'] = sum(u['updates'] for u in updates)
    q = QLearner(config['q_learning'], seed)
    results['q_learning'] = collect(initial.snapshot(), q, B + K, online_q=True)
    update_counts['q_learning'] = 0
    if exploration is None:
        shared_game = initial.snapshot()
        explorer = Controller(models['explorer'], seed)
        prefix = collect(shared_game, explorer, B)
    else:
        prefix, shared_game = exploration
        assert len(prefix) == B and shared_game.t == B
    # This is the ONLY learned-prefix collection for both main dual methods.
    post_exploration = shared_game.snapshot()
    if frozen_suffix is not None:
        assert not online
        suffix, updates = frozen_suffix, []
    else:
        exploit = Controller(clone_model(models['dual_neural']), seed + 10000)
        exploit.ingest(prefix)
        optimizer = torch.optim.Adam(exploit.model.parameters(), lr=config['training']['learning_rate'])
        suffix, updates = adapt(post_exploration.snapshot(), exploit, K, optimizer, config, online)
        if not online:
            assert_equal(exploit.model, models['dual_neural'])
    results['dual_neural'] = prefix + suffix
    update_counts['dual_neural'] = sum(u['updates'] for u in updates)
    q = QLearner(config['q_learning'], seed + 10000); q.replay(prefix)
    results['dual_q_table'] = prefix + collect(post_exploration.snapshot(), q, K, online_q=True)
    update_counts['dual_q_table'] = 0
    if 'q_explorer' in models:
        game = initial.snapshot()
        controller = Controller(clone_model(models['q_explorer']), seed)
        qprefix = collect(game, controller, B)
        q = QLearner(config['q_learning'], seed+10000); q.replay(qprefix)
        results['dual_q_aware'] = qprefix + collect(game, q, K, online_q=True)
        update_counts['dual_q_aware'] = 0
    reference = None
    if config['evaluation']['history_reference']:
        qc = dict(config['q_learning'], replay_passes=50, epsilon_start=0., epsilon_end=0.)
        ref = QLearner(qc, seed + 30000); ref.replay(prefix)
        ref_history = collect(post_exploration.snapshot(), ref, K, online_q=False)
        reference = sum(row['true_mean'] for row in diagnostic_rows(ref_history, world))
        # Only reference scoring accesses means. Reference action selection uses H.
    summaries = {}
    for method, history in results.items():
        summaries[method] = metrics(history, world, B, initial.opponent_kind, config,
                                    reference if method in ('dual_neural', 'dual_q_table') else None)
        summaries[method]['online_optimizer_steps'] = update_counts[method]
        summaries[method]['q_updates_enabled'] = method in ('q_learning', 'dual_q_table', 'dual_q_aware')
    return results, summaries


def evaluate_worlds(config, models, worlds, seed, mode, opponent_model=None, frozen_batch_size=32):
    if frozen_batch_size < 1:
        raise ValueError('frozen_batch_size must be positive')
    B, K = config['training']['exploration_steps'], config['training']['exploitation_steps']
    for start in range(0, len(worlds), frozen_batch_size):
        batch = worlds[start:start+frozen_batch_size]
        seeds = [seed * 100000 + w.seed for w in batch]
        if frozen_batch_size == 1:
            for world, s in zip(batch, seeds):
                yield world, evaluate_world(config, models, world, s, mode, opponent_model)
            continue
        games = [make_game(w, B+K, s, config, opponent_model) for w, s in zip(batch, seeds)]
        prefixes = collect_batched(games, models['explorer'], seeds, B)
        singles, suffixes = [None]*len(batch), [None]*len(batch)
        if mode == 'frozen':
            singles = collect_batched([make_game(w, B+K, s, config, opponent_model) for w,s in zip(batch,seeds)],
                                      models['single_policy'], seeds, B+K)
            suffixes = collect_batched([g.snapshot() for g in games], models['dual_neural'],
                                       [s+10000 for s in seeds], K, prefixes=prefixes)
        for i, (world, s) in enumerate(zip(batch, seeds)):
            yield world, evaluate_world(config, models, world, s, mode, opponent_model,
                                        exploration=(prefixes[i], games[i]),
                                        frozen_single=singles[i], frozen_suffix=suffixes[i])


def evaluate(config, regime, seed, out, device='cpu', frozen_batch_size=32):
    out = Path(out)
    checkpoint = torch.load(out / 'checkpoints' / 'trained.pt', map_location=device, weights_only=False)
    models = {}
    for key, state in checkpoint['models'].items():
        model = make_model(config, device); model.load_state_dict(state); model.eval(); models[key] = model
    baseline = {k: clone_model(m) for k, m in models.items()}
    worlds = world_set(config, regime, seed, 'test')
    train_worlds = world_set(config, regime, seed, 'train')
    assert not {w.fingerprint for w in worlds} & {w.fingerprint for w in train_worlds}
    opponent_model = load_opponent(config, device)
    root = out / 'evaluation'; root.mkdir(exist_ok=True)
    Bs = sorted(set(config['evaluation']['exploration_budgets'] + [config['training']['exploration_steps']]))
    Ks = sorted(set(config['evaluation']['exploitation_budgets'] + [config['training']['exploitation_steps']]))
    all_rows = []
    paired_rows = []
    for B in Bs:
        for K in Ks:
            cfg = copy.deepcopy(config)
            cfg['training']['exploration_steps'], cfg['training']['exploitation_steps'] = B, K
            for mode in config['evaluation']['modes']:
                folder = root / f'{mode}_B{B}_K{K}'; folder.mkdir(exist_ok=True)
                rows = []
                for world, (histories, summaries) in evaluate_worlds(cfg, models, worlds, seed, mode, opponent_model,
                                                                   frozen_batch_size):
                    paired_rows.append(dict(regime=regime, seed=seed, world_id=world.world_id, world_type=world.world_type,
                                            evaluation=mode, B=B, K=K,
                                            neural_minus_q_exploitation_mean=summaries['dual_neural']['exploitation_mean']-summaries['dual_q_table']['exploitation_mean']))
                    for method, history in histories.items():
                        row = dict(regime=regime, seed=seed, world_id=world.world_id, world_type=world.world_type,
                                   method=method, evaluation=mode, B=B, K=K,
                                   opponent=make_game(world, B+K, 0, config, opponent_model).opponent_kind,
                                   **summaries[method])
                        rows.append(row)
                        if config['logging']['save_trajectories']:
                            target = folder / f'{world.world_id}_{method}.jsonl'
                            with target.open('w') as f:
                                for r in public_rows(history, world, method, B):
                                    f.write(json.dumps(r) + '\n')
                            diag_dir = folder / 'diagnostics'; diag_dir.mkdir(exist_ok=True)
                            with (diag_dir / target.name).open('w') as f:
                                for r in diagnostic_rows(history, world):
                                    f.write(json.dumps(r) + '\n')
                write_json(folder / 'metrics.json', rows)
                all_rows.extend(rows)
    for k in models:
        assert_equal(models[k], baseline[k])
    write_json(root / 'metrics.json', all_rows)
    write_json(root / 'paired_differences.json', paired_rows)
    write_json(root / 'protocol.json', dict(
        frozen_batch_size=frozen_batch_size,
        frozen='Only neural parameters frozen; Q methods still update, report separately.',
        paired='One B-prefix collected, environment/opponent/mode/RNG cloned for both exploiters.',
        history_reference='50 observed-only reverse TD replay passes, greedy frozen Q policy on a clone; empirical comparator, not an upper bound.',
        reference_extra_interactions=len(worlds)*len(Bs)*sum(Ks)*len(config['evaluation']['modes']) if config['evaluation']['history_reference'] else 0,
        neural_world_reset='Fresh model clone and optimizer for every world, mode and budget.',
        uncertainty='Across held-out worlds within seed, hierarchical seed/world bootstrap when aggregating.',
        q_gamma=config['q_learning']['gamma'], neural_gamma=config['training']['gamma']))
    return all_rows
