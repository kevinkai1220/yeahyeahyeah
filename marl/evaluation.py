import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .environment import Game
from .opponents import NeuralController, make_opponent
from .qlearning import OnlineQLearner
from .oracle import oracle_values


def evaluate_rollout(task, model, kind, opponent_kind, pretrained, horizon, k, seed,
                     q_config=None, oracle=None):
    # Independent action RNG streams: different agents cannot consume opponent randomness.
    agent_seed, opponent_seed = np.random.SeedSequence(seed).spawn(2)
    agent_rng, opponent_rng = np.random.default_rng(agent_seed), np.random.default_rng(opponent_seed)
    opponent = make_opponent(opponent_kind, opponent_rng, pretrained)
    if kind == 'q_learning':
        agent = OnlineQLearner(agent_rng, **(q_config or {}))
    else:
        agent = NeuralController(model, agent_rng)
    game = Game(task, horizon)
    observations = game.reset()
    rewards, first_hit, oracle_start = [], None, None
    for t in range(horizon):
        if t == k and oracle is not None:
            oracle_start = float(oracle[(t,) + game.previous])
        if kind == 'q_learning':
            a = agent.act(observations[0])
        else:
            head = int(kind == 'dual_learned' and t < k)
            a = agent.act_info(observations[0], head, random_action=kind == 'dual_random' and t < k)[0]
        b = opponent.act(observations[1])
        # Only the evaluator sees combo metadata. 1-based completion round.
        if task.combos is not None and game.mode == 1 and first_hit is None:
            if (game.previous[0], a) == tuple(task.combos[0]):
                first_hit = t + 1
        next_observations, reward, done = game.step(a, b)
        if kind == 'q_learning':
            agent.update(observations[0], a, reward[0], next_observations[0], done)
        observations = next_observations
        rewards.append(reward)
    rewards = np.asarray(rewards)
    exploitation = rewards[k:, 0]
    cumulative = rewards[:, 0].cumsum()
    metrics = dict(total_reward=float(cumulative[-1]), mean_reward=float(rewards[:, 0].mean()),
                   exploitation_return=float(exploitation.sum()), exploitation_mean=float(exploitation.mean()),
                   adaptation_auc=float((exploitation.cumsum()/np.arange(1, horizon-k+1)).mean()),
                   opponent_total_reward=float(rewards[:, 1].sum()),
                   reward_difference=float((rewards[:, 0]-rewards[:, 1]).sum()),
                   combo_metadata_available=task.combos is not None,
                   combo_discovered=None if task.combos is None else first_hit is not None,
                   first_combo_round=first_hit,
                   combo_discovered_by_k=None if task.combos is None else first_hit is not None and first_hit <= k,
                   oracle_expected_return=oracle_start,
                   oracle_regret=None if oracle_start is None else oracle_start-float(exploitation.sum()))
    curves = dict(reward=rewards[:, 0], cumulative_reward=cumulative,
                  mean_reward=cumulative/np.arange(1, horizon+1),
                  reward_difference=(rewards[:, 0]-rewards[:, 1]).cumsum())
    return metrics, curves


def evaluate(tasks, dual_models, single_model, pretrained, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    horizon = config['horizon']
    curve_sums, counts, grouped = {}, defaultdict(int), defaultdict(list)
    with (output/'episodes.jsonl').open('w') as stream:
        for task_index, task in enumerate(tasks):
            for opponent_kind in ('random', 'fixed', 'ppo'):
                oracle = oracle_values(task, horizon, opponent_kind) if opponent_kind != 'ppo' else None
                for repetition in range(config['evaluation_rollouts']):
                    seed = [config['seed'], 400, task_index, ('random', 'fixed', 'ppo').index(opponent_kind), repetition]
                    # Single and online learners do not depend on K. Reusing the same seed
                    # yields the same full trajectory, scored at each K cutoff.
                    for k, dual in dual_models.items():
                        for kind in ('dual_learned', 'dual_random', 'dual_target', 'single_gru', 'q_learning'):
                            model = single_model if kind == 'single_gru' else dual
                            metrics, curves = evaluate_rollout(task, model, kind, opponent_kind, pretrained,
                                                              horizon, k, seed, config['q_learning'], oracle)
                            row = dict(task_index=task_index, task_id=task.fingerprint, k=k, agent=kind,
                                       opponent=opponent_kind, repetition=repetition, **metrics)
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                            key = (k, kind, opponent_kind)
                            grouped[key].append(metrics)
                            if key not in curve_sums:
                                curve_sums[key] = {name: np.zeros(horizon) for name in curves}
                            for name, curve in curves.items():
                                curve_sums[key][name] += curve
                            counts[key] += 1
    summaries = []
    for (k, kind, opponent), rows in grouped.items():
        summary = dict(k=k, agent=kind, opponent=opponent, rollouts=len(rows))
        for name in ('total_reward', 'mean_reward', 'exploitation_return', 'exploitation_mean',
                     'adaptation_auc', 'reward_difference', 'oracle_regret'):
            vals = [r[name] for r in rows if r[name] is not None]
            summary[name] = float(np.mean(vals)) if vals else None
        known = [r for r in rows if r['combo_metadata_available']]
        hits = [r['first_combo_round'] for r in known if r['combo_discovered']]
        summary['combo_discovery_rate'] = float(np.mean([r['combo_discovered'] for r in known])) if known else None
        summary['combo_discovery_rate_by_k'] = float(np.mean([r['combo_discovered_by_k'] for r in known])) if known else None
        summary['first_combo_round_conditional_mean'] = float(np.mean(hits)) if hits else None
        # Censored at T; report alongside rate, never label misses as discoveries.
        summary['restricted_mean_time_to_combo'] = float(np.mean([r['first_combo_round'] if r['combo_discovered'] else horizon for r in known])) if known else None
        summaries.append(summary)
    (output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    with (output/'curves.csv').open('w', newline='') as stream:
        names = ['reward', 'cumulative_reward', 'mean_reward', 'reward_difference']
        writer = csv.DictWriter(stream, fieldnames=['k', 'agent', 'opponent', 'round']+names)
        writer.writeheader()
        for (k, kind, opponent), sums in curve_sums.items():
            for t in range(horizon):
                writer.writerow(dict(k=k, agent=kind, opponent=opponent, round=t+1,
                                     **{name: sums[name][t]/counts[k, kind, opponent] for name in names}))
    return summaries
