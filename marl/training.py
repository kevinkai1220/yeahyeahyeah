import json
import time
from pathlib import Path
import numpy as np
import torch
from .models import RecurrentPolicy
from .opponents import make_opponent
from .rollout import collect_rollout
from .ppo import update_ppo
from .batched import collect_batch
from .offpolicy import SequenceReplay, update_vtrace


def train(tasks, config, output, name, dual=False, k=0, pretrained=None, pretraining=False):
    if config.get("gpu_pipeline", False):
        from .gpu_training import train_gpu
        return train_gpu(tasks, config, output, name, dual, k, pretrained, pretraining)
    output = Path(output)
    # Stage-specific seeds also make running CLI stages separately reproducible.
    stage = 10 if pretraining else 20 if not dual else 100+k
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], stage]))
    torch.manual_seed(config['seed'] * 10000 + stage)
    model = RecurrentPolicy(dual, config['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    kinds = ['random', 'fixed'] if pretraining else ['random', 'fixed', 'ppo']
    if not pretraining and pretrained is None:
        raise ValueError("Main training requires pretrained PPO opponent")
    updates = config['opponent_updates'] if pretraining else config['updates']
    if (output/f'{name}.jsonl').exists():
        raise FileExistsError(f"Training log already exists: {name}; use a new output directory")
    offpolicy = dual and config.get('exploit_learning', 'ppo') == 'vtrace'
    replay = SequenceReplay(config.get('replay_capacity', 1024)) if offpolicy else None
    acquisition_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], stage, 901]))
    replay_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], stage, 902]))
    tracker = None
    if dual and config.get('tracking', {}).get('enabled', False):
        from .tracking import TrainingTracker
        tracker = TrainingTracker(tasks, config, output, name, k, pretrained)
    started = time.monotonic()
    with (output/f'{name}.jsonl').open('w') as log, (output/f'{name}_issues.jsonl').open('w') as issues:
        try:
            if tracker is not None:
                tracker.capture(model, 0)
            for update in range(1, updates+1):
                trajectories = []
                opponent_counts = dict.fromkeys(kinds, 0)
                if config.get('batched_collection', False):
                    selected = acquisition_rng.choice(len(tasks), config['tasks_per_update'], replace=False)
                    batch_tasks, batch_kinds, agent_rngs, opponent_rngs = [], [], [], []
                    for task_index in selected:
                        for _ in range(config['rollouts_per_task']):
                            kind = str(acquisition_rng.choice(kinds))
                            opponent_counts[kind] += 1
                            batch_tasks.append(tasks[task_index])
                            batch_kinds.append(kind)
                            agent_rngs.append(np.random.default_rng(int(acquisition_rng.integers(2**63))))
                            opponent_rngs.append(np.random.default_rng(int(acquisition_rng.integers(2**63))))
                    trajectories, _, _ = collect_batch(batch_tasks, model, pretrained, batch_kinds,
                        config['horizon'], k, agent_rngs, opponent_rngs)
                else:
                    selected = rng.choice(len(tasks), config['tasks_per_update'], replace=False)
                    for task_index in selected:
                        for _ in range(config['rollouts_per_task']):
                            kind = str(rng.choice(kinds))
                            opponent_counts[kind] += 1
                            opponent = make_opponent(kind, np.random.default_rng(int(rng.integers(2**63))), pretrained)
                            trajectories.append(collect_rollout(tasks[task_index], model, opponent,
                                                                config['horizon'], k, rng))
                metrics = update_ppo(model, optimizer, trajectories, k, rng,
                    exploration_lambda=config.get('exploration_lambda', 0.),
                    train_heads=[1] if offpolicy else None, **config['ppo'])
                if offpolicy:
                    replay.add(trajectories)
                    metrics.update(update_vtrace(model, optimizer, replay, replay_rng,
                        updates=config.get('replay_updates', 16),
                        minibatch=config['ppo']['minibatch'],
                        value_coefficient=config['ppo']['value_coefficient'],
                        max_grad_norm=config['ppo']['max_grad_norm']))
                    metrics['replay_sequences'] = len(replay.items)
                metrics.update(update=update, elapsed_seconds=time.monotonic()-started,
                               mean_total_reward=float(np.mean([t.rewards.sum() for t in trajectories])),
                               opponent_counts=opponent_counts)
                if dual:
                    metrics['mean_exploitation_reward'] = float(np.mean([t.rewards[k:].mean() for t in trajectories]))
                log.write(json.dumps(metrics, allow_nan=False)+'\n')
                log.flush()
                for metric, value in metrics.items():
                    if metric.endswith('_entropy') and value < .1:
                        issues.write(json.dumps(dict(update=update, issue='low_policy_entropy', metric=metric, value=value))+'\n')
                        issues.flush()
                if update % config['checkpoint_every'] == 0 or update == updates:
                    model.save(output/f'{name}.pt', name=name, k=k, horizon=config['horizon'],
                               update=update, seed=config['seed'], objective='weighted_explore_plus_exploit' if dual else 'total',
                               exploration_lambda=config.get('exploration_lambda', 0.),
                               exploit_learning=config.get('exploit_learning', 'ppo'))
                    print(f'{name}: {update}/{updates}, loss={metrics["loss"]:.3f}', flush=True)
                if tracker is not None:
                    tracker.capture(model, update)
        except Exception as error:
            issues.write(json.dumps(dict(issue='training_failed', message=str(error)))+'\n')
            raise
    return model
