import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from .generator import make_splits, save_tasks, load_tasks
from .models import RecurrentPolicy
from .training import train
from .evaluation import evaluate


def validate(config):
    if config.get('gpu_pipeline', False):
        if config.get('exploit_learning', 'ppo') != 'ppo' or config.get('tracking', {}).get('enabled', False):
            raise ValueError('v2 GPU pipeline supports PPO without legacy tracking only')
        if not isinstance(config.get('exploit_pretrain_updates'), int) or config['exploit_pretrain_updates'] < 1:
            raise ValueError('Positive exploit_pretrain_updates required')
        if not 0 <= config.get('history_injection_probability', .75) <= 1:
            raise ValueError('Invalid history injection probability')
        if any(k < 2 for k in config['budgets']):
            raise ValueError('Combo history requires K >= 2')
    for key in ('horizon', 'training_tasks', 'evaluation_tasks', 'tasks_per_update',
                'rollouts_per_task', 'updates', 'opponent_updates', 'evaluation_rollouts',
                'checkpoint_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= config.get('exploration_lambda', 0.) <= 1:
        raise ValueError('exploration_lambda must be in [0, 1]')
    if config.get('exploit_learning', 'ppo') not in ('ppo', 'vtrace'):
        raise ValueError('Unknown exploit_learning')
    tracking = config.get('tracking', {})
    if tracking.get('enabled', False):
        for key, default in (('every', 5), ('tasks', 16), ('rollouts', 2)):
            if not isinstance(tracking.get(key, default), int) or tracking.get(key, default) < 1:
                raise ValueError(f'tracking.{key} must be a positive integer')
    if config['hidden'] != 128:
        raise ValueError('Phase 1 uses GRU hidden=128')
    if not config['budgets'] or len(set(config['budgets'])) != len(config['budgets']):
        raise ValueError('Budgets must be nonempty and unique')
    if not all(isinstance(k, int) and 0 < k < config['horizon'] for k in config['budgets']):
        raise ValueError('Each K must satisfy 0 < K < T')
    if config['tasks_per_update'] > config['training_tasks']:
        raise ValueError('tasks_per_update exceeds training_tasks')
    if config['seed'] < 0 or config['learning_rate'] <= 0:
        raise ValueError('Invalid seed or learning rate')
    ppo = config['ppo']
    if ppo['epochs'] < 1 or ppo['minibatch'] < 1 or not 0 < ppo['clip'] < 1:
        raise ValueError('Invalid PPO settings')


def prepare(output, config):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be empty for prepare/run; existing results are preserved')
    validate(config)
    output.mkdir(parents=True, exist_ok=True)
    train_tasks, evaluation_tasks = make_splits(config['seed'], config['training_tasks'], config['evaluation_tasks'], **config['generator'])
    save_tasks(output/'training_tasks.npz', train_tasks)
    save_tasks(output/'evaluation_tasks.npz', evaluation_tasks)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                    rule='alternating_R1_R2', observation='own_previous_action,other_previous_action,own_previous_reward,mode',
                    training_ids=[t.fingerprint for t in train_tasks], evaluation_ids=[t.fingerprint for t in evaluation_tasks])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def load_model(output, name, config, k=None):
    model, metadata = RecurrentPolicy.load(output/f'{name}.pt')
    if metadata['horizon'] != config['horizon'] or (k is not None and metadata['k'] != k):
        raise ValueError('Checkpoint/config horizon or budget mismatch')
    model.eval()
    model.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description='Phase 1 recurrent PPO and online Q-learning experiments')
    parser.add_argument('command', choices=['prepare', 'pretrain', 'train', 'evaluate', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Required for prepare/run; later stages use saved config')
    parser.add_argument('--seed', type=int, help='Override seed for prepare/run only')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command in ('prepare', 'run'):
        if args.config is None:
            parser.error('--config is required for prepare/run')
        config = json.loads(args.config.read_text())
        if args.seed is not None:
            config['seed'] = args.seed
        prepare(args.output, config)
    else:
        if args.config is not None or args.seed is not None:
            parser.error('Later stages use config.json saved in --output; omit --config/--seed')
        config = json.loads((args.output/'config.json').read_text())
        validate(config)
    if args.command == 'prepare':
        return
    tasks = load_tasks(args.output/'training_tasks.npz')
    if args.command in ('pretrain', 'run'):
        train(tasks, config, args.output, 'opponent', pretraining=True)
    if args.command in ('train', 'run'):
        opponent = load_model(args.output, 'opponent', config)
        train(tasks, config, args.output, 'single', pretrained=opponent)
        for k in config['budgets']:
            train(tasks, config, args.output, f'dual_k{k}', dual=True, k=k, pretrained=opponent)
    if args.command in ('evaluate', 'run'):
        evaluation_tasks = load_tasks(args.output/'evaluation_tasks.npz')
        if {t.fingerprint for t in tasks} & {t.fingerprint for t in evaluation_tasks}:
            raise ValueError('Training/evaluation task overlap')
        opponent = load_model(args.output, 'opponent', config)
        single = load_model(args.output, 'single', config)
        duals = {k: load_model(args.output, f'dual_k{k}', config, k) for k in config['budgets']}
        evaluate(evaluation_tasks, duals, single, opponent, config, args.output/'evaluation')
        print(f'Evaluation saved to {args.output / "evaluation"}', flush=True)


if __name__ == '__main__':
    main()
