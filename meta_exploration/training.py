import copy
import json
import time
import subprocess
import platform
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .config import REGIMES
from .environment import Game, generate_world, ACTIONS
from .learning import (Controller, QLearner, collect, make_model, clone_model,
                       assert_equal, targets, ppo_update, dual_targets, RolloutBatch)


def atomic_save(data, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def append_json(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


def world_set(config, regime, seed, split):
    types = REGIMES[regime]
    n = config['training']['train_worlds'] if split == 'train' else config['evaluation']['num_test_worlds']
    c = config['environment']
    # Same per-type world sequence across regimes, independent of evaluation budget.
    return [generate_world(seed * 1000000 + (i // len(types)) * 10 + int(types[i % len(types)][-1]),
                           types[i % len(types)], split, c['difficulty'], c['reward_sigma'],
                           c['opponent_dependence']) for i in range(n)]


def make_game(world, horizon, seed, config, opponent_model=None):
    kinds = config['environment']['opponents']
    # Stable across training regimes and budget ablations.
    kind = kinds[(world.seed // 10 + int(world.world_type[-1])) % len(kinds)]
    return Game(world, horizon, seed + 100, kind, seed + 200, opponent_model)


class HistoryCollector:
    """Only observed state enters action selection; every ordered pair is covered.

    coverage schedules pairs on R1/R2, random samples independently, repeat produces
    intentionally narrow history. No world parameters are available to this class.
    """
    def __init__(self, seed, strategy):
        self.rng = np.random.default_rng(seed)
        self.strategy = strategy
        self.pairs = list(self.rng.permutation(9))
        self.index, self.pending = 0, None
        self.repeat = int(self.rng.integers(3))

    def act(self, observation):
        if self.strategy == 'random':
            a = int(self.rng.integers(3))
        elif self.strategy == 'repeat':
            a = self.repeat
        elif observation.mode == 0:
            pair = self.pairs[self.index % 9]
            a, self.pending = int(pair // 3), int(pair % 3)
            self.index += 1
        else:
            a = self.pending if self.pending is not None else int(self.rng.integers(3))
        return a, {'collector': self.strategy, 'action_probs': ([1/3]*3 if self.strategy == 'random' else np.eye(3)[a].tolist())}


def observed_information(history):
    buckets = {}
    for t in history:
        if t.observation.mode:
            key = (t.observation.previous_own, t.action)
            buckets.setdefault(key, []).append(t.reward)
    means = np.array([np.mean(v) for v in buckets.values()])
    variance = np.mean([np.var(v) if len(v) > 1 else 1. for v in buckets.values()]) if buckets else 1.
    contrast = float(np.ptp(means) / np.sqrt(variance + 1.)) if len(means) > 1 else 0.
    # Coverage is primary; bounded observed contrast breaks equal-coverage ties.
    # Thus the high stratum includes full nine-pair histories whenever B>=18.
    return len(buckets) + contrast / (1 + contrast), dict(pair_coverage=len(buckets), observed_contrast=contrast,
                                          pair_counts={''.join(ACTIONS[a] for a in k): len(v) for k, v in buckets.items()})


def episode_returns(history, B):
    return dict(episode_return=float(sum(t.reward for t in history)),
                exploration_return=float(sum(t.reward for t in history[:B])),
                downstream_return=float(sum(t.reward for t in history[B:])))


def log_history(out, config, history, world, method, episode):
    if not config['logging']['save_trajectories']:
        return
    from .evaluation import public_rows, diagnostic_rows
    folder = out / 'training_histories' / method
    folder.mkdir(parents=True, exist_ok=True)
    name = f'episode_{episode:06d}.jsonl'
    with (folder / name).open('w') as f:
        for row in public_rows(history, world, method, config['training']['exploration_steps']):
            f.write(json.dumps(row) + '\n')
    diag = out / 'diagnostics' / 'training_histories' / method
    diag.mkdir(parents=True, exist_ok=True)
    with (diag / name).open('w') as f:
        for row in diagnostic_rows(history, world):
            f.write(json.dumps(row) + '\n')


def pretrain(config, worlds, out, device, seed, opponent_model=None):
    config = copy.deepcopy(config)
    config['training'].update(config['training'].get('pretraining_ppo', {}))
    t = config['training']; B, K = t['exploration_steps'], t['exploitation_steps']
    model = make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['learning_rate'])
    progress = out / 'checkpoints' / 'pretrain_resume.pt'
    start, total_interactions = 0, 0
    batch = RolloutBatch(model, optimizer, config)
    batch_size = t.get('rollout_batch_size', 1)
    if progress.exists():
        saved = torch.load(progress, map_location=device, weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, total_interactions = saved['episode'], saved['interactions']
        model.policy_version = saved.get('policy_version', 0)
        restore_rng(saved)
    ratios = np.array(t['history_sampling_ratios']); ratios = ratios / ratios.sum()
    for episode in range(start, t['pretraining_episodes']):
        world = worlds[episode % len(worlds)]
        candidates = []
        for j in range(t['history_pool_size']):
            s = seed * 100000 + episode * 100 + j
            game = make_game(world, B + K, s, config, opponent_model)
            collector = HistoryCollector(s, ('coverage', 'random', 'repeat')[j % 3])
            if t['privileged_curriculum'] and j == 0:
                # Explicit ablation only; never enabled in main pretraining.
                pair = np.unravel_index(world.bonus.mean(axis=(1, 3)).argmax(), (3, 3))
                collector.pairs = [int(pair[0] * 3 + pair[1])] * 9
            history = collect(game, collector, B)
            score, stats = observed_information(history)
            candidates.append((score, game, history, stats))
        # Rank strata depend ONLY on observed rewards/coverage. Labels are relative,
        # not assertions that the hidden optimum was discovered.
        candidates.sort(key=lambda x: x[0])
        groups = np.array_split(np.arange(len(candidates)), 3)
        # Weighted deficit scheduling, with one sample from each stratum first.
        counts = np.zeros(3, dtype=int)
        for previous_episode in range(episode):
            selected = previous_episode if previous_episode < 3 else int(np.argmax(ratios * (previous_episode + 1) - counts))
            counts[selected] += 1
        group = episode if episode < 3 else int(np.argmax(ratios * (episode + 1) - counts))
        idx = int(groups[group][episode % len(groups[group])])
        _, game, prefix, stats = candidates[idx]
        controller = Controller(model, seed + episode)
        controller.ingest(prefix)
        suffix = collect(game, controller, K)
        log_history(out, config, prefix + suffix, world, 'pretraining', episode)
        adv, ret = targets(suffix, config, B)
        metrics = batch.add(suffix, adv, ret, [x.observation.vector() for x in prefix], episode)
        boundary = (episode+1) % batch_size == 0 or episode+1 == t['pretraining_episodes']
        if boundary:
            append_json(out / 'training.jsonl', dict(stage='pretraining_update', episode=episode, **batch.flush()))
        total_interactions += B * len(candidates) + K
        append_json(out / 'training.jsonl', dict(stage='pretraining', episode=episode,
                    world_id=world.world_id, world_type=world.world_type,
                    return_mean=float(np.mean([x.reward for x in suffix])),
                    interactions=total_interactions, history_stratum=('uninformative', 'ambiguous', 'informative')[group],
                    stratum_definition='relative observed-score tercile', history_stats=stats,
                    **episode_returns(prefix+suffix, B), **metrics))
        if boundary:
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), episode=episode + 1,
                             interactions=total_interactions, policy_version=getattr(model, 'policy_version', 0), **rng_state()), progress)
    atomic_save(dict(model=model.state_dict(), config=config, train_fingerprints=[w.fingerprint for w in worlds],
                     interactions=total_interactions), out / 'checkpoints' / 'shared_exploitation_pretrain.pt')
    model.save(out / 'checkpoints' / 'three_action_policy.pt', purpose='optional frozen PPO opponent; trained on train worlds only')
    return model, total_interactions


def train(config, regime, seed, out, device='cpu', pretrain_only=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Refusing resume with different config')
    write_json(config_path, config)
    if not (out / 'provenance.json').exists():
        write_json(out / 'provenance.json', provenance(config, device, seed))
    torch.manual_seed(seed)
    worlds = world_set(config, regime, seed, 'train')
    tests = world_set(config, regime, seed, 'test')
    assert not {w.fingerprint for w in worlds} & {w.fingerprint for w in tests}
    diagnostics = out / 'diagnostics'; diagnostics.mkdir(exist_ok=True)
    write_json(diagnostics / 'worlds.json', [w.diagnostic() for w in worlds + tests])
    opponent_model = load_opponent(config, device)
    shared, pretrain_interactions = pretrain(config, worlds, out, device, seed, opponent_model)
    if pretrain_only:
        return
    single, exploiter = clone_model(shared), clone_model(shared)
    assert_equal(single, exploiter)
    models = {'single_policy': single, 'dual_neural': exploiter, 'explorer': make_model(config, device)}
    if config['training']['q_aware']:
        models['q_explorer'] = clone_model(models['explorer'])
    initial = {k: v.state_dict() for k, v in models.items()}
    init_path = out / 'checkpoints' / 'identical_initialization.pt'
    if not init_path.exists():
        atomic_save(initial, init_path)
    t = config['training']; B, K = t['exploration_steps'], t['exploitation_steps']
    optimizers = {k: torch.optim.Adam(m.parameters(), lr=t['learning_rate']) for k, m in models.items()}
    resume = out / 'checkpoints' / 'training_resume.pt'
    start = 0
    batches = {k: RolloutBatch(m, optimizers[k], config) for k,m in models.items()}
    batch_size = t.get('rollout_batch_size', 1)
    if resume.exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        start = ckpt['episode']
        for key in models:
            models[key].load_state_dict(ckpt['models'][key]); optimizers[key].load_state_dict(ckpt['optimizers'][key])
            models[key].policy_version = ckpt.get('policy_versions', {}).get(key, 0)
        restore_rng(ckpt)
    for episode in range(start, t['episodes']):
        world = worlds[episode % len(worlds)]; s = seed * 100000 + episode
        before = time.monotonic()
        game = make_game(world, B + K, s, config, opponent_model)
        controller = Controller(single, s)
        history = collect(game, controller, B + K)
        log_history(out, config, history, world, 'single_policy', episode)
        adv, ret = targets(history, config)
        cut = B if t['single_k_only'] else 0
        metrics = batches['single_policy'].add(history[cut:], adv[cut:], ret[cut:],
                             [x.observation.vector() for x in history[:cut]], episode)
        append_json(out / 'training.jsonl', dict(stage='single_policy', episode=episode,
                    world_id=world.world_id, world_type=world.world_type,
                    return_mean=float(np.mean([x.reward for x in history[B:]])),
                    objective=float(sum(x.reward * (t['exploration_lambda'] if i < B else 1.) for i, x in enumerate(history))),
                    interactions=(episode+1)*(B+K), runtime_seconds=time.monotonic()-before,
                    **episode_returns(history, B), **metrics))
        game = make_game(world, B + K, s, config, opponent_model)
        explorer = Controller(models['explorer'], s)
        prefix = collect(game, explorer, B)
        post_exploration = game.snapshot()
        exploit = Controller(exploiter, s + 10000); exploit.ingest(prefix)
        suffix = collect(game, exploit, K)
        full = prefix + suffix
        log_history(out, config, full, world, 'dual_neural', episode)
        # Separate critics: corrected prefix credit uses observed complete suffix.
        (pa, pr), (sa, sr) = dual_targets(prefix, suffix, config)
        for key, ctrl, hist, a, r, context in (
            ('explorer', explorer, prefix, pa, pr, []),
            ('dual_neural', exploit, suffix, sa, sr, [x.observation.vector() for x in prefix])):
            metrics = batches[key].add(hist, a, r, context, episode)
            append_json(out / 'training.jsonl', dict(stage=key, episode=episode,
                        world_id=world.world_id, world_type=world.world_type,
                        return_mean=float(np.mean([x.reward for x in suffix])),
                        interactions=(episode+1)*(B+K), **episode_returns(prefix+suffix, B), **metrics))
        # Training-world baseline diagnostics: no cross-world Q transfer and no
        # additional neural updates. The learned prefix is shared with dual-Q.
        q = QLearner(config['q_learning'], s)
        q_history = collect(make_game(world, B+K, s, config, opponent_model), q, B+K, online_q=True)
        paired_q = QLearner(config['q_learning'], s+10000); paired_q.replay(prefix)
        dual_q_history = prefix + collect(post_exploration, paired_q, K, online_q=True)
        for name, h in [('q_learning', q_history), ('dual_q_table', dual_q_history)]:
            log_history(out, config, h, world, name, episode)
            append_json(out / 'training.jsonl', dict(stage=name, episode=episode, world_id=world.world_id,
                        world_type=world.world_type, return_mean=float(np.mean([x.reward for x in h[B:]])),
                        interactions=(episode+1)*(B+K), updates=0, transitions=0,
                        protocol='training-world online Q diagnostic; table discarded after world',
                        **episode_returns(h, B)))
        if 'q_explorer' in models:
            game = make_game(world, B+K, s, config, opponent_model)
            qctrl = Controller(models['q_explorer'], s)
            prefix = collect(game, qctrl, B)
            q = QLearner(config['q_learning'], s); q.replay(prefix)
            suffix = collect(game, q, K, online_q=True)
            log_history(out, config, prefix + suffix, world, 'dual_q_aware', episode)
            # Monte Carlo downstream return supplies terminal credit at phase
            # boundary, without using Q values as neural critic predictions.
            downstream = sum((t['gamma'] ** i) * x.reward for i, x in enumerate(suffix))
            a, r = targets(prefix, config, bootstrap=downstream)
            metrics = batches['q_explorer'].add(prefix, a, r, episode_id=episode)
            append_json(out / 'training.jsonl', dict(stage='q_explorer', episode=episode,
                        return_mean=float(np.mean([x.reward for x in suffix])), interactions=(episode+1)*(B+K), **episode_returns(prefix+suffix, B), **metrics))
        if (episode+1) % batch_size == 0 or episode+1 == t['episodes']:
            for key, batch in batches.items():
                append_json(out / 'training.jsonl', dict(stage=key+'_update', episode=episode, **batch.flush()))
            atomic_save(dict(episode=episode+1, models={k: m.state_dict() for k, m in models.items()},
                             optimizers={k: o.state_dict() for k, o in optimizers.items()},
                             policy_versions={k:getattr(m, 'policy_version', 0) for k,m in models.items()},
                             **rng_state()), resume)
    atomic_save(dict(models={k: m.state_dict() for k, m in models.items()}, config=config), out / 'checkpoints' / 'trained.pt')
    n = sum(p.numel() for p in shared.parameters())
    write_json(out / 'costs.json', dict(pretraining_interactions=pretrain_interactions,
        per_method_training_interactions=t['episodes']*(B+K),
        single_optimized_transitions=t['episodes']*(K if t['single_k_only'] else B+K),
        dual_exploiter_optimized_transitions=t['episodes']*K,
        explorer_optimized_transitions=t['episodes']*B,
        single_optimizer_steps=math.ceil(t['episodes']/batch_size)*t['ppo_epochs'],
        dual_exploiter_optimizer_steps=math.ceil(t['episodes']/batch_size)*t['ppo_epochs'],
        extra_explorer_optimizer_steps=math.ceil(t['episodes']/batch_size)*t['ppo_epochs'],
        pretraining_optimizer_steps=math.ceil(t['pretraining_episodes']/t.get('pretraining_ppo', {}).get('rollout_batch_size', batch_size))*t.get('pretraining_ppo', {}).get('ppo_epochs', t['ppo_epochs']),
        rollout_batch_size=batch_size, advantage_normalization=t.get('advantage_normalization', 'legacy_episode'),
        neural_parameters=n, single_trainable_parameters=n, dual_trainable_parameters=2*n,
        dual_q_trainable_parameters=n, q_table_entries=96, q_baseline_training_interactions=t['episodes']*(B+K),
        q_baseline_transferred_parameters=0,
        physical_main_training_interactions=t['episodes']*(3*(B+K)+K),
        dual_q_shared_prefix_reused_interactions=t['episodes']*B,
        pretraining_optimized_transitions=t['pretraining_episodes']*K,
        dual_q_explorer_training='shared with dual_neural; additional K-step Q diagnostics, no Q-aware neural updates',
        q_aware_additional_interactions=t['episodes']*(B+K) if t['q_aware'] else 0))


def load_opponent(config, device):
    if 'ppo' not in config['environment']['opponents']:
        return None
    from marl.models import RecurrentPolicy
    path = config['environment']['opponent_checkpoint']
    if not path:
        raise ValueError('Set opponent_checkpoint to a three-action PPO checkpoint; legacy 4-action checkpoints are incompatible')
    model, _ = RecurrentPolicy.load(path)
    if model.n_actions != 3 or model.observation_size != 11:
        raise ValueError('Opponent checkpoint must have 3 actions / 11 observation features')
    model.eval()
    return model.to(device)


def rng_state():
    return dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(saved):
    if 'torch_rng' in saved:
        torch.set_rng_state(saved['torch_rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        if saved.get('cuda_rng') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])


def provenance(config, device, seed):
    def git(*args):
        return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    sources = list(Path('meta_exploration').glob('*.py')) + [Path('marl/models.py')]
    return dict(git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, cuda=torch.version.cuda, device=str(device), seed=seed,
                config=config, rollout_order='episode index modulo training world count; per-episode local RNG',
                checkpoint_boundary='complete rollout batch; no pending trajectories',
                checkpoint_rng='torch CPU/CUDA, numpy global; per-episode controller/environment RNG seeded by episode')
