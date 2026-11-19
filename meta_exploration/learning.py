"""Observed-only controllers, recurrent sequence PPO and tabular TD replay.

The encoder/GRU/actor/critic implementation is reused from marl.models. Sequence
replay and clipped surrogate follow marl.ppo, with GAE, context masks and versions.
"""
from dataclasses import dataclass
import copy
import time
import numpy as np
import torch
from torch.distributions import Categorical
from marl.models import RecurrentPolicy


def make_model(config, device='cpu'):
    return RecurrentPolicy(hidden=config['training']['hidden'], observation_size=11, n_actions=3).to(device)


def clone_model(model):
    return copy.deepcopy(model)


def assert_equal(a, b):
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def action_statistics(logits, values):
    """Categorical statistics with one device-to-host transfer for the whole batch."""
    logs = logits - logits.logsumexp(dim=-1, keepdim=True)
    return torch.cat((logs.softmax(-1), logs, values[..., None]), dim=-1).cpu().numpy()


def sample_statistics(stats, rng, version=0):
    p = stats[:3].astype(float)
    a = int(rng.choice(3, p=p / p.sum()))
    return a, dict(log_prob=float(stats[3+a]), value=float(stats[6]),
                   action_probs=p.tolist(), policy_version=version)


class Controller:
    def __init__(self, model, seed):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.hidden = None
        self.context = []
        self.version = getattr(model, "policy_version", 0)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def act(self, observation):
        x = observation.vector()
        logits, values, self.hidden = self.model(torch.tensor(x, device=self.device)[None, None], self.hidden)
        stats = action_statistics(logits[0, 0, 0], values[0, 0, 0])
        self.context.append(x)
        return sample_statistics(stats, self.rng, self.version)

    @torch.no_grad()
    def value(self, observation):
        # Bootstrap must not advance recurrent state.
        _, v, _ = self.model(torch.tensor(observation.vector(), device=self.device)[None, None], self.hidden)
        return float(v[0, 0, 0])

    @torch.no_grad()
    def refresh(self):
        self.hidden = None
        if self.context:
            _, _, self.hidden = self.model(torch.tensor(np.array(self.context), device=self.device)[None])

    def ingest(self, transitions):
        self.context = [t.observation.vector() for t in transitions]
        self.refresh()


@dataclass
class Transition:
    observation: object
    action: int
    reward: float
    next_observation: object
    done: bool
    opponent_action: int
    info: dict


class QLearner:
    """v2 TD update extended with schedule, configurable gamma and reverse replay."""
    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((2, 4, 4, 3))
        self.steps = 0
        self.update_count = 0
        self.update_seconds = 0.

    def act(self, observation):
        c = self.config
        f = min(1., self.steps / max(1, c['epsilon_decay_steps']))
        epsilon = c['epsilon_start'] + f * (c['epsilon_end'] - c['epsilon_start'])
        values = self.q[observation.q_state()]
        best = np.flatnonzero(values == values.max())
        p = np.full(3, epsilon / 3)
        p[best] += (1 - epsilon) / len(best)
        a = int(self.rng.choice(3, p=p))
        self.steps += 1
        return a, dict(epsilon=epsilon, q_values=values.tolist(), action_probs=p.tolist())

    def update(self, t):
        start = time.perf_counter()
        index = t.observation.q_state() + (t.action,)
        future = 0. if t.done else self.q[t.next_observation.q_state()].max()
        self.q[index] += self.config['alpha'] * (t.reward + self.config['gamma'] * future - self.q[index])
        self.update_count += 1
        self.update_seconds += time.perf_counter() - start

    def replay(self, history):
        # Prefix end is not terminal. Reverse backup propagates delayed reward.
        for _ in range(self.config['replay_passes']):
            for t in reversed(history):
                self.update(t)
        self.steps = len(history)


def gae(rewards, values, dones, bootstrap, gamma=1., gae_lambda=.95):
    rewards, values = np.asarray(rewards), np.asarray(values)
    advantages = np.zeros(len(rewards), dtype=np.float32)
    tail, next_value = 0., bootstrap
    for i in reversed(range(len(rewards))):
        continuation = 1. - float(dones[i])
        delta = rewards[i] + gamma * continuation * next_value - values[i]
        tail = delta + gamma * gae_lambda * continuation * tail
        advantages[i] = tail
        next_value = values[i]
    return advantages, advantages + values


def targets(history, config, start_step=0, bootstrap=0.):
    tr = config['training']
    B = tr['exploration_steps']
    rewards = [t.reward * (tr['exploration_lambda'] if start_step + i < B else 1.)
               for i, t in enumerate(history)]
    return gae(rewards, [t.info['value'] for t in history], [t.done for t in history],
               bootstrap, tr['gamma'], tr['gae_lambda'])


def dual_targets(prefix, suffix, config):
    """Train-only boundary: observed suffix return, never the other critic.

    GAE within B bootstraps at B with MC G_K. This is a sampled continuation
    target, not an environment termination. Lambda attenuates earlier credit;
    lambda=1 gives the complete weighted return minus the prefix critic.
    """
    tr = config['training']
    if tr.get('dual_credit', 'legacy_cross_critic') == 'legacy_cross_critic':
        a, r = targets(prefix + suffix, config)
        return (a[:len(prefix)], r[:len(prefix)]), (a[len(prefix):], r[len(prefix):])
    if suffix and not suffix[-1].done:
        raise ValueError('Observed-suffix credit requires complete training episodes')
    continuation = sum(tr['gamma'] ** i * t.reward for i, t in enumerate(suffix))
    return targets(prefix, config, bootstrap=continuation), targets(suffix, config, len(prefix))


def normalize_advantages(arrays, mode):
    """Actor-only normalization. Ragged arrays contain valid tokens only."""
    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    if mode == 'off':
        return arrays
    if mode == 'legacy_episode':
        return [(a-a.mean()) / (a.std()+1e-8) for a in arrays]
    if mode != 'batch':
        raise ValueError(f'Unknown advantage normalization {mode}')
    all_a = np.concatenate(arrays)
    return [(a-all_a.mean()) / (all_a.std()+1e-8) for a in arrays]


def ppo_batch_update(controller, optimizer, episodes, config, refresh=False, normalization=None):
    """episodes=(history, raw_advantages, critic_targets, context_vectors).

    Right padding is excluded from every loss/statistic by slicing valid tokens.
    GRU rows start at independent zero states; prefixes condition only their row.
    All old log probabilities must belong to this model's current version.
    """
    episodes = [e for e in episodes if len(e[0])]
    if not episodes:
        return dict(updates=0, transitions=0, loss=0.)
    model, device = controller.model, controller.device
    expected = getattr(model, 'policy_version', controller.version)
    versions = {t.info['policy_version'] for h, _, _, _ in episodes for t in h}
    if versions != {expected} or controller.version != expected:
        raise ValueError(f'Off-policy rollout versions {versions}, expected {expected}')
    tr = config['training']
    norm = normalization or tr.get('advantage_normalization', 'legacy_episode')
    normalized = normalize_advantages([e[1] for e in episodes], norm)
    sequences = []
    for h, a, r, prefix in episodes:
        if len(h) != len(a) or len(h) != len(r):
            raise ValueError('Target length mismatch')
        sequences.append(torch.as_tensor(np.array(list(prefix)+[t.observation.vector() for t in h]), device=device))
    x = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    actions = torch.tensor([t.action for h, _, _, _ in episodes for t in h], device=device)
    old_logs = torch.tensor([t.info['log_prob'] for h, _, _, _ in episodes for t in h], device=device)
    adv = torch.tensor(np.concatenate(normalized), device=device)
    ret = torch.tensor(np.concatenate([e[2] for e in episodes]), dtype=torch.float32, device=device)
    raw = np.concatenate([e[1] for e in episodes])
    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    stats = []
    start = time.monotonic()
    was_training = model.training
    model.train()  # cuDNN RNN backward requires the forward to run in train mode.
    try:
        for _ in range(tr['ppo_epochs']):
            logits, values, _ = model(x)
            valid_logits = torch.cat([logits[i, len(e[3]):len(e[3])+len(e[0]), 0] for i,e in enumerate(episodes)])
            valid_values = torch.cat([values[i, len(e[3]):len(e[3])+len(e[0]), 0] for i,e in enumerate(episodes)])
            dist = Categorical(logits=valid_logits)
            logs = dist.log_prob(actions)
            ratio = (logs-old_logs).exp()
            actor_loss = -torch.minimum(ratio*adv, ratio.clamp(1-tr['clip'],1+tr['clip'])*adv).mean()
            critic_loss = (valid_values-ret).square().mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + tr['value_coefficient']*critic_loss - tr['entropy_coefficient']*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            parameters = tuple(model.parameters())
            def grad_norm(term):
                grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                return float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)).detach())
            actor_grad = grad_norm(actor_loss)
            critic_grad = grad_norm(tr['value_coefficient']*critic_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_grad = torch.nn.utils.clip_grad_norm_(parameters, tr['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            stats.append(dict(loss=float(loss.detach()), actor_loss=float(actor_loss.detach()),
                critic_loss=float(critic_loss.detach()), entropy=float(entropy.detach()),
                actor_gradient_norm=actor_grad, critic_gradient_norm=critic_grad,
                gradient_norm=float(total_grad),
                approximate_kl=float(((ratio-1)-(logs-old_logs)).mean().detach()),
                clip_fraction=float(((ratio-1).abs()>tr['clip']).float().mean().detach()),
                value_explained_variance=float((1-(ret-valid_values).var(unbiased=False)/(ret.var(unbiased=False)+1e-8)).detach())))
        controller.version = expected+1
        model.policy_version = controller.version
        if refresh:
            controller.refresh()
    finally:
        model.train(was_training)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    return dict(updates=tr['ppo_epochs'], transitions=len(actions), episodes=len(episodes),
        policy_version=controller.version, rollout_policy_version=expected,
        normalization=norm, raw_advantage_mean=float(raw.mean()), raw_advantage_std=float(raw.std()),
        raw_advantage_min=float(raw.min()), raw_advantage_max=float(raw.max()),
        normalized_advantage_mean=float(adv.mean()), normalized_advantage_std=float(adv.std(unbiased=False)),
        normalized_episode_advantage_means=[float(a.mean()) for a in normalized],
        parameter_delta_norm=float((after-before).norm()), update_seconds=time.monotonic()-start,
        **{k:float(np.mean([row[k] for row in stats])) for k in stats[0]})


def ppo_update(controller, optimizer, history, advantages, returns, config, prefix=(), refresh=True):
    return ppo_batch_update(controller, optimizer, [(history, advantages, returns, prefix)], config, refresh)


class RolloutBatch:
    """One queue per policy/optimizer; collection never mutates its weights."""
    def __init__(self, model, optimizer, config):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.episodes, self.ids = [], []

    def add(self, history, advantages, returns, prefix=(), episode_id=None):
        self.episodes.append((history, advantages, returns, prefix))
        self.ids.append(episode_id)
        return dict(updates=0, transitions=0, loss=0., pending_batch=True)

    def flush(self):
        if not self.episodes:
            return None
        metrics = ppo_batch_update(Controller(self.model, 0), self.optimizer, self.episodes, self.config)
        metrics['rollout_episode_ids'] = list(self.ids)
        self.episodes, self.ids = [], []
        return metrics


def collect(game, actor, steps, online_q=False):
    history = []
    for _ in range(steps):
        observation = game.observation()
        action, info = actor.act(observation)
        nxt, reward, done, b = game.step(action)
        t = Transition(observation, action, reward, nxt, done, b, info)
        history.append(t)
        if online_q:
            actor.update(t)
    return history


@torch.no_grad()
def collect_batched(games, model, seeds, steps, prefixes=None):
    """Batch independent frozen policies sharing weights, never adaptation state.

    Each row keeps its own environment, RNG and GRU hidden state. CPU environment
    semantics and per-world action seeds are identical to scalar collection.
    """
    if len(games) != len(seeds):
        raise ValueError('One seed per game is required')
    if not games:
        return []
    device = next(model.parameters()).device
    rngs = [np.random.default_rng(s) for s in seeds]
    histories = [[] for _ in games]
    hidden = None
    if prefixes is not None:
        if len(prefixes) != len(games) or len({len(h) for h in prefixes}) != 1:
            raise ValueError('Equal-length independent prefixes are required')
        if prefixes[0]:
            context = np.array([[t.observation.vector() for t in h] for h in prefixes])
            _, _, hidden = model(torch.from_numpy(context).to(device))
    for _ in range(steps):
        observations = [game.observation() for game in games]
        x = torch.from_numpy(np.stack([o.vector() for o in observations])).to(device)[:, None]
        logits, values, hidden = model(x, hidden)
        stats = action_statistics(logits[:, 0, 0], values[:, 0, 0])
        for i, (game, observation, rng) in enumerate(zip(games, observations, rngs)):
            action, info = sample_statistics(stats[i], rng)
            nxt, reward, done, b = game.step(action)
            histories[i].append(Transition(observation, action, reward, nxt, done, b, info))
    return histories


def adapt(game, controller, steps, optimizer, config, online=False):
    history, updates = [], []
    interval = config['evaluation']['update_interval']
    while len(history) < steps:
        count = min(interval if online else steps, steps - len(history))
        prefix = list(controller.context)
        start = game.t
        segment = collect(game, controller, count)
        history.extend(segment)
        if online:
            bootstrap = 0. if segment[-1].done else controller.value(game.observation())
            adv, ret = targets(segment, config, start, bootstrap)
            updates.append(ppo_batch_update(controller, optimizer, [(segment, adv, ret, prefix)], config,
                                      refresh=not segment[-1].done,
                                      normalization=config['evaluation'].get('online_advantage_normalization', 'legacy_episode')))
    return history, updates
