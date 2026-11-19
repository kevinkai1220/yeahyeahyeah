"""Observed-only controllers, recurrent sequence PPO and tabular TD replay.

The encoder/GRU/actor/critic implementation is reused from marl.models. Sequence
replay and clipped surrogate follow marl.ppo, with GAE, context masks and versions.
"""
from dataclasses import dataclass
import copy
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


class Controller:
    def __init__(self, model, seed):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.hidden = None
        self.context = []
        self.version = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def act(self, observation):
        x = observation.vector()
        logits, values, self.hidden = self.model(torch.tensor(x, device=self.device)[None, None], self.hidden)
        dist = Categorical(logits=logits[0, 0, 0])
        p = dist.probs.cpu().numpy().astype(float)
        a = int(self.rng.choice(3, p=p / p.sum()))
        self.context.append(x)
        return a, dict(log_prob=float(dist.log_prob(torch.tensor(a, device=self.device))),
                       value=float(values[0, 0, 0]), action_probs=p.tolist(), policy_version=self.version)

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
        index = t.observation.q_state() + (t.action,)
        future = 0. if t.done else self.q[t.next_observation.q_state()].max()
        self.q[index] += self.config['alpha'] * (t.reward + self.config['gamma'] * future - self.q[index])

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


def ppo_update(controller, optimizer, history, advantages, returns, config, prefix=()):
    if not history:
        return dict(updates=0, transitions=0, loss=0.)
    versions = {t.info['policy_version'] for t in history}
    if versions != {controller.version}:
        raise ValueError(f'Off-policy rollout versions {versions}, expected {controller.version}')
    model, device = controller.model, controller.device
    tr = config['training']
    x = torch.tensor(np.array(list(prefix) + [t.observation.vector() for t in history]), device=device)[None]
    actions = torch.tensor([t.action for t in history], device=device)
    old_logs = torch.tensor([t.info['log_prob'] for t in history], device=device)
    adv = torch.tensor(np.asarray(advantages, dtype=np.float32), device=device)
    ret = torch.tensor(np.asarray(returns, dtype=np.float32), device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    losses, kls = [], []
    for _ in range(tr['ppo_epochs']):
        # Replay the prefix with current weights; gradients may flow through history,
        # but prefix actions do not enter the exploiter surrogate.
        logits, values, _ = model(x)
        dist = Categorical(logits=logits[0, len(prefix):, 0])
        logs = dist.log_prob(actions)
        ratio = (logs - old_logs).exp()
        surrogate = torch.minimum(ratio * adv, ratio.clamp(1-tr['clip'], 1+tr['clip']) * adv)
        loss = -surrogate.mean() + tr['value_coefficient'] * (values[0, len(prefix):, 0] - ret).square().mean()
        loss -= tr['entropy_coefficient'] * dist.entropy().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite PPO loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        kls.append(float(((ratio-1) - (logs-old_logs)).mean().detach()))
    controller.version += 1
    controller.refresh()
    return dict(updates=tr['ppo_epochs'], transitions=len(history), loss=float(np.mean(losses)),
                approximate_kl=float(np.mean(kls)), policy_version=controller.version)


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
            updates.append(ppo_update(controller, optimizer, segment, adv, ret, config, prefix))
    return history, updates
