"""Full-history replay actor-critic with V-trace off-policy correction.
Reference: Espeholt et al., ICML 2018, https://arxiv.org/abs/1802.01561.
The target head learns from all steps, including the exploration behavior head.
"""
from collections import deque
import numpy as np
import torch
from torch.distributions import Categorical


class SequenceReplay:
    def __init__(self, capacity=1024):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.items = deque(maxlen=capacity)

    def add(self, trajectories):
        self.items.extend(trajectories)

    def sample(self, count, rng):
        indices = rng.choice(len(self.items), count, replace=len(self.items) < count)
        return [self.items[int(i)] for i in indices]


@torch.no_grad()
def vtrace_targets(rewards, values, target_log_probs, behavior_log_probs,
                   rho_clip=1., c_clip=1.):
    """Complete episodes, gamma=1, zero terminal bootstrap; shape [batch,time]."""
    ratios = (target_log_probs - behavior_log_probs).clamp(max=20).exp()
    rho, c = ratios.clamp(max=rho_clip), ratios.clamp(max=c_clip)
    next_values = torch.cat((values[:, 1:], torch.zeros_like(values[:, :1])), 1)
    delta = rho * (rewards + next_values - values)
    correction = torch.zeros_like(values[:, 0])
    targets = torch.empty_like(values)
    for t in reversed(range(values.shape[1])):
        correction = delta[:, t] + c[:, t] * correction
        targets[:, t] = values[:, t] + correction
    next_targets = torch.cat((targets[:, 1:], torch.zeros_like(targets[:, :1])), 1)
    advantages = rho * (rewards + next_targets - values)
    return targets, advantages, ratios


def update_vtrace(model, optimizer, replay, rng, updates=16, minibatch=16,
                  value_coefficient=.5, max_grad_norm=.5):
    metrics = []
    for _ in range(updates):
        trajectories = replay.sample(minibatch, rng)
        obs = torch.from_numpy(np.stack([t.observations for t in trajectories]))
        actions = torch.from_numpy(np.stack([t.actions for t in trajectories]))
        behavior = torch.from_numpy(np.stack([t.log_probs for t in trajectories]))
        rewards = torch.from_numpy(np.stack([t.rewards for t in trajectories]))
        # Reconstruct history from episode start with current weights, no stale hidden states.
        logits, all_values, _ = model(obs)
        dist = Categorical(logits=logits[:, :, 0])
        log_probs, values = dist.log_prob(actions), all_values[:, :, 0]
        targets, advantages, ratios = vtrace_targets(rewards, values, log_probs, behavior)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = (values-targets).square().mean()
        loss = policy_loss + value_coefficient * value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite V-trace loss')
        optimizer.zero_grad()
        loss.backward()
        # Stacking heads gives unused heads zero gradients, not None. With Adam,
        # zero still applies stored momentum; explicitly skip exploration heads.
        for head in range(1, len(model.actors)):
            for module in (model.actors[head], model.critics[head]):
                for parameter in module.parameters():
                    parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        metrics.append(dict(target_entropy=dist.entropy().mean().item(),
            target_policy_loss=policy_loss.item(), target_value_loss=value_loss.item(),
            vtrace_loss=loss.item(), vtrace_gradient_norm=float(norm),
            importance_ratio_mean=ratios.mean().item(), importance_clipped_fraction=(ratios>1).float().mean().item()))
    return {key:float(np.mean([r[key] for r in metrics])) for key in metrics[0]}
