import numpy as np
import torch
from torch.distributions import Categorical


def return_targets(rewards, k, dual, exploration_lambda=0.):
    """Exploration: lambda * remaining prefix rewards + exploitation return."""
    if not 0 <= exploration_lambda <= 1:
        raise ValueError("exploration_lambda must be in [0, 1]")
    returns = np.flip(np.cumsum(np.flip(rewards, axis=-1), axis=-1), axis=-1).copy()
    if dual:
        if not 0 < k < rewards.shape[-1]:
            raise ValueError("Dual PPO requires 0 < K < T")
        returns[..., :k] = exploration_lambda * returns[..., :k] + (1-exploration_lambda) * returns[..., k:k+1]
    return returns


def update_ppo(model, optimizer, trajectories, k, rng, epochs=4, minibatch=16,
               clip=.2, value_coefficient=.5, max_grad_norm=.5,
               exploration_lambda=0., train_heads=None):
    # Keep time intact: shuffle independent rollouts only, no truncation/detach.
    obs = torch.from_numpy(np.stack([t.observations for t in trajectories]))
    actions = torch.from_numpy(np.stack([t.actions for t in trajectories]))
    old_logs = torch.from_numpy(np.stack([t.log_probs for t in trajectories]))
    old_values = torch.from_numpy(np.stack([t.values for t in trajectories]))
    heads = torch.from_numpy(np.stack([t.heads for t in trajectories]))
    rewards = np.stack([t.rewards for t in trajectories])
    returns = torch.from_numpy(return_targets(rewards, k, model.dual, exploration_lambda))
    advantages = returns - old_values
    for head in range(len(model.actors)):
        mask = heads == head
        selected = advantages[mask]
        advantages[mask] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    active_heads = list(range(len(model.actors))) if train_heads is None else train_heads
    metrics = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(trajectories))
        for start in range(0, len(order), minibatch):
            indices = order[start:start+minibatch]
            # hidden=None means zero initial state for every replayed rollout.
            all_logits, all_values, _ = model(obs[indices])
            chosen_heads = heads[indices]
            logits = all_logits.gather(2, chosen_heads[..., None, None].expand(-1, -1, 1, 4)).squeeze(2)
            values = all_values.gather(2, chosen_heads[..., None]).squeeze(2)
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions[indices])
            ratio = (log_probs - old_logs[indices]).exp()
            adv = advantages[indices]
            surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
            loss = torch.zeros(())
            row = {}
            for head in active_heads:
                mask = chosen_heads == head
                policy_loss = -surrogate[mask].mean()
                value_loss = (values[mask] - returns[indices][mask]).square().mean()
                loss = loss + policy_loss + value_coefficient * value_loss
                label = 'exploration' if head == 1 else 'target' if model.dual else 'single'
                row[f'{label}_entropy'] = distribution.entropy()[mask].mean().item()
                row[f'{label}_policy_loss'] = policy_loss.item()
                row[f'{label}_value_loss'] = value_loss.item()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss; objective was not modified")
            optimizer.zero_grad()
            loss.backward()
            # Adam momentum must not move heads absent from this loss.
            for inactive in set(range(len(model.actors))) - set(active_heads):
                for module in (model.actors[inactive], model.critics[inactive]):
                    for parameter in module.parameters():
                        parameter.grad = None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            row.update(loss=loss.item(), gradient_norm=float(norm),
                       approximate_kl=((ratio-1) - (log_probs-old_logs[indices])).mean().item(),
                       clip_fraction=((ratio-1).abs() > clip).float().mean().item())
            metrics.append(row)
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
