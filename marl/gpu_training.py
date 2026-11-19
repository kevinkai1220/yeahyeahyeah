"""Device-resident environments and full-sequence PPO; v2 staged curriculum."""
import json
import time
import numpy as np
import torch
from .models import RecurrentPolicy


def observation(a, b, reward, mode):
    return torch.cat((torch.nn.functional.one_hot(a, 5),
                      torch.nn.functional.one_hot(b, 5),
                      torch.nn.functional.one_hot(torch.full_like(a, mode), 2),
                      reward[:, None]), -1).float()


@torch.no_grad()
def collect(model, opponent, r1, r2, combos, horizon, k, warmup=False, injection=.75):
    device = r1.device
    n = len(r1)
    rows = torch.arange(n, device=device)
    a = b = torch.full((n,), 4, device=device, dtype=torch.long)
    reward = torch.zeros(n, 2, device=device)
    hidden = other_hidden = None
    kinds = torch.randint(3 if opponent is not None else 2, (n,), device=device)
    # Complete R1/R2 pair strictly before K, with randomized location.
    starts = 2 * torch.randint(max(1, k//2), (n,), device=device)
    inject = (torch.rand(n, device=device) < injection) & warmup & (k >= 2)
    records = [[] for _ in range(5)]
    for t in range(horizon):
        obs = observation(a, b, reward[:, 0], t % 2)
        logits, values, hidden = model(obs[:, None], hidden)
        head = 1 if model.dual and t < k and not warmup else 0
        dist = torch.distributions.Categorical(logits=logits[:, 0, head], validate_args=False)
        action = dist.sample()
        if warmup and t < k:
            action = torch.randint(4, (n,), device=device)
            for offset in (0, 1):
                action = torch.where(inject & (starts+offset == t), combos[:, offset], action)
        other = torch.randint(4, (n,), device=device)
        other = torch.where(kinds == 1, t % 4, other)
        if opponent is not None:
            ol, _, other_hidden = opponent(observation(b, a, reward[:, 1], t % 2)[:, None], other_hidden)
            sampled = torch.distributions.Categorical(logits=ol[:, 0, 0], validate_args=False).sample()
            other = torch.where(kinds == 2, sampled, other)
        reward = r1[rows, :, action, other] if t % 2 == 0 else r2[rows, :, a, b, action, other]
        for storage, value in zip(records, (obs, action, dist.log_prob(action), values[:, 0, head], reward[:, 0])):
            storage.append(value)
        a, b = action, other
    return tuple(torch.stack(items, 1) for items in records)


def optimize(model, optimizer, batch, k, warmup, config):
    obs, actions, old_log, old_value, rewards = batch
    returns = rewards.flip(1).cumsum(1).flip(1)
    if model.dual and not warmup:
        returns[:, :k] = config.get('exploration_lambda', 0.) * returns[:, :k] + (1-config.get('exploration_lambda', 0.)) * returns[:, k:k+1]
    # Forced/random prefix is burn-in context only, never an on-policy sample.
    region = slice(k, None) if warmup else slice(0, k) if model.dual else slice(None)
    head = 1 if model.dual and not warmup else 0
    adv = returns[:, region] - old_value[:, region]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    p = config['ppo']
    metrics = []
    for _ in range(p['epochs']):
        order = torch.randperm(len(obs), device=obs.device)
        for idx in order.split(p['minibatch']):
            logits, values, _ = model(obs[idx])
            dist = torch.distributions.Categorical(logits=logits[:, region, head], validate_args=False)
            log = dist.log_prob(actions[idx][:, region])
            ratio = (log-old_log[idx][:, region]).exp()
            actor = -torch.minimum(ratio*adv[idx], ratio.clamp(1-p['clip'], 1+p['clip'])*adv[idx]).mean()
            critic = (values[:, region, head]-returns[idx][:, region]).square().mean()
            loss = actor + p['value_coefficient']*critic
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            metrics.append(torch.stack((loss.detach(), dist.entropy().mean().detach())))
    loss, entropy = torch.stack(metrics).mean(0).cpu().tolist()
    return dict(loss=loss, entropy=entropy, mean_total_reward=rewards.sum(1).mean().item(),
                mean_exploitation_reward=rewards[:, k:].mean().item())


def train_gpu(tasks, config, output, name, dual=False, k=0, pretrained=None, pretraining=False):
    from pathlib import Path
    output = Path(output)
    if (output/f'{name}.jsonl').exists():
        raise FileExistsError(name)
    device = torch.device(config.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install CUDA PyTorch or explicitly set device=cpu for smoke')
    torch.manual_seed(config['seed']*10000 + (10 if pretraining else 100+k if dual else 20))
    model = RecurrentPolicy(dual, config['hidden']).to(device)
    if pretrained is not None:
        pretrained = pretrained.to(device).eval().requires_grad_(False)
    if not pretraining and pretrained is None:
        raise ValueError('Frozen PPO opponent required')
    r1 = torch.tensor(np.stack([t.r1 for t in tasks]), device=device, dtype=torch.float32)
    r2 = torch.tensor(np.stack([t.r2 for t in tasks]), device=device, dtype=torch.float32)
    if dual and any(t.combos is None for t in tasks):
        raise ValueError('Combo curriculum requires generator combo metadata')
    combos = torch.tensor(np.stack([t.combos[0] if t.combos is not None else [0,0] for t in tasks]), device=device)
    phases = [('exploit_pretrain', config['exploit_pretrain_updates']), ('exploration', config['updates'])] if dual else [('train', config['opponent_updates'] if pretraining else config['updates'])]
    start = time.monotonic()
    for phase, updates in phases:
        warmup = phase == 'exploit_pretrain'
        model.requires_grad_(True)
        if dual:
            if warmup:
                model.actors[1].requires_grad_(False); model.critics[1].requires_grad_(False)
            else:
                model.requires_grad_(False)
                model.actors[1].requires_grad_(True); model.critics[1].requires_grad_(True)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
        order = torch.empty(0, device=device, dtype=torch.long)
        with (output/f'{name}{"_exploit_pretrain" if warmup else ""}.jsonl').open('x') as log:
            for update in range(1, updates+1):
                count = config['tasks_per_update']
                while len(order) < count:
                    order = torch.cat((order, torch.randperm(len(tasks), device=device)))
                ids, order = order[:count].repeat_interleave(config['rollouts_per_task']), order[count:]
                batch = collect(model, pretrained, r1[ids], r2[ids], combos[ids], config['horizon'], k, warmup, config.get('history_injection_probability', .75))
                metrics = optimize(model, optimizer, batch, k, warmup, config)
                metrics.update(update=update, phase=phase, elapsed_seconds=time.monotonic()-start)
                log.write(json.dumps(metrics, allow_nan=False)+'\n'); log.flush()
                if update % config['checkpoint_every'] == 0 or update == updates:
                    suffix = '_exploit_pretrain' if warmup else ''
                    model.save(output/f'{name}{suffix}.pt', name=name, k=k, horizon=config['horizon'], update=update, phase=phase, seed=config['seed'])
                    print(f'{name} {phase}: {update}/{updates} {metrics}', flush=True)
    if pretrained is not None:
        pretrained.cpu()
    return model.cpu()
