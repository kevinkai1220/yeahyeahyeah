"""Batch inference across independent episodes; each has its own RNG and history."""
import numpy as np
import torch
from .environment import Game
from .rollout import Trajectory


def probabilities(logits):
    logs = logits.log_softmax(-1)
    probs = logs.exp().numpy().astype(np.float64)
    probs /= probs.sum(-1, keepdims=True)
    return probs, logs.numpy()


@torch.no_grad()
def collect_batch(tasks, model, pretrained, opponent_kinds, horizon, k,
                  agent_rngs, opponent_rngs, kind='dual_learned', trace=None):
    games = [Game(task, horizon) for task in tasks]
    observations = [game.reset() for game in games]
    n = len(tasks)
    obs_record = np.zeros((n,horizon,13),np.float32)
    actions = np.zeros((n,horizon),np.int64)
    logs = np.zeros((n,horizon),np.float32)
    values = np.zeros((n,horizon),np.float32)
    rewards = np.zeros((n,horizon),np.float32)
    exact_rewards = np.zeros((n,horizon,2),np.float64)
    heads = np.zeros((n,horizon),np.int64)
    opponent_actions = np.zeros((n,horizon),np.int64)
    hidden = other_hidden = None
    neural_indices = [i for i,x in enumerate(opponent_kinds) if x == 'ppo']
    for t in range(horizon):
        x = np.stack([o[0].vector() for o in observations])
        all_logits, all_values, hidden = model(torch.from_numpy(x)[:,None], hidden)
        head = int(model.dual and kind == 'dual_learned' and t < k)
        probs, log_probs = probabilities(all_logits[:,0,head])
        aa = [int(rng.integers(4) if kind == 'dual_random' and t < k else rng.choice(4,p=probs[i])) for i,rng in enumerate(agent_rngs)]
        bb = [0]*n
        if neural_indices:
            bx = np.stack([observations[i][1].vector() for i in neural_indices])
            bl, _, other_hidden = pretrained(torch.from_numpy(bx)[:,None],other_hidden)
            bp, _ = probabilities(bl[:,0,0])
            for j,i in enumerate(neural_indices):
                bb[i] = int(opponent_rngs[i].choice(4,p=bp[j]))
        for i,op in enumerate(opponent_kinds):
            if op == 'random': bb[i] = int(opponent_rngs[i].integers(4))
            elif op == 'fixed': bb[i] = t%4
            elif op != 'ppo': raise ValueError(op)
        obs_record[:,t] = x
        actions[:,t] = aa
        opponent_actions[:,t] = bb
        logs[:,t] = log_probs[np.arange(n),aa]
        if kind == 'dual_random' and t < k: logs[:,t] = -np.log(4.)
        values[:,t] = all_values[:,0,head].numpy()
        heads[:,t] = head
        for i,game in enumerate(games):
            next_obs, reward, _ = game.step(aa[i],bb[i])
            observations[i] = next_obs
            exact_rewards[i,t] = reward
        rewards[:,t] = exact_rewards[:,t,0]
        if trace is not None:
            all_probs = all_logits[:,0].softmax(-1).numpy().astype(np.float64)
            all_probs /= all_probs.sum(-1,keepdims=True)
            behavior = np.full_like(probs,.25) if kind == 'dual_random' and t < k else probs.copy()
            trace.append(dict(head=None if kind == 'dual_random' and t < k else head, policy_probabilities=all_probs.copy(),
                              behavior_probabilities=behavior, values=all_values[:,0].numpy().copy()))
    trajectories = [Trajectory(obs_record[i],actions[i],logs[i],values[i],rewards[i],heads[i]) for i in range(n)]
    return trajectories, exact_rewards, opponent_actions
