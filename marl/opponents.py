import numpy as np
import torch


class RandomOpponent:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        pass

    def act(self, observation):
        return int(self.rng.integers(4))


class FixedOpponent:
    def reset(self):
        self.t = 0

    def act(self, observation):
        action = self.t % 4
        self.t += 1
        return action


class NeuralController:
    """Private per-rollout recurrent state; frozen weights during acting."""
    def __init__(self, model, rng):
        self.model, self.rng = model, rng
        self.reset()

    def reset(self):
        self.hidden = None

    @torch.no_grad()
    def act_info(self, observation, head=0, random_action=False):
        x = torch.from_numpy(observation.vector())[None, None]
        logits, values, self.hidden = self.model(x, self.hidden)
        log_probs = logits[0, 0, head].log_softmax(-1)
        probs = log_probs.exp().numpy().astype(np.float64)
        probs /= probs.sum()
        action = int(self.rng.integers(4) if random_action else self.rng.choice(4, p=probs))
        return action, float(log_probs[action]), float(values[0, 0, head])

    def act(self, observation):
        return self.act_info(observation)[0]


def make_opponent(kind, rng, pretrained=None):
    if kind == 'random':
        opponent = RandomOpponent(rng)
    elif kind == 'fixed':
        opponent = FixedOpponent()
    elif kind == 'ppo' and pretrained is not None:
        opponent = NeuralController(pretrained, rng)
    else:
        raise ValueError(f"Unknown opponent or missing checkpoint: {kind}")
    opponent.reset()
    return opponent
