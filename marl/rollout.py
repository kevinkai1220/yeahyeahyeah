from dataclasses import dataclass
import numpy as np
from .environment import Game
from .opponents import NeuralController


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    heads: np.ndarray


def collect_rollout(task, model, opponent, horizon, k, rng):
    game = Game(task, horizon)
    observations = game.reset()
    controller = NeuralController(model, rng)
    opponent.reset()
    xs, actions, log_probs, values, rewards, heads = [], [], [], [], [], []
    for t in range(horizon):
        head = int(model.dual and t < k)
        a, log_prob, value = controller.act_info(observations[0], head)
        b = opponent.act(observations[1])
        xs.append(observations[0].vector())
        observations, reward, _ = game.step(a, b)
        actions.append(a)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward[0])
        heads.append(head)
    return Trajectory(np.asarray(xs), np.asarray(actions), np.asarray(log_probs, dtype=np.float32),
                      np.asarray(values, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(heads))
