"""Tensor-driven simultaneous game; no dependency on the task generator."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Observation:
    mode: int
    previous_own: int
    previous_other: int
    reward: float

    def vector(self):
        # Index 4 denotes no previous action. No tensor, time, ID or combo input.
        x = np.zeros(13, dtype=np.float32)
        x[self.previous_own] = 1
        x[5 + self.previous_other] = 1
        x[10 + self.mode] = 1
        x[12] = self.reward
        return x

    def q_state(self):
        return self.mode, self.previous_own, self.previous_other


@dataclass
class Task:
    r1: np.ndarray  # [player, a_A, a_B]
    r2: np.ndarray  # [player, prev_A, prev_B, a_A, a_B]
    combos: np.ndarray | None = None  # evaluator-only metadata

    def __post_init__(self):
        self.r1 = np.asarray(self.r1, dtype=np.float64).copy()
        self.r2 = np.asarray(self.r2, dtype=np.float64).copy()
        if self.r1.shape != (2, 4, 4) or self.r2.shape != (2, 4, 4, 4, 4):
            raise ValueError("Expected R1=(2,4,4), R2=(2,4,4,4,4), player first")
        if not np.isfinite(self.r1).all() or not np.isfinite(self.r2).all():
            raise ValueError("Rewards must be finite")
        if self.combos is not None:
            self.combos = np.asarray(self.combos, dtype=np.int64).copy()
            if self.combos.shape != (2, 2) or not ((0 <= self.combos) & (self.combos < 4)).all():
                raise ValueError("Combos must be two ordered action pairs")
            self.combos.setflags(write=False)
        self.r1.setflags(write=False)
        self.r2.setflags(write=False)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.r1.tobytes() + self.r2.tobytes()).hexdigest()


class Game:
    """R1 -> R2 -> R1; reward uses the mode before the transition."""
    def __init__(self, task: Task, horizon: int):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.task, self.horizon = task, horizon
        self.reset()

    def reset(self):
        self.t, self.mode = 0, 0
        self.previous = (4, 4)
        self.rewards = np.zeros(2)
        return self.observations()

    def observations(self):
        a, b = self.previous
        return (Observation(self.mode, a, b, float(self.rewards[0])),
                Observation(self.mode, b, a, float(self.rewards[1])))

    def step(self, a: int, b: int):
        if self.t >= self.horizon:
            raise RuntimeError("Rollout has ended")
        if a not in range(4) or b not in range(4):
            raise ValueError("Actions must be in {0,1,2,3}")
        if self.mode == 0:
            rewards = self.task.r1[:, a, b]
        else:
            pa, pb = self.previous
            rewards = self.task.r2[:, pa, pb, a, b]
        self.rewards = rewards.copy()
        self.previous = (a, b)
        self.mode = 1 - self.mode
        self.t += 1
        return self.observations(), self.rewards.copy(), self.t == self.horizon
