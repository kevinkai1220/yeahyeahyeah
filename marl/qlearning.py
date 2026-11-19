import numpy as np


class OnlineQLearner:
    """Fresh tabular learner per rollout; no reward/task metadata access."""
    def __init__(self, rng, alpha=.2, epsilon=.2, gamma=1.0):
        if not (0 < alpha <= 1 and 0 <= epsilon <= 1 and gamma == 1):
            raise ValueError("Expected 0<alpha<=1, 0<=epsilon<=1, gamma=1")
        self.rng, self.alpha, self.epsilon, self.gamma = rng, alpha, epsilon, gamma
        self.reset()

    def reset(self):
        self.q = np.zeros((2, 5, 5, 4), dtype=np.float64)

    def act(self, observation):
        values = self.q[observation.q_state()]
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, observation, action, reward, next_observation, done):
        index = observation.q_state() + (action,)
        future = 0.0 if done else self.q[next_observation.q_state()].max()
        self.q[index] += self.alpha * (reward + self.gamma * future - self.q[index])
