"""Exact expected finite-horizon best response to Random or cyclic Fixed."""
import numpy as np


def oracle_values(task, horizon, opponent):
    if opponent not in ('random', 'fixed'):
        raise ValueError("Exact oracle is only defined for Random and Fixed")
    # axes: time before action, prev_A, prev_B; 4 is the start sentinel.
    values = np.zeros((horizon+1, 5, 5), dtype=np.float64)
    for t in reversed(range(horizon)):
        probabilities = np.full(4, .25) if opponent == 'random' else np.eye(4)[t % 4]
        continuation = values[t+1, :4, :4]
        if t % 2 == 0:
            q = ((task.r1[0] + continuation) * probabilities[None, :]).sum(axis=-1)
            values[t, :, :] = q.max()
        else:
            q = ((task.r2[0] + continuation[None, None, :, :]) * probabilities).sum(axis=-1)
            values[t, :4, :4] = q.max(axis=-1)
    return values
