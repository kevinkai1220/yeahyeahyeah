"""Example task family; environment also accepts arbitrary external tensors."""
from pathlib import Path
import numpy as np
from .environment import Task


def generate_task(rng, safe_reward=0.6, prefix_cost=-3.0, combo_reward=9.0,
                  interaction=0.05):
    if not (prefix_cost < 0 and safe_reward > 0 and interaction >= 0):
        raise ValueError("Invalid generator reward parameters")
    # The lowest task scale is .8; allow worst-case interaction in both paths.
    if .8 * (prefix_cost + combo_reward - 2 * safe_reward) <= 4 * interaction:
        raise ValueError("Hidden pair must dominate the safe pair for every opponent action")
    r1 = np.zeros((2, 4, 4))
    r2 = np.zeros((2, 4, 4, 4, 4))
    combos = []
    for player in range(2):
        safe, u = rng.choice(4, 2, replace=False)
        v = int(rng.integers(4))
        combos.append((u, v))
        scale = rng.uniform(.8, 1.2)
        for a, b in np.ndindex(4, 4):
            own = (a, b)[player]
            r1[player, a, b] = scale * (safe_reward if own == safe else prefix_cost if own == u else 0)
        for pa, pb, a, b in np.ndindex(4, 4, 4, 4):
            prev, own = ((pa, a), (pb, b))[player]
            r2[player, pa, pb, a, b] = scale * (combo_reward if (prev, own) == (u, v) else safe_reward if own == safe else 0)
    r1 += rng.uniform(-interaction, interaction, r1.shape)
    r2 += rng.uniform(-interaction, interaction, r2.shape)
    return Task(r1, r2, np.array(combos))


def save_tasks(path, tasks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = np.stack([t.combos if t.combos is not None else np.full((2, 2), -1) for t in tasks])
    np.savez_compressed(path, r1=np.stack([t.r1 for t in tasks]),
                        r2=np.stack([t.r2 for t in tasks]), combos=metadata)


def load_tasks(path):
    with np.load(path, allow_pickle=False) as data:
        if len(data['r1']) != len(data['r2']):
            raise ValueError("Task array lengths differ")
        combos = data['combos'] if 'combos' in data else [None] * len(data['r1'])
        return [Task(a, b, None if c is None or np.all(c == -1) else c)
                for a, b, c in zip(data['r1'], data['r2'], combos, strict=True)]


def make_splits(seed, n_train, n_eval, **kwargs):
    train_seed, eval_seed = np.random.SeedSequence(seed).spawn(2)
    train_rng, eval_rng = np.random.default_rng(train_seed), np.random.default_rng(eval_seed)
    train = [generate_task(train_rng, **kwargs) for _ in range(n_train)]
    evaluation = [generate_task(eval_rng, **kwargs) for _ in range(n_eval)]
    if {t.fingerprint for t in train} & {t.fingerprint for t in evaluation}:
        raise RuntimeError("Training/evaluation overlap")
    return train, evaluation
