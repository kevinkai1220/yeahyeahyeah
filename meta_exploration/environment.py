"""Private world parameters, observed-only state, independent noise/opponent RNGs."""
from dataclasses import dataclass
import copy
import hashlib
import numpy as np

ACTIONS = ('A', 'B', 'C')
N_ACTIONS = 3
SENTINEL = 3


@dataclass(frozen=True)
class Observation:
    mode: int
    previous_own: int
    previous_other: int
    reward: float

    def vector(self):
        x = np.zeros(11, dtype=np.float32)
        x[self.previous_own] = 1
        x[4 + self.previous_other] = 1
        x[8 + self.mode] = 1
        x[10] = self.reward
        return x

    def q_state(self):
        # Preserve observed actions in R1, useful for predicting adaptive opponents.
        return self.mode, self.previous_own, self.previous_other


@dataclass
class World:
    world_id: str
    world_type: str
    base: np.ndarray
    bonus: np.ndarray  # [previous agent, previous opponent, agent, opponent]
    sigma: float
    split: str
    seed: int
    difficulty: str

    def __post_init__(self):
        self.base = np.array(self.base, dtype=float, copy=True)
        self.bonus = np.array(self.bonus, dtype=float, copy=True)
        assert self.base.shape == (3, 3) and self.bonus.shape == (3, 3, 3, 3)
        assert np.isfinite(self.base).all() and np.isfinite(self.bonus).all()
        assert self.sigma >= 0
        self.base.setflags(write=False)
        self.bonus.setflags(write=False)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.base.tobytes() + self.bonus.tobytes()).hexdigest()

    def mean(self, mode, previous, a, b):
        return float(self.base[a, b] + (self.bonus[previous[0], previous[1], a, b] if mode else 0))

    def diagnostic(self):
        return dict(world_id=self.world_id, world_type=self.world_type,
                    base=self.base.tolist(), bonus=self.bonus.tolist(), sigma=self.sigma,
                    split=self.split, seed=self.seed, difficulty=self.difficulty,
                    fingerprint=self.fingerprint)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k != 'fingerprint'})


def structural_validation(world):
    """Conservative sufficient trap check for every opponent pair, not just iid.

    Compare two-step pair returns. A same-action optimum may itself be a repeat.
    Type 3 must beat every repeat; Type 2 must beat immediate-greedy play.
    """
    if world.world_type == 'type1':
        return bool(np.all(world.bonus == 0))
    greedy = world.base.mean(axis=1).argmax()
    target_same = world.world_type == 'type2'
    candidates = [(a, c) for a in range(3) for c in range(3) if (a == c) == target_same]
    for a, c in candidates:
        if world.base[a].max() >= 0 or world.base[c].max() >= 0:
            continue
        good_min = min(world.base[a, b] + world.base[c, d] + world.bonus[a, b, c, d]
                       for b in range(3) for d in range(3))
        repeats = [greedy] if target_same else range(3)
        trap_max = max(world.base[x, b] + world.base[x, d] + world.bonus[x, b, x, d]
                       for x in repeats for b in range(3) for d in range(3))
        if good_min > trap_max + .5:
            return True
    return False


def generate_world(seed, world_type, split='train', difficulty='medium', sigma=1., opponent_dependence=0.):
    if world_type not in ('type1', 'type2', 'type3'):
        raise ValueError(world_type)
    if difficulty not in ('easy', 'medium', 'hard'):
        raise ValueError(difficulty)
    # Split is part of world generation, and reward sampling never touches this RNG.
    rng = np.random.default_rng(np.random.SeedSequence([seed, {'train': 17, 'test': 29}[split]]))
    for _ in range(1000):
        bonus = np.zeros((3, 3, 3, 3))
        if world_type == 'type1':
            gap = rng.uniform(2., 5.)
            base = rng.uniform(-1., 1., (3, 3))
            style = rng.integers(3)
            if style == 0:
                base[rng.integers(3)] += gap
            elif style == 1:
                base[rng.permutation(3), np.arange(3)] += gap
            else:
                risky = rng.integers(3)
                base[risky] += gap
                base[risky, rng.integers(3)] -= 2.5 * gap
        else:
            trap, x, y = rng.permutation(3)
            immediate = {'easy': 1.0, 'medium': 2.5, 'hard': 4.0}[difficulty]
            base = rng.uniform(-2.5, -1., (3, 3))
            base[trap] = immediate + rng.uniform(-.3, .3, 3)
            pair = (x, x if world_type == 'type2' else y)
            margin = {'easy': 7., 'medium': 4., 'hard': 1.5}[difficulty]
            strength = 2 * base[trap].max() - base[pair[0]].min() - base[pair[1]].min() + margin
            bonus[pair[0], :, pair[1], :] = strength + rng.uniform(0., 2.)
            # Additional same-family candidate, weak and not guaranteed beneficial.
            if difficulty == 'easy':
                second = (y, y) if world_type == 'type2' else (y, x)
                bonus[second[0], :, second[1], :] = rng.uniform(2., 4.)
            if opponent_dependence:
                active = bonus != 0
                bonus += active * rng.uniform(-opponent_dependence, opponent_dependence, bonus.shape)
        world = World(f'{split}-{world_type}-{seed}', world_type, base, bonus, sigma, split, seed, difficulty)
        if structural_validation(world):
            return world
    raise RuntimeError('World rejection sampler exhausted: reduce opponent_dependence')


class Game:
    """R1 -> R2 -> R1, no interruption. Only horizon ends an episode.

    Previous actions and reward remain visible in R1, as in v2. Opponent observes
    its own symmetric reward: mean(base[b,a] + bonus[pb,pa,b,a]) plus fresh noise.
    No current opponent action is available before agent action selection.
    """
    def __init__(self, world, horizon, noise_seed, opponent_kind='random', opponent_seed=0,
                 opponent_model=None):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self._world, self.horizon = world, horizon
        self.noise_rng = np.random.default_rng(noise_seed)
        self.opponent_rng = np.random.default_rng(opponent_seed)
        self.opponent_kind = opponent_kind
        self.opponent_model = opponent_model
        if opponent_kind not in ('random', 'fixed', 'constant', 'ppo'):
            raise ValueError(opponent_kind)
        if opponent_kind == 'ppo' and (opponent_model is None or opponent_model.n_actions != 3):
            raise ValueError('PPO opponent requires a three-action pretrained model')
        self.t, self.mode, self.previous = 0, 0, (3, 3)
        self.reward, self.other_reward = 0., 0.
        self.opponent_hidden = None

    def observation(self):
        return Observation(self.mode, *self.previous, self.reward)

    def step(self, a):
        if a not in range(3) or self.t >= self.horizon:
            raise ValueError('invalid action or finished episode')
        if self.opponent_kind == 'random':
            b = int(self.opponent_rng.integers(3))
        elif self.opponent_kind == 'constant':
            b = 0
        elif self.opponent_kind == 'fixed':
            b = self.t % 3
        else:
            import torch
            other = Observation(self.mode, self.previous[1], self.previous[0], self.other_reward)
            with torch.no_grad():
                device = next(self.opponent_model.parameters()).device
                logits, _, self.opponent_hidden = self.opponent_model(
                    torch.tensor(other.vector(), device=device)[None, None], self.opponent_hidden)
                p = logits[0, 0, 0].softmax(-1).cpu().numpy().astype(float)
            b = int(self.opponent_rng.choice(3, p=p / p.sum()))
        mean = self._world.mean(self.mode, self.previous, a, b)
        other_mean = self._world.mean(self.mode, self.previous[::-1], b, a)
        noise = self.noise_rng.normal(0., self._world.sigma, 2)
        self.reward, self.other_reward = mean + noise[0], other_mean + noise[1]
        self.previous, self.mode, self.t = (a, b), 1 - self.mode, self.t + 1
        return self.observation(), float(self.reward), self.t == self.horizon, b

    def snapshot(self):
        # Frozen opponent parameters may be shared; state/RNG/hidden are copied.
        memo = {id(self.opponent_model): self.opponent_model} if self.opponent_model is not None else {}
        return copy.deepcopy(self, memo)


def oracle_values(world, horizon, opponent, random_agent=False):
    """Adapted from marl.oracle: exact finite-horizon DP for iid/cyclic opponents."""
    if opponent not in ('random', 'fixed', 'constant'):
        raise ValueError('Exact oracle unavailable for recurrent PPO opponent')
    values = np.zeros((horizon + 1, 4, 4))
    for t in reversed(range(horizon)):
        p = np.full(3, 1 / 3) if opponent == 'random' else np.eye(3)[0 if opponent == 'constant' else t % 3]
        future = values[t + 1, :3, :3]
        if t % 2 == 0:
            q = ((world.base + future) * p).sum(-1)
            values[t] = q.mean() if random_agent else q.max()
        else:
            q = ((world.base + world.bonus + future) * p).sum(-1)
            values[t, :3, :3] = q.mean(-1) if random_agent else q.max(-1)
    return values
