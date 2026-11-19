import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from marl.environment import Game, Task, Observation
from marl.generator import generate_task, make_splits, save_tasks, load_tasks
from marl.models import RecurrentPolicy
from marl.opponents import make_opponent, NeuralController
from marl.qlearning import OnlineQLearner
from marl.ppo import return_targets, update_ppo
from marl.rollout import collect_rollout
from marl.oracle import oracle_values
from marl.evaluation import evaluate_rollout


class Phase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.rng = np.random.default_rng(7)
        self.task = generate_task(self.rng)

    def test_external_tensors_modes_rewards_and_role(self):
        r1 = np.arange(32).reshape(2, 4, 4)
        r2 = np.arange(512).reshape(2, 4, 4, 4, 4)
        game = Game(Task(r1, r2), 4)
        self.assertEqual(game.reset()[0].previous_own, 4)
        obs, reward, done = game.step(2, 1)
        np.testing.assert_array_equal(reward, r1[:, 2, 1])
        self.assertFalse(done)
        self.assertEqual((obs[1].previous_own, obs[1].previous_other), (1, 2))
        self.assertEqual(obs[1].reward, r1[1, 2, 1])
        np.testing.assert_array_equal(game.step(3, 0)[1], r2[:, 2, 1, 3, 0])
        np.testing.assert_array_equal(game.step(1, 2)[1], r1[:, 1, 2])
        # Last R2 uses the immediately preceding R1, not the earlier combo.
        np.testing.assert_array_equal(game.step(0, 3)[1], r2[:, 1, 2, 0, 3])
        with self.assertRaises(RuntimeError):
            game.step(0, 0)

    def test_generator_combo_independent_of_opponent_and_costly(self):
        for player in range(2):
            u, v = self.task.combos[player]
            for b1, b2 in np.ndindex(4, 4):
                prefix = (u, b1) if player == 0 else (b1, u)
                suffix = (v, b2) if player == 0 else (b2, v)
                cost = self.task.r1[(player,)+prefix]
                reward = self.task.r2[(player,)+prefix+suffix]
                self.assertLess(cost, -2)
                self.assertGreater(reward, 7)
                self.assertGreater(cost+reward, 4)

    def test_no_metadata_in_observation(self):
        obs = Game(self.task, 4).reset()[0]
        self.assertEqual(set(vars(obs)), {'mode', 'previous_own', 'previous_other', 'reward'})
        self.assertEqual(obs.vector().shape, (13,))

    def test_splits_and_external_roundtrip(self):
        train, evaluation = make_splits(0, 8, 8)
        self.assertFalse({t.fingerprint for t in train} & {t.fingerprint for t in evaluation})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tasks.npz'
            save_tasks(path, train + [Task(self.task.r1, self.task.r2)])
            recovered = load_tasks(path)
            self.assertEqual([t.fingerprint for t in recovered[:-1]], [t.fingerprint for t in train])
            self.assertIsNone(recovered[-1].combos)

    def test_exploration_returns_exclude_costs(self):
        rewards = np.array([[-3, 9, -3, 9, 1, 2]], dtype=np.float32)
        np.testing.assert_array_equal(return_targets(rewards, 3, True), [[12, 12, 12, 12, 3, 2]])
        np.testing.assert_array_equal(return_targets(rewards, 0, False), [[15, 18, 9, 12, 3, 2]])

    def test_oracle_matches_exhaustive_expectation(self):
        horizon = 4
        for opponent in ('random', 'fixed'):
            @lru_cache(None)
            def brute(t, pa, pb):
                if t == horizon:
                    return 0.
                bs = range(4) if opponent == 'random' else [t % 4]
                return max(sum((self.task.r1[0, a, b] if t % 2 == 0 else self.task.r2[0, pa, pb, a, b])
                               + brute(t+1, a, b) for b in bs)/len(bs) for a in range(4))
            values = oracle_values(self.task, horizon, opponent)
            self.assertAlmostEqual(values[0, 4, 4], brute(0, 4, 4))
            for pa, pb in np.ndindex(4, 4):
                self.assertAlmostEqual(values[1, pa, pb], brute(1, pa, pb))
        with self.assertRaises(ValueError):
            oracle_values(self.task, horizon, 'ppo')

    def test_q_update_terminal_reset_and_state(self):
        q = OnlineQLearner(self.rng, alpha=.5, epsilon=0)
        obs, nxt = Observation(0, 4, 4, 0), Observation(1, 0, 0, -3)
        q.q[nxt.q_state()] = 8
        q.update(obs, 0, -3, nxt, False)
        self.assertEqual(q.q[obs.q_state()+(0,)], 2.5)
        q.update(obs, 1, 9, nxt, True)
        self.assertEqual(q.q[obs.q_state()+(1,)], 4.5)
        q.reset()
        self.assertEqual(q.q.sum(), 0)

    def test_rollout_hidden_reset_and_no_weight_update(self):
        model = RecurrentPolicy(dual=True)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        def collect():
            return collect_rollout(self.task, model, make_opponent('fixed', self.rng), 6, 3, np.random.default_rng(42))
        first, second = collect(), collect()
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.heads, [1, 1, 1, 0, 0, 0])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        # Replayed recurrent log probabilities match stepwise rollout, including switch.
        logits, _, _ = model(torch.from_numpy(first.observations)[None])
        selected = logits[0, torch.arange(6), torch.from_numpy(first.heads)]
        logs = selected.log_softmax(-1).gather(1, torch.from_numpy(first.actions)[:, None]).squeeze(1)
        np.testing.assert_allclose(logs.detach().numpy(), first.log_probs, atol=1e-6)

    def test_target_gradients_reach_exploration_history(self):
        model = RecurrentPolicy(dual=True)
        obs = torch.randn(2, 6, 13, requires_grad=True)
        logits, values, _ = model(obs)
        (logits[:, 3:, 0].square().mean() + values[:, 3:, 0].square().mean()).backward()
        self.assertGreater(obs.grad[:, :3].abs().sum().item(), 0)
        self.assertIsNotNone(model.gru.weight_hh_l0.grad)

    def test_ppo_updates_both_heads_and_shared_gru(self):
        model = RecurrentPolicy(dual=True)
        trajectories = [collect_rollout(self.task, model, make_opponent('random', self.rng), 8, 3, self.rng) for _ in range(4)]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        metrics = update_ppo(model, torch.optim.Adam(model.parameters(), lr=.001), trajectories, 3, self.rng, epochs=2, minibatch=2)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        self.assertIn('exploration_entropy', metrics)
        for key in ('actors.0.weight', 'actors.1.weight', 'critics.0.weight', 'critics.1.weight', 'gru.weight_hh_l0'):
            self.assertFalse(torch.equal(before[key], model.state_dict()[key]), key)

    def test_checkpoint_roundtrip(self):
        model = RecurrentPolicy(dual=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'agent.pt'
            model.save(path, k=3)
            recovered, metadata = RecurrentPolicy.load(path)
            self.assertEqual(metadata['k'], 3)
            x = torch.randn(2, 6, 13)
            torch.testing.assert_close(model(x)[0], recovered(x)[0])

    def test_metrics_and_odd_k(self):
        model = RecurrentPolicy(dual=True)
        oracle = oracle_values(self.task, 8, 'fixed')
        metrics, curves = evaluate_rollout(self.task, model, 'dual_learned', 'fixed', None, 8, 3, 3, oracle=oracle)
        rewards = curves['reward']
        self.assertAlmostEqual(metrics['exploitation_return'], rewards[3:].sum())
        self.assertAlmostEqual(metrics['adaptation_auc'], np.mean(rewards[3:].cumsum()/np.arange(1, 6)))
        self.assertAlmostEqual(metrics['total_reward'], rewards.sum())
        self.assertAlmostEqual(metrics['oracle_regret'], metrics['oracle_expected_return']-rewards[3:].sum())

    def test_baseline_trajectory_does_not_depend_on_k(self):
        model = RecurrentPolicy()
        for kind in ('single_gru', 'q_learning'):
            _, curve1 = evaluate_rollout(self.task, model, kind, 'random', None, 8, 2, 123)
            _, curve2 = evaluate_rollout(self.task, model, kind, 'random', None, 8, 5, 123)
            np.testing.assert_array_equal(curve1['reward'], curve2['reward'])

    def test_unknown_combo_metadata_not_counted_as_failure(self):
        task = Task(self.task.r1, self.task.r2)
        metrics, _ = evaluate_rollout(task, None, 'q_learning', 'fixed', None, 6, 2, 123)
        self.assertIsNone(metrics['combo_discovered'])
        self.assertIsNone(metrics['first_combo_round'])


if __name__ == '__main__':
    unittest.main()
