import unittest
import numpy as np
import torch
from marl.environment import Game
from marl.generator import generate_task
from marl.models import RecurrentPolicy
from marl.gpu_training import collect, optimize

class V2Tests(unittest.TestCase):
    def test_injected_history_matches_environment(self):
        torch.set_num_threads(1)
        task = generate_task(np.random.default_rng(1))
        model = RecurrentPolicy(True)
        r1 = torch.tensor(np.stack([task.r1]), dtype=torch.float32)
        r2 = torch.tensor(np.stack([task.r2]), dtype=torch.float32)
        obs, actions, _, _, rewards = collect(model, None, r1, r2, torch.tensor(task.combos[:1]), 8, 3, True, 1.)
        self.assertEqual(actions[0,:2].tolist(), task.combos[0].tolist())
        game = Game(task, 8)
        for t in range(7):
            b = int(obs[0,t+1,5:10].argmax())
            _, reward, _ = game.step(int(actions[0,t]), b)
            self.assertAlmostEqual(rewards[0,t].item(), reward[0], places=5)
            np.testing.assert_allclose(obs[0,t+1].numpy(), game.observations()[0].vector(), atol=1e-6)

    def test_exploration_keeps_exploit_frozen(self):
        task = generate_task(np.random.default_rng(2))
        model = RecurrentPolicy(True)
        model.requires_grad_(False)
        model.actors[1].requires_grad_(True); model.critics[1].requires_grad_(True)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        batch = collect(model, None, torch.tensor(np.stack([task.r1]*4),dtype=torch.float32), torch.tensor(np.stack([task.r2]*4),dtype=torch.float32),torch.tensor(np.stack([task.combos[0]]*4)),8,3)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
        optimize(model,opt,batch,3,False,{'ppo':dict(epochs=1,minibatch=4,clip=.2,value_coefficient=.5,max_grad_norm=.5)})
        for name,v in model.state_dict().items():
            if not name.startswith(('actors.1','critics.1')):
                self.assertTrue(torch.equal(v,before[name]), name)
        self.assertFalse(torch.equal(model.actors[1].weight,before['actors.1.weight']))
