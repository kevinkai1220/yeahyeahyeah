import unittest
import numpy as np
import torch
from marl.ppo import return_targets, update_ppo
from marl.offpolicy import vtrace_targets, SequenceReplay, update_vtrace
from marl.batched import collect_batch
from marl.generator import generate_task
from marl.models import RecurrentPolicy
from marl.rollout import collect_rollout
from marl.opponents import make_opponent


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.rng=np.random.default_rng(4)

    def test_lambda_returns_and_boundary(self):
        r=np.array([[-3,9,-3,9,1,2]],np.float32)
        np.testing.assert_allclose(return_targets(r,3,True,.5),[[13.5,15,10.5,12,3,2]])
        np.testing.assert_allclose(return_targets(r,3,True,1),return_targets(r,0,False))
        for k in (2,3,4):
            base=return_targets(r,k,True,0)
            full=return_targets(r,k,True,1)
            np.testing.assert_allclose(return_targets(r,k,True,.5),(base+full)/2)
        with self.assertRaises(ValueError): return_targets(r,3,True,-1)

    def test_vtrace_on_policy_is_monte_carlo(self):
        r=torch.tensor([[-3.,9.,-3.,9.]])
        v=torch.tensor([[1.,2.,3.,4.]])
        logs=torch.zeros_like(r)
        target,adv,_=vtrace_targets(r,v,logs,logs)
        torch.testing.assert_close(target,torch.tensor([[12.,15.,6.,9.]]))
        torch.testing.assert_close(adv,target-v)
        self.assertFalse(target.requires_grad)

    def test_vtrace_offpolicy_hand_computation(self):
        r=torch.tensor([[2.,4.]])
        v=torch.tensor([[1.,2.]],requires_grad=True)
        # ratios .5 and 2; latter clips to one.
        targets,adv,ratios=vtrace_targets(r,v,torch.log(torch.tensor([[.25,.5]])),torch.log(torch.tensor([[.5,.25]])))
        torch.testing.assert_close(targets,torch.tensor([[3.5,4.]]))
        torch.testing.assert_close(adv,torch.tensor([[2.5,2.]]))
        self.assertFalse(targets.requires_grad)
        self.assertFalse(adv.requires_grad)

    def test_batch_matches_scalar_history_and_rng(self):
        model=RecurrentPolicy(True); other=RecurrentPolicy()
        tasks=[generate_task(self.rng) for _ in range(3)]
        kinds=['random','fixed','ppo']
        batch,_,_=collect_batch(tasks,model,other,kinds,8,3,
            [np.random.default_rng(i+10) for i in range(3)],
            [np.random.default_rng(i+20) for i in range(3)])
        for i in range(3):
            scalar=collect_rollout(tasks[i],model,make_opponent(kinds[i],np.random.default_rng(i+20),other),8,3,np.random.default_rng(i+10))
            np.testing.assert_array_equal(batch[i].actions,scalar.actions)
            np.testing.assert_allclose(batch[i].log_probs,scalar.log_probs,atol=1e-6)
            np.testing.assert_array_equal(batch[i].rewards,scalar.rewards)

    def test_offpolicy_updates_target_and_history_not_explore_head(self):
        model=RecurrentPolicy(True)
        tasks=[generate_task(self.rng) for _ in range(4)]
        ts,_,_=collect_batch(tasks,model,None,['fixed']*4,8,3,
            [np.random.default_rng(i) for i in range(4)],[self.rng]*4)
        replay=SequenceReplay(3); replay.add(ts)
        self.assertEqual(len(replay.items),3)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        # Populate exploration Adam momentum first, as in the real hybrid loop.
        update_ppo(model,optimizer,ts,3,self.rng,epochs=1,minibatch=4,train_heads=[1])
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=update_vtrace(model,optimizer,replay,self.rng,updates=2,minibatch=2)
        self.assertTrue(all(np.isfinite(x) for x in result.values()))
        for key in ('actors.0.weight','critics.0.weight','gru.weight_hh_l0'):
            self.assertFalse(torch.equal(before[key],model.state_dict()[key]))
        torch.testing.assert_close(before['actors.1.weight'],model.actors[1].weight,rtol=0,atol=0)
        torch.testing.assert_close(before['critics.1.weight'],model.critics[1].weight,rtol=0,atol=0)
        before={k:v.clone() for k,v in model.state_dict().items()}
        update_ppo(model,optimizer,ts,3,self.rng,epochs=1,minibatch=4,train_heads=[1])
        torch.testing.assert_close(before['actors.0.weight'],model.actors[0].weight,rtol=0,atol=0)
        torch.testing.assert_close(before['critics.0.weight'],model.critics[0].weight,rtol=0,atol=0)
        self.assertFalse(torch.equal(before['actors.1.weight'],model.actors[1].weight))

if __name__=='__main__': unittest.main()

class StudyEvaluationTests(unittest.TestCase):
    def test_batched_evaluation_matches_original_all_conditions(self):
        from marl.evaluation import evaluate_rollout
        from marl.study_evaluation import episode_metrics
        from marl.oracle import oracle_values
        torch.set_num_threads(1);torch.manual_seed(12)
        task=generate_task(np.random.default_rng(12))
        model=RecurrentPolicy(True);other=RecurrentPolicy()
        for kind in ('dual_learned','dual_random','dual_target'):
            for op in ('random','fixed','ppo'):
                for k in (2,3):
                    a,b=np.random.SeedSequence(123).spawn(2)
                    trajectories,rewards,other_actions=collect_batch([task],model,other,[op],8,k,
                        [np.random.default_rng(a)],[np.random.default_rng(b)],kind)
                    oracle=None if op=='ppo' else oracle_values(task,8,op)
                    actual=episode_metrics(task,trajectories[0],rewards[0],other_actions[0],k,oracle)
                    expected,_=evaluate_rollout(task,model,kind,op,other,8,k,123,oracle=oracle)
                    for key,value in expected.items():
                        if value is None: self.assertIsNone(actual[key],key)
                        else: self.assertAlmostEqual(actual[key],value,places=6,msg=key)
