import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from marl.batched import collect_batch
from marl.generator import generate_task
from marl.models import RecurrentPolicy
from marl.tracking import describe_episode,aggregate
from marl.training import train


class TrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(91)
        self.rng=np.random.default_rng(91)
        self.tasks=[generate_task(self.rng) for _ in range(2)]
        self.model=RecurrentPolicy(True)

    def test_trace_does_not_change_rollouts_or_rng(self):
        def run(trace):
            rngs=[np.random.default_rng(3+i) for i in range(2)]
            out=collect_batch(self.tasks,self.model,None,['fixed']*2,8,3,rngs,
                [np.random.default_rng(8+i) for i in range(2)],'dual_random',trace)
            return out,[r.random() for r in rngs]
        plain,pstate=run(None);trace=[];tracked,tstate=run(trace)
        np.testing.assert_array_equal(plain[1],tracked[1]);self.assertEqual(pstate,tstate)
        for a,b in zip(plain[0],tracked[0]):np.testing.assert_array_equal(a.actions,b.actions)
        self.assertIsNone(trace[0]['head'])
        np.testing.assert_array_equal(trace[0]['behavior_probabilities'],np.full((2,4),.25))
        self.assertEqual(trace[0]['policy_probabilities'].shape,(2,2,4))

    def test_first_hit_reuse_odd_boundary_and_censoring(self):
        trace=[]
        ts,rs,other=collect_batch(self.tasks,self.model,None,['fixed']*2,8,3,
            [np.random.default_rng(i) for i in range(2)],[self.rng]*2,trace=trace)
        u,v=self.tasks[0].combos[0];w=(int(v)+1)%4;z=(int(u)+1)%4
        ts[0].actions[:]=[u,w,u,v,u,v,z,w]
        ep,steps=describe_episode(self.tasks[0],ts[0],rs[0],other[0],trace,0,3)
        self.assertEqual(ep['first_combo_round'],4);self.assertEqual(ep['first_repeat_round'],6)
        self.assertEqual(ep['repeat_delay_rounds'],2)
        self.assertFalse(ep['discovered_by_k']);self.assertEqual(ep['exploit_reuse_pairs'],2)
        self.assertEqual(ep['exploit_reuse_hits'],1)
        self.assertFalse(steps[3]['previously_discovered']);self.assertTrue(steps[4]['previously_discovered'])
        ts[0].actions[:]=[z,w]*4
        miss,miss_steps=describe_episode(self.tasks[0],ts[0],rs[0],other[0],trace,0,3)
        self.assertIsNone(miss['first_combo_round']);self.assertTrue(miss['first_combo_censored'])
        result=aggregate([ep,miss],[steps,miss_steps],8,3)
        self.assertEqual(result['restricted_mean_first_combo_round'],6)
        self.assertEqual(result['discovery_rate'],.5)
        self.assertEqual(result['discovery_cdf'][3]['discovery_rate'],.5)
        self.assertEqual(result['after_discovery_by_pair_lag'][0]['combo_reuse_rate'],1)
        self.assertEqual(result['after_discovery_by_pair_lag'][0]['episodes'],1)

    def test_training_tracker_is_observational_and_preserves_checkpoints(self):
        cfg=json.loads(Path('configs/pilot.json').read_text())
        cfg.update(horizon=8,budgets=[3],updates=2,tasks_per_update=2,rollouts_per_task=1,
            checkpoint_every=1,batched_collection=True,exploit_learning='vtrace',replay_updates=1)
        cfg['ppo']=dict(cfg['ppo'],epochs=1,minibatch=2)
        opponent=RecurrentPolicy();opponent.requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            plain=Path(tmp)/'plain';plain.mkdir();tracked=Path(tmp)/'tracked';tracked.mkdir()
            a=train(self.tasks,cfg,plain,'dual_k3',dual=True,k=3,pretrained=opponent)
            cfg['tracking']=dict(enabled=True,every=1,tasks=2,rollouts=1)
            b=train(self.tasks,cfg,tracked,'dual_k3',dual=True,k=3,pretrained=opponent)
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(len(list((tracked/'checkpoints/dual_k3').glob('*.pt'))),3)
            with (tracked/'tracking/dual_k3/learning_curve.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),9)
            self.assertEqual({r['update'] for r in rows},{'0','1','2'})

if __name__=='__main__':unittest.main()
