import copy
import json
import threading
import types
import numpy as np
import pytest
import torch
from meta_exploration.config import load_config
from meta_exploration.environment import Game, generate_world
from meta_exploration.learning import (Controller, collect, collect_batched, make_model, targets, ppo_update,
                                      action_statistics)
from meta_exploration.training import world_set
from meta_exploration.evaluation import evaluate_worlds


def config():
    torch.set_num_threads(1)
    return load_config('configs/meta/smoke.json')


def equal_history(a,b):
    assert len(a)==len(b)
    assert [t.action for t in a]==[t.action for t in b]
    assert [t.reward for t in a]==[t.reward for t in b]
    assert [t.observation for t in a]==[t.observation for t in b]
    for x,y in zip(a,b):
        if 'value' in x.info:
            np.testing.assert_allclose(x.info['value'],y.info['value'],atol=1e-6)
            np.testing.assert_allclose(x.info['log_prob'],y.info['log_prob'],atol=1e-6)


def test_single_transfer_statistics_match_categorical():
    torch.manual_seed(1)
    logits, values = torch.randn(5,3), torch.randn(5)
    dist = torch.distributions.Categorical(logits=logits)
    stats = action_statistics(logits, values)
    np.testing.assert_array_equal(stats[:,:3],dist.probs.numpy())
    np.testing.assert_array_equal(stats[:,3:6],dist.logits.numpy())
    np.testing.assert_array_equal(stats[:,6],values.numpy())


def test_batched_collection_matches_scalar_including_prefix():
    c=config(); torch.manual_seed(2)
    model = make_model(c)
    games=[Game(generate_world(i,'type3'), 10, i) for i in range(5)]
    controls=[Controller(model,i) for i in range(5)]
    reference=[collect(g.snapshot(),ctrl,3) for g,ctrl in zip(games,controls)]
    prefix=collect_batched(games,model,list(range(5)),3)
    for a,b in zip(reference,prefix): equal_history(a,b)
    scalar=[]
    for i,g in enumerate(games):
        ctrl=Controller(model,i+10);ctrl.ingest(prefix[i])
        scalar.append(collect(g.snapshot(),ctrl,7))
    batch=collect_batched(games,model,[i+10 for i in range(5)],7,prefixes=prefix)
    for a,b in zip(scalar,batch): equal_history(a,b)
    assert all(g.t==10 for g in games)


@pytest.mark.parametrize('mode',['online','frozen'])
def test_batched_evaluation_preserves_pairing_and_isolation(mode):
    c=config();torch.manual_seed(5)
    models={k:make_model(c) for k in ['single_policy','dual_neural','explorer']}
    worlds=world_set(c,'B',0,'test')
    scalar=list(evaluate_worlds(c,models,worlds,0,mode,frozen_batch_size=1))
    batched=list(evaluate_worlds(c,models,worlds,0,mode,frozen_batch_size=2))
    for (w,(a,am)),(v,(b,bm)) in zip(scalar,batched):
        assert w.world_id==v.world_id
        for method in a:
            equal_history(a[method],b[method])
            assert am[method]['online_optimizer_steps']==bm[method]['online_optimizer_steps']
        for x,y in zip(b['dual_neural'][:3],b['dual_q_table'][:3]): assert x is y


def test_skip_refresh_keeps_ppo_weights_identical():
    c=config();torch.manual_seed(5)
    m=make_model(c); clone=copy.deepcopy(m)
    a,b=Controller(m,1),Controller(clone,1)
    game=Game(generate_world(1,'type2'),8,1)
    h=collect(game.snapshot(),a,8);h2=collect(game.snapshot(),b,8)
    adv,ret=targets(h,c)
    for ctrl,hist,refresh in [(a,h,True),(b,h2,False)]:
        ppo_update(ctrl,torch.optim.Adam(ctrl.model.parameters(),lr=.001),hist,adv,ret,c,refresh=refresh)
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in m.state_dict().items())


def test_multiple_workers_per_gpu_dispatch_independent_jobs(tmp_path,monkeypatch):
    import meta_exploration.hardware as hardware
    c=config()
    monkeypatch.setattr(hardware,'hardware_report',lambda:{'usable_devices':[0,1]})
    barrier=threading.Barrier(4)
    calls=[];lock=threading.Lock()
    def fake_run(cmd,**kwargs):
        if cmd[0]=='nvidia-smi': return types.SimpleNamespace(returncode=0,stdout='',stderr='')
        with lock: calls.append((cmd[cmd.index('--device')+1],cmd[cmd.index('--out')+1]))
        barrier.wait(timeout=10)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(hardware.subprocess,'run',fake_run)
    hardware.launch(c,'configs/meta/smoke.json',tmp_path,['B'],[0,1,2,3],workers_per_gpu=2)
    assert sorted(d for d,_ in calls)==['cuda:0','cuda:0','cuda:1','cuda:1']
    assert len({p for _,p in calls})==4
    assert json.loads((tmp_path/'scheduling.json').read_text())['workers_per_gpu']==2
