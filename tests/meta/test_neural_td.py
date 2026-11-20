import copy
import numpy as np
import pytest
import torch
from torch import nn
from meta_exploration.environment import Game,Observation,generate_world
from meta_exploration.learning import Transition,Controller,ppo_batch_update,normalize_advantages
from meta_exploration.neural_td import (QNetwork,NeuralTD,MetaExplorer,Replay,double_dqn_targets,episode,
    explorer_targets,leave_one_out,FEATURE_SIZE)
from meta_exploration.neural_td_experiment import config,train,model_digest

@pytest.fixture
def cfg():
    torch.set_num_threads(1);torch.manual_seed(3)
    c=config('configs/meta/neural_td_pilot.json')
    c['neural_td'].update(hidden=16,batch_size=8,prefix_updates=1,k_updates=1,pretrain_interactions=16)
    c['training'].update(hidden=16,exploration_steps=3,exploitation_steps=5,episodes=2,rollout_batch_size=2,ppo_epochs=1,train_worlds=2)
    c['evaluation']['num_test_worlds']=2
    return c

class Constant(nn.Module):
    def __init__(self,values):super().__init__();self.register_buffer('v',torch.tensor(values,dtype=torch.float32))
    def forward(self,x):return self.v.expand(len(x),-1)


def test_double_dqn_selection_terminal_and_gamma():
    x=torch.zeros(2,11);online=Constant([1,5,2]);target=Constant([100,3,10])
    for gamma in [.99,1.]:
        y=double_dqn_targets(online,target,x,torch.tensor([2.,2.]),torch.tensor([0.,1.]),gamma)
        torch.testing.assert_close(y,torch.tensor([2+gamma*3,2.]))
        assert not y.requires_grad


def test_known_one_transition_td_improves(cfg):
    d=dict(cfg['neural_td'],gamma=0.,learning_rate=.01)
    q=NeuralTD(QNetwork(16),d,'w',0);o=Observation(0,3,3,0.)
    t=Transition(o,1,4.,o,True,0,{})
    q.observe(t,'w');before=abs(q.values(o)[1]-4)
    q.update(150)
    assert abs(q.values(o)[1]-4)<.05 and abs(q.values(o)[1]-4)<before

@pytest.mark.parametrize('kind,pair',[('type2',(0,0)),('type2',(1,1)),('type2',(2,2)),('type3',(0,1)),('type3',(1,0)),('type3',(2,1))])
def test_ordered_pair_statistics_and_no_mode_terminal(cfg,kind,pair):
    g=Game(generate_world(3,kind,sigma=0),6,0,'constant');q=NeuralTD(QNetwork(16),cfg['neural_td'],g._world.world_id,0)
    for a in pair:
        o=g.observation();n,r,done,b=g.step(a);q.observe(Transition(o,a,r,n,done,b,{}),g._world.world_id)
        assert not done
    assert q.pair_counts[3*pair[0]+pair[1]]==1 and q.pair_counts.sum()==1
    assert g.mode==0 and all(not r[4] for r in q.replay.data)
    assert q.features(g.observation()).shape==(FEATURE_SIZE,)


def test_replay_and_optimizer_world_isolation(cfg):
    initial=QNetwork(16);d=cfg['neural_td'];a=NeuralTD(initial,d,'A',1);b=NeuralTD(initial,d,'B',1)
    assert model_digest(a.model)==model_digest(b.model)==model_digest(initial)
    assert model_digest(a.target)==model_digest(initial)
    o=Observation(0,3,3,0);t=Transition(o,0,3.,o,False,0,{})
    a.observe(t,'A');a.update()
    assert model_digest(a.model)!=model_digest(initial)
    assert not b.optimizer.state and not b.replay.data
    assert model_digest(b.target)==model_digest(initial)
    with pytest.raises(ValueError,match='Cross-world'):b.observe(t,'A')
    assert not b.replay.data and b.steps==0

@pytest.mark.parametrize('timing',['online','batch'])
@pytest.mark.parametrize('online_k',[False,True])
def test_prefix_adaptation_timing_budgets_and_frozen_k(cfg,timing,online_k):
    d=dict(cfg['neural_td'],prefix_timing=timing,prefix_updates=2)
    initial=QNetwork(16);before=model_digest(initial);g=Game(generate_world(3,'type3'),8,0)
    h,_,learner,diag=episode(g,initial,d,3,5,'coverage',0,online_k=online_k)
    assert not h[2].done and h[3].observation.mode==1 and h[-1].done
    assert diag['prefix']['td_updates']==6
    assert diag['total']['td_updates']==6+(5 if online_k else 0)
    assert diag['k_parameter_delta']>0 if online_k else diag['k_parameter_delta']==0
    assert model_digest(learner.model)!=before and model_digest(initial)==before
    assert not any(isinstance(m,nn.GRU) for m in learner.model.modules())
    assert learner.replay.inserted==(8 if online_k else 3)


def test_downstream_actor_signal_and_mc_credit(cfg):
    m=MetaExplorer(16);initial=QNetwork(16);g=Game(generate_world(2,'type3'),8,0)
    h,prefix,_,_=episode(g,initial,cfg['neural_td'],3,5,'meta',0,m)
    episodes=[]
    for r in [1.,10.]:
        suffix=copy.deepcopy(h[3:])
        for t in suffix:t.reward=r
        a,ret=explorer_targets(prefix,suffix,cfg)
        episodes.append((prefix,a,ret,[]))
    np.testing.assert_allclose(episodes[1][2]-episodes[0][2],45.,atol=1e-5)
    norm=normalize_advantages([e[1] for e in episodes],'batch')
    assert norm[0].mean()<0<norm[1].mean()
    x=torch.as_tensor(np.array([t.observation.vector() for t in prefix]))[None]
    logits,_,_=m(x);logs=torch.distributions.Categorical(logits=logits[0,:,0]).log_prob(torch.tensor([t.action for t in prefix]))
    grads=[]
    for a in norm:
        gs=torch.autograd.grad(-(logs*torch.tensor(a)).mean(),tuple(m.actors.parameters()),retain_graph=True)
        grads.append(torch.cat([g.flatten() for g in gs]))
    assert torch.dot(*grads)<0
    optimizer=torch.optim.Adam(m.parameters(),lr=.001)
    stats=ppo_batch_update(Controller(m,0),optimizer,episodes,cfg)
    assert stats['parameter_delta_norm']>0
    with pytest.raises(ValueError,match='Off-policy'):ppo_batch_update(Controller(m,0),optimizer,episodes,cfg)


def test_loo_baseline_preserves_targets(cfg):
    m=MetaExplorer(16);h,p,_,_=episode(Game(generate_world(2,'type2'),8,0),QNetwork(16),cfg['neural_td'],3,5,'meta',0,m)
    a,r=explorer_targets(p,h[3:],cfg)
    eps=[(p,a,r,[]),(p,a+10,r+10,[])]
    centered=leave_one_out(eps,['w','w'])
    assert centered[0][1].mean()<centered[1][1].mean()
    np.testing.assert_array_equal(centered[0][2],r)
    with pytest.raises(ValueError,match='LOO'):leave_one_out(eps,['w','other'])


def test_train_fixes_initial_and_resume(cfg,tmp_path):
    out=tmp_path/'run';q,m=train(cfg,'A3',0,out)
    fingerprint=model_digest(q);trained=model_digest(m)
    q2,m2=train(cfg,'A3',0,out,resume=True)
    assert model_digest(q2)==fingerprint and model_digest(m2)==trained
    import json
    costs=json.loads((out/'costs.json').read_text())
    assert costs['main_interactions']==16 and costs['inner_td_updates']==16
    worlds=json.loads((out/'worlds_diagnostic.json').read_text())
    assert not {w['fingerprint'] for w in worlds if w['split']=='train'}&{w['fingerprint'] for w in worlds if w['split']=='test'}

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_backward_device(cfg,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    q=NeuralTD(QNetwork(16).to(device),cfg['neural_td'],'w',0)
    o=Observation(0,3,3,0);q.observe(Transition(o,1,3.,o,False,0,{}),'w');q.update()
    assert q.updates==1


def test_timing_same_observed_data_and_update_count(cfg):
    initial=QNetwork(16);g=Game(generate_world(3,'type3'),8,0)
    outputs=[]
    for timing in ['online','batch']:
        h,_,learner,diag=episode(g.snapshot(),initial,dict(cfg['neural_td'],prefix_timing=timing),3,5,'coverage',0,online_k=False)
        outputs.append((h,diag))
    assert [(t.action,t.reward,t.done) for t in outputs[0][0][:3]]==[(t.action,t.reward,t.done) for t in outputs[1][0][:3]]
    assert outputs[0][1]['prefix']['td_updates']==outputs[1][1]['prefix']['td_updates']
    assert outputs[0][1]['b_parameter_delta']>0 and outputs[1][1]['b_parameter_delta']>0


def test_existing_explorer_prefix_is_paired(cfg):
    from meta_exploration.learning import make_model,collect
    g=Game(generate_world(3,'type3'),8,0);model=make_model(cfg)
    h,_,_,_=episode(g.snapshot(),QNetwork(16),cfg['neural_td'],3,5,'existing',9,existing=model)
    ref=collect(g.snapshot(),Controller(model,9),3)
    assert [(t.action,t.reward,t.opponent_action) for t in h[:3]]==[(t.action,t.reward,t.opponent_action) for t in ref]


def test_small_evaluation_and_no_silent_missing_checkpoint(cfg,tmp_path):
    from meta_exploration.neural_td_experiment import evaluate
    import json
    cfg['neural_td'].update(evaluation_budgets=[3],evaluation_timings=['online','batch'])
    cfg['evaluation']['num_test_worlds']=1;cfg['logging']['save_trajectories']=False
    source=tmp_path/'train';train(cfg,'A3',0,source)
    rows=evaluate(cfg,'A3',0,source,tmp_path/'evaluation')
    assert len(rows)==20  # 4 available TD explorers + Q-only, 2 modes, 2 timings
    td=[r for r in rows if r['method'].startswith('neural_td')]
    assert len({r['initial_sha256'] for r in td})==1
    assert all(r['compute']['k_updates']==0 for r in td if r['evaluation']=='frozen_k')
    assert all(r['compute']['k_updates']>0 for r in td if r['evaluation']=='online_k')
    assert 'neural_td_existing' in json.loads((tmp_path/'evaluation/provenance.json').read_text())['missing_methods']
    with pytest.raises(ValueError,match='fresh'):evaluate(cfg,'A3',0,source,tmp_path/'evaluation')


def test_hard_target_update_and_target_no_gradient(cfg):
    d=dict(cfg['neural_td'],target_interval=2)
    initial=QNetwork(16);q=NeuralTD(initial,d,'w',0);o=Observation(0,3,3,0.)
    q.observe(Transition(o,1,5.,o,False,0,{}),'w');q.update()
    assert model_digest(q.target)==model_digest(initial)
    assert model_digest(q.model)!=model_digest(initial)
    q.update()
    assert model_digest(q.target)==model_digest(q.model)
    assert all(p.grad is None and not p.requires_grad for p in q.target.parameters())

@pytest.mark.parametrize('kind',['type1','type2','type3'])
def test_metrics_support_all_world_types(cfg,kind):
    from meta_exploration.neural_td_experiment import diagnostic_scores
    from meta_exploration.learning import collect
    from meta_exploration.training import HistoryCollector
    w=generate_world(7,kind);g=Game(w,8,0)
    h=collect(g,HistoryCollector(0,'coverage'),8)
    score=diagnostic_scores(h,w,3,cfg,'random')
    if kind=='type1':assert score['evidence_group']=='not_applicable' and not score['discovered_but_not_utilized']
    else:assert score['optimal_combo_utilization'] is not None


def test_launcher_excludes_failed_gpu_and_unique_outputs(tmp_path,monkeypatch):
    import types
    import meta_exploration.neural_td_launch as launcher
    monkeypatch.setattr(launcher,'hardware_report',lambda:dict(usable_devices=[1],devices=[dict(index=0,usable=False),dict(index=1,usable=True)]))
    commands=[]
    def fake_run(cmd,**kwargs):commands.append(cmd);return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess,'run',fake_run)
    launcher.launch('config.json',tmp_path/'launch',['A2','A3'],[0,1],require_gpu=True)
    assert len(commands)==4
    assert all(cmd[cmd.index('--device')+1]=='cuda:1' for cmd in commands)
    assert len({cmd[cmd.index('--out')+1] for cmd in commands})==4
    with pytest.raises(ValueError,match='Duplicate'):launcher.launch('c',tmp_path/'bad',['A2'],[0,0])


def test_paired_speed_keeps_censored_worlds():
    from meta_exploration.adaptation import speed_metrics
    from meta_exploration.neural_td_experiment import paired_td_speed
    rows=[]
    for method,rewards in [('neural_td_meta',[0.]*8),('neural_td_random',[10.]*8)]:
        rows.append(dict(regime='A3',world_type='type3',seed=0,world_id='w',prefix_timing='online',evaluation='frozen_k',
            B=3,K=8,method=method,adaptation_speed=speed_metrics(rewards,[10.]*8,[0.]*8,dict(window=2,hold=2))))
    result=paired_td_speed(rows,20)
    r=next(r for r in result if r['metric']=='rmst_0.9' and r['normalization']=='absolute')
    assert r['mean']==5  # nonreacher capped at 8, reference confirmed at 3
    assert next(r for r in result if r['metric']=='reach_0.9')['mean']==-1
