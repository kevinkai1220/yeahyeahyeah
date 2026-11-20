import copy
import numpy as np
import pytest
import torch
from meta_exploration.config import load_config
from meta_exploration.environment import Game, generate_world
from meta_exploration.learning import Controller, collect, make_model, targets, ppo_batch_update, clone_model
from meta_exploration.history_diagnostics import (run_world, probabilities, observed_stats, action_shuffle,
    digest_state, ordered_probability, linear_probe)

@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    c=load_config('configs/meta/smoke.json')
    c['training'].update(exploration_steps=5,exploitation_steps=8,rollout_batch_size=2,
        advantage_normalization='batch',dual_credit='observed_suffix',gradient_diagnostics=True)
    return c

@pytest.mark.parametrize('mode',['frozen','online'])
def test_common_snapshot_odd_boundary_and_no_model_leak(cfg,mode):
    model=make_model(cfg);models={k:clone_model(model) for k in ['single_policy','dual_neural','explorer']}
    before=copy.deepcopy(model.state_dict())
    rows,probes=run_world(cfg,models,generate_world(8,'type3'),11,mode,candidates=2)
    assert len({r['snapshot_hash'] for r in rows})==1
    neural=[r for r in rows if r['method']=='dual_neural']
    assert len(neural)==9
    assert all(r['ordered_diagnostic']['first_action_probability'] is None for r in neural)
    for r in neural:
        assert r['suffix'][0]['mode']==2
        assert len(r['suffix'])==8
        assert r['optimizer_steps']>0 if mode=='online' else r['optimizer_steps']==0
    for m in models.values():
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert len(probes)==2


def test_ingest_and_observed_probe_no_last_reward_leak(cfg):
    model=make_model(cfg);game=Game(generate_world(1,'type2'),20,4)
    h=collect(game,Controller(model,7),6)
    a=Controller(model,0);a.ingest(h)
    b=Controller(model,0)
    for t in h:b.act(t.observation)
    torch.testing.assert_close(a.hidden,b.hidden)
    changed=copy.deepcopy(h);changed[-1].reward+=999
    np.testing.assert_array_equal(observed_stats(h),observed_stats(changed))
    assert sum(observed_stats(h)[:12])==5
    scrambled=action_shuffle(h,0)
    for previous,current in zip(scrambled,scrambled[1:]):
        assert current.observation.previous_own==previous.action
    assert digest_state(game)==digest_state(game.snapshot())


def test_ordered_probe_is_nonmutating_and_distinguishes_modes(cfg):
    model=make_model(cfg);game=Game(generate_world(1,'type3'),20,4)
    before=digest_state(game)
    row=ordered_probability(model,[],game,(0,2))
    assert row['ordered_product_probability']==pytest.approx(row['first_action_probability']*row['second_action_probability'])
    assert digest_state(game)==before
    game.step(0)
    assert ordered_probability(model,[],game,(0,2))['first_action_probability'] is None

@pytest.mark.parametrize('minibatch',[0,1,2])
def test_global_normalization_cadence_ragged_and_gradient_logs(cfg,minibatch):
    cfg['training'].update(sequence_minibatch_size=minibatch,ppo_epochs=2)
    model=make_model(cfg);episodes=[]
    for i,length in enumerate([5,8,7]):
        h=collect(Game(generate_world(i,'type3'),length,i),Controller(model,i),length)
        a,r=targets(h,cfg);episodes.append((h,a,r,[]))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    stat=ppo_batch_update(Controller(model,0),opt,episodes,cfg)
    assert stat['updates']==2*int(np.ceil(3/(minibatch or 3)))
    assert stat['transitions']==20 and stat['optimized_token_passes']==40
    assert stat['normalized_advantage_mean']==pytest.approx(0,abs=1e-6)
    assert stat['normalized_advantage_std']==pytest.approx(1,abs=1e-6)
    assert -1.00001<=stat['shared_gradient_cosine']<=1.00001
    assert stat['post_clipping_gradient_norm']<=cfg['training']['max_grad_norm']+1e-5
    assert stat['actor_head_parameter_delta']>0 and stat['shared_parameter_delta']>0
    with pytest.raises(ValueError,match='Off-policy'):
        ppo_batch_update(Controller(model,0),opt,episodes,cfg)


def test_critic_scaling_does_not_change_actor_targets(cfg):
    model=make_model(cfg);h=collect(Game(generate_world(1,'type3'),8,2),Controller(model,0),8)
    a,r=targets(h,cfg);changed=copy.deepcopy(cfg);changed['training']['critic_loss_scale']=10
    a2,r2=targets(h,changed)
    np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(r,r2)


def test_probe_requires_independent_seed(cfg):
    assert 'undefined' in linear_probe([dict(model='x',seed=0,world_id='a',hidden=[0],targets=[1])])[0]['status']


def test_parameter_group_clipping_and_cost_counter(cfg,tmp_path):
    from meta_exploration.training import train
    cfg['training'].update(gradient_clipping='parameter_groups',sequence_minibatch_size=1,
        episodes=2,pretraining_episodes=3,train_worlds=1,ppo_epochs=1)
    cfg['logging']['save_trajectories']=False
    train(cfg,'A3',0,tmp_path)
    import json
    costs=json.loads((tmp_path/'costs.json').read_text())
    assert costs['single_optimizer_steps']==2
    assert costs['extra_explorer_optimizer_steps']==2
    assert costs['pretraining_optimizer_steps']==3
    rows=[json.loads(l) for l in (tmp_path/'training.jsonl').read_text().splitlines()]
    for r in rows:
        if r['stage'].endswith('_update'):
            assert r['shared_post_clip_gradient_norm']<=cfg['training']['max_grad_norm']+1e-5
            assert r['actor_head_post_clip_gradient_norm']<=cfg['training']['max_grad_norm']+1e-5


def test_oracle_constant_deterministic(cfg):
    from meta_exploration.optimization_pilot import oracle_rollout
    w=generate_world(1,'type3',sigma=0.)
    row=oracle_rollout(Game(w,12,0,'constant'),w)
    assert row['realized_return']==pytest.approx(row['oracle_expected_return'])
    assert len(set(tuple(p) for p in row['pairs']))==1


def test_fixed_checkpoint_probe_world_split():
    from meta_exploration.history_diagnostics import within_checkpoint_probes
    rows=[dict(model='x',seed=seed,world_id=f'{seed}-{i}',hidden=[i,i*i],targets=[i]) for seed in [0,1] for i in range(12)]
    results=within_checkpoint_probes(rows)
    assert len(results)==2
    for r in results:
        assert not set(r['train_worlds'])&set(r['test_worlds'])
        assert r['skill_vs_train_mean'][0]>.9


def test_observed_evidence_repetition_no_hidden_access(cfg):
    from meta_exploration.history_diagnostics import observed_evidence_candidate,evidence_score
    g=Game(generate_world(1,'type3',sigma=0.),40,0,'constant')
    h=observed_evidence_candidate(g,32,0)
    assert len(h)==32 and g.t==0
    pairs=[(t.observation.previous_own,t.action) for t in h[16:] if t.observation.mode]
    assert len(set(pairs))==1 and len(pairs)==8
    assert np.isfinite(evidence_score(h))
