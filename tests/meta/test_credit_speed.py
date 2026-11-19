import copy
import numpy as np
import pytest
import torch
from meta_exploration.config import load_config
from meta_exploration.learning import (Controller, collect, make_model, targets, dual_targets,
    normalize_advantages, ppo_batch_update, adapt, Transition)
from meta_exploration.environment import Game, generate_world, Observation, oracle_values
from meta_exploration.adaptation import speed_metrics, reference_curves


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    c=load_config('configs/meta/smoke.json')
    c['training'].update(advantage_normalization='batch',dual_credit='observed_suffix',rollout_batch_size=2)
    return c


@pytest.mark.parametrize('lam',[1.,.95])
def test_downstream_batch_advantage_and_actor_gradient(cfg,lam):
    cfg['training'].update(gae_lambda=lam,exploration_steps=3,value_coefficient=0.,entropy_coefficient=0.,ppo_epochs=1)
    torch.manual_seed(5)
    model=make_model(cfg)
    h=collect(Game(generate_world(1,'type3'),8,2),Controller(model,1),8)
    # Identical sampled prefix, different OBSERVED suffix returns, zero critics.
    episodes=[]
    for reward in [1.,10.]:
        copy_h=copy.deepcopy(h)
        for t in copy_h: t.info['value']=0.
        for t in copy_h[:3]: t.reward=0.
        for t in copy_h[3:]: t.reward=reward
        (a,r),_=dual_targets(copy_h[:3],copy_h[3:],cfg)
        episodes.append((copy_h[:3],a,r,[]))
    norm=normalize_advantages([x[1] for x in episodes],'batch')
    assert norm[0].mean()<0<norm[1].mean()
    legacy=normalize_advantages([x[1] for x in episodes],'legacy_episode')
    np.testing.assert_allclose(legacy[0],legacy[1],atol=2e-6)
    # Identical action scores cancel at lambda=1 when both counterfactual prefixes
    # are updated together. Verify each episode's contribution, not a false claim
    # that their SUM must favor one identical action over itself.
    x=torch.tensor(np.array([t.observation.vector() for t in h[:3]]))[None]
    logits,_,_=model(x)
    logs=torch.distributions.Categorical(logits=logits[0,:,0]).log_prob(torch.tensor([t.action for t in h[:3]]))
    gradients=[]
    for a in norm:
        grads=torch.autograd.grad(-(logs*torch.tensor(a)).mean(),tuple(model.actors.parameters()),retain_graph=True)
        gradients.append(torch.cat([g.flatten() for g in grads]))
    assert gradients[0].norm()>0 and gradients[1].norm()>0
    assert torch.dot(gradients[0],gradients[1])<0


def test_dual_boundary_independent_of_suffix_critic(cfg):
    h=collect(Game(generate_world(1,'type2'),8,0),Controller(make_model(cfg),0),8)
    a=dual_targets(h[:3],h[3:],cfg)[0]
    changed=copy.deepcopy(h)
    for t in changed[3:]:t.info['value']+=10000
    b=dual_targets(changed[:3],changed[3:],cfg)[0]
    np.testing.assert_allclose(a,b)
    cfg['training']['gae_lambda']=1.
    a,r=dual_targets(h[:3],h[3:],cfg)[0]
    expected=sum(t.reward for t in h[3:])+cfg['training']['exploration_lambda']*sum(t.reward for t in h[:3])
    assert r[0]==pytest.approx(expected,abs=1e-5)
    with pytest.raises(ValueError,match='complete'):
        dual_targets(h[:3],h[3:-1],cfg)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batch_backward_versions_ragged_eval_mode(cfg,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable; run on GPU server')
    torch.manual_seed(0)
    model=make_model(cfg,device).eval()
    episodes=[]
    for length in [5,8]:
        ctrl=Controller(model,0)
        assert ctrl.hidden is None
        h=collect(Game(generate_world(length,'type2'),length,0),ctrl,length)
        a,r=targets(h,cfg)
        episodes.append((h,a,r,[]))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    m=ppo_batch_update(Controller(model,0),opt,episodes,cfg)
    assert m['transitions']==13 and m['parameter_delta_norm']>0 and not model.training
    with pytest.raises(ValueError,match='Off-policy'):
        ppo_batch_update(Controller(model,0),opt,episodes,cfg)
    # Adapt from eval mode, exercising cuDNN forward/backward mode management.
    adapt(Game(generate_world(2,'type1'),8,0),Controller(model,0),8,opt,cfg,True)
    assert not model.training


def test_hidden_independence_and_no_padding_loss(cfg):
    model=make_model(cfg)
    game=Game(generate_world(2,'type2'),8,0)
    a=Controller(model,10);b=Controller(model,10)
    collect(game,a,3)
    h1=collect(Game(generate_world(2,'type2'),8,0),b,8)
    h2=collect(Game(generate_world(2,'type2'),8,0),Controller(model,10),8)
    assert [t.info for t in h1]==[t.info for t in h2]
    episodes=[]
    for h in [h1[:3],h2]:
        adv,ret=targets(h,cfg);episodes.append((h,adv,ret,[]))
    # Zero learning rate: joint forward must match isolated recurrent forwards.
    m=ppo_batch_update(Controller(model,0),torch.optim.Adam(model.parameters(),lr=0),episodes,cfg)
    assert abs(m['approximate_kl'])<1e-6


def test_speed_confirmed_crossing_censoring_and_undefined():
    c=dict(window=3,hold=2,thresholds=[.25,.5,.75,.9])
    s=speed_metrics([1]*8,[1]*8,[0]*8,c)
    assert s['absolute']['time_to_90pct']==4
    assert s['absolute']['normalized_adaptation_auc']==6
    s=speed_metrics([0]*8,[1]*8,[0]*8,c)
    e=s['absolute']['events']['0.25']
    assert e['time'] is None and not e['event'] and e['restricted_time']==8
    s=speed_metrics([10]*8,[0]*8,[0]*8,c)
    assert not s['absolute']['defined'] and not s['baseline_normalized']['defined']
    s=speed_metrics([10]*8,[1]*8,[1]*8,c)
    assert s['absolute']['curve'][-1]==10 and not s['baseline_normalized']['defined']
    s=speed_metrics([1,1],[1,1],[0,0],c)
    assert s['absolute']['normalized_adaptation_auc'] is None


@pytest.mark.parametrize('opponent',['random','fixed','constant'])
@pytest.mark.parametrize('B',[2,3])
def test_reference_matches_dp_odd_mode_and_random(opponent,B):
    w=generate_world(3,'type3',sigma=0)
    g=Game(w,B+7,1,opponent)
    for i in range(B):g.step(i%3)
    for random in [False,True]:
        curve=reference_curves(w,B,7,g.previous,opponent,random)
        dp=oracle_values(w,B+7,opponent,random_agent=random)
        assert curve.sum()==pytest.approx(dp[B,*g.previous])


def test_speed_spike_does_not_satisfy_hold():
    s=speed_metrics([0,0,100,0,0],[1]*5,[0]*5,dict(window=1,hold=2))
    assert s['absolute']['time_to_25pct'] is None


def test_lambda_one_matches_legacy_and_point95_separates_critics(cfg):
    h=collect(Game(generate_world(1,'type3'),8,0),Controller(make_model(cfg),0),8)
    cfg['training']['gae_lambda']=1.
    corrected=dual_targets(h[:3],h[3:],cfg)[0]
    cfg['training']['dual_credit']='legacy_cross_critic'
    legacy=dual_targets(h[:3],h[3:],cfg)[0]
    np.testing.assert_allclose(corrected,legacy,atol=1e-5)
    cfg['training']['gae_lambda']=.95
    legacy=dual_targets(h[:3],h[3:],cfg)[0]
    cfg['training']['dual_credit']='observed_suffix'
    corrected=dual_targets(h[:3],h[3:],cfg)[0]
    assert not np.allclose(corrected,legacy)


def test_intervention_consistency_and_snapshot(cfg):
    from meta_exploration.diagnostics import intervene,history_interventions
    game=Game(generate_world(2,'type3'),11,0)
    h=collect(game,Controller(make_model(cfg),2),8)
    original=copy.deepcopy(h)
    for kind in ['real','reward_shuffled','action_relabelled']:
        modified=intervene(h,kind,0)
        assert modified[-1].next_observation==game.observation()
        for a,b in zip(modified,modified[1:]):assert a.next_observation==b.observation
        assert [t.reward for t in h]==[t.reward for t in original]
    cfg['training'].update(exploration_steps=8,exploitation_steps=8)
    models={k:make_model(cfg) for k in ['single_policy','explorer','dual_neural']}
    before={k:copy.deepcopy(m.state_dict()) for k,m in models.items()}
    a=history_interventions(cfg,models,generate_world(2,'type3'),0)
    b=history_interventions(cfg,models,generate_world(2,'type3'),0)
    assert [r['action_sequence'] for r in a]==[r['action_sequence'] for r in b]
    assert all(r['post_B_observation']==a[0]['post_B_observation'] for r in a)
    assert all(torch.equal(v,before[k][key]) for k,m in models.items() for key,v in m.state_dict().items())


def test_batch_resume_and_optimizer_budget(cfg,tmp_path,monkeypatch):
    import meta_exploration.training as training
    cfg['training'].update(episodes=5,pretraining_episodes=4)
    training.train(cfg,'B',0,tmp_path/'full')
    original=training.atomic_save
    def interrupt(data,path):
        original(data,path)
        if str(path).endswith('training_resume.pt') and data['episode']==2:
            raise RuntimeError('interrupted batch')
    monkeypatch.setattr(training,'atomic_save',interrupt)
    with pytest.raises(RuntimeError,match='interrupted batch'):training.train(cfg,'B',0,tmp_path/'resume')
    monkeypatch.setattr(training,'atomic_save',original)
    training.train(cfg,'B',0,tmp_path/'resume')
    full=torch.load(tmp_path/'full/checkpoints/trained.pt',weights_only=False)['models']
    resumed=torch.load(tmp_path/'resume/checkpoints/trained.pt',weights_only=False)['models']
    for k in full:
        assert all(torch.equal(v,resumed[k][key]) for key,v in full[k].items())
    import json
    cost=json.loads((tmp_path/'full/costs.json').read_text())
    assert cost['single_optimizer_steps']==3*cfg['training']['ppo_epochs']
    updates=[json.loads(x) for x in (tmp_path/'full/training.jsonl').read_text().splitlines() if 'rollout_episode_ids' in x]
    assert any(r['rollout_episode_ids']==[4] for r in updates)


def test_speed_rmst_and_paired_all_censored(tmp_path):
    import json
    from meta_exploration.adaptation import aggregate_speed
    rows=[]
    for seed in [0,1]:
        for method in ['single_policy','dual_neural']:
            rows.append(dict(regime='A1',evaluation='frozen',world_type='type1',B=2,K=4,method=method,seed=seed,world_id=str(seed),
                adaptation_speed=speed_metrics([0]*4,[1]*4,[0]*4,dict(window=1,hold=2))))
    a=aggregate_speed(rows,tmp_path,10)
    assert all(r['mean']==4 for r in a if r['metric']=='rmst_0.9')
    assert all(r['mean']==0 for r in a if r['metric']=='reach_at_K_0.9')
    paired=json.loads((tmp_path/'paired_speed.json').read_text())
    assert all(r['mean']==0 for r in paired)
    with pytest.raises(ValueError,match='Duplicate'):
        aggregate_speed(rows+rows,tmp_path,10)


def test_backfill_uses_raw_not_aggregate(cfg,tmp_path):
    import json
    from meta_exploration.training import train
    from meta_exploration.evaluation import evaluate
    from meta_exploration.adaptation import backfill_raw
    source=tmp_path/'source'
    train(cfg,'B',0,source)
    original=evaluate(cfg,'B',0,source)
    recovered=backfill_raw(source,tmp_path/'backfilled')
    assert len(recovered)==len(original)
    for a,b in zip(original,recovered):
        assert a['adaptation_speed']==b['adaptation_speed']
    # Metric files alone cannot recover paths/crossing times.
    for p in (source/'evaluation').glob('*/*.jsonl'):p.unlink()
    missing=backfill_raw(source,tmp_path/'missing')
    assert missing==[]
    audit=json.loads((tmp_path/'missing/backfill_audit.json').read_text())
    assert audit['n_missing']==len(original)


def test_budget_evaluation_isolated_copy(cfg,tmp_path):
    import json,hashlib
    from meta_exploration.training import train,write_json
    from meta_exploration.diagnostics import evaluate_budgets
    source=tmp_path/'source/seed_0'
    train(cfg,'B',0,source)
    write_json(source/'identity.json',dict(regime='B',seed=0))
    before=hashlib.sha256((source/'checkpoints/trained.pt').read_bytes()).hexdigest()
    evaluate_budgets(source.parent,tmp_path/'budget',budgets=(1,3),K=5)
    assert hashlib.sha256((source/'checkpoints/trained.pt').read_bytes()).hexdigest()==before
    rows=json.loads((tmp_path/'budget/seed_0/evaluation/metrics.json').read_text())
    assert {r['B'] for r in rows}=={1,3}
    assert all(r['K']==5 for r in rows)
    assert all('adaptation_compute' in r for r in rows)
    assert all(r['adaptation_compute']['td_updates']==r['B']+r['K'] for r in rows if r['method']=='q_learning')
    assert all(r['adaptation_compute']['td_updates']==r['B']*cfg['q_learning']['replay_passes']+r['K'] for r in rows if r['method']=='dual_q_table')
