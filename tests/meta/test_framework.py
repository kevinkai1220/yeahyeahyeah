import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from meta_exploration.config import load_config
from meta_exploration.environment import World, Game, Observation, generate_world, structural_validation, oracle_values
from meta_exploration.learning import (QLearner, Transition, Controller, collect, gae, targets,
                                      make_model, clone_model, assert_equal, ppo_update, adapt)
from meta_exploration.training import train, world_set, HistoryCollector, observed_information
from meta_exploration.evaluation import evaluate_world, evaluate
from meta_exploration.hardware import job_path


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return load_config('configs/meta/smoke.json')


@pytest.mark.parametrize('typ', ['type1', 'type2', 'type3'])
@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_generation_dependency_and_trap(typ, difficulty):
    for seed in range(10):
        w = generate_world(seed, typ, difficulty=difficulty)
        assert structural_validation(w)
        assert w.fingerprint == generate_world(seed, typ, difficulty=difficulty).fingerprint
        for a in range(3):
            for c in range(3):
                if typ == 'type1' or (typ == 'type2' and a != c) or (typ == 'type3' and a == c):
                    assert np.all(w.bonus[a, :, c, :] == 0)
        for a in range(3):
            for b in range(3):
                assert w.mean(0, (3, 3), a, b) == w.base[a, b]
                if typ == 'type1':
                    assert w.mean(1, (0, 2), a, b) == w.base[a, b]


def test_reward_noise_independent_reproducible_and_fixed_mean():
    w = generate_world(4, 'type3', sigma=1)
    deterministic = World(w.world_id, w.world_type, w.base, w.bonus, 0., w.split, w.seed, w.difficulty)
    a, b = Game(w, 5000, 4, 'fixed'), Game(deterministic, 5000, 4, 'fixed')
    fingerprint = w.fingerprint
    noises = []
    for i in range(5000):
        noises.append(a.step(i % 3)[1] - b.step(i % 3)[1])
    assert abs(np.mean(noises)) < .05
    assert .95 < np.std(noises) < 1.05
    assert w.fingerprint == fingerprint
    a, b = Game(w, 10, 8), Game(w, 10, 8)
    assert [a.step(i % 3)[1] for i in range(10)] == [b.step(i % 3)[1] for i in range(10)]


def test_mode_reset_not_terminal_and_no_leakage():
    w = generate_world(8, 'type2')
    game = Game(w, 3, 0)
    assert game.observation().previous_own == 3
    _, _, done, _ = game.step(1)
    assert game.mode == 1 and not done
    observation, _, done, _ = game.step(2)
    assert game.mode == 0 and not done and observation.previous_own == 2
    assert set(vars(observation)) == {'mode','previous_own','previous_other','reward'}
    assert observation.vector().shape == (11,)
    assert game.step(0)[2]
    with pytest.raises(ValueError):
        game.step(0)


def test_snapshot_odd_boundary():
    w = generate_world(2, 'type3')
    game = Game(w, 8, 5)
    game.step(0)
    fork = game.snapshot()
    assert fork.mode == 1
    for a in [1, 2, 0]:
        assert game.step(a) == fork.step(a)
    fork.step(0)
    assert game.t == 4 and fork.t == 5


def test_gae_bootstrap_and_downstream_credit(config):
    a, ret = gae([0, 0, 10], [0, 0, 0], [False, False, True], 999, 1., 1.)
    np.testing.assert_allclose(ret, [10, 10, 10])
    a, ret = gae([1, 2], [0, 0], [False, False], 7, 1., 1.)
    np.testing.assert_allclose(ret, [10, 9])
    config['training']['exploration_steps'] = 2
    config['training']['exploration_lambda'] = .1
    config['training']['gae_lambda'] = 1.
    o = Observation(0, 3, 3, 0)
    history = [Transition(o, 0, r, o, i==2, 0, {'value': 0}) for i, r in enumerate([10,20,30])]
    _, r = targets(history, config)
    np.testing.assert_allclose(r, [33, 32, 30])


def test_q_reverse_td_delayed_credit(config):
    qc = dict(config['q_learning'], alpha=1., gamma=1., replay_passes=1)
    q = QLearner(qc, 0)
    o0, o1, o2 = Observation(0,3,3,0), Observation(1,0,1,-2), Observation(0,1,0,10)
    t0 = Transition(o0,0,-2,o1,False,1,{})
    t1 = Transition(o1,1,10,o2,False,0,{})
    q.q[o2.q_state()] = 3
    q.replay([t0,t1])
    assert q.q[o1.q_state()+(1,)] == 13  # R2 -> R1 retains bootstrap
    assert q.q[o0.q_state()+(0,)] == 11
    q2 = QLearner(qc, 0)
    assert np.all(q2.q == 0)


def test_shared_backbone_and_versions(config):
    torch.manual_seed(0)
    model = make_model(config); other = clone_model(model)
    assert_equal(model, other)
    controller = Controller(model, 0)
    game = Game(generate_world(2,'type2'), 8, 1)
    history = collect(game, controller, 8)
    adv, ret = targets(history, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    ppo_update(controller, optimizer, history, adv, ret, config)
    assert any(not torch.equal(v, other.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError, match='Off-policy'):
        ppo_update(controller, optimizer, history, adv, ret, config)


def test_coverage_only_observations():
    actor = HistoryCollector(5, 'coverage')
    game = Game(generate_world(1, 'type3'), 18, 2)
    history = collect(game, actor, 18)
    assert observed_information(history)[1]['pair_coverage'] == 9
    assert not hasattr(actor, 'world')


def test_split_and_output_uniqueness(config):
    for regime in ['A1','A2','A3','B']:
        train_worlds = world_set(config, regime, 0, 'train')
        test_worlds = world_set(config, regime, 0, 'test')
        assert not {w.fingerprint for w in train_worlds} & {w.fingerprint for w in test_worlds}
    paths = {job_path('/tmp/test', r, s) for r in ['A1','A2','A3','B'] for s in range(5)}
    assert len(paths) == 20


def test_oracle_matches_bruteforce_fixed():
    import itertools
    w = generate_world(3, 'type3', sigma=0)
    vals = oracle_values(w, 4, 'fixed')
    returns = []
    for actions in itertools.product(range(3), repeat=4):
        g = Game(w, 4, 0, 'fixed')
        returns.append(sum(g.step(a)[1] for a in actions))
    assert vals[0,3,3] == pytest.approx(max(returns))


def test_frozen_online_world_independence_and_pairing(config):
    torch.manual_seed(3)
    models = {k: make_model(config) for k in ['single_policy','dual_neural','explorer']}
    before = {k: clone_model(m) for k,m in models.items()}
    w = generate_world(20,'type3',split='test')
    a, _ = evaluate_world(config, models, w, 4, 'online')
    evaluate_world(config, models, generate_world(21,'type2',split='test'), 5, 'online')
    b, _ = evaluate_world(config, models, w, 4, 'online')
    for key in models:
        assert_equal(models[key], before[key])
    for method in a:
        assert [t.reward for t in a[method]] == [t.reward for t in b[method]]
    frozen, summary = evaluate_world(config, models, w, 4, 'frozen')
    B = config['training']['exploration_steps']
    for t, u in zip(frozen['dual_neural'][:B], frozen['dual_q_table'][:B]):
        assert t is u
    assert frozen['dual_neural'][B].observation == frozen['dual_q_table'][B].observation
    assert all(r['online_optimizer_steps']==0 for r in summary.values())


def test_end_to_end_checkpoint_resume_and_logs(config, tmp_path):
    train(config, 'B', 0, tmp_path)
    initial = torch.load(tmp_path/'checkpoints/identical_initialization.pt', weights_only=False)
    assert all(torch.equal(v, initial['dual_neural'][k]) for k,v in initial['single_policy'].items())
    trained = torch.load(tmp_path/'checkpoints/trained.pt', weights_only=False)
    train(config, 'B', 0, tmp_path)
    resumed = torch.load(tmp_path/'checkpoints/trained.pt', weights_only=False)
    for model in trained['models']:
        assert all(torch.equal(v, resumed['models'][model][k]) for k,v in trained['models'][model].items())
    rows = evaluate(config, 'B', 0, tmp_path)
    assert len(rows) == 24
    paths = list((tmp_path/'evaluation').glob('*/*.jsonl'))
    assert len(paths) == 24
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            assert 'true_mean' not in record and 'true_bonus' not in record and 'world_type' not in record
    strata = {json.loads(line)['history_stratum'] for line in (tmp_path/'training.jsonl').read_text().splitlines()
              if json.loads(line)['stage']=='pretraining'}
    assert strata == {'informative','uninformative','ambiguous'}


def test_partial_training_resume_matches_uninterrupted(config, tmp_path, monkeypatch):
    import meta_exploration.training as module
    train(config, 'B', 7, tmp_path/'full')
    original = module.atomic_save
    stopped = False
    def interrupt(data, path):
        nonlocal stopped
        original(data, path)
        if Path(path).name == 'training_resume.pt' and data['episode'] == 1 and not stopped:
            stopped = True
            raise RuntimeError('simulated interruption after checkpoint')
    monkeypatch.setattr(module, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(config, 'B', 7, tmp_path/'interrupted')
    monkeypatch.setattr(module, 'atomic_save', original)
    train(config, 'B', 7, tmp_path/'interrupted')
    a = torch.load(tmp_path/'full/checkpoints/trained.pt', weights_only=False)['models']
    b = torch.load(tmp_path/'interrupted/checkpoints/trained.pt', weights_only=False)['models']
    for model in a:
        assert all(torch.equal(v, b[model][k]) for k,v in a[model].items())


def test_launcher_dynamic_jobs_retry_and_unique_outputs(config, tmp_path, monkeypatch):
    import threading
    import types
    import meta_exploration.hardware as module
    monkeypatch.setattr(module, 'hardware_report', lambda: {'usable_devices':[0,1]})
    calls, attempts = [], {}
    lock = threading.Lock()
    def fake_run(cmd, **kwargs):
        if cmd[0] == 'nvidia-smi':
            return types.SimpleNamespace(returncode=0, stdout='0, 20, 100', stderr='')
        out = cmd[cmd.index('--out')+1]
        with lock:
            attempts[out] = attempts.get(out,0)+1
            calls.append((out,cmd[cmd.index('--device')+1]))
            fail = out.endswith('A1/seed_0') and attempts[out] == 1
        return types.SimpleNamespace(returncode=1 if fail else 0)
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    module.launch(config, 'configs/meta/smoke.json', tmp_path, ['A1','B'], [0,1])
    assert len({p for p,d in calls}) == 4
    assert len(calls) == 5
    assert all(d in ('cuda:0','cuda:1') for p,d in calls)
    assert json.loads((tmp_path/'failures.json').read_text()) == []


def test_output_lock_rejects_competing_worker(tmp_path, monkeypatch):
    import fcntl
    import sys
    from meta_exploration.__main__ import main
    with (tmp_path/'.job.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(sys, 'argv', ['meta', 'run', '--config', 'configs/meta/smoke.json', '--out', str(tmp_path)])
        with pytest.raises(RuntimeError, match='Another worker owns'):
            main()


def test_recurrent_opponent_snapshot_and_action_space(config):
    opponent = make_model(config)
    game = Game(generate_world(0,'type2'), 8, 2, 'ppo', 3, opponent)
    game.step(0)
    clone = game.snapshot()
    assert clone.opponent_model is game.opponent_model
    assert torch.equal(clone.opponent_hidden, game.opponent_hidden)
    for a in [2,1,0]:
        assert clone.step(a) == game.step(a)


def test_optional_q_aware_and_privileged_flags(config, tmp_path):
    config['training']['q_aware'] = True
    config['training']['privileged_curriculum'] = True
    config['training']['single_k_only'] = True
    config['training']['gamma'] = .99
    train(config, 'B', 1, tmp_path)
    rows = evaluate(config, 'B', 1, tmp_path)
    assert {r['method'] for r in rows} == {'single_policy','q_learning','dual_neural','dual_q_table','dual_q_aware'}


def test_hierarchical_bootstrap_and_plots(config, tmp_path):
    from meta_exploration.plotting import bootstrap, plot_results
    mean, lo, hi = bootstrap({0:[1.,2.],1:[3.,4.]}, 100)
    assert mean == 2.5 and lo <= mean <= hi
    train(config, 'B', 2, tmp_path)
    evaluate(config, 'B', 2, tmp_path)
    summary = plot_results(tmp_path)
    assert summary
    assert (tmp_path/'figures/aggregated.csv').exists()
    for stem in ['history_type1','training_progress','budget_discovery_online_K5','auc_online_B3_K5',
                 'generalization_online_B3_K5','adaptation_B_online_3_5_all']:
        assert (tmp_path/f'figures/{stem}.png').stat().st_size > 1000
        assert (tmp_path/f'figures/{stem}.pdf').stat().st_size > 1000


def test_informative_stratum_guarantees_pair_coverage(config, tmp_path):
    config['training']['exploration_steps'] = 18
    config['training']['pretraining_episodes'] = 6
    train(config, 'B', 0, tmp_path, pretrain_only=True)
    records = [json.loads(x) for x in (tmp_path/'training.jsonl').read_text().splitlines()]
    informative = [r for r in records if r.get('history_stratum')=='informative']
    assert informative and all(r['history_stats']['pair_coverage']==9 for r in informative)


@pytest.mark.parametrize('typ',['type2','type3'])
def test_exact_sequence_bonus_and_no_bonus_after_reset(typ):
    world = generate_world(4, typ, sigma=0)
    pair = np.unravel_index(world.bonus.mean(axis=(1,3)).argmax(), (3,3))
    game = Game(world, 4, 0, 'fixed')
    _, r0, _, _ = game.step(int(pair[0]))
    _, r1, done, _ = game.step(int(pair[1]))
    _, r2, _, _ = game.step(int(pair[1]))
    assert r0 == world.base[pair[0],0]
    assert r1 == world.base[pair[1],1] + world.bonus[pair[0],0,pair[1],1]
    assert r2 == world.base[pair[1],2] and not done
    assert world.fingerprint == generate_world(4, typ, sigma=9).fingerprint
