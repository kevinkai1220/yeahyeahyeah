import copy
import json
from pathlib import Path

DEFAULT = {
    'environment': {'n_actions': 3, 'reward_sigma': 1., 'difficulty': 'medium',
                    'opponent_dependence': 0., 'opponents': ['random', 'fixed'],
                    'opponent_checkpoint': None},
    'training': {'train_worlds': 60, 'pretraining_episodes': 300, 'episodes': 1200,
                 'hidden': 64, 'exploration_steps': 32, 'exploitation_steps': 128,
                 'exploration_lambda': .1, 'gamma': 1., 'gae_lambda': .95,
                 'learning_rate': .0003, 'ppo_epochs': 4, 'clip': .2,
                 'value_coefficient': .5, 'entropy_coefficient': .01, 'max_grad_norm': .5,
                 'history_pool_size': 6, 'history_sampling_ratios': [.4, .3, .3],
                 'privileged_curriculum': False, 'single_k_only': False, 'q_aware': False, 'rollout_batch_size': 1,
                 'advantage_normalization': 'legacy_episode', 'dual_credit': 'legacy_cross_critic'},
    'q_learning': {'alpha': .2, 'gamma': .99, 'epsilon_start': .3, 'epsilon_end': .05,
                   'epsilon_decay_steps': 160, 'replay_passes': 10},
    'evaluation': {'num_test_worlds': 100, 'seeds': [0, 1, 2, 3, 4],
                   'modes': ['frozen', 'online'], 'update_interval': 32,
                   'auc_windows': [32, 64, 128], 'exploration_budgets': [8, 16, 32, 64],
                   'exploitation_budgets': [128], 'history_reference': True,
                   'bootstrap_replicates': 1000},
    'hardware': {'device': 'auto', 'use_all_available_gpus': True, 'cpu_workers': 1,
                 'threads_per_worker': 1, 'max_retries': 1},
    'logging': {'save_trajectories': True, 'save_plots': True, 'checkpoint_interval': 20},
}
REGIMES = {'A1': ['type1'], 'A2': ['type2'], 'A3': ['type3'], 'B': ['type1', 'type2', 'type3']}


def merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    c = copy.deepcopy(DEFAULT)
    if path:
        merge(c, json.loads(Path(path).read_text()))
    t, e = c['training'], c['evaluation']
    assert c['environment']['n_actions'] == 3
    assert t['exploration_steps'] > 0 and t['exploitation_steps'] > 0
    assert 0 <= t['gamma'] <= 1 and 0 <= t['gae_lambda'] <= 1
    assert 0 <= t['exploration_lambda'] <= 1
    assert t.get('sequence_minibatch_size', 0) >= 0
    assert t.get('critic_loss_scale', 1.) > 0
    assert t.get('gradient_clipping', 'global') in ('global', 'parameter_groups')
    assert t['rollout_batch_size'] >= 1
    assert t['advantage_normalization'] in ('batch', 'legacy_episode', 'off')
    assert t['dual_credit'] in ('observed_suffix', 'legacy_cross_critic')
    assert t['train_worlds'] >= 1 and t['pretraining_episodes'] >= 3
    assert t['history_pool_size'] >= 3 and len(t['history_sampling_ratios']) == 3
    assert all(x >= 0 for x in t['history_sampling_ratios']) and sum(t['history_sampling_ratios']) > 0
    assert e['update_interval'] > 0 and e['num_test_worlds'] > 0
    assert 0 <= c['q_learning']['gamma'] <= 1 and 0 < c['q_learning']['alpha'] <= 1
    assert set(e['modes']) <= {'online', 'frozen'}
    return c
