"""Replay held-out repetition 0 and measure reuse after first combo discovery."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from marl import evaluation
from marl.environment import Game
from marl.generator import load_tasks
from marl.cli import load_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/pilot_seed0'
OUT = RUN / 'combo_diagnostic'

class TracedGame(Game):
    latest = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = []
        TracedGame.latest = self
    def step(self, a, b):
        self.actions.append(a)
        return super().step(a, b)

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads((RUN / 'config.json').read_text())
    tasks = load_tasks(RUN / 'evaluation_tasks.npz')
    opponent = load_model(RUN, 'opponent', config)
    single = load_model(RUN, 'single', config)
    duals = {k: load_model(RUN, f'dual_k{k}', config, k) for k in config['budgets']}
    original = {}
    for line in (RUN / 'evaluation/episodes.jsonl').open():
        r = json.loads(line)
        if r['repetition'] == 0:
            original[r['task_index'],r['opponent'],r['k'],r['agent']] = r
    evaluation.Game = TracedGame
    OUT.mkdir(exist_ok=True)
    rows = []
    stream = (OUT/'episodes.jsonl').open('w')
    for ti, task in enumerate(tasks):
        for oi, op in enumerate(('random','fixed','ppo')):
            for k, dual in duals.items():
                for kind in ('dual_learned','dual_random','dual_target','single_gru','q_learning'):
                    metrics, _ = evaluation.evaluate_rollout(task, single if kind == 'single_gru' else dual,
                        kind, op, opponent, config['horizon'], k,
                        [config['seed'],400,ti,oi,0], config['q_learning'])
                    old = original[ti,op,k,kind]
                    assert abs(metrics['total_reward']-old['total_reward']) < 1e-6
                    assert metrics['first_combo_round'] == old['first_combo_round']
                    actions = TracedGame.latest.actions
                    u,v = task.combos[0]
                    hits = [i for i in range(1,len(actions),2) if actions[i-1] == u and actions[i] == v]
                    first = hits[0] if hits else None
                    # Only complete pairs whose R1 starts in exploitation and after discovery.
                    eligible = [i for i in range(1,len(actions),2) if first is not None and i-1 > first and i-1 >= k]
                    prefixes = sum(actions[i-1] == u for i in eligible)
                    reuse = sum(actions[i-1] == u and actions[i] == v for i in eligible)
                    row = dict(task=ti,opponent=op,k=k,agent=kind,first_combo_round=None if first is None else first+1,
                        opportunities=len(eligible),prefixes=prefixes,reuse=reuse,
                        next_pair_hit=None if first is None or first+2 >= len(actions) else first+2 in hits,
                        total_hits=len(hits),exploitation_mean=metrics['exploitation_mean'])
                    rows.append(row)
                    stream.write(json.dumps(row, default=lambda x: x.item())+'\n')
        if (ti+1)%16 == 0:
            stream.flush()
            print(f'{ti+1}/{len(tasks)} tasks complete',flush=True)
    stream.close()
    summary=[]
    for k in config['budgets']:
        for kind in ('dual_learned','dual_random','dual_target','single_gru','q_learning'):
            group=[r for r in rows if r['k']==k and r['agent']==kind]
            eligible=[r for r in group if r['opportunities']]
            opportunities=sum(r['opportunities'] for r in eligible)
            prefixes=sum(r['prefixes'] for r in eligible)
            reuse=sum(r['reuse'] for r in eligible)
            summary.append(dict(k=k,agent=kind,episodes=len(group),discovered=sum(r['first_combo_round'] is not None for r in group),
                eligible_episodes=len(eligible),opportunities=opportunities,
                reuse_rate=reuse/opportunities if opportunities else None,
                prefix_rate=prefixes/opportunities if opportunities else None,
                completion_given_prefix=reuse/prefixes if prefixes else None,
                any_reuse_rate=float(np.mean([r['reuse']>0 for r in eligible])),
                next_pair_rate=float(np.mean([r['next_pair_hit'] for r in group if r['next_pair_hit'] is not None]))))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x: x.item())+'\n')
    print(json.dumps(summary,indent=2,default=lambda x: x.item()))

if __name__ == '__main__':
    main()
