"""Tracked training with fixed monitoring tasks and preserved checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from marl.cli import validate,load_model
from marl.generator import load_tasks
from marl.training import train
from marl.study_evaluation import evaluate_study

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'runs/pilot_seed0')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithms',nargs='+',choices=['ppo','vtrace'],default=['vtrace'])
    p.add_argument('--lambdas',nargs='+',type=float,default=[0.,.5,1.])
    p.add_argument('--budgets',nargs='+',type=int,default=[50])
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--track-every',type=int,default=5)
    p.add_argument('--diagnostic-tasks',type=int,default=16)
    p.add_argument('--diagnostic-rollouts',type=int,default=2)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if any(n<1 for n in (args.updates,args.track_every,args.diagnostic_tasks,args.diagnostic_rollouts)):
        p.error('Counts and intervals must be positive')
    if len(set(args.lambdas))!=len(args.lambdas) or any(not 0<=x<=1 for x in args.lambdas):p.error('Unique lambdas in [0,1] required')
    if len(set(args.algorithms))!=len(args.algorithms):p.error('Algorithms must be unique')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    config=json.loads((args.source/'config.json').read_text())
    config.update(budgets=args.budgets,updates=args.updates,batched_collection=True,replay_capacity=1024,replay_updates=16,
        tracking=dict(enabled=True,every=args.track_every,tasks=args.diagnostic_tasks,rollouts=args.diagnostic_rollouts))
    if args.smoke:
        config.update(updates=2,tasks_per_update=2,rollouts_per_task=2,replay_updates=2,evaluation_rollouts=1)
        config['ppo']=dict(config['ppo'],epochs=1,minibatch=2)
        config['tracking']=dict(enabled=True,every=1,tasks=2,rollouts=1)
    validate(config)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=dict(source=str(args.source.resolve()),arguments=vars(args)|{'source':str(args.source),'output':str(args.output)},
        note='Training seed and task splits inherited from source. Fixed independent diagnostic tasks and common rollout seeds at update 0 and every tracking interval. Final evaluation never used for updates. Replay has additional optimization compute.',
        code_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list((ROOT/'marl').glob('*.py'))+[Path(__file__).resolve()]})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.time();completed=[]
    def status(state,**extra):
        dest=args.output/'status.json';temp=dest.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state,elapsed_seconds=time.time()-started,completed=completed,**extra),indent=2)+'\n');temp.replace(dest)
    status('running')
    try:
        for algorithm in args.algorithms:
            for weight in args.lambdas:
                case=args.output/f'{algorithm}_lambda{str(float(weight)).replace(".","p")}'
                case.mkdir();cfg=dict(config,exploit_learning=algorithm,exploration_lambda=weight)
                (case/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
                for filename in ('training_tasks.npz','evaluation_tasks.npz','opponent.pt','single.pt'):
                    shutil.copy2(args.source/filename,case/filename)
                tasks=load_tasks(case/'training_tasks.npz');opponent=load_model(case,'opponent',cfg)
                for k in cfg['budgets']:
                    status('running',active=f'{case.name}/train_k{k}')
                    train(tasks,cfg,case,f'dual_k{k}',dual=True,k=k,pretrained=opponent)
                eval_tasks=load_tasks(case/'evaluation_tasks.npz')
                if args.smoke:eval_tasks=eval_tasks[:2]
                duals={k:load_model(case,f'dual_k{k}',cfg,k) for k in cfg['budgets']}
                status('running',active=f'{case.name}/evaluate')
                evaluate_study(eval_tasks,duals,opponent,cfg,case/'evaluation',
                    baseline_path=None if args.smoke else args.source/'evaluation/episodes.jsonl')
                completed.append(case.name)
                subprocess.run([sys.executable,'-m','scripts.summarize_tracking',str(args.output)],cwd=ROOT,check=True)
        status('completed')
    except Exception as error:
        status('failed',error=str(error));raise

if __name__=='__main__':main()
