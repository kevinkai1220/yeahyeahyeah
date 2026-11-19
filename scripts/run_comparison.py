"""Run 2 learning algorithms x 3 exploration-return weights, with shared tasks/baselines.
Usage: python -m scripts.run_comparison [--smoke] [--output PATH]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from marl.cli import load_model, validate
from marl.generator import load_tasks
from marl.training import train
from marl.study_evaluation import evaluate_study

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/pilot_seed0'


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def prepare(output,smoke):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    source_config=json.loads((SOURCE/'config.json').read_text())
    manifest=dict(source=str(SOURCE),started_unix=time.time(),smoke=smoke,
        protocol='2 algorithms x 3 lambda; full-history replay; uniform FIFO capacity=1024; terminal gamma=1',
        baseline_reuse='Frozen opponent and single model plus existing single/Q evaluation; no baseline retraining',
        collection='Batched independent episode RNGs, separate acquisition/update/replay streams; PPO lambda=0 retrained',
        compute='PPO:16 joint gradient steps/update; V-trace:16 exploration PPO + 16 target replay steps/update',
        source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('training_tasks.npz','evaluation_tasks.npz','opponent.pt','single.pt')},
        code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'marl').glob('*.py'))+[Path(__file__).resolve()]})
    write_json(output/'manifest.json',manifest)
    cases=[]
    for algorithm in ('ppo','vtrace'):
        for weight in (0.,.5,1.):
            name=f'{algorithm}_lambda{str(weight).replace(".","p")}'
            case=output/name;case.mkdir()
            config=dict(source_config,exploration_lambda=weight,exploit_learning=algorithm,
                batched_collection=True,replay_capacity=1024,replay_updates=16)
            if smoke:
                config.update(updates=2,tasks_per_update=2,rollouts_per_task=2,evaluation_rollouts=1,
                    checkpoint_every=1,replay_updates=2)
                config['ppo']=dict(config['ppo'],epochs=1,minibatch=2)
            validate(config)
            write_json(case/'config.json',config)
            for filename in ('training_tasks.npz','evaluation_tasks.npz','opponent.pt','single.pt'):
                shutil.copy2(SOURCE/filename,case/filename)
            cases.append(case)
    return cases


def worker(case,stage,k,smoke):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=json.loads((case/'config.json').read_text())
    pretrained=load_model(case,'opponent',config)
    if stage=='train':
        tasks=load_tasks(case/'training_tasks.npz')
        train(tasks,config,case,f'dual_k{k}',dual=True,k=k,pretrained=pretrained)
    else:
        tasks=load_tasks(case/'evaluation_tasks.npz')
        if smoke: tasks=tasks[:2]
        duals={budget:load_model(case,f'dual_k{budget}',config,budget) for budget in config['budgets']}
        baseline=None if smoke else SOURCE/'evaluation/episodes.jsonl'
        evaluate_study(tasks,duals,pretrained,config,case/'evaluation',baseline)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/lambda_offpolicy_seed0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker',choices=['train','evaluate'])
    parser.add_argument('--k',type=int)
    args=parser.parse_args()
    if args.worker:
        worker(args.output,args.worker,args.k,args.smoke);return
    cases=prepare(args.output,args.smoke)
    started=time.time();completed=[]
    status_path=args.output/'status.json'
    write_json(status_path,dict(state='running',pid=os.getpid(),started_unix=started,completed=[]))
    def job(case,stage,k=None):
        command=[sys.executable,'-u','-m','scripts.run_comparison','--worker',stage,'--output',str(case)]
        if k is not None:command+=['--k',str(k)]
        if args.smoke:command+=['--smoke']
        label=f'{case.name}/{stage}'+('' if k is None else f'_k{k}')
        logpath=case/(f'{stage}'+('' if k is None else f'_k{k}')+'.process.log')
        with logpath.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{label} failed; see {logpath}')
        return label
    try:
        # Two CPU workers, each Torch single-threaded. No distributed agents involved.
        with ThreadPoolExecutor(max_workers=2) as executor:
            for stage in ('train','evaluate'):
                jobs=[executor.submit(job,case,stage,k) for case in cases for k in ((10,25,50) if stage=='train' else (None,))]
                for future in as_completed(jobs):
                    label=future.result();completed.append(label)
                    print(label+' complete',flush=True)
                    write_json(status_path,dict(state='running',pid=os.getpid(),started_unix=started,elapsed_seconds=time.time()-started,completed=completed))
        if not args.smoke:
            subprocess.run([sys.executable, '-m', 'scripts.summarize_comparison', str(args.output)], cwd=ROOT, check=True)
        write_json(status_path,dict(state='completed',started_unix=started,elapsed_seconds=time.time()-started,completed=completed))
    except Exception as error:
        write_json(status_path,dict(state='failed',error=str(error),completed=completed,elapsed_seconds=time.time()-started))
        raise

if __name__=='__main__': main()
