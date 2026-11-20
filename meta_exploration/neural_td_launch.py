"""Explicit dynamic independent-job launcher; no automatic full-study invocation."""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from .hardware import hardware_report
from .training import write_json


def launch(config,out,regimes,seeds,existing_root=None,cpu_workers=1,require_gpu=False):
    if len(set(regimes))!=len(regimes) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate jobs would collide')
    root=Path(out)
    if root.exists():raise ValueError('Launcher requires a new study root; resume individual runs explicitly')
    report=hardware_report()
    devices=[f'cuda:{i}' for i in report['usable_devices']]
    if require_gpu and not devices:raise RuntimeError('No GPU passed arithmetic probe')
    devices=devices or ['cpu']*cpu_workers
    if not devices:raise ValueError('At least one worker required')
    root.mkdir(parents=True);write_json(root/'hardware.json',report)
    write_json(root/'launch.json',dict(config=str(Path(config).resolve()),regimes=regimes,seeds=seeds,devices=devices))
    jobs=queue.Queue()
    for regime in regimes:
        for seed in seeds:jobs.put((regime,seed))
    results=[];lock=threading.Lock()
    def worker(device):
        while True:
            try:regime,seed=jobs.get_nowait()
            except queue.Empty:return
            target=root/regime/f'seed_{seed}';target.parent.mkdir(exist_ok=True)
            log=root/f'{regime}_seed_{seed}.log'
            cmd=[sys.executable,'-m','meta_exploration.neural_td_experiment','run','--config',config,
                 '--regime',regime,'--seed',str(seed),'--out',str(target),'--device',device]
            if existing_root:cmd+=['--existing-root',existing_root]
            before=time.monotonic()
            with log.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            with lock:
                results.append(dict(regime=regime,seed=seed,device=device,exit_code=result.returncode,seconds=time.monotonic()-before,log=str(log)))
                write_json(root/'jobs.json',results)
            jobs.task_done()
    threads=[threading.Thread(target=worker,args=(d,)) for d in devices]
    for t in threads:t.start()
    for t in threads:t.join()
    if len(results)!=len(regimes)*len(seeds) or any(r['exit_code'] for r in results):raise RuntimeError('Worker failures; see jobs.json (no output overwritten)')
    write_json(root/'complete.json',dict(jobs=len(results)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/meta/neural_td_pilot.json');p.add_argument('--out',required=True)
    p.add_argument('--regimes',nargs='+',choices=['A1','A2','A3','B'],default=['A2','A3'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--existing-root')
    p.add_argument('--cpu-workers',type=int,default=1);p.add_argument('--require-gpu',action='store_true')
    a=p.parse_args();launch(a.config,a.out,a.regimes,a.seeds,a.existing_root,a.cpu_workers,a.require_gpu)

if __name__=='__main__':main()
