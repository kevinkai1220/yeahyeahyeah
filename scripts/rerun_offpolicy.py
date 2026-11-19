"""Correct inactive-head Adam momentum, reuse unaffected PPO results, rerun V-trace."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .run_comparison import ROOT,prepare,write_json


def main():
    output=ROOT/'runs/lambda_offpolicy_seed0_corrected'
    previous=ROOT/'runs/lambda_offpolicy_seed0'
    cases=prepare(output,False)
    for case in cases:
        if case.name.startswith('ppo_'):
            shutil.copytree(previous/case.name,case,dirs_exist_ok=True)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest.update(correction='Set inactive actor/critic grads=None before Adam step to prevent momentum-only updates. Shared GRU remains trainable in both phases.',
        reused_ppo_from=str(previous),supersedes_offpolicy_from=str(previous))
    write_json(output/'manifest.json',manifest)
    dest=output/'source_snapshot'
    for dirname in ('marl','scripts','tests'):
        for p in (ROOT/dirname).glob('*.py'):
            target=dest/dirname/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    started=time.time();completed=[f'{case.name}/reused_ppo' for case in cases if case.name.startswith('ppo_')]
    status=output/'status.json'
    write_json(status,dict(state='running',started_unix=started,completed=completed))
    def job(case,stage,k=None):
        cmd=[sys.executable,'-u','-m','scripts.run_comparison','--worker',stage,'--output',str(case)]
        if k is not None:cmd+=['--k',str(k)]
        label=f'{case.name}/{stage}'+('' if k is None else f'_k{k}')
        with (case/(stage+('' if k is None else f'_k{k}')+'.process.log')).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        return label
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            offpolicy=[case for case in cases if case.name.startswith('vtrace_')]
            for stage in ('train','evaluate'):
                jobs=[pool.submit(job,case,stage,k) for case in offpolicy for k in ((10,25,50) if stage=='train' else (None,))]
                for future in as_completed(jobs):
                    label=future.result();completed.append(label);print(label+' complete',flush=True)
                    write_json(status,dict(state='running',started_unix=started,elapsed_seconds=time.time()-started,completed=completed))
        subprocess.run([sys.executable,'-m','scripts.summarize_comparison',str(output)],cwd=ROOT,check=True)
        write_json(status,dict(state='completed',started_unix=started,elapsed_seconds=time.time()-started,completed=completed))
    except Exception as error:
        write_json(status,dict(state='failed',error=str(error),completed=completed));raise

if __name__=='__main__':main()
