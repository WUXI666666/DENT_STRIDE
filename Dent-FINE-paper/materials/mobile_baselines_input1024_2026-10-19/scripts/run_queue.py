from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, MODELS, run_dir, write, now, disk_guard, PYTHON, FASTER_PYTHON


def command_for(family,preflight=False):
    python=FASTER_PYTHON if family=='faster_rcnn' else PYTHON
    name='run_transformer.py' if family in ['deim_s','rtdetrv2_s','dome_detr_s'] else 'run_faster.py' if family=='faster_rcnn' else 'run_ultralytics.py'
    command=[python,'-u',str(ROOT/'scripts'/name)]
    if family!='faster_rcnn':command.append(family)
    if preflight:command.append('--preflight')
    return command


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--only',choices=MODELS);args=parser.parse_args()
    lock=(ROOT/'queue_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',
          OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',
          WANDB_DISABLED='true',COMET_MODE='DISABLED',TORCH_HOME=str(Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10/pretrained_cache')))
    (ROOT/'logs').mkdir(exist_ok=True)
    done=[];failed=[]
    for family in [args.only] if args.only else MODELS:
        out=run_dir(family)
        if not args.preflight and (out/'COMPLETED').exists():done.append(family);continue
        disk_guard();command=command_for(family,args.preflight)
        write(ROOT/('preflight_status.json' if args.preflight else 'queue_gpu0.json'),
              {'state':'preflight' if args.preflight else 'running','family':family,'completed':done,'failed':failed,
               'command':command,'pid':os.getpid(),'gpu':0,'updated_at':now()})
        log=ROOT/'logs'/f'{family}{"_preflight" if args.preflight else ""}.log'
        with log.open('a',buffering=1) as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
            if not result.returncode and not args.preflight:
                for entry in ['profile_benchmark.py','finalize.py']:
                    python=FASTER_PYTHON if family=='faster_rcnn' else PYTHON
                    result=subprocess.run([python,'-u',str(ROOT/'scripts'/entry),family],stdout=stream,
                                          stderr=subprocess.STDOUT,cwd=ROOT,env=env)
                    if result.returncode:break
        if result.returncode:
            failed.append({'family':family,'returncode':result.returncode,'log':str(log)})
            # Preserve all evidence; no retry/overwrite from an unreviewed failed state.
            if shutil_free_gib()<3:break
        else:done.append(family)
    write(ROOT/('preflight_status.json' if args.preflight else 'queue_gpu0.json'),
          {'state':'completed' if not failed else 'completed_with_failures','completed':done,'failed':failed,
           'pid':os.getpid(),'updated_at':now()})
    if failed:raise SystemExit(1)


def shutil_free_gib():
    import shutil
    return shutil.disk_usage(ROOT).free/1024**3


if __name__=='__main__':main()
