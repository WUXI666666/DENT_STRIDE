"""Detached, best-only reruns of the two affected Mobile 1024 comparisons."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from common import ROOT, MODELS, OLD, PYTHON, run_dir, write, now, sha, disk_guard
from finalize import refresh_summary

REPAIR=ROOT/'repairs_20261020'
FAMILIES=['rtdetr_l','rtdetrv2_s']


def environment():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               WANDB_MODE='disabled',WANDB_DISABLED='true',COMET_MODE='DISABLED',
               TORCH_HOME=str(OLD/'pretrained_cache'))
    return env


def retained_checksums():
    result={}
    for family in MODELS:
        if family in FAMILIES:continue
        out=run_dir(family)
        summary=json.loads((out/'metrics_summary.json').read_text())
        assert (out/'COMPLETED').is_file()
        for path in [out/'metrics_summary.json',Path(summary['checkpoint'])]:
            result[str(path)]=sha(path)
    return result


def launch():
    if (REPAIR/'launch.json').exists():
        raise RuntimeError('A repair launch record already exists; inspect before retrying')
    for name in ['unit_tests.json','diagnostic_replay.json']:
        assert json.loads((REPAIR/name).read_text())['status']=='passed'
    for family in FAMILIES:
        assert json.loads((REPAIR/'smoke'/family/'seed_0/SMOKE_PASSED.json').read_text())['status']=='passed'
        out=run_dir(family)
        assert not (out/'best.pth').exists() and not (out/'weights/best.pt').exists()
        assert not (out/'log.txt').exists() and not (out/'validation_coco.jsonl').exists()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for entry in protocol['datasets'].values():
        assert sha(entry['annotation'])==entry['sha256']
    for entry in protocol['configs'].values():
        assert sha(entry['path'])==entry['sha256']
    disk_guard()
    manifest={'seed':0,'epochs':160,'input':[1024,1024],'gpu':0,'families':FAMILIES,
              'started_at':now(),'retained_six':retained_checksums(),
              'scripts':{str(p):sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},
              'run_paths':{f:str(run_dir(f)) for f in FAMILIES},
              'checkpoint_policy':'One best checkpoint in each repaired run; no last/periodic weights',
              'restarts':'Fresh seed-0 training from original pretrained initialization, not the invalid dental run',
              'repaired_runs_sha256':sha(ROOT/'repaired_runs.json')}
    write(REPAIR/'frozen_manifest.json',manifest)
    refresh_summary()
    log=(REPAIR/'queue.log').open('a',buffering=1)
    command=[PYTHON,'-u',str(Path(__file__).resolve()),'--worker']
    process=subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    write(REPAIR/'launch.json',{'pid':process.pid,'command':command,**manifest})
    print(json.dumps({'pid':process.pid,'models':FAMILIES,'gpu':0,'root':str(REPAIR)}))


def worker():
    lock=(ROOT/'queue_gpu0.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((REPAIR/'frozen_manifest.json').read_text())
    for path,expected in manifest['scripts'].items():assert sha(path)==expected, path
    assert sha(ROOT/'repaired_runs.json')==manifest['repaired_runs_sha256']
    done=[f for f in MODELS if f not in FAMILIES and (run_dir(f)/'COMPLETED').exists()]
    failed=[]
    env=environment()
    def status(state,**extra):
        value={'state':state,'completed':done,'failed':failed,'pid':os.getpid(),'gpu':0,
               'updated_at':now(),'repair_root':str(REPAIR),**extra}
        write(REPAIR/'status.json',value)
        write(ROOT/'queue_gpu0.json',value)
    for family in FAMILIES:
        disk_guard()
        out=run_dir(family);out.mkdir(parents=True,exist_ok=True)
        runner='run_ultralytics.py' if family=='rtdetr_l' else 'run_transformer.py'
        log=REPAIR/f'{family}.log'
        jobs=[('training',[PYTHON,'-u',str(ROOT/'scripts'/runner),family]),
              ('profiling',[PYTHON,'-u',str(ROOT/'scripts/profile_benchmark.py'),family]),
              ('finalizing',[PYTHON,'-u',str(ROOT/'scripts/finalize.py'),family])]
        with log.open('a',buffering=1) as stream:
            for stage,command in jobs:
                if stage=='training' and (out/'TRAINING_COMPLETED').exists():continue
                status('running',family=family,stage=stage,command=command,log=str(log))
                print(now(),family,stage,flush=True)
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    failed.append({'family':family,'stage':stage,'returncode':result.returncode,'log':str(log)})
                    break
            else:
                done.append(family)
        refresh_summary()
    for path,expected in manifest['retained_six'].items():assert sha(path)==expected,path
    status('completed' if not failed else 'completed_with_failures',retained_six_unchanged=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker()
    else:launch()
