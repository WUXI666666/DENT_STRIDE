"""Detached two-GPU queue for independent, frozen 160-epoch DSLR runs."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent
CHILD=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)


def verify(manifest,job=None):
    freeze=json.loads((ROOT/'LAUNCH_FREEZE.json').read_text())
    assert sha(ROOT/'execution_manifest.json')==freeze['execution_manifest_sha256']
    for row in manifest['source_files']+manifest['inputs']+manifest['script_files']:
        assert sha(row['path'])==row['sha256'],'Changed frozen input: '+row['path']
    assert shutil.disk_usage(ROOT).free>manifest['minimum_free_bytes'],'Free disk below 8GiB safety reserve'
    if job:
        assert json.loads(Path(job['job_file']).read_text())==job
        assert sha(job['config'])==job['config_sha256']


def claim(manifest,gpu,preferred=None):
    with (ROOT/'claims.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=[j for j in manifest['jobs'] if not (ROOT/'claims'/(j['id']+'.json')).exists()]
        if preferred:
            jobs.sort(key=lambda j:j['id']!=preferred)
        if not jobs:return None
        job=jobs[0]
        atomic(ROOT/'claims'/(job['id']+'.json'),{'job':job['id'],'gpu':gpu,'worker_pid':os.getpid(),
            'status':'claimed','unix':time.time()})
        return job


def aggregate(manifest):
    with (ROOT/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for job in manifest['jobs']:
            path=Path(job['output_dir'])/'COMPLETED.json'
            if not path.exists():continue
            result=json.loads((path.parent/'metrics_summary.json').read_text())
            for mode,record in result['prior_results'].items():
                rows.append({'Configuration':job['configuration'],'Seed':job['seed'],'Prior':mode,
                    **{k:record['metrics_percent'][k] for k in ['AP','AP50','AP75','AR100']},
                    'Best epoch':result['best_epoch_one_based'],'Params':result['params_total'],
                    'Checkpoint':str(path.parent/'best.pth')})
        if rows:
            tmp=ROOT/'results_summary.csv.tmp'
            with tmp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            os.replace(tmp,ROOT/'results_summary.csv')
        atomic(ROOT/'overall_progress.json',{'completed':len(rows)//4,'scheduled':len(manifest['jobs']),
            'all_completed':len(rows)//4==len(manifest['jobs'])})


def main():
    global CHILD
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--first-job');args=parser.parse_args()
    manifest=json.loads((ROOT/'execution_manifest.json').read_text());gpu=args.gpu
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lockroot=ROOT.parent/'.gpu_queue_locks';lockroot.mkdir(exist_ok=True)
    gpulock=(lockroot/(uuid+'.lock')).open('a');fcntl.flock(gpulock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'gpu':gpu,'gpu_uuid':uuid,'worker_pid':os.getpid(),'state':'preparing','completed':[],'started_unix':time.time()}
    statefile=ROOT/f'gpu{gpu}_state.json';atomic(statefile,state);job=None
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',DETERMINISTIC='0',STRICT_DETERMINISTIC='0')
    def terminate(signum,frame):
        if CHILD is not None and CHILD.poll() is None:os.killpg(CHILD.pid,signal.SIGTERM)
        state['state']='interrupted';atomic(statefile,state)
        if job:atomic(Path(job['output_dir'])/'INTERRUPTED.json',{'job':job['id'],'signal':signum,'unix':time.time()})
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def run(script,log,phase):
        global CHILD
        cmd=[manifest['python'],'-u',str(ROOT/script),job['job_file']]
        state.update(state=phase,command=cmd);atomic(statefile,state)
        with log.open('a') as f:
            CHILD=subprocess.Popen(cmd,cwd=job['native_cwd'],env=env,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=CHILD.pid;atomic(statefile,state);code=CHILD.wait()
        CHILD=None;state.pop('child_pid',None)
        if code:raise RuntimeError(f'{phase} failed, exit {code}: {log}')
    try:
        preferred=args.first_job
        while True:
            verify(manifest)
            pids=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).splitlines()
            others=[int(p.strip()) for p in pids if p.strip().isdigit()]
            if others:
                state.update(state='waiting_for_gpu',external_compute_pids=others);atomic(statefile,state)
                time.sleep(30);continue
            job=claim(manifest,gpu,preferred);preferred=None
            if job is None:
                state.update(state='queue_drained',finished_unix=time.time());atomic(statefile,state);aggregate(manifest);return
            verify(manifest,job);output=Path(job['output_dir'])
            assert not output.exists(),'Refusing to overwrite previous or partial run: '+str(output)
            output.mkdir(parents=True);shutil.copy2(job['config'],output/'config.yml');atomic(output/'job.json',job)
            state.update(current_job=job['id'],output_dir=str(output));atomic(statefile,state)
            existing=ROOT/'preflight'/job['id']/'preflight.json'
            if existing.exists():
                check=json.loads(existing.read_text());assert check['passed'] and check['config_sha256']==job['config_sha256']
                shutil.copy2(existing,output/'preflight.json')
            else:run('smoke_view_run.py',output/'preflight.log','preflight')
            verify(manifest,job)
            run('train_view_run.py',output/'console.log','training')
            verify(manifest,job)
            run('evaluate_view_runs.py',output/'evaluation.log','best_checkpoint_replay_and_view_priors')
            result=json.loads((output/'metrics_summary.json').read_text())
            assert result['verified_best_checkpoint'] and result['checkpoint_sha256']==sha(output/'best.pth')
            removed=[]
            for name in ['best_stg1.pth','best_stg2.pth','last.pth']:
                path=output/name
                if path.exists():
                    assert path.parent.resolve()==output.resolve() and not path.is_symlink()
                    path.unlink();removed.append(name)
            assert sorted(p.name for p in output.glob('*.pth'))==['best.pth']
            atomic(output/'checkpoint_retention.json',{'retained':'best.pth',
                'removed_temporary_stage_checkpoints':removed,'native_stage_rollback_preserved':True})
            atomic(output/'COMPLETED.json',{'job':job['id'],'gpu':gpu,'checkpoint_sha256':result['checkpoint_sha256'],
                'metrics_sha256':sha(output/'metrics_summary.json'),'finished_unix':time.time()})
            atomic(ROOT/'claims'/(job['id']+'.json'),{'job':job['id'],'gpu':gpu,'status':'completed','finished_unix':time.time()})
            state['completed'].append(job['id']);state['state']='job_completed';atomic(statefile,state)
            aggregate(manifest);print('JOB_COMPLETED',job['id'],result['metrics_percent']['AP'],flush=True);job=None
    except Exception as exc:
        state.update(state='failed_stopped',error=str(exc),traceback=traceback.format_exc());atomic(statefile,state)
        if job:
            output=Path(job['output_dir']);output.mkdir(parents=True,exist_ok=True)
            atomic(output/'FAILED.json',{'job':job['id'],'gpu':gpu,'error':str(exc),'unix':time.time()})
        print(state['traceback'],flush=True);raise


if __name__=='__main__':main()
