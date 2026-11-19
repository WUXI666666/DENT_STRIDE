"""One detached serial worker per GPU. Stop safely on an error or changed input."""
import argparse
from pathlib import Path
from hashlib import sha256
import csv
import fcntl
import json
import importlib.util
import os
import signal
import shutil
import statistics
import subprocess
import time
import traceback

RECOVERY=Path(__file__).resolve().parent
ROOT=RECOVERY.parent
CHILD=None

def frozen_module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def recovery_guard():
    record=json.loads((RECOVERY/'RECOVERY_FREEZE.json').read_text())
    assert sha(ROOT/'execution_manifest.json')==record['original_manifest_sha256']
    for name,digest in record['scripts'].items():
        assert sha(RECOVERY/name)==digest,'Recovery sidecar changed: '+name

def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)

def verify(manifest,job,verify_source=False):
    recovery_guard()
    freeze=json.loads((ROOT/'LAUNCH_FREEZE.json').read_text())
    assert sha(ROOT/'execution_manifest.json')==freeze['execution_manifest_sha256'],'Frozen execution manifest changed'
    assert json.loads(Path(job['job_file']).read_text())==job,'Job JSON differs from frozen manifest'
    assert sha(job['config'])==job['config_sha256'],'Configuration changed'
    for p in manifest['input_files']:
        # Manuscripts may be edited while training; dataset/config sources may not.
        if p['kind']!='manuscript':assert sha(p['path'])==p['sha256'],p['path']
    assert sha(manifest['pretrained']['path'])==manifest['pretrained']['sha256']
    assert sha(manifest['image_manifest'])==manifest['image_manifest_sha256']
    for r in json.loads(Path(manifest['image_manifest']).read_text()):
        st=Path(r['path']).stat();assert st.st_size==r['size'] and st.st_mtime_ns==r['mtime_ns'],'Image changed: '+r['path']
    for n,h in manifest['scripts'].items():assert sha(ROOT/n)==h,'Execution code changed: '+n
    if verify_source:
        for r in manifest['source_files']:assert sha(r['snapshot'])==r['sha256'],r['snapshot']
    if shutil.disk_usage(ROOT).free<manifest['disk_min_free_bytes']:raise RuntimeError('Free space below 8 GiB safety reserve')

def aggregate():
    with (ROOT/'aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for marker in sorted((ROOT/'runs').glob('*/seed_*/COMPLETED.json')):
            r=json.loads((marker.parent/'metrics_summary.json').read_text());m=r['metrics_percent']
            rows.append({'configuration':r['configuration'],'cohort':r['cohort'],'seed':r['seed'],'AP':m['AP'],'AP50':m['AP50'],'AP75':m['AP75'],'AR100':m['AR100'],'best_epoch_zero_based':r['best_epoch_zero_based'],'best_epoch_one_based':r['best_epoch_one_based'],'params_total':r['params_total'],'metrics':str(marker.parent/'metrics_summary.json')})
        if rows:
            tmp=ROOT/'results_summary.csv.tmp'
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            os.replace(tmp,ROOT/'results_summary.csv')
            summary=[]
            for name in sorted({r['configuration'] for r in rows}):
                group=[r for r in rows if r['configuration']==name]
                if {r['seed'] for r in group}!={0,1,2}:continue
                summary.append({'configuration':name,'seeds':[0,1,2],'metrics':{k:{'mean':statistics.mean(r[k] for r in group),'SD':statistics.stdev(r[k] for r in group)} for k in ['AP','AP50','AP75','AR100']}})
            atomic(ROOT/'three_seed_mean_sd.json',summary)
        manifest=json.loads((ROOT/'execution_manifest.json').read_text())
        if len(rows)==len(manifest['jobs']) and not (ROOT/'manuscript_data').exists():
            with (ROOT/'collect_tables.log').open('a') as log:
                result=subprocess.run([manifest['python'],str(ROOT/'collect_tables.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('All training markers exist but manuscript table-data verification failed; see collect_tables.log')

def main():
    global CHILD
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args()
    manifest=json.loads((ROOT/'execution_manifest.json').read_text());gpu=args.gpu
    assert gpu==0,'This recovery is restricted to the stopped GPU0 queue'
    recovery_guard()
    evidence=frozen_module('frozen_recovery_evidence','collect_tables.py')
    replay=frozen_module('frozen_recovery_native','evaluate_native.py')
    lock=(ROOT/f'gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statefile=ROOT/f'gpu{gpu}_state.json'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',DETERMINISTIC='0',STRICT_DETERMINISTIC='0')
    state={'gpu':gpu,'worker_pid':os.getpid(),'started_unix':time.time(),'completed':[],'state':'preparing','recovery_script':str(Path(__file__))}
    atomic(statefile,state)
    def terminate(signum,frame):
        if CHILD is not None and CHILD.poll() is None:os.killpg(CHILD.pid,signal.SIGTERM)
        state['state']='interrupted';atomic(statefile,state);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def run(command,log,phase):
        global CHILD
        state['state']=phase;state['command']=command;atomic(statefile,state)
        with log.open('a') as handle:
            CHILD=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=job['native_cwd'],stdin=subprocess.DEVNULL,start_new_session=True)
            state['child_pid']=CHILD.pid;atomic(statefile,state)
            rc=CHILD.wait()
        CHILD=None
        if rc:raise RuntimeError(f'{phase} failed with exit {rc}; see {log}')
    def native_or_verified(out,job,mode=None,factor=None):
        filename=f'native_prior_{mode}_metrics.json' if mode else f'native_factor_{factor}_metrics.json' if factor else 'native_coco_metrics.json'
        if not (out/filename).exists():
            command=[manifest['python'],'-u',str(ROOT/'evaluate_native.py'),job['job_file']]
            if mode:command+=['--prior',mode]
            if factor:command+=['--factor-inference',factor]
            run(command,out/f'recovery_{filename}.log','native_best_checkpoint_replay')
        evidence.native_record(job,out/filename,sha(out/'best.pth'),
                               mode_kind='prior' if mode else 'factor' if factor else None,mode=mode or factor)
    def export_or_verified(out,job,mode=None,factor=None):
        destination=out/('replay_prior_'+mode if mode else 'replay_factor_'+factor if factor else 'evaluation')
        capture=out/('native_candidate_export_'+('prior_'+mode if mode else 'factor_'+factor if factor else 'default'))
        metrics=destination/'metrics.json'
        if not metrics.exists():
            command=[manifest['python'],'-u',str(RECOVERY/'export_native.py'),job['job_file']]
            if mode:command+=['--prior',mode]
            if factor:command+=['--factor-inference',factor]
            run(command,out/f'recovery_export_{mode or factor or "default"}.log','native_prediction_export')
        summary=json.loads(metrics.read_text())
        assert summary['checkpoint_sha256']==sha(out/'best.pth')
        assert summary['config_sha256']==job['config_sha256']
        assert summary['prediction_export_recovery_version']=='2026-10-09 native-forward observation'
        assert all(abs(v)<=2e-6 for v in summary['native_export_minus_reference_fraction'].values())
        provenance=json.loads((capture/'export_provenance.json').read_text())
        for filename,digest in provenance['files'].items():assert sha(destination/filename)==digest
    try:
        for job in sorted([j for j in manifest['jobs'] if j['gpu']==gpu],key=lambda j:j['queue_order']):
            env.update(PYTHONHASHSEED=str(job['seed']),PYTHONPATH=job['implementation'])
            state['current_job']=job['id'];state['output_dir']=job['output_dir'];atomic(statefile,state)
            verify(manifest,job,True)
            out=Path(job['output_dir']);marker=out/'COMPLETED.json'
            if marker.exists():
                evidence.verify_job(job);state['completed'].append(job['id']);continue
            # Prelaunch smoke output can be reused only with a matching config.
            training_done=(out/'TRAINING_STARTED.json').exists()
            if training_done:
                assert job['id']=='dslr_cmfa_dualopt_seed1','Only the audited finished training can be recovered'
                replay.validate_log(out)
                native_or_verified(out,job)
            out.mkdir(parents=True,exist_ok=True);shutil.copy2(job['config'],out/'config.yml');atomic(out/'job.json',job)
            if not (out/'preflight.json').exists():run([manifest['python'],'-u',str(ROOT/'smoke_controlled.py'),job['job_file']],out/'preflight.log','preflight')
            check=json.loads((out/'preflight.json').read_text());assert check['passed'] and check['config_sha256']==job['config_sha256']
            verify(manifest,job)
            if not training_done:
                run([manifest['python'],'-u',str(ROOT/'train_historical.py'),job['job_file']],out/'console.log','training')
            native_or_verified(out,job)
            export_or_verified(out,job)
            if job['configuration'] in ['pano_factor_prototype','pano_historical_spatialshape']:
                for mode in (['none','center','shape','both'] if job['configuration']=='pano_historical_spatialshape' else ['center','shape','both']):
                    native_or_verified(out,job,mode=mode)
                    export_or_verified(out,job,mode=mode)
            if job['seed']==0 and job['configuration'] in ['mobile_1024_full','internal_mobile_factor_train_off']:
                mode='off' if job['configuration']=='mobile_1024_full' else 'on'
                native_or_verified(out,job,factor=mode)
                export_or_verified(out,job,factor=mode)
            result=json.loads((out/'metrics_summary.json').read_text())
            removed=[]
            for name in ['best_stg1.pth','best_stg2.pth','last.pth']:
                p=out/name
                if p.exists():p.unlink();removed.append(name)
            atomic(out/'checkpoint_retention.json',{'retained_weight':'best.pth','removed_task_owned_temporary_weights':removed,'best_replay_verified':True})
            atomic(marker,{'job':job['id'],'checkpoint_sha256':sha(out/'best.pth'),'metrics_sha256':sha(out/'metrics_summary.json'),'native_metrics_sha256':sha(out/'native_coco_metrics.json'),'finished_unix':time.time()})
            evidence.verify_job(job)
            state['completed'].append(job['id']);state['state']='job_completed';atomic(statefile,state);aggregate()
        state['state']='all_completed';state['finished_unix']=time.time();atomic(statefile,state)
    except Exception as e:
        state.update(state='failed_stopped',error=str(e),traceback=traceback.format_exc());atomic(statefile,state)
        print(state['traceback'],flush=True);raise

if __name__=='__main__':main()
