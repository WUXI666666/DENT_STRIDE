"""Start both fresh-restart queues once; refuse duplicate or busy-GPU launch."""
from pathlib import Path
from hashlib import sha256
import fcntl
import json
import shutil
import subprocess
import time

SIDE=Path(__file__).resolve().parent
ROOT=SIDE.parent


def sha(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(2**20),b''):h.update(part)
    return h.hexdigest()


def main():
    assert not (SIDE/'LAUNCHED_RESTART.json').exists(),'Already launched; inspect running queues instead'
    ready=json.loads((SIDE/'RESTART_READY.json').read_text())
    assert ready['ready']
    assert sha(SIDE/'RECOVERY_FREEZE.json')==ready['plan_sha256']
    plan=json.loads((SIDE/'RECOVERY_FREEZE.json').read_text())
    manifest=json.loads((ROOT/'execution_manifest.json').read_text())
    assert sha(ROOT/'execution_manifest.json')==plan['original_manifest_sha256']
    for name,digest in plan['scripts'].items():assert sha(SIDE/name)==digest,name
    for name,digest in manifest['scripts'].items():assert sha(ROOT/name)==digest,name
    by_id={job['id']:job for job in manifest['jobs']}
    assigned=[jid for rows in plan['gpu_assignments'].values() for jid in rows]
    assert len(assigned)==len(set(assigned))==19
    for jid in assigned:
        job=by_id[jid];out=Path(job['output_dir'])
        assert json.loads(Path(job['job_file']).read_text())==job
        assert sha(job['config'])==job['config_sha256']
        assert not (out/'TRAINING_STARTED.json').exists(),'Existing partial run; refuse overwrite'
        assert not (out/'COMPLETED.json').exists(),'Do not retrain completed results'
        assert not list(out.rglob('*.pth')),'Unexpected retained checkpoint in fresh run'
    for record in plan['completed_jobs_preserved']:
        job=by_id[record['job_id']];out=Path(job['output_dir'])
        assert sha(out/'best.pth')==record['checkpoint_sha256']
        assert sha(out/'metrics_summary.json')==record['summary_sha256']
        assert sha(out/'native_coco_metrics.json')==record['native_metrics_sha256']
    assert shutil.disk_usage(ROOT).free>=manifest['disk_min_free_bytes']
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in rows.splitlines():
        gpu,memory,utilization=map(int,line.split(','))
        if gpu in [0,1]:assert memory<1000 and utilization<10,f'GPU{gpu} is busy'
    for gpu in [0,1]:
        with (ROOT/f'gpu{gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launches=[]
    for gpu in [0,1]:
        log=SIDE/f'gpu{gpu}_restart_queue.log'
        with log.open('x') as handle:
            command=[manifest['python'],'-u',str(SIDE/'queue_worker_restart.py'),'--gpu',str(gpu)]
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        launches.append({'gpu':gpu,'worker_pid':process.pid,'command':command,'log':str(log),
                         'jobs':len(plan['gpu_assignments'][str(gpu)]),
                         'first_job':plan['gpu_assignments'][str(gpu)][0]})
    record={'launched_unix':time.time(),'original_manifest_sha256':plan['original_manifest_sha256'],
            'restart_plan_sha256':ready['plan_sha256'],'workers':launches,
            'training_resume':False,'initialization':'Frozen historical pretrained initialization, not interrupted checkpoint',
            'assistant_monitoring':'Startup confirmation only; no ongoing polling'}
    (SIDE/'LAUNCHED_RESTART.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
