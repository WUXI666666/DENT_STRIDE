"""Freeze and launch the stopped GPU0 queue without editing frozen originals."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import os
import subprocess
import sys
import time

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parent


def sha(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20),b''):
            h.update(part)
    return h.hexdigest()


def main():
    assert not (SIDE/'LAUNCHED_RECOVERY.json').exists(), 'Recovery already launched'
    assert not (SIDE/'RECOVERY_FREEZE.json').exists(), 'Recovery already frozen'
    original=json.loads((ROOT/'execution_manifest.json').read_text())
    freeze=json.loads((ROOT/'LAUNCH_FREEZE.json').read_text())
    assert sha(ROOT/'execution_manifest.json')==freeze['execution_manifest_sha256']
    for name,digest in original['scripts'].items():
        assert sha(ROOT/name)==digest, 'Frozen original script changed: '+name
    state=json.loads((ROOT/'gpu0_state.json').read_text())
    assert state['state']=='failed_stopped' and state['current_job']=='dslr_cmfa_dualopt_seed1'
    try:
        os.kill(state['worker_pid'],0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Original GPU0 PID is still present; inspect instead of launching')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in rows.splitlines():
        index,memory,utilization=map(int,line.split(','))
        if index==0:
            assert memory<1000 and utilization<10,'GPU0 is busy'
    job=next(j for j in original['jobs'] if j['id']=='dslr_cmfa_dualopt_seed1')
    run=Path(job['output_dir'])
    record=json.loads((run/'evaluation/metrics.json').read_text())
    assert record['prediction_export_recovery_version']=='2026-10-09 native-forward observation'
    assert record['checkpoint_sha256']==sha(run/'best.pth')
    assert all(abs(v)<=2e-6 for v in record['native_export_minus_reference_fraction'].values())
    diagnosis=json.loads((SIDE/'export_diagnosis.json').read_text())
    assert diagnosis['before_hashes']=={name:sha(run/name) for name in diagnosis['before_hashes']}
    plan={'created_unix':time.time(),'original_manifest_sha256':sha(ROOT/'execution_manifest.json'),
          'scripts':{name:sha(SIDE/name) for name in ['export_native.py','queue_worker_recovery.py','launch_recovery.py']},
          'gpu':0,'assigned_jobs':[j['id'] for j in original['jobs'] if j['gpu']==0],
          'recovered_finished_training':job['id'],
          'changes':'Prediction export observes native EMA forward only. All training/source/config/split/checkpoint selection settings remain frozen. Preserve failed export and state; do not interrupt GPU1.',
          'assistant_monitoring':'Startup check only; no recurring assistant monitoring'}
    (SIDE/'RECOVERY_FREEZE.json').write_text(json.dumps(plan,indent=2))
    with (SIDE/'gpu0_recovery_queue.log').open('x') as stream:
        command=[original['python'],'-u',str(SIDE/'queue_worker_recovery.py'),'--gpu','0']
        worker=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,start_new_session=True)
    launched={'gpu':0,'worker_pid':worker.pid,'command':command,'launched_unix':time.time(),
              'original_manifest_sha256':plan['original_manifest_sha256'],
              'recovery_freeze_sha256':sha(SIDE/'RECOVERY_FREEZE.json')}
    (SIDE/'LAUNCHED_RECOVERY.json').write_text(json.dumps(launched,indent=2))
    print(json.dumps(launched,indent=2),flush=True)


if __name__=='__main__':
    main()
