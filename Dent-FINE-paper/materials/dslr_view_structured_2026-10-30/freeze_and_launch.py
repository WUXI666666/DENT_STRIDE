"""Launch the two detached workers once, after discarded checks pass."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not (ROOT/'LAUNCHED.json').exists(),'Already launched; inspect existing workers instead of duplicating'
    manifest=json.loads((ROOT/'execution_manifest.json').read_text())
    for job in manifest['jobs']:
        report=json.loads((ROOT/'preflight'/job['id']/'preflight.json').read_text())
        assert report['passed'] and report['config_sha256']==job['config_sha256']
    for fname in ['view_identity_runtime.py','view_prior_ops.py','train_view_run.py','smoke_view_run.py',
                  'evaluate_view_runs.py','queue_worker.py','prepare_experiments.py','freeze_and_launch.py']:
        manifest['script_files'].append({'path':str(ROOT/fname),'sha256':sha(ROOT/fname)})
    (ROOT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'LAUNCH_FREEZE.json').write_text(json.dumps({'execution_manifest_sha256':sha(ROOT/'execution_manifest.json'),
        'frozen_unix':time.time()},indent=2))
    workers=[];env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    for gpu,first in [(0,manifest['jobs'][0]['id']),(1,manifest['jobs'][1]['id'])]:
        command=[manifest['python'],'-u',str(ROOT/'queue_worker.py'),'--gpu',str(gpu),'--first-job',first]
        with (ROOT/f'gpu{gpu}_queue.log').open('a') as f:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'gpu':gpu,'pid':proc.pid,'first_job':first,'command':command})
    marker={'workers':workers,'launched_unix':time.time(),'jobs':[j['id'] for j in manifest['jobs']],
        'policy':'three independent seed-0 160-epoch runs; dynamic claim of remaining third job; no persistent assistant monitoring'}
    (ROOT/'LAUNCHED.json').write_text(json.dumps(marker,indent=2));print(json.dumps(marker,indent=2),flush=True)


if __name__=='__main__':main()
