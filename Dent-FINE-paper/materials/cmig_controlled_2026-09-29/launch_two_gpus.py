"""Launch only after both first-job preflights pass; no training-progress monitoring."""
from pathlib import Path
from hashlib import sha256
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent

def main():
    assert not (ROOT/'LAUNCHED.json').exists(),'Already launched; inspect existing queue, do not double-launch'
    path=ROOT/'execution_manifest.json';manifest=json.loads(path.read_text())
    # Finalize executable fingerprints before any trainer has started.
    assert not list((ROOT/'runs').glob('*/seed_*/TRAINING_STARTED.json'))
    for name in manifest['scripts']:
        manifest['scripts'][name]=sha256((ROOT/name).read_bytes()).hexdigest()
    manifest['launch_script_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for gpu in [0,1]:
        job=next(j for j in manifest['jobs'] if j['gpu']==gpu)
        pre=json.loads((Path(job['output_dir'])/'preflight.json').read_text())
        assert pre['passed'] and pre['config_sha256']==job['config_sha256']
    launches=[]
    for gpu in [0,1]:
        logfile=ROOT/f'gpu{gpu}_queue.log'
        with logfile.open('a') as out:
            command=[manifest['python'],'-u',str(ROOT/'queue_worker.py'),'--gpu',str(gpu)]
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launches.append({'gpu':gpu,'pid':p.pid,'command':command,'log':str(logfile),'jobs':sum(j['gpu']==gpu for j in manifest['jobs'])})
    record={'launched_unix':time.time(),'manifest_sha256':sha256(path.read_bytes()).hexdigest(),'workers':launches,'assistant_monitoring':'Startup confirmation only, as requested.'}
    (ROOT/'LAUNCHED.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
