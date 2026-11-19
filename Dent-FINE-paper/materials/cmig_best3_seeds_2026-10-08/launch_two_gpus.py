"""Start two detached workers once; startup checks do not monitor training progress."""
from pathlib import Path
from hashlib import sha256
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def main():
    assert not (ROOT/'LAUNCHED.json').exists(), 'Already launched; never double-launch'
    manifest_path=ROOT/'execution_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert (ROOT/'LAUNCH_FREEZE.json').is_file(),'Final reviewed launch freeze missing'
    freeze=json.loads((ROOT/'LAUNCH_FREEZE.json').read_text())
    assert sha256(manifest_path.read_bytes()).hexdigest()==freeze['execution_manifest_sha256'],'Frozen manifest changed'
    assert shutil.disk_usage(ROOT).free>=manifest['disk_min_free_bytes'],'Insufficient free disk reserve'
    for name,digest in manifest['scripts'].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    for job in manifest['jobs']:
        assert json.loads(Path(job['job_file']).read_text())==job,job['id']
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in gpu_rows.strip().splitlines():
        idx,mem,util=map(int,line.split(','))
        if idx in [0,1]:
            assert mem<1000 and util<10, f'GPU {idx} busy; do not start over another task'
    for gpu in [0,1]:
        job=next(j for j in manifest['jobs'] if j['gpu']==gpu)
        pre=json.loads((Path(job['output_dir'])/'preflight.json').read_text())
        assert pre['passed'] and pre['config_sha256']==job['config_sha256']
    launches=[]
    for gpu in [0,1]:
        logfile=ROOT/f'gpu{gpu}_queue.log'
        with logfile.open('a') as out:
            command=[manifest['python'],'-u',str(ROOT/'queue_worker.py'),'--gpu',str(gpu)]
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launches.append({'gpu':gpu,'pid':process.pid,'command':command,'log':str(logfile),'jobs':sum(j['gpu']==gpu for j in manifest['jobs'])})
    record={'launched_unix':time.time(),'manifest_sha256':sha256(manifest_path.read_bytes()).hexdigest(),'workers':launches,'assistant_monitoring':'Startup confirmation only; user requested no ongoing monitoring.'}
    (ROOT/'LAUNCHED.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
