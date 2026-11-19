"""Start a detached GPU0 serial queue only after all eight preflights pass."""
from __future__ import annotations
import json
import os
import subprocess
from common import ROOT, MODELS, PYTHON, write, now, disk_guard, sha


def main():
    if (ROOT/'launch.json').exists():raise RuntimeError('Already launched; inspect the queue instead.')
    for family in MODELS:
        record=ROOT/'preflight'/f'{family}.json'
        assert record.is_file() and json.loads(record.read_text())['status']=='passed', family
    disk_guard()
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='0')
    command=[PYTHON,'-u',str(ROOT/'scripts/run_queue.py')]
    with (ROOT/'queue_gpu0.log').open('a') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,cwd=ROOT,env=env)
    record={'pid':process.pid,'command':command,'root':str(ROOT),'gpu':0,'models':MODELS,
            'seed':0,'epochs':160,'input':[1024,1024],'started_at':now(),
            'scripts':{p.name:sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}}
    write(ROOT/'launch.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
