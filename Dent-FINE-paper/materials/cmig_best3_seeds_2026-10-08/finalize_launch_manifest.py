"""Explicit one-time final freeze after prelaunch review; never change jobs/data."""
from pathlib import Path
from hashlib import sha256
import json
import importlib.metadata
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (ROOT/'LAUNCHED.json').exists()
    assert not (ROOT/'LAUNCH_FREEZE.json').exists(),'Launch already frozen'
    assert not list((ROOT/'runs').glob('*/seed_*/TRAINING_STARTED.json'))
    path=ROOT/'execution_manifest.json'
    original=path.read_bytes()
    manifest=json.loads(original)
    for job in manifest['jobs']:
        assert json.loads(Path(job['job_file']).read_text())==job
        assert sha(job['config'])==job['config_sha256']
    for source in manifest['source_files']:
        assert sha(source['snapshot'])==source['sha256']
    for source in manifest['input_files']:
        if source['kind']!='manuscript':
            assert sha(source['path'])==source['sha256']
    previous=manifest['scripts'].copy()
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'packages':{name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','scipy','calflops','pycocotools']}}
    (ROOT/'software_environment.json').write_text(json.dumps(environment,indent=2))
    frozen_packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock.txt').write_text(frozen_packages)
    manifest['environment_files']={name:sha(ROOT/name) for name in ['software_environment.json','requirements-lock.txt']}
    for name in [*previous,'launch_two_gpus.py','finalize_launch_manifest.py','test_historical_execution.py','collect_tables.py']:
        manifest['scripts'][name]=sha(ROOT/name)
    record={'finalized_unix':time.time(),'preparation_manifest_sha256':sha256(original).hexdigest(),
            'changes':'Prelaunch review only: strict native prior/factor replays; job JSON integrity; separate evaluation TensorBoard; launcher hash and disk guard; verified table-data collector after all 30 completions. No jobs, configs, source, splits or assets changed.',
            'previous_script_sha256':previous,'final_script_sha256':manifest['scripts']}
    with (ROOT/'preparation_manifest.json').open('xb') as stream:stream.write(original)
    manifest['launch_review']=record
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    record['execution_manifest_sha256']=sha(path)
    (ROOT/'LAUNCH_FREEZE.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'jobs':len(manifest['jobs']),'launch_frozen':True,'manifest_sha256':sha(path)},indent=2))


if __name__=='__main__':main()
