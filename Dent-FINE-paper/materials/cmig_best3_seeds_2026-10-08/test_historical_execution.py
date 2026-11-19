"""Prelaunch static checks and isolated replay of archived, unchanged checkpoints."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse
import json
import os
import shutil
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native',choices=['Mobile','Pano','DSLR']);parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
    manifest=json.loads((ROOT/'execution_manifest.json').read_text())
    assert len(manifest['jobs'])==30
    assert Counter(j['phase'] for j in manifest['jobs'])=={'primary':9,'internal':21}
    main_jobs=[j for j in manifest['jobs'] if j['phase']=='primary']
    assert {(j['configuration'],j['seed']) for j in main_jobs}=={
        (configuration,seed)
        for configuration in ['mobile_1024_full','pano_historical_spatialshape','dslr_cmfa_dualopt']
        for seed in [0,1,2]
    }
    references=[j for j in manifest['jobs'] if j.get('scope_role')=='internal_reference']
    assert len(references)==1 and references[0]['id']=='internal_dslr_cmfa_adamw_seed0'
    for j in manifest['jobs']:
        assert json.loads(Path(j['job_file']).read_text())==j
        assert sha256(Path(j['config']).read_bytes()).hexdigest()==j['config_sha256']
        cfg=yaml.safe_load(Path(j['config']).read_text())
        assert cfg['epoches']==160 and cfg['train_dataloader']['total_batch_size']==8
        if j['configuration'] in ['mobile_1024_full','dslr_cmfa_dualopt','pano_historical_spatialshape']:
            anchor=yaml.safe_load(Path(j['historical_anchor_config']).read_text())
            for item in [cfg,anchor]:
                item.pop('output_dir',None);item.pop('summary_dir',None)
            assert cfg==anchor
    print('STATIC_30_JOB_CHECK_PASSED',flush=True)
    if not args.native:return
    anchors=json.loads((ROOT/'historical_mapping.json').read_text())['anchors']
    anchor=anchors[args.native]
    destination=ROOT/'verification'/('native_'+args.native.lower())
    assert not destination.exists(),'Verification already exists; inspect it instead of overwrite'
    destination.mkdir(parents=True)
    source=Path(anchor['checkpoint']);before=sha256(source.read_bytes()).hexdigest()
    assert before==anchor['checkpoint_sha256']
    # Private copy, never a symlink or a write through a shared historical inode.
    shutil.copy2(source,destination/'best.pth')
    shutil.copy2(source.parent/'log.txt',destination/'log.txt')
    shutil.copy2(anchor['config'],destination/'config.yml')
    impl=ROOT/'source_snapshots'/anchor['snapshot_name']
    job={'id':'verification_'+args.native.lower(),'configuration':'archived_native_verification','cohort':args.native,'seed':0,
         'implementation':str(impl),'config':anchor['config'],'config_sha256':anchor['config_sha256'],
         'output_dir':str(destination),'controls':{},'native_cwd':'/home/wuxi/Dent-FINE'}
    job_path=destination/'job.json';job_path.write_text(json.dumps(job,indent=2))
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONHASHSEED='0',PYTHONPATH=str(impl),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',DETERMINISTIC='0',STRICT_DETERMINISTIC='0')
    subprocess.run([sys.executable,str(ROOT/'evaluate_native.py'),str(job_path)],cwd='/home/wuxi/Dent-FINE',env=environment,check=True)
    assert sha256(source.read_bytes()).hexdigest()==before,'Archived checkpoint changed'
    print('ARCHIVED_NATIVE_REPLAY_PASSED',args.native,flush=True)


if __name__=='__main__':main()
