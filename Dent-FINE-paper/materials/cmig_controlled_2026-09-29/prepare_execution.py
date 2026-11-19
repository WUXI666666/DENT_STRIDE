"""Freeze the explicitly authorized original-split, two-GPU experiment queue."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from datetime import datetime, timezone
import csv
import json
import os
import shutil
import yaml

ROOT=Path(__file__).resolve().parent
PYTHON='/home/wuxi/miniconda3/envs/deimv2/bin/python'

def sha(p): return sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def main():
    assert not (ROOT/'execution_manifest.json').exists(), 'Execution already frozen'
    rows=list(csv.DictReader((ROOT/'primary_experiments.csv').open(encoding='utf-8-sig')))
    jobs=[]
    def add(name,cohort,seed,c,controls=None,phase='primary'):
        c=deepcopy(c)
        # HGNetv2 concatenates this directory with a basename, requiring '/'.
        c['HGNetv2']['local_model_dir']='/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/'
        out=ROOT/'runs'/name/f'seed_{seed}';c['output_dir']=str(out)
        c['summary_dir']=str(out/'tensorboard')
        path=ROOT/'configs'/name/f'seed_{seed}.yml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(c,sort_keys=False))
        jobs.append({'id':f'{name}_seed{seed}','configuration':name,'cohort':cohort,'seed':seed,'phase':phase,'config':str(path),'config_sha256':sha(path),'output_dir':str(out),'implementation':str(ROOT/'source_snapshots'/('mobile' if cohort=='Mobile' else 'local')),'controls':controls or {},'epochs':160,'expected_input':c['eval_spatial_size'][0]})
    for r in rows:
        add(r['configuration'],r['cohort'],int(r['seed']),yaml.safe_load(Path(r['config']).read_text()))
    def base(name):return yaml.safe_load(Path(next(j['config'] for j in jobs if j['configuration']==name and j['seed']==0)).read_text())
    mobile=base('mobile_1024_full')
    for name,controls in [
        ('cmfa_no_gates',{'cmfa_channel':False,'cmfa_spatial':False}),
        ('cmfa_channel_only',{'cmfa_channel':True,'cmfa_spatial':False}),
        ('cmfa_spatial_only',{'cmfa_channel':False,'cmfa_spatial':True}),
        ('hooa_no_highpass',{'hooa_highpass':False}),
        ('hooa_no_second_order',{'hooa_second_order':False}),
        ('hooa_neither',{'hooa_highpass':False,'hooa_second_order':False}),
    ]: add('internal_mobile_'+name,'Mobile',0,mobile,controls,'internal')
    fusion_off=deepcopy(mobile);fusion_off['DFINETransformer']['factor_fusion_train']=False
    add('internal_mobile_factor_train_off','Mobile',0,fusion_off,{'factor_inference_fusion':False},'internal')
    for name,loss in [('quadrant','quad'),('position','tooth'),('dentition','dent')]:
        c=deepcopy(fusion_off);c['DEIMCriterion']['losses'].remove(loss)
        add('internal_mobile_factor_no_'+name,'Mobile',0,c,{'factor_inference_fusion':False},'internal')
    c=deepcopy(mobile);c['DFINETransformer']['use_tooth_prototype']=False;c['DEIMCriterion']['losses'].remove('proto')
    add('internal_mobile_proto_off','Mobile',0,c,phase='internal')
    for k,t in [(1,.2),(2,.2),(4,.1),(4,.5)]:
        c=deepcopy(mobile);c['DFINETransformer']['prototype_count']=k;c['DFINETransformer']['prototype_tau']=t
        add(f'internal_mobile_proto_k{k}_tau{str(t).replace(".","p")}','Mobile',0,c,phase='internal')
    dslr=base('dslr_cmfa_dualopt')
    for name,rb,decay in [('rollback_only',.005,0.),('extra_decay_only',0.,.01)]:
        c=deepcopy(dslr);c['optimizer'].update(rollback_weight=rb,realtime_decay=decay)
        add('internal_dslr_'+name,'DSLR',0,c,phase='internal')
    for name,nwd,arch in [('neither',False,False),('nwd_only',True,False),('arch_only',False,True)]:
        c=deepcopy(mobile)
        if not nwd:c['DEIMCriterion']['losses'].remove('nwd')
        c['DEIMCriterion']['use_arch_order_loss']=arch
        add('internal_mobile_background_'+name,'Mobile',0,c,phase='internal')
    assert len(jobs)==62
    # Main full-model seed0 runs first; then all main comparisons before internals.
    priority={'mobile_1024_full':0,'pano_factor_prototype':1,'dslr_cmfa_dualopt':2}
    jobs.sort(key=lambda j:(j['phase']!='primary',j['seed'],priority.get(j['configuration'],10),j['configuration']))
    loads={0:0.,1:0.};order={0:0,1:0}
    for j in jobs:
        gpu=min(loads,key=loads.get);j['gpu']=gpu;j['queue_order']=order[gpu];order[gpu]+=1
        loads[gpu]+=4 if j['cohort']=='Mobile' and j['expected_input']==1024 else 2
    source=json.loads((ROOT/'source_snapshot_manifest.json').read_text())
    for row in source: assert sha(row['snapshot'])==row['sha256']
    inputs=json.loads((ROOT/'audit/source_manifest.json').read_text())
    for row in inputs:assert sha(row['path'])==row['sha256']
    image_records=list(csv.DictReader((ROOT/'audit/image_hash_manifest.csv').open(encoding='utf-8-sig')))
    for r in image_records:
        assert sha(r['path'])==r['bytes_sha256']
        st=Path(r['path']).stat();r['size']=st.st_size;r['mtime_ns']=st.st_mtime_ns
    write(ROOT/'frozen_image_manifest.json',image_records)
    for j in jobs:
        p=ROOT/'jobs'/f"{j['id']}.json";j['job_file']=str(p);write(p,j)
    backbone='/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth'
    scripts=['controlled_runtime.py','train_controlled.py','evaluate_controlled.py','smoke_controlled.py','queue_worker.py']
    manifest={'frozen_at':datetime.now(timezone.utc).isoformat(),'authorization':'Use unchanged original splits; two GPUs; no assistant monitoring after confirmed startup. All primary seeds0/1/2, internal seed0. Known overlap retained and disclosed.','python':PYTHON,'primary_jobs':42,'internal_jobs':20,'historical_runs_reused':0,'jobs':jobs,'source_files':source,'input_files':inputs,'image_manifest':str(ROOT/'frozen_image_manifest.json'),'image_manifest_sha256':sha(ROOT/'frozen_image_manifest.json'),'pretrained':{'path':backbone,'sha256':sha(backbone)},'scripts':{s:sha(ROOT/s) for s in scripts},'gpus':[0,1],'disk_min_free_bytes':8*1024**3,'stage_policy':'Native per-cohort stage behavior preserved. Only unrelated last/stage2 writes suppressed; necessary stage1 retained until successful replay.'}
    write(ROOT/'execution_manifest.json',manifest)
    print(json.dumps({'jobs':len(jobs),'gpu_jobs':dict(Counter(j['gpu'] for j in jobs)),'first_jobs':{g:next(j['id'] for j in jobs if j['gpu']==g) for g in [0,1]}},indent=2))

if __name__=='__main__':
    from collections import Counter
    main()
