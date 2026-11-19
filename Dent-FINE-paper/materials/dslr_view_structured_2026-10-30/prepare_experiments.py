"""Freeze three independently trained incremental DSLR configurations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'cmig_selected_configs_multiseed_2026-10-11'
SOURCE=REFERENCE/'source_snapshots/local'
PYTHON='/home/wuxi/miniconda3/envs/deimv2/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    basepath=REFERENCE/'configs/dslr_cmfa_extra_decay_only/seed_0.yml'
    base=yaml.safe_load(basepath.read_text())
    prior=ROOT/'priors/view_specific_tooth_priors.pt'
    assert prior.exists()
    for folder in ['configs','jobs','claims']:(ROOT/folder).mkdir(exist_ok=True)
    specs=[('D01_cmfa_factor',False,False),('D02_cmfa_factor_viewproto',True,False),
           ('D03_cmfa_hooa_factor_viewproto',True,True)]
    jobs=[]
    for name,proto,hooa in specs:
        config=deepcopy(base);output=ROOT/'runs'/name/'seed_0'
        assert not output.exists(),'Refusing to overwrite a run: '+str(output)
        config['output_dir']=str(output);config['summary_dir']=str(output/'tensorboard')
        config['HybridEncoder']['use_hooa_attention']=hooa
        decoder=config['DFINETransformer'];decoder['use_factor_heads']=True
        decoder.update(factor_alpha_quad=0.0,factor_alpha_tooth=0.0,factor_alpha_dent=0.0,
            factor_fusion_train=False,use_fdi_factor_prior=False,fdi_factor_logit_scale=0.0,
            use_tooth_prototype=proto,prototype_count=4,prototype_tau=.50,
            prototype_fusion_weight=0.0,prototype_fusion_train=False)
        loss=config['DEIMCriterion'];loss['losses']=['mal','boxes','local','quad','tooth','dent']+(['proto'] if proto else [])
        # Match the actual Mobile supervised head coefficients, not the inert
        # native decoder.prototype_weight field.
        loss['weight_dict'].update(loss_quad=.08,loss_tooth=.08,loss_dent=.03,loss_proto=.03)
        path=ROOT/'configs'/(name+'_seed0.yml');path.write_text(yaml.safe_dump(config,sort_keys=False))
        job={'id':name+'_seed0','configuration':name,'cohort':'DSLR','seed':0,'epochs':160,
            'config':str(path),'config_sha256':sha(path),'implementation':str(SOURCE),
            'train_script':str(SOURCE/'train.py'),'native_cwd':'/home/wuxi/Dent-FINE',
            'python':PYTHON,'output_dir':str(output),'expected_input':640,
            'runtime_settings':{'num_views':5,'attach_dataset_metadata':True,'guard_mixed_view_augmentations':True},
            'view_prototype':proto,'HOOA':hooa,'factor_supervision':True,
            'factor_fusion_train':False,'factor_fusion_inference':False,
            'prior_path':str(prior),'prior_sha256':sha(prior),
            'prior_weights':{'center':.20,'shape':.05},'prior_modes':['none','center','shape','both'],
            'checkpoint_selection':'maximum native no-prior validation COCO AP; same selected EMA replayed in all prior modes',
            'initialization':'independent fresh detector with the same ImageNet-pretrained HGNetv2-B0 backbone; no detector resume/tuning',
            'baseline_config':str(basepath),'baseline_config_sha256':sha(basepath),
            'job_file':str(ROOT/'jobs'/(name+'_seed0.json'))}
        Path(job['job_file']).write_text(json.dumps(job,indent=2));jobs.append(job)
    manifest={'python':PYTHON,'source':str(SOURCE),'native_cwd':'/home/wuxi/Dent-FINE',
        'jobs':jobs,'dataset_policy':'existing DSLR splits unchanged',
        'gpu_ids':[0,1],'minimum_free_bytes':8*1024**3,'source_files':[],
        'inputs':[],'script_files':[],'fresh_training':True,
        'native_augmentation':'legacy640 multiscale; Mosaic0 MixUp0 crop.8 zoom.5',
        'native_optimizer':'AdamW base weight_decay .0001 plus realtime_decay .01; rollback_weight0'}
    for path in sorted(SOURCE.rglob('*.py')):
        manifest['source_files'].append({'path':str(path),'sha256':sha(path)})
    for path in [basepath,Path(base['train_dataloader']['dataset']['ann_file']),
                 Path(base['val_dataloader']['dataset']['ann_file']),prior,
                 ROOT/'priors/image_view_manifest.json',
                 Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth')]:
        manifest['inputs'].append({'path':str(path),'sha256':sha(path)})
    (ROOT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PREPARED',[(j['id'],j['view_prototype'],j['HOOA']) for j in jobs],flush=True)


if __name__=='__main__':main()
