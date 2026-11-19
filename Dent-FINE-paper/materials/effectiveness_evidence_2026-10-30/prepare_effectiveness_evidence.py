"""Controlled retraining evidence, not visual contrast manipulation."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

HERE=Path(__file__).resolve().parent
MATERIALS=Path('/home/wuxi/Dent-FINE-paper/materials')
CONTROL=MATERIALS/'cmig_controlled_2026-09-29'
REFERENCE=MATERIALS/'cmig_best3_seeds_2026-10-08'
DATA={
 'Mobile':('/home/wuxi/coco/valid/annotations_fdi_types.json','/home/wuxi/coco/valid'),
 'DSLR':('/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json',
         '/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val')}


def read(p):
 p=Path(p)
 if p.suffix=='.gz':
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(p.read_text())


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def config_differences(a,b,p=''):
 differences=[]
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in ['output_dir','summary_dir']:differences+=config_differences(a.get(k),b.get(k),p+'.'+k)
 elif a!=b:differences.append({'field':p,'before':a,'after':b})
 return differences


def load_run(root,config,seed):
 run=root/'runs'/config/f'seed_{seed}'
 native=run/'native_coco_metrics.json'
 if not native.exists():native=run/'metrics_summary.json'
 metric=read(native)
 evaluation=read(run/'evaluation/metrics.json')
 assert evaluation['checkpoint_sha256']==metric['checkpoint_sha256']
 ckpt=run/'best.pth'
 assert sha(ckpt)==metric['checkpoint_sha256']
 conf=root/'configs'/config/f'seed_{seed}.yml'
 job=read(root/'jobs'/f'{config}_seed{seed}.json')
 return {'configuration':config,'seed':seed,'metric_file':str(native),
         'checkpoint':str(ckpt),'checkpoint_sha256':metric['checkpoint_sha256'],
         'config':str(conf),'config_data':yaml.safe_load(conf.read_text()),'implementation':job['implementation'],
         'controls':job.get('controls',{}),'metrics':metric['metrics_percent'],
         'fixed_threshold':evaluation['fixed_threshold'],
         'prediction_file':str(run/'evaluation/predictions_coco.json.gz'),
         'prediction_sha256':sha(run/'evaluation/predictions_coco.json.gz'),
         'evaluation_minus_native_AP':evaluation['metrics_percent']['AP']-metric['metrics_percent']['AP']}


def dataset(cohort):
 a=read(DATA[cohort][0]);ims={i['id']:i for i in a['images']};gt=defaultdict(list)
 for g in a['annotations']:gt[g['image_id']].append(g)
 return a,ims,gt,{c['id']:int(c.get('fdi',c['name'])) for c in a['categories']}


def predictions(run):
 out=defaultdict(list)
 for r in read(run['prediction_file']):
  if r['score']>=.30:out[r['image_id']].append(r)
 return out


def match(pp,gg):
 a=np.asarray([p['bbox'] for p in pp],float).reshape(-1,4)
 b=np.asarray([g['bbox'] for g in gg],float).reshape(-1,4)
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
 inter=np.maximum(hi-lo,0).prod(-1)
 ov=inter/np.maximum(a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-inter,1e-12)
 st={g['id']:{'status':'missed','prediction':None,'IoU':None} for g in gg}
 for r,c in zip(*linear_sum_assignment(-ov)):
  if ov[r,c]>=.50:st[gg[c]['id']]={'status':'correct' if pp[r]['category_id']==gg[c]['category_id'] else 'wrong','prediction':pp[r],'IoU':float(ov[r,c])}
 return st,{'TP':sum(s['status']=='correct' for s in st.values()),'GT':len(gg),'predictions':len(pp)}


def select_cases(cohort,before,after,kind='identity'):
 ann,ims,gt,fdi=dataset(cohort);pp0=predictions(before);pp1=predictions(after);inventory=[]
 for iid,gg in gt.items():
  s0,c0=match(pp0[iid],gg);s1,c1=match(pp1[iid],gg)
  for g in gg:
   a,b=s0[g['id']],s1[g['id']]
   target=fdi[g['category_id']]
   if b['status']!='correct':continue
   if kind=='localization':
    eligible=a['status']=='correct' and .50<=a['IoU']<.75 and b['IoU']>=.80 and b['IoU']-a['IoU']>=.12
   elif kind=='identity':eligible=a['status']!='correct'
   else:
    if a['status']!='wrong':continue
    old=fdi[a['prediction']['category_id']]
    if kind=='quadrant':eligible=(old>=50)==(target>=50) and old%10==target%10 and old//10!=target//10
    elif kind=='position':eligible=old//10==target//10 and old%10!=target%10
    elif kind=='dentition':eligible=(old>=50)!=(target>=50) and old%10==target%10
    else:raise ValueError(kind)
   if not eligible:continue
   record=ims[iid];fraction=g['bbox'][2]*g['bbox'][3]/(record['width']*record['height'])
   gain=c1['TP']-c0['TP']
   rank=(int(gain>=0),gain, (b['IoU']-a['IoU']) if kind=='localization' else 0,
         int(a['status']=='wrong'),fraction,b['IoU'])
   inventory.append((rank,{'cohort':cohort,'image_id':iid,'image_record':record,
            'image_file':str(Path(DATA[cohort][1])/record['file_name']),
            'target_annotation':g,'target_FDI':target,'kind':kind,'before':a,'after':b,
            'before_FDI':fdi[a['prediction']['category_id']] if a['prediction'] else None,
            'after_FDI':target,'counts_before':c0,'counts_after':c1}))
 assert inventory,(cohort,kind,'no qualified example')
 chosen=max(inventory,key=lambda x:x[0])[1]
 chosen['image_sha256']=sha(chosen['image_file'])
 chosen['eligible_examples']=len(inventory)
 return chosen


def main():
 out={'data':DATA,'groups':{},'note':'All controlled results are internal validation, not independent-test or universal-configuration claims.'}
 rows=[]
 for optimizer in ['adamw','dualopt']:
  for seed in [0,1,2]:
   a=load_run(CONTROL,'dslr_base_'+optimizer,seed)
   b=load_run(CONTROL,'dslr_cmfa_'+optimizer,seed)
   diff=config_differences(a['config_data'],b['config_data'])
   assert diff==[{'field':'.HybridEncoder.use_cmfa_fusion','before':False,'after':True}],diff
   assert a['implementation']==b['implementation'] and a['controls']==b['controls']
   rows.append({'optimizer':optimizer,'seed':seed,'before_AP':a['metrics']['AP'],
                'after_AP':b['metrics']['AP'],'delta_AP':b['metrics']['AP']-a['metrics']['AP'],
                'before':a,'after':b,'config_differences':diff})
 summaries=[]
 for opt in ['adamw','dualopt']:
  rr=[r for r in rows if r['optimizer']==opt];d=np.array([r['delta_AP'] for r in rr])
  summaries.append({'optimizer':opt,'seeds':[0,1,2],'paired_AP_deltas':d.tolist(),
       'mean_paired_delta_AP':float(d.mean()),'sample_SD_paired_delta_AP':float(d.std(ddof=1)),
       'before_mean_AP':float(np.mean([r['before_AP'] for r in rr])),
       'after_mean_AP':float(np.mean([r['after_AP'] for r in rr]))})
 a,b=rows[0]['before'],rows[0]['after']
 out['groups']['CMFA']={'runs':rows,'summaries':summaries,
       'cases':[select_cases('DSLR',a,b,'identity'),select_cases('DSLR',a,b,'localization')],
       'role':'CMFA learning effect in the controlled DSLR configuration; not the later selected extra-decay-only recipe',
       'DualOpt_definition':'base AdamW plus both extra decay and rollback'}
 specs=[('Full HOOA','mobile_1024_full'),('No high-pass','internal_mobile_hooa_no_highpass'),
        ('No second-order','internal_mobile_hooa_no_second_order'),('Neither','internal_mobile_hooa_neither')]
 hr=[{'label':lab,**load_run(REFERENCE,cfg,0)} for lab,cfg in specs]
 for r in hr:
  assert not config_differences(hr[0]['config_data'],r['config_data'])
 old=read(MATERIALS/'cmig_revision_2026-10-11/prediction_figures/prediction_figure_sources.json')['figures']['Figure_8_HOOA_same_image']
 out['groups']['HOOA']={'runs':hr,'cases':old['cases'],
       'prototype_temperature':.20,'seed':0,
       'controls_definition':'No second-order replaces L*sigmoid(C) with C; no high-pass replaces E with zeros; concat width is unchanged.'}
 fr=[{'label':lab,**load_run(REFERENCE,cfg,0)} for lab,cfg in [
     ('All factors','internal_mobile_factor_train_off'),('No quadrant','internal_mobile_factor_no_quadrant'),
     ('No position','internal_mobile_factor_no_position'),('No dentition','internal_mobile_factor_no_dentition')]]
 differences=[]
 for r in fr[1:]:differences.append({'label':r['label'],'differences':config_differences(fr[0]['config_data'],r['config_data'])})
 assert all(r['config_data']['DFINETransformer']['factor_fusion_train']==False for r in fr)
 cases=[select_cases('Mobile',r,fr[0],kind) for r,kind in zip(fr[1:],['quadrant','position','dentition'])]
 out['groups']['Factor']={'runs':fr,'cases':cases,'config_differences':differences,
       'prototype_temperature':.20,'seed':0,'factor_fusion':'disabled during training and inference for all controls'}
 (HERE/'controlled_evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print('CMFA',json.dumps(summaries,indent=2))
 for group in ['CMFA','Factor']:
  print(group,'CASES',[{k:c.get(k) for k in ['image_id','kind','target_FDI','before_FDI','after_FDI','before','after']} for c in out['groups'][group]['cases']])
 print('ALL_NATIVE_EXPORT_AP_DIFFERENCES',[(r['configuration'],r['evaluation_minus_native_AP']) for r in hr+fr])


if __name__=='__main__':main()
