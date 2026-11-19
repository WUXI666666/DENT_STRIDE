"""Check plotted labels against raw queries, validate artifacts, package PNGs."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def label_check(run,case,prediction):
 p=Path(run['prediction_file']).parent/'pre_topk_candidates.npz';z=np.load(p)
 at=int(np.flatnonzero(z['image_ids']==case['image_id'])[0])
 if 'image_record' in case:im=case['image_record']
 else:
  ann=read('/home/wuxi/coco/valid/annotations_fdi_types.json')
  im=next(r for r in ann['images'] if r['id']==case['image_id'])
 b=z['boxes_cxcywh'][at].astype(float)*np.array([im['width'],im['height'],im['width'],im['height']])
 b[:,:2]-=b[:,2:]/2
 err=np.max(abs(b-np.array(prediction['bbox'])),axis=1)
 eligible=np.flatnonzero(err<.03);assert len(eligible),case['image_id']
 prob=1/(1+np.exp(-np.clip(z['logits'][at],-80,80)))
 q=min(eligible,key=lambda j:abs(float(prob[j,prediction['category_id']])-prediction['score']))
 score_error=abs(float(prob[q,prediction['category_id']])-prediction['score'])
 assert score_error<1e-5
 assert int(prob[q].argmax())==prediction['category_id'],(run['configuration'],case['image_id'])
 return {'configuration':run['configuration'],'image_id':case['image_id'],'query':int(q),
          'displayed_category_is_query_top1':True,'bbox_max_error_pixels':float(err[q]),'score_error':score_error,
          'raw_source':str(p)}


def main():
 data=read(HERE/'controlled_evidence.json');checks=[]
 r=data['groups']['CMFA']['runs'][0]
 for c in data['groups']['CMFA']['cases']:
  checks+=[label_check(r['before'],c,c['before']['prediction']),label_check(r['after'],c,c['after']['prediction'])]
 factor=data['groups']['Factor']
 for r,c in zip(factor['runs'][1:],factor['cases']):
  assert r['controls'].get('factor_inference_fusion') is False
  checks+=[label_check(r,c,c['before']['prediction']),label_check(factor['runs'][0],c,c['after']['prediction'])]
 assert factor['runs'][0]['controls'].get('factor_inference_fusion') is False
 hooa=data['groups']['HOOA'];before=next(r for r in hooa['runs'] if r['label']=='Neither')
 after=next(r for r in hooa['runs'] if r['label']=='Full HOOA')
 for c in hooa['cases']:
  checks+=[label_check(before,c,c['statuses']['Neither branch']['prediction']),label_check(after,c,c['statuses']['Full HOOA']['prediction'])]
 occ=read(HERE/'factor_occlusion_provenance.json')
 assert occ['patches_per_image']==841 and occ['stride']==[32,32]
 assert all(r['unoccluded_frozen_query_max_error']==0 for r in occ['records'])
 attr=read(HERE/'factor_head_attribution_provenance.json')
 assert attr['checkpoint_sha256']==occ['checkpoint_sha256'] and attr['passive_hook_prediction_max_difference']==0
 attr['gradient_scope']='The native encoder top-k query-content initialization includes detach; gradients describe remaining differentiable paths, not complete input-perturbation effects. Conditional input occlusion is provided separately.'
 (HERE/'factor_head_attribution_provenance.json').write_text(json.dumps(attr,indent=2))
 fig=read(HERE/'figure_manifest.json');artifacts=[]
 for name in fig:
  im=Image.open(HERE/(name+'.png'));assert im.info['dpi'][0]>=599
  ET.parse(HERE/(name+'.svg'));assert (HERE/(name+'.pdf')).read_bytes().startswith(b'%PDF')
  artifacts.append({'figure':name,'PNG_size':list(im.size),'DPI':im.info['dpi'][0]})
 report={'query_label_checks':checks,'artifacts':artifacts,'occlusion_patches_per_image':841,
         'unoccluded_query_freezing_invariance':True,'passive_hook_invariance':True,
         'weights_and_manuscripts_modified':False,'new_training_started':False}
 (HERE/'verification_report.json').write_text(json.dumps(report,indent=2))
 selected=['Figure_CMFA_controlled_evidence','Figure_HOOA_controlled_evidence','Figure_Factor_supervision_evidence',
           'Figure_Factor_head_occlusion_heatmaps','Figure_Factor_head_occlusion_overlay','Figure_Factor_head_confusions']
 archive=HERE/'Recommended_effectiveness_figures_PNG.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in selected:
   for ext in ['.png','_preview.png']:z.write(HERE/(name+ext),name+ext)
  for f in ['图片使用建议与结论范围.txt','figure_captions.json']:z.write(HERE/f,f)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print(json.dumps({'checked_query_labels':len(checks),'figures':len(artifacts),'archive_MB':round(archive.stat().st_size/1024/1024,2)},indent=2))


if __name__=='__main__':main()
