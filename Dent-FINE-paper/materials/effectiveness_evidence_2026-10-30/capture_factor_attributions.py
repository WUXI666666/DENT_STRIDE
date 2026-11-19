"""Query-specific factor-head attributions from actual trained logits.

Saves conventional Grad-CAM and elementwise gradient-times-activation
(HiResCAM form). Both are explanatory diagnostics, not native attention.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
job=json.loads((ROOT/'jobs/mobile_proto_k4_tau0p50_seed2.json').read_text())
sys.path.insert(0,job['implementation']);sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from controlled_runtime import install_controls

TARGETS=[(115,64),(77,11),(182,46)]


def target_iou(bb,g):
 lo=np.maximum(bb[:,:2]-bb[:,2:]/2,g[:2]-g[2:]/2)
 hi=np.minimum(bb[:,:2]+bb[:,2:]/2,g[:2]+g[2:]/2)
 inter=np.maximum(hi-lo,0).prod(-1)
 return inter/(bb[:,2:].prod(-1)+g[2:].prod()-inter+1e-9)


def main():
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 install_controls(job['controls'])
 cfg=YAMLConfig(job['config'],device='cuda:0',use_amp=False)
 cfg.yaml_cfg['HGNetv2']['pretrained']=False
 cfg.yaml_cfg['val_dataloader']['num_workers']=0;cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
 model=cfg.model.cuda().eval()
 checkpoint=Path(job['output_dir'])/'best.pth'
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model.load_state_dict(state['ema']['module'],strict=True);del state
 for p in model.parameters():p.requires_grad_(False)
 annotation=Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
 ann=json.loads(annotation.read_text());mapping={r['id']:int(r['name']) for r in ann['categories']}
 ds=cfg.val_dataloader.dataset;cache={};handles=[]
 for name,module in [('feature',model.encoder.hooa_attention[0]),('quadrant',model.decoder.dec_quad_head[-1]),
                      ('position',model.decoder.dec_toothpos_head[-1]),('dentition',model.decoder.dec_dent_head[-1])]:
  handles.append(module.register_forward_hook(lambda m,i,o,key=name:cache.__setitem__(key,o)))
 records=[]
 check_sample=check_logits=None
 for iid,code in TARGETS:
  sample,target=ds[list(ds.ids).index(iid)]
  sample=sample.as_subclass(torch.Tensor).unsqueeze(0).cuda().requires_grad_(True)
  cache.clear();out=model(sample)
  gt=next(g for g in ann['annotations'] if g['image_id']==iid and mapping[g['category_id']]==code)
  im=next(i for i in ann['images'] if i['id']==iid)
  x,y,w,h=gt['bbox'];g=np.array([x+w/2,y+h/2,w,h])/np.array([im['width'],im['height'],im['width'],im['height']])
  bb=out['pred_boxes'][0].detach().cpu().numpy();iou=target_iou(bb,g)
  confidence=out['pred_logits'][0,:,gt['category_id']].detach().sigmoid().cpu().numpy()
  confidence[iou<.50]=-1;query=int(confidence.argmax())
  assert confidence[query]>=.30
  probs=out['pred_logits'][0,query].detach().sigmoid()
  assert int(probs.argmax())==gt['category_id'],(iid,code,int(probs.argmax()))
  feature=cache['feature']
  saved={};heads={}
  H,W=feature.shape[-2:]
  xx=(np.arange(W)+.5)/W;yy=(np.arange(H)+.5)/H
  in_box=(xx[None,:]>=x/im['width'])&(xx[None,:]<=(x+w)/im['width'])&\
         (yy[:,None]>=y/im['height'])&(yy[:,None]<=(y+h)/im['height'])
  for name,label in [('quadrant',code//10-1),('position',code%10-1),('dentition',0 if code>=50 else 1)]:
   logits=cache[name][0,query];prob=logits.softmax(-1)
   competitor=logits.detach().clone();competitor[label]=-torch.inf;other=int(competitor.argmax())
   head={'target_index':label,'predicted_index':int(logits.argmax()),'competitor_index':other,
         'probabilities':prob.detach().cpu().tolist(),'target_probability':float(prob[label].detach()),
         'target_logit':float(logits[label].detach()),'target_minus_competitor_logit':float((logits[label]-logits[other]).detach()),
         'attribution_metrics':{}}
   assert head['predicted_index']==label,(iid,name,head)
   for target_name,scalar in [('logit',logits[label]),('margin',logits[label]-logits[other])]:
    grad=torch.autograd.grad(scalar,feature,retain_graph=True)[0]
    raw_maps={'gradcam':(feature*grad.mean((2,3),keepdim=True)).sum(1).relu()[0],
              'hirescam':(feature*grad).sum(1).relu()[0]}
    for method,map_tensor in raw_maps.items():
     values=map_tensor.detach().cpu().numpy();key=f'{name}_{target_name}_{method}'
     assert np.isfinite(values).all() and values.max()>0,key
     saved[key]=values
     peak=np.unravel_index(int(values.argmax()),values.shape)
     head['attribution_metrics'][key]={'positive_mass_inside_GT_box':float(values[in_box].sum()/values.sum()),
          'GT_box_grid_fraction':float(in_box.mean()),'maximum_inside_GT_box':bool(in_box[peak]),
          'raw_maximum':float(values.max()),'raw_positive_sum':float(values.sum()),'gradient_norm':float(grad.norm())}
   heads[name]=head
  path=HERE/f'factor_head_attributions_image_{iid}_FDI_{code}.npz'
  np.savez_compressed(path,**saved)
  record={'image_id':iid,'FDI':code,'query_index':query,'FDI_confidence':float(confidence[query]),
          'query_IoU':float(iou[query]),'image_record':im,'target_annotation':gt,
          'heads':heads,'maps':str(path),'feature_shape':list(feature.shape)}
  records.append(record)
  print('HEAD_CAM',json.dumps({'image_id':iid,'FDI':code,'query':query,
                'heads':{n:{'p':h['target_probability'],'mass_in_GT':h['attribution_metrics'][n+'_margin_hirescam']['positive_mass_inside_GT_box']} for n,h in heads.items()}}),flush=True)
  check_sample=sample.detach().clone().requires_grad_(True);check_logits=out['pred_logits'].detach().clone()
  del feature,out,sample,scalar,grad,raw_maps
 for handle in handles:handle.remove()
 plain=model(check_sample)
 err=float((plain['pred_logits']-check_logits).abs().max());assert err==0,err
 report={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'config':job['config'],'seed':2,'prototype_temperature':.50,'device':'GPU0 FP32, no AMP, TF32 disabled',
    'feature_layer':'encoder.hooa_attention.0 output','feature_coordinate_frame':'entire resized image, no GT masking during attribution',
    'query_selection':'highest correct-FDI sigmoid score among queries with IoU>=0.50 to the stated GT box; require score>=0.30',
    'scalar_targets':{'logit':'ground-truth factor component logit','margin':'ground-truth component logit minus the largest other component logit'},
    'methods':{'gradcam':'ReLU(sum_c A_c * spatial_mean(d scalar/d A_c))',
               'hirescam':'ReLU(sum_c A_c * (d scalar/d A_c))'},
    'interpretation':'Head-specific gradient attributions, not spatial attention outputs or a proof of module efficacy. No architectural faithfulness guarantee is assumed for this Transformer detector.',
    'gradient_scope':'The native encoder top-k query-content initialization includes detach; gradients describe remaining differentiable paths, not complete input-perturbation effects. Conditional input occlusion is provided separately.',
    'passive_hook_prediction_max_difference':err,'records':records}
 (HERE/'factor_head_attribution_provenance.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
