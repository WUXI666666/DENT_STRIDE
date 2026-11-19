"""Conditional input-occlusion sensitivity, including detached query-content paths.

Original encoder anchor selection/order is frozen, but query content, decoder
states, factor logits, and boxes are recomputed from every occluded input.
This is a diagnostic, not native perturbed-image detector performance.
"""
from pathlib import Path
import json
import sys
import types

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
job=json.loads((ROOT/'jobs/mobile_proto_k4_tau0p50_seed2.json').read_text())
sys.path.insert(0,job['implementation']);sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from controlled_runtime import install_controls


def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 install_controls(job['controls'])
 cfg=YAMLConfig(job['config'],device='cuda:0',use_amp=False)
 cfg.yaml_cfg['HGNetv2']['pretrained']=False
 cfg.yaml_cfg['val_dataloader']['num_workers']=0;cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
 model=cfg.model.cuda().eval();ckpt=torch.load(Path(job['output_dir'])/'best.pth',map_location='cpu',weights_only=False)
 model.load_state_dict(ckpt['ema']['module'],strict=True);del ckpt
 ds=cfg.val_dataloader.dataset;decoder=model.decoder;native=decoder._select_topk;state={'indices':None}
 captured={};handles=[]
 for name,module in [('quadrant',decoder.dec_quad_head[-1]),('position',decoder.dec_toothpos_head[-1]),('dentition',decoder.dec_dent_head[-1])]:
  handles.append(module.register_forward_hook(lambda m,i,o,key=name:captured.__setitem__(key,o.detach())))

 def fixed_topk(self,memory,logits,anchors,topk):
  if state['indices'] is None:
   assert self.query_select_method=='default'
   ids=torch.topk(logits.max(-1).values,topk,dim=-1).indices
   state['indices']=ids.detach().clone()
   expected=native(memory,logits,anchors,topk)
  else:ids=state['indices'].expand(memory.shape[0],-1);expected=None
  mem=memory.gather(1,ids[...,None].expand(-1,-1,memory.shape[-1]))
  anc=anchors.gather(1,ids[...,None].expand(-1,-1,anchors.shape[-1]))
  if expected is not None:
   assert torch.equal(mem,expected[0]) and torch.equal(anc,expected[2])
  return mem,None,anc

 decoder._select_topk=types.MethodType(fixed_topk,decoder)
 base=json.loads((HERE/'factor_head_attribution_provenance.json').read_text())
 records=[];size=128;stride=32;batch_size=8
 positions=[(y,x) for y in range(0,1024-size+1,stride) for x in range(0,1024-size+1,stride)]
 with torch.inference_mode():
  for reference in base['records']:
   iid=reference['image_id'];sample,target=ds[list(ds.ids).index(iid)]
   sample=sample.as_subclass(torch.Tensor).unsqueeze(0).cuda();state['indices']=None
   outputs=model(sample);baseline={k:v.clone() for k,v in captured.items()}
   # Select the same foreground object under the frozen anchor ordering.
   gt=reference['target_annotation'];im=reference['image_record'];x,y,w,h=gt['bbox']
   g=torch.tensor([(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']],device='cuda')
   bb=outputs['pred_boxes'][0];lo=torch.maximum(bb[:,:2]-bb[:,2:]/2,g[:2]-g[2:]/2)
   hi=torch.minimum(bb[:,:2]+bb[:,2:]/2,g[:2]+g[2:]/2);inter=(hi-lo).clamp_min(0).prod(-1)
   overlap=inter/(bb[:,2:].prod(-1)+g[2:].prod()-inter+1e-9)
   score=outputs['pred_logits'][0,:,gt['category_id']].sigmoid();score=torch.where(overlap>=.5,score,-torch.ones_like(score))
   q=int(score.argmax());assert score[q]>=.30
   frozen=model(sample)
   zero_error=float((frozen['pred_logits']-outputs['pred_logits']).abs().max());assert zero_error==0
   fill=sample.mean((2,3),keepdim=True)
   patch_results={name:[] for name in baseline};margin0={};targets={};competitors={}
   for name in baseline:
    heads=reference['heads'][name];t=heads['target_index'];c=heads['competitor_index']
    targets[name]=t;competitors[name]=c;margin0[name]=float(baseline[name][0,q,t]-baseline[name][0,q,c])
   for start in range(0,len(positions),batch_size):
    batch=sample.expand(min(batch_size,len(positions)-start),-1,-1,-1).clone()
    for at,(py,px) in enumerate(positions[start:start+batch_size]):batch[at,:,py:py+size,px:px+size]=fill[0]
    model(batch)
    for name,values in captured.items():
     margins=values[:,q,targets[name]]-values[:,q,competitors[name]]
     patch_results[name].extend((margin0[name]-margins).cpu().tolist())
   arrays={'patch_xy':np.array([(x,y) for y,x in positions])};summaries={}
   for name,values in patch_results.items():
    dense=np.zeros((1024,1024),np.float32);den=np.zeros_like(dense)
    for value,(py,px) in zip(values,positions):dense[py:py+size,px:px+size]+=value;den[py:py+size,px:px+size]+=1
    dense/=den
    arrays[name+'_patch_margin_drop']=np.array(values,dtype=np.float32)
    arrays[name+'_dense_margin_drop']=dense
    vmax=int(np.argmax(values));vy,vx=positions[vmax]
    summaries[name]={'baseline_margin':margin0[name],'target_index':targets[name],'baseline_competitor_index':competitors[name],
       'maximum_margin_drop':float(max(values)),'minimum_margin_drop':float(min(values)),
       'strongest_patch_xy':[vx,vy],'strongest_patch_center_xy_normalized':[(vx+size/2)/1024,(vy+size/2)/1024]}
   path=HERE/f'factor_occlusion_image_{iid}_FDI_{reference["FDI"]}.npz';np.savez_compressed(path,**arrays)
   records.append({'reference':reference,'query_index':q,'unoccluded_frozen_query_max_error':zero_error,'maps':str(path),'heads':summaries})
   print('OCCLUSION',iid,summaries,flush=True)
 decoder._select_topk=native
 for handle in handles:handle.remove()
 report={'checkpoint':base['checkpoint'],'checkpoint_sha256':base['checkpoint_sha256'],'seed':2,'prototype_temperature':.50,
    'input_size':[1024,1024],'occluder_size':[size,size],'stride':[stride,stride],'patches_per_image':len(positions),
    'occluder_fill':'per-image RGB channel means','scalar':'correct factor logit minus the strongest unoccluded alternative (fixed competitor)',
    'spatial_map':'average baseline-minus-occluded margin over all sliding patches covering each input pixel; signed values retained',
    'query_conditioning':'unoccluded encoder top-k anchor indices/order held fixed; content and all downstream states recomputed for every perturbed input',
    'reason':'Native query-content initialization is detached in autograd, so input occlusion complements partial gradient-path explanations.',
    'limitations':'Conditional-query sensitivity, not native detector accuracy on occluded images or a proof of clinical faithfulness. Mean-color masks may introduce artificial edges.',
    'records':records}
 (HERE/'factor_occlusion_provenance.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
