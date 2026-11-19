"""Probe genuine gradients from each factor logit to a spatial feature map."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
HERE=Path(__file__).resolve().parent
job=json.loads((ROOT/'jobs/mobile_proto_k4_tau0p50_seed2.json').read_text())
sys.path.insert(0,job['implementation']);sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from controlled_runtime import install_controls


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    install_controls(job['controls'])
    cfg=YAMLConfig(job['config'],device='cuda:0',use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['val_dataloader']['num_workers']=0
    cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
    model=cfg.model.cuda().eval()
    ckpt=torch.load(Path(job['output_dir'])/'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['ema']['module'],strict=True)
    del ckpt
    for parameter in model.parameters():parameter.requires_grad_(False)
    ann=json.loads(Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']).read_text())
    fdi={r['id']:int(r['name']) for r in ann['categories']}
    ds=cfg.val_dataloader.dataset
    sample,target=ds[list(ds.ids).index(148)]
    sample=sample.as_subclass(torch.Tensor).unsqueeze(0).cuda().requires_grad_(True)
    cache={};handles=[]
    for name,module in [('feature',model.encoder.hooa_attention[0]),
                        ('quadrant',model.decoder.dec_quad_head[-1]),
                        ('position',model.decoder.dec_toothpos_head[-1]),
                        ('dentition',model.decoder.dec_dent_head[-1])]:
        handles.append(module.register_forward_hook(lambda m,i,o,key=name:cache.__setitem__(key,o)))
    out=model(sample)
    code=55
    gt=next(g for g in ann['annotations'] if g['image_id']==148 and fdi[g['category_id']]==code)
    im=next(i for i in ann['images'] if i['id']==148)
    xywh=np.asarray(gt['bbox']);gtbox=np.array([xywh[0]+xywh[2]/2,xywh[1]+xywh[3]/2,xywh[2],xywh[3]])/np.array([im['width'],im['height'],im['width'],im['height']])
    bb=out['pred_boxes'][0].detach().cpu().numpy()
    lo=np.maximum(bb[:,:2]-bb[:,2:]/2,gtbox[:2]-gtbox[2:]/2)
    hi=np.minimum(bb[:,:2]+bb[:,2:]/2,gtbox[:2]+gtbox[2:]/2)
    inter=np.maximum(hi-lo,0).prod(-1);iou=inter/(bb[:,2:].prod(-1)+gtbox[2:].prod()-inter+1e-9)
    cid=gt['category_id']
    confidence=out['pred_logits'][0,:,cid].detach().sigmoid().cpu().numpy()
    confidence[iou<.5]=-1
    query=int(confidence.argmax());print('QUERY',query,'IoU',float(iou[query]),'FDI confidence',float(confidence[query]),'requires_grad',cache['feature'].requires_grad,flush=True)
    saved={}
    for name,k in [('quadrant',code//10-1),('position',code%10-1),('dentition',0 if code>=50 else 1)]:
        score=cache[name][0,query,k]
        grad=torch.autograd.grad(score,cache['feature'],retain_graph=True)[0]
        cam=(grad.mean((2,3),keepdim=True)*cache['feature']).sum(1).relu()[0]
        positive_local=(grad*cache['feature']).sum(1).relu()[0]
        print(name,'logit',float(score.detach()),'prediction',int(cache[name][0,query].argmax()),'target_probability',float(cache[name][0,query].softmax(-1)[k].detach()),'gradnorm',float(grad.norm()),'cammax',float(cam.max()),'localmax',float(positive_local.max()),flush=True)
        saved[name+'_gradcam']=cam.detach().cpu().numpy();saved[name+'_grad_times_activation']=positive_local.detach().cpu().numpy()
    for h in handles:h.remove()
    np.savez_compressed(HERE/'probe_factor_cam_image_148.npz',**saved)


if __name__=='__main__':main()
