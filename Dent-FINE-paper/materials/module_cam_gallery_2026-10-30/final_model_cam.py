"""Complement controlled comparisons with the final Mobile EMA's mechanisms."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'cmig_selected_configs_multiseed_2026-10-11'
TARGETS=[(115,64),(77,11),(182,46),(44,51),(118,42),(24,62)]


def capture():
    job=json.loads((ROOT/'jobs/mobile_proto_k4_tau0p50_seed2.json').read_text())
    sys.path.insert(0,job['implementation']);sys.path.insert(0,str(ROOT))
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls
    install_controls(job['controls']);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=YAMLConfig(job['config'],device='cpu',use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['val_dataloader']['num_workers']=0;cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
    model=cfg.model.cpu().eval();checkpoint=Path(job['output_dir'])/'best.pth'
    st=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(st['ema']['module'],strict=True);del st
    for p in model.parameters():p.requires_grad_(False)
    ds=cfg.val_dataloader.dataset;ann=json.loads(Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']).read_text())
    ims={r['id']:r for r in ann['images']};mapping={r['id']:int(r['name']) for r in ann['categories']}
    cache={};handles=[];modules=dict(model.named_modules())
    handles.append(model.encoder.register_forward_hook(lambda m,i,o:cache.__setitem__('finest',o[0])))
    for name in ['encoder.top_down_fusions.1','encoder.hooa_attention.0.edge_branch']:
        handles.append(modules[name].register_forward_hook(lambda m,i,o,key=name:cache.__setitem__(key,o)))
    records=[]
    for iid,code in TARGETS:
        sample,_=ds[list(ds.ids).index(iid)];sample=sample.as_subclass(torch.Tensor).unsqueeze(0).requires_grad_(True)
        cache.clear();out=model(sample);im=ims[iid]
        gt=next(g for g in ann['annotations'] if g['image_id']==iid and mapping[g['category_id']]==code)
        x,y,w,h=gt['bbox'];W,H=im['width'],im['height'];cid=gt['category_id']
        bb=out['pred_boxes'][0].detach().numpy();norm=np.array([(x+w/2)/W,(y+h/2)/H,w/W,h/H])
        lo=np.maximum(bb[:,:2]-bb[:,2:]/2,norm[:2]-norm[2:]/2)
        hi=np.minimum(bb[:,:2]+bb[:,2:]/2,norm[:2]+norm[2:]/2)
        inter=np.maximum(hi-lo,0).prod(-1);iou=inter/(bb[:,2:].prod(-1)+norm[2:].prod()-inter+1e-9)
        labels=out['pred_logits'][0].detach().argmax(-1).numpy()
        score=out['pred_logits'][0,:,cid].detach().sigmoid().numpy();score[(iou<.5)|(labels!=cid)]=-1
        q=int(score.argmax())
        if score[q]<.3:
            print('NO_QUALIFIED_TARGET',iid,code,flush=True);continue
        nativebb=bb[q].copy();nativebb[:2]-=nativebb[2:]/2;nativebb*=np.array([W,H,W,H])
        case={'case_id':f'FinalMobile_image{iid}_FDI{code}','group':'FinalMobile','cohort':'Mobile',
            'image_id':iid,'image_record':im,'image_file':str(Path(cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'])/im['file_name']),
            'target_annotation':gt,'target_FDI':code,'after':{'prediction':{'bbox':nativebb.tolist(),'category_id':cid,'FDI':code,'score':float(score[q])}}}
        arrays={};scalars={}
        def cams(scalar,layer,key):
            grad=torch.autograd.grad(scalar,layer,retain_graph=True)[0]
            arrays[key+'_gradcam']=(grad.mean((2,3),keepdim=True)*layer).sum(1).relu()[0].detach().numpy()
            arrays[key+'_hirescam']=(grad*layer).sum(1).relu()[0].detach().numpy()
            scalars[key]=float(scalar.detach())
        target=out['pred_logits'][0,q,cid]
        cams(target,cache['finest'],'Detection')
        cams(target,cache['encoder.top_down_fusions.1'],'CMFA')
        cams(target,cache['encoder.hooa_attention.0.edge_branch'],'HOOA_edge')
        prototype_score=out['pred_proto_logits'][0,q,cid]
        cams(prototype_score,cache['finest'],'Prototype')
        cosine=torch.einsum('d,ckd->ck',F.normalize(out['query_feats'][0,q],dim=-1),F.normalize(model.decoder.tooth_prototypes,dim=-1))
        arrays['prototype_cosine_allclasses']=cosine.detach().numpy()
        arrays['prototype_class_logits']=out['pred_proto_logits'][0,q].detach().numpy()
        path=HERE/(case['case_id']+'.npz');np.savez_compressed(path,**arrays)
        record={'case':case,'case_id':case['case_id'],'raw_maps':str(path),'query':q,'IoU':float(iou[q]),
            'scalars':scalars,'prototype_classifier_correct':int(out['pred_proto_logits'][0,q].argmax())==cid,
            'prototype_fusion_weight':model.decoder.prototype_fusion_weight}
        records.append(record);print('FINAL_CAPTURE',case['case_id'],'query',q,'prototype_correct',record['prototype_classifier_correct'],flush=True)
        last_sample=sample.detach();last_q=q;last_cid=cid;last_scalar=scalars['Detection']
    for h in handles:h.remove()
    with torch.inference_mode():test=model(last_sample)
    err=abs(float(test['pred_logits'][0,last_q,last_cid])-last_scalar);assert err<1e-5
    report={'job':job,'checkpoint':str(checkpoint),'seed':2,'temperature':.50,'records':records,
        'passive_hook_logit_error':err,'device':'CPU FP32','weights_modified':False,
        'scope':'Mechanism examples from one final Mobile EMA; these are not module-isolation comparisons. Native detached content initialization is not traversed by gradients.'}
    (HERE/'final_mobile_capture.json').write_text(json.dumps(report,indent=2))


def render():
    import render_gallery as r
    data=json.loads((HERE/'final_mobile_capture.json').read_text());scored=[]
    for rec in data['records']:
        c=rec['case'];r.CAPTURE['FinalMobile','after']=r.CAPTURE.get(('FinalMobile','after'),{})
        r.CAPTURE['FinalMobile','after'][c['case_id']]=rec
        m=np.load(rec['raw_maps'])
        metrics={key:r.measure(m[key+'_hirescam'],c) for key in ['Detection','CMFA','HOOA_edge','Prototype']}
        rec['spatial_diagnostics']=metrics
        rank=(rec['prototype_classifier_correct'],sum(v['peak_inside_GT'] for v in metrics.values()),
            min(v['enrichment'] for v in metrics.values()))
        scored.append((rank,rec))
    scored.sort(key=lambda z:z[0],reverse=True)
    def plate(records,name,method):
        H=.45+len(records)*1.48;fig=r.plt.figure(figsize=(8.35,H));ph=1.1/H
        keys=['Detection','CMFA','HOOA_edge','Prototype']
        for row,rec in enumerate(records):
            c=rec['case'];bottom=1-(.35+row*1.48+1.12)/H
            for col in range(5):
                ax=fig.add_axes([.036+col*.192,bottom,.173,ph])
                if col==0:r.prediction(ax,c,'after')
                else:r.cam(ax,c,'after',keys[col-1]+'_'+method)
                if row==0:ax.set_title(['Prediction','Detection CAM','CMFA CAM','HOOA edge CAM','Prototype CAM'][col],fontsize=9,pad=6)
            fig.text(.004,bottom+ph+.005,f'({chr(97+row)})',ha='left',va='bottom')
        caption=('Module-specific attributions from the final DENT-STRIDE-Mobile EMA (seed 2, K=4, temperature 0.50). '
            'All columns within a row use the same trained model, image and native ordinary query. Predictions are orange; '
            'teal dashed boxes are ground truth. Detection CAM uses the finest encoder output, CMFA CAM uses the finer '
            'top-down fusion output, and HOOA edge CAM uses the high-pass branch tensor. These three explain the same '
            'reference-FDI detection logit. Prototype CAM explains the actual auxiliary prototype class score at the '
            'finest encoder output. Each positive map is normalized by its own full-image maximum before cropping, with '
            'bilinear interpolation for display. These maps show relative spatial attribution, not comparable contribution '
            'strength across modules. Prototype inference fusion weight is zero. Gradients follow differentiable paths '
            'but do not traverse detached top-k content initialization. This is a mechanism illustration, not a comparison '
            'between independently retrained models; efficacy requires the accompanying controlled ablations. '
            f'The method is {"Grad-CAM" if method=="gradcam" else "positive elementwise gradient-times-activation (HiResCAM form)"}. Cases: '+
            '; '.join(f"({chr(97+i)}) image {v['case']['image_id']}, FDI {v['case']['target_FDI']}" for i,v in enumerate(records))+'.')
        r.save(fig,name,caption,{'final_checkpoint':data['checkpoint'],'cases':[v['case_id'] for v in records],'method':method})
    for _,rec in scored:plate([rec],rec['case_id']+'_modules_HiResCAM','hirescam')
    plate([x[1] for x in scored],'FinalMobile_all_cases_HiResCAM','hirescam')
    plate([x[1] for x in scored[:3]],'Recommended_FinalMobile_modules_HiResCAM','hirescam')
    plate([x[1] for x in scored[:3]],'Recommended_FinalMobile_modules_GradCAM','gradcam')
    data['recommendations']=[x[1]['case_id'] for x in scored[:3]]
    (HERE/'final_mobile_capture.json').write_text(json.dumps(data,indent=2))
    (HERE/'final_mobile_captions.json').write_text(json.dumps(r.CAPTIONS,indent=2))
    (HERE/'final_mobile_figure_manifest.json').write_text(json.dumps(r.MANIFEST,indent=2))


if __name__=='__main__':
    if '--render' in sys.argv:render()
    else:capture()
