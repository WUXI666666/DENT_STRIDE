"""Real query-targeted CAMs and module activations, with no model/source writes."""
import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent


def get_module(model,name):
    return dict(model.named_modules())[name]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',required=True);ap.add_argument('--side',required=True)
    args=ap.parse_args()
    plan=json.loads((HERE/'gallery_plan.json').read_text())
    group=next(g for g in plan['groups'] if g['name']==args.group)
    source=group[args.side];job=source['job']
    sys.path.insert(0,job['implementation']);sys.path.insert(0,str(Path(job['config']).parents[2]))
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    install_controls(job['controls'])
    cfg=YAMLConfig(job['config'],device='cpu',use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['val_dataloader']['num_workers']=0;cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
    model=cfg.model.cpu().eval()
    state=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['ema']['module'],strict=True);del state
    for p in model.parameters():p.requires_grad_(False)
    ds=cfg.val_dataloader.dataset;cache={};handles=[]
    handles.append(model.encoder.register_forward_hook(lambda m,i,o:cache.__setitem__('feature',o[0])))
    names=[]
    if args.side=='after' and args.group=='CMFA':
        for root in ['encoder.top_down_fusions.1','encoder.bottom_up_fusions.0']:
            names += [root,root+'.channel_gate',root+'.spatial_gate']
    if args.side=='after' and args.group in ['HOOA','Prototype']:
        names += ['encoder.hooa_attention.0','encoder.hooa_attention.0.local_branch',
            'encoder.hooa_attention.0.context_branch','encoder.hooa_attention.0.edge_branch',
            'encoder.hooa_attention.0.channel_gate','encoder.hooa_attention.0.spatial_gate']
    for name in names:
        module=get_module(model,name)
        def record(m,inputs,out,key=name):
            cache[key]=out
            if key in ['encoder.hooa_attention.0','encoder.top_down_fusions.1','encoder.bottom_up_fusions.0']:
                for k,x in enumerate(inputs):cache[key+f'_input{k}']=x
        handles.append(module.register_forward_hook(record))
    reports=[]
    for case in group['cases']:
        start=time.monotonic();cache.clear()
        sample,_=ds[list(ds.ids).index(case['image_id'])]
        sample=sample.as_subclass(torch.Tensor).unsqueeze(0).requires_grad_(True)
        output=model(sample);feature=cache['feature'];cid=case['target_annotation']['category_id'];comp=case['competitor_category']
        # CPU/GPU arithmetic can reorder nearly tied encoder top-k queries.
        # Identify the archived object by its box AND its native top1 category,
        # not by assuming the archived query index survives across devices.
        im=case['image_record'];scale=np.array([im['width'],im['height'],im['width'],im['height']])
        allbb=output['pred_boxes'][0].detach().numpy().copy()
        allbb[:,:2]-=allbb[:,2:]/2;allbb*=scale
        expected=case[args.side]['prediction'];labels=output['pred_logits'][0].detach().argmax(-1).numpy()
        delta=np.max(np.abs(allbb-np.array(expected['bbox']))/scale,axis=1)
        delta[labels!=expected['category_id']]=np.inf
        q=int(delta.argmin());assert delta[q]<.001,(case['case_id'],'no corresponding native query',float(delta[q]))
        logits=output['pred_logits'][0,q];label=int(logits.argmax());bb=allbb[q]
        boxerr=float(np.max(np.abs(bb-np.array(expected['bbox']))))
        live_prediction={'bbox':bb.tolist(),'category_id':label,'score':float(logits.sigmoid().max().detach())}
        assert live_prediction['score']>=.30
        arrays={};scores={}
        def cam(scalar,key,layer=feature):
            grad=torch.autograd.grad(scalar,layer,retain_graph=True,allow_unused=False)[0]
            maps={'gradcam':(grad.mean((2,3),keepdim=True)*layer).sum(1).relu()[0],
                  'hirescam':(grad*layer).sum(1).relu()[0]}
            for method,value in maps.items():
                a=value.detach().float().numpy();assert np.isfinite(a).all()
                arrays[key+'_'+method]=a
            scores[key]=float(scalar.detach())
        cam(logits[cid],'FDI_logit')
        cam(logits[cid]-logits[comp],'FDI_margin')
        if args.side=='after':
            for name,value in cache.items():
                if name=='feature':continue
                a=value.detach().float()
                key=name.replace('.','_')
                if 'gate' in name:
                    arrays[key]=a[0].numpy()
                else:arrays[key+'_RMS']=a.square().mean(1).sqrt()[0].numpy()
            if args.group=='CMFA':
                layer=cache['encoder.top_down_fusions.1']
                cam(logits[cid]-logits[comp],'CMFA_output_margin',layer)
            if args.group=='HOOA':
                prefix='encoder.hooa_attention.0'
                for name in [prefix+'_input0',prefix,prefix+'.local_branch',prefix+'.context_branch',prefix+'.edge_branch']:
                    cam(logits[cid]-logits[comp],name.replace('.','_')+'_margin',cache[name])
                local,context=cache[prefix+'.local_branch'],cache[prefix+'.context_branch']
                arrays['HOOA_second_order_RMS']=(local*context.sigmoid()).detach().square().mean(1).sqrt()[0].numpy()
                arrays['HOOA_residual_RMS']=(cache[prefix]-cache[prefix+'_input0']).detach().square().mean(1).sqrt()[0].numpy()
                scores['HOOA_alpha']=float(.30*model.encoder.hooa_attention[0].res_scale.sigmoid())
            if args.group=='Prototype':
                feats=F.normalize(output['query_feats'][0,q],dim=-1)
                protos=F.normalize(model.decoder.tooth_prototypes,dim=-1)
                sims=torch.einsum('d,ckd->ck',feats,protos)
                actual=torch.logsumexp(sims/model.decoder.prototype_tau,dim=-1)
                np.testing.assert_allclose(actual.detach().numpy(),output['pred_proto_logits'][0,q].detach().numpy(),atol=1e-6)
                other=actual.detach().clone();other[cid]=-torch.inf;proto_comp=int(other.argmax())
                cam(actual[cid],'Prototype_logit')
                cam(actual[cid]-actual[proto_comp],'Prototype_margin')
                arrays['prototype_cosine_allclasses']=sims.detach().numpy()
                arrays['prototype_class_logits']=actual.detach().numpy()
                scores.update({'prototype_temperature':float(model.decoder.prototype_tau),
                    'prototype_competitor_category':proto_comp,'prototype_class_prediction':int(actual.argmax()),
                    'prototype_fusion_weight':float(model.decoder.prototype_fusion_weight)})
        stem=case['case_id']+'_'+args.side
        np.savez_compressed(HERE/(stem+'.npz'),**arrays)
        report={'case_id':case['case_id'],'side':args.side,'feature_shape':list(feature.shape),
            'query':q,'target_category':cid,'competitor_category':comp,'scores':scores,
            'query_top1_category':label,'archived_query':case[args.side]['query'],'live_prediction':live_prediction,
            'bbox_max_error_pixels':boxerr,'bbox_max_normalized_error':float(delta[q]),'raw_maps':str(HERE/(stem+'.npz')),
            'map_ranges':{k:[float(np.min(v)),float(np.max(v))] for k,v in arrays.items()},
            'elapsed_seconds':time.monotonic()-start}
        reports.append(report)
        print('CAPTURED',args.group,args.side,case['case_id'],f'{report["elapsed_seconds"]:.1f}s',flush=True)
        del output,feature,logits,sample
    for h in handles:h.remove()
    # Compare the last native prediction against an unhooked forward.
    check,_=ds[list(ds.ids).index(case['image_id'])]
    with torch.inference_mode():plain=model(check.as_subclass(torch.Tensor).unsqueeze(0))
    p=plain['pred_logits'][0,q]
    assert int(p.argmax())==label
    maxerr=abs(float(p[cid])-scores['FDI_logit']);assert maxerr<1e-5,maxerr
    report={'group':args.group,'side':args.side,'source':source,
        'method':'Grad-CAM and positive elementwise gradient-times-activation (HiResCAM form)',
        'device':'CPU FP32, no AMP; checkpoint EMA loaded strictly',
        'feature':'finest encoder output, shared stride-8 feature level',
        'target':'GT FDI logit or its margin against one fixed baseline competitor at the matched native query',
        'scope':'Gradients do not traverse native detached top-k query-content initialization. These are partial-path explanations, not a complete causal account or a clinical faithfulness guarantee.',
        'passive_hook_logit_error':maxerr,'records':reports}
    (HERE/f'{args.group}_{args.side}_capture.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
