"""CPU-only passive capture from the final Mobile seed-2 EMA checkpoint."""
from pathlib import Path
from hashlib import sha256
import argparse
import importlib.util
import json
import sys
import time
import numpy as np

DEST=Path(__file__).resolve().parent
TRAIN=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
BASE=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/scripts/export_mechanism_evidence.py')
spec=importlib.util.spec_from_file_location('capture_geometry',BASE)
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)

def main():
    p=argparse.ArgumentParser();p.add_argument('--image-id',type=int,required=True);p.add_argument('--limit',type=int,default=220);a=p.parse_args()
    job=json.loads((TRAIN/'jobs/mobile_proto_k4_tau0p50_seed2.json').read_text())
    sys.path.insert(0,job['implementation']);sys.path.insert(0,str(TRAIN))
    import torch
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls
    from scipy.optimize import linear_sum_assignment
    torch.set_num_threads(4)
    install_controls(job['controls'])
    cfg=YAMLConfig(job['config'],device='cpu',use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['val_dataloader']['total_batch_size']=1
    cfg.yaml_cfg['val_dataloader']['num_workers']=0
    model=cfg.model.cpu().eval()
    ckpt=Path(job['output_dir'])/'best.pth'
    checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
    state=checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    model.load_state_dict(state,strict=True);del checkpoint,state
    ann_path=Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']);annotations=json.loads(ann_path.read_text())
    cats=sorted(annotations['categories'],key=lambda c:c['id'])
    codes=np.array([int(str(c.get('fdi',c['name'])).split()[0]) for c in cats]);category_ids=np.array([c['id'] for c in cats])
    assert len(codes)==52
    gt_by={i['id']:[] for i in annotations['images']}
    for row in annotations['annotations']:gt_by[row['image_id']].append(row)
    lookup={c['id']:i for i,c in enumerate(cats)}
    modules=dict(model.named_modules());cmfa_names=[n for n,m in modules.items() if m.__class__.__name__=='AdaptiveInterleaveFusion'];hooa_names=[n for n,m in modules.items() if m.__class__.__name__=='HighOrderOmniAttention']
    assert len(cmfa_names)==4 and len(hooa_names)==3
    cmfa_name,hooa_name=cmfa_names[0],hooa_names[0];cmfa,hooa=modules[cmfa_name],modules[hooa_name];decoder=model.decoder
    proto=torch.nn.functional.normalize(decoder.tooth_prototypes.detach(),dim=-1).cpu().numpy()
    train_path=Path(cfg.yaml_cfg['train_dataloader']['dataset']['ann_file']);train_ann=json.loads(train_path.read_text())
    supports=np.array([sum(r['category_id']==cid for r in train_ann['annotations']) for cid in category_ids])
    assert proto.shape==(52,4,256);assert list(codes[supports==0])==[18,28,38,48]
    np.savez_compressed(DEST/'prototype_bank.npz',normalized_prototypes=proto,fdi_codes=codes,train_support=supports)
    cache={};capture_on=False;handles=[]
    def store(name,tensor):
        if capture_on:cache[name]=tensor.detach()
    def pre_cmfa(module,inputs):
        if capture_on:cache['_cmfa_a']=inputs[0].detach();cache['_cmfa_b']=inputs[1].detach()
    def out_cmfa(module,inputs,output):
        if not capture_on:return
        ga,gb=cache['_cmfa_gate'].split(module.channels,1);xa,xb=cache['_cmfa_a'],cache['_cmfa_b']
        cross=torch.cat([xa+xb*gb,xb+xa*ga],1)
        actual=output.detach();reconstructed=module.out_proj(cross*cache['_cmfa_spatial'])
        error=float((reconstructed-actual).abs().max());assert error<2e-5
        cache['CMFA_input_A']=xa.float().square().mean(1).sqrt();cache['CMFA_input_B']=xb.float().square().mean(1).sqrt();cache['CMFA_output']=actual.float().square().mean(1).sqrt()
        cache['CMFA_gA']=ga.flatten(1);cache['CMFA_gB']=gb.flatten(1);cache['CMFA_Gs']=cache['_cmfa_spatial'].squeeze(1);cache['_cmfa_reconstruction_error']=error
    def pre_hooa(module,inputs):
        if capture_on:cache['_hooa_input']=inputs[0].detach()
    def pre_project(module,inputs):
        if capture_on:
            xx=inputs[0].detach();channels=xx.shape[1]//3
            cache['HOOA_second_order']=xx[:,channels:2*channels].float().square().mean(1).sqrt()
            error=float((xx[:,channels:2*channels]-cache['_hooa_local']*cache['_hooa_context'].sigmoid()).abs().max());assert error<2e-5
            cache['_hooa_interaction_error']=error
    def out_hooa(module,inputs,output):
        if not capture_on:return
        xx=cache['_hooa_input'];proj=cache['_hooa_project'];cg=cache['_hooa_channel'];sg=cache['_hooa_spatial']
        actual=output.detach();alpha=.30*module.res_scale.sigmoid();reconstructed=xx+alpha*proj*cg*sg
        error=float((reconstructed-actual).abs().max());assert error<2e-5
        for src,name in [('_hooa_local','HOOA_local'),('_hooa_context','HOOA_context'),('_hooa_edge','HOOA_edge')]:cache[name]=cache[src].float().square().mean(1).sqrt()
        cache['HOOA_input']=xx.float().square().mean(1).sqrt();cache['HOOA_output']=actual.float().square().mean(1).sqrt();cache['HOOA_channel_gate']=cg.flatten(1);cache['HOOA_spatial_gate']=sg.squeeze(1)
        cache['_hooa_reconstruction_error']=error;cache['_hooa_alpha']=float(alpha)
    handles.extend([cmfa.register_forward_pre_hook(pre_cmfa),cmfa.register_forward_hook(out_cmfa),hooa.register_forward_pre_hook(pre_hooa),hooa.register_forward_hook(out_hooa),hooa.out_proj.register_forward_pre_hook(pre_project)])
    for module,name in [(cmfa.channel_gate,'_cmfa_gate'),(cmfa.spatial_gate,'_cmfa_spatial'),(hooa.local_branch,'_hooa_local'),(hooa.context_branch,'_hooa_context'),(hooa.edge_branch,'_hooa_edge'),(hooa.out_proj,'_hooa_project'),(hooa.channel_gate,'_hooa_channel'),(hooa.spatial_gate,'_hooa_spatial')]:handles.append(module.register_forward_hook(lambda m,i,o,key=name:store(key,o)))
    factor_cache={}
    for name,layer in [('quadrant',decoder.dec_quad_head[-1]),('position',decoder.dec_toothpos_head[-1]),('dentition',decoder.dec_dent_head[-1])]:handles.append(layer.register_forward_hook(lambda m,i,o,key=name:factor_cache.__setitem__(key,o.detach().numpy())))
    matrices={'quadrant':np.zeros((8,8),np.int64),'position':np.zeros((8,8),np.int64),'dentition':np.zeros((2,2),np.int64)}
    matched=[];checks=[];seen=[];captured=[];selected_cpu_logits=None
    original=np.load(Path(job['output_dir'])/'evaluation/pre_topk_candidates.npz');original_idx={int(i):j for j,i in enumerate(original['image_ids'])}
    start=time.time()
    for index,(samples,targets) in enumerate(cfg.val_dataloader):
        iid=int(targets[0]['image_id']);capture_on=iid==a.image_id;cache.clear();seen.append(iid)
        with torch.inference_mode():outputs=model(samples.cpu())
        scores=outputs['pred_logits'].sigmoid().max(-1).values.detach().numpy()[0];bb=outputs['pred_boxes'].detach().numpy()[0]
        replay=outputs['pred_logits'].detach().numpy()[0];old_index=original_idx[iid]
        # Near-tied encoder top-k candidates can permute query indices between
        # native GPU batches and CPU batch=1; compare candidate boxes after a
        # one-to-one alignment, not raw query-array position.
        old_boxes=original['boxes_cxcywh'][old_index];distance=np.abs(bb[:,None,:]-old_boxes[None,:,:]).sum(-1);ra,rb=linear_sum_assignment(distance)
        aligned_box_error=np.abs(bb[ra]-old_boxes[rb]);aligned_logit_error=np.abs(replay[ra]-original['logits'][old_index][rb])
        diagnostic={'image_id':iid,'unaligned_logit_max_difference':float(np.max(np.abs(replay-original['logits'][old_index]))),'aligned_box_max_difference':float(aligned_box_error.max()),'aligned_box_median_difference':float(np.median(aligned_box_error)),'aligned_logit_max_difference':float(aligned_logit_error.max()),'aligned_logit_median_difference':float(np.median(aligned_logit_error))}
        checks.append(diagnostic);assert diagnostic['aligned_box_median_difference']<.005,diagnostic
        if index==0:print('CPU_FIRST_DIAGNOSTIC',diagnostic,'input_shape',list(samples.shape),flush=True)
        gg=gt_by[iid];w,h=map(float,targets[0]['orig_size'].tolist());box=bb.copy();box[:,:2]-=box[:,2:]/2;box*=np.array([w,h,w,h]);keep=np.flatnonzero(scores>=.30)
        if len(keep) and gg:
            overlap=geometry.overlap(box[keep],np.array([r['bbox'] for r in gg]));rr,cc=linear_sum_assignment(-overlap)
            for r,c in zip(rr,cc):
                if overlap[r,c]<.50:continue
                query=int(keep[r]);fdi=int(codes[lookup[gg[c]['category_id']]]);truths={'quadrant':fdi//10-1,'position':fdi%10-1,'dentition':0 if fdi>=50 else 1}
                row={'image_id':iid,'query':query,'FDI':fdi,'IoU':float(overlap[r,c])}
                for name in matrices:
                    pred=int(factor_cache[name][0,query].argmax());matrices[name][truths[name],pred]+=1;row[name+'_true']=truths[name];row[name+'_pred']=pred
                matched.append(row)
        if capture_on:
            selected_cpu_logits=replay.copy()
            data={k:v[0].numpy().astype(np.float32) for k,v in cache.items() if not k.startswith('_')}
            for k,v in cache.items():
                if isinstance(v,(float,int)):data[k]=np.array(v)
            path=DEST/f'internal_tensors_image_{iid}.npz';np.savez_compressed(path,**data)
            captured.append({'image_id':iid,'CMFA_module':cmfa_name,'HOOA_module':hooa_name,'shape_of_input':list(samples.shape),'raw_tensor_file':str(path),'CMFA_reconstruction_max_abs_error':cache['_cmfa_reconstruction_error'],'HOOA_reconstruction_max_abs_error':cache['_hooa_reconstruction_error'],'HOOA_interaction_max_abs_error':cache['_hooa_interaction_error'],'HOOA_residual_coefficient':cache['_hooa_alpha']})
        if (index+1)%10==0 or capture_on:print('CPU_CAPTURE',index+1,iid,'matches',len(matched),'seconds',round(time.time()-start,1),flush=True)
        if index+1>=a.limit:break
    for handle in handles:handle.remove()
    assert captured,('Selected case not captured',a.image_id)
    # A second unhooked forward on the selected image verifies hook invariance.
    for samples,targets in cfg.val_dataloader:
        if int(targets[0]['image_id'])!=a.image_id:continue
        with torch.inference_mode():plain=model(samples.cpu())
        hook_invariance_error=float(np.max(np.abs(plain['pred_logits'].numpy()[0]-selected_cpu_logits)))
        assert hook_invariance_error<1e-6,hook_invariance_error
        break
    np.savez_compressed(DEST/'factor_confusion_raw.npz',**matrices)
    (DEST/'factor_matched_queries.json').write_text(json.dumps(matched,indent=2))
    prov={'configuration':job['configuration'],'seed':2,'device':'CPU FP32','checkpoint':str(ckpt),'checkpoint_sha256':sha256(ckpt.read_bytes()).hexdigest(),'config_sha256':job['config_sha256'],'annotation_file':str(ann_path),'annotation_sha256':sha256(ann_path.read_bytes()).hexdigest(),'train_annotation_file':str(train_path),'train_annotation_sha256':sha256(train_path.read_bytes()).hexdigest(),'selected_cases':captured,'captured_queries_total':len(matched),'images_evaluated':seen,'dataset_image_count':len(annotations['images']),'complete_validation_split':len(seen)==len(annotations['images']),'capture_matching':'Unique ordinary queries with maximum sigmoid score >=0.30; class-agnostic maximum-total-IoU Hungarian localization and IoU>=0.50, distinct from post-top-k detector matching.','factor_labels':{'quadrant':'FDI first digit minus one','position':'FDI second digit minus one','dentition':'0 primary, 1 permanent'},'attention_display':'Channel RMS responses and all 256 channel-gate entries; raw sigmoid gates; no Grad-CAM.','CPU_vs_archived_GPU_candidate_diagnostics':checks,'CPU_query_order_note':'Native encoder top-k near-ties can reorder queries between CPU batch=1 and archived GPU batches; raw query-array differences are not output identity discrepancies. CPU forward supplies only internal response/head diagnostics, not primary AP endpoints.','passive_hook_prediction_invariance_max_abs_difference':hook_invariance_error,'prototype_temperature':.50,'prototype_count':4,'unsupported_FDI_codes':[18,28,38,48]}
    (DEST/'mechanism_provenance.json').write_text(json.dumps(prov,indent=2));print('CAPTURE_COMPLETE',len(seen),len(matched),flush=True)

if __name__=='__main__':main()
