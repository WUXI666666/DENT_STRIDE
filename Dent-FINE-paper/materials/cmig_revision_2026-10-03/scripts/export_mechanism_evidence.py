"""Capture genuine internal tensors and factor predictions from a frozen EMA model."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import sys
import numpy as np

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
TRAIN = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')


def overlap(a, b):
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.clip(hi-lo, 0, None).prod(-1)
    union = a[:, 2:].prod(-1)[:, None] + b[:, 2:].prod(-1)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-ids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    job = json.loads((TRAIN/'jobs/mobile_1024_full_seed0.json').read_text())
    sys.path.insert(0, job['implementation'])
    sys.path.insert(0, str(TRAIN))
    import torch
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls
    from scipy.optimize import linear_sum_assignment
    torch.set_num_threads(4)
    install_controls(job['controls'])
    cfg = YAMLConfig(job['config'], device='cuda', use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.cuda().eval()
    ckpt = Path(job['output_dir'])/'best.pth'
    checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['ema']['module'], strict=True)
    del checkpoint
    post = cfg.postprocessor.cuda().eval()
    annotations = json.loads(Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']).read_text())
    cats = sorted(annotations['categories'], key=lambda c:c['id'])
    codes = np.array([int(str(c.get('fdi', c['name'])).split()[0]) for c in cats])
    category_ids = np.array([c['id'] for c in cats])
    assert len(codes) == 52
    # Match the implementation's explicit clinical FDI map.
    assert np.array_equal(codes, np.array(list(range(11,19))+list(range(21,29))+list(range(31,39))+list(range(41,49))+list(range(51,56))+list(range(61,66))+list(range(71,76))+list(range(81,86))))
    gt_by = {i['id']:[] for i in annotations['images']}
    for row in annotations['annotations']: gt_by[row['image_id']].append(row)
    lookup = {c['id']:i for i,c in enumerate(cats)}
    selected = set(args.image_ids)
    modules = dict(model.named_modules())
    cmfa_names = [n for n,m in modules.items() if m.__class__.__name__ == 'AdaptiveInterleaveFusion']
    hooa_names = [n for n,m in modules.items() if m.__class__.__name__ == 'HighOrderOmniAttention']
    assert len(cmfa_names)==4 and len(hooa_names)==3
    cmfa_name, hooa_name = cmfa_names[0], hooa_names[0]
    cmfa, hooa = modules[cmfa_name], modules[hooa_name]
    decoder = model.decoder
    proto = torch.nn.functional.normalize(decoder.tooth_prototypes.detach(), dim=-1).cpu().numpy()
    dest = ROOT/'figures/mechanism'
    dest.mkdir(parents=True, exist_ok=True)
    train_ann = json.loads(Path(cfg.yaml_cfg['train_dataloader']['dataset']['ann_file']).read_text())
    supports = np.array([sum(r['category_id']==cid for r in train_ann['annotations']) for cid in category_ids])
    np.savez_compressed(dest/'prototype_bank.npz', normalized_prototypes=proto, fdi_codes=codes, train_support=supports)
    captured = {}
    batch = {'ids':[], 'indices':[], 'cache':{}}
    handles=[]

    def subset(tensor):
        return tensor[batch['indices']].detach()

    def store(name, tensor, rms=False):
        if not batch['indices']: return
        val=subset(tensor)
        if rms: val=val.float().square().mean(dim=1).sqrt()
        batch['cache'][name]=val

    def pre_cmfa(module, inputs):
        if batch['indices']:
            batch['cache']['_cmfa_a']=subset(inputs[0]); batch['cache']['_cmfa_b']=subset(inputs[1])

    def out_cmfa(module, inputs, output):
        if not batch['indices']: return
        cache=batch['cache']; ga,gb=cache['_cmfa_gate'].split(module.channels,1)
        a,b=cache['_cmfa_a'],cache['_cmfa_b']
        cross=torch.cat([a+b*gb,b+a*ga],1)
        actual=subset(output)
        reconstructed=module.out_proj(cross*cache['_cmfa_spatial'])
        error=float((reconstructed-actual).abs().max())
        assert error < 2e-5, ('CMFA capture does not reconstruct the executed forward',error)
        cache['CMFA_input_A']=a.float().square().mean(1).sqrt()
        cache['CMFA_input_B']=b.float().square().mean(1).sqrt()
        cache['CMFA_output']=actual.float().square().mean(1).sqrt()
        cache['CMFA_gA']=ga.flatten(1);cache['CMFA_gB']=gb.flatten(1)
        cache['CMFA_Gs']=cache['_cmfa_spatial'].squeeze(1)
        cache['_cmfa_reconstruction_error']=error

    def pre_hooa(module, inputs):
        if batch['indices']: batch['cache']['_hooa_input']=subset(inputs[0])

    def pre_project(module, inputs):
        if batch['indices']:
            x=subset(inputs[0]); channels=x.shape[1]//3
            cache=batch['cache'];cache['HOOA_second_order']=x[:,channels:2*channels].float().square().mean(1).sqrt()
            error=float((x[:,channels:2*channels]-cache['_hooa_local']*cache['_hooa_context'].sigmoid()).abs().max())
            assert error < 2e-5
            cache['_hooa_interaction_error']=error

    def out_hooa(module, inputs, output):
        if not batch['indices']: return
        cache=batch['cache'];x=cache['_hooa_input'];proj=cache['_hooa_project'];cg=cache['_hooa_channel'];sg=cache['_hooa_spatial']
        actual=subset(output);alpha=.30*module.res_scale.sigmoid()
        reconstructed=x+alpha*proj*cg*sg
        error=float((reconstructed-actual).abs().max())
        assert error < 2e-5, ('HOOA capture does not reconstruct the executed forward',error)
        for src,name in [('_hooa_local','HOOA_local'),('_hooa_context','HOOA_context'),('_hooa_edge','HOOA_edge')]:
            cache[name]=cache[src].float().square().mean(1).sqrt()
        cache['HOOA_input']=x.float().square().mean(1).sqrt()
        cache['HOOA_output']=actual.float().square().mean(1).sqrt()
        cache['HOOA_channel_gate']=cg.flatten(1);cache['HOOA_spatial_gate']=sg.squeeze(1)
        cache['_hooa_reconstruction_error']=error;cache['_hooa_alpha']=float(alpha)

    handles.extend([cmfa.register_forward_pre_hook(pre_cmfa),cmfa.register_forward_hook(out_cmfa),hooa.register_forward_pre_hook(pre_hooa),hooa.register_forward_hook(out_hooa),hooa.out_proj.register_forward_pre_hook(pre_project)])
    for module,name in [(cmfa.channel_gate,'_cmfa_gate'),(cmfa.spatial_gate,'_cmfa_spatial'),(hooa.local_branch,'_hooa_local'),(hooa.context_branch,'_hooa_context'),(hooa.edge_branch,'_hooa_edge'),(hooa.out_proj,'_hooa_project'),(hooa.channel_gate,'_hooa_channel'),(hooa.spatial_gate,'_hooa_spatial')]:
        handles.append(module.register_forward_hook(lambda m,i,o,key=name:store(key,o)))
    factor_cache={}
    for name,layer in [('quadrant',decoder.dec_quad_head[-1]),('position',decoder.dec_toothpos_head[-1]),('dentition',decoder.dec_dent_head[-1])]:
        handles.append(layer.register_forward_hook(lambda m,i,o,key=name:factor_cache.__setitem__(key,o.detach().cpu().numpy())))
    matrices={'quadrant':np.zeros((8,8),np.int64),'position':np.zeros((8,8),np.int64),'dentition':np.zeros((2,2),np.int64)}
    matched=[];prediction_check=[]
    original=np.load(Path(job['output_dir'])/'evaluation/pre_topk_candidates.npz')
    original_idx={int(iid):i for i,iid in enumerate(original['image_ids'])}
    for samples,targets in cfg.val_dataloader:
        ids=[int(t['image_id']) for t in targets]
        batch['ids']=ids;batch['indices']=[i for i,iid in enumerate(ids) if iid in selected];batch['cache']={}
        with torch.inference_mode():outputs=model(samples.cuda())
        scores=outputs['pred_logits'].sigmoid().max(-1).values.detach().cpu().numpy()
        bb=outputs['pred_boxes'].detach().cpu().numpy()
        # Capturing tensors must leave the evaluated predictions unchanged.
        replay=outputs['pred_logits'].detach().cpu().numpy()
        for b,iid in enumerate(ids):
            o=original_idx[iid]
            error=float(np.max(np.abs(replay[b]-original['logits'][o])))
            assert error < 5e-4, ('Hooks changed predictions',iid,error)
            prediction_check.append(error)
            gg=gt_by[iid]
            w,h=map(float,targets[b]['orig_size'].tolist())
            box=bb[b].copy();box[:,:2]-=box[:,2:]/2;box*=np.array([w,h,w,h])
            keep=np.flatnonzero(scores[b]>=.30)
            if len(keep) and gg:
                mat=overlap(box[keep],np.array([r['bbox'] for r in gg]))
                rr,cc=linear_sum_assignment(-mat)
                for r,c in zip(rr,cc):
                    if mat[r,c]<.50:continue
                    query=int(keep[r]);cid=lookup[gg[c]['category_id']];fdi=int(codes[cid])
                    truths={'quadrant':fdi//10-1,'position':fdi%10-1,'dentition':0 if fdi>=50 else 1}
                    row={'image_id':iid,'query':query,'FDI':fdi,'IoU':float(mat[r,c])}
                    for name in matrices:
                        pred=int(factor_cache[name][b,query].argmax());matrices[name][truths[name],pred]+=1
                        row[name+'_true']=truths[name];row[name+'_pred']=pred
                    matched.append(row)
        for ix,b in enumerate(batch['indices']):
            iid=ids[b]
            data={k:v[ix].cpu().float().numpy() for k,v in batch['cache'].items() if not k.startswith('_')}
            for k,v in batch['cache'].items():
                if isinstance(v,(float,int)):data[k]=np.array(v)
            np.savez_compressed(dest/f'internal_tensors_image_{iid}.npz',**data)
            captured[iid]={'image_id':iid,'CMFA_module':cmfa_name,'HOOA_module':hooa_name,'shape_of_input':list(samples.shape),'raw_tensor_file':str(dest/f'internal_tensors_image_{iid}.npz'),'CMFA_reconstruction_max_abs_error':batch['cache']['_cmfa_reconstruction_error'],'HOOA_reconstruction_max_abs_error':batch['cache']['_hooa_reconstruction_error'],'HOOA_interaction_max_abs_error':batch['cache']['_hooa_interaction_error'],'HOOA_residual_coefficient':batch['cache']['_hooa_alpha']}
        print('CAPTURE_BATCH',ids,'matches',len(matched),flush=True)
    for handle in handles:handle.remove()
    assert selected==set(captured)
    np.savez_compressed(dest/'factor_confusion_raw.npz',**matrices)
    (dest/'factor_matched_queries.json').write_text(json.dumps(matched,indent=2))
    provenance={'configuration':job['configuration'],'seed':0,'checkpoint':str(ckpt),'checkpoint_sha256':sha256(ckpt.read_bytes()).hexdigest(),'config_sha256':job['config_sha256'],'selected_cases':list(captured.values()),'captured_queries_total':len(matched),'capture_matching':'Unique ordinary queries with maximum sigmoid score >=0.30; class-agnostic maximum-total-IoU Hungarian localization and IoU>=0.50. This is separate from post-top-k detection matching.','factor_labels':{'quadrant':'FDI first digit minus one','position':'FDI second digit minus one','dentition':'0 primary, 1 permanent'},'attention_display':'Response RMS across channels; channel gates retain all256 channels; spatial gates use actual scalar values, not Grad-CAM.','prediction_invariance_max_abs_error':max(prediction_check)}
    (dest/'mechanism_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('MECHANISM_CAPTURE_COMPLETE',len(matched),flush=True)


if __name__=='__main__':main()
