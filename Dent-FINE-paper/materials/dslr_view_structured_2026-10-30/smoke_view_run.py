"""Discarded full-model batch checks, before any formal initialization."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('--report-dir')
    args=parser.parse_args()
    job=json.loads(Path(args.job).read_text());sys.path.insert(0,job['implementation'])
    from view_identity_runtime import install_view_identity_runtime
    install_view_identity_runtime(job['runtime_settings'])
    from engine.core import YAMLConfig
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    cfg=YAMLConfig(job['config'],device='cuda',use_amp=True)
    model=cfg.model.cuda();criterion=cfg.criterion.cuda();optimizer=cfg.optimizer;scaler=cfg.scaler
    train=cfg.train_dataloader;assert train.batch_size==8
    train.set_epoch(4);ds=train.dataset;ids=[];seen=set()
    for idx,iid in enumerate(ds.ids):
        view=int(ds.coco.imgs[iid]['view_id'])
        if view not in seen:ids.append(idx);seen.add(view)
        if len(seen)==5:break
    for i in range(len(ds)):
        if len(ids)==8:break
        if i not in ids:ids.append(i)
    batch=[ds[i] for i in ids];samples,targets=train.collate_fn(batch)
    samples=samples.cuda();targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
    views=torch.cat([t['view_id'] for t in targets]);assert set(views.tolist())==set(range(5))
    model.train();torch.cuda.reset_peak_memory_stats()
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):out=model(samples,targets=targets)
        with torch.autocast('cuda',enabled=False):losses=criterion(out,targets,epoch=4,step=0,global_step=4*len(train),epoch_step=len(train))
        loss=sum(losses.values());assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        if torch.isfinite(norm):
            break
        scaler.step(optimizer);scaler.update()
    else:raise RuntimeError('AMP did not reach finite gradients')
    for key in ['quad','tooth','dent']:
        assert 'loss_'+key in losses and torch.isfinite(losses['loss_'+key])
    proto_grad=None
    if job['view_prototype']:
        bank=model.decoder.tooth_prototypes
        assert tuple(bank.shape)==(5,52,4,256)
        assert out['pred_proto_valid_mask'].all()
        assert 'loss_proto' in losses and losses['loss_proto'].item()>0
        proto_grad=bank.grad.detach().abs().flatten(1).sum(1).cpu().tolist();assert all(v>0 for v in proto_grad)
    else:assert 'loss_proto' not in losses
    scaler.step(optimizer);scaler.update()
    cfg.ema.to('cuda').update(model)
    info={'job_id':job['id'],'passed':True,'config_sha256':job['config_sha256'],
        'input_shape':list(samples.shape),'known_views':views.tolist(),'active_losses':{k:float(v.detach()) for k,v in losses.items()},
        'gradient_norm':float(norm),'AMP_scale_attempts':attempt,'prototype_view_gradient_L1':proto_grad,
        'params_total':sum(p.numel() for p in model.parameters()),'peak_training_bytes':torch.cuda.max_memory_allocated(),
        'preflight_updates_discarded':True}
    del loss,losses,out,samples,targets,batch
    optimizer.zero_grad(set_to_none=True);model.eval();val=cfg.val_dataloader;assert val.batch_size==16
    samples,targets=val.collate_fn([val.dataset[i] for i in range(16)])
    assert list(samples.shape)==[16,3,640,640]
    with torch.no_grad():
        out=model(samples.cuda())
        pred=cfg.postprocessor.cuda()(out,torch.stack([t['orig_size'] for t in targets]).cuda())
    assert len(pred)==16 and all(torch.isfinite(p['scores']).all() for p in pred)
    if job['view_prototype']:assert not out['pred_proto_valid_mask'].any()
    output=Path(args.report_dir or job['output_dir']);output.mkdir(parents=True,exist_ok=True)
    (output/'preflight.json').write_text(json.dumps(info,indent=2))
    print('PREFLIGHT_PASSED',job['id'],info['input_shape'],info['params_total'],flush=True)


if __name__=='__main__':main()
