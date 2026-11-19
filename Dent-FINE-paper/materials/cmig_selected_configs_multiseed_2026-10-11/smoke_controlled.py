"""Discarded real-batch forward/backward checks before each formal run."""
import argparse
from pathlib import Path
import json
import sys
import random
import numpy as np

def main():
    a=argparse.ArgumentParser();a.add_argument('job');args=a.parse_args()
    job=json.loads(Path(args.job).read_text());sys.path.insert(0,job['implementation'])
    import torch
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls,validate_modules
    torch.set_num_threads(4)
    module_tests=validate_modules()
    install_controls(job['controls'])
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    cfg=YAMLConfig(job['config'],device='cuda',use_amp=True)
    model=cfg.model.cuda();criterion=cfg.criterion.cuda();optimizer=cfg.optimizer;scaler=cfg.scaler
    train=cfg.train_dataloader;val=cfg.val_dataloader
    assert train.batch_size==8 and val.batch_size==16
    train.set_epoch(4)
    dataset=train.dataset
    idx=sorted(range(len(dataset)),key=lambda i:len(dataset.coco.imgToAnns.get(dataset.ids[i],[])),reverse=True)[:8]
    items=[dataset[i] for i in idx]
    random.seed(1)
    samples,targets=train.collate_fn(items)
    samples=samples.cuda();targets=[{k:v.cuda() for k,v in t.items()} for t in targets]
    for t in targets:
        assert len(t['boxes'])==len(t['labels'])
        assert torch.isfinite(t['boxes']).all()
        if len(t['labels']):assert t['labels'].min()>=0 and t['labels'].max()<cfg.yaml_cfg['num_classes']
    torch.cuda.reset_peak_memory_stats();model.train()
    for attempt in range(12):
        optimizer.zero_grad()
        with torch.autocast('cuda'): outputs=model(samples,targets=targets)
        with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=4,step=0,global_step=4*len(train),epoch_step=len(train))
        loss=sum(losses.values());assert torch.isfinite(loss), 'Nonfinite training loss'
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scaler.step(optimizer);scaler.update()
        if torch.isfinite(norm):break
    else:raise RuntimeError('No finite gradients after AMP scale adaptation')
    keys=list(losses)
    for name in ['quad','tooth','dent','proto','nwd']:
        expected=name in cfg.yaml_cfg['DEIMCriterion']['losses']
        actual=any(k=='loss_'+name or k.startswith('loss_'+name+'_') for k in keys)
        assert expected==actual,(name,expected,actual)
    if job['cohort']=='Mobile':
        assert ('loss_arch' in losses)==cfg.yaml_cfg['DEIMCriterion']['use_arch_order_loss']
    info={'input_shape':list(samples.shape),'finite_loss':float(loss.detach()),'finite_gradient_norm':float(norm),'amp_scale_adaptation_attempts':attempt,'active_loss_keys':keys,'peak_training_allocated_bytes':torch.cuda.max_memory_allocated(),'params_total':sum(p.numel() for p in model.parameters())}
    del outputs,losses,loss,samples,targets,items
    optimizer.zero_grad(set_to_none=True)
    model.eval();val.set_epoch(0)
    samples,targets=val.collate_fn([val.dataset[i] for i in range(16)])
    assert list(samples.shape)==[16,3,job['expected_input'],job['expected_input']]
    with torch.no_grad():
        out=model(samples.cuda());pred=cfg.postprocessor.cuda()(out,torch.stack([t['orig_size'] for t in targets]).cuda())
    assert len(pred)==16 and all(torch.isfinite(p['scores']).all() and torch.isfinite(p['boxes']).all() for p in pred)
    info.update(passed=True,job_id=job['id'],config_sha256=job['config_sha256'],controls=job['controls'],module_tests=module_tests,preflight_updates_discarded=True)
    output=Path(job['output_dir']);output.mkdir(parents=True,exist_ok=True)
    (output/'preflight.json').write_text(json.dumps(info,indent=2))
    print('PREFLIGHT_PASSED',job['id'],flush=True)

if __name__=='__main__':main()
