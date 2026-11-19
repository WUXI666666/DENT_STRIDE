from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

from common import ROOT, OLD, PROJECT, run_dir, sha, write, now, disk_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('family',choices=['deim_s','rtdetrv2_s','dome_detr_s']);p.add_argument('--preflight',action='store_true');p.add_argument('--preflight-batch',type=int,default=2);p.add_argument('--integration-smoke',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    source = {'deim_s':OLD/'upstream/DEIM', 'rtdetrv2_s':OLD/'upstream/RT-DETR/rtdetrv2_pytorch',
              'dome_detr_s':PROJECT/'third_party/dome_detr'}[a.family]
    prefix='engine' if a.family=='deim_s' else 'src'
    sys.path.insert(0,str(source)); sys.path.insert(0,str(OLD/'scripts'))
    from run_official import BestOnlyStorage, export_verified_best, percentage_metrics
    core=importlib.import_module(prefix+'.core');misc=importlib.import_module(prefix+'.misc.dist_utils')
    solvers=importlib.import_module(prefix+'.solver');solver_module=importlib.import_module(prefix+'.solver.det_solver')
    os.environ['TORCH_HOME']=str(OLD/'pretrained_cache')
    misc.setup_distributed(seed=0)
    out=(ROOT/'repairs_20261020/smoke'/a.family/'seed_0') if a.integration_smoke else run_dir(a.family)
    out.mkdir(parents=True,exist_ok=True)
    os.chdir(out)  # Keep framework incidental runtime files out of the original code tree.
    config=ROOT/'configs'/f'{a.family}_phone.yml'
    cfg=core.YAMLConfig(str(config),device='cuda',use_amp=True,seed=0,output_dir=str(out),summary_dir=str(out/'summary'))
    if a.integration_smoke:
        cfg.epoches=1
        cfg.yaml_cfg['epoches']=1
        class ShortLoader:
            def __init__(self,wrapped):self.wrapped=wrapped
            def __len__(self):return 2
            def __iter__(self):
                from itertools import islice
                yield from islice(iter(self.wrapped),2)
            def __getattr__(self,name):return getattr(self.wrapped,name)
        cfg._train_dataloader=ShortLoader(cfg.train_dataloader)
    model=cfg.model
    if a.family=='dome_detr_s':
        # Native DOME ignores num_top_queries and returns Q sorted candidates.
        # Apply the declared common 300-output cap after its native score ranking.
        class Cap300(torch.nn.Module):
            def __init__(self,native):super().__init__();self.native=native
            def forward(self,*args,**kwargs):
                results=self.native(*args,**kwargs)
                return [{k:v[:300] for k,v in r.items()} for r in results]
        cfg._postprocessor=Cap300(cfg.postprocessor)
    if a.preflight:
        cfg.yaml_cfg['train_dataloader']['num_workers']=0
        cfg.yaml_cfg['train_dataloader']['total_batch_size']=a.preflight_batch
        loader=cfg.train_dataloader;loader.set_epoch(0)
        sample,target=next(iter(loader)); assert list(sample.shape)==[a.preflight_batch,3,1024,1024]
        model=model.cuda().train();criterion=cfg.criterion.cuda()
        target=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in target]
        with torch.autocast('cuda',dtype=torch.float16):
            outputs=model(sample.cuda(),targets=target);losses=criterion(outputs,target);loss=sum(losses.values())
        assert torch.isfinite(loss); loss.backward()
        model.eval()
        with torch.no_grad():results=model(sample[:1].cuda())
        assert results['pred_logits'].shape[-1]==52
        with torch.no_grad():
            candidates=cfg.postprocessor.cuda()(results,torch.tensor([[1024,1024]],device='cuda'))
        assert all(len(r['scores'])<=300 for r in candidates)
        write(ROOT/'preflight'/f'{a.family}.json',{'status':'passed','at':now(),'input_shape':list(sample.shape),
               'loss':float(loss.detach()),'output_shape':list(results['pred_logits'].shape),'checkpoint_written':False,
               'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'config_sha256':sha(config)})
        misc.cleanup();return
    if (out/'log.txt').exists() or (out/'best.pth').exists():
        raise RuntimeError('Refuse to overwrite existing training; inspect status first.')
    disk_guard()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(config)==protocol['configs'][a.family+'_phone']['sha256']
    params=sum(x.numel() for x in model.parameters())
    write(out/'run_protocol.json',{'family':a.family,'seed':0,'input':1024,'epochs':cfg.epoches,'started_at':now(),
          'source':str(source),'config':str(config),'config_sha256':sha(config),'params':params,
          'runner_sha256':sha(__file__),'integration_smoke':a.integration_smoke,
          'backbone_initialization':'Fresh native ImageNet backbone; no dental checkpoint tuning'})
    solver=solvers.TASKS[cfg.yaml_cfg['task']](cfg)
    hook=BestOnlyStorage(solver,solver_module,misc,out,0);hook.install()
    # Profiling is performed separately, without random draws or a second GPU graph in the training process.
    # RT-DETRv2 has no solver-level stats function. Do not require or inject it.
    has_stats=hasattr(solver_module,'stats')
    original_stats=getattr(solver_module,'stats',None)
    if has_stats:
        solver_module.stats=lambda _cfg:(params,{'profile':'Native-forward count is measured separately after training.'})
    started=time.monotonic()
    try:
        solver.fit();hook.restore()
        logs=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
        assert [r['epoch'] for r in logs]==list(range(cfg.epoches))
        selected=max(logs,key=lambda r:r['test_coco_eval_bbox'][0])
        assert selected['epoch']==hook.best_metrics['epoch_zero_based']
        verified=export_verified_best(solver,hook.original_evaluate,out,hook.best_metrics)
        if a.integration_smoke:
            assert ROOT/'repairs_20261020/smoke' in out.parents
            (out/'best.pth').unlink()  # This new pipeline-test weight only, after exact replay.
            write(out/'SMOKE_PASSED.json',{'status':'passed','scientific_result':False,
                  'coverage':'Native fit, two optimizer batches, full validation, best-only I/O, exact selected-checkpoint replay',
                  'retained_weight_count':0,'at':now()})
            return
        write(out/'metrics_summary.json',{'family':a.family,'seed':0,'input':[1024,1024],
              'epochs_completed':160,'best_epoch_zero_based':selected['epoch'], 'best_epoch_one_based':selected['epoch']+1,
              'metrics_fraction':verified['metrics_fraction'],'metrics_percent':verified['metrics_percentage_points'],
              'final_epoch_metrics':percentage_metrics(logs[-1]['test_coco_eval_bbox']),
              'params':params,'checkpoint':str(out/'best.pth'),'checkpoint_sha256':verified['checkpoint_sha256'],
              'elapsed_seconds':time.monotonic()-started,'peak_training_validation_cuda_bytes':torch.cuda.max_memory_allocated(),
              'native_stage1_RAM_restores':hook.stage1_loads,'completed_at':now()})
        assert list(out.rglob('*.pth'))==[out/'best.pth']
        write(out/'status.json',{'state':'training_and_best_replay_completed','at':now()})
        (out/'TRAINING_COMPLETED').write_text(now())
    except BaseException as e:
        hook.restore();write(out/'status.json',{'state':'failed','error':repr(e),'at':now(),'epoch_zero_based':hook.epoch});raise
    finally:
        if has_stats:solver_module.stats=original_stats
        misc.cleanup()


if __name__=='__main__':main()
