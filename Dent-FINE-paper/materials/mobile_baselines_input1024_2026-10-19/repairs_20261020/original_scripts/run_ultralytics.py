from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import types

from common import ROOT, OLD, PROJECT, run_dir, write, sha, now, disk_guard, ANN, coco_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('family',choices=['rtdetr_l','yolov8s','yolo11s','caf_yolo_s']);p.add_argument('--preflight',action='store_true');p.add_argument('--integration-smoke',action='store_true');a=p.parse_args()
    if a.family=='caf_yolo_s':sys.path.insert(0,str(PROJECT/'third_party/CAF-YOLO'))
    import torch
    import numpy as np
    if not hasattr(np,'trapz'):np.trapz=np.trapezoid
    import ultralytics
    from ultralytics import YOLO, RTDETR
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils import SETTINGS
    from ultralytics.utils import torch_utils
    for integration in ['wandb','clearml','comet','mlflow','neptune','raytune','dvc']:
        if integration in SETTINGS:dict.__setitem__(SETTINGS,integration,False)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    out=(ROOT/'smoke'/a.family/'seed_0') if a.integration_smoke else run_dir(a.family)
    out.mkdir(parents=True,exist_ok=True)
    weights={'yolov8s':PROJECT/'outputs/viewnet3_dslr_baseline_comparison/weights/yolov8s.pt',
             'yolo11s':PROJECT/'outputs/viewnet3_dslr_baseline_comparison/weights/yolo11s.pt',
             'rtdetr_l':PROJECT/'weights/rtdetr-l.pt',
             'caf_yolo_s':PROJECT/'outputs/viewnet3_dslr_baseline_comparison/weights/yolov8s.pt'}[a.family]
    if a.family=='caf_yolo_s':
        alias=ROOT/'configs/yolov8s-CAFMAttention.yaml'
        if not alias.exists():alias.symlink_to(PROJECT/'third_party/CAF-YOLO/ultralytics/cfg/models/v8/yolov8-CAFMAttention.yaml')
        model=YOLO(str(alias));model.load(str(weights))
    else:model=(RTDETR if a.family=='rtdetr_l' else YOLO)(str(weights))
    payload=json.loads(ANN.read_text())
    image_ids={Path(x['file_name']).name:int(x['id']) for x in payload['images']}
    unwrap=getattr(torch_utils,'unwrap_model',getattr(torch_utils,'de_parallel',lambda x:x))
    base=RTDETRTrainer if a.family=='rtdetr_l' else DetectionTrainer
    args=dict(data=str(ROOT/'configs/mobile_fdi52.yaml'),epochs=160,imgsz=1024,batch=8,workers=4,
              device=0,project=str(out.parent),name=out.name,exist_ok=True,pretrained=True,optimizer='AdamW',
              lr0=.0002,momentum=.9,weight_decay=.0001,warmup_epochs=3.,warmup_bias_lr=0.,cos_lr=False,
              patience=160,seed=0,deterministic=True,amp=a.family!='caf_yolo_s',cache=False,rect=False,
              save=True,save_period=-1,plots=False,flipud=0.,fliplr=0.,mosaic=0.,mixup=0.,
              save_json=True,conf=.001,iou=.7,max_det=300)
    if a.integration_smoke:args.update(epochs=1,patience=1,workers=0)

    def capture(self, predn, batch_or_filename):
        filename=batch_or_filename['im_file'] if isinstance(batch_or_filename,dict) else batch_or_filename
        iid=image_ids[Path(filename).name]
        if isinstance(predn,dict):
            bb=predn['bboxes'].detach().cpu().tolist();ss=predn['conf'].detach().cpu().tolist();cc=predn['cls'].detach().cpu().tolist()
        else:
            rows=predn.detach().cpu().tolist();bb=[x[:4] for x in rows];ss=[x[4] for x in rows];cc=[x[5] for x in rows]
        for (x1,y1,x2,y2),score,cls in zip(bb,ss,cc):
            assert 0<=int(cls)<52
            self.jdict.append({'image_id':iid,'category_id':int(cls),
                               'bbox':[x1,y1,x2-x1,y2-y1],'score':float(score)})

    class ControlledTrainer(base):
        def get_dataloader(self,*pos,**kw):
            loader=super().get_dataloader(*pos,**kw)
            mode=kw.get('mode',pos[3] if len(pos)>3 else 'train')
            if a.integration_smoke and mode=='train':
                class ShortLoader:
                    def __init__(self,wrapped):self.wrapped=wrapped
                    def __len__(self):return 2
                    def __iter__(self):
                        from itertools import islice
                        yield from islice(iter(self.wrapped),2)
                    def __getattr__(self,name):return getattr(self.wrapped,name)
                return ShortLoader(loader)
            return loader

        def build_dataset(self,img_path,mode='train',batch=None):
            data=super().build_dataset(img_path,mode,batch)
            if mode=='val' and hasattr(data,'rect'):
                data.rect=False
                if hasattr(data,'batch_shapes'):data.batch_shapes=None
                data.transforms=data.build_transforms(hyp=self.args)
            return data

        def validate(self):
            disk_guard()
            self.validator.pred_to_json=types.MethodType(capture,self.validator)
            self.validator.args.save_json=True
            original_amp=self.amp
            self.amp=False  # The common COCO selection endpoint is FP32 for every framework.
            try:metrics=self.validator(self)
            finally:self.amp=original_amp
            values=coco_metrics(self.validator.jdict)
            fitness=values['AP']
            if not np.isfinite(fitness):raise FloatingPointError('Nonfinite COCO AP')
            self._common_metrics=values
            self._common_predictions=deepcopy(self.validator.jdict)
            self._common_epoch=int(self.epoch)
            metrics['metrics/mAP50(B)']=values['AP50']
            metrics['metrics/mAP50-95(B)']=values['AP']
            metrics.pop('fitness',None)
            if self.best_fitness is None or fitness>self.best_fitness:self.best_fitness=fitness
            with (out/'validation_coco.jsonl').open('a') as f:
                f.write(json.dumps({'epoch_zero_based':self.epoch,'metrics_fraction':values})+'\n')
            write(out/'status.json',{'state':'training','epoch_zero_based':self.epoch,'AP':fitness*100,
                                      'best_AP':self.best_fitness*100,'at':now()})
            return metrics,fitness

        def save_model(self):
            if not hasattr(self,'_common_metrics'):return
            ap=self._common_metrics['AP']
            if ap<=getattr(self,'_saved_common_best',float('-inf')):return
            disk_guard()
            selected=deepcopy(unwrap(self.ema.ema)).cpu().float()
            if hasattr(selected,'criterion'):selected.criterion=None
            state={'epoch':self.epoch,'best_fitness':ap,'model':None,'ema':selected,
                   'updates':self.ema.updates,'optimizer':None,'train_args':vars(self.args),
                   'version':ultralytics.__version__,'date':now(),
                   'comparison_metadata':{'seed':0,'epoch_zero_based':self.epoch,'metrics_fraction':self._common_metrics}}
            self.wdir.mkdir(parents=True,exist_ok=True)
            pending=self.wdir/'best.pending.pt';torch.save(state,pending);pending.replace(self.best)
            self._saved_common_best=ap
            write(out/'best_metrics.json',{'epoch_zero_based':self.epoch,'epoch_one_based':self.epoch+1,
                  'metrics_fraction':self._common_metrics,'selection':'maximum FP32 canonical COCO AP, earliest tie'})
            from common import coco_metrics as export
            export(self._common_predictions,out/'evaluation')
            print('RETAINED_BEST_COCO_AP',self.epoch,ap,flush=True)

        def final_eval(self):
            # Avoid stock conversion of selected FP32 weights to FP16 and avoid a last.pt.
            # The evaluation predictions retained by save_model come from the selected EMA.
            return

    write(out/('preflight_protocol.json' if a.preflight else 'run_protocol.json'),
          {'family':a.family,'args':args,'initialization':str(weights),'initialization_sha256':sha(weights),
           'package':ultralytics.__file__,'version':ultralytics.__version__,'started_at':now(),
           'checkpoint_policy':'Global best canonical COCO AP only, FP32 EMA; no last/periodic files'})
    if a.preflight:
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import RTDETRDetectionModel
        target_class=RTDETRDetectionModel if a.family=='rtdetr_l' else type(model.model)
        net=target_class(cfg=deepcopy(model.model.yaml),nc=52,verbose=False)
        net.load(model.model)
        net=net.cuda().train()
        net.args=get_cfg(overrides=args)
        net.nc=52
        net.names={x['id']:x['name'] for x in payload['categories']}
        batch={'img':torch.zeros(2,3,1024,1024,device='cuda'),
               'batch_idx':torch.tensor([0,1],device='cuda'),
               'cls':torch.tensor([[0.],[1.]],device='cuda'),
               'bboxes':torch.tensor([[.5,.5,.1,.1],[.5,.5,.1,.1]],device='cuda')}
        with torch.autocast('cuda',dtype=torch.float16,enabled=args['amp']):loss=net(batch)
        loss=loss[0] if isinstance(loss,(tuple,list)) else loss
        loss=sum(loss.values()) if isinstance(loss,dict) else loss.sum()
        assert torch.isfinite(loss);loss.backward()
        write(ROOT/'preflight'/f'{a.family}.json',{'status':'passed','at':now(),'input_shape':[2,3,1024,1024],
              'loss':float(loss.detach()),'checkpoint_written':False,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        return
    if (out/'validation_coco.jsonl').exists() or (out/'weights/best.pt').exists():
        raise RuntimeError('Refuse to mix with an existing training run.')
    disk_guard();started=time.monotonic()
    model.train(trainer=ControlledTrainer,**args)
    rows=[json.loads(x) for x in (out/'validation_coco.jsonl').read_text().splitlines() if x.strip()]
    assert [x['epoch_zero_based'] for x in rows]==list(range(1 if a.integration_smoke else 160))
    selected=json.loads((out/'best_metrics.json').read_text())
    best=max(rows,key=lambda x:x['metrics_fraction']['AP'])
    assert best['epoch_zero_based']==selected['epoch_zero_based']
    checkpoint=out/'weights/best.pt'
    values=selected['metrics_fraction']
    if a.integration_smoke:
        assert checkpoint.is_file()
        assert ROOT/'smoke' in checkpoint.parents
        write(out/'SMOKE_PASSED.json',{'status':'passed','scientific_result':False,
              'coverage':'Two optimizer batches, full canonical COCO validation, global-best storage and API reload',
              'retained_weight_count':0,'at':now()})
        checkpoint.unlink()  # Only this newly generated, verified pipeline-test artifact.
        return
    write(out/'metrics_summary.json',{'family':a.family,'seed':0,'input':[1024,1024],'epochs_completed':160,
          'metrics_fraction':values,'metrics_percent':{k:v*100 if v>=0 else None for k,v in values.items()},
          'best_epoch_zero_based':selected['epoch_zero_based'],'best_epoch_one_based':selected['epoch_one_based'],
          'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
          'elapsed_seconds':time.monotonic()-started,'peak_training_validation_cuda_bytes':torch.cuda.max_memory_allocated(),
          'selected_prediction_source':'Canonical COCO FP32 validation of the exact retained EMA', 'completed_at':now()})
    assert list(out.rglob('*.pt'))==[checkpoint]
    (out/'TRAINING_COMPLETED').write_text(now())


if __name__=='__main__':main()
