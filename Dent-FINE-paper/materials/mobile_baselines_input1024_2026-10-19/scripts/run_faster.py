from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from common import ROOT, PROJECT, TRAIN_ANN, ANN, run_dir, sha, write, now, disk_guard


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    sys.path.insert(0,'/home/wuxi/detectron2')
    source=PROJECT/'tools/dental/train_faster_rcnn_viewnet3_dslr_fdi52.py'
    spec=importlib.util.spec_from_file_location('native_faster_runner',source)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    out=run_dir('faster_rcnn')
    original=native.build_config
    def build_config(*args,**kwargs):
        cfg,weights=original(*args,**kwargs)
        cfg=cfg.clone();cfg.defrost();cfg.SOLVER.AMP.ENABLED=False;cfg.freeze()
        return cfg,weights
    native.build_config=build_config
    original_make=native.make_trainer_class
    def make_trainer_class():
        import torch
        from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
        from detectron2.data import transforms as T
        from detectron2.engine import hooks
        Parent=original_make()
        class DiskGuard(hooks.HookBase):
            def before_step(self):
                if self.trainer.iter%100==0:
                    disk_guard()
                    write(out/'status.json',{'state':'training','iteration':self.trainer.iter,
                          'max_iterations':self.trainer.max_iter,'at':now()})
        class SquareTrainer(Parent):
            @classmethod
            def build_train_loader(cls,cfg):
                mapper=DatasetMapper(cfg,is_train=True,augmentations=[T.Resize((1024,1024))])
                return build_detection_train_loader(cfg,mapper=mapper)
            @classmethod
            def build_test_loader(cls,cfg,dataset_name):
                mapper=DatasetMapper(cfg,is_train=False,augmentations=[T.Resize((1024,1024))])
                return build_detection_test_loader(cfg,dataset_name,mapper=mapper)
            def build_hooks(self):return super().build_hooks()+[DiskGuard()]
        return SquareTrainer
    native.make_trainer_class=make_trainer_class
    weights=Path('/home/wuxi/.torch/iopath_cache/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl')
    sys.argv=[str(source),'--train-ann',str(TRAIN_ANN),'--val-ann',str(ANN),
        '--train-images',str(TRAIN_ANN.parent),'--val-images',str(ANN.parent),'--output-dir',str(out),
        '--num-classes','52','--epochs','160','--batch-size','4','--num-workers','4',
        '--min-size','1024','--max-size','1024','--base-lr','0.005','--seed','0','--gpu-id','0','--weights',str(weights)]
    if a.preflight:
        import torch
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.data import DatasetMapper
        from detectron2.data.datasets import load_coco_json
        from detectron2.data import transforms as T
        args=native.parse_args();native.resolve_and_validate_args(args)
        records=load_coco_json(str(TRAIN_ANN),str(TRAIN_ANN.parent))
        cfg,_=native.build_config(args,'mobile_preflight_train','mobile_preflight_val',math.ceil(len(records)/4))
        net=build_model(cfg).train();DetectionCheckpointer(net).load(str(weights))
        mapper=DatasetMapper(cfg,is_train=True,augmentations=[T.Resize((1024,1024))])
        batch=[mapper(x) for x in records[:4]]
        assert all(list(x['image'].shape)==[3,1024,1024] for x in batch)
        from detectron2.utils.events import EventStorage
        with EventStorage():
            loss=sum(net(batch).values());assert torch.isfinite(loss);loss.backward()
        write(ROOT/'preflight/faster_rcnn.json',{'status':'passed','at':now(),'input_shape':[4,3,1024,1024],
              'loss':float(loss.detach()),'checkpoint_written':False,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        return
    if (out/'metrics.json').exists() or (out/'model_best.pth').exists():raise RuntimeError('Existing run; refuse overwrite.')
    # Native main refuses a nonempty output; queue stdout is kept in ROOT/logs.
    started=time.monotonic();disk_guard();native.main()
    data=json.loads((out/'coco_eval/coco_metrics.json').read_text())
    checkpoint=out/'model_best.pth'
    vals=data['metrics_all']
    write(out/'metrics_summary.json',{'family':'faster_rcnn','seed':0,'input':[1024,1024],'epochs_completed':160,
          'metrics_fraction':vals,'metrics_percent':{k:v*100 if v>=0 else None for k,v in vals.items()},
          'best_epoch_one_based':data['training']['best_epoch_by_validation_AP'],
          'best_epoch_zero_based':data['training']['best_epoch_by_validation_AP']-1,
          'params':data['num_parameters'],'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
          'elapsed_seconds':time.monotonic()-started,'resize':'Exact square in both train/test DatasetMapper',
          'completed_at':now()})
    (out/'TRAINING_COMPLETED').write_text(now())


if __name__=='__main__':main()
