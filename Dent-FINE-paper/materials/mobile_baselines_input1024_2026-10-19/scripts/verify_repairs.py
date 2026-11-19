"""Adapter regression tests and diagnostic replay; never final experimental results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from common import ROOT, ANN, write, now, coco_metrics, sha
from coco_export import canonical_predictions, assert_native_coordinates


def unit_tests():
    import torch
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.models.yolo.detect.val import DetectionValidator
    predictions={'bboxes':torch.tensor([[128.,256.,640.,768.]]),
                 'conf':torch.tensor([.914326]),'cls':torch.tensor([7.])}
    batch={'im_file':'/test/73.jpg','ori_shape':(1536,2048)}
    ids={'73.jpg':42}
    canonical=canonical_predictions(predictions,batch,ids,rtdetr=True,imgsz=1024)
    assert canonical[0]['bbox']==[256.,384.,1024.,768.]
    assert canonical[0]['image_id']==42
    assert predictions['bboxes'].tolist()==[[128.,256.,640.,768.]]
    stub=SimpleNamespace(jdict=[],args=SimpleNamespace(imgsz=1024),class_map=list(range(52)))
    RTDETRValidator.pred_to_json(stub,predictions,batch)
    assert_native_coordinates(canonical,stub.jdict)
    stub.class_map=list(range(1,53));stub.jdict=[]
    RTDETRValidator.pred_to_json(stub,predictions,batch)
    assert_native_coordinates(canonical,stub.jdict,class_map=stub.class_map)
    stub.class_map=list(range(52))
    unscaled=canonical_predictions(predictions,batch,ids,rtdetr=False)
    assert unscaled[0]['bbox']!=canonical[0]['bbox']  # The old bug is caught.
    stub.jdict=[]
    DetectionValidator.pred_to_json(stub,predictions,batch)
    assert_native_coordinates(unscaled,stub.jdict)
    old_tensor=torch.cat([predictions['bboxes'],predictions['conf'][:,None],predictions['cls'][:,None]],dim=1)
    assert canonical_predictions(old_tensor,'/test/73.jpg',ids)==unscaled
    empty={k:v[:0] for k,v in predictions.items()}
    assert canonical_predictions(empty,batch,ids,rtdetr=True)==[]
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO()
    gt.dataset={'info':{},'images':[{'id':42,'width':2048,'height':1536}],
                'categories':[{'id':7,'name':'test'}],
                'annotations':[{'id':1,'image_id':42,'category_id':7,'bbox':[256,384,1024,768],
                                'area':1024*768,'iscrowd':0}]}
    gt.createIndex()
    dt=gt.loadRes(canonical)
    ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    assert ev.stats[0]>.999999
    bad=gt.loadRes(unscaled)
    broken=COCOeval(gt,bad,'bbox');broken.evaluate();broken.accumulate();broken.summarize()
    assert broken.stats[0]==0
    write(ROOT/'repairs_20261020/unit_tests.json',{'status':'passed','at':now(),
          'cases':['RT-DETR anisotropic original-image scaling','native rounded-export geometry equivalence',
                   'native YOLO geometry preserved','CAF tensor geometry preserved','input not mutated',
                   'empty output','exact synthetic COCO AP=1; old export AP=0'],
          'correct_AP':float(ev.stats[0]),'old_incorrect_AP':float(broken.stats[0]),
          'coco_export_sha256':sha(ROOT/'scripts/coco_export.py')})


def replay():
    import torch
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.utils import SETTINGS
    torch.set_num_threads(4)
    for key in ['wandb','clearml','comet','mlflow','neptune','raytune','dvc']:
        if key in SETTINGS:dict.__setitem__(SETTINGS,key,False)
    mapping={Path(x['file_name']).name:int(x['id']) for x in json.loads(ANN.read_text())['images']}
    observed={}
    class DiagnosticValidator(RTDETRValidator):
        def pred_to_json(self,predn,pbatch):
            rows=canonical_predictions(predn,pbatch,mapping,rtdetr=True,imgsz=self.args.imgsz)
            saved=self.jdict;self.jdict=[]
            try:
                super().pred_to_json(predn,pbatch)
                assert_native_coordinates(rows,self.jdict,class_map=self.class_map)
            finally:self.jdict=saved
            self.jdict.extend(rows)
        def finalize_metrics(self):
            super().finalize_metrics()
            self.diagnostic_coco=coco_metrics(self.jdict,ROOT/'repairs_20261020/verification/rtdetr_l_replay/evaluation')
            observed['coco']=self.diagnostic_coco
            print('DIAGNOSTIC_REPLAY_COCO',self.diagnostic_coco,flush=True)
    checkpoint=ROOT/'runs/rtdetr_l/seed_0/weights/best.pt'
    model=RTDETR(str(checkpoint))
    model.val(validator=DiagnosticValidator,data=str(ROOT/'configs/mobile_fdi52.yaml'),imgsz=1024,
              batch=8,device=0,half=False,workers=4,rect=False,conf=.001,max_det=300,iou=.7,
              plots=False,save_json=True,project=str(ROOT/'repairs_20261020/verification'),
              name='rtdetr_l_replay',exist_ok=True)
    values=observed['coco']
    assert values['AP']>.05, 'Geometry correction should recover nontrivial AP for this trained diagnostic checkpoint'
    write(ROOT/'repairs_20261020/diagnostic_replay.json',{'status':'passed','at':now(),
          'scientific_result':False,'reason':'Checkpoint selection in the old run was invalid; this is only an exporter regression check',
          'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'metrics_fraction':values,
          'geometry_checked_against_native_for_every_detection':True,'validation_image_count':len(mapping)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    unit_tests()
    if args.replay:replay()
