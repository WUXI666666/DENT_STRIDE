from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from common import ROOT, OLD, PROJECT, ANN, run_dir, write, now, sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('family');a=ap.parse_args()
    if a.family=='caf_yolo_s':sys.path.insert(0,str(PROJECT/'third_party/CAF-YOLO'))
    if a.family=='faster_rcnn':sys.path.insert(0,'/home/wuxi/detectron2')
    import torch
    import numpy as np
    from calflops import calculate_flops
    torch.set_num_threads(4);torch.manual_seed(0)
    out=run_dir(a.family)
    summary=json.loads((out/'metrics_summary.json').read_text());checkpoint=Path(summary['checkpoint'])
    if a.family in ['deim_s','rtdetrv2_s','dome_detr_s']:
        source={'deim_s':OLD/'upstream/DEIM','rtdetrv2_s':OLD/'upstream/RT-DETR/rtdetrv2_pytorch',
                'dome_detr_s':PROJECT/'third_party/dome_detr'}[a.family]
        prefix='engine' if a.family=='deim_s' else 'src';sys.path.insert(0,str(source))
        core=importlib.import_module(prefix+'.core');cfg=core.YAMLConfig(str(ROOT/'configs'/f'{a.family}_phone.yml'))
        cfg.yaml_cfg['PResNet' if a.family=='rtdetrv2_s' else 'HGNetv2']['pretrained']=False
        model=cfg.model
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'],strict=True)
        del state
    elif a.family=='faster_rcnn':
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        cfg=get_cfg();cfg.merge_from_file(str(out/'config.yaml'));cfg.MODEL.DEVICE='cpu'
        detector=build_model(cfg);DetectionCheckpointer(detector).load(str(checkpoint))
        class Forward(torch.nn.Module):
            def __init__(self,m):super().__init__();self.model=m
            def forward(self,x):
                return self.model.inference([{'image':im,'height':1024,'width':1024} for im in x],do_postprocess=False)
        model=Forward(detector)
    else:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=state.get('ema') if state.get('ema') is not None else state['model']
        del state
    model=model.cpu().float().eval()
    params=sum(p.numel() for p in model.parameters())
    from PIL import Image
    image_metadata=sorted(json.loads(ANN.read_text())['images'],key=lambda row:row['id'])[0]
    image_path=ANN.parent/image_metadata['file_name']
    with Image.open(image_path) as image:
        image=image.convert('RGB').resize((1024,1024),Image.Resampling.BILINEAR)
        array=np.array(image,copy=True)
    x=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
    if a.family=='faster_rcnn':x=x[:,[2,1,0]]*255
    counts=[]
    hook=model.register_forward_hook(lambda _m,_i,o:counts.append(int(o['pred_logits'].shape[1]))
                                    if isinstance(o,dict) and 'pred_logits' in o else None)
    with torch.no_grad():
        flops,macs,_=calculate_flops(model=model,args=[x],output_as_string=False,print_detailed=False,print_results=False)
    hook.remove()
    result={'family':a.family,'input_shape':[1,3,1024,1024],'params':params,'Params_M':params/1e6,
            'GFLOPs':float(flops)/1e9,'GMACs':float(macs)/1e9,'tool':'calflops 0.3.2',
            'form':'Native unfused eval forward, FP32; no external postprocessing or unsupported operations',
            'profile_input':'First canonical validation image, square bilinear resize to 1024; RGB [0,1], or BGR [0,255] for Faster R-CNN',
            'image_path':str(image_path),'image_sha256':sha(image_path),'image_id':image_metadata['id'],
            'dynamic_compute_note':'Representative-image count, not an average over the cohort',
            'query_counts':counts,'checkpoint_sha256':sha(checkpoint),'at':now()}
    write(out/'compute_profile.json',result)
    model=model.cuda().eval();x=x.cuda();torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(30):model(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for _ in range(100):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();torch.cuda.synchronize();elapsed.append(start.elapsed_time(end))
    v=np.asarray(elapsed)
    benchmark={'family':a.family,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
               'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'precision':'FP32',
               'batch':1,'shape':[1,3,1024,1024],'warmup':30,'repetitions':100,
               'timing_scope':'CUDA-event native forward only; resident representative-image input; no data loading/external postprocessing',
               'image_path':str(image_path),'image_id':image_metadata['id'],
               'mean_ms':float(v.mean()),'SD_ms':float(v.std(ddof=1)),'P50_ms':float(np.percentile(v,50)),
               'P95_ms':float(np.percentile(v,95)),'images_per_second':1000/float(v.mean()),
               'peak_inference_allocated_bytes':torch.cuda.max_memory_allocated(),'at':now()}
    write(out/'inference_benchmark.json',benchmark)
    print(json.dumps({'compute':result,'benchmark':benchmark},indent=2))


if __name__=='__main__':main()
