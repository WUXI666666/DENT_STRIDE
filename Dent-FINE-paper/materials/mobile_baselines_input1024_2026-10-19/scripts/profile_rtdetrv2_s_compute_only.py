"""Measure the fixed RT-DETRv2-S graph without training or changing run status."""
from pathlib import Path
import json
import sys

from common import ROOT, OLD, ANN, run_dir, sha, write, now


def main():
    import numpy as np
    import torch
    from PIL import Image
    from calflops import calculate_flops
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source=OLD/'upstream/RT-DETR/rtdetrv2_pytorch'
    sys.path.insert(0,str(source))
    from src.core import YAMLConfig
    config=ROOT/'configs/rtdetrv2_s_phone.yml'
    config_hash=sha(config)
    cfg=YAMLConfig(str(config))
    cfg.yaml_cfg['PResNet']['pretrained']=False
    model=cfg.model.cpu().float().eval()
    checkpoint=run_dir('rtdetrv2_s')/'best.pth'
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        weights=state['ema']['module'] if state.get('ema') is not None else state['model']
        model.load_state_dict(weights,strict=True)
        del weights,state
    assert list(model.encoder.eval_spatial_size)==[1024,1024]
    assert list(model.decoder.eval_spatial_size)==[1024,1024]
    metadata=sorted(json.loads(ANN.read_text())['images'],key=lambda row:row['id'])[0]
    image_path=ANN.parent/metadata['file_name']
    with Image.open(image_path) as im:
        im=im.convert('RGB').resize((1024,1024),Image.Resampling.BILINEAR)
        image=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).unsqueeze(0).float()/255
    shapes=[]
    hook=model.register_forward_hook(lambda _m,_i,out:shapes.append({
        key:list(value.shape) for key,value in out.items() if isinstance(value,torch.Tensor)}))
    params=sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        flops,macs,_=calculate_flops(model=model,args=[image],output_as_string=False,
                                   print_detailed=False,print_results=False)
    hook.remove()
    assert shapes[-1]['pred_logits']==[1,300,52]
    assert shapes[-1]['pred_boxes']==[1,300,4]
    assert sha(config)==config_hash
    result={'family':'rtdetrv2_s','model':'Official RT-DETRv2-S (ResNet18-vd)',
            'cohort':'Mobile','num_classes':52,'num_queries':300,'input_shape':[1,3,1024,1024],
            'params':params,'Params_M':params/1e6,'GFLOPs':float(flops)/1e9,'GMACs':float(macs)/1e9,
            'tool':'calflops 0.3.2','precision':'FP32','device':'CPU',
            'form':'Native unfused evaluation-mode forward; no external postprocessing or unsupported operations',
            'profile_input':'First canonical validation photograph, square bilinear resize, RGB [0,1]',
            'image_path':str(image_path),'image_sha256':sha(image_path),'image_id':metadata['id'],
            'source':str(source),'source_config':str(config),'config_sha256':config_hash,
            'in_memory_override':{'PResNet.pretrained':False},'original_config_unchanged':True,
            'checkpoint':str(checkpoint) if checkpoint.is_file() else None,
            'checkpoint_sha256':sha(checkpoint) if checkpoint.is_file() else None,
            'training_status_note':'Compute-only measurement; partial checkpoint does not establish completed training or final AP',
            'output_shapes':shapes[-1],'at':now()}
    destination=ROOT/'measurements/rtdetrv2_s_input1024_compute.json'
    write(destination,result)
    print(json.dumps(result,indent=2))
    print('SAVED',destination)


if __name__=='__main__':main()
