"""Run one native trainer with isolated ablation controls and verified best retention."""
import argparse
from pathlib import Path
import json
import os
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');args=ap.parse_args()
    job=json.loads(Path(args.job).read_text())
    sys.path.insert(0,job['implementation'])
    import torch
    import engine
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from engine.solver import det_solver
    from controlled_runtime import install_controls
    assert Path(engine.__file__).resolve().parent==Path(job['implementation'])/'engine'
    torch.set_num_threads(4)
    dist_utils.setup_distributed(0,'builtin',seed=job['seed'])
    install_controls(job['controls'])
    out=Path(job['output_dir']);out.mkdir(parents=True,exist_ok=True)
    cfg=YAMLConfig(job['config'],seed=job['seed'],device='cuda',use_amp=True,output_dir=str(out))
    current={'ap':None,'saved_ap':-1.,'stats':None,'epoch':None}
    native_train_epoch=det_solver.train_one_epoch
    def train_epoch(*a,**kw):
        current['epoch']=int(kw.get('epoch',a[7] if len(a)>7 else -1))
        return native_train_epoch(*a,**kw)
    det_solver.train_one_epoch=train_epoch
    native_evaluate=det_solver.evaluate
    def evaluate(*a,**kw):
        result=native_evaluate(*a,**kw)
        current['ap']=float(result[0]['coco_eval_bbox'][0]);current['stats']=result[0]
        return result
    det_solver.evaluate=evaluate
    native_save=dist_utils.save_on_master
    def save(state,path,*a,**kw):
        path=Path(path)
        if path.parent.resolve()==out.resolve():
            # last and stage2 are not read by these native training loops.
            # Stage1 is required for the local trainer's native phase resets.
            if path.name=='last.pth' or path.name=='best_stg2.pth' or path.name.startswith('checkpoint'):
                return
            if path.name=='best.pth':
                ap_value=current['ap']
                if ap_value is None or ap_value<=current['saved_ap']:return
                tmp=out/'best.pending.pth'
                native_save(state,tmp,*a,**kw);os.replace(tmp,path)
                current['saved_ap']=ap_value
                (out/'best_selection.json').write_text(json.dumps({'epoch_zero_based':current['epoch'],'native_state_last_epoch':int(state['last_epoch']),'AP_fraction':ap_value,'selection':'global maximum validation AP; first epoch on ties','metrics':current['stats']},indent=2))
                return
        return native_save(state,path,*a,**kw)
    dist_utils.save_on_master=save
    (out/'TRAINING_STARTED.json').write_text(json.dumps({'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':job['seed'],'started_unix':time.time(),'controls':job['controls']},indent=2))
    print('CONTROLLED_TRAINING_STARTED',job['id'],flush=True)
    TASKS[cfg.yaml_cfg['task']](cfg).fit()
    dist_utils.cleanup()

if __name__=='__main__':main()
