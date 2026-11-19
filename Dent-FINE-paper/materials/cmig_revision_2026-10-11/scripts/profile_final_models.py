"""Complexity measurements of the actual new seed-0 checkpoints, CPU only."""
from pathlib import Path
import json
import importlib.util
import sys

PAPER=Path('/home/wuxi/Dent-FINE-paper')
TRAIN=PAPER/'materials/cmig_best3_seeds_2026-10-08'
ROOT=PAPER/'materials/cmig_revision_2026-10-11'
spec=importlib.util.spec_from_file_location('complexity_measurement',PAPER/'materials/cmig_final_2026-10-06/extra_evaluations/evaluate_extra.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=TRAIN
module.OUT=ROOT/'model_measurements'
configuration=sys.argv[1]
job=json.loads((TRAIN/'jobs'/(configuration+'_seed0.json')).read_text())
sys.path.insert(0,job['implementation'])
sys.path.insert(0,str(TRAIN))
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
cfg=module.configure(job,device='cpu')
module.profile(job,cfg)
