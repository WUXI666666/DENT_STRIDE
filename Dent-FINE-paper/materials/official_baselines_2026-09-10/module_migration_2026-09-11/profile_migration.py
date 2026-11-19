#!/usr/bin/env python3
"""Independent forward-only profiles; never consume the training RNG."""
import copy
import json
import sys
import gc

import torch
from calflops import calculate_flops

from prepare_migration import HERE, ROOT, write_json
sys.path.insert(0, str(ROOT / 'upstream/DEIM'))
from engine.core import YAMLConfig
from paper_modules import install_modules


def main():
    torch.set_num_threads(4)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for task, frozen in protocol['runs'].items():
        cfg = YAMLConfig(frozen['config'], HGNetv2={'pretrained': False})
        model = install_modules(cfg, frozen['modules']).eval()
        deployed = copy.deepcopy(model).deploy().eval()
        flops, macs, _ = calculate_flops(deployed, input_shape=(1, 3, 640, 640),
                                        output_as_string=False, print_detailed=False, print_results=False)
        result = {'task': task, 'params_training_total': sum(p.numel() for p in model.parameters()),
                  'params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'params_deploy_total': sum(p.numel() for p in deployed.parameters()),
                  'GMACs': float(macs)/1e9, 'GFLOPs': float(flops)/1e9,
                  'profile': 'calflops; deployed model forward; FP32 CPU; batch 1; 640x640; excludes postprocessing (including SpatialShape)'}
        write_json(HERE / 'profiles' / f'{task}.json', result)
        print(json.dumps(result), flush=True)
        del cfg, model, deployed
        gc.collect()


if __name__ == '__main__':
    main()
