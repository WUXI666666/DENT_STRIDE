#!/usr/bin/env python3
"""Profile independently; never alter training RNG or upstream source files."""
from __future__ import annotations
import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import torch
from calflops import calculate_flops

from prepare_comparison import ROOT, MODELS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['deim_s', 'rtdetrv2_s', 'local_plain'], required=True)
    parser.add_argument('--cohort', choices=['phone', 'dentex', 'dslr'], default='phone')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.source == 'local_plain':
        sys.path.insert(0, '/home/wuxi/Dent-FINE-paper/materials/scripts')
        from export_deim_predictions import RESET_OVERRIDES
        from engine.core import YAMLConfig, yaml_utils
        suffix = {'phone': 'phone', 'dentex': 'dentex', 'dslr': 'viewnet3_dslr_fdi52'}[args.cohort]
        updates = yaml_utils.parse_cli(RESET_OVERRIDES + ['HGNetv2.pretrained=False',
            'DFINETransformer.use_mr=False', 'DFINETransformer.use_tooth_query_embed=False',
            'DEIMCriterion.weight_dict.loss_arch=0.0'])
        cfg = YAMLConfig(f'/home/wuxi/Dent-FINE/configs/deim_dfine/deim_hgnetv2_s_{suffix}.yml', **updates)
    else:
        spec = MODELS[args.source]
        sys.path.insert(0, str(spec['code']))
        prefix = 'engine' if args.source == 'deim_s' else 'src'
        core = importlib.import_module(f'{prefix}.core')
        key = 'HGNetv2' if args.source == 'deim_s' else 'PResNet'
        cfg = core.YAMLConfig(str(ROOT / 'configs' / f'{args.source}_{args.cohort}.yml'), **{key: {'pretrained': False}})
    model = cfg.model.eval()
    schema = {n: {'shape': list(p.shape), 'numel': p.numel(), 'trainable': p.requires_grad}
              for n,p in model.named_parameters()}
    train_params = sum(p.numel() for p in model.parameters())
    deploy = copy.deepcopy(model).deploy().eval()
    flops, macs, _ = calculate_flops(model=deploy, input_shape=(1,3,640,640),
                                    output_as_string=False, print_detailed=False, print_results=False)
    result = {'source': args.source, 'cohort': args.cohort, 'params_training_total': train_params,
              'params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'params_deploy_total': sum(p.numel() for p in deploy.parameters()),
              'GMACs': float(macs)/1e9, 'GFLOPs': float(flops)/1e9,
              'profile': 'calflops, deployed model forward only, FP32, CPU, batch 1, 640x640; excludes postprocessing',
              'parameter_schema': schema}
    output = ROOT / 'profiles' / f'{args.source}_{args.cohort}.json'
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'parameter_schema'}, indent=2))


if __name__ == '__main__':
    main()
