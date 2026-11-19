#!/usr/bin/env python3
"""Native official DEIM training plus the frozen paper modules; best weight only."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from prepare_migration import HERE, ROOT, TASKS, sha256, write_json

sys.path.insert(0, str(ROOT / 'scripts'))
from run_official import BestOnlyStorage, export_verified_best, percentage_metrics, now


def prepare_smoke(task, frozen):
    config = yaml.safe_load(Path(frozen['config']).read_text())
    config.update(epoches=2, print_freq=1)
    for split, count in [('train', 16), ('val', 8)]:
        dataset = config[f'{split}_dataloader']['dataset']
        data = json.loads(Path(dataset['ann_file']).read_text())
        chosen = sorted(data['images'], key=lambda x: x['id'])[:count]
        ids = {x['id'] for x in chosen}
        data['images'] = chosen
        data['annotations'] = [a for a in data['annotations'] if a['image_id'] in ids]
        path = HERE / 'smoke_data' / f'{task}_{split}.json'
        write_json(path, data)
        dataset['ann_file'] = str(path)
        config[f'{split}_dataloader']['num_workers'] = 0
    config['train_dataloader']['collate_fn']['stop_epoch'] = 1
    path = HERE / 'smoke_data' / f'{task}.yml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_num_threads(4)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    frozen = protocol['runs'][args.task]
    spec = frozen['modules']
    assert sha256(frozen['config']) == frozen['config_sha256']
    source = ROOT / 'upstream/DEIM'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == protocol['upstream_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    assert sha256(ROOT / 'protocol.json') == protocol['base_protocol_sha256']
    if spec['spatial_shape']:
        assert sha256(spec['prior_file']) == spec['prior_sha256']
    if not args.smoke:
        validation = json.loads((HERE / 'preflight_validation.json').read_text())
        assert validation['passed']
        assert all(sha256(path) == digest for path, digest in validation['source_sha256'].items())
        evidence = json.loads((ROOT / 'dataset_evidence.json').read_text())[spec['cohort']]
        assert all(sha256(evidence[s]['annotation']) == evidence[s]['sha256'] for s in ('train', 'val'))
    output = HERE / 'smoke_runs' / args.task if args.smoke else Path(frozen['output'])
    if (output / 'COMPLETED').exists():
        print(f'Already completed: {output}')
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'log.txt').exists() or (output / 'best.pth').exists():
        raise RuntimeError(f'Refusing to overwrite an incomplete run: {output}')
    os.chdir(source)
    sys.path.insert(0, str(source))
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS as SOLVERS
    from paper_modules import install_modules, audit_model
    solver_module = importlib.import_module('engine.solver.det_solver')
    dist_utils.setup_distributed(seed=0)
    config = prepare_smoke(args.task, frozen) if args.smoke else Path(frozen['config'])
    cfg = YAMLConfig(str(config), seed=0, output_dir=str(output),
                     summary_dir=str(output / 'summary'), device='cuda', use_amp=True)
    model = install_modules(cfg, spec)
    baseline_profile = json.loads((ROOT / 'profiles' / f"deim_s_{spec['cohort']}.json").read_text())
    audit = audit_model(model, baseline_profile['parameter_schema'], spec)
    write_json(output / 'architecture_audit.json', audit)
    for name, module in list(sys.modules.items()):
        if name == 'engine' or name.startswith('engine.'):
            module_path = getattr(module, '__file__', None)
            if module_path:
                assert Path(module_path).resolve().is_relative_to(source.resolve()), (name, module_path)
    pretrained = Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth')
    write_json(output / 'run_protocol.json', {
        'task': args.task, 'cohort': spec['cohort'], 'seed': 0, 'gpu': args.gpu,
        'smoke_only': args.smoke, 'epochs': cfg.epoches, 'started_at': now(),
        'config': str(config), 'config_sha256': sha256(config), 'upstream_commit': protocol['upstream_commit'],
        'upstream_source_tree_clean': True, 'migration_protocol_sha256': sha256(HERE / 'protocol.json'),
        'runner_sha256': sha256(__file__), 'modules_sha256': sha256(HERE / 'paper_modules.py'),
        'pretrained_backbone': str(pretrained), 'pretrained_sha256': sha256(pretrained),
        'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'modules': spec, 'params_training_total': audit['params_training_total'],
        'baseline': frozen['baseline_metrics'], 'command': sys.argv,
        'storage': 'one best.pth; native stage-1 state kept in RAM; DualOpt anchors serialized',
    })
    solver = SOLVERS[cfg.yaml_cfg['task']](cfg)
    hook = BestOnlyStorage(solver, solver_module, dist_utils, output, 0)
    hook.install()
    started = time.monotonic()
    try:
        solver.fit()
        hook.restore()
        rows = [json.loads(line) for line in (output / 'log.txt').read_text().splitlines() if line.strip()]
        assert [r['epoch'] for r in rows] == list(range(cfg.epoches))
        best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
        assert hook.best_metrics['epoch_zero_based'] == best['epoch']
        assert abs(hook.best_ap - best['test_coco_eval_bbox'][0]) < 1e-12
        verified = export_verified_best(solver, hook.original_evaluate, output, hook.best_metrics)
        summary = {
            'model': 'Official DEIM-S + paper modules', 'task': args.task, 'cohort': spec['cohort'], 'seed': 0,
            'epochs_completed': len(rows), 'best_epoch_zero_based': best['epoch'],
            'final_epoch_zero_based': rows[-1]['epoch'], 'best_metrics': verified['metrics_percentage_points'],
            'final_epoch_metrics': percentage_metrics(rows[-1]['test_coco_eval_bbox']),
            'params_training_total': audit['params_training_total'],
            'params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'additional_parameters_over_official': audit['additional_parameters'],
            'elapsed_seconds': time.monotonic() - started,
            'stage1_RAM_restores': hook.stage1_loads, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'checkpoint_sha256': verified['checkpoint_sha256'], 'completed_at': now(),
            'smoke_only': args.smoke,
        }
        profile_path = HERE / 'profiles' / f'{args.task}.json'
        if profile_path.exists():
            summary['compute_profile'] = json.loads(profile_path.read_text())
        if not args.smoke:
            baseline = json.loads(Path(frozen['baseline_metrics']).read_text())
            summary['delta_vs_official_deim_seed0'] = {
                key: summary['best_metrics'][key] - baseline['best_metrics'][key]
                for key in ['AP', 'AP50', 'AP75', 'AR100']}
        assert list(output.rglob('*.pth')) == [output / 'best.pth']
        write_json(output / 'metrics_summary.json', summary)
        write_json(output / 'status.json', {'state': 'completed', **summary})
        (output / 'COMPLETED').write_text(now() + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        hook.restore()
        write_json(output / 'status.json', {'state': 'failed', 'error': repr(exc),
                   'epoch_zero_based': hook.epoch, 'best_retained': (output / 'best.pth').exists(),
                   'updated_at': now(), 'pid': os.getpid()})
        raise
    finally:
        dist_utils.cleanup()


if __name__ == '__main__':
    main()
