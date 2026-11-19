#!/usr/bin/env python3
"""Frozen new-augmentation experiment; never overwrite an earlier research run."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from prepare_migration import HERE, ROOT, TASKS, sha256, write_json

sys.path.insert(0, str(ROOT / 'scripts'))
from run_official import BestOnlyStorage, METRICS, percentage_metrics, now


def verify_frozen(protocol, task, require_preflight=True):
    run = protocol['runs'][task]
    assert sha256(run['config']) == run['config_sha256'], 'Config changed after freezing'
    assert yaml.safe_load(Path(run['config']).read_text())['paper_modules'] == run['modules']
    source = Path(protocol['upstream'])
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == protocol['upstream_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    for path, digest in protocol['source_sha256'].items():
        assert sha256(path) == digest, f'Frozen runtime changed: {path}'
    for path, digest in protocol['dataset_annotations'].items():
        assert sha256(path) == digest, f'Original annotation changed: {path}'
    weight = protocol['pretrained_backbone']
    assert sha256(weight['path']) == weight['sha256']
    spec = run['modules']
    if spec['spatial_shape']:
        assert sha256(spec['prior_file']) == spec['prior_sha256']
    if require_preflight:
        preflight = json.loads((HERE / 'preflight_validation.json').read_text())
        assert preflight['passed']
        assert preflight['protocol_sha256'] == sha256(HERE / 'protocol.json')
        assert preflight['source_sha256'] == protocol['source_sha256']
    return run


def install_runtime():
    source = ROOT / 'upstream/DEIM'
    sys.path.insert(0, str(source))
    from augmentation_runtime import install_augmentation_runtime
    install_augmentation_runtime()
    from paper_modules import install_modules, audit_model
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS as SOLVERS
    return YAMLConfig, dist_utils, SOLVERS, install_modules, audit_model


def export_verified_best(solver, native_evaluate, output, best_metrics):
    """Existing evaluator export, with hook restoration and explicit metric ordering."""
    state = torch.load(output / 'best.pth', map_location='cpu', weights_only=False)
    solver.load_state_dict(state)
    del state
    model = solver.ema.module if solver.ema else solver.model
    rows = []
    original_update = solver.evaluator.update

    def capture(predictions):
        for image_id, prediction in predictions.items():
            boxes = prediction['boxes'].detach().cpu().tolist()
            labels = prediction['labels'].detach().cpu().tolist()
            scores = prediction['scores'].detach().cpu().tolist()
            for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
                rows.append(dict(image_id=int(image_id), category_id=int(label),
                                 bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
        return original_update(predictions)

    solver.evaluator.update = capture
    try:
        stats, evaluator = native_evaluate(model, solver.criterion, solver.postprocessor,
                                          solver.val_dataloader, solver.evaluator, solver.device)
    finally:
        solver.evaluator.update = original_update
    observed = np.asarray(stats['coco_eval_bbox'], dtype=float)
    expected = np.asarray([best_metrics['metrics_fraction'][name] for name in METRICS], dtype=float)
    assert np.allclose(observed, expected, atol=2e-6, rtol=0), (observed, expected)
    prediction_file = output / 'best_predictions.coco.json.gz'
    with gzip.open(prediction_file, 'wt', encoding='utf-8') as handle:
        json.dump(rows, handle, separators=(',', ':'))
    ev = evaluator.coco_eval['bbox']
    np.savez_compressed(output / 'best_coco_arrays.npz', precision=ev.eval['precision'],
                        recall=ev.eval['recall'], scores=ev.eval.get('scores', np.array([])),
                        category_ids=np.asarray(ev.params.catIds))
    result = dict(verified_at=now(), matches_selected_checkpoint=True,
                  metrics_fraction=dict(zip(METRICS, observed.tolist())),
                  metrics_percentage_points=percentage_metrics(observed),
                  prediction_count=len(rows), prediction_sha256=sha256(prediction_file),
                  checkpoint_sha256=sha256(output / 'best.pth'))
    write_json(output / 'verified_coco_metrics.json', result)
    return result


def smoke_config(task, run):
    config = yaml.safe_load(Path(run['config']).read_text())
    # The native scheduler divides by warmup_iter even at iteration0; keep one
    # positive smoke-only warmup step. Formal configs retain warmup_iter=50.
    config.update(epoches=3, print_freq=1, flat_epoch=0, no_aug_epoch=1, warmup_iter=1)
    for split, count in [('train', 16), ('val', 16)]:
        dataset = config[f'{split}_dataloader']['dataset']
        data = json.loads(Path(dataset['ann_file']).read_text())
        selected = sorted(data['images'], key=lambda row: row['id'])[:count]
        ids = {row['id'] for row in selected}
        data['images'] = selected
        data['annotations'] = [row for row in data['annotations'] if row['image_id'] in ids]
        path = HERE / 'smoke_data' / f'{task}_{split}.json'
        write_json(path, data)
        dataset['ann_file'] = str(path)
        config[f'{split}_dataloader']['num_workers'] = 0
    # Test strong augmentation immediately and the RAM stage-1 restore at epoch2.
    config['train_dataloader']['dataset']['transforms']['policy']['epoch'] = [0, 1, 2]
    config['train_dataloader']['collate_fn'].update(stop_epoch=2, mixup_epochs=[0, 1], mixup_prob=1.0)
    path = HERE / 'smoke_data' / f'{task}.yml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--gpu', type=int, default=0, choices=[0])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--memory-limit-mib', type=int, default=9216)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib*1024**2/total, device=0)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    frozen = verify_frozen(protocol, args.task, require_preflight=not args.smoke)
    spec = frozen['modules']
    output = HERE / 'smoke_runs' / args.task if args.smoke else Path(frozen['output'])
    if (output / 'COMPLETED').is_file():
        print(f'Already completed; preserving: {output}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'log.txt').exists() or (output / 'best.pth').exists():
        raise RuntimeError(f'Refusing to overwrite or inaccurately resume a partial run: {output}')
    if shutil.disk_usage(output).free < 5*1024**3:
        raise RuntimeError('Less than 5GiB free before starting; no weights removed')
    YAMLConfig, dist_utils, SOLVERS, install_modules, audit_model = install_runtime()
    os.chdir(protocol['upstream'])
    dist_utils.setup_distributed(seed=0)
    config = smoke_config(args.task, frozen) if args.smoke else Path(frozen['config'])
    cfg = YAMLConfig(str(config), seed=0, output_dir=str(output),
                     summary_dir=str(output / 'summary'), device='cuda', use_amp=True)
    model = install_modules(cfg, spec)
    profile = json.loads((ROOT / 'profiles' / f"deim_s_{spec['cohort']}.json").read_text())
    audit = audit_model(model, profile['parameter_schema'], spec)
    write_json(output / 'architecture_audit.json', audit)
    for name, module in list(sys.modules.items()):
        if name == 'engine' or name.startswith('engine.'):
            file = getattr(module, '__file__', None)
            if file:
                assert Path(file).resolve().is_relative_to(Path(protocol['upstream']).resolve())
    run_record = dict(task=args.task, cohort=spec['cohort'], seed=0, gpu=args.gpu,
                      epochs=cfg.epoches, smoke_only=args.smoke, started_at=now(),
                      config=str(config), config_sha256=sha256(config),
                      protocol_sha256=sha256(HERE / 'protocol.json'),
                      upstream_commit=protocol['upstream_commit'], modules=spec,
                      source_sha256=protocol['source_sha256'],
                      dataset_annotations=protocol['dataset_annotations'],
                      pretrained_backbone=protocol['pretrained_backbone'],
                      python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      params_training_total=audit['params_training_total'],
                      memory_limit_mib=args.memory_limit_mib, command=sys.argv,
                      baseline_metrics=frozen.get('baseline_metrics'),
                      historical_reference=frozen.get('historical_reference'),
                      storage='one global best.pth; native stage-1 rollback snapshot in RAM',
                      interpretation='new seed-0 internal-validation recipe; not independent testing')
    write_json(output / 'run_protocol.json', run_record)
    solver = SOLVERS[cfg.yaml_cfg['task']](cfg)
    solver_module = importlib.import_module('engine.solver.det_solver')
    hook = BestOnlyStorage(solver, solver_module, dist_utils, output, 0)
    hook.install()
    start = time.monotonic()
    try:
        solver.fit()
        hook.restore()
        rows = [json.loads(line) for line in (output / 'log.txt').read_text().splitlines() if line.strip()]
        assert [row['epoch'] for row in rows] == list(range(cfg.epoches))
        best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
        assert best['epoch'] == hook.best_metrics['epoch_zero_based']
        verified = export_verified_best(solver, hook.original_evaluate, output, hook.best_metrics)
        summary = dict(model='Official DEIM-S + selected paper modules' if any(spec[k] for k in ['cmfa','structured','spatial_shape','dualopt']) else 'Official Plain DEIM-S',
                       task=args.task, cohort=spec['cohort'], seed=0, smoke_only=args.smoke,
                       epochs_completed=len(rows), final_epoch_zero_based=rows[-1]['epoch'],
                       best_epoch_zero_based=best['epoch'], best_metrics=verified['metrics_percentage_points'],
                       final_epoch_metrics=percentage_metrics(rows[-1]['test_coco_eval_bbox']),
                       params_training_total=audit['params_training_total'],
                       params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       additional_parameters_over_official=audit['additional_parameters'],
                       elapsed_seconds=time.monotonic()-start, stage1_RAM_restores=hook.stage1_loads,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                       checkpoint_sha256=verified['checkpoint_sha256'], completed_at=now(),
                       config_sha256=sha256(config), protocol_sha256=sha256(HERE / 'protocol.json'))
        baseline_path = frozen.get('baseline_metrics')
        if not args.smoke and baseline_path and not frozen.get('is_plain_control', False):
            bp = Path(baseline_path)
            if bp.exists() and (bp.parent / 'COMPLETED').exists():
                baseline = json.loads(bp.read_text())
                peer_task = spec['cohort']+'_plain'
                assert baseline['config_sha256'] == protocol['runs'][peer_task]['config_sha256']
                assert baseline['protocol_sha256'] == summary['protocol_sha256']
                summary['matched_baseline'] = dict(path=str(bp), sha256=sha256(bp),
                                                   delta_AP={key: summary['best_metrics'][key]-baseline['best_metrics'][key]
                                                             for key in ['AP','AP50','AP75','AR100']})
            else:
                summary['matched_baseline'] = {'available': False,
                    'note': 'Same-recipe Plain has not completed; no module-gain claim versus historical baselines.'}
        assert list(output.rglob('*.pth')) == [output / 'best.pth']
        write_json(output / 'metrics_summary.json', summary)
        write_json(output / 'status.json', dict(state='completed', **summary))
        (output / 'COMPLETED').write_text(now()+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        hook.restore()
        write_json(output / 'status.json', dict(state='failed', error=repr(exc),
                   epoch_zero_based=hook.epoch, best_retained=(output / 'best.pth').exists(),
                   updated_at=now(), pid=os.getpid()))
        raise
    finally:
        dist_utils.cleanup()


if __name__ == '__main__':
    main()
