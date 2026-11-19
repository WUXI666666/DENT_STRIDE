#!/usr/bin/env python3
"""Native upstream training with storage-only hooks and independent AP selection.

No upstream model, criterion, matching, augmentation implementation or optimizer
is patched. DEIM's original stage-1 rollbacks remain byte-for-byte snapshots in
RAM. Only the globally AP-best checkpoint is written to the experiment folder.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import inspect
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from prepare_comparison import ROOT, MODELS, COHORTS, sha256, write_json

METRICS = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']


def percentage_metrics(values):
    # COCO uses -1 for an undefined size/category slice, not a negative AP.
    return {k: 100*float(v) if float(v) >= 0 else None for k,v in zip(METRICS, values)}


def now():
    return datetime.now().astimezone().isoformat()


class BestOnlyStorage:
    """Wrap native solver I/O without altering its stage-local training decisions."""
    def __init__(self, solver, module, dist, output, seed):
        self.solver, self.module, self.dist = solver, module, dist
        self.output, self.seed = output, seed
        self.epoch = -1
        self.best_ap = -math.inf
        self.best_metrics = None
        self.stage1 = None
        self.stage1_loads = 0
        self.original_save = torch.save
        self.original_dist_save = dist.save_on_master
        self.original_train = module.train_one_epoch
        self.original_evaluate = module.evaluate
        self.original_resume = solver.load_resume_state
        self.signature = inspect.signature(self.original_train)

    def install(self):
        self.dist.save_on_master = self.save_native
        torch.save = self.save_tensor
        self.module.train_one_epoch = self.train_epoch
        self.module.evaluate = self.evaluate
        self.solver.load_resume_state = types.MethodType(lambda solver, path: self.load_native(path), self.solver)

    def restore(self):
        self.dist.save_on_master = self.original_dist_save
        torch.save = self.original_save
        self.module.train_one_epoch = self.original_train
        self.module.evaluate = self.original_evaluate
        self.solver.load_resume_state = self.original_resume

    def save_native(self, state, path, *args, **kwargs):
        path = Path(path)
        if path.name == 'NaN.pth':
            raise FloatingPointError('Native training generated non-finite boxes; no invalid weights retained')
        if path.parent.resolve() == self.output.resolve():
            if path.name == 'best_stg1.pth':
                stream = io.BytesIO()
                self.original_save(state, stream)
                self.stage1 = stream.getvalue()
            # Global best is saved by the independent evaluation hook below.
            return
        raise RuntimeError(f'Unexpected native checkpoint destination: {path}')

    def save_tensor(self, obj, file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            if path.parent.resolve() == (self.output / 'eval').resolve():
                return  # These are evaluator caches, not required for training.
        return self.original_save(obj, file, *args, **kwargs)

    def load_native(self, path):
        if Path(path).name == 'best_stg1.pth':
            if self.stage1 is None:
                raise RuntimeError('Stage-1 RAM snapshot missing')
            self.solver.load_state_dict(torch.load(io.BytesIO(self.stage1), map_location='cpu', weights_only=False))
            self.stage1_loads += 1
            return
        return self.original_resume(path)

    def train_epoch(self, *args, **kwargs):
        self.epoch = int(self.signature.bind(*args, **kwargs).arguments['epoch'])
        if shutil.disk_usage(self.output).free < 3 * 1024**3:
            raise RuntimeError('Less than 3 GiB free: stopped safely; existing best.pth retained')
        write_json(self.output / 'status.json', {'state': 'training', 'epoch_zero_based': self.epoch,
                   'epochs': self.solver.cfg.epoches, 'updated_at': now(), 'pid': os.getpid(),
                   'best_AP': self.best_ap if math.isfinite(self.best_ap) else None})
        return self.original_train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        stats, evaluator = self.original_evaluate(*args, **kwargs)
        values = [float(x) for x in stats['coco_eval_bbox']]
        if not math.isfinite(values[0]):
            raise FloatingPointError('Nonfinite validation AP')
        if self.epoch >= 0 and values[0] > self.best_ap:
            self.best_ap = values[0]
            self.best_metrics = {
                'epoch_zero_based': self.epoch, 'epoch_one_based': self.epoch + 1,
                'seed': self.seed, 'metrics_fraction': dict(zip(METRICS, values)),
                'metrics_percentage_points': percentage_metrics(values),
                'selected_at': now(), 'evaluation_weights': 'EMA' if self.solver.ema else 'model',
            }
            state = self.solver.state_dict()
            state['comparison_metadata'] = copy.deepcopy(self.best_metrics)
            temp = self.output / 'best.pth.tmp'
            self.original_save(state, temp)
            temp.replace(self.output / 'best.pth')
            write_json(self.output / 'best_metrics.json', self.best_metrics)
            print(f'[official-comparison] retained global best AP={self.best_ap:.8f}, epoch={self.epoch}', flush=True)
        write_json(self.output / 'status.json', {'state': 'epoch_completed', 'epoch_zero_based': self.epoch,
                   'epochs': self.solver.cfg.epoches, 'updated_at': now(), 'pid': os.getpid(),
                   'best_AP': self.best_ap, 'current_AP': values[0], 'stage1_RAM_restores': self.stage1_loads})
        return stats, evaluator


def export_verified_best(solver, native_evaluate, output, best_metrics):
    state = torch.load(output / 'best.pth', map_location='cpu', weights_only=False)
    solver.load_state_dict(state)
    module = solver.ema.module if solver.ema else solver.model
    rows = []
    original_update = solver.evaluator.update

    def capture(predictions):
        for image_id, prediction in predictions.items():
            boxes = prediction['boxes'].detach().cpu().tolist()
            labels = prediction['labels'].detach().cpu().tolist()
            scores = prediction['scores'].detach().cpu().tolist()
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                rows.append({'image_id': int(image_id), 'category_id': int(label),
                             'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(score)})
        return original_update(predictions)

    solver.evaluator.update = capture
    stats, evaluator = native_evaluate(module, solver.criterion, solver.postprocessor,
                                       solver.val_dataloader, solver.evaluator, solver.device)
    solver.evaluator.update = original_update
    observed = np.asarray(stats['coco_eval_bbox'], dtype=float)
    expected = np.asarray(list(best_metrics['metrics_fraction'].values()), dtype=float)
    if not np.allclose(observed, expected, atol=2e-6, rtol=0):
        raise RuntimeError(f'Best-checkpoint re-evaluation mismatch: expected={expected}, observed={observed}')
    prediction_file = output / 'best_predictions.coco.json.gz'
    with gzip.open(prediction_file, 'wt', encoding='utf-8') as handle:
        json.dump(rows, handle, separators=(',', ':'))
    ev = evaluator.coco_eval['bbox']
    np.savez_compressed(output / 'best_coco_arrays.npz', precision=ev.eval['precision'],
                        recall=ev.eval['recall'], scores=ev.eval.get('scores', np.array([])),
                        category_ids=np.asarray(ev.params.catIds))
    result = {'verified_at': now(), 'matches_selected_checkpoint': True,
              'metrics_fraction': dict(zip(METRICS, observed.tolist())),
              'metrics_percentage_points': percentage_metrics(observed),
              'prediction_count': len(rows), 'prediction_sha256': sha256(prediction_file),
              'checkpoint_sha256': sha256(output / 'best.pth')}
    write_json(output / 'verified_coco_metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--cohort', choices=COHORTS, required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'pretrained_cache'))
    torch.set_num_threads(4)
    spec = MODELS[args.model]
    output = ROOT / ('smoke_runs' if args.smoke else 'runs') / args.model / args.cohort / f'seed_{args.seed}'
    if (output / 'COMPLETED').exists():
        print(f'Already completed: {output}')
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'log.txt').exists() or (output / 'best.pth').exists():
        raise RuntimeError(f'Refusing to overwrite or mix an existing incomplete run: {output}')
    os.chdir(spec['code'])
    sys.path.insert(0, str(spec['code']))
    prefix = 'engine' if args.model == 'deim_s' else 'src'
    core = importlib.import_module(f'{prefix}.core')
    misc = importlib.import_module(f'{prefix}.misc.dist_utils')
    solver_module = importlib.import_module(f'{prefix}.solver.det_solver')
    solvers = importlib.import_module(f'{prefix}.solver')
    assert Path(solver_module.__file__).resolve().is_relative_to(spec['code'].resolve())
    for name, module in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + '.'):
            source = getattr(module, '__file__', None)
            if source and not Path(source).resolve().is_relative_to(spec['code'].resolve()):
                raise RuntimeError(f'Non-upstream module imported: {name}: {source}')
    misc.setup_distributed(seed=args.seed)
    config = ROOT / 'configs' / (f'{args.model}_smoke.yml' if args.smoke else f'{args.model}_{args.cohort}.yml')
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    if not args.smoke:
        assert sha256(config) == protocol['configs'][f'{args.model}_{args.cohort}']['sha256']
    commit = subprocess.check_output(['git', '-C', str(spec['repo']), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == protocol['models'][args.model]['commit']
    assert not subprocess.check_output(['git', '-C', str(spec['repo']), 'status', '--porcelain'], text=True).strip()
    cfg = core.YAMLConfig(str(config), seed=args.seed, output_dir=str(output),
                          summary_dir=str(output / 'summary'), device='cuda', use_amp=True)
    # This is the same first model construction that native solver.train performs.
    # No profiling or additional random draws are inserted into the training process.
    model = cfg.model
    pretrained_file = (Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth')
                       if args.model == 'deim_s' else
                       ROOT / 'pretrained_cache/hub/checkpoints/ResNet18_vd_pretrained_from_paddle.pth')
    if not pretrained_file.is_file():
        raise RuntimeError(f'Pretrained backbone provenance missing: {pretrained_file}')
    write_json(output / 'run_protocol.json', {
        'model': args.model, 'cohort': args.cohort, 'seed': args.seed, 'gpu': args.gpu,
        'smoke_only': args.smoke, 'config': str(config), 'config_sha256': sha256(config),
        'upstream_url': spec['url'], 'commit': commit, 'source_tree_clean': True,
        'runner_sha256': sha256(__file__), 'protocol_sha256': sha256(ROOT / 'protocol.json'),
        'started_at': now(), 'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'pretrained_backbone': str(pretrained_file), 'pretrained_sha256': sha256(pretrained_file),
        'params_training_total': sum(p.numel() for p in model.parameters()),
        'command': sys.argv, 'storage': 'one best.pth; native stage-1 checkpoint held in RAM',
    })
    solver = solvers.TASKS[cfg.yaml_cfg['task']](cfg)
    hook = BestOnlyStorage(solver, solver_module, misc, output, args.seed)
    hook.install()
    started = time.monotonic()
    try:
        solver.fit()
        hook.restore()
        log_rows = [json.loads(line) for line in (output / 'log.txt').read_text().splitlines() if line.strip()]
        assert [r['epoch'] for r in log_rows] == list(range(cfg.epoches)), 'Missing/duplicate epochs'
        expected_row = max(log_rows, key=lambda r: r['test_coco_eval_bbox'][0])
        assert hook.best_metrics['epoch_zero_based'] == expected_row['epoch']
        assert abs(hook.best_ap - expected_row['test_coco_eval_bbox'][0]) < 1e-12
        verified = export_verified_best(solver, hook.original_evaluate, output, hook.best_metrics)
        summary = {
            'model': args.model, 'cohort': args.cohort, 'seed': args.seed,
            'epochs_completed': len(log_rows), 'final_epoch_zero_based': log_rows[-1]['epoch'],
            'best_epoch_zero_based': hook.best_metrics['epoch_zero_based'],
            'best_metrics': verified['metrics_percentage_points'],
            'final_epoch_metrics': percentage_metrics(log_rows[-1]['test_coco_eval_bbox']),
            'params_training_total': sum(p.numel() for p in cfg.model.parameters()),
            'params_trainable': sum(p.numel() for p in cfg.model.parameters() if p.requires_grad),
            'stage1_RAM_restores': hook.stage1_loads, 'elapsed_seconds': time.monotonic()-started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
            'checkpoint_sha256': verified['checkpoint_sha256'], 'completed_at': now(),
        }
        profile_file = ROOT / 'profiles' / f'{args.model}_{args.cohort}.json'
        if profile_file.is_file():
            profile = json.loads(profile_file.read_text())
            summary['compute_profile'] = {k: profile[k] for k in ['GMACs', 'GFLOPs', 'params_deploy_total', 'profile']}
        write_json(output / 'metrics_summary.json', summary)
        weights = list(output.rglob('*.pth'))
        assert weights == [output / 'best.pth'], weights
        write_json(output / 'status.json', {'state': 'completed', **summary})
        (output / 'COMPLETED').write_text(now() + '\n')
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        hook.restore()
        write_json(output / 'status.json', {'state': 'failed', 'error': repr(exc), 'epoch_zero_based': hook.epoch,
                   'updated_at': now(), 'pid': os.getpid(), 'best_retained': (output / 'best.pth').exists()})
        raise
    finally:
        misc.cleanup()


if __name__ == '__main__':
    main()
