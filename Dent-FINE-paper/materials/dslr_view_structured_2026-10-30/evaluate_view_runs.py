"""Native FP32 EMA replay and same-weight, acquisition-view prior evaluation.

Usage: python evaluate_view_runs.py JOB_JSON
       python evaluate_view_runs.py --self-test

The saved no-prior checkpoint is selected by the native training evaluator.
Exactly one native forward pass per validation image is observed. Center/shape
scores are then added to the complete [Q,52] logits BEFORE native top-300
selection; no target tooth classes are used to choose an acquisition view.
No source, dataset, checkpoint, training log or saved configuration is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
METRIC_NAMES = (
    'AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
    'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge',
)
MODES = ('none', 'center', 'shape', 'both')
REPLAY_TOLERANCE = 2e-6  # Metric fractions, i.e. 0.0002 AP percentage points.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def completed_training_log(path, epochs=160):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != epochs or [row['epoch'] for row in rows] != list(range(epochs)):
        raise RuntimeError(f'Training log must contain exactly {epochs} unique, ordered epochs 0..{epochs-1}')
    for row in rows:
        values = row['test_coco_eval_bbox']
        if len(values) != len(METRIC_NAMES) or not all(math.isfinite(float(x)) for x in values):
            raise RuntimeError('Training log contains invalid COCO statistics')
    return rows, max(rows, key=lambda row: row['test_coco_eval_bbox'][0])


def metric_dict(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != len(METRIC_NAMES) or not np.isfinite(values).all():
        raise RuntimeError('Invalid native COCO metrics')
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def coco_records(predictions):
    """Use the native tensor subtraction for xyxy->xywh before serialization."""
    import torch
    records = []
    for image_id, prediction in predictions.items():
        boxes = prediction['boxes']
        x1, y1, x2, y2 = boxes.unbind(1)
        xywh = torch.stack((x1, y1, x2 - x1, y2 - y1), 1).tolist()
        for label, score, box in zip(prediction['labels'].tolist(), prediction['scores'].tolist(), xywh):
            records.append({'image_id': int(image_id), 'category_id': int(label),
                            'bbox': box, 'score': float(score)})
    return records


def fixed_endpoints(records, gt, score=.30, iou=.50):
    """Existing descriptive localization-first Hungarian protocol, unchanged.

    This is not COCO matching and is not a clinical charting-error estimate.
    Every retained query/category prediction is included, without added NMS.
    """
    from scipy.optimize import linear_sum_assignment
    grouped = {int(i): [] for i in gt.imgs}
    for prediction in records:
        if prediction['score'] >= score:
            grouped[int(prediction['image_id'])].append(prediction)
    totals = Counter()
    rows = []
    for image_id in sorted(grouped):
        predictions = grouped[image_id]
        annotations = [a for a in gt.imgToAnns.get(image_id, []) if not a.get('iscrowd', 0)]
        labels = Counter(p['category_id'] for p in predictions)
        localized = correct = 0
        if predictions and annotations:
            a = np.asarray([p['bbox'] for p in predictions], dtype=np.float64)
            b = np.asarray([g['bbox'] for g in annotations], dtype=np.float64)
            lo = np.maximum(a[:, None, :2], b[None, :, :2])
            hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
            intersection = np.clip(hi - lo, 0, None).prod(-1)
            union = a[:, 2:].prod(-1)[:, None] + b[:, 2:].prod(-1)[None, :] - intersection
            overlaps = intersection / np.maximum(union, 1e-12)
            rr, cc = linear_sum_assignment(-overlaps)
            for r, c in zip(rr, cc):
                if overlaps[r, c] >= iou:
                    localized += 1
                    correct += int(predictions[r]['category_id'] == annotations[c]['category_id'])
        row = {'image_id': image_id, 'GT': len(annotations), 'predictions': len(predictions),
               'localized_matches': localized, 'correct_FDI_matches': correct,
               'TP': correct, 'FP': len(predictions) - correct, 'FN': len(annotations) - correct,
               'duplicate_image': int(any(n > 1 for n in labels.values())),
               'duplicate_count': sum(max(n - 1, 0) for n in labels.values())}
        rows.append(row)
        totals.update({k: v for k, v in row.items() if k != 'image_id'})
    def ratio(a, b):
        return a / b if b else None
    result = {
        'score_threshold': score, 'IoU_threshold': iou, 'counts': dict(totals),
        'Precision': ratio(totals['TP'], totals['predictions']),
        'Recall': ratio(totals['TP'], totals['GT']),
        'F1': ratio(2 * totals['TP'], 2 * totals['TP'] + totals['FP'] + totals['FN']),
        'matched_tooth_FDI_accuracy': ratio(totals['correct_FDI_matches'], totals['localized_matches']),
        'localization_coverage': ratio(totals['localized_matches'], totals['GT']),
        'raw_duplicate_image_rate': ratio(totals['duplicate_image'], len(grouped)),
        'matching': 'class-agnostic one-to-one Hungarian maximum summed IoU, then IoU>=0.50',
        'denominators': 'P:retained predictions; R:GT; FDI:localized matches; coverage:GT; duplicates:images',
    }
    return result, rows


class ObservedLoader:
    """Read acquisition metadata without altering native validation tensors."""
    def __init__(self, loader, state, resolver):
        self.loader, self.state, self.resolver = loader, state, resolver
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for samples, targets in self.loader:
            ids = [int(t['image_id'].item()) for t in targets]
            views = [self.resolver(self.dataset.coco.imgs[i]) for i in ids]
            if any(v < 0 for v in views):
                raise RuntimeError('Validation image lacks explicit acquisition-view metadata')
            for target, expected in zip(targets, views):
                if 'view_id' not in target or int(target['view_id'].item()) != expected:
                    raise RuntimeError('Loader target view ID disagrees with COCO acquisition metadata')
            self.state['current_ids'], self.state['current_views'] = ids, views
            yield samples, targets


def evaluate_job(job_path):
    import torch
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    implementation = Path(job.get('implementation') or job['implementation_dir']).resolve()
    config = Path(job['config']).resolve()
    output = Path(job['output_dir']).resolve()
    checkpoint, log = output / 'best.pth', output / 'log.txt'
    prior_path = Path(job.get('prior_path') or job.get('view_prior_path') or
                      ROOT / 'priors/view_specific_tooth_priors.pt').resolve()
    if job.get('epochs', 160) != 160:
        raise RuntimeError('This experiment specifies an independent 160-epoch budget')
    rows, best = completed_training_log(log)
    hashes_before = {'checkpoint_sha256': sha(checkpoint), 'config_sha256': sha(config),
                     'training_log_sha256': sha(log), 'view_prior_sha256': sha(prior_path)}
    if job.get('config_sha256') and job['config_sha256'] != hashes_before['config_sha256']:
        raise RuntimeError('Configuration hash differs from the frozen job')
    if job.get('prior_sha256') and job['prior_sha256'] != hashes_before['view_prior_sha256']:
        raise RuntimeError('View-specific training prior hash differs from the frozen job')
    weights = job.get('prior_weights', {'center': .2, 'shape': .05})
    center_weight, shape_weight = float(weights['center']), float(weights['shape'])
    if not all(math.isfinite(x) and x >= 0 for x in (center_weight, shape_weight)):
        raise ValueError('Prior weights must be explicit, nonnegative and finite')

    sys.path.insert(0, str(implementation))
    from view_identity_runtime import install_view_identity_runtime, metadata_view_id
    installation = install_view_identity_runtime(job.get('runtime_settings', {}))
    from view_prior_ops import load_view_prior, apply_view_priors
    import engine
    from engine.solver import det_solver, _solver
    from engine.data.dataset.coco_eval import CocoEvaluator
    if Path(engine.__file__).resolve().parent != implementation / 'engine':
        raise RuntimeError(f'Unexpected engine import: {engine.__file__}')
    torch.set_num_threads(4)
    prior = load_view_prior(prior_path)
    state = {'calls': 0, 'all_ids': [], 'all_views': [], 'sizes': [], 'logits': [], 'boxes': [],
             'predictions': {mode: {} for mode in MODES}, 'prior_metrics': {}}
    native_setup = _solver.BaseSolver._setup
    native_evaluate = det_solver.evaluate

    def observe_setup(solver, *args, **kwargs):
        result = native_setup(solver, *args, **kwargs)
        state['solver'] = solver
        state['params_total'] = sum(p.numel() for p in solver.model.parameters())
        state['params_trainable'] = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)
        return result

    def observe_evaluate(model, criterion, postprocessor, loader, evaluator, device):
        if state['calls']:
            raise RuntimeError('Expected exactly one native best-checkpoint evaluation')
        state['calls'] += 1
        if torch.is_autocast_enabled():
            raise RuntimeError('Native validation must run without autocast')
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise RuntimeError('Evaluation model must contain FP32 floating parameters')
        for flag in ('use_dental_spatial_prior', 'use_dental_shape_prior',
                     'use_dental_appearance_prior', 'use_dental_view_prior', 'use_global_assignment'):
            if bool(getattr(postprocessor, flag, False)):
                raise RuntimeError(f'Native no-prior selection requires {flag}=False')
        if postprocessor.num_classes != 52 or postprocessor.num_top_queries != 300:
            raise RuntimeError('Expected 52 FDI classes and native top-300 selection')
        if postprocessor.remap_mscoco_category:
            raise RuntimeError('FDI52 labels must not use MSCOCO remapping')
        gt = loader.dataset.coco
        if set(int(i) for i in gt.cats) != set(range(52)):
            raise RuntimeError('Validation category IDs must be the fixed zero-based FDI52 map')
        state['gt'] = gt
        state['category_map'] = [{'label_index': int(i), 'FDI': int(gt.cats[i].get('fdi', gt.cats[i]['name']))}
                                 for i in sorted(gt.cats)]
        observed_loader = ObservedLoader(loader, state, metadata_view_id)

        def observe_postprocessor(outputs, sizes):
            logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
            if logits.dtype != torch.float32 or boxes.dtype != torch.float32:
                raise RuntimeError('Native FP32 candidate outputs unexpectedly changed precision')
            if logits.shape[1:] != (300, 52) or boxes.shape != (*logits.shape[:2], 4):
                raise RuntimeError(f'Unexpected pre-top-k shapes: {logits.shape}, {boxes.shape}')
            if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
                raise RuntimeError('Nonfinite candidate tensors')
            ids, views = state['current_ids'], state['current_views']
            if len(ids) != logits.shape[0]:
                raise RuntimeError('Observed loader/postprocessor batch mismatch')
            if any(i in state['predictions']['none'] for i in ids):
                raise RuntimeError('A validation image was evaluated more than once')
            state['all_ids'].extend(ids)
            state['all_views'].extend(views)
            state['sizes'].append(sizes.detach().cpu().numpy())
            state['logits'].append(logits.detach().cpu().numpy())
            state['boxes'].append(boxes.detach().cpu().numpy())
            native_results = postprocessor(outputs, sizes)
            for mode in MODES:
                if mode == 'none':
                    results = native_results
                else:
                    corrected = apply_view_priors(logits, boxes, views, prior,
                        spatial_weight=center_weight if mode in ('center', 'both') else 0.,
                        shape_weight=shape_weight if mode in ('shape', 'both') else 0.)
                    revised = dict(outputs, pred_logits=corrected)
                    results = postprocessor(revised, sizes)
                for image_id, prediction in zip(ids, results):
                    state['predictions'][mode][image_id] = {
                        name: tensor.detach().cpu() for name, tensor in prediction.items()}
            return native_results

        result = native_evaluate(model, criterion, observe_postprocessor, observed_loader, evaluator, device)
        state['prior_metrics']['none'] = metric_dict(result[0]['coco_eval_bbox'])
        expected = set(int(i) for i in gt.imgs)
        if set(state['all_ids']) != expected or len(state['all_ids']) != len(expected):
            raise RuntimeError('Validation image coverage differs from the locked existing split')
        for mode in MODES[1:]:
            mode_evaluator = CocoEvaluator(gt, ['bbox'])
            # Reuse native tensor bbox conversion, without a Python float round-trip.
            mode_evaluator.update(state['predictions'][mode])
            mode_evaluator.synchronize_between_processes()
            mode_evaluator.accumulate()
            print('SAME_WEIGHT_ACQUISITION_VIEW_PRIOR', mode, flush=True)
            mode_evaluator.summarize()
            state['prior_metrics'][mode] = metric_dict(mode_evaluator.coco_eval['bbox'].stats)
        return result

    _solver.BaseSolver._setup = observe_setup
    det_solver.evaluate = observe_evaluate
    evaluation_dir = output / 'view_evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    train_script = implementation / 'train.py'
    original_cwd, original_argv = Path.cwd(), sys.argv[:]
    argv = [str(train_script), '-c', str(config), '--seed', str(job.get('seed', 0)),
            '--use-amp', '-d', 'cuda', '--resume', str(checkpoint), '--test-only',
            '--output-dir', str(evaluation_dir), '--summary-dir', str(evaluation_dir / 'tensorboard')]
    started = time.time()
    try:
        os.chdir(job.get('native_cwd', '/home/wuxi/Dent-FINE'))
        sys.argv = argv
        runpy.run_path(str(train_script), run_name='__main__')
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        _solver.BaseSolver._setup = native_setup
        det_solver.evaluate = native_evaluate
    if state['calls'] != 1:
        raise RuntimeError('No native validation pass was captured')
    solver = state['solver']
    if solver.ema is None:
        raise RuntimeError('These experiments select the EMA checkpoint, but no EMA was loaded')
    hashes_after = {'checkpoint_sha256': sha(checkpoint), 'config_sha256': sha(config),
                    'training_log_sha256': sha(log), 'view_prior_sha256': sha(prior_path)}
    if hashes_after != hashes_before:
        raise RuntimeError('Checkpoint, configuration, log or training-derived priors changed during replay')
    logged_metrics = metric_dict(best['test_coco_eval_bbox'])
    native_metrics = state['prior_metrics']['none']
    differences = {name: native_metrics[name] - logged_metrics[name] for name in METRIC_NAMES}
    if any(abs(differences[name]) > REPLAY_TOLERANCE for name in ('AP', 'AP50', 'AP75', 'AR100')):
        atomic_json(evaluation_dir / 'REPLAY_MISMATCH.json',
                    {'native': native_metrics, 'logged': logged_metrics, 'differences_fraction': differences,
                     'tolerance_fraction': REPLAY_TOLERANCE, **hashes_before})
        raise RuntimeError('Native FP32 no-prior best replay differs from logged selection by >2e-6')

    np.savez_compressed(evaluation_dir / 'pre_topk_candidates.npz',
        image_ids=np.asarray(state['all_ids'], dtype=np.int64),
        view_ids=np.asarray(state['all_views'], dtype=np.int64),
        original_sizes=np.concatenate(state['sizes']),  # Native order: width, height.
        original_sizes_wh=np.concatenate(state['sizes']),
        logits=np.concatenate(state['logits']), boxes_cxcywh=np.concatenate(state['boxes']))
    prior_results = {}
    for mode in MODES:
        predictions = state['predictions'][mode]
        records = coco_records(predictions)
        with gzip.open(evaluation_dir / f'predictions_{mode}_coco.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(records, stream, separators=(',', ':'), allow_nan=False)
        fixed, per_image = fixed_endpoints(records, state['gt'])
        atomic_json(evaluation_dir / f'per_image_counts_{mode}.json', per_image)
        ids = state['all_ids']
        np.savez_compressed(evaluation_dir / f'predictions_{mode}.npz',
            image_ids=np.asarray(ids, dtype=np.int64),
            view_ids=np.asarray(state['all_views'], dtype=np.int64),
            labels=np.stack([predictions[i]['labels'].numpy() for i in ids]),
            scores=np.stack([predictions[i]['scores'].numpy() for i in ids]),
            boxes_xyxy=np.stack([predictions[i]['boxes'].numpy() for i in ids]))
        metrics = state['prior_metrics'][mode]
        prior_results[mode] = {'mode': mode,
            'center_weight': center_weight if mode in ('center', 'both') else 0.,
            'shape_weight': shape_weight if mode in ('shape', 'both') else 0.,
            'metrics_fraction': metrics, 'metrics_percent': {k: v * 100 for k, v in metrics.items()},
            'delta_AP_points_vs_none': 100 * (metrics['AP'] - native_metrics['AP']),
            'fixed_threshold': fixed}
    provenance = {
        'job_id': job.get('id', job_path.stem), 'configuration': job.get('configuration', job_path.stem),
        'cohort': 'DSLR', 'seed': int(job.get('seed', 0)), **hashes_before,
        'implementation': str(implementation), 'native_train_script_sha256': sha(train_script),
        'identity_runtime_sha256': sha(ROOT / 'view_identity_runtime.py'),
        'view_prior_ops_sha256': sha(ROOT / 'view_prior_ops.py'),
        'evaluator_script_sha256': sha(__file__), 'runtime_installation': installation,
        'native_arguments': argv, 'precision': 'native FP32 evaluation, EMA model, no autocast',
        'evaluated_ema': True, 'params_total': state['params_total'],
        'params_trainable': state['params_trainable'], 'validation_images': len(state['all_ids']),
        'validation_boxes': len(state['gt'].anns), 'candidate_shape': list(np.concatenate(state['logits']).shape),
        'view_counts': dict(Counter(str(v) for v in state['all_views'])), 'FDI_label_map': state['category_map'],
        'view_routing': 'explicit images[] acquisition view metadata, checked against tensor loader targets',
        'candidate_export': 'untruncated logits and normalized cxcywh before score activation and native top300',
        'prior_role': 'fixed training-derived per-view center/log-shape score correction before top300, not a training loss',
        'postprocessing': 'native top300, no added score threshold, NMS or global assignment for COCO',
        'checkpoint_selection': 'maximum native validation COCO bbox AP, no priors',
        'best_epoch_zero_based': int(best['epoch']), 'best_epoch_one_based': int(best['epoch']) + 1,
        'final_epoch_zero_based': 159, 'final_epoch_one_based': 160, 'completed_epochs': len(rows),
        'checkpoint_internal_last_epoch': int(solver.last_epoch),
        'checkpoint_epoch_note': 'Native stage rollback can reset internal last_epoch; best epoch is from full training log',
        'logged_best_metrics_fraction': logged_metrics, 'native_minus_logged_fraction': differences,
        'native_replay_tolerance_fraction': REPLAY_TOLERANCE,
        'verified_best_checkpoint': True, 'authoritative_native_evaluation': True,
        'independent_test': False, 'split_changed': False,
        'evaluation_seconds': time.time() - started,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
    }
    atomic_json(evaluation_dir / 'evaluation_provenance.json', provenance)
    atomic_json(output / 'prior_results.json', {'provenance': provenance, 'modes': prior_results})
    with (output / 'prior_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Mode', 'Center weight', 'Shape weight', 'AP', 'AP50', 'AP75', 'AR100', 'Delta AP vs none'])
        for mode in MODES:
            result = prior_results[mode]
            writer.writerow([mode, result['center_weight'], result['shape_weight'],
                *[result['metrics_percent'][k] for k in ('AP', 'AP50', 'AP75', 'AR100')],
                result['delta_AP_points_vs_none']])
    summary = dict(provenance, metrics_fraction=native_metrics,
                   metrics_percent={k: v * 100 for k, v in native_metrics.items()},
                   fixed_threshold=prior_results['none']['fixed_threshold'],
                   prior='none', prior_weights={'center': center_weight, 'shape': shape_weight},
                   prior_results=prior_results, checkpoint=str(checkpoint),
                   evaluation_directory=str(evaluation_dir),
                   primary_metric_source='native evaluator, no-prior EMA best checkpoint')
    atomic_json(output / 'native_coco_metrics.json', summary)
    atomic_json(output / 'best_selection.json',
                {'epoch_zero_based': int(best['epoch']), 'epoch_one_based': int(best['epoch']) + 1,
                 'AP_fraction': logged_metrics['AP'], 'checkpoint_sha256': hashes_before['checkpoint_sha256']})
    atomic_json(output / 'metrics_summary.json', summary)
    print('VIEW_EVALUATION_COMPLETE', job.get('id', job_path.stem),
          {m: prior_results[m]['metrics_percent']['AP'] for m in MODES}, flush=True)


def self_test():
    """CPU-only API, pre-top-k ordering, size conversion and log-budget tests."""
    import torch
    sys.path.insert(0, '/home/wuxi/Dent-FINE')
    from engine.deim.postprocessor import PostProcessor
    from view_prior_ops import load_view_prior, apply_view_priors
    from view_identity_runtime import metadata_view_id
    torch.manual_seed(19)
    logits = torch.randn(2, 300, 52)
    boxes = torch.rand(2, 300, 4).clamp_min(.01)
    sizes = torch.tensor([[3264, 2448], [2048, 1536]])
    post = PostProcessor(num_classes=52, num_top_queries=300, remap_mscoco_category=False)
    prior = load_view_prior(ROOT / 'priors/view_specific_tooth_priors.pt')
    views = [0, 4]
    baseline = post({'pred_logits': logits, 'pred_boxes': boxes}, sizes)
    no_change = apply_view_priors(logits, boxes, views, prior, spatial_weight=0., shape_weight=0.)
    assert torch.equal(logits, no_change)
    corrected = apply_view_priors(logits, boxes, views, prior, spatial_weight=.2, shape_weight=.05)
    assert corrected.shape == (2, 300, 52) and torch.isfinite(corrected).all()
    altered = post({'pred_logits': corrected, 'pred_boxes': boxes}, sizes)
    assert any(not torch.equal(a['labels'], b['labels']) for a, b in zip(baseline, altered))
    assert all(p['labels'].shape == (300,) and p['boxes'].shape == (300, 4) for p in altered)
    records = coco_records({11: baseline[0], 12: baseline[1]})
    assert len(records) == 600 and set(p['image_id'] for p in records) == {11, 12}
    first = baseline[0]['boxes'][0]
    assert records[0]['bbox'] == torch.stack((first[0], first[1], first[2] - first[0], first[3] - first[1])).tolist()
    assert metadata_view_id({'view': 'right_buccal', 'view_id': 4}) == 4
    with tempfile.TemporaryDirectory(prefix='dslr_view_eval_test_') as temp:
        path = Path(temp) / 'log.txt'
        rows = [{'epoch': i, 'test_coco_eval_bbox': [.5 + i / 10000] * 12} for i in range(160)]
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        parsed, selected = completed_training_log(path)
        assert len(parsed) == 160 and selected['epoch'] == 159
        path.write_text('\n'.join(json.dumps(row) for row in rows[:-1]))
        try:
            completed_training_log(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete 159-epoch training log was accepted')
    print('SELF_TEST_PASSED: FP32 full candidate prior scoring, native top300, native xywh, strict 160-epoch budget')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', nargs='?')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.job:
        evaluate_job(args.job)
    else:
        parser.error('JOB_JSON is required unless --self-test is supplied')
