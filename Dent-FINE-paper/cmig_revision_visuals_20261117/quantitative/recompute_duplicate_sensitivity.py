#!/usr/bin/env python3
"""CPU-only post-hoc exact-duplicate sensitivity analysis; no model inference.

Run using the existing environment with pycocotools:
  /home/wuxi/miniconda3/envs/deimv2/bin/python recompute_duplicate_sensitivity.py

The saved predictions must reproduce the native AP before restricted evaluation.
All five completed seeds are retained; selection flags come from the archived
2026-10-12 selection artifact, never from the restricted evaluation results.
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import importlib.metadata
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml

OUT = Path(__file__).resolve().parent
MATERIALS = Path('/home/wuxi/Dent-FINE-paper/materials')
RUN_ROOT = MATERIALS / 'cmig_selected_configs_multiseed_2026-10-11/runs'
SELECTION_FILE = MATERIALS / 'cmig_selected_seed_revision_2026-10-12/quantitative/quantitative_data.json'
AUDIT_FILE = MATERIALS / 'cmig_controlled_2026-09-29/audit/dataset_audit.json'
SPEC = {'Pano': 'pano_spatialshape_original', 'Mobile': 'mobile_proto_k4_tau0p50'}
METRICS = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
           'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_sha256(path):
    # Identical definition to the archived source audit; no resizing/EXIF rotation.
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        h = hashlib.sha256(struct.pack('>II', *rgb.size))
        h.update(rgb.tobytes())
    return h.hexdigest()


def evaluate(gt, predictions, image_ids, log):
    with contextlib.redirect_stdout(log):
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: (float(value) * 100 if value >= 0 else None)
            for name, value in zip(METRICS, evaluator.stats)}


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def mean_sd(values):
    return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)),
            'n_seeds': len(values)}


def main():
    selection = json.loads(SELECTION_FILE.read_text())
    audit = json.loads(AUDIT_FILE.read_text())
    selection_by_key = {(x['cohort'], int(x['seed'])): x for x in selection['per_seed']}
    result = {
        'analysis': 'Post-hoc exact cross-split pixel-duplicate exclusion sensitivity',
        'no_training_or_inference': True,
        'no_reselection': True,
        'evaluation': 'pycocotools COCOeval bbox; native saved predictions; unchanged maxDets=[1,10,100] and category set; AP uses IoU 0.50:0.05:0.95',
        'python': sys.version,
        'pycocotools': importlib.metadata.version('pycocotools'),
        'selection_file': str(SELECTION_FILE),
        'selection_file_sha256': sha256(SELECTION_FILE),
        'selection_policy': selection['selection_policy'],
        'audit_file': str(AUDIT_FILE),
        'audit_file_sha256': sha256(AUDIT_FILE),
        'pixel_hash_definition': audit['pixel_hash_definition'],
        'limitations': [
            'Sensitivity analysis does not change training or reverse leakage exposure.',
            'It excludes only known exact pixel duplicates, not near duplicates or unidentified patient overlap.',
            'All checkpoints were selected using the original validation split; this is not independent test evaluation.',
            'The three reported seeds were selected after viewing five completed runs; summaries are descriptive and no significance inference is made.',
        ],
        'cohorts': {}, 'per_seed': [], 'summaries': {},
    }
    flat_rows = []
    duplicate_rows = []
    with (OUT / 'duplicate_sensitivity_evaluation.log').open('w') as log:
        for cohort, configuration in SPEC.items():
            cohort_audit = audit['cohorts'][cohort]
            config0 = yaml.safe_load((RUN_ROOT / configuration / 'seed_0/config.yml').read_text())
            datasets = {split: config0[loader]['dataset'] for split, loader in
                        [('train', 'train_dataloader'), ('validation', 'val_dataloader')]}
            for split, dataset in datasets.items():
                assert str(dataset['ann_file']) == cohort_audit['splits'][split]['annotation_file']
                assert sha256(dataset['ann_file']) == cohort_audit['splits'][split]['annotation_sha256']
            groups = cohort_audit['content_duplicates']['decoded_rgb_sha256']['cross_split_groups']
            remove_ids = set()
            for group in groups:
                for member in group['members']:
                    path = Path(datasets[member['split']]['img_folder']) / member['file_name']
                    actual_sha = pixel_sha256(path)
                    assert actual_sha == group['hash'], (cohort, path, actual_sha, group['hash'])
                    duplicate_rows.append({'cohort': cohort, **member, 'path': str(path),
                                           'decoded_rgb_sha256_verified': actual_sha})
                    if member['split'] == 'validation':
                        remove_ids.add(int(member['image_id']))
            with contextlib.redirect_stdout(log):
                gt = COCO(datasets['validation']['ann_file'])
            full_ids = sorted(gt.getImgIds())
            kept_ids = sorted(set(full_ids) - remove_ids)
            counts = {
                'full_images': len(full_ids), 'retained_images': len(kept_ids),
                'removed_image_ids': sorted(remove_ids),
                'full_annotations': len(gt.getAnnIds(imgIds=full_ids)),
                'retained_annotations': len(gt.getAnnIds(imgIds=kept_ids)),
                'annotation_file': datasets['validation']['ann_file'],
                'annotation_sha256': sha256(datasets['validation']['ann_file']),
                'classes': len(gt.getCatIds()),
                'lost_supported_category_ids': sorted({a['category_id'] for a in gt.dataset['annotations']} -
                    {a['category_id'] for a in gt.dataset['annotations'] if a['image_id'] in kept_ids}),
            }
            assert len(remove_ids) == (1 if cohort == 'Pano' else 4)
            result['cohorts'][cohort] = counts
            for seed in range(5):
                run = RUN_ROOT / configuration / f'seed_{seed}'
                native_file = run / 'native_coco_metrics.json'
                native = json.loads(native_file.read_text())
                manifest = selection_by_key[(cohort, seed)]
                predictions_file = run / 'evaluation/predictions_coco.json.gz'
                checkpoint_file = run / 'best.pth'
                config_file = run / 'config.yml'
                assert sha256(checkpoint_file) == native['checkpoint_sha256'] == manifest['checkpoint_sha256']
                assert sha256(config_file) == native['config_sha256']
                assert sha256(native_file) == manifest['native_metric_sha256']
                assert native['configuration'] == configuration and native['seed'] == seed
                with gzip.open(predictions_file, 'rt') as stream:
                    predictions = json.load(stream)
                assert {x['image_id'] for x in predictions} == set(full_ids)
                log.write(f'\n=== {cohort}, seed {seed}: original split ===\n')
                original = evaluate(gt, predictions, full_ids, log)
                errors = {key: original[key] - native['metrics_percent'][key]
                          for key in ['AP', 'AP50', 'AP75', 'AR100']}
                assert max(map(abs, errors.values())) < 1e-7, errors
                assert abs(original['AP'] - manifest['AP']) < 1e-7
                log.write(f'\n=== {cohort}, seed {seed}: exclude cross-split exact duplicates ===\n')
                restricted = evaluate(gt, predictions, kept_ids, log)
                per_seed = {
                    'cohort': cohort, 'configuration': configuration, 'seed': seed,
                    'selected_for_reported_mean': manifest['selected'],
                    'representative': manifest['representative'],
                    'original_metrics_percent': original, 'restricted_metrics_percent': restricted,
                    'delta_metrics_percentage_points': {key: restricted[key] - original[key]
                        for key in ['AP', 'AP50', 'AP75', 'AR100']},
                    'native_reproduction_error_percentage_points': errors,
                    'native_metric_path': str(native_file), 'native_metric_sha256': sha256(native_file),
                    'prediction_file': str(predictions_file), 'prediction_sha256': sha256(predictions_file),
                    'checkpoint_path': str(checkpoint_file), 'checkpoint_sha256_verified': native['checkpoint_sha256'],
                    'config_path': str(config_file), 'config_sha256_verified': native['config_sha256'],
                }
                result['per_seed'].append(per_seed)
                row = {key: per_seed[key] for key in ['cohort', 'configuration', 'seed', 'selected_for_reported_mean', 'representative']}
                row.update({key: counts[key] for key in ['full_images', 'retained_images', 'full_annotations', 'retained_annotations']})
                for key in ['AP', 'AP50', 'AP75', 'AR100']:
                    row[f'original_{key}'] = original[key]
                    row[f'restricted_{key}'] = restricted[key]
                    row[f'delta_{key}_pp'] = restricted[key] - original[key]
                flat_rows.append(row)
                print(f"{cohort} seed {seed} selected={manifest['selected']}: {original['AP']:.6f} -> {restricted['AP']:.6f}, delta={restricted['AP'] - original['AP']:+.6f}; native error={max(map(abs, errors.values())):.3g}", flush=True)
    summary_rows = []
    for cohort in SPEC:
        result['summaries'][cohort] = {}
        for subset in ['all_five', 'reported_three', 'fixed_012']:
            records = [r for r in result['per_seed'] if r['cohort'] == cohort and
                       (subset == 'all_five' or (subset == 'reported_three' and r['selected_for_reported_mean']) or
                        (subset == 'fixed_012' and r['seed'] in [0, 1, 2]))]
            summary = {'seeds': [r['seed'] for r in records]}
            for key in ['AP', 'AP50', 'AP75', 'AR100']:
                summary[f'original_{key}'] = mean_sd([r['original_metrics_percent'][key] for r in records])
                summary[f'restricted_{key}'] = mean_sd([r['restricted_metrics_percent'][key] for r in records])
                summary[f'delta_{key}_pp'] = mean_sd([r['delta_metrics_percentage_points'][key] for r in records])
            result['summaries'][cohort][subset] = summary
            summary_rows.append({'cohort': cohort, 'subset': subset, 'seeds': ','.join(map(str,summary['seeds'])),
                                 **{f'{field}_{stat}': values[stat] for field, values in summary.items() if field != 'seeds'
                                    for stat in ['mean', 'sample_sd', 'n_seeds']}})
    (OUT / 'duplicate_sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    write_csv(OUT / 'duplicate_sensitivity_per_seed.csv', flat_rows)
    write_csv(OUT / 'duplicate_sensitivity_summary.csv', summary_rows)
    write_csv(OUT / 'verified_duplicate_members.csv', duplicate_rows)


if __name__ == '__main__':
    main()
