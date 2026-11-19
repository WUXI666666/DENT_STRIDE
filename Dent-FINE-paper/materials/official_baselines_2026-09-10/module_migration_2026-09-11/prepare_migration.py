#!/usr/bin/env python3
"""Freeze the three authorized seed-0 ports; never edit either source project."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = Path('/home/wuxi/Dent-FINE')
TASKS = {
    'phone_dualopt': {'cohort': 'phone', 'cmfa': False, 'structured': False, 'spatial_shape': False, 'dualopt': True},
    'dentex_spatial_shape': {'cohort': 'dentex', 'cmfa': False, 'structured': True, 'spatial_shape': True, 'dualopt': False},
    'dslr_cmfa_dualopt': {'cohort': 'dslr', 'cmfa': True, 'structured': False, 'spatial_shape': False, 'dualopt': True},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def protected_files():
    files = set()
    for folder in ('engine', 'configs', 'tools'):
        files.update(p for p in (LOCAL / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yml', '.yaml', '.sh'))
    files.update(p for p in LOCAL.glob('*.py') if p.is_file())
    files.update(p for p in Path('/home/wuxi/Dent-FINE-paper').glob('*.docx') if p.is_file())
    return {str(p): sha256(p) for p in sorted(files)}


def covariance(points, min_var=1e-3):
    if not points:
        return torch.zeros(2), torch.eye(2) / min_var, False
    x = torch.as_tensor(points, dtype=torch.float32)
    mu = x.mean(0)
    if len(points) == 1:
        cov = torch.eye(2) * min_var
    else:
        centered = x - mu
        cov = centered.t().matmul(centered) / max(1, len(points) - 1)
        cov = cov + torch.eye(2) * min_var
    return mu, torch.linalg.inv(cov), True


def build_train_prior(annotation, fdi_codes):
    data = json.loads(Path(annotation).read_text())
    images = {int(x['id']): x for x in data['images']}
    centers, shapes = defaultdict(list), defaultdict(list)
    assert sorted(c['id'] for c in data['categories']) == list(range(32))
    for ann in data['annotations']:
        cid = int(ann['category_id'])
        image = images[int(ann['image_id'])]
        width, height = float(image['width']), float(image['height'])
        x, y, w, h = map(float, ann['bbox'])
        assert 0 <= cid < 32 and min(w, h, width, height) > 0
        centers[cid].append([(x + 0.5 * w) / width, (y + 0.5 * h) / height])
        shapes[cid].append([max(w / width, 1e-5), max(h / height, 1e-5)])
    tensors = {prefix + suffix: [] for prefix in ('spatial', 'shape') for suffix in ('_mu', '_inv_cov', '_valid')}
    for cid in range(32):
        shape_points = torch.as_tensor(shapes[cid], dtype=torch.float32).clamp_min(1e-5).log().tolist()
        for prefix, points in (('spatial', centers[cid]), ('shape', shape_points)):
            mu, inverse, valid = covariance(points)
            tensors[prefix + '_mu'].append(mu.tolist())
            tensors[prefix + '_inv_cov'].append(inverse.tolist())
            tensors[prefix + '_valid'].append(valid)
    prior = {'num_classes': 32, 'fdi_label_order': fdi_codes,
             'annotation_file': annotation, 'annotation_sha256': sha256(annotation),
             'min_variance': 1e-3, 'support': [len(centers[c]) for c in range(32)], **tensors}
    # Independent comparison with the prior used by the earlier paper recipe.
    reference_path = LOCAL / 'outputs/ablation_all_new_modules/priors/dentex_tooth_templates.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    matches = {key: torch.equal(torch.as_tensor(value, dtype=reference[key].dtype), reference[key])
               for key, value in tensors.items()}
    assert all(matches.values()), matches
    assert reference['fdi_label_order'].tolist() == fdi_codes
    prior['reference_verification'] = {'path': str(reference_path), 'sha256': sha256(reference_path),
                                       'all_template_tensors_exactly_equal': True}
    return prior


def main():
    if (HERE / 'protocol.json').exists():
        raise RuntimeError('Migration protocol already frozen; refusing overwrite')
    torch.set_num_threads(4)
    base_protocol = json.loads((ROOT / 'protocol.json').read_text())
    evidence = json.loads((ROOT / 'dataset_evidence.json').read_text())
    upstream = ROOT / 'upstream/DEIM'
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    code_revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert code_revision == base_protocol['models']['deim_s']['commit']
    write_json(HERE / 'protected_files_before.json', protected_files())
    fdi_codes = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                 38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]
    prior_path = HERE / 'dentex_train_templates.json'
    write_json(prior_path, build_train_prior(base_protocol['datasets']['dentex']['train_ann'], fdi_codes))
    runs = {}
    for task, settings in TASKS.items():
        spec = copy.deepcopy(settings)
        cohort = spec['cohort']
        base_config = ROOT / 'configs' / f'deim_s_{cohort}.yml'
        assert sha256(base_config) == base_protocol['configs'][f'deim_s_{cohort}']['sha256']
        for split in ('train', 'val'):
            entry = evidence[cohort][split]
            assert sha256(entry['annotation']) == entry['sha256']
        config = yaml.safe_load(base_config.read_text())
        if spec['dualopt']:
            config['optimizer'].update(type='DualOptAdamW', rollback_weight=0.005, realtime_decay=0.01,
                                       layer_decay_power=1.5, layer_decay_min=0.5, layer_decay_max=1.5)
        if spec['structured']:
            config['DEIMCriterion']['losses'] += ['quad', 'tooth', 'dent', 'proto']
            config['DEIMCriterion']['weight_dict'].update(loss_quad=0.10, loss_tooth=0.10, loss_dent=0.05, loss_proto=0.05)
            spec.update(prototype_count=4, prototype_tau=0.20, fdi_codes=fdi_codes,
                        factor_inference_fusion=0.0, prototype_inference_fusion=0.0)
        if spec['spatial_shape']:
            spec.update(prior_file=str(prior_path), prior_sha256=sha256(prior_path),
                        spatial_prior_weight=0.2, shape_prior_weight=0.05,
                        view_prior_weight=0.2, view_type='panorama')
        config['paper_modules'] = spec
        config['output_dir'] = str(ROOT / 'runs/deim_s_paper_modules' / task / 'seed_0')
        config_path = HERE / 'configs' / f'{task}.yml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        baseline = ROOT / 'runs/deim_s' / cohort / 'seed_0/metrics_summary.json'
        assert (baseline.parent / 'COMPLETED').exists()
        runs[task] = {'config': str(config_path), 'config_sha256': sha256(config_path),
                      'baseline_config': str(base_config), 'baseline_config_sha256': sha256(base_config),
                      'baseline_metrics': str(baseline), 'baseline_metrics_sha256': sha256(baseline),
                      'output': config['output_dir'], 'modules': spec}
    write_json(HERE / 'protocol.json', {
        'frozen_at': datetime.now().astimezone().isoformat(), 'upstream_commit': code_revision,
        'upstream': str(upstream), 'base_protocol_sha256': sha256(ROOT / 'protocol.json'),
        'scope': 'User selected only the three primary recipes; no module sweep',
        'runs': runs, 'seeds': [0], 'epochs': 160,
        'initialization': 'Fresh official ImageNet HGNetv2-B0 backbone; no trained detector checkpoint fine-tuning',
        'selection': 'Maximum validation COCO AP; earliest epoch on exact tie',
        'scaffold': 'Official DEIM-D-FINE-S with unchanged native decoder, nearest upsampling and sinusoidal positions',
        'matching': 'Every native model/data/training setting equals its completed official DEIM cohort baseline except the explicitly listed modules/losses/optimizer/prior',
        'denoising_queries': 100,
        'denoising_note': 'Retain the official comparison config (100), not the historical local setting (80)',
        'input': 'Native 480-800 multiscale training before epoch 130; fixed 640x640 validation',
        'batches': {'train': 8, 'validation': 16}, 'amp': True, 'ema': True,
        'postprocessing': 'Native top-300; no score cutoff, NMS, uniqueness assignment or non-paper prior; SpatialShape only for DENTEX',
        'storage': 'One global best.pth per run; required native stage-1 rollback stored in RAM',
        'excluded': ['QueryFusion', 'DySample', 'learned row/column positions', 'query/graph/ROI refiners',
                     'HOOA', 'CEDv2', 'SMWG', 'MPF', 'UCM', 'all other unselected modules'],
        'interpretation': 'New seed-0 migration experiments on existing internal-validation splits, not new three-seed means or independent tests',
    })
    print(json.dumps({'prepared': str(HERE), 'tasks': list(runs), 'prior_verified': True}, indent=2))


if __name__ == '__main__':
    main()
