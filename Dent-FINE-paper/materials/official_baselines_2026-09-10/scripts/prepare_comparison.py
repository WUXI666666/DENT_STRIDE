#!/usr/bin/env python3
"""Freeze the upstream comparison without changing source datasets or upstream code."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path('/home/wuxi/Dent-FINE')
COHORTS = {
    'phone': {
        'classes': 52,
        'train_images': '/home/wuxi/coco/train',
        'val_images': '/home/wuxi/coco/valid',
        'train_ann': '/home/wuxi/coco/train/annotations_fdi_types.json',
        'val_ann': '/home/wuxi/coco/valid/annotations_fdi_types.json',
        'label_order': 'fdi52',
    },
    'dentex': {
        'classes': 32,
        'train_images': '/home/wuxi/DENTEX2023/train2017',
        'val_images': '/home/wuxi/DENTEX2023/val2017',
        'train_ann': '/home/wuxi/DENTEX2023/annotations/instances_train2017.json',
        'val_ann': '/home/wuxi/DENTEX2023/annotations/instances_val2017.json',
        'label_order': 'dentex_enumeration32',
    },
    'dslr': {
        'classes': 52,
        'train_images': '/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/train',
        'val_images': '/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val',
        'train_ann': '/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_train_fdi52.json',
        'val_ann': '/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json',
        'label_order': 'fdi52',
    },
}
MODELS = {
    'deim_s': {
        'label': 'Official DEIM-D-FINE-S (HGNetv2-B0)',
        'repo': ROOT / 'upstream/DEIM',
        'code': ROOT / 'upstream/DEIM',
        'base': 'configs/deim_dfine/deim_hgnetv2_s_coco.yml',
        'url': 'https://github.com/Intellindust-AI-Lab/DEIM',
    },
    'rtdetrv2_s': {
        'label': 'Official RT-DETRv2-S (ResNet18-vd)',
        'repo': ROOT / 'upstream/RT-DETR',
        'code': ROOT / 'upstream/RT-DETR/rtdetrv2_pytorch',
        'base': 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
        'url': 'https://github.com/lyuwenyu/RT-DETR',
    },
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def merge(a, b):
    a = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            a[k] = merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


def load_yaml(path):
    path = Path(path)
    obj = yaml.safe_load(path.read_text()) or {}
    result = {}
    for name in obj.pop('__include__', []):
        result = merge(result, load_yaml(path.parent / name))
    return merge(result, obj)


def profile_dataset(spec):
    profiles = {}
    maps = []
    for split in ('train', 'val'):
        ann = Path(spec[f'{split}_ann'])
        folder = Path(spec[f'{split}_images'])
        data = json.loads(ann.read_text())
        ids = {i['id'] for i in data['images']}
        cats = sorted(data['categories'], key=lambda c: c['id'])
        assert [c['id'] for c in cats] == list(range(spec['classes']))
        assert len(ids) == len(data['images'])
        assert len({a['id'] for a in data['annotations']}) == len(data['annotations'])
        assert all((folder / i['file_name']).is_file() for i in data['images'])
        assert all(a['image_id'] in ids and 0 <= a['category_id'] < spec['classes']
                   and a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in data['annotations'])
        maps.append(cats)
        profiles[split] = {
            'annotation': str(ann), 'sha256': sha256(ann), 'image_root': str(folder),
            'images': len(ids), 'annotations': len(data['annotations']),
            'category_map': cats, 'annotation_id_zero_present': any(a['id'] == 0 for a in data['annotations']),
        }
    assert maps[0] == maps[1], 'Train/validation label maps differ'
    return profiles


def build_config(model, cohort):
    spec = MODELS[model]
    ds = COHORTS[cohort]
    cfg = load_yaml(spec['code'] / spec['base'])
    cfg.update(num_classes=ds['classes'], remap_mscoco_category=False,
               epoches=160, checkpoint_freq=1000000, print_freq=100,
               device='cuda', use_amp=True, use_ema=True, sync_bn=False)
    for split, batch in [('train', 8), ('val', 16)]:
        loader = cfg[f'{split}_dataloader']
        loader.pop('batch_size', None)
        loader.update(total_batch_size=batch, num_workers=8)
        loader['dataset'].update(img_folder=ds[f'{split}_images'], ann_file=ds[f'{split}_ann'])
    if model == 'deim_s':
        # Architecture/loss/solver are imported from the pristine official repository.
        # Dataset-specific augmentation and schedule match the existing local controls.
        local_cfg = load_yaml(LOCAL / 'configs/deim_dfine/deim_hgnetv2_s_phone.yml')
        cfg['optimizer'] = local_cfg['optimizer']
        for key in ['lrsheduler', 'lr_gamma', 'warmup_iter', 'flat_epoch', 'no_aug_epoch']:
            cfg[key] = local_cfg[key]
        cfg['ema'] = local_cfg['ema']
        cfg['lr_scheduler'] = local_cfg['lr_scheduler']
        cfg['lr_warmup_scheduler'] = local_cfg['lr_warmup_scheduler']
        cfg['train_dataloader']['dataset']['transforms'] = copy.deepcopy(local_cfg['train_dataloader']['dataset']['transforms'])
        cfg['train_dataloader']['collate_fn'] = copy.deepcopy(local_cfg['train_dataloader']['collate_fn'])
        cfg['HGNetv2']['local_model_dir'] = '/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/'
        cfg['HGNetv2']['pretrained'] = True
    else:
        # Do not insert DEIM matching/MixUp or DENT-STRIDE modules into RT-DETRv2.
        # Retain official S optimizer, loss and fixed-size collator. Align image ops
        # and the augmentation stop to the established dental protocol.
        tr = cfg['train_dataloader']['dataset']['transforms']
        tr['ops'] = [op for op in tr['ops'] if op['type'] != 'RandomHorizontalFlip']
        tr['policy']['epoch'] = 130
        cfg['train_dataloader']['collate_fn'].update(scales=None, stop_epoch=130)
    return cfg


def main():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; do not overwrite an active comparison.')
    dataset_evidence = {c: profile_dataset(s) for c, s in COHORTS.items()}
    repositories = {}
    configs = {}
    for model, spec in MODELS.items():
        commit = subprocess.check_output(['git', '-C', str(spec['repo']), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(spec['repo']), 'status', '--porcelain'], text=True)
        assert not status.strip(), status
        repositories[model] = {'url': spec['url'], 'commit': commit, 'label': spec['label'],
                               'code': str(spec['code']), 'base_config': spec['base']}
        for cohort in COHORTS:
            cfg = build_config(model, cohort)
            path = ROOT / 'configs' / f'{model}_{cohort}.yml'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[f'{model}_{cohort}'] = {'path': str(path), 'sha256': sha256(path)}
        # Tiny, separately marked tests. A stage transition verifies the RAM-only cache.
        cfg = build_config(model, 'phone')
        cfg.update(epoches=2, print_freq=1)
        for split, count in [('train', 16), ('val', 8)]:
            data = json.loads(Path(COHORTS['phone'][f'{split}_ann']).read_text())
            chosen = sorted(data['images'], key=lambda i: i['id'])[:count]
            ids = {i['id'] for i in chosen}
            data['images'] = chosen
            data['annotations'] = [a for a in data['annotations'] if a['image_id'] in ids]
            path = ROOT / 'smoke_data' / f'{split}.json'
            write_json(path, data)
            cfg[f'{split}_dataloader']['dataset']['ann_file'] = str(path)
            cfg[f'{split}_dataloader']['num_workers'] = 0
        if model == 'deim_s':
            cfg['train_dataloader']['collate_fn']['stop_epoch'] = 1
        smoke = ROOT / 'configs' / f'{model}_smoke.yml'
        smoke.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(ROOT / 'dataset_evidence.json', dataset_evidence)
    write_json(ROOT / 'protocol.json', {
        'frozen_at': datetime.now().astimezone().isoformat(),
        'models': repositories, 'datasets': COHORTS, 'configs': configs,
        'seeds': [0, 1, 2], 'epochs': 160, 'train_batch': 8, 'validation_batch': 16,
        'base_and_evaluation_size': [640, 640],
        'selection': 'maximum validation COCO bbox AP; earliest epoch on an exact tie',
        'checkpoint_policy': 'one global best.pth per run; stage-1 rollback snapshot in RAM only; no last or periodic weights',
        'endpoints': ['AP', 'AP50', 'AP75', 'AR100'],
        'postprocessing': 'native top-300 query-class predictions, no score cutoff, no NMS or dental priors',
        'pretraining': 'official ImageNet backbone initialization; no full COCO detector fine-tuning',
        'recipe_notes': [
            'Official DEIM-D-FINE-S uses the established dental augmentation/optimizer/schedule; upstream architecture, loss and training steps are unchanged.',
            'RT-DETRv2-S retains its official optimizer, warmup, matching and fixed-size/no-MixUp collator; it is not an all-hyperparameter-matched DEIM ablation.',
            'DEIM training uses native 480-800 multi-scale sampling around base 640 until epoch 130; evaluation is fixed 640x640.',
            'Original annotations and dataset membership are not edited. Existing internal-validation limitations remain.',
            'Existing local Plain DEIM and RT-DETR-l results are retained as separate comparators, not replaced.',
            'Only the source-backed framework comparison is new; no result-dependent recipe selection or seed exclusion.',
        ],
    })
    print(json.dumps({'prepared': str(ROOT), 'experiments': 18, 'datasets': dataset_evidence}, indent=2))


if __name__ == '__main__':
    main()
