"""Shared, source-backed protocol for the eight Mobile 1024 baselines."""
from __future__ import annotations
from pathlib import Path
import contextlib
import gzip
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/wuxi/Dent-FINE')
OLD = Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10')
ANN = Path('/home/wuxi/coco/valid/annotations_fdi_types.json')
TRAIN_ANN = Path('/home/wuxi/coco/train/annotations_fdi_types.json')
MODELS = ['deim_s', 'rtdetr_l', 'rtdetrv2_s', 'faster_rcnn', 'caf_yolo_s', 'yolov8s', 'yolo11s', 'dome_detr_s']
NAMES = dict(zip(MODELS, ['Official DEIM-s', 'RT-DETR-L', 'RT-DETRv2-S', 'Faster R-CNN R50-FPN',
                          'CAF-YOLO-s', 'YOLOv8-s', 'YOLO11-s', 'Dome-DETR-S']))
KEYS = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
PYTHON = '/home/wuxi/miniconda3/envs/deimv2/bin/python'
FASTER_PYTHON = '/home/wuxi/miniconda3/envs/mask2former/bin/python'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    pending.replace(path)


def run_dir(family):
    return ROOT / 'runs' / family / 'seed_0'


def disk_guard():
    free = shutil.disk_usage(ROOT).free
    if free < 3*1024**3:
        raise RuntimeError('Less than 3 GiB free: stop without deleting old data; retain best checkpoint.')
    return free


def coco_metrics(predictions, destination=None):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': list(gt.imgs.values()), 'categories': list(gt.cats.values()), 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(gt.imgs)
        ev.params.catIds = sorted(gt.cats)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    values = dict(zip(KEYS, map(float, ev.stats)))
    if destination:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/'coco_arrays.npz', precision=ev.eval['precision'],
                            recall=ev.eval['recall'], scores=ev.eval['scores'],
                            category_ids=ev.params.catIds, iou_thresholds=ev.params.iouThrs,
                            recall_thresholds=ev.params.recThrs)
        with gzip.open(destination/'predictions.coco.json.gz', 'wt') as f:
            json.dump(predictions, f, separators=(',', ':'))
    return values


def fixed_metrics(predictions, destination):
    import importlib.util
    from pycocotools.coco import COCO
    source = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11/evaluate_controlled.py')
    spec = importlib.util.spec_from_file_location('existing_fixed_endpoints', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN))
    metrics, rows = module.fixed_endpoints(predictions, gt, score=.30, iou=.50)
    write(Path(destination)/'per_image_counts.json', rows)
    return metrics
