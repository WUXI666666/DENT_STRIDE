"""Re-evaluate archived comparator predictions on frozen validation annotations (CPU)."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from hashlib import sha256
import contextlib
import importlib.util
import io
import json
import math
import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUT = Path(__file__).resolve().parent
LOCAL = Path('/home/wuxi/Dent-FINE')
PAPER = Path('/home/wuxi/Dent-FINE-paper')
CONFIG_ROOT = PAPER / 'materials/cmig_controlled_2026-09-29'
NAMES = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
         'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
FDI52 = [q * 10 + p for q in [1, 2, 3, 4] for p in range(1, 9)] + [q * 10 + p for q in [5, 6, 7, 8] for p in range(1, 6)]
PANO32 = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def canonical(dataset):
    images = [{k: r[k] for k in ['id', 'file_name', 'width', 'height']} for r in dataset['images']]
    categories = [{k: r[k] for k in ['id', 'name']} for r in dataset['categories']]
    annotations = [{k: r.get(k, 0) for k in ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'ignore']}
                   for r in dataset['annotations']]
    return {'images': sorted(images, key=lambda x: x['id']),
            'categories': sorted(categories, key=lambda x: x['id']),
            'annotations': sorted(annotations, key=lambda x: x['id'])}


def canonical_hash(dataset):
    return sha256(json.dumps(canonical(dataset), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('fixed_evaluator', CONFIG_ROOT / 'evaluate_controlled.py')
    fixed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed_module)
    paths = [
        ('Pano', 'YOLOv8-s', LOCAL / 'outputs/cross_dataset_baseline_comparison/dentex/runs/yolov8s', 'pano_factor_prototype',
         LOCAL / 'outputs/cross_dataset_baseline_comparison/dentex/yolo_dataset/export_manifest.json'),
        ('Pano', 'RT-DETR-L', LOCAL / 'outputs/cross_dataset_baseline_comparison/dentex/runs/rtdetr_l', 'pano_factor_prototype',
         LOCAL / 'outputs/cross_dataset_baseline_comparison/dentex/yolo_dataset/export_manifest.json'),
        ('Mobile', 'RT-DETR-L', LOCAL / 'outputs/cross_dataset_baseline_comparison/phone/runs/rtdetr_l', 'mobile_1024_full',
         LOCAL / 'outputs/cross_dataset_baseline_comparison/phone/yolo_dataset/export_manifest.json'),
        ('DSLR', 'YOLOv8-s', LOCAL / 'outputs/viewnet3_dslr_baseline_comparison/runs/yolov8s_fdi52_no_flip', 'dslr_cmfa_dualopt',
         LOCAL / 'outputs/viewnet3_dslr_baseline_comparison/yolo_dataset/export_manifest.json'),
        ('DSLR', 'RT-DETR-L', LOCAL / 'outputs/cross_dataset_baseline_comparison/dslr/runs/rtdetr_l', 'dslr_cmfa_dualopt',
         LOCAL / 'outputs/cross_dataset_baseline_comparison/dslr/yolo_dataset/export_manifest.json'),
    ]
    evidence = []
    for cohort, model, run, controlled_name, export_path in paths:
        metadata_path = run / 'coco_eval/coco_metrics.json'
        metadata = read(metadata_path)
        cfg = yaml.safe_load((CONFIG_ROOT / 'configs' / controlled_name / 'seed_0.yml').read_text())
        current_ann = Path(cfg['val_dataloader']['dataset']['ann_file'])
        source_ann = Path(metadata['annotation_file'])
        source, current = read(source_ann), read(current_ann)
        assert canonical(source) == canonical(current), (cohort, model, 'annotation mismatch')
        pred_path = run / 'coco_eval/coco_predictions.json'
        predictions = read(pred_path)
        valid_images = {r['id'] for r in current['images']}
        valid_categories = {r['id'] for r in current['categories']}
        assert {r['image_id'] for r in predictions} <= valid_images
        assert {r['category_id'] for r in predictions} <= valid_categories
        assert all(0 <= r['score'] <= 1 and all(math.isfinite(x) for x in r['bbox']) and r['bbox'][2] >= 0 and r['bbox'][3] >= 0 for r in predictions)
        args = yaml.safe_load((run / 'args.yaml').read_text())
        assert args['epochs'] == 160 and args['seed'] == 0
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            gt = COCO(str(current_ann)); dt = gt.loadRes(predictions)
            evaluator = COCOeval(gt, dt, 'bbox')
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        stats = {k: float(v) for k, v in zip(NAMES, evaluator.stats)}
        for key in ['AP', 'AP50', 'AP75', 'AR100']:
            assert abs(stats[key] - metadata['metrics'][key]) < 1e-12, (model, cohort, key)
        fixed, counts = fixed_module.fixed_endpoints(predictions, gt)
        export = read(export_path) if export_path.is_file() else None
        class_map = PANO32 if cohort == 'Pano' else FDI52
        class_map_valid = export is None or len(export['classes']) == len(class_map)
        if cohort != 'Pano' and export is not None:
            assert [int(export['classes'][str(i)]) for i in range(len(class_map))] == class_map
        if export is not None:
            assert canonical(read(export['val_annotation_source'])) == canonical(current)
        source_only_annotation_fields = sorted(set().union(*(set(r) for r in source['annotations'])) - set().union(*(set(r) for r in current['annotations'])))
        current_only_annotation_fields = sorted(set().union(*(set(r) for r in current['annotations'])) - set().union(*(set(r) for r in source['annotations'])))
        item = {'cohort': cohort, 'model': model, 'prediction_path': str(pred_path), 'prediction_sha256': sha(pred_path),
                'metadata_path': str(metadata_path), 'weight_path': metadata['weights'],
                'weight_sha256': sha(metadata['weights']) if Path(metadata['weights']).is_file() else None,
                'source_annotation_path': str(source_ann), 'source_annotation_sha256': sha(source_ann),
                'current_annotation_path': str(current_ann), 'current_annotation_sha256': sha(current_ann),
                'serialized_annotations_identical': sha(source_ann) == sha(current_ann),
                'COCO_bbox_content_identical': True, 'canonical_annotation_sha256': canonical_hash(current),
                'source_only_annotation_fields': source_only_annotation_fields,
                'current_only_annotation_fields': current_only_annotation_fields,
                'image_dir': metadata['image_dir'], 'GT_image_count': len(valid_images),
                'GT_instance_count': len(current['annotations']), 'prediction_count': len(predictions),
                'unique_prediction_image_count': len({r['image_id'] for r in predictions}),
                'category_ids_valid': True, 'category_map_FDI': class_map,
                'training_export_manifest': str(export_path) if export else None,
                'class_map_matches_training_export': class_map_valid,
                'reevaluated_COCO_metrics_fraction': stats,
                'reevaluated_COCO_metrics_percent': {k: 100 * v for k, v in stats.items()},
                'fixed_threshold': fixed,
                'params_total': metadata['num_parameters'], 'seed': args['seed'], 'epochs': args['epochs'],
                'training_input': args['imgsz'], 'validation_input': metadata['inference']['imgsz'],
                'prediction_threshold': metadata['inference']['confidence'],
                'prediction_limit': metadata['inference']['max_det'],
                'native_postprocessing': metadata['inference'], 'framework': metadata['framework'],
                'pretraining_path': args['model'], 'reuse_for_same_image_qualitative': True,
                'comparison_type': 'Archived seed-0 framework comparator. Same validation annotations/category mapping; native training, input and output rules differ.'}
        evidence.append(item)
        directory = OUT / 'external_reevaluation' / (cohort + '_' + model.replace('-', '_'))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'coco_evaluation.log').write_text(capture.getvalue())
        (directory / 'per_image_counts.json').write_text(json.dumps(counts, indent=2))
        np.savez_compressed(directory / 'coco_precision_recall.npz', precision=evaluator.eval['precision'], recall=evaluator.eval['recall'],
                            scores=evaluator.eval['scores'], category_ids=evaluator.params.catIds,
                            IoU_thresholds=evaluator.params.iouThrs, recall_thresholds=evaluator.params.recThrs)
        print(cohort, model, item['reevaluated_COCO_metrics_percent']['AP'], item['serialized_annotations_identical'], flush=True)
    payload = {'snapshot_time_Asia_Shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
               'entries': evidence,
               'main_visual_table_scope': {'Table3': 'Single-run local DEIM-D-FINE, official DEIM-s, RT-DETRv2-S and historical selected recipe in each cohort.',
                                          'Table4': 'DSLR 10-method broad comparator set including RT-DETR-L.',
                                          'supplement_TableS5': 'Pano/Mobile broad 6-method comparator sets including RT-DETR-L; Pano highest broad AP is RT-DETR-L51.978, not YOLOv8-s51.431.'}}
    (OUT / 'external_prediction_sources.json').write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
