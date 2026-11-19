"""Canonical image IDs and unrounded coordinates, respecting each native validator."""
from __future__ import annotations
import math
from pathlib import Path


def canonical_predictions(predn, batch_or_filename, image_ids, *, rtdetr=False, imgsz=1024):
    metadata = batch_or_filename if isinstance(batch_or_filename, dict) else None
    filename = metadata['im_file'] if metadata is not None else batch_or_filename
    image_id = image_ids[Path(filename).name]
    if isinstance(predn, dict):
        boxes = predn['bboxes'].detach().clone().cpu()
        scores = predn['conf'].detach().cpu().tolist()
        classes = predn['cls'].detach().cpu().tolist()
    else:
        rows = predn.detach().cpu()
        boxes = rows[:, :4].clone()
        scores = rows[:, 4].tolist()
        classes = rows[:, 5].tolist()
    if rtdetr:
        # RTDETRValidator.scale_preds is deliberately a no-op. Unlike YOLO,
        # its pred_to_json performs independent width/height scaling here.
        if metadata is None:
            raise RuntimeError('RT-DETR export requires original image dimensions')
        height, width = metadata['ori_shape']
        if not isinstance(imgsz, (int, float)):
            raise RuntimeError('The fixed-square RT-DETR comparison expects scalar imgsz')
        boxes[:, [0, 2]] *= float(width) / float(imgsz)
        boxes[:, [1, 3]] *= float(height) / float(imgsz)
    result = []
    for (x1, y1, x2, y2), score, cls in zip(boxes.tolist(), scores, classes):
        if not 0 <= int(cls) < 52:
            raise RuntimeError(f'Unexpected class index {cls}')
        if not all(math.isfinite(x) for x in [x1, y1, x2, y2, score]):
            raise FloatingPointError('Non-finite prediction')
        result.append({'image_id': image_id, 'category_id': int(cls),
                       'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(score)})
    return result


def assert_native_coordinates(canonical, native, class_map=None):
    """Compare geometry/classes/scores while allowing native JSON rounding."""
    if len(canonical) != len(native):
        raise AssertionError('Native and canonical exports have different detection counts')
    for actual, expected in zip(canonical, native):
        native_class = actual['category_id'] if class_map is None else class_map[actual['category_id']]
        assert native_class == expected['category_id']
        assert abs(actual['score'] - expected['score']) <= 0.0000051
        for a, b in zip(actual['bbox'], expected['bbox']):
            # Native xyxy->center-xywh->top-left arithmetic is FP32 before
            # rounding to 0.001 px; canonical Python subtraction is unrounded.
            assert abs(a-b) <= 0.002, (actual, expected)
