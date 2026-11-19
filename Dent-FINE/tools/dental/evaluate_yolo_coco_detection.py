#!/usr/bin/env python3
"""Evaluate a trained Ultralytics detection model with COCO bbox metrics.

This writes predictions in the same canonical FDI52 category IDs as the
View-Net3 COCO validation annotations, then evaluates with pycocotools. It
therefore supplies AP/AP50/AP75/AR100 on the same protocol used by DEIM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import RTDETR, YOLO
from ultralytics import __version__ as ultralytics_version


METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--model-family", choices=("yolo", "rtdetr"), default="yolo")
    return parser.parse_args()


def chunks(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def model_parameter_count(model: Any) -> int:
    network = getattr(model, "model", None)
    if network is None:
        return 0
    return sum(parameter.numel() for parameter in network.parameters())


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing model weights: {args.weights}")
    if not args.ann_file.is_file():
        raise FileNotFoundError(f"Missing annotations: {args.ann_file}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {args.image_dir}")

    coco_gt = COCO(str(args.ann_file))
    category_ids = sorted(coco_gt.getCatIds())
    if category_ids != list(range(len(category_ids))):
        raise ValueError(f"Expected contiguous zero-based categories, found {category_ids}")
    image_ids = sorted(coco_gt.getImgIds())
    images = [coco_gt.loadImgs(image_id)[0] for image_id in image_ids]
    for image in images:
        image_path = args.image_dir / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Annotation image is absent: {image_path}")

    model = RTDETR(str(args.weights)) if args.model_family == "rtdetr" else YOLO(str(args.weights))
    predictions: list[dict[str, Any]] = []
    for batch_images in chunks(images, args.batch):
        paths = [str(args.image_dir / str(image["file_name"])) for image in batch_images]
        results = model.predict(
            source=paths,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            stream=False,
        )
        if len(results) != len(batch_images):
            raise RuntimeError(f"Prediction count mismatch: {len(results)} results for {len(batch_images)} images")
        for image, result in zip(batch_images, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            width = float(image["width"])
            height = float(image["height"])
            for raw_box, score, class_index in zip(xyxy, scores, classes):
                category_id = int(class_index)
                if category_id not in category_ids:
                    raise ValueError(f"Model emitted invalid FDI52 index {category_id}")
                x1, y1, x2, y2 = (float(value) for value in raw_box)
                x1 = min(max(x1, 0.0), width)
                y1 = min(max(y1, 0.0), height)
                x2 = min(max(x2, 0.0), width)
                y2 = min(max(y2, 0.0), height)
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0.0 or box_height <= 0.0:
                    continue
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, box_width, box_height],
                        "score": float(score),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "coco_predictions.json"
    predictions_path.write_text(json.dumps(predictions) + "\n")
    if not predictions:
        raise RuntimeError("No predictions were produced; refusing to report misleading COCO metrics")

    coco_dt = coco_gt.loadRes(str(predictions_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats.tolist())}
    metrics = {
        "framework": "Ultralytics",
        "model_family": args.model_family,
        "ultralytics_version": ultralytics_version,
        "weights": str(args.weights.resolve()),
        "annotation_file": str(args.ann_file.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "image_count": len(images),
        "prediction_count": len(predictions),
        "num_parameters": model_parameter_count(model),
        "inference": {
            "imgsz": args.imgsz,
            "confidence": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": str(args.device),
        },
        "metrics": stats,
        "prediction_file": str(predictions_path.resolve()),
    }
    metrics_path = args.output_dir / "coco_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # Avoid noisy device-selection edge cases when this is invoked by the
    # serial shell queue; Ultralytics itself receives the selected device.
    torch.set_grad_enabled(False)
    main()
