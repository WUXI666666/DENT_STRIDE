#!/usr/bin/env python3
"""Train/evaluate YOLOv8-s, YOLO11-s, RT-DETR-l and CAF-YOLO-s.

The launcher applies one paper-facing protocol to a supplied contiguous
zero-based COCO/YOLO dental dataset: 160 epochs, 640 px, seed 0, AdamW, and no
geometric flips. Completed runs are resumable and retain their selected best
checkpoint plus common COCO prediction/metric JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/wuxi/miniconda3/envs/deimv2/bin/python")
YOLO = Path("/home/wuxi/miniconda3/envs/deimv2/bin/yolo")
EVALUATOR = ROOT / "tools/dental/evaluate_yolo_coco_detection.py"
CAF_RUNNER = ROOT / "tools/dental/run_caf_yolo_viewnet3_dslr_fdi52.py"
WEIGHTS = ROOT / "outputs/viewnet3_dslr_baseline_comparison/weights"


def command(args: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def evaluate(
    weights: Path,
    annotation: Path,
    images: Path,
    output: Path,
    family: str,
    batch: int,
    env: dict[str, str],
) -> None:
    command([
        str(PYTHON), str(EVALUATOR), "--weights", str(weights),
        "--ann-file", str(annotation), "--image-dir", str(images),
        "--output-dir", str(output), "--imgsz", "640", "--device", "0",
        "--batch", str(batch), "--conf", "0.001", "--iou", "0.7",
        "--max-det", "300", "--model-family", family,
    ], env)


def train_ultralytics(
    name: str,
    init_weight: str,
    family: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    run = args.output_dir / "runs" / name
    best, last = run / "weights/best.pt", run / "weights/last.pt"
    metrics = run / "coco_eval/coco_metrics.json"
    if (run / "COMPLETED").is_file() and best.is_file() and metrics.is_file():
        print(f"[skip] {name}", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    (run / "RUNNING").touch()
    (run / "FAILED").unlink(missing_ok=True)
    try:
        if last.is_file() and family == "rtdetr":
            # The generic ``yolo detect train resume=...`` CLI reconstructs a
            # DetectionModel before reading the checkpoint and therefore
            # mistakes RT-DETR's decoder for a YOLO Detect head.  Load through
            # the architecture-specific API so the serialized trainer state is
            # restored without changing the model family.
            train = [
                str(PYTHON), "-c",
                (
                    "from ultralytics import RTDETR; "
                    f"RTDETR({str(last)!r}).train(resume=True, device=0)"
                ),
            ]
        elif last.is_file():
            train = [str(YOLO), "detect", "train", f"resume={last}", "device=0"]
        else:
            train = [
                str(YOLO), "detect", "train", f"model={init_weight}", f"data={args.data}",
                f"epochs={args.epochs}", "imgsz=640", f"batch={args.batch}",
                f"workers={args.workers}", "device=0", f"project={args.output_dir / 'runs'}",
                f"name={name}", "exist_ok=True", "pretrained=True", "optimizer=AdamW",
                "lr0=0.0002", "momentum=0.9", "weight_decay=0.0001",
                "warmup_epochs=3.0", "warmup_bias_lr=0.0", "cos_lr=False",
                f"patience={args.epochs}", "seed=0", "deterministic=True", "amp=True",
                "cache=False", "rect=False", "save=True", "save_period=-1",
                "plots=True", "flipud=0.0", "fliplr=0.0",
            ]
        command(train, env)
        if not best.is_file():
            raise FileNotFoundError(f"Training did not create {best}")
        evaluate(best, args.annotation, args.images, run / "coco_eval", family, args.batch, env)
        for checkpoint in (run / "weights").glob("*.pt"):
            if checkpoint != best:
                checkpoint.unlink()
        (run / "RUNNING").unlink(missing_ok=True)
        (run / "FAILED").unlink(missing_ok=True)
        (run / "COMPLETED").touch()
    except Exception:
        (run / "RUNNING").unlink(missing_ok=True)
        (run / "FAILED").touch()
        raise


def train_caf(args: argparse.Namespace, env: dict[str, str]) -> None:
    run = args.output_dir / "runs/caf_yolo_s"
    command([
        str(PYTHON), str(CAF_RUNNER), "--data", str(args.data),
        "--weights", str(WEIGHTS / "yolov8s.pt"), "--run-dir", str(run),
        "--ann-file", str(args.annotation), "--image-dir", str(args.images),
        "--python", str(PYTHON), "--epochs", str(args.epochs), "--imgsz", "640",
        "--batch", str(args.batch), "--workers", str(args.workers), "--device", "0",
        "--seed", "0",
    ], env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-from", choices=("yolov8s", "yolo11s", "rtdetr_l", "caf_yolo_s"), default="yolov8s")
    parser.add_argument("--only", choices=("yolov8s", "yolo11s", "rtdetr_l", "caf_yolo_s"),
                        help="Run exactly one family (useful for filling a single missing cohort/model cell).")
    args = parser.parse_args()
    args.data, args.annotation, args.images, args.output_dir = (
        args.data.resolve(), args.annotation.resolve(), args.images.resolve(), args.output_dir.resolve()
    )
    for path in (args.data, args.annotation):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.images.is_dir():
        raise FileNotFoundError(args.images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    order = ["yolov8s", "yolo11s", "rtdetr_l", "caf_yolo_s"]
    selected = [args.only] if args.only else order[order.index(args.start_from):]
    if "yolov8s" in selected:
        train_ultralytics("yolov8s", str(WEIGHTS / "yolov8s.pt"), "yolo", args, env)
    if "yolo11s" in selected:
        train_ultralytics("yolo11s", str(WEIGHTS / "yolo11s.pt"), "yolo", args, env)
    if "rtdetr_l" in selected:
        train_ultralytics("rtdetr_l", str(ROOT / "weights/rtdetr-l.pt"), "rtdetr", args, env)
    if "caf_yolo_s" in selected:
        train_caf(args, env)
    summary = {
        "dataset_yaml": str(args.data), "annotation": str(args.annotation),
        "images": str(args.images), "gpu": args.gpu, "epochs": args.epochs,
        "input": 640, "seed": 0, "flips": False, "models": selected,
    }
    (args.output_dir / "yolo_family_suite.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
