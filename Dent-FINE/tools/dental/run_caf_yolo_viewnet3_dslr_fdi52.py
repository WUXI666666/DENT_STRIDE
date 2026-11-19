#!/usr/bin/env python3
"""Train the official CAF-YOLO implementation on View-Net3 DSLR FDI52.

The upstream CAF-YOLO repository is a fork of Ultralytics 8.0.228 that adds a
CAFMAttention block to the P4 feature level.  It is deliberately imported from
``third_party/CAF-YOLO`` rather than the globally installed Ultralytics package.

This launcher follows the existing DSLR baseline protocol:

* canonical 750/190-image FDI52 train/validation split;
* 160 epochs, 640px input, batch size 8, seed 0, AdamW;
* no horizontal or vertical flip, because flips invalidate FDI tooth-position
  labels; and
* the upstream author's non-AMP training setting.  Their vendored Ultralytics
  AMP self-test depends on a demonstration image that is not distributed in
  the official CAF-YOLO source archive, whereas the CAF training entry point
  explicitly uses ``amp=False``;
* the official YOLOv8-s COCO checkpoint is *partially* transferred into the
  CAF-YOLO-s model.  The incompatible 80-to-52-class detection-head tensors
  are intentionally not transferred by Ultralytics' shape-intersection loader.

It is safe to invoke again after an interruption: a valid ``weights/last.pt``
is resumed.  When training and the common COCO re-evaluation both complete,
only ``weights/best.pt`` is retained.  The generic summary script can then read
``coco_eval/coco_metrics.json`` from this run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT_DIR / "outputs" / "viewnet3_dslr_baseline_comparison"
DEFAULT_SOURCE_DIR = ROOT_DIR / "third_party" / "CAF-YOLO"
DEFAULT_DATA = DEFAULT_BASE / "yolo_dataset" / "viewnet3_dslr_fdi52.yaml"
DEFAULT_WEIGHTS = DEFAULT_BASE / "weights" / "yolov8s.pt"
DEFAULT_RUN_DIR = DEFAULT_BASE / "runs" / "caf_yolo_s_fdi52"
DEFAULT_ANN = ROOT_DIR / "datasets" / "viewnet3_dslr_fdi52" / "annotations" / "tooth_instances_val_fdi52.json"
DEFAULT_IMAGES = Path("/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val")
EVALUATOR = ROOT_DIR / "tools" / "dental" / "evaluate_yolo_coco_detection.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ann-file", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for COCO re-evaluation")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="Visible CUDA device index (default: GPU 0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the CAF-YOLO-s graph and partially preload YOLOv8-s without training an epoch.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run common pycocotools re-evaluation after a successful training run.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str, *, directory: bool = False) -> None:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        noun = "directory" if directory else "file"
        raise FileNotFoundError(f"Missing {description} {noun}: {path}")


def get_source_revision(source_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_scale_s_alias(source_dir: Path, run_dir: Path) -> Path:
    """Make a non-invasive filename alias so upstream scale inference selects ``s``.

    CAF-YOLO's official YAML is named ``yolov8-CAFMAttention.yaml``.  Upstream
    Ultralytics determines the compound scale from the filename, and would
    otherwise silently choose the first (nano) scale.  A symlink named
    ``yolov8s-CAFMAttention.yaml`` retains the exact official YAML content while
    making the intended small model explicit.
    """

    source_yaml = source_dir / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-CAFMAttention.yaml"
    require_path(source_yaml, "official CAF-YOLO model YAML")
    alias_dir = run_dir / "model_config"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / "yolov8s-CAFMAttention.yaml"
    if alias.is_symlink() and alias.resolve() == source_yaml.resolve():
        return alias
    if alias.exists() or alias.is_symlink():
        # This is a narrow, recoverable replacement of an auto-generated alias,
        # never a user-provided configuration file.
        alias.unlink()
    alias.symlink_to(source_yaml)
    return alias


def import_source_yolo(source_dir: Path):
    """Import YOLO from the checked-out CAF-YOLO fork, never global Ultralytics."""

    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    # This script starts a fresh interpreter, but guard against an interactive
    # invocation which previously imported the globally installed package.
    for name in tuple(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            module_file = getattr(sys.modules[name], "__file__", "") or ""
            if module_file and not str(Path(module_file).resolve()).startswith(source):
                del sys.modules[name]
    import ultralytics  # pylint: disable=import-outside-toplevel
    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
    from ultralytics.utils import SETTINGS  # pylint: disable=import-outside-toplevel

    module_file = Path(ultralytics.__file__).resolve()
    if not str(module_file).startswith(source):
        raise RuntimeError(f"CAF-YOLO import isolation failed: resolved ultralytics to {module_file}")

    # The upstream 8.0.228 fork auto-registers third-party tracking callbacks
    # from its mutable SETTINGS map.  In this workspace a newer wandb rejects
    # an absolute local output directory as a project name before epoch zero.
    # This experiment is intentionally local/reproducible, so disable external
    # services before Trainer imports the integrations. TensorBoard remains on.
    for integration in ("wandb", "clearml", "comet", "mlflow", "neptune", "raytune", "dvc"):
        if integration in SETTINGS:
            SETTINGS[integration] = False
    return YOLO, ultralytics


def install_compatibility_shims(epochs: int) -> None:
    """Bridge two upstream-version gaps without changing CAF-YOLO itself.

    CAF-YOLO targets a NumPy release where ``np.trapz`` still exists.  NumPy
    2.x renamed it to ``trapezoid``; the aliases are mathematically identical.
    Its resume path also intentionally restores every checkpoint argument and
    only permits image/batch overrides.  Permit the queue's explicit epoch
    budget to override an old default patience value, so a recovered run still
    satisfies the stated 160-epoch protocol.
    """

    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    from ultralytics.engine.trainer import BaseTrainer  # pylint: disable=import-outside-toplevel

    if getattr(BaseTrainer.check_resume, "_dentfine_protocol_shim", False):
        return
    original_check_resume = BaseTrainer.check_resume

    def patched_check_resume(self, overrides):
        original_check_resume(self, overrides)
        if self.resume:
            self.args.epochs = epochs
            self.args.patience = epochs

    patched_check_resume._dentfine_protocol_shim = True  # type: ignore[attr-defined]
    BaseTrainer.check_resume = patched_check_resume


def build_pretrained_model(YOLO: Any, model_yaml: Path, weights: Path):
    """Build CAF-YOLO-s and let the upstream loader transfer compatible tensors."""

    model = YOLO(str(model_yaml))
    scale = getattr(model.model, "yaml", {}).get("scale")
    if scale != "s":
        raise RuntimeError(f"CAF-YOLO model scale must be 's', got {scale!r} from {model_yaml}")
    # ``Model.load`` calls the upstream DetectionModel.load(), which intersects
    # keys and shapes.  Thus the YOLOv8-s backbone/neck weights transfer while
    # class-dependent detector tensors remain newly initialized for FDI52.
    model.load(str(weights))
    return model


def parameter_count(model: Any) -> int:
    network = getattr(model, "model", model)
    return sum(parameter.numel() for parameter in network.parameters())


def build_fdi52_target_model(source_model: Any):
    """Mirror the upstream trainer's 52-class reconstruction for smoke checks."""

    from ultralytics.nn.tasks import DetectionModel  # pylint: disable=import-outside-toplevel

    target = DetectionModel(deepcopy(source_model.model.yaml), nc=52, verbose=False)
    target.load(source_model.model)
    return target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_inputs(args: argparse.Namespace) -> None:
    require_path(args.source_dir, "CAF-YOLO source", directory=True)
    require_path(args.data, "FDI52 data YAML")
    require_path(args.weights, "YOLOv8-s initialization weight")
    if not args.smoke_test:
        require_path(args.ann_file, "validation COCO annotations")
        require_path(args.image_dir, "validation image", directory=True)
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs, imgsz and batch must be positive; workers must be non-negative")


def run_common_evaluator(args: argparse.Namespace, best_weight: Path) -> None:
    """Run the existing FDI52 COCO evaluator under the same CAF-YOLO import path."""

    require_path(EVALUATOR, "common YOLO COCO evaluator")
    env = os.environ.copy()
    source = str(args.source_dir.resolve())
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        args.python,
        str(EVALUATOR),
        "--weights",
        str(best_weight),
        "--ann-file",
        str(args.ann_file),
        "--image-dir",
        str(args.image_dir),
        "--output-dir",
        str(args.run_dir / "coco_eval"),
        "--imgsz",
        str(args.imgsz),
        "--device",
        str(args.device),
        "--batch",
        str(args.batch),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    ]
    subprocess.run(command, check=True, env=env)


def retain_best_weight(run_dir: Path) -> None:
    """Prune only transient CAF-YOLO checkpoint files after a fully completed run."""

    weights_dir = run_dir / "weights"
    best_weight = weights_dir / "best.pt"
    if not best_weight.is_file():
        raise FileNotFoundError(f"Refusing checkpoint cleanup: selected best weight is absent: {best_weight}")
    for checkpoint in weights_dir.glob("*.pt"):
        if checkpoint.resolve() != best_weight.resolve():
            checkpoint.unlink()


def smoke_test(args: argparse.Namespace, YOLO: Any, ultralytics: Any, model_yaml: Path) -> None:
    model = build_pretrained_model(YOLO, model_yaml, args.weights)
    target = build_fdi52_target_model(model)
    detect_head = target.model[-1]
    target.eval()
    with torch.inference_mode():
        smoke_output = target(torch.zeros((1, 3, 64, 64)))
    if not isinstance(smoke_output, tuple) or len(smoke_output) < 1:
        raise RuntimeError("CAF-YOLO FDI52 smoke forward returned an unexpected output type")
    payload = {
        "status": "passed",
        "source_dir": str(args.source_dir.resolve()),
        "source_revision": get_source_revision(args.source_dir),
        "ultralytics_version": getattr(ultralytics, "__version__", None),
        "model_yaml": str(model_yaml.resolve()),
        "model_scale": model.model.yaml.get("scale"),
        "num_parameters": parameter_count(target),
        "initialization": {
            "source_weight": str(args.weights.resolve()),
            "mode": "upstream shape-intersection partial transfer",
            "class_count_after_build": int(detect_head.nc),
            "forward_shape": list(smoke_output[0].shape),
        },
        "training_would_use": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(args.device),
            "seed": args.seed,
            "optimizer": "AdamW",
            "amp": False,
            "flipud": 0.0,
            "fliplr": 0.0,
        },
    }
    write_json(args.run_dir / "caf_yolo_smoke_test.json", payload)
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.data = args.data.resolve()
    args.weights = args.weights.resolve()
    args.run_dir = args.run_dir.resolve()
    args.ann_file = args.ann_file.resolve()
    args.image_dir = args.image_dir.resolve()
    verify_inputs(args)

    # Keep the lifecycle markers directly in the result directory so the live
    # baseline table can safely distinguish interrupted, failed and completed runs.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = make_scale_s_alias(args.source_dir, args.run_dir)
    YOLO, ultralytics = import_source_yolo(args.source_dir)
    install_compatibility_shims(args.epochs)
    if args.smoke_test:
        smoke_test(args, YOLO, ultralytics, model_yaml)
        return

    weights_dir = args.run_dir / "weights"
    last_weight = weights_dir / "last.pt"
    best_weight = weights_dir / "best.pt"
    completed = args.run_dir / "COMPLETED"
    running = args.run_dir / "RUNNING"
    failed = args.run_dir / "FAILED"

    # A completed train can be re-evaluated after an evaluator interruption, but
    # never retrained or overwritten.  It is then pruned again defensively.
    if completed.is_file() and best_weight.is_file():
        if not args.skip_eval and not (args.run_dir / "coco_eval" / "coco_metrics.json").is_file():
            run_common_evaluator(args, best_weight)
        retain_best_weight(args.run_dir)
        print(f"CAF-YOLO-s is already complete: {args.run_dir}")
        return

    if (args.run_dir / "results.csv").is_file() and not last_weight.is_file():
        raise RuntimeError(
            f"Refusing to overwrite a partial CAF-YOLO run without resumable last.pt: {args.run_dir}. "
            "Inspect or recover that directory before restarting."
        )

    failed.unlink(missing_ok=True)
    completed.unlink(missing_ok=True)
    running.touch()
    try:
        if last_weight.is_file():
            print(f"Resuming CAF-YOLO-s from {last_weight}")
            model = YOLO(str(last_weight))
            model.train(resume=True, patience=args.epochs)
        else:
            print("Starting CAF-YOLO-s with partial YOLOv8-s initialization")
            model = build_pretrained_model(YOLO, model_yaml, args.weights)
            initialization = {
                "source_dir": str(args.source_dir),
                "source_revision": get_source_revision(args.source_dir),
                "ultralytics_version": getattr(ultralytics, "__version__", None),
                "model_yaml": str(model_yaml.resolve()),
                "model_scale": model.model.yaml.get("scale"),
                "num_parameters": parameter_count(model),
                "pretrained_weight": str(args.weights),
                "pretrained_transfer": "upstream shape-intersection partial transfer",
                "optimizer": "AdamW",
                "amp": False,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "seed": args.seed,
                "flipud": 0.0,
                "fliplr": 0.0,
            }
            write_json(args.run_dir / "initialization.json", initialization)
            model.train(
                data=str(args.data),
                epochs=args.epochs,
                # Prevent early stopping from shortening the explicitly
                # declared 160-epoch paper protocol.
                patience=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                project=str(args.run_dir.parent),
                name=args.run_dir.name,
                exist_ok=True,
                pretrained=True,
                optimizer="AdamW",
                lr0=0.0002,
                momentum=0.9,
                weight_decay=0.0001,
                warmup_epochs=3.0,
                warmup_bias_lr=0.0,
                close_mosaic=30,
                cos_lr=False,
                seed=args.seed,
                deterministic=True,
                # Preserve the authors' supplied CAF-YOLO training setting.
                # It also avoids the fork's unrelated missing-demo-image AMP
                # validator before the first real dataset batch.
                amp=False,
                cache=False,
                rect=False,
                save=True,
                save_period=-1,
                plots=True,
                flipud=0.0,
                fliplr=0.0,
            )

        if not best_weight.is_file():
            raise RuntimeError(f"CAF-YOLO training returned without a selected best.pt: {best_weight}")
        if not args.skip_eval:
            run_common_evaluator(args, best_weight)
        retain_best_weight(args.run_dir)
        running.unlink(missing_ok=True)
        completed.touch()
    except BaseException:
        running.unlink(missing_ok=True)
        failed.touch()
        raise


if __name__ == "__main__":
    main()
