#!/usr/bin/env python3
"""Train a bbox-only Faster R-CNN baseline on View-Net3 DSLR FDI52.

This script deliberately keeps the conventional Detectron2 Faster R-CNN
R50-FPN architecture separate from Dent-FINE.  It uses the exact canonical
FDI52 COCO JSON files used by the existing DSLR baselines, evaluates once per
nominal epoch, and selects the checkpoint with the highest validation COCO
bbox AP.  At successful completion the output directory contains exactly one
model checkpoint: ``model_best.pth``.

The final ``coco_eval/coco_metrics.json`` stores AP/AP50/AP75/AR100 on the
same 0--1 scale as the existing YOLO COCO-metric artifacts.  The raw
Detectron2 percentages are retained separately for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


ROOT_DIR = Path("/home/wuxi/Dent-FINE")
DEFAULT_TRAIN_ANN = ROOT_DIR / "datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_train_fdi52.json"
DEFAULT_VAL_ANN = ROOT_DIR / "datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json"
DEFAULT_TRAIN_IMAGES = Path(
    "/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/train"
)
DEFAULT_VAL_IMAGES = Path(
    "/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val"
)
DEFAULT_OUTPUT = ROOT_DIR / "outputs/viewnet3_dslr_baseline_comparison/runs/faster_rcnn_r50_fpn_fdi52"

MODEL_NAME = "Faster R-CNN R50-FPN"
MODEL_ZOO_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
BEST_CHECKPOINT_NAME = "model_best"
NUM_CLASSES = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bbox-only Detectron2 Faster R-CNN R50-FPN on View-Net3 DSLR FDI52."
    )
    parser.add_argument("--train-ann", type=Path, default=DEFAULT_TRAIN_ANN)
    parser.add_argument("--val-ann", type=Path, default=DEFAULT_VAL_ANN)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-classes", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--base-lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument(
        "--lr-drop-epochs",
        type=int,
        nargs="+",
        default=(120, 150),
        help="Epochs at which the SGD learning rate is multiplied by GAMMA (default: 120 150).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU index. The script enforces CUDA_VISIBLE_DEVICES to this value.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional pretrained checkpoint/path. Defaults to Detectron2's official COCO Faster R-CNN R50-FPN 3x checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the best completed epoch checkpoint if one is present after an interrupted run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a new run to reuse a non-completed output directory without resuming.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow rerunning an output directory already marked COMPLETED.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate annotations, images, label mapping, and the derived schedule without constructing a model or training.",
    )
    return parser.parse_args()


def validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def resolve_and_validate_args(args: argparse.Namespace) -> None:
    for field in ("train_ann", "val_ann", "train_images", "val_images", "output_dir"):
        setattr(args, field, Path(getattr(args, field)).expanduser().resolve())
    for field in ("train_ann", "val_ann"):
        path = getattr(args, field)
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")
    for field in ("train_images", "val_images"):
        path = getattr(args, field)
        if not path.is_dir():
            raise NotADirectoryError(f"Missing image directory: {path}")
    for value, name in (
        (args.num_classes, "--num-classes"),
        (args.epochs, "--epochs"),
        (args.batch_size, "--batch-size"),
        (args.min_size, "--min-size"),
        (args.max_size, "--max-size"),
    ):
        validate_positive(value, name)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative")
    if args.max_size < args.min_size:
        raise ValueError("--max-size must be at least --min-size")
    if args.base_lr <= 0.0:
        raise ValueError("--base-lr must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("--momentum must be in [0, 1)")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    invalid_drops = [epoch for epoch in args.lr_drop_epochs if epoch <= 0 or epoch >= args.epochs]
    if invalid_drops:
        raise ValueError(
            "Every --lr-drop-epochs value must lie in [1, epochs - 1]; "
            f"got {invalid_drops} for {args.epochs} epochs"
        )


def verify_coco_fdi52(path: Path, image_dir: Path) -> dict[str, Any]:
    """Validate the canonical JSON before Detectron2 consumes it.

    The DSLR annotations intentionally use class IDs 0--51.  COCO's common
    positive-ID convention does not apply here, but both Detectron2 and
    pycocotools handle this mapping correctly when category metadata is
    registered explicitly.  Validate it up front so a malformed split cannot
    silently train an incompatible classifier.
    """

    payload = json.loads(path.read_text())
    required = {"images", "annotations", "categories"}
    absent = required.difference(payload)
    if absent:
        raise ValueError(f"{path} is not a COCO annotation file; absent keys: {sorted(absent)}")
    category_ids = sorted(int(category["id"]) for category in payload["categories"])
    expected = list(range(NUM_CLASSES))
    if category_ids != expected:
        raise ValueError(
            f"{path} must expose exactly canonical category IDs {expected[0]}..{expected[-1]}, "
            f"found {category_ids}"
        )
    image_ids = [int(image["id"]) for image in payload["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path} contains duplicate image IDs")
    image_files = [image_dir / str(image["file_name"]) for image in payload["images"]]
    missing_files = [file for file in image_files if not file.is_file()]
    if missing_files:
        examples = ", ".join(str(file) for file in missing_files[:3])
        raise FileNotFoundError(f"{len(missing_files)} annotated image files are missing, e.g. {examples}")
    image_id_set = set(image_ids)
    annotation_category_ids = {int(annotation["category_id"]) for annotation in payload["annotations"]}
    if not annotation_category_ids.issubset(set(expected)):
        raise ValueError(f"{path} has annotation categories outside FDI52: {sorted(annotation_category_ids)}")
    invalid_image_ids = {
        int(annotation["image_id"])
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) not in image_id_set
    }
    if invalid_image_ids:
        raise ValueError(f"{path} has annotations pointing at unknown image IDs: {sorted(invalid_image_ids)[:10]}")
    return {
        "path": str(path),
        "image_dir": str(image_dir),
        "images": len(payload["images"]),
        "annotations": len(payload["annotations"]),
        "categories": len(category_ids),
        "used_category_ids": sorted(annotation_category_ids),
    }


def configure_gpu(gpu_id: int) -> None:
    """Pin the process to an explicit physical GPU before importing torch."""

    desired = str(gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != desired:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} conflicts with --gpu-id={gpu_id}. "
            "Refusing to run on an unintended GPU."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = desired


def dataset_names() -> tuple[str, str]:
    # Names are intentionally specific: DatasetCatalog is process-global.
    return "viewnet3_dslr_fdi52_faster_rcnn_train", "viewnet3_dslr_fdi52_faster_rcnn_val"


def normalize_in_memory_image_dimensions(
    records: list[dict[str, Any]],
    *,
    annotation_file: Path,
    dataset_name: str,
) -> dict[str, Any]:
    """Align Detectron2 records with decoded image sizes without editing COCO.

    The canonical FDI52 JSON is shared by every baseline and must remain an
    immutable provenance artifact.  A handful of exported DSLR JPEGs differ
    from their source-image metadata by one pixel.  Detectron2 deliberately
    rejects that discrepancy in ``DatasetMapper``.  Correct only the
    *in-memory* record dimensions, after proving that every bounding box is
    valid in the decoded image coordinate system.
    """

    from detectron2.data.detection_utils import read_image
    from detectron2.structures import BoxMode

    tolerance = 1e-4
    corrections: list[dict[str, Any]] = []
    checked_annotations = 0
    for record in records:
        file_name = str(record.get("file_name", ""))
        if not file_name:
            raise ValueError(f"{dataset_name} has a record without file_name: {record!r}")
        image = read_image(file_name, format="BGR")
        if image.ndim < 2:
            raise ValueError(f"Unable to read a two-dimensional image from {file_name}")
        actual_height, actual_width = (int(image.shape[0]), int(image.shape[1]))
        declared_width, declared_height = (int(record["width"]), int(record["height"]))
        image_id = int(record.get("image_id", -1))

        for annotation_index, annotation in enumerate(record.get("annotations", [])):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"{dataset_name} image_id={image_id} ({file_name}) has an invalid bbox at "
                    f"annotation index {annotation_index}: {bbox!r}"
                )
            try:
                x0, y0, x1, y1 = (
                    float(value)
                    for value in BoxMode.convert(
                        bbox, annotation["bbox_mode"], BoxMode.XYXY_ABS
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{dataset_name} image_id={image_id} ({file_name}) has an unreadable bbox at "
                    f"annotation index {annotation_index}: {bbox!r}"
                ) from error
            if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
                raise ValueError(
                    f"{dataset_name} image_id={image_id} ({file_name}) has a non-finite bbox at "
                    f"annotation index {annotation_index}: {bbox!r}"
                )
            if (
                x1 <= x0
                or y1 <= y0
                or x0 < -tolerance
                or y0 < -tolerance
                or x1 > actual_width + tolerance
                or y1 > actual_height + tolerance
            ):
                raise ValueError(
                    f"{dataset_name} image_id={image_id} ({file_name}) bbox at annotation index "
                    f"{annotation_index} lies outside decoded image bounds {actual_width}x{actual_height}: "
                    f"{bbox!r} (declared {declared_width}x{declared_height})"
                )
            checked_annotations += 1

        if (declared_width, declared_height) != (actual_width, actual_height):
            # This mutates only the catalog record returned to Detectron2.  The
            # canonical COCO JSON on disk is never opened for writing.
            record["width"] = actual_width
            record["height"] = actual_height
            corrections.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "declared_width": declared_width,
                    "declared_height": declared_height,
                    "decoded_width": actual_width,
                    "decoded_height": actual_height,
                }
            )

    return {
        "dataset_name": dataset_name,
        "annotation_file": str(annotation_file.resolve()),
        "policy": "in-memory record dimensions only; source COCO JSON is unchanged",
        "records_checked": len(records),
        "annotations_checked": checked_annotations,
        "corrected_record_count": len(corrections),
        "corrections": corrections,
    }


def get_dimension_audit(train_name: str, val_name: str) -> dict[str, Any]:
    """Retrieve the per-split audit attached by the custom COCO loaders."""

    from detectron2.data import MetadataCatalog

    audits: dict[str, Any] = {}
    for split, name in (("train", train_name), ("validation", val_name)):
        audit = getattr(MetadataCatalog.get(name), "fdi52_dimension_audit", None)
        if not isinstance(audit, dict):
            raise RuntimeError(f"Missing in-memory dimension audit for registered dataset {name}")
        audits[split] = audit
    return {
        "schema_version": 1,
        "policy": "Decoded dimensions normalize only Detectron2 in-memory records; canonical JSON is unchanged.",
        "splits": audits,
    }


def write_dimension_audit(output_dir: Path, audit: dict[str, Any]) -> Path:
    """Persist the bounded data-integrity evidence alongside the run logs."""

    path = output_dir / "dataset_dimension_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return path


def make_dimension_normalizing_coco_loader(
    annotation_file: Path,
    image_dir: Path,
    dataset_name: str,
):
    """Build a cached catalog callable with its own split-local audit state."""

    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import load_coco_json

    records_cache: Optional[list[dict[str, Any]]] = None

    def loader() -> list[dict[str, Any]]:
        nonlocal records_cache
        if records_cache is None:
            records_cache = load_coco_json(str(annotation_file), str(image_dir), dataset_name=dataset_name)
            audit = normalize_in_memory_image_dimensions(
                records_cache,
                annotation_file=annotation_file,
                dataset_name=dataset_name,
            )
            MetadataCatalog.get(dataset_name).set(fdi52_dimension_audit=audit)
        return records_cache

    return loader


def register_fdi52_datasets(
    train_ann: Path,
    train_images: Path,
    val_ann: Path,
    val_images: Path,
) -> tuple[str, str]:
    from detectron2.data import DatasetCatalog, MetadataCatalog

    train_name, val_name = dataset_names()
    registrations = (
        (train_name, train_ann, train_images),
        (val_name, val_ann, val_images),
    )
    for name, ann_path, image_dir in registrations:
        if name not in DatasetCatalog.list():
            # Match register_coco_instances metadata while using a loader that
            # leaves the shared COCO JSON untouched.
            DatasetCatalog.register(name, make_dimension_normalizing_coco_loader(ann_path, image_dir, name))
            MetadataCatalog.get(name).set(
                json_file=str(ann_path), image_root=str(image_dir), evaluator_type="coco"
            )
        # load_coco_json populates the category-ID mapping lazily when the
        # dataset callable is first materialized.
        DatasetCatalog.get(name)
        metadata = MetadataCatalog.get(name)
        mapping = dict(getattr(metadata, "thing_dataset_id_to_contiguous_id", {}))
        expected_mapping = {class_id: class_id for class_id in range(NUM_CLASSES)}
        if mapping != expected_mapping:
            raise RuntimeError(
                f"Detectron2 did not preserve the canonical FDI52 label mapping for {name}: {mapping}"
            )
        if len(getattr(metadata, "thing_classes", [])) != NUM_CLASSES:
            raise RuntimeError(f"{name} has {len(metadata.thing_classes)} classes, expected {NUM_CLASSES}")
    return train_name, val_name


def build_config(
    args: argparse.Namespace,
    train_name: str,
    val_name: str,
    iterations_per_epoch: int,
) -> tuple[Any, str]:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIG))
    pretrained_weights = args.weights or model_zoo.get_checkpoint_url(MODEL_ZOO_CONFIG)
    cfg.MODEL.WEIGHTS = pretrained_weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    # FDI labels encode left/right semantics. A naive horizontal flip leaves
    # category IDs unchanged and would therefore create anatomically invalid
    # examples, so it is intentionally disabled.
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MOMENTUM = args.momentum
    cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    cfg.SOLVER.MAX_ITER = args.epochs * iterations_per_epoch
    cfg.SOLVER.STEPS = tuple(sorted({epoch * iterations_per_epoch for epoch in args.lr_drop_epochs}))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000.0
    cfg.SOLVER.WARMUP_ITERS = min(args.warmup_iters, max(0, cfg.SOLVER.MAX_ITER - 1))
    cfg.SOLVER.WARMUP_METHOD = "linear"
    # DefaultTrainer's periodic snapshots are removed in the custom trainer.
    # Keep this enormous to guard against accidental base-class behavior.
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER + 1
    cfg.TEST.EVAL_PERIOD = iterations_per_epoch
    cfg.TEST.PRECISE_BN.ENABLED = False
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.SEED = args.seed
    cfg.CUDNN_BENCHMARK = False
    cfg.freeze()
    return cfg, pretrained_weights


def make_trainer_class():
    """Construct a trainer that never emits periodic model snapshots.

    The single retained *during-training* best checkpoint must remain a real
    Detectron2 resume checkpoint.  In particular, it needs the trainer,
    optimizer and scheduler states in addition to model weights; otherwise an
    interruption would silently restart the iteration counter and optimizer.
    ``cleanup_after_success`` converts that one artifact to a model-only
    checkpoint once evaluation has completed, when resume state is no longer
    needed.
    """

    from detectron2.engine import DefaultTrainer, hooks
    from detectron2.evaluation import COCOEvaluator

    class ResumableBestCheckpointer(hooks.BestCheckpointer):
        """Persist best-score state so an interrupted run cannot regress it.

        Detectron2's stock ``BestCheckpointer`` is intentionally stateless.
        That is fine for a one-shot job, but on resume it would regard the
        first post-resume validation result as a new best and could overwrite
        the actual selected checkpoint.  Saving the two scalar fields alongside
        the ordinary trainer state keeps the best-weight criterion continuous.
        """

        def state_dict(self) -> dict[str, Any]:
            return {"best_metric": self.best_metric, "best_iter": self.best_iter}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.best_metric = state_dict.get("best_metric")
            self.best_iter = state_dict.get("best_iter")

        def _best_checking(self) -> None:
            metric_tuple = self.trainer.storage.latest().get(self._val_metric)
            if metric_tuple is None:
                self._logger.warning(
                    "Given val metric %s does not seem to be computed/stored. "
                    "Will not be checkpointing based on it.",
                    self._val_metric,
                )
                return
            latest_metric, metric_iter = metric_tuple
            is_better = self.best_metric is None or self._compare(latest_metric, self.best_metric)
            if not is_better:
                self._logger.info(
                    "Not saving as latest eval score for %s is %.5f, not better than "
                    "best score %.5f @ iteration %s.",
                    self._val_metric,
                    latest_metric,
                    self.best_metric,
                    self.best_iter,
                )
                return
            if not self._update_best(latest_metric, metric_iter):
                return
            # Update *before* serializing the trainer so the checkpoint contains
            # the selected score itself, rather than the preceding best score.
            self._checkpointer.save(self._file_prefix, iteration=metric_iter)
            self._logger.info(
                "Saved best model as eval score for %s is %.5f @ iteration %s.",
                self._val_metric,
                self.best_metric,
                self.best_iter,
            )

    class BestOnlyFasterRCNNTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg: Any, dataset_name: str, output_folder: Optional[str] = None):
            output_folder = output_folder or os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(
                dataset_name,
                tasks=("bbox",),
                distributed=False,
                output_dir=output_folder,
                max_dets_per_image=100,
                use_fast_impl=False,
            )

        def build_hooks(self):
            all_hooks = super().build_hooks()
            # Removing the default PeriodicCheckpointer prevents 160 expensive
            # 300+ MB optimizer snapshots.  Reuse DefaultTrainer's ordinary
            # checkpointer for the one selected best checkpoint so a later
            # --resume restores model, optimizer, scheduler and trainer state.
            all_hooks = [hook for hook in all_hooks if not isinstance(hook, hooks.PeriodicCheckpointer)]
            self.best_model_checkpointer = self.checkpointer
            best_hook = ResumableBestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.best_model_checkpointer,
                val_metric="bbox/AP",
                mode="max",
                file_prefix=BEST_CHECKPOINT_NAME,
            )
            for index, hook in enumerate(all_hooks):
                if isinstance(hook, hooks.EvalHook):
                    # EvalHook must write bbox/AP into EventStorage first.
                    all_hooks.insert(index + 1, best_hook)
                    break
            else:
                raise RuntimeError("Unable to install the best-checkpoint hook after the evaluation hook")
            return all_hooks

    return BestOnlyFasterRCNNTrainer


def read_best_training_metric(metrics_path: Path, iterations_per_epoch: int) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    if metrics_path.is_file():
        for line in metrics_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = row.get("bbox/AP")
            iteration = row.get("iteration")
            if not isinstance(score, (float, int)) or not isinstance(iteration, int):
                continue
            if best is None or float(score) > float(best["AP_percent"]):
                best = {"AP_percent": float(score), "iteration": int(iteration)}
    if best is None:
        raise RuntimeError(f"No validation bbox/AP entries found in {metrics_path}")
    best["epoch"] = int(best["iteration"] // iterations_per_epoch + 1)
    return best


def run_official_coco_bbox_evaluation(annotation_file: Path, prediction_file: Path) -> tuple[dict[str, float], int, int]:
    """Return standard COCO bbox metrics on a 0--1 scale, including AR100."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = json.loads(prediction_file.read_text())
    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError(f"No bbox predictions found in {prediction_file}")
    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.catIds = sorted(coco_gt.getCatIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
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
    metrics = {name: float(value) for name, value in zip(names, evaluator.stats.tolist())}
    return metrics, len(predictions), len(coco_gt.getImgIds())


def model_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_final_metrics(
    args: argparse.Namespace,
    cfg: Any,
    pretrained_weights: str,
    best_training: dict[str, Any],
    final_results: dict[str, Any],
    parameter_count: int,
    dimension_audit: dict[str, Any],
    dimension_audit_path: Path,
) -> Path:
    prediction_file = args.output_dir / "coco_eval/coco_instances_results.json"
    if not prediction_file.is_file():
        raise FileNotFoundError(f"Detectron2 evaluation did not write predictions: {prediction_file}")
    stats, prediction_count, image_count = run_official_coco_bbox_evaluation(args.val_ann, prediction_file)
    raw_bbox = final_results.get("bbox", {})
    raw_metrics = {name: float(value) for name, value in raw_bbox.items() if isinstance(value, (float, int))}
    primary = {name: stats[name] for name in ("AP", "AP50", "AP75", "AR100")}
    payload = {
        "framework": "Detectron2",
        "method": MODEL_NAME,
        "architecture": "Faster R-CNN with ResNet-50-FPN backbone",
        "weights": str((args.output_dir / f"{BEST_CHECKPOINT_NAME}.pth").resolve()),
        "best_weight": str((args.output_dir / f"{BEST_CHECKPOINT_NAME}.pth").resolve()),
        "pretrained_weights": pretrained_weights,
        "annotation_file": str(args.val_ann),
        "image_dir": str(args.val_images),
        "image_count": image_count,
        "prediction_count": prediction_count,
        "num_parameters": parameter_count,
        "metrics": primary,
        "metrics_all": stats,
        "detectron2_metrics_percent": raw_metrics,
        "training": {
            "epochs_target": args.epochs,
            "batch_size": args.batch_size,
            "iterations_per_epoch": cfg.TEST.EVAL_PERIOD,
            "max_iterations": cfg.SOLVER.MAX_ITER,
            "lr_drop_epochs": sorted(int(step // cfg.TEST.EVAL_PERIOD) for step in cfg.SOLVER.STEPS),
            "best_epoch_by_validation_AP": best_training["epoch"],
            "best_iteration_by_validation_AP": best_training["iteration"],
            "best_validation_AP_percent": best_training["AP_percent"],
            "optimizer": "SGD",
            "base_lr": args.base_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "horizontal_flip": False,
        },
        "data_integrity": {
            "dimension_normalization": dimension_audit["policy"],
            "dimension_audit_file": str(dimension_audit_path.resolve()),
            "train_corrected_record_count": dimension_audit["splits"]["train"]["corrected_record_count"],
            "validation_corrected_record_count": dimension_audit["splits"]["validation"][
                "corrected_record_count"
            ],
        },
        "inference": {
            "min_size": args.min_size,
            "max_size": args.max_size,
            "score_threshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "requested_physical_gpu": args.gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "prediction_file": str(prediction_file.resolve()),
    }
    metrics_path = args.output_dir / "coco_eval/coco_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return metrics_path


def is_descendant(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def cleanup_after_success(output_dir: Path, best_weight: Path) -> None:
    """Leave only the selected model-only checkpoint and light JSON/log evidence."""

    output_dir = output_dir.resolve()
    best_weight = best_weight.resolve()
    if not is_descendant(best_weight, output_dir):
        raise RuntimeError(f"Refusing cleanup: best checkpoint lies outside output directory: {best_weight}")
    if not best_weight.is_file():
        raise FileNotFoundError(f"Best checkpoint is absent: {best_weight}")
    # COCOEvaluator writes a temporary prediction tensor at every evaluation.
    # The final COCO JSON and metrics are preserved separately in coco_eval.
    intermediate_inference = output_dir / "inference"
    if intermediate_inference.is_dir():
        shutil.rmtree(intermediate_inference)
    for candidate in output_dir.rglob("*.pth"):
        if candidate.resolve() != best_weight:
            candidate.unlink()
    for candidate in output_dir.rglob("*.pkl"):
        candidate.unlink()
    # The active best checkpoint deliberately included resume state.  Once the
    # selected model has been independently evaluated, retain only the model
    # tensors to honor the disk-budget policy without weakening interruption
    # recovery during training.
    import torch

    checkpoint = torch.load(best_weight, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError(f"Cannot slim selected checkpoint with no model state: {best_weight}")
    temporary_path = best_weight.with_suffix(".tmp")
    try:
        torch.save({"model": checkpoint["model"]}, temporary_path)
        temporary_path.replace(best_weight)
    finally:
        temporary_path.unlink(missing_ok=True)
    checkpoint_marker = output_dir / "last_checkpoint"
    if checkpoint_marker.is_file():
        checkpoint_marker.unlink()


def output_is_nonempty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def reset_output_directory(output_dir: Path) -> None:
    """Clear a prior run only after the caller explicitly passed --overwrite."""

    if output_dir.exists():
        shutil.rmtree(output_dir)


def print_dry_run_report(
    args: argparse.Namespace,
    train_info: dict[str, Any],
    val_info: dict[str, Any],
    iterations_per_epoch: int,
    detectron2_info: dict[str, Any],
) -> None:
    report = {
        "status": "DRY_RUN_OK",
        "model": MODEL_NAME,
        "dataset": "View-Net3 DSLR canonical FDI52",
        "train": train_info,
        "validation": val_info,
        "classes": NUM_CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "iterations_per_epoch": iterations_per_epoch,
        "max_iterations": args.epochs * iterations_per_epoch,
        "eval_period_iterations": iterations_per_epoch,
        "lr_drop_iterations": [epoch * iterations_per_epoch for epoch in sorted(args.lr_drop_epochs)],
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "checkpoint_policy": f"best-only ({BEST_CHECKPOINT_NAME}.pth after success)",
        "detectron2": detectron2_info,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def validate_detectron2_setup(
    args: argparse.Namespace,
    train_info: dict[str, Any],
    val_info: dict[str, Any],
    iterations_per_epoch: int,
) -> dict[str, Any]:
    """Exercise Detectron2 registration and configuration without building/training a model."""

    configure_gpu(args.gpu_id)
    import detectron2
    import torch
    import torchvision
    from detectron2.data import DatasetCatalog

    train_name, val_name = register_fdi52_datasets(
        args.train_ann, args.train_images, args.val_ann, args.val_images
    )
    observed_train = len(DatasetCatalog.get(train_name))
    observed_val = len(DatasetCatalog.get(val_name))
    if observed_train != train_info["images"] or observed_val != val_info["images"]:
        raise RuntimeError("Detectron2 dataset registration changed the validated split cardinality")
    dimension_audit = get_dimension_audit(train_name, val_name)
    cfg, pretrained_weights = build_config(args, train_name, val_name, iterations_per_epoch)
    return {
        "detectron2_version": getattr(detectron2, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "registered_train_records": observed_train,
        "registered_validation_records": observed_val,
        "dataset_dimension_audit": dimension_audit,
        "roi_head_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "mask_on": cfg.MODEL.MASK_ON,
        "evaluation_period": cfg.TEST.EVAL_PERIOD,
        "max_iterations": cfg.SOLVER.MAX_ITER,
        "pretrained_weights": pretrained_weights,
    }


def main() -> int:
    global NUM_CLASSES
    args = parse_args()
    NUM_CLASSES = args.num_classes
    resolve_and_validate_args(args)
    train_info = verify_coco_fdi52(args.train_ann, args.train_images)
    val_info = verify_coco_fdi52(args.val_ann, args.val_images)
    iterations_per_epoch = math.ceil(train_info["images"] / args.batch_size)
    if args.dry_run:
        detectron2_info = validate_detectron2_setup(args, train_info, val_info, iterations_per_epoch)
        print_dry_run_report(args, train_info, val_info, iterations_per_epoch, detectron2_info)
        return 0

    completed_marker = args.output_dir / "COMPLETED"
    best_weight = args.output_dir / f"{BEST_CHECKPOINT_NAME}.pth"
    if completed_marker.is_file() and best_weight.is_file() and not args.force:
        print(f"[skip] Completed Faster R-CNN run already exists: {args.output_dir}")
        return 0
    if output_is_nonempty(args.output_dir) and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"Output directory is non-empty: {args.output_dir}. Use --resume after an interruption, "
            "or --overwrite to deliberately start a new run."
        )
    if args.resume and not (args.output_dir / "last_checkpoint").is_file():
        raise RuntimeError(
            f"--resume was requested but no safe model checkpoint marker exists in {args.output_dir}. "
            "Use --overwrite to start a new run."
        )
    if args.overwrite and output_is_nonempty(args.output_dir):
        reset_output_directory(args.output_dir)

    configure_gpu(args.gpu_id)
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog
    from detectron2.engine import default_setup
    from detectron2.evaluation import COCOEvaluator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after selecting the requested GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible CUDA device after GPU pinning, found "
            f"{torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )

    train_name, val_name = register_fdi52_datasets(
        args.train_ann, args.train_images, args.val_ann, args.val_images
    )
    # Loading the DatasetCatalog verifies that Detectron2 itself observes the
    # same number of samples used to derive the nominal epoch length.
    if len(DatasetCatalog.get(train_name)) != train_info["images"]:
        raise RuntimeError("Detectron2 train-dataset length disagrees with the validated COCO JSON")
    if len(DatasetCatalog.get(val_name)) != val_info["images"]:
        raise RuntimeError("Detectron2 validation-dataset length disagrees with the validated COCO JSON")
    dimension_audit = get_dimension_audit(train_name, val_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimension_audit_path = write_dimension_audit(args.output_dir, dimension_audit)
    (args.output_dir / "RUNNING").touch()
    for marker in (args.output_dir / "FAILED", args.output_dir / "COMPLETED"):
        marker.unlink(missing_ok=True)
    cfg, pretrained_weights = build_config(args, train_name, val_name, iterations_per_epoch)
    default_setup(cfg, args)
    Trainer = make_trainer_class()
    trainer = Trainer(cfg)

    try:
        trainer.resume_or_load(resume=args.resume)
        trainer.train()
        if not best_weight.is_file():
            raise RuntimeError("Training completed without model_best.pth")
        best_training = read_best_training_metric(args.output_dir / "metrics.json", iterations_per_epoch)

        # Evaluate the selected model, rather than the final SGD iteration.
        DetectionCheckpointer(trainer.model).load(str(best_weight))
        final_evaluator = COCOEvaluator(
            val_name,
            tasks=("bbox",),
            distributed=False,
            output_dir=str(args.output_dir / "coco_eval"),
            max_dets_per_image=100,
            use_fast_impl=False,
        )
        final_results = Trainer.test(cfg, trainer.model, evaluators=[final_evaluator])
        metrics_path = write_final_metrics(
            args,
            cfg,
            pretrained_weights,
            best_training,
            final_results,
            model_parameter_count(trainer.model),
            dimension_audit,
            dimension_audit_path,
        )
        cleanup_after_success(args.output_dir, best_weight)
        (args.output_dir / "RUNNING").unlink(missing_ok=True)
        (args.output_dir / "COMPLETED").touch()
        print(f"Completed {MODEL_NAME}; metrics: {metrics_path}")
        return 0
    except BaseException:
        (args.output_dir / "RUNNING").unlink(missing_ok=True)
        (args.output_dir / "FAILED").touch()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
