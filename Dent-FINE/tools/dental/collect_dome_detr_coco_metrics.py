#!/usr/bin/env python3
"""Normalize an official Dome-DETR COCO evaluation into the baseline schema.

Dome-DETR prints the standard pycocotools 12-statistic bbox table but does not
emit a portable JSON summary.  This helper deliberately parses the evaluation
log (or, as a fallback, the official per-epoch JSON log) and writes the same
0--1 metric fields used by the DSLR YOLO/DEIM comparison table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STAT_NAMES = (
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Dome --test-only stdout/stderr log")
    parser.add_argument(
        "--train-log",
        type=Path,
        default=None,
        help="Optional official Dome JSON-lines log.txt fallback.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", default="Dome-DETR-S")
    parser.add_argument("--epoch", type=int, default=None)
    return parser.parse_args()


def _metric_from_line(line: str) -> tuple[str, float] | None:
    """Map one standard COCOeval summary line to its stable metric key."""

    kind = "AP" if "Average Precision" in line else "AR" if "Average Recall" in line else None
    if kind is None or "area=" not in line or "maxDets=" not in line:
        return None
    if not re.search(r"area=\s*all\b", line):
        return None

    max_dets_match = re.search(r"maxDets=\s*(\d+)", line)
    value_match = re.search(r"=\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$", line)
    if max_dets_match is None or value_match is None:
        return None
    max_dets = int(max_dets_match.group(1))
    value = float(value_match.group(1))

    iou_50_95 = re.search(r"IoU=\s*0\.50\s*:\s*0\.95", line) is not None
    iou_50 = re.search(r"IoU=\s*0\.50\b(?!\s*:)", line) is not None
    iou_75 = re.search(r"IoU=\s*0\.75\b(?!\s*:)", line) is not None
    if kind == "AP" and max_dets == 100:
        if iou_50_95:
            return "AP", value
        if iou_50:
            return "AP50", value
        if iou_75:
            return "AP75", value
    if kind == "AR" and iou_50_95 and max_dets in (1, 10, 100):
        return f"AR{max_dets}", value
    return None


def parse_coco_stdout(log_path: Path) -> dict[str, float]:
    if not log_path.is_file():
        return {}
    values: dict[str, float] = {}
    # A run may contain more than one evaluation.  Retain the final occurrence
    # of every metric, which is the --test-only evaluation for the selected
    # checkpoint.
    for line in log_path.read_text(errors="replace").splitlines():
        parsed = _metric_from_line(line)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    required = {"AP", "AP50", "AP75", "AR1", "AR10", "AR100"}
    return values if required.issubset(values) else {}


def parse_official_json_log(log_path: Path | None) -> tuple[dict[str, float], list[float] | None]:
    if log_path is None or not log_path.is_file():
        return {}, None
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        try:
            payload: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats = payload.get("test_coco_eval_bbox")
        if not isinstance(stats, list) or len(stats) < len(STAT_NAMES):
            continue
        try:
            floats = [float(value) for value in stats[: len(STAT_NAMES)]]
        except (TypeError, ValueError):
            continue
        return dict(zip(STAT_NAMES, floats)), floats
    return {}, None


def main() -> None:
    args = parse_args()
    args.log = args.log.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.train_log = args.train_log.expanduser().resolve() if args.train_log else None
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Selected Dome checkpoint is missing: {args.checkpoint}")

    metrics = parse_coco_stdout(args.log)
    raw_stats: list[float] | None = None
    source = str(args.log)
    if not metrics:
        metrics, raw_stats = parse_official_json_log(args.train_log)
        source = str(args.train_log) if metrics and args.train_log else source
    required = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(
            "Could not recover a complete COCO bbox summary for Dome-DETR; "
            f"missing {missing}. Evaluation log: {args.log}"
        )

    output = {
        "method": args.method,
        "metric_protocol": "COCO bbox, canonical View-Net3 DSLR FDI52 validation split",
        "metric_scale": "0-1",
        "checkpoint": str(args.checkpoint),
        # Match the common cross-framework schema consumed by
        # summarize_viewnet3_dslr_baselines.py.
        "weights": str(args.checkpoint),
        "best_weight": str(args.checkpoint),
        "source_log": source,
        "metrics": {key: metrics[key] for key in STAT_NAMES if key in metrics},
        **{key: metrics[key] for key in STAT_NAMES if key in metrics},
    }
    if raw_stats is not None:
        output["raw_coco_stats"] = raw_stats
    if args.epoch is not None:
        output["selected_epoch"] = args.epoch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
