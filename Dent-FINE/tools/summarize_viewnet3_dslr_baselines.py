#!/usr/bin/env python3
"""Build a live, evaluator-aligned baseline comparison table for View-Net3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# The first exploratory YOLO runs used Ultralytics' default horizontal flip.
# That is anatomically invalid for fixed FDI left/right labels, so they are
# retained on disk only as audit artifacts and intentionally excluded here.
# The no-flip reruns below, DEIM, and the three external baselines form the
# paper-facing, protocol-aligned comparison.
EXPERIMENTS = (
    ("YOLOv8-s", "yolov8s_fdi52_no_flip", "yolo", "Ultralytics", "AdamW", "weights/best.pt"),
    ("YOLO11-s", "yolo11s_fdi52_no_flip", "yolo", "Ultralytics", "AdamW", "weights/best.pt"),
    ("CAF-YOLO-s", "caf_yolo_s_fdi52", "yolo", "CAF-YOLO (Ultralytics 8.0.228)", "AdamW", "weights/best.pt"),
    ("Faster R-CNN R50-FPN", "faster_rcnn_r50_fpn_fdi52", "json", "Detectron2", "SGD", "model_best.pth"),
    ("Dome-DETR-S", "dome_detr_s_fdi52", "dome", "Dome-DETR", "AdamW", "best.pth"),
    ("DEIM-s", "deim_s_fdi52", "deim", "Dent-FINE", "AdamW", "best.pth"),
)


def clean(value: Any) -> float | None:
    if value in (None, ""):
        return None
    value = float(value)
    return value if value >= 0 else None


def parse_deim_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    by_epoch: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch, values = row.get("epoch"), row.get("test_coco_eval_bbox")
        if isinstance(epoch, int) and isinstance(values, list):
            by_epoch[epoch] = row
    if not by_epoch:
        return None
    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    best = max(rows, key=lambda row: (row.get("test_coco_eval_bbox") or [-1])[0])
    last = rows[-1]
    values = best["test_coco_eval_bbox"]
    return {
        "epochs_logged": len(rows),
        "last_epoch": last["epoch"],
        "best_epoch": best["epoch"],
        "AP": clean(values[0] if len(values) > 0 else None),
        "AP50": clean(values[1] if len(values) > 1 else None),
        "AP75": clean(values[2] if len(values) > 2 else None),
        "AR100": clean(values[8] if len(values) > 8 else None),
        "params": best.get("n_parameters"),
    }


def parse_yolo_results(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    metric_column = next((name for name in rows[0] if name.strip() == "metrics/mAP50-95(B)"), None)
    if metric_column is None:
        return {"epochs_logged": len(rows), "last_epoch": len(rows) - 1}
    valid = [row for row in rows if row.get(metric_column) not in (None, "")]
    if not valid:
        return {"epochs_logged": len(rows), "last_epoch": len(rows) - 1}
    best = max(valid, key=lambda row: float(row[metric_column]))
    return {
        "epochs_logged": len(rows),
        "last_epoch": int(float(rows[-1].get("epoch", len(rows) - 1))),
        "best_epoch": int(float(best.get("epoch", 0))),
        "native_AP": clean(best.get(metric_column)),
        "native_AP50": clean(best.get("metrics/mAP50(B)")),
    }


def read_coco_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    return {
        "AP": clean(metrics.get("AP")),
        "AP50": clean(metrics.get("AP50")),
        "AP75": clean(metrics.get("AP75")),
        "AR100": clean(metrics.get("AR100")),
        "params": data.get("num_parameters"),
        "predictions": data.get("prediction_count"),
        "ultralytics_version": data.get("ultralytics_version"),
    }


def fmt(value: Any, scale: float = 100.0) -> str:
    return "-" if value in (None, "") else f"{float(value) * scale:.3f}"


def fmt_delta(value: Any) -> str:
    return "-" if value in (None, "") else f"{float(value) * 100.0:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison"),
    )
    parser.add_argument("--expected-epochs", type=int, default=160)
    args = parser.parse_args()
    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be positive")
    expected_last_epoch = args.expected_epochs - 1
    base = args.base_dir
    base.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for order, (method, name, framework, framework_name, optimizer, weight_relative) in enumerate(EXPERIMENTS, 1):
        run_dir = base / "runs" / name
        record: dict[str, Any] = {
            "order": order,
            "method": method,
            "framework": framework_name,
            "dataset": "View-Net3 DSLR FDI52",
            "input": "640×640",
            "classes": "52 canonical FDI labels",
            "epochs_target": args.expected_epochs,
            "seed": 0,
            "run_dir": str(run_dir),
            "status": "queued",
        }
        record["optimizer"] = optimizer
        if framework == "yolo":
            weight_path = run_dir / weight_relative
            record["best_weight"] = str(weight_path) if weight_path.is_file() else ""
            parsed = parse_yolo_results(run_dir / "results.csv")
            coco_metrics = read_coco_metrics(run_dir / "coco_eval" / "coco_metrics.json")
            if parsed:
                record.update(parsed)
            if coco_metrics:
                record.update(coco_metrics)
        elif framework == "deim":
            weight_path = run_dir / weight_relative
            record["best_weight"] = str(weight_path) if weight_path.is_file() else ""
            parsed = parse_deim_log(run_dir / "log.txt")
            if parsed:
                record.update(parsed)
        else:
            weight_path = run_dir / weight_relative
            record["best_weight"] = str(weight_path) if weight_path.is_file() else ""
            parsed = None
            coco_metrics = read_coco_metrics(run_dir / "coco_eval" / "coco_metrics.json")
            if coco_metrics:
                record.update(coco_metrics)
                training = json.loads((run_dir / "coco_eval" / "coco_metrics.json").read_text()).get("training", {})
                if isinstance(training, dict):
                    record["best_epoch"] = training.get("best_epoch_by_validation_AP", training.get("best_epoch"))
                    record["epochs_logged"] = training.get("epochs_logged", training.get("epochs_target"))
                    record["last_epoch"] = training.get("last_epoch", args.expected_epochs - 1)
            if framework == "dome":
                parsed = parse_deim_log(run_dir / "log.txt")
                if parsed:
                    # The Dome runner also writes a final standard COCO JSON;
                    # retain its log-derived training epoch and parameter count
                    # only as traceability.  Do not overwrite the selected
                    # checkpoint's final COCO metrics with a per-epoch entry.
                    for key in ("best_epoch", "epochs_logged", "last_epoch", "params"):
                        if record.get(key) in (None, "") and parsed.get(key) is not None:
                            record[key] = parsed[key]

        if (run_dir / "FAILED").is_file():
            record["status"] = "failed"
        elif (run_dir / "COMPLETED").is_file():
            record["status"] = "done"
        elif record.get("last_epoch", -1) >= expected_last_epoch:
            record["status"] = "evaluating" if not record.get("AP") else "done"
        elif (run_dir / "RUNNING").is_file():
            record["status"] = "running"
        elif parsed:
            record["status"] = "partial"
        records.append(record)

    complete = sorted(
        (record for record in records if record["status"] == "done" and record.get("AP") is not None),
        key=lambda record: float(record["AP"]),
        reverse=True,
    )
    best_ap = complete[0]["AP"] if complete else None
    for rank, record in enumerate(complete, 1):
        record["rank"] = rank
        record["delta_vs_best"] = float(record["AP"]) - float(best_ap)

    fields = [
        "order", "rank", "method", "framework", "dataset", "input", "classes", "epochs_target", "seed", "optimizer",
        "status", "epochs_logged", "last_epoch", "best_epoch", "AP", "AP50", "AP75", "AR100", "native_AP", "native_AP50",
        "delta_vs_best", "params", "predictions", "ultralytics_version", "run_dir", "best_weight",
    ]
    csv_path = base / "baseline_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: "" if record.get(field) is None else record.get(field, "") for field in fields})

    lines = [
        "# View-Net3 DSLR FDI52 Baseline Comparison",
        "",
        "- Split and label space: identical 750-image train / 190-image validation cohort with canonical zero-based FDI52 labels.",
        "- Training budget: 160 epochs, 640×640 input, seed 0, and one GPU. Batch size is 8 except Faster R-CNN (4); each framework retains only its final selected best weight.",
        "- Geometric horizontal/vertical flips are disabled because FDI labels encode left/right tooth position. The initial exploratory YOLO artifacts with default horizontal flip are excluded from this paper-facing table.",
        "- Primary metrics are standard COCO bbox metrics on the same validation JSON. Native Ultralytics mAP is retained separately for traceability.",
        "",
        "| Rank | Method | Framework | Status | Epochs | Best epoch | COCO AP | AP50 | AP75 | AR100 | Δ vs best | Params | Best weight |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        params = "-" if record.get("params") in (None, "") else f"{int(record['params']) / 1e6:.2f}M"
        lines.append(
            f"| {record.get('rank', '-')} | {record['method']} | {record['framework']} | {record['status']} | "
            f"{record.get('epochs_logged', '-')} | {record.get('best_epoch', '-')} | {fmt(record.get('AP'))} | "
            f"{fmt(record.get('AP50'))} | {fmt(record.get('AP75'))} | {fmt(record.get('AR100'))} | "
            f"{fmt_delta(record.get('delta_vs_best'))} | {params} | `{record['best_weight'] or '-'}` |"
        )
    lines += [
        "",
        "## Reproducibility",
        "",
        "- YOLOv8-s / YOLO11-s: Ultralytics detection models initialized from their official pretrained small checkpoints and optimized with AdamW, with FDI-safe flips disabled.",
        "- CAF-YOLO-s: authors' CAFMAttention YOLOv8 fork, small scale, initialized by compatible weights from the official YOLOv8-s checkpoint and trained with the authors' non-AMP setting.",
        "- Faster R-CNN R50-FPN: Detectron2 implementation initialized from the official COCO R50-FPN 3x checkpoint and optimized with SGD.",
        "- Dome-DETR-S: authors' official HGNetv2-B0 small configuration, adapted only for the canonical FDI52 data paths, class count, COCO evaluator, and 640-pixel protocol.",
        "- DEIM-s: HGNetv2-B0 / 3-layer DEIM configuration initialized from the project-local pretrained HGNetv2 weights; all Dent-FINE additions and architecture-prior loss are disabled, with standard AdamW.",
        "- Evaluation artifacts: `runs/<model>/coco_eval/coco_metrics.json`; DEIM also records evaluator results in `runs/deim_s_fdi52/log.txt`.",
        "",
        "## Files",
        "",
        f"- CSV: `{csv_path}`",
        f"- Markdown: `{base / 'baseline_summary.md'}`",
        "",
    ]
    markdown_path = base / "baseline_summary.md"
    markdown_path.write_text("\n".join(lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
