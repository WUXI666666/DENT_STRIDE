#!/usr/bin/env python3
"""Summarize the original combo-module recipes on the View-Net3 DSLR cohort.

The source cross-dataset table contains the same twelve recipes for DENTEX and
COCO.  Two of those recipes were already completed in the earlier View-Net3
Best4 queue, so this summary deliberately reads those two results in place
instead of duplicating their training or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    ("viewnet3_spatial_shape_dualopt", "SpatialShape + DualOpt", "new"),
    ("viewnet3_prototype_dualopt", "Prototype + DualOpt", "new"),
    ("viewnet3_cmfa_dualopt", "CMFA + DualOpt", "reused"),
    ("viewnet3_mpf_dualopt", "MPF + DualOpt", "reused"),
    ("viewnet3_cmfa_hooa_dualopt", "CMFA + HOOA + DualOpt", "new"),
    ("viewnet3_cmfa_hooa_prototype", "CMFA + HOOA + Prototype", "new"),
    ("viewnet3_cmfa_hooa_spatial_shape", "CMFA + HOOA + SpatialShape", "new"),
    ("viewnet3_cmfa_hooa_spatial_shape_dualopt", "CMFA + HOOA + SpatialShape + DualOpt", "new"),
    ("viewnet3_cmfa_hooa_cedv2", "CMFA + HOOA + CEDv2", "new"),
    ("viewnet3_cmfa_hooa_cedv2_dualopt", "CMFA + HOOA + CEDv2 + DualOpt", "new"),
    ("viewnet3_cmfa_hooa_cedv2_spatial_shape", "CMFA + HOOA + CEDv2 + SpatialShape", "new"),
    ("viewnet3_cmfa_hooa_cedv2_spatial_shape_dualopt", "CMFA + HOOA + CEDv2 + SpatialShape + DualOpt", "new"),
]


def clean_metric(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if value < 0 else value


def parse_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    # A resumed experiment may append an epoch that already exists in log.txt.
    # Keep its most recent record so the total table remains stable.
    by_epoch: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = row.get("test_coco_eval_bbox")
        epoch = row.get("epoch")
        if isinstance(metrics, list) and isinstance(epoch, int):
            by_epoch[epoch] = row
    if not by_epoch:
        return None

    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    best = max(rows, key=lambda row: (row.get("test_coco_eval_bbox") or [-1])[0])
    last = rows[-1]
    best_metrics = best["test_coco_eval_bbox"]
    last_metrics = last["test_coco_eval_bbox"]
    parsed: dict[str, Any] = {
        "epochs_logged": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": last["epoch"],
        "best_epoch": best["epoch"],
        "params": best.get("n_parameters"),
        "train_lr_best": best.get("train_lr"),
        "train_loss_best": best.get("train_loss"),
    }
    for index, metric in enumerate(("ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100")):
        parsed[metric] = clean_metric(best_metrics[index] if len(best_metrics) > index else None)
    for index, metric in enumerate(("final_ap", "final_ap50", "final_ap75")):
        parsed[metric] = clean_metric(last_metrics[index] if len(last_metrics) > index else None)
    return parsed


def status_for(exp_dir: Path, parsed: dict[str, Any] | None, expected_last_epoch: int) -> str:
    if (exp_dir / "FAILED").is_file():
        return "failed"
    if parsed is None:
        return "queued"
    if parsed.get("last_epoch", -1) >= expected_last_epoch:
        return "done"
    if (exp_dir / "RUNNING").is_file():
        return "running"
    return "partial"


def fmt(value: Any, scale: float = 100.0) -> str:
    return "-" if value in (None, "") else f"{float(value) * scale:.3f}"


def fmt_delta(value: Any) -> str:
    return "-" if value in (None, "") else f"{float(value) * 100.0:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_combo_modules"),
    )
    parser.add_argument(
        "--inherited-dir",
        type=Path,
        default=Path("/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_best4_dualopt_combinations"),
    )
    parser.add_argument("--expected-epochs", type=int, default=160)
    args = parser.parse_args()
    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be positive")

    base = args.base_dir
    base.mkdir(parents=True, exist_ok=True)
    expected_last_epoch = args.expected_epochs - 1
    records: list[dict[str, Any]] = []
    for order, (experiment, method, source) in enumerate(EXPERIMENTS, 1):
        exp_dir = (args.inherited_dir if source == "reused" else base) / experiment
        parsed = parse_log(exp_dir / "log.txt")
        record: dict[str, Any] = {
            "order": order,
            "dataset": "View-Net3 DSLR FDI52",
            "experiment": experiment,
            "method": method,
            "source": "reused from Best4 queue" if source == "reused" else "combo-recipe queue",
            "status": status_for(exp_dir, parsed, expected_last_epoch),
            "output_dir": str(exp_dir),
            "log_path": str(exp_dir / "log.txt"),
            "best_weight": str(exp_dir / "best.pth") if (exp_dir / "best.pth").is_file() else "",
        }
        if parsed:
            record.update(parsed)
        records.append(record)

    completed = sorted(
        (record for record in records if record["status"] == "done" and record.get("ap") is not None),
        key=lambda record: float(record["ap"]),
        reverse=True,
    )
    best_ap = completed[0]["ap"] if completed else None
    for rank, record in enumerate(completed, 1):
        record["rank"] = rank
        record["delta_vs_best_completed"] = float(record["ap"]) - float(best_ap)

    fields = [
        "order", "rank", "dataset", "experiment", "method", "source", "status",
        "epochs_logged", "first_epoch", "last_epoch", "best_epoch", "ap", "ap50", "ap75",
        "aps", "apm", "apl", "ar1", "ar10", "ar100", "final_ap", "final_ap50", "final_ap75",
        "delta_vs_best_completed", "params", "train_lr_best", "train_loss_best",
        "output_dir", "log_path", "best_weight",
    ]
    csv_path = base / "ablation_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: "" if record.get(field) is None else record.get(field, "") for field in fields})

    lines = [
        "# View-Net3 DSLR FDI52: Original Combination-Module Recipes",
        "",
        "- Dataset: 750 training images / 190 validation images; canonical 52-class FDI label space.",
        "- Recipes: the 12 unique module recipes in `ablation_combo_modules_cross_dataset`; its DENTEX and COCO sections repeat this same recipe set.",
        "- CMFA+DualOpt and MPF+DualOpt are reused from the completed Best4 DSLR queue; the other 10 recipes run in this queue.",
        "- The SpatialShape recipes use templates built only from the DSLR training split; `dental_view_type=unknown` avoids imposing a panorama prior on five-view DSLR data.",
        "- Completed runs retain only `best.pth`. AP values are COCO bbox percentage points; ranks include completed rows only.",
        "",
        "| Order | Rank | Method | Status | Source | Epochs | Best epoch | AP | AP50 | AP75 | AR100 | Δ vs best completed | Params | Experiment |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        params = "-" if record.get("params") in (None, "") else f"{int(record['params']) / 1e6:.2f}M"
        lines.append(
            f"| {record['order']} | {record.get('rank', '-')} | {record['method']} | {record['status']} | "
            f"{record['source']} | {record.get('epochs_logged', '-')} | {record.get('best_epoch', '-')} | "
            f"{fmt(record.get('ap'))} | {fmt(record.get('ap50'))} | {fmt(record.get('ap75'))} | "
            f"{fmt(record.get('ar100'))} | {fmt_delta(record.get('delta_vs_best_completed'))} | {params} | "
            f"`{record['experiment']}` |"
        )
    lines += ["", "## Files", "", f"- CSV: `{csv_path}`", f"- Markdown: `{base / 'ablation_summary.md'}`", ""]
    markdown_path = base / "ablation_summary.md"
    markdown_path.write_text("\n".join(lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
