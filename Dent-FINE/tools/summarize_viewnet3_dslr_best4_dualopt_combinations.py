#!/usr/bin/env python3
"""Build a live total table for the View-Net3 DSLR Best4+DualOpt ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    ("viewnet3_dualopt_only", "DualOpt-only", ()),
    ("viewnet3_cmfa_dualopt", "CMFA + DualOpt", ("CMFA",)),
    ("viewnet3_smwg_dualopt", "SMWG + DualOpt", ("SMWG",)),
    ("viewnet3_ucm_dualopt", "UCM + DualOpt", ("UCM",)),
    ("viewnet3_mpf_dualopt", "MPF + DualOpt", ("MPF",)),
    ("viewnet3_cmfa_smwg_dualopt", "CMFA + SMWG + DualOpt", ("CMFA", "SMWG")),
    ("viewnet3_cmfa_ucm_dualopt", "CMFA + UCM + DualOpt", ("CMFA", "UCM")),
    ("viewnet3_cmfa_mpf_dualopt", "CMFA + MPF + DualOpt", ("CMFA", "MPF")),
    ("viewnet3_smwg_ucm_dualopt", "SMWG + UCM + DualOpt", ("SMWG", "UCM")),
    ("viewnet3_smwg_mpf_dualopt", "SMWG + MPF + DualOpt", ("SMWG", "MPF")),
    ("viewnet3_ucm_mpf_dualopt", "UCM + MPF + DualOpt", ("UCM", "MPF")),
    ("viewnet3_cmfa_smwg_ucm_dualopt", "CMFA + SMWG + UCM + DualOpt", ("CMFA", "SMWG", "UCM")),
    ("viewnet3_cmfa_smwg_mpf_dualopt", "CMFA + SMWG + MPF + DualOpt", ("CMFA", "SMWG", "MPF")),
    ("viewnet3_cmfa_ucm_mpf_dualopt", "CMFA + UCM + MPF + DualOpt", ("CMFA", "UCM", "MPF")),
    ("viewnet3_smwg_ucm_mpf_dualopt", "SMWG + UCM + MPF + DualOpt", ("SMWG", "UCM", "MPF")),
    ("viewnet3_cmfa_smwg_ucm_mpf_dualopt", "CMFA + SMWG + UCM + MPF + DualOpt", ("CMFA", "SMWG", "UCM", "MPF")),
]
METRICS = ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100", "final_ap", "final_ap50", "final_ap75")


def clean_metric(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if value < 0 else value


def parse_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    # Resumed runs can append repeated epoch numbers. Keep the latest record of
    # each epoch so epochs_logged remains meaningful.
    by_epoch: dict[int, dict[str, Any]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = record.get("test_coco_eval_bbox")
        epoch = record.get("epoch")
        if not isinstance(metrics, list) or not isinstance(epoch, int):
            continue
        by_epoch[epoch] = record
    if not by_epoch:
        return None

    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    best = max(rows, key=lambda row: (row.get("test_coco_eval_bbox") or [-1])[0])
    last = rows[-1]
    best_metrics = best["test_coco_eval_bbox"]
    last_metrics = last["test_coco_eval_bbox"]
    record: dict[str, Any] = {
        "epochs_logged": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": last["epoch"],
        "best_epoch": best["epoch"],
        "params": best.get("n_parameters"),
        "train_lr_best": best.get("train_lr"),
        "train_loss_best": best.get("train_loss"),
    }
    for index, metric in enumerate(("ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100")):
        record[metric] = clean_metric(best_metrics[index] if len(best_metrics) > index else None)
    for index, metric in enumerate(("final_ap", "final_ap50", "final_ap75")):
        record[metric] = clean_metric(last_metrics[index] if len(last_metrics) > index else None)
    return record


def status_for(exp_dir: Path, parsed: dict[str, Any] | None, expected_last_epoch: int) -> str:
    if (exp_dir / "FAILED").exists():
        return "failed"
    if parsed is None:
        return "queued"
    if parsed.get("last_epoch", -1) >= expected_last_epoch:
        return "done"
    if (exp_dir / "RUNNING").exists():
        return "running"
    return "partial"


def fmt(value: Any, scale: float = 100.0) -> str:
    return "-" if value in (None, "") else f"{float(value) * scale:.3f}"


def fmt_delta(value: Any) -> str:
    return "-" if value in (None, "") else f"{float(value) * 100.0:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_best4_dualopt_combinations"))
    parser.add_argument("--expected-epochs", type=int, default=160)
    args = parser.parse_args()
    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be positive")

    base = args.base_dir
    base.mkdir(parents=True, exist_ok=True)
    expected_last_epoch = args.expected_epochs - 1
    records: list[dict[str, Any]] = []

    for order, (experiment, method, enabled) in enumerate(EXPERIMENTS, 1):
        exp_dir = base / experiment
        parsed = parse_log(exp_dir / "log.txt")
        record: dict[str, Any] = {
            "order": order,
            "dataset": "View-Net3 DSLR FDI52",
            "experiment": experiment,
            "method": method,
            "enabled_modules": "+".join(enabled) if enabled else "none",
            "cmfa": "yes" if "CMFA" in enabled else "no",
            "smwg": "yes" if "SMWG" in enabled else "no",
            "ucm": "yes" if "UCM" in enabled else "no",
            "mpf": "yes" if "MPF" in enabled else "no",
            "optimizer": "DualOptAdamW",
            "output_dir": str(exp_dir),
            "log_path": str(exp_dir / "log.txt"),
            "best_weight": str(exp_dir / "best.pth") if (exp_dir / "best.pth").exists() else "",
            "best_stg1_weight": str(exp_dir / "best_stg1.pth") if (exp_dir / "best_stg1.pth").exists() else "",
            "best_stg2_weight": str(exp_dir / "best_stg2.pth") if (exp_dir / "best_stg2.pth").exists() else "",
            "last_weight": str(exp_dir / "last.pth") if (exp_dir / "last.pth").exists() else "",
        }
        if parsed:
            record.update(parsed)
        record["status"] = status_for(exp_dir, parsed, expected_last_epoch)
        records.append(record)

    completed = sorted(
        (record for record in records if record["status"] == "done" and record.get("ap") is not None),
        key=lambda record: float(record["ap"]), reverse=True,
    )
    best_ap = completed[0]["ap"] if completed else None
    for rank, record in enumerate(completed, 1):
        record["rank"] = rank
        record["delta_vs_best_completed"] = float(record["ap"]) - float(best_ap)

    fields = [
        "order", "rank", "dataset", "experiment", "method", "enabled_modules",
        "cmfa", "smwg", "ucm", "mpf", "optimizer", "status", "epochs_logged",
        "first_epoch", "last_epoch", "best_epoch", "ap", "ap50", "ap75", "aps",
        "apm", "apl", "ar1", "ar10", "ar100", "final_ap", "final_ap50", "final_ap75",
        "delta_vs_best_completed", "params", "train_lr_best", "train_loss_best",
        "output_dir", "log_path", "best_weight", "best_stg1_weight", "best_stg2_weight", "last_weight",
    ]
    csv_path = base / "ablation_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: "" if record.get(field) is None else record.get(field, "") for field in fields})

    lines = [
        "# View-Net3 DSLR FDI52: Best4 + DualOpt Combination Ablation",
        "",
        f"- Dataset: 750 training images / 190 validation images; canonical 52-class FDI label space.",
        f"- Queue: 16 serial experiments on one GPU; expected length: {args.expected_epochs} epochs each.",
        "- Every row uses DualOptAdamW. Only CMFA, SMWG, UCM, and MPF vary.",
        "- AP values are COCO bbox percentage points. Ranks and deltas include completed runs only.",
        "",
        "| Order | Rank | Modules | Status | Epochs | Best epoch | AP | AP50 | AP75 | AR100 | Δ vs best completed | Params | Experiment |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        params = "-" if record.get("params") in (None, "") else f"{int(record['params']) / 1e6:.2f}M"
        lines.append(
            f"| {record['order']} | {record.get('rank', '-')} | {record['enabled_modules']} | {record['status']} | "
            f"{record.get('epochs_logged', '-')} | {record.get('best_epoch', '-')} | {fmt(record.get('ap'))} | "
            f"{fmt(record.get('ap50'))} | {fmt(record.get('ap75'))} | {fmt(record.get('ar100'))} | "
            f"{fmt_delta(record.get('delta_vs_best_completed'))} | {params} | `{record['experiment']}` |"
        )
    lines += ["", "## Files", "", f"- CSV: `{csv_path}`", f"- Markdown: `{base / 'ablation_summary.md'}`", ""]
    markdown_path = base / "ablation_summary.md"
    markdown_path.write_text("\n".join(lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
