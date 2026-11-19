#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("dentex_spatial_shape_dualopt", "DENTEX2023", "SpatialShape+DualOpt"),
    ("dentex_prototype_dualopt", "DENTEX2023", "Prototype+DualOpt"),
    ("dentex_cmfa_dualopt", "DENTEX2023", "CMFA+DualOpt"),
    ("dentex_mpf_dualopt", "DENTEX2023", "MPF+DualOpt"),
    ("dentex_cmfa_hooa_dualopt", "DENTEX2023", "CMFA+HOOA+DualOpt"),
    ("dentex_cmfa_hooa_prototype", "DENTEX2023", "CMFA+HOOA+Prototype"),
    ("dentex_cmfa_hooa_spatial_shape", "DENTEX2023", "CMFA+HOOA+SpatialShape"),
    ("dentex_cmfa_hooa_spatial_shape_dualopt", "DENTEX2023", "CMFA+HOOA+SpatialShape+DualOpt"),
    ("dentex_cmfa_hooa_cedv2", "DENTEX2023", "CMFA+HOOA+CEDv2"),
    ("dentex_cmfa_hooa_cedv2_dualopt", "DENTEX2023", "CMFA+HOOA+CEDv2+DualOpt"),
    ("dentex_cmfa_hooa_cedv2_spatial_shape", "DENTEX2023", "CMFA+HOOA+CEDv2+SpatialShape"),
    ("dentex_cmfa_hooa_cedv2_spatial_shape_dualopt", "DENTEX2023", "CMFA+HOOA+CEDv2+SpatialShape+DualOpt"),
    ("coco_spatial_shape_dualopt", "COCO-/home/wuxi/coco", "SpatialShape+DualOpt"),
    ("coco_prototype_dualopt", "COCO-/home/wuxi/coco", "Prototype+DualOpt"),
    ("coco_cmfa_dualopt", "COCO-/home/wuxi/coco", "CMFA+DualOpt"),
    ("coco_mpf_dualopt", "COCO-/home/wuxi/coco", "MPF+DualOpt"),
    ("coco_cmfa_hooa_dualopt", "COCO-/home/wuxi/coco", "CMFA+HOOA+DualOpt"),
    ("coco_cmfa_hooa_prototype", "COCO-/home/wuxi/coco", "CMFA+HOOA+Prototype"),
    ("coco_cmfa_hooa_spatial_shape", "COCO-/home/wuxi/coco", "CMFA+HOOA+SpatialShape"),
    ("coco_cmfa_hooa_spatial_shape_dualopt", "COCO-/home/wuxi/coco", "CMFA+HOOA+SpatialShape+DualOpt"),
    ("coco_cmfa_hooa_cedv2", "COCO-/home/wuxi/coco", "CMFA+HOOA+CEDv2"),
    ("coco_cmfa_hooa_cedv2_dualopt", "COCO-/home/wuxi/coco", "CMFA+HOOA+CEDv2+DualOpt"),
    ("coco_cmfa_hooa_cedv2_spatial_shape", "COCO-/home/wuxi/coco", "CMFA+HOOA+CEDv2+SpatialShape"),
    ("coco_cmfa_hooa_cedv2_spatial_shape_dualopt", "COCO-/home/wuxi/coco", "CMFA+HOOA+CEDv2+SpatialShape+DualOpt"),
]


def clean_metric(value):
    if value is None:
        return None
    value = float(value)
    return None if value < 0 else value


def parse_log(log_path):
    rows = []
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("test_coco_eval_bbox"):
            rows.append(record)
    if not rows:
        return None

    best = max(rows, key=lambda r: (r.get("test_coco_eval_bbox") or [-1])[0])
    last = rows[-1]
    b = best["test_coco_eval_bbox"]
    lb = last["test_coco_eval_bbox"]
    parsed = {
        "epochs_logged": len(rows),
        "first_epoch": rows[0].get("epoch"),
        "last_epoch": last.get("epoch"),
        "best_epoch": best.get("epoch"),
        "ap": b[0] if len(b) > 0 else None,
        "ap50": b[1] if len(b) > 1 else None,
        "ap75": b[2] if len(b) > 2 else None,
        "aps": b[3] if len(b) > 3 else None,
        "apm": b[4] if len(b) > 4 else None,
        "apl": b[5] if len(b) > 5 else None,
        "ar1": b[6] if len(b) > 6 else None,
        "ar10": b[7] if len(b) > 7 else None,
        "ar100": b[8] if len(b) > 8 else None,
        "final_ap": lb[0] if len(lb) > 0 else None,
        "final_ap50": lb[1] if len(lb) > 1 else None,
        "final_ap75": lb[2] if len(lb) > 2 else None,
        "params": best.get("n_parameters"),
        "train_lr_best": best.get("train_lr"),
        "train_loss_best": best.get("train_loss"),
    }
    for key in ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100", "final_ap", "final_ap50", "final_ap75"):
        parsed[key] = clean_metric(parsed.get(key))
    return parsed


def fmt(value, scale=100.0):
    if value in (None, ""):
        return "-"
    return f"{float(value) * scale:.3f}"


def fmt_delta(value):
    if value in (None, ""):
        return "-"
    return f"{float(value) * 100.0:+.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/home/wuxi/Dent-FINE/outputs/ablation_combo_modules_cross_dataset")
    args = parser.parse_args()

    base = Path(args.base_dir)
    records = []
    for order, (experiment, dataset, method) in enumerate(EXPERIMENTS, 1):
        exp_dir = base / experiment
        parsed = parse_log(exp_dir / "log.txt")
        record = {
            "order": order,
            "dataset": dataset,
            "experiment": experiment,
            "method": method,
            "status": "missing",
            "output_dir": str(exp_dir),
            "log_path": str(exp_dir / "log.txt"),
            "best_weight": str(exp_dir / "best.pth") if (exp_dir / "best.pth").exists() else "",
            "best_stg1_weight": str(exp_dir / "best_stg1.pth") if (exp_dir / "best_stg1.pth").exists() else "",
            "best_stg2_weight": str(exp_dir / "best_stg2.pth") if (exp_dir / "best_stg2.pth").exists() else "",
        }
        if parsed:
            record.update(parsed)
            record["status"] = "done" if record.get("last_epoch") is not None and record["last_epoch"] >= 159 else "partial"
        records.append(record)

    for dataset in sorted({r["dataset"] for r in records}):
        done = sorted(
            [r for r in records if r["dataset"] == dataset and r.get("ap") is not None],
            key=lambda r: r["ap"],
            reverse=True,
        )
        best_ap = done[0]["ap"] if done else None
        for rank, record in enumerate(done, 1):
            record["rank_in_dataset"] = rank
            record["delta_vs_dataset_best"] = record["ap"] - best_ap if best_ap is not None else None

    fieldnames = [
        "order", "dataset", "rank_in_dataset", "experiment", "method", "status",
        "epochs_logged", "first_epoch", "last_epoch", "best_epoch",
        "ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100",
        "final_ap", "final_ap50", "final_ap75", "delta_vs_dataset_best",
        "params", "train_lr_best", "train_loss_best",
        "output_dir", "log_path", "best_weight", "best_stg1_weight", "best_stg2_weight",
    ]
    csv_path = base / "ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: "" if record.get(key) is None else record.get(key, "") for key in fieldnames})

    lines = [
        "# Combination Ablation Summary",
        "",
        f"Base directory: `{base}`",
        "",
        "Metrics are COCO bbox metrics. AP values are percentage points; undefined scale metrics are shown as `-`.",
        "",
    ]
    for dataset in ("DENTEX2023", "COCO-/home/wuxi/coco"):
        table = sorted(
            [r for r in records if r["dataset"] == dataset],
            key=lambda r: (r.get("ap") is not None, r.get("ap") or -1),
            reverse=True,
        )
        lines += [
            f"## {dataset}",
            "",
            "| Rank | Experiment | Method | Status | Best AP | AP50 | AP75 | APs | APm | APl | AR100 | Best Epoch | Last AP | Delta vs Best | Params |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for record in table:
            params = "-"
            if record.get("params") not in (None, ""):
                params = f"{int(record['params']) / 1e6:.2f}M"
            lines.append(
                f"| {record.get('rank_in_dataset', '-')} | `{record['experiment']}` | {record['method']} | {record['status']} | "
                f"{fmt(record.get('ap'))} | {fmt(record.get('ap50'))} | {fmt(record.get('ap75'))} | {fmt(record.get('aps'))} | "
                f"{fmt(record.get('apm'))} | {fmt(record.get('apl'))} | {fmt(record.get('ar100'))} | "
                f"{record.get('best_epoch', '-')} | {fmt(record.get('final_ap'))} | {fmt_delta(record.get('delta_vs_dataset_best'))} | {params} |"
            )
        lines.append("")

    lines += [
        "## Files",
        "",
        f"- CSV: `{csv_path}`",
        f"- Markdown: `{base / 'ablation_summary.md'}`",
        "",
    ]
    md_path = base / "ablation_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
