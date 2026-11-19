#!/usr/bin/env python
"""Run serial coco ablations for DUG-HOIF, D-SWFM, and D-CED."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COCO_DATASET = {
    "name": "coco",
    "base_config": "configs/deim_dfine/deim_hgnetv2_s_phone.yml",
    "train_ann": "/home/wuxi/coco/train/annotations_fdi_types.json",
    "val_ann": "/home/wuxi/coco/valid/annotations_fdi_types.json",
    "train_img": "/home/wuxi/coco/train",
    "val_img": "/home/wuxi/coco/valid",
    "fdi_order": "from_categories",
    "view": "unknown",
}


BASE_DUG = {
    "use_dug_hoif_fusion": True,
    "dug_hoif_use_interleave": True,
    "dug_hoif_use_multibranch": True,
    "dug_hoif_use_uncertainty_routing": True,
    "dug_hoif_use_high_order": True,
    "dug_hoif_use_omni_gate": True,
    "dug_hoif_res_init": -2.0,
}

STABLE_DUG_FLAGS = {
    "dug_hoif_use_interleave": True,
    "dug_hoif_use_multibranch": True,
    "dug_hoif_use_uncertainty_routing": False,
    "dug_hoif_use_high_order": False,
    "dug_hoif_use_omni_gate": False,
    "dug_hoif_res_init": -2.0,
}

BASE_DUG_STABLE = {
    "use_dug_hoif_fusion": True,
    **STABLE_DUG_FLAGS,
}


BASE_DSWFM = {
    "use_dswfm": True,
    "dswfm_use_dynamic_fourier": True,
    "dswfm_use_haar_detail": True,
    "dswfm_use_bidirectional": True,
    "dswfm_res_init": -2.4,
    "paper_refine_apply": "proj",
}


BASE_DCED = {
    "use_dced_cascade": True,
    "dced_stages": 1,
    "dced_use_dug_hoif": True,
    "dced_res_init": -2.2,
    "paper_refine_apply": "proj",
}


EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "Original DEIM/D-FINE encoder",
        "updates": {},
    },
    {
        "name": "dug_interleave",
        "description": "+ interleaved two-way injection only",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG,
                "dug_hoif_use_multibranch": False,
            },
        },
    },
    {
        "name": "dug_multibranch",
        "description": "+ local/context/edge branches without routing",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG,
                "dug_hoif_use_uncertainty_routing": False,
                "dug_hoif_use_high_order": False,
                "dug_hoif_use_omni_gate": False,
            },
        },
    },
    {
        "name": "dug_routing",
        "description": "+ local-variance uncertainty routing",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG,
                "dug_hoif_use_high_order": False,
                "dug_hoif_use_omni_gate": False,
            },
        },
    },
    {
        "name": "dug_high_order",
        "description": "+ second-/third-order dental interaction",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG,
                "dug_hoif_use_omni_gate": False,
            },
        },
    },
    {
        "name": "dug_full",
        "description": "+ omni-axis gate / full DUG-HOIF",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG,
            },
        },
    },
    {
        "name": "dswfm",
        "description": "D-SWFM only",
        "updates": {
            "HybridEncoder": {
                **BASE_DSWFM,
            },
        },
    },
    {
        "name": "dced_lite",
        "description": "D-CED with normalized base fusion only",
        "updates": {
            "HybridEncoder": {
                **BASE_DCED,
                "dced_use_dug_hoif": False,
            },
        },
    },
    {
        "name": "dced_dug_hoif",
        "description": "D-CED with DUG-HOIF fusion units",
        "updates": {
            "HybridEncoder": {
                **BASE_DCED,
                **STABLE_DUG_FLAGS,
            },
        },
    },
    {
        "name": "full_dental_neck",
        "description": "DUG-HOIF + D-SWFM + D-CED",
        "updates": {
            "HybridEncoder": {
                **BASE_DUG_STABLE,
                **BASE_DSWFM,
                **BASE_DCED,
                "dced_use_dug_hoif": False,
            },
        },
    },
]


OPTIONAL_EXPERIMENTS = [
    {
        "name": "dswfm_static_fourier",
        "description": "D-SWFM with static Fourier band prior",
        "default": False,
        "updates": {
            "HybridEncoder": {
                **BASE_DSWFM,
                "dswfm_use_dynamic_fourier": False,
            },
        },
    },
    {
        "name": "dswfm_highpass_only",
        "description": "D-SWFM with high-pass detail instead of Haar detail",
        "default": False,
        "updates": {
            "HybridEncoder": {
                **BASE_DSWFM,
                "dswfm_use_haar_detail": False,
            },
        },
    },
    {
        "name": "dswfm_fine_to_coarse",
        "description": "D-SWFM without coarse-to-fine semantic compensation",
        "default": False,
        "updates": {
            "HybridEncoder": {
                **BASE_DSWFM,
                "dswfm_use_bidirectional": False,
            },
        },
    },
    {
        "name": "dced_stage2",
        "description": "D-CED with two cascade stages",
        "default": False,
        "updates": {
            "HybridEncoder": {
                **BASE_DCED,
                **STABLE_DUG_FLAGS,
                "dced_stages": 2,
            },
        },
    },
]


ALL_EXPERIMENTS = EXPERIMENTS + OPTIONAL_EXPERIMENTS


def deep_merge(dst: Dict, src: Dict) -> Dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def subset_coco(ann_file: Path, out_file: Path, max_images: Optional[int]) -> Path:
    if not max_images or max_images <= 0:
        return ann_file
    with ann_file.open("r") as f:
        coco = json.load(f)
    keep_images = coco["images"][:max_images]
    keep_ids = {img["id"] for img in keep_images}
    subset = {
        **coco,
        "images": keep_images,
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep_ids],
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(subset, f)
    return out_file


def write_config(base_config: Path, out_file: Path, updates: Dict) -> None:
    cfg = {"__include__": [str(base_config.resolve())]}
    deep_merge(cfg, updates)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_prior(train_ann: Path, prior_path: Path, fdi_order: str) -> None:
    from build_tooth_templates import build_templates

    build_templates(train_ann, prior_path, fdi_order, min_var=1e-3)


def parse_metrics(output_dir: Path) -> Dict:
    log_file = output_dir / "log.txt"
    if not log_file.exists():
        return {}

    rows = []
    with log_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {}

    best = max(rows, key=lambda r: r.get("test_coco_eval_bbox", [float("-inf")])[0])
    last = rows[-1]
    bbox = best.get("test_coco_eval_bbox", [])
    return {
        "best_epoch": best.get("epoch"),
        "best_AP": bbox[0] if len(bbox) > 0 else None,
        "best_AP50": bbox[1] if len(bbox) > 1 else None,
        "best_AP75": bbox[2] if len(bbox) > 2 else None,
        "best_APs": bbox[3] if len(bbox) > 3 else None,
        "best_APm": bbox[4] if len(bbox) > 4 else None,
        "best_APl": bbox[5] if len(bbox) > 5 else None,
        "last_epoch": last.get("epoch"),
        "last_train_loss": last.get("train_loss"),
        "n_parameters": best.get("n_parameters") or last.get("n_parameters"),
    }


def is_complete(output_dir: Path, epochs: int) -> bool:
    metrics = parse_metrics(output_dir)
    last_epoch = metrics.get("last_epoch")
    return isinstance(last_epoch, int) and last_epoch >= epochs - 1


def selected_experiments(names: Optional[Iterable[str]]) -> List[Dict]:
    if not names:
        return [exp for exp in ALL_EXPERIMENTS if exp.get("default", True)]
    selected = set(names)
    exps = [exp for exp in ALL_EXPERIMENTS if exp["name"] in selected]
    missing = selected - {exp["name"] for exp in exps}
    if missing:
        raise ValueError(f"Unknown experiments: {sorted(missing)}")
    return exps


def write_summary(rows: List[Dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "ablation_summary.csv"
    md_path = output_root / "ablation_summary.md"
    baseline = next(
        (row.get("best_AP") for row in rows
         if row.get("experiment") == "baseline" and isinstance(row.get("best_AP"), float)),
        None,
    )
    fields = [
        "dataset", "experiment", "description", "status", "delta_AP", "best_epoch",
        "best_AP", "best_AP50", "best_AP75", "best_APs", "best_APm", "best_APl",
        "last_epoch", "last_train_loss", "n_parameters", "output_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row_ap = row.get("best_AP")
            if isinstance(baseline, float) and isinstance(row_ap, float):
                row["delta_AP"] = row_ap - baseline
            writer.writerow({k: row.get(k) for k in fields})

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    with md_path.open("w") as f:
        f.write("| Dataset | Experiment | Description | Status | Best Epoch | AP | Delta | AP50 | AP75 | APs | APm | APl | Params | Output |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            row_ap = row.get("best_AP")
            delta = row_ap - baseline if isinstance(baseline, float) and isinstance(row_ap, float) else None
            f.write(
                f"| {row.get('dataset')} | {row.get('experiment')} | {row.get('description')} | "
                f"{row.get('status')} | {fmt(row.get('best_epoch'))} | {fmt(row_ap)} | {fmt(delta)} | "
                f"{fmt(row.get('best_AP50'))} | {fmt(row.get('best_AP75'))} | "
                f"{fmt(row.get('best_APs'))} | {fmt(row.get('best_APm'))} | {fmt(row.get('best_APl'))} | "
                f"{fmt(row.get('n_parameters'))} | {row.get('output_dir')} |\n"
            )
        f.write("\nNotes:\n")
        f.write("- Dataset is fixed to `/home/wuxi/coco` train/valid annotations.\n")
        f.write("- Default sequence: Baseline, DUG-HOIF component ablations, D-SWFM, D-CED-lite, D-CED w/ DUG-HOIF, Full dental neck.\n")
        f.write("- The table is rewritten after each serial experiment, so interrupted runs keep partial results.\n")


def append_command(command_file: Path, cmd: List[str]) -> None:
    command_file.parent.mkdir(parents=True, exist_ok=True)
    with command_file.open("a") as f:
        f.write(" ".join(cmd) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-config", type=Path, default=Path(COCO_DATASET["base_config"]))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/coco_dental_neck_ablation_fixed"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--stop-epoch", type=int, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    config_root = output_root / "configs" / COCO_DATASET["name"]
    subset_root = output_root / "subsets" / COCO_DATASET["name"]
    prior_root = output_root / "priors"
    command_file = output_root / "serial_commands.sh"
    experiments = selected_experiments(args.experiments)
    rows: List[Dict] = [
        {
            "dataset": COCO_DATASET["name"],
            "experiment": exp["name"],
            "description": exp["description"],
            "status": "pending",
            "output_dir": str(output_root / COCO_DATASET["name"] / exp["name"]),
        }
        for exp in experiments
    ]
    write_summary(rows, output_root)

    if command_file.exists():
        command_file.unlink()

    train_ann = subset_coco(
        Path(COCO_DATASET["train_ann"]),
        subset_root / "train_subset.json",
        args.max_train_images,
    )
    val_ann = subset_coco(
        Path(COCO_DATASET["val_ann"]),
        subset_root / "val_subset.json",
        args.max_val_images,
    )
    prior_path = prior_root / "coco_tooth_templates.pt"
    if not args.dry_run:
        build_prior(train_ann, prior_path, COCO_DATASET["fdi_order"])

    for row_idx, exp in enumerate(experiments):
        out_dir = output_root / COCO_DATASET["name"] / exp["name"]
        updates = {
            "output_dir": str(out_dir),
            "epoches": args.epochs,
            "device": args.device,
            "dental_prior_path": str(prior_path),
            "keep_only_best_checkpoints": True,
            "train_dataloader": {
                "total_batch_size": args.train_batch_size,
                "num_workers": args.num_workers,
                "dataset": {
                    "img_folder": COCO_DATASET["train_img"],
                    "ann_file": str(train_ann),
                },
                "collate_fn": {},
            },
            "val_dataloader": {
                "total_batch_size": args.val_batch_size,
                "num_workers": args.num_workers,
                "dataset": {
                    "img_folder": COCO_DATASET["val_img"],
                    "ann_file": str(val_ann),
                },
            },
            "PostProcessor": {
                "dental_view_type": COCO_DATASET["view"],
            },
        }
        if args.stop_epoch is not None:
            updates["train_dataloader"]["dataset"]["transforms"] = {
                "policy": {"epoch": args.stop_epoch},
            }
            updates["train_dataloader"]["collate_fn"]["stop_epoch"] = args.stop_epoch

        deep_merge(updates, copy.deepcopy(exp["updates"]))
        cfg_path = config_root / f"{exp['name']}.yml"
        write_config(REPO_ROOT / args.base_config, cfg_path, updates)

        cmd = [args.python, "train.py", "-c", str(cfg_path)]
        if not args.no_amp:
            cmd.append("--use-amp")
        if args.seed is not None:
            cmd.append(f"--seed={args.seed}")
        append_command(command_file, cmd)

        status = "dry_run"
        if args.skip_existing and is_complete(out_dir, args.epochs):
            status = "skipped"
            print(f"\n[Coco Neck Ablation] skip completed {exp['name']}")
        elif out_dir.exists() and not args.dry_run:
            shutil.rmtree(out_dir)

        if status == "skipped":
            pass
        elif args.dry_run:
            print(" ".join(cmd))
        else:
            print(f"\n[Coco Neck Ablation] {exp['name']}")
            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"

        metrics = parse_metrics(out_dir)
        rows[row_idx] = {
            "dataset": COCO_DATASET["name"],
            "experiment": exp["name"],
            "description": exp["description"],
            "status": status,
            "output_dir": str(out_dir),
            **metrics,
        }
        write_summary(rows, output_root)

    write_summary(rows, output_root)
    print(f"\nCommands written to {command_file}")
    print(f"Summary written to {output_root / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
