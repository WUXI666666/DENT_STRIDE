#!/usr/bin/env python3
"""Train the official Dome-DETR-S implementation on one dental dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOME = ROOT / "third_party/dome_detr"
BASE_CONFIG = DOME / "configs/dome/Dome-S-ViewNet3-DSLR-FDI52.yml"
PYTHON = Path("/home/wuxi/miniconda3/envs/deimv2/bin/python")
EXPORTER = ROOT / "tools/dental/export_dome_detr_coco_predictions.py"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tuning", type=Path, default=None,
                        help="Optional checkpoint used as model-only initialization for target-domain tuning.")
    parser.add_argument("--max-num-select", type=int, default=None,
                        help="Optional bounded internal DOME encoder-candidate cap.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = args.output_dir / "best.pth"
    metrics = args.output_dir / "coco_eval/coco_metrics.json"
    if (args.output_dir / "COMPLETED").is_file() and best.is_file() and metrics.is_file():
        print(f"[skip] {args.output_dir}")
        return

    updates = [
        f"num_classes={args.num_classes}", "remap_mscoco_category=False",
        f"train_dataloader.dataset.img_folder={args.train_images.resolve()}",
        f"train_dataloader.dataset.ann_file={args.train_ann.resolve()}",
        f"val_dataloader.dataset.img_folder={args.val_images.resolve()}",
        f"val_dataloader.dataset.ann_file={args.val_ann.resolve()}",
        f"train_dataloader.total_batch_size={args.batch}",
        f"val_dataloader.total_batch_size={args.batch}",
        f"train_dataloader.num_workers={args.workers}",
        f"val_dataloader.num_workers={args.workers}",
        f"epoches={args.epochs}", "eval_spatial_size=[640,640]",
        "train_dataloader.collate_fn.base_size=640",
        "train_dataloader.collate_fn.base_size_repeat=null",
    ]
    if args.max_num_select is not None:
        updates.append(f"DomeTransformer.max_num_select={args.max_num_select}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    (args.output_dir / "run_protocol.json").write_text(json.dumps({
        "train_annotation": str(args.train_ann.resolve()),
        "validation_annotation": str(args.val_ann.resolve()),
        "train_images": str(args.train_images.resolve()),
        "validation_images": str(args.val_images.resolve()),
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "gpu": args.gpu,
        "tuning_checkpoint": str(args.tuning.resolve()) if args.tuning is not None else None,
        "max_num_select": args.max_num_select,
        "qualitative_only": args.tuning is not None or args.max_num_select is not None,
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "RUNNING").touch()
    (args.output_dir / "FAILED").unlink(missing_ok=True)
    try:
        candidates = [args.output_dir / name for name in ("best_stg2.pth", "last.pth", "best_stg1.pth", "best.pth")]
        resume = next((path for path in candidates if path.is_file()), None)
        command = [
            str(PYTHON), "train.py", "-c", str(BASE_CONFIG), "--use-amp", "--seed=0",
            "-d", "cuda", "--output-dir", str(args.output_dir),
        ]
        if resume:
            command += ["-r", str(resume)]
        elif args.tuning is not None:
            if not args.tuning.is_file():
                raise FileNotFoundError(args.tuning)
            command += ["-t", str(args.tuning.resolve())]
        command += ["-u", *updates]
        run(command, cwd=DOME, env=env)
        selected = next((path for path in (args.output_dir / "best_stg2.pth", args.output_dir / "best_stg1.pth", args.output_dir / "last.pth", best) if path.is_file()), None)
        if selected is None:
            raise FileNotFoundError("Dome-DETR training produced no checkpoint")
        if selected != best:
            selected.replace(best)
        export_command = [
            str(PYTHON), str(EXPORTER), "--config", str(BASE_CONFIG),
            "--checkpoint", str(best), "--annotation", str(args.val_ann.resolve()),
            "--output-dir", str(args.output_dir / "coco_eval"), "--device", "cuda:0",
            "--batch", str(args.batch), "--workers", str(args.workers),
        ]
        for update in updates:
            export_command += ["-u", update]
        run(export_command, cwd=ROOT, env=env)
        for checkpoint in args.output_dir.glob("*.pth"):
            if checkpoint != best:
                checkpoint.unlink()
        (args.output_dir / "RUNNING").unlink(missing_ok=True)
        (args.output_dir / "FAILED").unlink(missing_ok=True)
        (args.output_dir / "COMPLETED").touch()
    except Exception:
        (args.output_dir / "RUNNING").unlink(missing_ok=True)
        (args.output_dir / "FAILED").touch()
        raise


if __name__ == "__main__":
    main()
