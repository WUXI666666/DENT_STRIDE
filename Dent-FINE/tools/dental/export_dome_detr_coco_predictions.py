#!/usr/bin/env python3
"""Export Dome-DETR detections and common COCO metrics for dental datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DOME = Path("/home/wuxi/Dent-FINE/third_party/dome_detr")
sys.path.insert(0, str(DOME))
from src.core import YAMLConfig, yaml_utils  # noqa: E402


METRICS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload.get("ema"), dict) and isinstance(payload["ema"].get("module"), dict):
        return payload["ema"]["module"]
    if isinstance(payload.get("model"), dict):
        return payload["model"]
    raise KeyError(f"No model state in {path}")


def xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = map(float, box.tolist())
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-u", "--update", action="append", default=[])
    args = parser.parse_args()

    updates = list(args.update) + [
        f"val_dataloader.total_batch_size={args.batch}",
        f"val_dataloader.num_workers={args.workers}",
    ]
    cfg = YAMLConfig(str(args.config), **yaml_utils.parse_cli(updates))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    device = torch.device(args.device)
    model = cfg.model.to(device).eval()
    missing, unexpected = model.load_state_dict(state(args.checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/config mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")
    postprocessor = cfg.postprocessor.to(device).eval()

    rows: list[dict] = []
    with torch.inference_mode():
        for samples, targets in cfg.val_dataloader:
            samples = samples.to(device)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(samples, targets=targets)
            for key in ("pred_logits", "pred_boxes"):
                if key in outputs:
                    outputs[key] = outputs[key].float()
            sizes = torch.stack([target["orig_size"] for target in targets])
            results = postprocessor(outputs, sizes)
            for target, result in zip(targets, results):
                image_id = int(target["image_id"].item())
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    rows.append({
                        "image_id": image_id, "category_id": int(label.item()),
                        "bbox": xyxy_to_xywh(box.detach().cpu()), "score": float(score.item()),
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "coco_predictions.json"
    prediction_path.write_text(json.dumps(rows) + "\n")
    coco = COCO(str(args.annotation))
    evaluator = COCOeval(coco, coco.loadRes(str(prediction_path)), "bbox")
    evaluator.params.imgIds = sorted(coco.getImgIds())
    evaluator.params.catIds = sorted(coco.getCatIds())
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    payload = {
        "framework": "Dome-DETR", "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()), "annotation": str(args.annotation.resolve()),
        "prediction_count": len(rows), "num_parameters": sum(p.numel() for p in model.parameters()),
        "metrics": {name: float(value) for name, value in zip(METRICS, evaluator.stats.tolist())},
        "prediction_file": str(prediction_path.resolve()), "updates": updates,
    }
    (args.output_dir / "coco_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
