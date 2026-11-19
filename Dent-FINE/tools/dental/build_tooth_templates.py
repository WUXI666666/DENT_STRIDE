#!/usr/bin/env python
"""Build spatial and bbox-shape tooth templates from COCO annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.deim.utils_arch import default_fdi52_order, resolve_fdi_label_order


def _category_order(coco: Dict, order_arg, num_classes: int) -> List[int]:
    if order_arg == "from_categories":
        cats = sorted(coco.get("categories", []), key=lambda x: int(x["id"]))
        values = []
        for cat in cats[:num_classes]:
            try:
                values.append(int(cat["name"]))
            except Exception:
                values.append(int(cat["id"]))
        return values
    resolved = resolve_fdi_label_order(order_arg, num_classes=num_classes)
    if resolved is None:
        return default_fdi52_order()[:num_classes]
    return resolved


def _cov_inv(points: List[List[float]], default_var: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if len(points) == 0:
        return torch.zeros(2), torch.eye(2) / default_var, False
    x = torch.as_tensor(points, dtype=torch.float32)
    mu = x.mean(dim=0)
    if len(points) == 1:
        cov = torch.eye(2) * default_var
    else:
        centered = x - mu
        cov = centered.t().matmul(centered) / max(1, len(points) - 1)
        cov = cov + torch.eye(2) * default_var
    return mu, torch.linalg.inv(cov), True


def build_templates(ann_file: Path, output: Path, fdi_order, min_var: float) -> Dict:
    with ann_file.open("r") as f:
        coco = json.load(f)

    images = {int(img["id"]): img for img in coco["images"]}
    max_cat = max(int(cat["id"]) for cat in coco["categories"])
    num_classes = max_cat + 1
    centers = defaultdict(list)
    shapes = defaultdict(list)

    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        if cid < 0 or cid >= num_classes:
            continue
        img = images.get(int(ann["image_id"]))
        if img is None:
            continue
        w_img = float(img["width"])
        h_img = float(img["height"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0 or w_img <= 0 or h_img <= 0:
            continue
        centers[cid].append([(x + 0.5 * w) / w_img, (y + 0.5 * h) / h_img])
        shapes[cid].append([max(w / w_img, 1e-5), max(h / h_img, 1e-5)])

    spatial_mu, spatial_inv_cov, spatial_valid = [], [], []
    shape_mu, shape_inv_cov, shape_valid = [], [], []
    for cid in range(num_classes):
        mu, inv, valid = _cov_inv(centers[cid], min_var)
        spatial_mu.append(mu)
        spatial_inv_cov.append(inv)
        spatial_valid.append(valid)

        shape_points = torch.as_tensor(shapes[cid], dtype=torch.float32).clamp_min(1e-5).log().tolist()
        mu, inv, valid = _cov_inv(shape_points, min_var)
        shape_mu.append(mu)
        shape_inv_cov.append(inv)
        shape_valid.append(valid)

    fdi_label_order = _category_order(coco, fdi_order, num_classes)
    prior = {
        "num_classes": num_classes,
        "fdi_label_order": torch.as_tensor(fdi_label_order, dtype=torch.long),
        "spatial_mu": torch.stack(spatial_mu),
        "spatial_inv_cov": torch.stack(spatial_inv_cov),
        "spatial_valid": torch.as_tensor(spatial_valid, dtype=torch.bool),
        "shape_mu": torch.stack(shape_mu),
        "shape_inv_cov": torch.stack(shape_inv_cov),
        "shape_valid": torch.as_tensor(shape_valid, dtype=torch.bool),
        "annotation_file": str(ann_file),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior, output)

    summary = {
        "annotation_file": str(ann_file),
        "output": str(output),
        "num_classes": num_classes,
        "valid_spatial_classes": int(prior["spatial_valid"].sum().item()),
        "valid_shape_classes": int(prior["shape_valid"].sum().item()),
        "fdi_label_order": fdi_label_order,
    }
    with output.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fdi-order", default="from_categories")
    parser.add_argument("--min-var", type=float, default=1e-3)
    args = parser.parse_args()
    summary = build_templates(args.ann_file, args.output, args.fdi_order, args.min_var)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
