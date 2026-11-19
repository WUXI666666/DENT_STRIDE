#!/usr/bin/env python3
"""Export Phone A07 Proto-Aux evidence from its selected checkpoint.

This script deliberately uses the A07 patched implementation and the selected
checkpoint/config, rather than the older ``coco_single_prototype`` artifact
used by the original Figure 10 generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/wuxi/Dent-FINE")
PAPER = Path("/home/wuxi/Dent-FINE-paper")
IMPL = REPO / "experiments/a07_proto_aux_ap_patch_1024"
CONFIG = REPO / "configs/paper_cohort_search_20260915/phone_a07_proto_aux_C05_seed0.yml"
CHECKPOINT = REPO / "outputs/paper_cohort_search_20260915/phone_a07_proto_aux_C05/seed_0/best.pth"
ANN = Path("/home/wuxi/coco/valid/annotations_fdi_types.json")
TRAIN_ANN = Path("/home/wuxi/coco/train/annotations_fdi_types.json")
SCORE = 0.30
IOU = 0.50

sys.path.insert(0, str(IMPL))
from engine.core import YAMLConfig, yaml_utils  # noqa: E402


def checkpoint_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("ema", payload.get("model", payload)) if isinstance(payload, dict) else payload
    if isinstance(state, dict) and "module" in state and isinstance(state["module"], dict):
        state = state["module"]
    return state


def phone_order() -> list[int]:
    data = json.loads(ANN.read_text())
    return [int(row["name"]) for row in sorted(data["categories"], key=lambda row: int(row["id"]))]


def configure_publication_font() -> str:
    """Use a Times-compatible serif face and record the actual installed font.

    The host image does not ship Microsoft Times New Roman.  Nimbus Roman is
    its metric-compatible TeX/URW substitute; choosing it explicitly avoids
    silently falling back to a sans-serif face in publication figures.
    """
    candidates = [
        ("Times New Roman", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        ("Times New Roman", "/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
        ("Nimbus Roman", "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"),
        ("Liberation Serif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
    ]
    for family, filename in candidates:
        path = Path(filename)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
            except Exception:
                pass
            plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
            return family
    plt.rcParams.update({"font.family": "serif", "axes.unicode_minus": False})
    return "serif (system fallback)"


def prototype_panel(out: Path) -> dict:
    state = checkpoint_state(CHECKPOINT)
    prototypes = state["decoder.tooth_prototypes"].float().numpy()
    order = np.asarray(phone_order())
    normalized = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-9)
    centroids = normalized.mean(axis=1)
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-9)
    similarity = centroids @ centroids.T
    cohesion = []
    cohesion_min = []
    for rows in normalized:
        pair = rows @ rows.T
        values = pair[np.triu_indices(pair.shape[0], k=1)]
        cohesion.append(float(values.mean()))
        cohesion_min.append(float(values.min()))

    actual_font = configure_publication_font()
    train = json.loads(TRAIN_ANN.read_text())
    train_categories = {int(row["id"]): int(row["name"]) for row in train["categories"]}
    support = defaultdict(int)
    for row in train["annotations"]:
        support[train_categories[int(row["category_id"])]] += 1
    support_by_fdi = {str(code): int(support[code]) for code in order}
    zero_support = [int(code) for code in order if support[code] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.18, .82]})
    unsupported = {int(code) for code in zero_support}
    masked_similarity = similarity.copy()
    for idx, code in enumerate(order):
        if int(code) in unsupported:
            masked_similarity[idx, :] = np.nan
            masked_similarity[:, idx] = np.nan
    similarity_cmap = plt.get_cmap("coolwarm").copy()
    similarity_cmap.set_bad("#E6E6E6")
    image = axes[0].imshow(np.ma.masked_invalid(masked_similarity), cmap=similarity_cmap, vmin=-1, vmax=1, interpolation="nearest")
    ticks = np.arange(len(order))
    axes[0].set_xticks(ticks, order, rotation=90, fontsize=7)
    axes[0].set_yticks(ticks, order, fontsize=7)
    axes[0].set_xlabel("FDI code")
    axes[0].set_ylabel("FDI code")
    axes[0].set_title("Prototype centroid cosine similarity\n(gray = no training support)")
    fig.colorbar(image, ax=axes[0], fraction=.045, pad=.03)
    y = np.arange(len(order))
    palette = ["#2F6B9A" if code // 10 in (1, 5) else "#5A9E92" if code // 10 in (2, 6) else "#D88931" if code // 10 in (3, 7) else "#765B9A" for code in order]
    finite = np.asarray([np.nan if int(code) in zero_support else value for code, value in zip(order, cohesion)], dtype=float)
    finite_min = np.asarray([np.nan if int(code) in zero_support else value for code, value in zip(order, cohesion_min)], dtype=float)
    axes[1].barh(y, finite, color=palette, edgecolor="#17324D", linewidth=.25)
    axes[1].scatter(finite_min, y, s=9, color="white", edgecolor="#17324D", linewidth=.6)
    for yi, code in zip(y, order):
        if int(code) in zero_support:
            axes[1].text(-.96, yi, "N/A", ha="left", va="center", fontsize=7, color="#6E6E6E")
    axes[1].set_yticks(y, order, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("Within-class prototype cosine")
    axes[1].set_title("Per-class prototype cohesion")
    axes[1].grid(axis="x", alpha=.18)
    fig.suptitle("Phone A07 Proto Aux: learned identity structure", fontsize=15, color="#17324D")
    fig.tight_layout(rect=(0, 0, 1, .95))
    output = out / "Figure_10A_phone_A07_proto_aux_prototypes.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"figure": str(output), "checkpoint": str(CHECKPOINT), "config": str(CONFIG), "prototype_shape": list(prototypes.shape), "fdi_order": order.tolist(), "cohesion_mean": {str(c): (None if int(c) in zero_support else v) for c, v in zip(order, cohesion)}, "cohesion_min": {str(c): (None if int(c) in zero_support else v) for c, v in zip(order, cohesion_min)}, "training_support_by_fdi": support_by_fdi, "zero_support_fdi": zero_support, "font_requested": "Times New Roman", "font_used": actual_font}


def iou_matrix(gt_xywh: list[list[float]], pred_xyxy: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(gt_xywh), len(pred_xyxy)), dtype=np.float32)
    for i, (gx, gy, gw, gh) in enumerate(gt_xywh):
        for j, (px1, py1, px2, py2) in enumerate(pred_xyxy):
            x1, y1 = max(gx, px1), max(gy, py1)
            x2, y2 = min(gx + gw, px2), min(gy + gh, py2)
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            matrix[i, j] = inter / max(gw * gh + pa - inter, 1e-9)
    return matrix


def auxiliary_confusion(out: Path, device: str, batch_size: int = 1, workers: int = 0) -> dict:
    data = json.loads(ANN.read_text())
    images_gt: dict[int, list[dict]] = defaultdict(list)
    for row in data["annotations"]:
        images_gt[int(row["image_id"])].append(row)
    category_to_fdi = {int(row["id"]): int(row["name"]) for row in data["categories"]}
    cfg = YAMLConfig(str(CONFIG), **yaml_utils.parse_cli([f"val_dataloader.total_batch_size={int(batch_size)}", f"val_dataloader.num_workers={int(workers)}"]))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.to(device).eval()
    model.load_state_dict(checkpoint_state(CHECKPOINT), strict=True)
    postprocessor = cfg.postprocessor.to(device).eval()
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for key, module in [("quadrant", model.decoder.dec_quad_head[-1]), ("position", model.decoder.dec_toothpos_head[-1]), ("dentition", model.decoder.dec_dent_head[-1])]:
        handles.append(module.register_forward_hook(lambda _m, _i, value, name=key: captured.__setitem__(name, value.detach())))
    matrices = {"quadrant": np.zeros((8, 8), dtype=np.int64), "position": np.zeros((8, 8), dtype=np.int64), "dentition": np.zeros((2, 2), dtype=np.int64)}
    matched = 0
    coco_rows: list[dict] = []
    raw_rows: list[dict] = []
    with torch.inference_mode():
        for samples, targets in cfg.val_dataloader:
            outputs = model(samples.to(device))
            sizes = torch.stack([row["orig_size"] for row in targets]).to(device)
            for key in ("pred_logits", "pred_boxes", "query_feats", "pred_proto_logits"):
                if key in outputs and torch.is_tensor(outputs[key]):
                    outputs[key] = outputs[key].float()
            final_results = postprocessor(outputs, sizes)
            boxes = torchvision.ops.box_convert(outputs["pred_boxes"], "cxcywh", "xyxy") * sizes.repeat(1, 2).unsqueeze(1)
            confidence = outputs["pred_logits"].sigmoid().amax(dim=-1)
            for b, (target, result) in enumerate(zip(targets, final_results)):
                image_id = int(target["image_id"].item())
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
                    coco_rows.append({"image_id": image_id, "category_id": int(label.item()), "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], "score": float(score.item())})
                raw_prob = outputs["pred_logits"][b].sigmoid()
                raw_scores, raw_labels = raw_prob.max(dim=-1)
                for query_id, (box, score, label) in enumerate(zip(boxes[b], raw_scores, raw_labels)):
                    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
                    raw_rows.append({"image_id": image_id, "query_id": query_id, "category_id": int(label.item()), "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], "score": float(score.item())})
                gt = images_gt[int(target["image_id"].item())]
                keep = torch.nonzero(confidence[b] >= SCORE, as_tuple=False).flatten()
                if not len(gt) or keep.numel() == 0:
                    continue
                matrix = iou_matrix([row["bbox"] for row in gt], boxes[b, keep].cpu().numpy())
                rows, cols = linear_sum_assignment(1.0 - matrix)
                for gi, local in zip(rows, cols):
                    if matrix[gi, local] < IOU:
                        continue
                    q = int(keep[int(local)].item())
                    code = category_to_fdi[int(gt[int(gi)]["category_id"])]
                    truths = (code // 10 - 1, code % 10 - 1, 0 if code >= 50 else 1)
                    for key, truth in zip(("quadrant", "position", "dentition"), truths):
                        pred = int(captured[key][b, q].argmax().item())
                        matrices[key][truth, pred] += 1
                    matched += 1
    for handle in handles:
        handle.remove()
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "coco_predictions.json").write_text(json.dumps(coco_rows), encoding="utf-8")
    (prediction_dir / "raw_query_predictions.json").write_text(json.dumps(raw_rows), encoding="utf-8")
    actual_font = configure_publication_font()
    labels = {"quadrant": [str(x) for x in range(1, 9)], "position": [str(x) for x in range(1, 9)], "dentition": ["Primary", "Permanent"]}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1, 1, .75]})
    accuracies = {}
    for ax, key in zip(axes, ("quadrant", "position", "dentition")):
        counts = matrices[key]; norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        names = labels[key]; ax.set_xticks(np.arange(len(names)), names); ax.set_yticks(np.arange(len(names)), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        accuracies[key] = float(np.trace(counts) / max(counts.sum(), 1)); ax.set_title(f"{key.capitalize()} ({accuracies[key]*100:.1f}%)")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                if counts[i, j]: ax.text(j, i, f"{norm[i,j]*100:.0f}%", ha="center", va="center", fontsize=6, color="white" if norm[i,j]>.55 else "#17324D")
    fig.suptitle("Phone A07 Proto Aux: auxiliary factor heads", fontsize=15, color="#17324D")
    # Keep protocol details in the caption/provenance rather than as tiny text
    # inside the figure; this is cleaner for the manuscript two-column layout.
    fig.tight_layout(rect=(0, 0, 1, .94))
    output = out / "Figure_10B_phone_A07_auxiliary_heads.png"; fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return {"figure": str(output), "checkpoint": str(CHECKPOINT), "config": str(CONFIG), "matched_teeth": matched, "accuracy": accuracies, "counts": {k: v.tolist() for k, v in matrices.items()}, "score_threshold": SCORE, "matching_iou": IOU, "prediction_files": {"coco": str(prediction_dir / "coco_predictions.json"), "raw_queries": str(prediction_dir / "raw_query_predictions.json")}, "font_requested": "Times New Roman", "font_used": actual_font}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=PAPER / "materials/protoaux_revision_assets_2026-09-18"); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"prototype": prototype_panel(args.output_dir), "auxiliary_heads": auxiliary_confusion(args.output_dir, args.device, args.batch_size, args.workers), "inference": {"batch_size": int(args.batch_size), "workers": int(args.workers), "amp": False}}
    (args.output_dir / "protoaux_evidence_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
