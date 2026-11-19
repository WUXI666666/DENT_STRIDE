#!/usr/bin/env python3
"""Build DSLR per-view center/log-shape templates from training annotations only.

This is a standalone addition: neither annotations nor the original project are
changed. Validation annotations are read exclusively for metadata/split checks.
The score matches the existing SpatialShape implementation: regularized full
2-D covariance, Mahalanobis penalty, relative clipping, addition before top-k.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

VIEW_TO_ID = {
    "frontal": 0,
    "upper_occlusal": 1,
    "lower_occlusal": 2,
    "left_buccal": 3,
    "right_buccal": 4,
}
VIEW_NAMES = [name for name, _ in sorted(VIEW_TO_ID.items(), key=lambda pair: pair[1])]
FDI52 = [q * 10 + p for q in range(1, 5) for p in range(1, 9)] + [
    q * 10 + p for q in range(5, 9) for p in range(1, 6)
]
DEFAULT_ANNOTATIONS = Path("/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_checked(path: Path):
    data = json.loads(path.read_text())
    cats = sorted(data["categories"], key=lambda row: int(row["id"]))
    assert [int(row["id"]) for row in cats] == list(range(52)), "Expected contiguous FDI52 IDs"
    assert [int(row["name"]) for row in cats] == FDI52, "FDI map must exactly match current DSLR"
    images = {int(row["id"]): row for row in data["images"]}
    assert len(images) == len(data["images"]), "Duplicate image IDs"
    assert len({int(row["id"]) for row in data["annotations"]}) == len(data["annotations"])
    patient_views = defaultdict(list)
    for image in images.values():
        view, view_id = image.get("view"), int(image.get("view_id", -1))
        assert view in VIEW_TO_ID and VIEW_TO_ID[view] == view_id, f"Unknown/conflicting view metadata: {image}"
        assert image.get("patient_id"), f"Missing patient ID: {image}"
        if "patient_key" in image:
            assert image["patient_key"] == image["patient_id"]
        assert float(image["width"]) > 0 and float(image["height"]) > 0
        patient_views[image["patient_id"]].append(view)
    invalid_boxes, outside_boxes = [], []
    box_counts = Counter()
    for ann in data["annotations"]:
        assert int(ann["image_id"]) in images
        cid = int(ann["category_id"])
        assert 0 <= cid < 52
        if "original_fdi" in ann:
            assert int(ann["original_fdi"]) == FDI52[cid]
        image = images[int(ann["image_id"])]
        bbox = torch.tensor(ann["bbox"], dtype=torch.float64)
        if not torch.isfinite(bbox).all() or len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            invalid_boxes.append(int(ann["id"]))
        elif float(bbox[0]) < 0 or float(bbox[1]) < 0 or float(bbox[0] + bbox[2]) > float(image["width"]) + 1 or float(bbox[1] + bbox[3]) > float(image["height"]) + 1:
            outside_boxes.append(int(ann["id"]))
        box_counts[image["view"]] += 1
    assert not invalid_boxes, f"Invalid boxes {invalid_boxes[:10]}"
    check = {
        "annotation_file": str(path.resolve()), "annotation_sha256": sha256(path),
        "image_count": len(images), "box_count": len(data["annotations"]),
        "patient_count": len(patient_views),
        "images_by_view": dict(Counter(image["view"] for image in images.values())),
        "boxes_by_view": dict(box_counts),
        "patients_with_incomplete_or_duplicate_views": {
            patient: views for patient, views in patient_views.items() if sorted(views) != sorted(VIEW_NAMES)
        },
        "invalid_boxes": invalid_boxes, "outside_image_boxes": outside_boxes,
        "present_fdi": sorted({FDI52[int(ann["category_id"])] for ann in data["annotations"]}),
    }
    return data, images, check


def moments(points, ridge: float):
    """Unbiased covariance + ridge I, as in existing build_tooth_templates.py."""
    if not points:
        return torch.zeros(2), torch.eye(2) * ridge, torch.eye(2) / ridge, False
    x = torch.as_tensor(points, dtype=torch.float64)
    mu = x.mean(0)
    delta = x - mu
    covariance = delta.T @ delta / max(len(x) - 1, 1)
    covariance = (covariance + covariance.T) * 0.5
    # Remove only numerical negative eigenvalues; do not shrink genuine covariance.
    values, vectors = torch.linalg.eigh(covariance)
    covariance = (vectors * values.clamp_min(0.0).unsqueeze(0)) @ vectors.T
    covariance = covariance + torch.eye(2, dtype=x.dtype) * ridge
    inverse = torch.linalg.inv(covariance)
    assert torch.linalg.eigvalsh(covariance).min() >= ridge * (1.0 - 1e-8)
    return mu.float(), covariance.float(), inverse.float(), True


def build(train_path: Path, val_path: Path, output_dir: Path, ridge: float = 1e-3):
    assert ridge > 0
    train, images, train_check = read_checked(train_path)
    val, val_images, val_check = read_checked(val_path)
    train_patients = {row["patient_id"] for row in images.values()}
    val_patients = {row["patient_id"] for row in val_images.values()}
    overlap = sorted(train_patients & val_patients)
    filenames = sorted({row["file_name"] for row in images.values()} & {row["file_name"] for row in val_images.values()})
    assert not overlap, f"Patient identifier overlap: {overlap}"
    assert not filenames, f"Filename overlap: {filenames}"
    assert not train_check["patients_with_incomplete_or_duplicate_views"]
    assert not val_check["patients_with_incomplete_or_duplicate_views"]
    centers, shapes = defaultdict(list), defaultdict(list)
    for ann in train["annotations"]:
        image = images[int(ann["image_id"])]
        key = (int(image["view_id"]), int(ann["category_id"]))
        x, y, w, h = map(float, ann["bbox"])
        width, height = float(image["width"]), float(image["height"])
        centers[key].append([(x + 0.5 * w) / width, (y + 0.5 * h) / height])
        shapes[key].append(torch.tensor([max(w / width, 1e-5), max(h / height, 1e-5)]).double().log().tolist())
    prior = {
        "schema_version": 1,
        "num_views": 5, "num_classes": 52,
        "view_names": VIEW_NAMES, "view_ids": list(range(5)),
        "fdi_label_order": torch.tensor(FDI52),
        "annotation_file": str(train_path.resolve()),
        "annotation_sha256": train_check["annotation_sha256"],
        "covariance_ridge": float(ridge),
        "shape_representation": "log(normalized_width), log(normalized_height)",
        "unsupported_class_policy": "zero score, no evidence, no hard visibility penalty",
    }
    counts = torch.zeros(5, 52, dtype=torch.long)
    for prefix, groups in [("spatial", centers), ("shape", shapes)]:
        means, covariances, inverse_covariances, valid_flags = [], [], [], []
        for vid in range(5):
            rows = [moments(groups[(vid, cid)], ridge) for cid in range(52)]
            means.append(torch.stack([row[0] for row in rows]))
            covariances.append(torch.stack([row[1] for row in rows]))
            inverse_covariances.append(torch.stack([row[2] for row in rows]))
            valid_flags.append(torch.tensor([row[3] for row in rows]))
            counts[vid] = torch.tensor([len(groups[(vid, cid)]) for cid in range(52)])
        prior[prefix + "_mu"] = torch.stack(means)
        prior[prefix + "_cov"] = torch.stack(covariances)
        prior[prefix + "_inv_cov"] = torch.stack(inverse_covariances)
        prior[prefix + "_valid"] = torch.stack(valid_flags)
    prior["counts"] = counts
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = output_dir / "view_specific_tooth_priors.pt"
    torch.save(prior, prior_path)
    # Individual view files are also compatible with the original PostProcessor.
    for vid, name in enumerate(VIEW_NAMES):
        single = {key: (value[vid] if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == (5, 52) else value)
                  for key, value in prior.items()}
        single.update(view_id=vid, view_name=name)
        torch.save(single, output_dir / f"tooth_priors_{name}.pt")
    serializable = {key: (value.tolist() if torch.is_tensor(value) else value) for key, value in prior.items()}
    (output_dir / "view_specific_tooth_priors.json").write_text(json.dumps(serializable, indent=2) + "\n")
    manifest = {
        "view_to_id": VIEW_TO_ID,
        "fdi_label_order": FDI52,
        "train": {str(iid): {"view_id": int(row["view_id"]), "view": row["view"], "patient_id": row["patient_id"], "file_name": row["file_name"]} for iid, row in images.items()},
        "val": {str(iid): {"view_id": int(row["view_id"]), "view": row["view"], "patient_id": row["patient_id"], "file_name": row["file_name"]} for iid, row in val_images.items()},
    }
    (output_dir / "image_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = {
        "train": train_check, "val": val_check, "view_to_id": VIEW_TO_ID,
        "patient_identifier_overlap": overlap, "filename_overlap": filenames,
        "content_duplicate_audit": "Not repeated by this helper; identifier and filename checks are not content hashing.",
        "prior_source": "training annotations only; validation coordinates/classes never used for moments",
        "view_class_counts": {name: {str(FDI52[cid]): int(counts[vid, cid]) for cid in range(52)} for vid, name in enumerate(VIEW_NAMES)},
        "supported_classes_by_view": {name: int((counts[vid] > 0).sum()) for vid, name in enumerate(VIEW_NAMES)},
        "sparse_classes_lt5_by_view": {name: [FDI52[cid] for cid in range(52) if 0 < int(counts[vid, cid]) < 5] for vid, name in enumerate(VIEW_NAMES)},
        "prior_file": str(prior_path), "prior_sha256": sha256(prior_path),
    }
    (output_dir / "data_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, default=DEFAULT_ANNOTATIONS / "tooth_instances_train_fdi52.json")
    parser.add_argument("--val-ann", type=Path, default=DEFAULT_ANNOTATIONS / "tooth_instances_val_fdi52.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "priors")
    parser.add_argument("--min-var", type=float, default=1e-3)
    args = parser.parse_args()
    checks = build(args.train_ann, args.val_ann, args.output_dir, args.min_var)
    print(json.dumps({key: checks[key] for key in ["supported_classes_by_view", "sparse_classes_lt5_by_view", "prior_file", "prior_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
