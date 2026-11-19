#!/usr/bin/env python3
"""Prepare View-Net3 DSLR tooth annotations for Dent-FINE detection.

The source annotations use FDI codes (for example, 11 and 85) as COCO
``category_id`` values.  Dent-FINE expects contiguous model labels.  This
tool leaves the source dataset untouched and writes a derived COCO annotation
pair in the canonical 52-class FDI order used by the existing phone models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FDI52_ORDER = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
    51, 52, 53, 54, 55,
    61, 62, 63, 64, 65,
    71, 72, 73, 74, 75,
    81, 82, 83, 84, 85,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(dataset: dict[str, Any], split: str) -> None:
    required = {"images", "annotations", "categories"}
    missing = required.difference(dataset)
    if missing:
        raise ValueError(f"{split}: source JSON is missing keys: {sorted(missing)}")

    image_ids = [int(item["id"]) for item in dataset["images"]]
    annotation_ids = [int(item["id"]) for item in dataset["annotations"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{split}: duplicate image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(f"{split}: duplicate annotation IDs")

    source_categories = {int(item["id"]) for item in dataset["categories"]}
    unsupported = source_categories.difference(FDI52_ORDER)
    if unsupported:
        raise ValueError(f"{split}: unsupported FDI category IDs: {sorted(unsupported)}")

    image_set = set(image_ids)
    for annotation in dataset["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in source_categories:
            raise ValueError(f"{split}: annotation has unknown category ID {category_id}")
        if int(annotation["image_id"]) not in image_set:
            raise ValueError(f"{split}: annotation references an unknown image")
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"{split}: invalid bbox in annotation {annotation.get('id')}")


def convert(source: dict[str, Any], fdi_to_label: dict[int, int]) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    for annotation in source["annotations"]:
        item = dict(annotation)
        original_fdi = int(item["category_id"])
        item["category_id"] = fdi_to_label[original_fdi]
        item["original_fdi"] = original_fdi
        annotations.append(item)

    return {
        "info": {
            "description": "View-Net3 DSLR tooth detection annotations remapped for Dent-FINE",
            "label_space": "canonical FDI52 order, contiguous 0-based model labels",
        },
        "licenses": source.get("licenses", []),
        "images": source["images"],
        "annotations": annotations,
        "categories": [
            {
                "id": label,
                "name": str(fdi),
                "supercategory": "tooth",
                "fdi": fdi,
            }
            for label, fdi in enumerate(FDI52_ORDER)
        ],
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52"),
    )
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    source_annotations = args.source_root / "annotations"
    output_annotations = args.output_root / "annotations"
    output_annotations.mkdir(parents=True, exist_ok=True)
    fdi_to_label = {fdi: label for label, fdi in enumerate(FDI52_ORDER)}

    manifest: dict[str, Any] = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "label_space": "FDI52 contiguous 0-based labels",
        "fdi_label_order": FDI52_ORDER,
        "splits": {},
    }

    for split in ("train", "val"):
        source_path = source_annotations / f"tooth_instances_{split}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source annotation: {source_path}")
        source = json.loads(source_path.read_text())
        validate_source(source, split)

        if args.check_images:
            image_root = args.source_root / "images" / split
            missing_images = [
                image["file_name"]
                for image in source["images"]
                if not (image_root / image["file_name"]).is_file()
            ]
            if missing_images:
                preview = ", ".join(missing_images[:5])
                raise FileNotFoundError(f"{split}: missing {len(missing_images)} images, e.g. {preview}")

        converted = convert(source, fdi_to_label)
        output_path = output_annotations / f"tooth_instances_{split}_fdi52.json"
        write_json(output_path, converted)
        present_fdi = sorted({int(item["category_id"]) for item in source["annotations"]})
        manifest["splits"][split] = {
            "source_annotation": str(source_path),
            "source_sha256": sha256(source_path),
            "converted_annotation": str(output_path),
            "images": len(source["images"]),
            "annotations": len(source["annotations"]),
            "present_fdi": present_fdi,
            "present_model_labels": [fdi_to_label[fdi] for fdi in present_fdi],
        }
        print(f"Prepared {split}: {len(source['images'])} images, {len(source['annotations'])} annotations -> {output_path}")

    write_json(args.output_root / "conversion_manifest.json", manifest)
    print(f"Wrote manifest: {args.output_root / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
