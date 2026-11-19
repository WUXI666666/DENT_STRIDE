#!/usr/bin/env python3
"""Export the canonical View-Net3 DSLR FDI52 COCO annotations to YOLO detect.

The source images are never copied: ``images/{train,val}`` in the output tree
are symlinks to the original View-Net3 image directories.  Labels retain the
canonical zero-based FDI52 indices used by the DEIM experiments, including the
ten classes that have no instances in this cohort.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    required = {"images", "annotations", "categories"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} is not a COCO detection file; missing {sorted(missing)}")
    return data


def validate_categories(train: dict[str, Any], val: dict[str, Any]) -> list[dict[str, Any]]:
    categories = sorted(train["categories"], key=lambda row: int(row["id"]))
    expected = list(range(len(categories)))
    ids = [int(row["id"]) for row in categories]
    if ids != expected:
        raise ValueError(f"Training category IDs must be contiguous zero-based; received {ids}")
    val_categories = sorted(val["categories"], key=lambda row: int(row["id"]))
    if [(int(row["id"]), str(row["name"])) for row in categories] != [
        (int(row["id"]), str(row["name"])) for row in val_categories
    ]:
        raise ValueError("Training and validation category definitions differ")
    if len(categories) != 52:
        raise ValueError(f"Expected canonical FDI52 labels, found {len(categories)} categories")
    return categories


def ensure_image_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {target}")
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise FileExistsError(f"Refusing to replace non-symlink directory: {link}")
        if link.resolve() != target:
            raise RuntimeError(f"Existing image symlink points elsewhere: {link} -> {link.resolve()}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def export_split(
    data: dict[str, Any],
    source_images: Path,
    labels_dir: Path,
    num_classes: int,
) -> dict[str, int]:
    source_images = source_images.resolve()
    if not source_images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source_images}")
    images = {int(row["id"]): row for row in data["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"Annotation {annotation.get('id')} references unknown image {image_id}")
        annotations[image_id].append(annotation)

    labels_dir.mkdir(parents=True, exist_ok=True)
    label_count = 0
    for image_id, image in sorted(images.items()):
        file_name = Path(str(image["file_name"]))
        image_path = source_images / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image file is absent: {image_path}")
        width = float(image["width"])
        height = float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid size for image {image_id}: {(width, height)}")

        rows: list[str] = []
        for annotation in annotations.get(image_id, []):
            category_id = int(annotation["category_id"])
            if not 0 <= category_id < num_classes:
                raise ValueError(f"Annotation {annotation.get('id')} has invalid FDI52 class {category_id}")
            x, y, box_w, box_h = (float(value) for value in annotation["bbox"])
            if box_w <= 0 or box_h <= 0:
                raise ValueError(f"Annotation {annotation.get('id')} has non-positive bbox")
            # The verified View-Net3 boxes lie in bounds. Clamp only for
            # round-off robustness without silently accepting invalid boxes.
            x2, y2 = x + box_w, y + box_h
            if x < -1e-3 or y < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3:
                raise ValueError(f"Annotation {annotation.get('id')} bbox lies outside image {image_id}")
            x = min(max(x, 0.0), width)
            y = min(max(y, 0.0), height)
            x2 = min(max(x2, 0.0), width)
            y2 = min(max(y2, 0.0), height)
            clipped_w, clipped_h = x2 - x, y2 - y
            if clipped_w <= 0 or clipped_h <= 0:
                raise ValueError(f"Annotation {annotation.get('id')} becomes empty after clipping")
            cx = (x + x2) * 0.5 / width
            cy = (y + y2) * 0.5 / height
            bw = clipped_w / width
            bh = clipped_h / height
            rows.append(f"{category_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            label_count += 1

        label_path = labels_dir / file_name.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return {"images": len(images), "annotations": label_count}


def write_image_list(data: dict[str, Any], path: Path, split: str) -> None:
    """Write paths through the output symlink without resolving that symlink.

    Ultralytics resolves a directory given directly in the dataset YAML, which
    would turn the output ``images`` symlink back into the original source tree
    and make it look for labels there. A text image list preserves the output
    pathname, so the normal ``images -> labels`` mapping points to this export.
    """
    rows = [f"./{split}/{Path(str(image['file_name'])).as_posix()}" for image in sorted(data["images"], key=lambda x: int(x["id"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def main() -> None:
    args = parse_args()
    train = load_coco(args.train_ann)
    val = load_coco(args.val_ann)
    categories = validate_categories(train, val)
    output_dir = args.output_dir.resolve()
    ensure_image_link(output_dir / "images" / "train", args.train_images)
    ensure_image_link(output_dir / "images" / "val", args.val_images)

    train_stats = export_split(train, args.train_images, output_dir / "labels" / "train", len(categories))
    val_stats = export_split(val, args.val_images, output_dir / "labels" / "val", len(categories))
    write_image_list(train, output_dir / "images" / "train.txt", "train")
    write_image_list(val, output_dir / "images" / "val.txt", "val")

    names = {int(category["id"]): str(category["name"]) for category in categories}
    data_yaml = {
        "path": str(output_dir),
        "train": "images/train.txt",
        "val": "images/val.txt",
        "nc": len(categories),
        "names": names,
    }
    with (output_dir / "viewnet3_dslr_fdi52.yaml").open("w") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False, allow_unicode=True)

    manifest = {
        "format": "YOLO detect",
        "label_space": "canonical FDI52, zero-based class IDs",
        "train_annotation_source": str(args.train_ann.resolve()),
        "val_annotation_source": str(args.val_ann.resolve()),
        "train_images_source": str(args.train_images.resolve()),
        "val_images_source": str(args.val_images.resolve()),
        "output_dir": str(output_dir),
        "image_list_mode": "text lists preserve labels inside this output directory while images remain symlinked",
        "classes": names,
        "train": train_stats,
        "val": val_stats,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
