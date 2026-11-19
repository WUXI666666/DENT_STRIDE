#!/usr/bin/env python3
"""Export any contiguous zero-based COCO detection dataset to YOLO format.

Images remain in their source directories and are exposed through symlinks.
Horizontal/vertical augmentation policy is deliberately left to the trainer;
the paper-facing runners disable both because FDI labels encode anatomy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    missing = {"images", "annotations", "categories"}.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing COCO keys: {sorted(missing)}")
    return payload


def categories(train: dict[str, Any], val: dict[str, Any]) -> list[dict[str, Any]]:
    left = sorted(train["categories"], key=lambda row: int(row["id"]))
    right = sorted(val["categories"], key=lambda row: int(row["id"]))
    signature = lambda rows: [(int(row["id"]), str(row["name"])) for row in rows]
    if signature(left) != signature(right):
        raise ValueError("Training and validation category definitions differ")
    ids = [int(row["id"]) for row in left]
    if ids != list(range(len(ids))):
        raise ValueError(f"Category IDs must be contiguous and zero-based, found {ids}")
    return left


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"Existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def export_split(payload: dict[str, Any], image_dir: Path, label_dir: Path) -> dict[str, int]:
    images = {int(row["id"]): row for row in payload["images"]}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["annotations"]:
        grouped[int(row["image_id"])].append(row)
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_id, image in sorted(images.items()):
        name = Path(str(image["file_name"]))
        if not (image_dir / name).is_file():
            raise FileNotFoundError(image_dir / name)
        width, height = float(image["width"]), float(image["height"])
        rows: list[str] = []
        for annotation in grouped.get(image_id, []):
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x2 = min(width, max(0.0, x + box_width))
            y2 = min(height, max(0.0, y + box_height))
            x1 = min(width, max(0.0, x))
            y1 = min(height, max(0.0, y))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Empty bbox in annotation {annotation.get('id')}")
            rows.append(
                f"{int(annotation['category_id'])} "
                f"{((x1+x2)/2/width):.8f} {((y1+y2)/2/height):.8f} "
                f"{((x2-x1)/width):.8f} {((y2-y1)/height):.8f}"
            )
            count += 1
        target = label_dir / name.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows) + ("\n" if rows else ""))
    return {"images": len(images), "annotations": count}


def write_list(payload: dict[str, Any], path: Path, split: str) -> None:
    rows = [f"./{split}/{Path(str(row['file_name'])).as_posix()}" for row in sorted(payload["images"], key=lambda row: int(row["id"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yaml-name", default="dataset.yaml")
    parser.add_argument("--label-space", default="contiguous zero-based FDI labels")
    args = parser.parse_args()

    train, val = load(args.train_ann), load(args.val_ann)
    cats = categories(train, val)
    output = args.output_dir.resolve()
    ensure_link(output / "images/train", args.train_images)
    ensure_link(output / "images/val", args.val_images)
    train_stats = export_split(train, args.train_images.resolve(), output / "labels/train")
    val_stats = export_split(val, args.val_images.resolve(), output / "labels/val")
    write_list(train, output / "images/train.txt", "train")
    write_list(val, output / "images/val.txt", "val")
    names = {int(row["id"]): str(row["name"]) for row in cats}
    yaml_path = output / args.yaml_name
    yaml_path.write_text(yaml.safe_dump({
        "path": str(output), "train": "images/train.txt", "val": "images/val.txt",
        "nc": len(cats), "names": names,
    }, sort_keys=False, allow_unicode=True))
    manifest = {
        "format": "YOLO detect", "label_space": args.label_space,
        "train_annotation_source": str(args.train_ann.resolve()),
        "val_annotation_source": str(args.val_ann.resolve()),
        "train_images_source": str(args.train_images.resolve()),
        "val_images_source": str(args.val_images.resolve()),
        "yaml": str(yaml_path), "classes": names,
        "train": train_stats, "val": val_stats,
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
