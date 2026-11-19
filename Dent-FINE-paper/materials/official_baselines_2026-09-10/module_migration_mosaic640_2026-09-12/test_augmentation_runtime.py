#!/usr/bin/env python3
"""CPU-only native-equivalence checks for the new experiment's metadata fix.

No output artifacts, checkpoints, original configs or datasets are modified.
The final JSON line is suitable for the parent's preflight evidence log.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "upstream/DEIM"))

from engine.core import GLOBAL_CONFIG  # noqa: E402
from engine.data.dataset.coco_dataset import CocoDetection  # noqa: E402
from engine.data.dataloader import BatchImageCollateFunction  # noqa: E402
from engine.data.transforms import Compose  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.transforms.v2 import SanitizeBoundingBoxes  # noqa: E402
from engine.data.transforms._transforms import RandomIoUCrop  # noqa: E402
from augmentation_runtime import (AlignedSanitizeBoundingBoxes, AlignedBatchImageCollateFunction,
                                  install_augmentation_runtime)  # noqa: E402


def seed(value):
    random.seed(value)
    torch.manual_seed(value)


def rng_snapshot():
    return random.getstate(), torch.get_rng_state().clone()


def same_rng(left, right):
    assert left[0] == right[0], "Python RNG changed relative to native augmentation"
    assert torch.equal(left[1], right[1]), "Torch RNG changed relative to native augmentation"


def expected_pixel_area(boxes, side=640):
    # Final train boxes are normalized cxcywh.  Both source configurations use
    # the native ConvertBoxes and the fixed-size 640 collator.
    return boxes[:, 2] * boxes[:, 3] * side * side


def compare_sample(native, aligned):
    assert torch.equal(native[0], aligned[0]), "Images differ from native pipeline"
    left, right = native[1], aligned[1]
    for key in left:
        if key not in ("area", "iscrowd"):
            assert torch.equal(left[key], right[key]), f"Unexpected target change: {key}"
    count = len(right["labels"])
    for key in ("boxes", "area", "iscrowd"):
        assert len(right[key]) == count, (key, len(right[key]), count)
    assert tuple(aligned[0].shape) == (3, 640, 640)
    assert torch.isfinite(right["boxes"]).all()
    assert torch.isfinite(right["area"]).all()
    assert (right["area"] >= 0).all()
    assert torch.allclose(right["area"], expected_pixel_area(right["boxes"]), rtol=3e-5, atol=0.01)


def training_dataset(task, aligned):
    path = ROOT / "module_migration_2026-09-11/configs" / f"{task}.yml"
    config = yaml.safe_load(path.read_text())
    dataset = copy.deepcopy(config["train_dataloader"]["dataset"])
    transforms = dataset.pop("transforms")
    dataset.pop("type")
    transforms.pop("type")
    for op in transforms["ops"]:
        if op["type"] == "Mosaic":
            op.update(output_size=640, rotation_range=10, translation_range=[0.1, 0.1],
                      scaling_range=[0.5, 1.5], probability=1.0, fill_value=0,
                      use_cache=False, max_cached_images=50, random_pop=True)
        elif op["type"] == "RandomIoUCrop":
            op["p"] = 0.5
        elif op["type"] == "SanitizeBoundingBoxes" and aligned:
            op["type"] = "AlignedSanitizeBoundingBoxes"
    transforms["mosaic_prob"] = 1.0
    return CocoDetection(transforms=Compose(**transforms), **dataset), config


def synthetic_sanitizer_test():
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            [[1., 1., 5., 6.], [4., 4., 4., 8.], [2., 3., 6., 8.]],
            format="XYXY", canvas_size=(16, 16)),
        "labels": torch.tensor([3, 4, 5]),
        "area": torch.tensor([999, 999, 999]),
        "iscrowd": torch.tensor([1, 0, 0]),
        "image_id": torch.tensor([123]),
    }
    native = SanitizeBoundingBoxes()(copy.deepcopy(target))
    aligned = AlignedSanitizeBoundingBoxes()(copy.deepcopy(target))
    assert torch.equal(native["boxes"], aligned["boxes"])
    assert torch.equal(native["labels"], aligned["labels"])
    assert aligned["iscrowd"].tolist() == [1, 0], "Real crowd flags must be preserved"
    assert aligned["area"].tolist() == [20., 20.]
    without_crowd = copy.deepcopy(target)
    del without_crowd["iscrowd"]
    without_crowd = AlignedSanitizeBoundingBoxes()(without_crowd)
    assert "iscrowd" not in without_crowd, "Do not synthesize absent crowd flags"


def crop_removal_test():
    """Exercise a real crop that removes a box, not just a skipped crop call."""
    image = tv_tensors.Image(torch.arange(3 * 64 * 64).reshape(3, 64, 64).float())
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            [[2., 2., 12., 12.], [50., 2., 62., 12.],
             [2., 50., 12., 62.], [50., 50., 62., 62.]],
            format="XYXY", canvas_size=(64, 64)),
        "labels": torch.tensor([0, 1, 2, 3]),
        "area": torch.tensor([100., 120., 120., 144.]),
        "iscrowd": torch.tensor([1, 0, 1, 0]),
    }
    crop = RandomIoUCrop(p=0.5)
    for value in range(100):
        seed(value)
        cropped = crop(image.clone(), copy.deepcopy(target))
        native = SanitizeBoundingBoxes()(*copy.deepcopy(cropped))
        if 0 < len(native[1]["labels"]) < 4:
            before = rng_snapshot()
            aligned = AlignedSanitizeBoundingBoxes()(*copy.deepcopy(cropped))
            same_rng(before, rng_snapshot())
            assert torch.equal(native[0], aligned[0])
            assert torch.equal(native[1]["boxes"], aligned[1]["boxes"])
            assert torch.equal(native[1]["labels"], aligned[1]["labels"])
            labels = aligned[1]["labels"]
            assert torch.equal(aligned[1]["iscrowd"], target["iscrowd"][labels])
            assert len(aligned[1]["area"]) == len(labels)
            return {"seed": value, "remaining_labels": labels.tolist(),
                    "cropped_image_shape": list(cropped[0].shape)}
    raise AssertionError("No actual crop removal was exercised")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    native_entry = dict(GLOBAL_CONFIG["SanitizeBoundingBoxes"])
    installation = install_augmentation_runtime()
    assert installation == install_augmentation_runtime(), "Installation must be idempotent"
    assert native_entry == GLOBAL_CONFIG["SanitizeBoundingBoxes"]
    synthetic_sanitizer_test()
    report = {"passed": True, "device": "cpu", "installation": installation,
              "explicit_crop_removal": crop_removal_test(), "cohorts": {}}
    for task in ("phone_dualopt", "dentex_spatial_shape", "dslr_cmfa_dualopt"):
        native, config = training_dataset(task, False)
        aligned, _ = training_dataset(task, True)
        cohort = {"stages": {}, "native_metadata_mismatch_observed": False}
        for epoch in (4, 70, 130):
            native.set_epoch(epoch)
            aligned.set_epoch(epoch)
            native_batch, aligned_batch = [], []
            counts = []
            for index in range(args.samples):
                value = 123 + index
                seed(value)
                left = native[index]
                after_left = rng_snapshot()
                seed(value)
                right = aligned[index]
                after_right = rng_snapshot()
                same_rng(after_left, after_right)
                compare_sample(left, right)
                native_batch.append(left)
                aligned_batch.append(right)
                counts.append(len(right[1]["labels"]))
                if len(left[1]["area"]) != len(left[1]["labels"]):
                    cohort["native_metadata_mismatch_observed"] = True
                source_count = right[1]["image_id"].numel()
                assert source_count == (4 if epoch == 4 else 1)
            # Probability1 forces the critical compound Mosaic+MixUp path at
            # epoch4; the same native MixUp is used on both branches.
            kwargs = dict(base_size=640, base_size_repeat=None, stop_epoch=130,
                          mixup_prob=1.0, mixup_epochs=[4, 80])
            native_collate = BatchImageCollateFunction(**kwargs)
            aligned_collate = AlignedBatchImageCollateFunction(**kwargs)
            native_collate.set_epoch(epoch)
            aligned_collate.set_epoch(epoch)
            seed(891)
            batched_left = native_collate(native_batch)
            after_left = rng_snapshot()
            seed(891)
            batched_right = aligned_collate(aligned_batch)
            after_right = rng_snapshot()
            same_rng(after_left, after_right)
            for image_l, target_l, image_r, target_r in zip(
                    batched_left[0], batched_left[1], batched_right[0], batched_right[1]):
                compare_sample((image_l, target_l), (image_r, target_r))
            cohort["stages"][str(epoch)] = {"sample_box_counts": counts,
                "collated_box_counts": [len(t["labels"]) for t in batched_right[1]],
                "batch_shape": list(batched_right[0].shape),
                "native_image_box_label_rng_equivalence": True}
        val = copy.deepcopy(config["val_dataloader"]["dataset"])
        transforms = val.pop("transforms")
        transforms.pop("type")
        val.pop("type")
        assert all(op["type"] != "AlignedSanitizeBoundingBoxes" for op in transforms["ops"])
        val_dataset = CocoDetection(transforms=Compose(**transforms), **val)
        _, val_target = val_dataset[0]
        assert val_target["image_id"].numel() == 1
        assert val_target["orig_size"].numel() == 2
        # Evaluation GT areas and original-size metadata must not be rewritten
        # to resized pixel area by the training-only alignment transform.
        _, untransformed_val_target = val_dataset.load_item(0)
        for key in ("area", "iscrowd", "orig_size", "image_id"):
            assert torch.equal(val_target[key], untransformed_val_target[key])
        cohort["validation_unmodified"] = True
        report["cohorts"][task] = cohort
    print("AUGMENTATION_RUNTIME_TEST_JSON=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
