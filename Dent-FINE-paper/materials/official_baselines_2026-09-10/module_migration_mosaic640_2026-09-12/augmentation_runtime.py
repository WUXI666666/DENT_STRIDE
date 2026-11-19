"""Process-local target metadata alignment for the Mosaic-640 experiment.

Call ``install_augmentation_runtime()`` after importing the frozen official
``engine`` package and before constructing YAMLConfig.  In this experiment's
training YAML, replace both ``SanitizeBoundingBoxes`` type names with
``AlignedSanitizeBoundingBoxes`` and the train collator type with
``AlignedBatchImageCollateFunction``.  Validation remains unchanged.

The frozen official Mosaic implementation has no internal sanitizer: it applies
Resize, assembles four sources and applies RandomAffine.  Thus replacing the two
outer sanitizers is sufficient; the Mosaic class and its RNG are not patched.
Its concatenated image_id/idx/orig_size values are deliberately retained as
native training-only source metadata, not fabricated into a clinical image ID.

This changes neither native box filtering nor image/label tensors.  It applies
the same native keep mask to existing COCO area/iscrowd tensors and recomputes
existing area from the resulting boxes.  Missing crowd flags are not invented.
"""
from __future__ import annotations

from collections.abc import Mapping
import sys
from typing import Any

import torch
import torchvision.transforms.v2 as T
from torchvision.ops import box_area, box_convert
from engine.data.dataloader import BatchImageCollateFunction


def _target(inputs: Any) -> Mapping:
    if isinstance(inputs, Mapping):
        return inputs
    if isinstance(inputs, (tuple, list)) and len(inputs) > 1 and isinstance(inputs[1], Mapping):
        return inputs[1]
    raise TypeError("Expected a detection target dict or (image, target[, dataset]).")


class AlignedSanitizeBoundingBoxes(T.SanitizeBoundingBoxes):
    """Native box/label sanitization with synchronized existing COCO metadata."""

    def __init__(self, min_size: float = 1.0, min_area: float = 1.0):
        super().__init__(min_size=min_size, min_area=min_area, labels_getter=self._labels)

    @staticmethod
    def _labels(inputs):
        target = _target(inputs)
        # Let the native sanitizer validate tensor types and matching lengths.
        # Every requested field is returned by identity, so its native keep mask
        # is applied to exactly those objects, including nonzero crowd flags.
        return tuple(target[key] for key in ("labels", "area", "iscrowd") if key in target)

    def forward(self, *inputs):
        output = super().forward(*inputs)
        target = _target(output)
        if "area" in target:
            boxes = target["boxes"]
            fmt = boxes.format.value.lower()
            coordinates = boxes.as_subclass(torch.Tensor)
            xyxy = coordinates if fmt == "xyxy" else box_convert(coordinates, fmt, "xyxy")
            # Keep the floating box dtype rather than truncating subpixel areas
            # to an integer dtype from an original COCO annotation.
            target["area"] = box_area(xyxy)
        return output


class AlignedBatchImageCollateFunction(BatchImageCollateFunction):
    """Preserve native MixUp while also mixing existing crowd flags by source."""

    def apply_mixup(self, images, targets):
        output_images, output_targets = super().apply_mixup(images, targets)
        if output_targets is not targets:
            # The native function deep-copies the targets iff MixUp is applied.
            # Its source permutation is a one-position positive roll, exactly
            # matching the image permutation.  No random decision is repeated.
            shifted_targets = targets[-1:] + targets[:-1]
            height, width = output_images.shape[-2:]
            for output, left, right in zip(output_targets, targets, shifted_targets):
                if "iscrowd" in left or "iscrowd" in right:
                    if "iscrowd" not in left or "iscrowd" not in right:
                        raise ValueError("Cannot mix inconsistently present crowd flags without inventing labels")
                    output["iscrowd"] = torch.cat([left["iscrowd"], right["iscrowd"]])
                if "area" in output:
                    # The frozen DEIM collator receives normalized cxcywh boxes.
                    # This experiment disables subsequent random resizing by
                    # setting base_size_repeat=null in its training YAML.
                    boxes = output["boxes"]
                    output["area"] = boxes[:, 2] * boxes[:, 3] * width * height
        return output_images, output_targets


def install_augmentation_runtime() -> dict:
    """Register only this experiment's explicit YAML type; no upstream edits."""
    from engine.core import GLOBAL_CONFIG, register

    for cls in (AlignedSanitizeBoundingBoxes, AlignedBatchImageCollateFunction):
        name = cls.__name__
        if name in GLOBAL_CONFIG:
            entry = GLOBAL_CONFIG[name]
            if entry.get("_pymodule") is not sys.modules[__name__]:
                raise RuntimeError(f"Unexpected existing registration for {name}")
        else:
            register(name=name)(cls)
    return {
        "registered_transform": "AlignedSanitizeBoundingBoxes",
        "registered_train_collator": "AlignedBatchImageCollateFunction",
        "native_mosaic_modified": False,
        "native_sanitizer_registration_modified": False,
        "changes": ["align existing area/iscrowd with native keep mask", "recompute existing box area",
                    "MixUp concatenates existing crowd flags with native source permutation"],
        "composite_source_metadata": "retained unchanged; training-only, not clinical image metadata",
    }
