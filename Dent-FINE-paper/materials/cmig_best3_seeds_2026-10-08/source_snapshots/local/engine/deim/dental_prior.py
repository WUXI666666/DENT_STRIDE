"""
Dental tooth-ID priors used by Dent-FINE post-processing.

The module is intentionally light-weight: priors are computed from COCO-style
annotations once, then loaded by the PostProcessor for spatial/shape fusion and
one-to-one assignment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .utils_arch import (
    default_fdi52_order,
    enumeration32_universal_to_fdi_order,
    resolve_fdi_label_order,
)


VIEW_VISIBLE_FDI = {
    "panorama": enumeration32_universal_to_fdi_order(),
    "unknown": default_fdi52_order(),
    "frontal": [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43],
    "front": [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43],
    "left": [21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38],
    "right": [11, 12, 13, 14, 15, 16, 17, 18, 41, 42, 43, 44, 45, 46, 47, 48],
    "upper_occlusal": [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower_occlusal": [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48],
}


@lru_cache(maxsize=8)
def _load_prior_cpu(path: str) -> Dict[str, torch.Tensor]:
    try:
        prior = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        prior = torch.load(path, map_location="cpu")
    if not isinstance(prior, dict):
        raise ValueError(f"Dental prior at {path} must be a dict, got {type(prior)}")
    return prior


def load_prior(
    path: Optional[Union[str, Path]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    if path in (None, "", "null"):
        return None
    prior = _load_prior_cpu(str(path))
    out = {}
    for key, value in prior.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        else:
            out[key] = value
    return out


def spatial_template_score(
    boxes_cxcywh: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    clamp: float = 20.0,
) -> torch.Tensor:
    """Return Mahalanobis log-score [B, Q, C] from normalized box centers."""
    if prior is None or "spatial_mu" not in prior or "spatial_inv_cov" not in prior:
        return boxes_cxcywh.new_zeros((*boxes_cxcywh.shape[:2], 0))
    mu = prior["spatial_mu"]
    inv_cov = prior["spatial_inv_cov"]
    valid = prior.get("spatial_valid", torch.ones(mu.shape[0], device=mu.device, dtype=torch.bool))
    centers = boxes_cxcywh[..., :2]
    diff = centers[:, :, None, :] - mu[None, None, :, :]
    score = -0.5 * torch.einsum("bqci,cij,bqcj->bqc", diff, inv_cov, diff)
    score = score.clamp(min=-float(clamp), max=0.0)
    return torch.where(valid.view(1, 1, -1), score, torch.zeros_like(score))


def shape_template_score(
    boxes_cxcywh: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    clamp: float = 20.0,
) -> torch.Tensor:
    """Return bbox-shape template score [B, Q, C] using log normalized w/h."""
    if prior is None or "shape_mu" not in prior or "shape_inv_cov" not in prior:
        return boxes_cxcywh.new_zeros((*boxes_cxcywh.shape[:2], 0))
    mu = prior["shape_mu"]
    inv_cov = prior["shape_inv_cov"]
    valid = prior.get("shape_valid", torch.ones(mu.shape[0], device=mu.device, dtype=torch.bool))
    wh = boxes_cxcywh[..., 2:].clamp_min(1e-5).log()
    diff = wh[:, :, None, :] - mu[None, None, :, :]
    score = -0.5 * torch.einsum("bqci,cij,bqcj->bqc", diff, inv_cov, diff)
    score = score.clamp(min=-float(clamp), max=0.0)
    return torch.where(valid.view(1, 1, -1), score, torch.zeros_like(score))


def appearance_prototype_score(
    query_feats: torch.Tensor,
    prior: Dict[str, torch.Tensor],
    tau: float = 0.07,
) -> Optional[torch.Tensor]:
    if query_feats is None or prior is None or "appearance_prototypes" not in prior:
        return None
    proto = prior["appearance_prototypes"]
    if proto.numel() == 0:
        return None
    feats = F.normalize(query_feats, dim=-1)
    proto = F.normalize(proto, dim=-1)
    sim = torch.einsum("bqd,ckd->bqck", feats, proto)
    return torch.logsumexp(sim / max(float(tau), 1e-6), dim=-1)


def visible_class_log_prior(
    num_classes: int,
    view_type: Optional[str],
    fdi_label_order: Optional[Union[str, Sequence[int]]],
    device: torch.device,
    dtype: torch.dtype,
    invisible_penalty: float = -2.0,
) -> torch.Tensor:
    key = (view_type or "unknown").strip().lower()
    visible_fdi = set(VIEW_VISIBLE_FDI.get(key, VIEW_VISIBLE_FDI["unknown"]))
    order = resolve_fdi_label_order(fdi_label_order, num_classes=num_classes)
    if order is None:
        order = default_fdi52_order()[:num_classes]
    values = [0.0 if int(fdi) in visible_fdi else float(invisible_penalty) for fdi in order]
    if len(values) < num_classes:
        values.extend([0.0] * (num_classes - len(values)))
    return torch.as_tensor(values[:num_classes], device=device, dtype=dtype)
