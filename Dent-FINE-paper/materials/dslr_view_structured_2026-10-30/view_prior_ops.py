"""Lightweight view-specific SpatialShape scoring, before top-k selection.

Unknown view IDs fail explicitly: they must never silently select the frontal
bank. Absent FDI/view combinations yield a neutral zero penalty, not invented
anatomical evidence. Caller must use observed acquisition metadata, not GT teeth.
"""
from __future__ import annotations

import torch


def load_view_prior(path, device=None):
    prior = torch.load(path, map_location=device or "cpu", weights_only=True)
    assert prior["num_views"] == 5 and prior["num_classes"] == 52
    return prior


def view_template_scores(boxes_cxcywh, view_ids, prior, clamp=20.0):
    """Return center and log-shape penalties [B,Q,52] for each image's view."""
    assert boxes_cxcywh.ndim == 3 and boxes_cxcywh.shape[-1] == 4
    views = torch.as_tensor(view_ids, dtype=torch.long, device=boxes_cxcywh.device).reshape(-1)
    assert views.numel() == boxes_cxcywh.shape[0]
    if torch.any(views < 0) or torch.any(views >= int(prior["num_views"])):
        raise ValueError(f"Missing/invalid acquisition view IDs: {views.tolist()}")
    dtype, device = boxes_cxcywh.dtype, boxes_cxcywh.device
    scores = []
    for prefix, points in [("spatial", boxes_cxcywh[..., :2]), ("shape", boxes_cxcywh[..., 2:].clamp_min(1e-5).log())]:
        mu = prior[prefix + "_mu"].to(device=device, dtype=dtype)[views]
        inv = prior[prefix + "_inv_cov"].to(device=device, dtype=dtype)[views]
        valid = prior[prefix + "_valid"].to(device=device)[views]
        diff = points[:, :, None, :] - mu[:, None, :, :]
        score = -0.5 * torch.einsum("bqci,bcij,bqcj->bqc", diff, inv, diff)
        score = score.clamp(min=-float(clamp), max=0.0)
        scores.append(torch.where(valid[:, None, :], score, torch.zeros_like(score)))
    return tuple(scores)


def apply_view_priors(logits, boxes_cxcywh, view_ids, prior, spatial_weight=1.0, shape_weight=0.2):
    """Same relative correction as original PostProcessor, but per-image view.

    Weights are explicit caller/config hyperparameters, not tuned by this helper.
    This is score correction only, not a geometry training loss.
    """
    center, shape = view_template_scores(boxes_cxcywh, view_ids, prior)
    if logits.shape != center.shape:
        raise ValueError(f"Logit/class count mismatch: {logits.shape} vs {center.shape}")
    result = logits
    for score, weight in [(center, spatial_weight), (shape, shape_weight)]:
        if weight:
            relative = (score - score.amax(dim=-1, keepdim=True)).clamp(-6.0, 0.0)
            result = result + float(weight) * relative
    return result
