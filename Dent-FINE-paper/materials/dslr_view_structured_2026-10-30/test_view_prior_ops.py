"""CPU-only integration checks, no training or mutation of old files."""
from pathlib import Path

import torch

from build_view_priors import VIEW_TO_ID
from view_prior_ops import apply_view_priors, load_view_prior, view_template_scores


def main():
    base = Path(__file__).resolve().parent
    prior = load_view_prior(base / "priors/view_specific_tooth_priors.pt")
    assert VIEW_TO_ID == {"frontal": 0, "upper_occlusal": 1, "lower_occlusal": 2, "left_buccal": 3, "right_buccal": 4}
    assert int(prior["counts"].sum()) == 10760
    for prefix in ["spatial", "shape"]:
        cov = prior[prefix + "_cov"]
        assert torch.isfinite(cov).all()
        assert torch.linalg.eigvalsh(cov).min() > 0
        assert (cov.diagonal(dim1=-2, dim2=-1) >= 0).all()
        identity = cov @ prior[prefix + "_inv_cov"]
        assert torch.allclose(identity, torch.eye(2).expand_as(identity), atol=2e-5)
    boxes = torch.tensor([[[0.3, 0.4, 0.1, 0.2], [0.6, 0.6, 0.15, 0.1]]]).repeat(5, 1, 1)
    center, shape = view_template_scores(boxes, range(5), prior)
    assert center.shape == shape.shape == (5, 2, 52)
    assert torch.isfinite(center).all() and torch.isfinite(shape).all()
    assert center.max() <= 0 and center.min() >= -20
    assert shape.max() <= 0 and shape.min() >= -20
    for vid in range(5):
        absent = ~prior["spatial_valid"][vid]
        assert (center[vid, :, absent] == 0).all()
        assert (shape[vid, :, absent] == 0).all()
        # Existing code's unbatched per-view Mahalanobis formula must agree.
        for prefix, values, expected in [("spatial", boxes[vid, :, :2], center[vid]), ("shape", boxes[vid, :, 2:].log(), shape[vid])]:
            diff = values[:, None, :] - prior[prefix + "_mu"][vid][None, :, :]
            native = (-0.5 * torch.einsum("qci,cij,qcj->qc", diff, prior[prefix + "_inv_cov"][vid], diff)).clamp(-20, 0)
            native = torch.where(prior[prefix + "_valid"][vid][None, :], native, torch.zeros_like(native))
            assert torch.allclose(native, expected, atol=1e-5)
    logits = torch.randn_like(center)
    assert torch.equal(apply_view_priors(logits, boxes, range(5), prior, 0, 0), logits)
    corrected = apply_view_priors(logits, boxes, range(5), prior, 1, 0.2)
    assert torch.isfinite(corrected).all()
    assert ((corrected - logits) <= 1e-6).all()
    assert ((corrected - logits) >= -7.2 - 1e-5).all()
    try:
        view_template_scores(boxes, [0, 1, 2, 3, -1], prior)
        raise AssertionError("Missing view must not silently select a prior")
    except ValueError:
        pass
    print("PASS: five-view mapping; train-only10760 counts; SPD covariances; native score equivalence; invalid neutral; zero-weight invariance; bounds; missing-view rejection")


if __name__ == "__main__":
    main()
