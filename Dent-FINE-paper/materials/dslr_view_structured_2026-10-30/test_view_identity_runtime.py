"""CPU checks for the isolated view-conditioned identity extension."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "cmig_selected_configs_multiseed_2026-10-11/source_snapshots/local"
sys.path.insert(0, str(SOURCE))
from view_identity_runtime import (batch_view_ids, install_view_identity_runtime,
                                   metadata_view_id, view_prototype_logits,
                                   view_prototype_loss)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11030)
    checks = {}
    assert metadata_view_id({}) == -1
    assert metadata_view_id({"view_id": 0, "view": "frontal"}) == 0
    assert metadata_view_id({"view_id": 1, "view": "upper_occ"}) == 1
    try:
        metadata_view_id({"view_id": 0, "view": "right_buccal"})
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting view metadata was accepted")
    checks["metadata_consistency_and_unknown"] = True

    bank = nn.Parameter(torch.randn(5, 7, 4, 8))
    query = torch.randn(3, 6, 8, requires_grad=True)
    vids = torch.tensor([1, 4, -1])
    actual = view_prototype_logits(query, bank, vids, .5)
    normalized_q = F.normalize(query, dim=-1)
    manual = torch.stack([
        torch.logsumexp(torch.einsum("qd,ckd->qck", normalized_q[b], F.normalize(bank[v], dim=-1)) / .5, dim=-1)
        for b, v in enumerate([1, 4])
    ])
    torch.testing.assert_close(actual[:2], manual)
    assert torch.count_nonzero(actual[2]) == 0
    checks["normalized_logsumexp_matches_native_equation"] = True

    idx = [(torch.tensor([1, 4]), torch.tensor([0, 1]))] * 3
    targets = [{"labels": torch.tensor([2, 5])}] * 3
    output = {"pred_proto_logits": actual, "pred_proto_valid_mask": vids >= 0}
    loss = view_prototype_loss(output, targets, idx)["loss_proto"]
    expected = F.cross_entropy(actual[:2, [1, 4]].reshape(-1, 7), torch.tensor([2, 5, 2, 5]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for view in (0, 2, 3):
        assert torch.count_nonzero(bank.grad[view]) == 0
    assert bank.grad[1].abs().sum() > 0 and bank.grad[4].abs().sum() > 0
    assert torch.count_nonzero(query.grad[2]) == 0
    checks["gradient_separates_view_banks_and_skips_unknown"] = True

    isolated_bank = nn.Parameter(torch.randn(5, 7, 4, 8))
    unknown_logits = view_prototype_logits(query.detach(), isolated_bank, torch.full((3,), -1), .5)
    zero_loss = view_prototype_loss({"pred_proto_logits": unknown_logits, "pred_proto_valid_mask": torch.zeros(3, dtype=torch.bool)}, targets, idx)["loss_proto"]
    zero_loss.backward()
    assert zero_loss.item() == 0 and isolated_bank.grad is not None
    assert torch.count_nonzero(isolated_bank.grad) == 0
    checks["all_unknown_zero_loss_keeps_bank_in_graph"] = True

    info = install_view_identity_runtime({"num_views": 5})
    from engine.deim.dfine_decoder import DFINETransformer
    from engine.data.dataloader import BatchImageCollateFunction
    from engine.data.transforms.mosaic import Mosaic

    for constructor, kwargs in [(BatchImageCollateFunction, {"mixup_prob": .5}), (Mosaic, {"probability": .5})]:
        try:
            constructor(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Mixed-view augmentation was accepted")
    checks["mixed_view_augmentation_guards"] = True

    model = DFINETransformer(num_classes=52, hidden_dim=32, num_queries=5,
                             feat_channels=[16, 32, 64], feat_strides=[8, 16, 32],
                             num_levels=3, num_layers=2, nhead=4, dim_feedforward=64,
                             num_denoising=0, eval_spatial_size=None,
                             use_factor_heads=True, use_tooth_prototype=True,
                             prototype_count=4, prototype_tau=.5,
                             prototype_fusion_weight=0., prototype_fusion_train=False)
    assert model.tooth_prototypes.shape == (5, 52, 4, 32)
    feats = [torch.randn(2, c, size, size) for c, size in [(16, 8), (32, 4), (64, 2)]]
    metadata = [{"view_id": torch.tensor([0])}, {"view_id": torch.tensor([3])}]
    model.eval()
    with torch.no_grad():
        with_view = model(feats, metadata)
        without_view = model(feats)
        model.use_tooth_prototype = False
        baseline = model(feats)
        model.use_tooth_prototype = True
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(with_view[key], baseline[key], rtol=0, atol=0)
        torch.testing.assert_close(without_view[key], baseline[key], rtol=0, atol=0)
    assert with_view["pred_proto_valid_mask"].tolist() == [True, True]
    assert without_view["pred_proto_valid_mask"].tolist() == [False, False]
    checks["evaluation_scores_boxes_identical_with_unused_prototype_branch"] = True

    training_targets = [{"view_id": torch.tensor([v]), "labels": torch.tensor([0, 9]),
                         "boxes": torch.tensor([[.4, .5, .2, .3], [.6, .5, .2, .3]])}
                        for v in (0, 3)]
    model.train()
    output = model(feats, training_targets)
    assert output["pred_proto_logits"].shape == (2, 5, 52)
    assert "pred_quad_logits" in output and "pred_tooth_logits" in output and "pred_dent_logits" in output
    assert len(output["aux_outputs"]) == 1
    output["pred_proto_logits"].sum().backward()
    assert model.tooth_prototypes.grad[0].abs().sum() > 0
    assert model.tooth_prototypes.grad[3].abs().sum() > 0
    for v in (1, 2, 4):
        assert model.tooth_prototypes.grad[v].abs().sum() == 0
    checks["native_decoder_training_factor_and_aux_outputs_preserved"] = True

    from engine.deim.matcher import HungarianMatcher
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.optim.ema import ModelEMA
    model.zero_grad(set_to_none=True)
    model.num_denoising = 4
    # Enabling DN after construction needs the native embedding, as the tiny
    # test decoder was initially instantiated without DN.
    model.denoising_class_embed = nn.Embedding(53, 32, padding_idx=52)
    output_dn = model(feats, training_targets)
    assert "dn_outputs" in output_dn and output_dn["pred_proto_logits"].shape == (2, 5, 52)
    matcher = HungarianMatcher({"cost_class": 2., "cost_bbox": 5., "cost_giou": 2.}, use_focal_loss=True)
    criterion = DEIMCriterion(matcher, {"loss_quad": .08, "loss_tooth": .08,
                                       "loss_dent": .03, "loss_proto": .03},
                              ["mal", "boxes", "quad", "tooth", "dent", "proto"],
                              num_classes=52, mal_alpha=.5)
    losses = criterion(output_dn, training_targets)
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert losses["loss_proto"].item() > 0 and any("loss_quad_dn" in name for name in losses)
    assert all(value.item() == 0 for name, value in losses.items() if name.startswith("loss_proto_"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = ModelEMA(model, decay=.999)
    sum(losses.values()).backward()
    optimizer.step()
    ema.update(model)
    assert ema.module.tooth_prototypes.shape == (5, 52, 4, 32)
    checks["native_criterion_denoising_optimizer_and_ema_step"] = True

    payload = {"checks": checks, "runtime": info, "passed": all(checks.values()),
               "torch_version": torch.__version__, "device": "cpu"}
    (ROOT / "view_identity_test_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
