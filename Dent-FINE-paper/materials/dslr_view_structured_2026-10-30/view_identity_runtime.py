"""Isolated DSLR identity-supervision extension; no source-tree files are edited.

Call ``install_view_identity_runtime(settings)`` after adding the frozen local
source directory to sys.path and before constructing YAMLConfig components.
The original factor heads, denoising handling and auxiliary losses are retained.
The only identity algorithm added here is a separate prototype bank per known
acquisition view, selected from explicit image metadata rather than tooth labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


VIEW_NAMES = ("frontal", "upper_occlusal", "lower_occlusal", "left_buccal", "right_buccal")
_VIEW_ALIASES = {
    "front": 0, "frontal": 0,
    "upper_occ": 1, "upper_occlusal": 1,
    "lower_occ": 2, "lower_occlusal": 2,
    "left_buccal": 3, "right_buccal": 4,
}
_INSTALLATION: dict[str, Any] | None = None


def metadata_view_id(metadata: Mapping[str, Any], num_views: int = 5) -> int:
    """Resolve explicit view metadata; unknown is -1 and never silently view 0."""
    raw = metadata.get("view_id")
    view_id = -1
    if raw is not None:
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                raise ValueError("view_id must contain exactly one value")
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw:
            raise ValueError(f"view_id must be an integer, received {raw!r}")
        view_id = int(raw)
        if view_id < -1 or view_id >= num_views:
            raise ValueError(f"view_id {view_id} outside -1 or [0,{num_views - 1}]")
    name = metadata.get("view")
    if isinstance(name, str):
        normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
        named_id = _VIEW_ALIASES.get(normalized, -1)
        if named_id >= 0:
            if view_id >= 0 and view_id != named_id:
                raise ValueError(f"Conflicting explicit view metadata: {raw!r}, {name!r}")
            view_id = named_id
    return view_id


def batch_view_ids(targets: Sequence[Mapping[str, Any]] | None, batch_size: int,
                   device: torch.device, num_views: int = 5) -> torch.Tensor:
    if targets is None:
        return torch.full((batch_size,), -1, dtype=torch.long, device=device)
    if len(targets) != batch_size:
        raise ValueError("Target batch length differs from feature batch size")
    resolved = []
    for target in targets:
        # Native MixUp concatenates annotations without per-instance view IDs.
        # Failing loudly is preferable to assigning cross-view teeth one bank.
        if "mixup" in target:
            raise ValueError("View-specific prototype supervision forbids mixed-view MixUp targets")
        resolved.append(metadata_view_id(target, num_views=num_views))
    return torch.tensor(resolved, dtype=torch.long, device=device)


def view_prototype_logits(query_feats: torch.Tensor, prototypes: torch.Tensor,
                          view_ids: torch.Tensor, tau: float) -> torch.Tensor:
    """Normalized-query/prototype cosine logits with native LogSumExp over K.

    Input bank [V,C,K,D], output [B,Q,C]. Unknown-view logits are zero and must
    be excluded by ``view_prototype_loss``. No prototype score is fused into
    detection logits in these experiments.
    """
    if prototypes.ndim != 4 or query_feats.ndim != 3:
        raise ValueError("Expected [V,C,K,D] bank and [B,Q,D] query features")
    if query_feats.shape[-1] != prototypes.shape[-1]:
        raise ValueError("Prototype/query embedding dimensions differ")
    if view_ids.shape != (query_feats.shape[0],):
        raise ValueError("Expected one view ID per batch image")
    if tau <= 0:
        raise ValueError("Prototype temperature must be strictly positive")
    valid = (view_ids >= 0) & (view_ids < prototypes.shape[0])
    # Clamp exists only to make the batched gather legal; unknown entries are
    # zero-masked below and excluded from CE. It is not a view-0 fallback.
    banks = prototypes[view_ids.clamp(0, prototypes.shape[0] - 1)]
    feats = F.normalize(query_feats, dim=-1)
    banks = F.normalize(banks, dim=-1)
    similarity = torch.einsum("bqd,bckd->bqck", feats, banks)
    logits = torch.logsumexp(similarity / float(tau), dim=-1)
    return torch.where(valid[:, None, None], logits, torch.zeros_like(logits))


def view_prototype_loss(outputs: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                        indices: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Native matched-query mean CE, with only explicitly known-view matches."""
    logits = outputs.get("pred_proto_logits")
    if logits is None:
        # Encoder/pre/DN/aux branches do not expose prototype query features in
        # the native implementation. Preserve that training-only final-head scope.
        return {"loss_proto": outputs["pred_logits"].sum() * 0.0}
    valid = outputs.get("pred_proto_valid_mask")
    if valid is None:
        raise RuntimeError("View prototype logits are missing the explicit validity mask")
    selected_logits, selected_labels = [], []
    for b, (src_indices, tgt_indices) in enumerate(indices):
        if bool(valid[b].item()) and src_indices.numel():
            selected_logits.append(logits[b, src_indices])
            selected_labels.append(targets[b]["labels"][tgt_indices])
    if not selected_logits:
        return {"loss_proto": logits.sum() * 0.0}
    return {"loss_proto": F.cross_entropy(torch.cat(selected_logits), torch.cat(selected_labels), reduction="mean")}


def install_view_identity_runtime(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Install process-local wrappers once, retaining native construction schema.

    Settings accepted: num_views (5), attach_dataset_metadata (True), and
    guard_mixed_view_augmentations (True). Prototype K/tau/loss weights and
    factor flags remain explicit in the saved experiment YAML, not hidden here.
    """
    global _INSTALLATION
    cfg = dict(settings or {})
    num_views = int(cfg.get("num_views", 5))
    if num_views != 5:
        raise ValueError("This DSLR metadata contract has exactly five known views")
    effective = {
        "num_views": num_views,
        "attach_dataset_metadata": bool(cfg.get("attach_dataset_metadata", True)),
        "guard_mixed_view_augmentations": bool(cfg.get("guard_mixed_view_augmentations", True)),
    }
    if _INSTALLATION is not None:
        if effective != _INSTALLATION["settings"]:
            raise RuntimeError("Identity runtime already installed with different settings")
        return _INSTALLATION

    from engine.deim.dfine_decoder import DFINETransformer
    from engine.deim.deim_criterion import DEIMCriterion
    from engine.data.dataset.coco_dataset import CocoDetection

    native_init = DFINETransformer.__init__
    native_forward = DFINETransformer.forward
    native_load_item = CocoDetection.load_item
    native_loss = DEIMCriterion.loss_proto_ce

    @wraps(native_init)
    def init_with_view_banks(self, *args, **kwargs):
        native_init(self, *args, **kwargs)
        self.view_prototype_num_views = num_views
        if self.use_tooth_prototype:
            if self.prototype_fusion_weight != 0.0 or self.prototype_fusion_train:
                raise ValueError("View-specific prototype branch is training-only: prototype fusion must be disabled")
            old = self.tooth_prototypes
            bank = old.detach().new_empty((num_views, self.num_classes, self.prototype_count, self.hidden_dim))
            nn.init.normal_(bank, mean=0.0, std=0.02)
            self.tooth_prototypes = nn.Parameter(bank)

    @wraps(native_forward)
    def forward_with_view_context(self, feats, targets=None):
        if not self.use_tooth_prototype:
            return native_forward(self, feats, targets)
        view_ids = batch_view_ids(targets, feats[0].shape[0], feats[0].device, num_views=num_views)
        previous = getattr(self, "_active_prototype_view_ids", None)
        self._active_prototype_view_ids = view_ids
        try:
            outputs = native_forward(self, feats, targets)
        finally:
            if previous is None:
                del self._active_prototype_view_ids
            else:
                self._active_prototype_view_ids = previous
        outputs["pred_proto_valid_mask"] = view_ids >= 0
        outputs["pred_proto_view_ids"] = view_ids
        return outputs

    def prototype_logits_by_explicit_view(self, query_feats):
        view_ids = getattr(self, "_active_prototype_view_ids", None)
        if view_ids is None:
            raise RuntimeError("View-specific prototype logits require the explicit forward view context")
        return view_prototype_logits(query_feats, self.tooth_prototypes, view_ids, self.prototype_tau)

    @wraps(native_loss)
    def criterion_view_proto(self, outputs, targets, indices, num_boxes, **kwargs):
        return view_prototype_loss(outputs, targets, indices)

    @wraps(native_load_item)
    def load_with_explicit_view(self, idx):
        image, target = native_load_item(self, idx)
        image_id = int(target["image_id"].item())
        metadata = self.coco.imgs[image_id]
        target["view_id"] = torch.tensor([metadata_view_id(metadata, num_views)], dtype=torch.long)
        return image, target

    DFINETransformer.__init__ = init_with_view_banks
    DFINETransformer.forward = forward_with_view_context
    DFINETransformer._prototype_logits = prototype_logits_by_explicit_view
    DEIMCriterion.loss_proto_ce = criterion_view_proto
    if effective["attach_dataset_metadata"]:
        CocoDetection.load_item = load_with_explicit_view

    if effective["guard_mixed_view_augmentations"]:
        from engine.data.dataloader import BatchImageCollateFunction
        from engine.data.transforms.mosaic import Mosaic
        import inspect
        native_collate_init = BatchImageCollateFunction.__init__
        collate_signature = inspect.signature(native_collate_init)
        native_mosaic_init = Mosaic.__init__
        mosaic_signature = inspect.signature(native_mosaic_init)

        @wraps(native_collate_init)
        def guarded_collate(self, *args, **kwargs):
            bound = collate_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if float(bound.arguments["mixup_prob"]) != 0.0:
                raise ValueError("View-specific identity runtime requires MixUp probability 0")
            native_collate_init(self, *args, **kwargs)

        @wraps(native_mosaic_init)
        def guarded_mosaic(self, *args, **kwargs):
            bound = mosaic_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if float(bound.arguments["probability"]) != 0.0:
                raise ValueError("View-specific identity runtime requires Mosaic probability 0")
            native_mosaic_init(self, *args, **kwargs)

        BatchImageCollateFunction.__init__ = guarded_collate
        Mosaic.__init__ = guarded_mosaic

    _INSTALLATION = {
        "settings": effective,
        "view_names": list(VIEW_NAMES),
        "bank_shape_contract": "[5, num_classes, prototype_count, hidden_dim]",
        "unknown_view_policy": "skip prototype CE; zero unused prototype logits; no view-0 fallback",
        "prototype_aggregation": "LogSumExp_K(cosine(normalized_query, normalized_prototype)/tau)",
        "prototype_supervision_scope": "native final decoder main-query matcher indices only",
        "factor_supervision_scope": "native factor losses, including available auxiliary and denoising heads",
        "prototype_inference_fusion": False,
    }
    return _INSTALLATION
