"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from scipy.optimize import linear_sum_assignment

from ..core import register
from .dental_prior import (
    appearance_prototype_score,
    load_prior,
    shape_template_score,
    spatial_template_score,
    visible_class_log_prior,
)


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category',
        'dental_prior_path',
        'fdi_label_order',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        dental_prior_path=None,
        fdi_label_order=None,
        use_dental_spatial_prior=False,
        spatial_prior_weight=1.0,
        use_dental_shape_prior=False,
        shape_prior_weight=0.2,
        use_dental_appearance_prior=False,
        appearance_prior_weight=0.5,
        appearance_tau=0.07,
        use_dental_view_prior=False,
        dental_view_type='unknown',
        view_prior_weight=1.0,
        use_global_assignment=False,
        assignment_score_threshold=0.01,
        assignment_keep_unassigned=True,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.dental_prior_path = dental_prior_path
        self.fdi_label_order = fdi_label_order
        self.use_dental_spatial_prior = bool(use_dental_spatial_prior)
        self.spatial_prior_weight = float(spatial_prior_weight)
        self.use_dental_shape_prior = bool(use_dental_shape_prior)
        self.shape_prior_weight = float(shape_prior_weight)
        self.use_dental_appearance_prior = bool(use_dental_appearance_prior)
        self.appearance_prior_weight = float(appearance_prior_weight)
        self.appearance_tau = float(appearance_tau)
        self.use_dental_view_prior = bool(use_dental_view_prior)
        self.dental_view_type = dental_view_type
        self.view_prior_weight = float(view_prior_weight)
        self.use_global_assignment = bool(use_global_assignment)
        self.assignment_score_threshold = float(assignment_score_threshold)
        self.assignment_keep_unassigned = bool(assignment_keep_unassigned)
        self.deploy_mode = False

    @staticmethod
    def _soft_prior(score):
        score = score - score.amax(dim=-1, keepdim=True)
        return score.clamp(min=-6.0, max=0.0)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def _apply_dental_priors(self, outputs, logits, boxes):
        if not (
            self.use_dental_spatial_prior
            or self.use_dental_shape_prior
            or self.use_dental_appearance_prior
            or self.use_dental_view_prior
        ):
            return logits

        logits = logits.clone()
        prior = load_prior(self.dental_prior_path, logits.device, logits.dtype)
        if self.use_dental_spatial_prior and prior is not None:
            score = spatial_template_score(boxes, prior)
            if score.shape[-1] == logits.shape[-1]:
                logits = logits + self.spatial_prior_weight * self._soft_prior(score)
        if self.use_dental_shape_prior and prior is not None:
            score = shape_template_score(boxes, prior)
            if score.shape[-1] == logits.shape[-1]:
                logits = logits + self.shape_prior_weight * self._soft_prior(score)
        if self.use_dental_appearance_prior and prior is not None:
            app = appearance_prototype_score(outputs.get('query_feats'), prior, tau=self.appearance_tau)
            if app is not None and app.shape == logits.shape:
                logits = logits + self.appearance_prior_weight * app
        if self.use_dental_view_prior:
            view_prior = visible_class_log_prior(
                self.num_classes,
                self.dental_view_type,
                self.fdi_label_order,
                logits.device,
                logits.dtype,
            )
            logits = logits + self.view_prior_weight * view_prior.view(1, 1, -1)
        return logits

    def _assignment_results(self, logits, boxes):
        all_scores = logits.sigmoid() if self.use_focal_loss else logits.softmax(-1)
        results = []
        for prob, box in zip(all_scores, boxes):
            cost = (-prob).detach().cpu().numpy()
            row, col = linear_sum_assignment(cost)
            row = torch.as_tensor(row, device=prob.device, dtype=torch.long)
            col = torch.as_tensor(col, device=prob.device, dtype=torch.long)
            scores = prob[row, col]
            keep = scores >= self.assignment_score_threshold
            row, col, scores = row[keep], col[keep], scores[keep]

            if self.assignment_keep_unassigned:
                flat_scores, flat_index = torch.topk(
                    prob.flatten(), min(self.num_top_queries, prob.numel()), dim=-1
                )
                flat_rows = flat_index // self.num_classes
                flat_cols = mod(flat_index, self.num_classes)
                assigned_rows = set(int(v) for v in row.detach().cpu().tolist())
                extra_rows, extra_cols, extra_scores = [], [], []
                for r, c, s in zip(flat_rows.tolist(), flat_cols.tolist(), flat_scores.tolist()):
                    if r in assigned_rows:
                        continue
                    extra_rows.append(r)
                    extra_cols.append(c)
                    extra_scores.append(s)
                    if len(extra_rows) + int(row.numel()) >= self.num_top_queries:
                        break
                if extra_rows:
                    row = torch.cat([row, torch.as_tensor(extra_rows, device=prob.device, dtype=torch.long)])
                    col = torch.cat([col, torch.as_tensor(extra_cols, device=prob.device, dtype=torch.long)])
                    scores = torch.cat([scores, torch.as_tensor(extra_scores, device=prob.device, dtype=prob.dtype)])

            if scores.numel() > self.num_top_queries:
                scores, order = torch.topk(scores, self.num_top_queries)
                row, col = row[order], col[order]
            else:
                order = torch.argsort(scores, descending=True)
                scores, row, col = scores[order], row[order], col[order]
            results.append(dict(labels=col, boxes=box[row], scores=scores))
        return results

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        logits = self._apply_dental_priors(outputs, logits, boxes)
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_global_assignment:
            results = self._assignment_results(logits, bbox_pred)
            if self.remap_mscoco_category:
                from ..data.dataset import mscoco_label2category
                for result in results:
                    result['labels'] = torch.tensor(
                        [mscoco_label2category[int(x.item())] for x in result['labels'].flatten()],
                        device=result['labels'].device,
                    ).reshape(result['labels'].shape)
            return results

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
