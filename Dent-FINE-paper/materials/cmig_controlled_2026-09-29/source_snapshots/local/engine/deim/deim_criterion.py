"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register
from .utils_arch import fit_arch_t, fdi_sequence_targets, fdi_index_map


@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,# +++ MrDETR O2M 相关 +++
        use_fdi_anatomy_loss=False,
        use_dsfed_dual_scale_kd=False,
        dsfed_kd_temperature=2.0,
        dsfed_kd_weight=0.05,
        dsfed_box_kd_weight=0.02,
        dsfed_kd_topk_ratio=0.20,
        dsfed_kd_min_conf=0.10,
        dsfed_final_kd_weight=0.02,
        use_tooth_size_prior_loss=False,
        nwd_normalizer=0.08,
        o2m_topk=6,          # 每个 GT 保留的候选 query 数
        o2m_iou_thr=0.1,     # IoU 阈值（筛掉太差的候选）
        o2m_cls_thr=0.02,    # 分类分数阈值（sigmoid 后）
        mr_loss_on=('focal','boxes'),  # O2M 分支启用的损失类型（'focal' 或 'vfl'/'mal' 任选其一 + 'boxes'）

        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        # __init__ 内保存
        self.o2m_topk = o2m_topk
        self.o2m_iou_thr = o2m_iou_thr
        self.o2m_cls_thr = o2m_cls_thr
        self.mr_loss_on = tuple(mr_loss_on)

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.weight_dict.setdefault('loss_focal', 1.0)
        self.weight_dict.setdefault('loss_bbox', 5.0)
        self.weight_dict.setdefault('loss_giou', 2.0)
        # MR 支路（在 criterion 里会 fallback 到主干权重，这里显式压低）
        self.weight_dict.setdefault('loss_focal_mr', 0.25)
        self.weight_dict.setdefault('loss_bbox_mr', 1.0)
        self.weight_dict.setdefault('loss_giou_mr', 0.5)
        self.weight_dict.setdefault('loss_quad', 0.10)
        self.weight_dict.setdefault('loss_tooth', 0.10)
        self.weight_dict.setdefault('loss_dent', 0.05)
        self.weight_dict.setdefault('loss_proto', 0.05)
        self.weight_dict.setdefault('loss_arch', 0.25)  # 可在配置里覆盖
        self.weight_dict.setdefault('loss_nwd', 1.0)
        self.weight_dict.setdefault('loss_fdi_factor', 0.20)
        self.weight_dict.setdefault('loss_fdi_unique', 0.05)
        self.weight_dict.setdefault('loss_fdi_dent', 0.05)
        self.weight_dict.setdefault('loss_dsfed_global_kd', 1.0)
        self.weight_dict.setdefault('loss_dsfed_local_kd', 1.0)
        self.weight_dict.setdefault('loss_dsfed_box_kd', 1.0)
        self.weight_dict.setdefault('loss_tooth_size_prior', 0.03)


        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.use_fdi_anatomy_loss = bool(use_fdi_anatomy_loss)
        self.use_dsfed_dual_scale_kd = bool(use_dsfed_dual_scale_kd)
        self.dsfed_kd_temperature = float(dsfed_kd_temperature)
        self.dsfed_kd_weight = float(dsfed_kd_weight)
        self.dsfed_box_kd_weight = float(dsfed_box_kd_weight)
        self.dsfed_kd_topk_ratio = float(dsfed_kd_topk_ratio)
        self.dsfed_kd_min_conf = float(dsfed_kd_min_conf)
        self.dsfed_final_kd_weight = float(dsfed_final_kd_weight)
        self.use_tooth_size_prior_loss = bool(use_tooth_size_prior_loss)
        self.nwd_normalizer = float(nwd_normalizer)
    @torch.no_grad()
    def build_o2m_indices(self, logits: torch.Tensor, boxes: torch.Tensor, targets, 
                        topk: int, iou_thr: float, cls_thr: float):
        """
        基于分类分数和 IoU 的简单 O2M 策略：
        - 对每个样本、每个 GT（标签 c、框 g）：
            1) 取该类别 c 的 sigmoid 分数向量 s = σ(logits[:, c])
            2) 计算所有 query 预测框与 g 的 IoU
            3) 过滤 s>=cls_thr 且 IoU>=iou_thr 的候选，按 s 排序取 top-k
        - 允许同一 query 被多个 GT 选中（真正 O2M）
        返回：indices 列表 [(src_idx, tgt_idx)] * bs
        """
        B, Q, C = logits.shape
        device = logits.device
        indices = []

        # [B, Q, 4] cxcywh -> xyxy
        from .box_ops import box_cxcywh_to_xyxy
        pred_xyxy = box_cxcywh_to_xyxy(boxes)

        for b in range(B):
            tgt = targets[b]
            num_gt = len(tgt['labels'])
            if num_gt == 0:
                indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                                torch.zeros(0, dtype=torch.int64, device=device)))
                continue

            # 当前样本的预测
            logit_b = logits[b]         # [Q, C]
            prob_b  = logit_b.sigmoid() # [Q, C]
            box_b   = pred_xyxy[b]      # [Q, 4]

            src_list, tgt_list = [], []
            for j in range(num_gt):
                c = int(tgt['labels'][j].item())
                g = tgt['boxes'][j].unsqueeze(0)            # cxcywh
                g_xyxy = box_cxcywh_to_xyxy(g)              # [1,4]

                # IoU: [Q]
                from .box_ops import box_iou
                ious, _ = box_iou(box_b, g_xyxy)            # [Q,1]
                ious = ious.squeeze(1)

                # 类别分数: [Q]
                cls_score = prob_b[:, c]

                # 过滤
                mask = (cls_score >= cls_thr) & (ious >= iou_thr)
                cand_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

                if cand_idx.numel() == 0:
                    fallback_score = (cls_score * ious).nan_to_num(0.)
                    k = max(1, min(topk, Q))  # ← 至少 1
                    cand_idx = torch.topk(fallback_score, k=k, largest=True)[1]
                # 选 top-k（按分类分数）
                k = min(topk, cand_idx.numel())
                top_idx = cand_idx[torch.topk(cls_score[cand_idx], k=k, largest=True)[1]]

                src_list.append(top_idx)
                tgt_list.append(torch.full((k,), j, dtype=torch.int64, device=device))

            src_idx = torch.cat(src_list, dim=0)
            tgt_idx = torch.cat(tgt_list, dim=0)
            indices.append((src_idx, tgt_idx))
        return indices

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        target = target.to(dtype=src_logits.dtype, device=src_logits.device)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target = target.to(dtype=src_logits.dtype, device=src_logits.device)

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target = target.to(dtype=src_logits.dtype, device=src_logits.device)

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_nwd(self, outputs, targets, indices, num_boxes, **kwargs):
        """Normalized Gaussian Wasserstein Distance loss for small dense boxes."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0:
            return {'loss_nwd': outputs['pred_boxes'].sum() * 0.0}

        center_dist = (src_boxes[:, :2] - target_boxes[:, :2]).pow(2).sum(dim=-1)
        size_dist = (src_boxes[:, 2:] - target_boxes[:, 2:]).pow(2).sum(dim=-1) / 4.0
        wasserstein = torch.sqrt((center_dist + size_dist).clamp_min(1e-12))
        nwd = torch.exp(-wasserstein / max(self.nwd_normalizer, 1e-6))
        return {'loss_nwd': (1.0 - nwd).sum() / num_boxes}

    def loss_dsfed_dual_scale_kd(self, outputs):
        """
        DSFedMed-inspired dual-scale mutual distillation for one detector.
        Detection uses sigmoid multi-label logits rather than softmax masks, so
        the KD is class-wise BCE consistency on reliable queries. The final head
        teaches lightweight aux/pre/encoder heads, and a small reverse term
        lets the final head absorb stable multi-scale consensus instead of only
        training auxiliary outputs.
        """
        if 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
            return {}
        final_logits = outputs['pred_logits']
        final_boxes = outputs['pred_boxes']
        teacher_logits = final_logits.detach()
        teacher_boxes = final_boxes.detach()
        if teacher_logits.numel() == 0:
            zero = final_logits.sum() * 0.0
            return {
                'loss_dsfed_global_kd': zero,
                'loss_dsfed_local_kd': zero,
                'loss_dsfed_box_kd': zero,
            }

        students = []
        for item in outputs.get('aux_outputs', []):
            students.append(item)
        if 'pre_outputs' in outputs:
            students.append(outputs['pre_outputs'])
        for item in outputs.get('enc_aux_outputs', []):
            students.append(item)

        valid_students = []
        for item in students:
            if not isinstance(item, dict):
                continue
            if 'pred_logits' not in item or 'pred_boxes' not in item:
                continue
            if item['pred_logits'].shape == teacher_logits.shape and item['pred_boxes'].shape == teacher_boxes.shape:
                valid_students.append(item)
        if not valid_students:
            zero = final_logits.sum() * 0.0
            return {
                'loss_dsfed_global_kd': zero,
                'loss_dsfed_local_kd': zero,
                'loss_dsfed_box_kd': zero,
            }

        t = max(self.dsfed_kd_temperature, 1e-4)
        teacher_prob = torch.sigmoid(teacher_logits.float() / t)
        teacher_conf = teacher_logits.sigmoid().amax(dim=-1)
        reliable = teacher_conf >= self.dsfed_kd_min_conf
        _, num_queries = teacher_conf.shape
        k = max(1, min(num_queries, int(round(num_queries * self.dsfed_kd_topk_ratio))))
        topk_idx = torch.topk(teacher_conf, k=k, dim=1, largest=True).indices
        topk_mask = torch.zeros_like(teacher_conf, dtype=torch.bool)
        topk_mask.scatter_(1, topk_idx, True)
        local_mask = topk_mask & reliable
        global_weight = teacher_conf.detach().clamp_min(0.02)

        global_losses = []
        local_losses = []
        box_losses = []
        consensus_probs = []
        consensus_boxes = []
        gather_cls = topk_idx.unsqueeze(-1).expand(-1, -1, teacher_logits.shape[-1])
        gather_box = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        teacher_prob_local = torch.gather(teacher_prob, dim=1, index=gather_cls)
        teacher_boxes_local = torch.gather(teacher_boxes, dim=1, index=gather_box)
        local_weight = torch.gather(global_weight, dim=1, index=topk_idx)
        local_weight = local_weight * local_mask.gather(1, topk_idx).to(local_weight.dtype)
        local_den = local_weight.sum().clamp_min(1.0)

        for student in valid_students:
            student_logits = student['pred_logits'].float()
            kd_map = F.binary_cross_entropy_with_logits(
                student_logits / t,
                teacher_prob,
                reduction='none',
            ).mean(dim=-1) * (t ** 2)
            global_losses.append((kd_map * global_weight).sum() / global_weight.sum().clamp_min(1.0))

            student_logits_local = torch.gather(student_logits, dim=1, index=gather_cls)
            local_kd = F.binary_cross_entropy_with_logits(
                student_logits_local / t,
                teacher_prob_local,
                reduction='none',
            ).mean(dim=-1) * (t ** 2)
            local_losses.append((local_kd * local_weight).sum() / local_den)

            student_boxes_local = torch.gather(student['pred_boxes'], dim=1, index=gather_box)
            box_l1 = (student_boxes_local - teacher_boxes_local).abs().mean(dim=-1)
            box_losses.append((box_l1 * local_weight).sum() / local_den)
            consensus_probs.append(torch.sigmoid(student_logits.detach() / t))
            consensus_boxes.append(student['pred_boxes'].detach())

        global_loss = torch.stack(global_losses).mean()
        local_loss = torch.stack(local_losses).mean()
        box_loss = torch.stack(box_losses).mean()

        if consensus_probs and self.dsfed_final_kd_weight > 0:
            aux_prob = torch.stack(consensus_probs, dim=0).mean(dim=0)
            aux_box = torch.stack(consensus_boxes, dim=0).mean(dim=0)
            blend_prob = (0.70 * teacher_prob + 0.30 * aux_prob).detach()
            final_global = F.binary_cross_entropy_with_logits(
                final_logits.float() / t,
                blend_prob,
                reduction='none',
            ).mean(dim=-1) * (t ** 2)
            final_global = (final_global * global_weight).sum() / global_weight.sum().clamp_min(1.0)
            aux_box_local = torch.gather(aux_box, dim=1, index=gather_box)
            final_box_local = torch.gather(final_boxes, dim=1, index=gather_box)
            final_box = (final_box_local - aux_box_local).abs().mean(dim=-1)
            final_box = (final_box * local_weight).sum() / local_den
            global_loss = global_loss + self.dsfed_final_kd_weight * final_global
            box_loss = box_loss + self.dsfed_final_kd_weight * final_box

        return {
            'loss_dsfed_global_kd': global_loss * self.dsfed_kd_weight * self.weight_dict['loss_dsfed_global_kd'],
            'loss_dsfed_local_kd': local_loss * self.dsfed_kd_weight * self.weight_dict['loss_dsfed_local_kd'],
            'loss_dsfed_box_kd': box_loss * self.dsfed_box_kd_weight * self.weight_dict['loss_dsfed_box_kd'],
        }

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    den = (self.num_pos + self.num_neg)
                    if isinstance(den, torch.Tensor):
                        den = den.clamp(min=1e-6)
                    else:
                        den = max(float(den), 1e-6)
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / den


        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'nwd': self.loss_nwd,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
                # === NEW ===
    'quad': self.loss_quad_ce,
    'tooth': self.loss_tooth_ce,
    'dent': self.loss_dent_ce,
    'proto': self.loss_proto_ce,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    def _labels_to_factor(self, target_labels, kind='quad'):
        # 借助 fdi_index_map，把 cls -> FDI -> (quad, tooth, dent)
        from .utils_arch import fdi_index_map
        _map = fdi_index_map(self.num_classes)      # {FDI->cls_id}
        id2fdi = {v:k for k,v in _map.items()}
        vals = []
        for lab in target_labels.tolist():
            fdi = id2fdi.get(int(lab), None)
            if fdi is None:
                vals.append(0)  # 兜底
            else:
                q, p = int(fdi)//10, int(fdi)%10
                if kind=='quad':  vals.append(min(max(q - 1, 0), 7))      # 0..7
                elif kind=='tooth':vals.append(min(max(p - 1, 0), 7))      # 0..7
                else:              vals.append(0 if int(fdi)>=50 else 1)  # 乳=0, 恒=1
        return torch.as_tensor(vals, device=target_labels.device, dtype=torch.long)

    def _ce_on_factor(self, logits, targets, indices, num_boxes, kind):
        # logits: [B,Q,K]
        idx = self._get_src_permutation_idx(indices)
        tcls = torch.cat([t["labels"][J] for t,(_,J) in zip(targets, indices)])  # [M]
        if logits is None:
            return {f'loss_{kind}': tcls.float().sum() * 0.0}
        if tcls.numel() == 0:
            return {f'loss_{kind}': logits.sum() * 0.0}
        src = logits[idx]                     # [M,K]
        tfactor = self._labels_to_factor(tcls, kind=kind)                        # [M]
        loss = F.cross_entropy(src, tfactor, reduction='mean')
        return {f'loss_{kind}': loss}

    def loss_quad_ce(self, outputs, targets, indices, num_boxes, **kwargs):
        return self._ce_on_factor(outputs.get('pred_quad_logits'), targets, indices, num_boxes, 'quad')
    def loss_tooth_ce(self, outputs, targets, indices, num_boxes, **kwargs):
        return self._ce_on_factor(outputs.get('pred_tooth_logits'), targets, indices, num_boxes, 'tooth')
    def loss_dent_ce(self, outputs, targets, indices, num_boxes, **kwargs):
        return self._ce_on_factor(outputs.get('pred_dent_logits'), targets, indices, num_boxes, 'dent')

    def loss_proto_ce(self, outputs, targets, indices, num_boxes, **kwargs):
        logits = outputs.get('pred_proto_logits')
        idx = self._get_src_permutation_idx(indices)
        tcls = torch.cat([t["labels"][J] for t,(_,J) in zip(targets, indices)])
        if logits is None:
            return {'loss_proto': tcls.float().sum() * 0.0}
        if tcls.numel() == 0:
            return {'loss_proto': logits.sum() * 0.0}
        loss = F.cross_entropy(logits[idx], tcls, reduction='mean')
        return {'loss_proto': loss}

    def _class_fdi_tensors(self, device):
        _map = fdi_index_map(self.num_classes)
        id2fdi = {int(v): int(k) for k, v in _map.items()}
        quad = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        tooth = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        dent = torch.ones(self.num_classes, dtype=torch.long, device=device)
        jaw = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        valid = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        for cid in range(self.num_classes):
            fdi = id2fdi.get(cid)
            if fdi is None:
                continue
            q, p = divmod(int(fdi), 10)
            quad[cid] = min(max(q - 1, 0), 7)
            tooth[cid] = min(max(p - 1, 0), 7)
            dent[cid] = 0 if int(fdi) >= 50 else 1
            jaw[cid] = 0 if q in (1, 2, 5, 6) else 1
            valid[cid] = True
        return quad, tooth, dent, jaw, valid

    @staticmethod
    def _factor_logsumexp_loss(log_probs, class_factor, target_factor, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(class_factor, dtype=torch.bool)
        losses = []
        for row, factor in enumerate(target_factor):
            mask = valid_mask & (class_factor == factor)
            if mask.any():
                losses.append(-torch.logsumexp(log_probs[row, mask], dim=-1))
        if not losses:
            return log_probs.sum() * 0.0
        return torch.stack(losses).mean()

    def loss_fdi_anatomy_final(self, outputs, targets, indices, num_boxes):
        if self.num_classes not in (32, 52) or 'pred_logits' not in outputs:
            zero = next(iter(outputs.values())).sum() * 0.0
            return {
                'loss_fdi_factor': zero,
                'loss_fdi_unique': zero,
                'loss_fdi_dent': zero,
            }

        logits = outputs['pred_logits']
        device = logits.device
        quad_map, tooth_map, dent_map, _, valid = self._class_fdi_tensors(device)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = logits.sum() * 0.0
            return {
                'loss_fdi_factor': zero,
                'loss_fdi_unique': zero,
                'loss_fdi_dent': zero,
            }

        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_valid = valid[target_classes]
        if not target_valid.any():
            zero = logits.sum() * 0.0
            return {
                'loss_fdi_factor': zero,
                'loss_fdi_unique': zero,
                'loss_fdi_dent': zero,
            }

        src_logits = logits[idx][target_valid]
        target_classes = target_classes[target_valid]
        log_probs = F.log_softmax(src_logits.float(), dim=-1)

        target_quad = quad_map[target_classes]
        target_tooth = tooth_map[target_classes]
        target_dent = dent_map[target_classes]

        factor_loss = 0.5 * self._factor_logsumexp_loss(log_probs, quad_map, target_quad, valid)
        factor_loss = factor_loss + self._factor_logsumexp_loss(log_probs, tooth_map, target_tooth, valid)

        if dent_map[valid].unique().numel() > 1:
            dent_loss = self._factor_logsumexp_loss(log_probs, dent_map, target_dent, valid)
        else:
            dent_loss = logits.sum() * 0.0

        probs = logits.sigmoid().float()
        present = torch.zeros(logits.shape[0], self.num_classes, dtype=probs.dtype, device=device)
        for b, tgt in enumerate(targets):
            labels = tgt["labels"]
            labels = labels[(labels >= 0) & (labels < self.num_classes)]
            if labels.numel() > 0:
                present[b, labels.unique()] = 1.0
        class_mass = probs.sum(dim=1)
        duplicate_loss = F.relu(class_mass - 1.0).pow(2).mean()
        max_prob = probs.max(dim=1).values
        missing_loss = (present * (1.0 - max_prob).pow(2)).sum() / present.sum().clamp_min(1.0)
        unique_loss = duplicate_loss + 0.5 * missing_loss

        return {
            'loss_fdi_factor': factor_loss,
            'loss_fdi_unique': unique_loss,
            'loss_fdi_dent': dent_loss,
        }

    @staticmethod
    def _tooth_size_rank(fdi):
        digit = int(fdi) % 10
        if int(fdi) >= 50:
            primary = {1: 0.85, 2: 0.90, 3: 1.00, 4: 1.12, 5: 1.22}
            return primary.get(digit, 1.0)
        permanent = {1: 0.85, 2: 0.88, 3: 1.02, 4: 1.08, 5: 1.12, 6: 1.35, 7: 1.30, 8: 1.18}
        return permanent.get(digit, 1.0)

    def loss_tooth_size_prior(self, outputs, targets, indices):
        if self.num_classes not in (32, 52) or 'pred_boxes' not in outputs:
            zero = next(iter(outputs.values())).sum() * 0.0
            return zero

        pred_boxes = outputs['pred_boxes']
        device = pred_boxes.device
        fdi2cls = fdi_index_map(self.num_classes)
        id2fdi = {int(cid): int(fdi) for fdi, cid in fdi2cls.items()}
        total = pred_boxes.sum() * 0.0
        groups = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() <= 1:
                continue
            src_idx = src_idx.to(device)
            tgt_idx = tgt_idx.to(device)
            boxes = pred_boxes[b, src_idx]
            labels = targets[b]['labels'][tgt_idx].to(device)
            fdis = [id2fdi.get(int(v), None) for v in labels.detach().cpu().tolist()]
            ranks = [self._tooth_size_rank(f) if f is not None else None for f in fdis]
            keep = torch.tensor([r is not None for r in ranks], device=device, dtype=torch.bool)
            if keep.sum() <= 1:
                continue
            boxes = boxes[keep]
            labels = labels[keep]
            ranks = torch.tensor([r for r in ranks if r is not None], device=device, dtype=boxes.dtype)
            fdis_kept = [f for f, ok in zip(fdis, keep.detach().cpu().tolist()) if ok]
            jaws = torch.tensor(
                [1 if (int(f) // 10) in (1, 2, 5, 6) else -1 for f in fdis_kept],
                device=device,
                dtype=torch.long,
            )
            dent = torch.tensor([1 if int(f) < 50 else -1 for f in fdis_kept], device=device, dtype=torch.long)
            size = boxes[..., 2:4].clamp_min(1e-4).prod(dim=-1).sqrt()

            for jaw in (-1, 1):
                for dentition in (-1, 1):
                    mask = (jaws == jaw) & (dent == dentition)
                    if mask.sum() <= 1:
                        continue
                    s = size[mask]
                    r = ranks[mask]
                    ds = s[:, None] - s[None, :]
                    dr = r[:, None] - r[None, :]
                    pair = dr.abs() > 0.08
                    if not pair.any():
                        continue
                    sign = dr.sign()
                    margin = 0.01 * dr.abs().clamp_max(0.5)
                    total = total + F.relu(-(ds * sign) + margin)[pair].mean()
                    groups += 1

        if groups == 0:
            return total
        return total / groups

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local', 'nwd'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local', 'nwd'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local', 'nwd'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'nwd'])
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # +++ MrDETR: O2M 分支损失（仅当模型输出含 mr_outputs 时） +++
        # 在 criterion.forward 里，移出 DN 分支外单独计算：
        if ('mr_outputs' in outputs) and (outputs['mr_outputs'] is not None):
            mr_logits_stack = outputs['mr_outputs']['logits']
            mr_boxes_stack  = outputs['mr_outputs']['boxes']
            if ('dn_meta' in outputs) and (outputs['dn_meta'] is not None):
                dn_q = int(outputs['dn_meta']['dn_num_split'][0])
                mr_logits_stack = mr_logits_stack[..., dn_q:, :]
                mr_boxes_stack  = mr_boxes_stack[...,  dn_q:, :]

            if (mr_logits_stack is not None) and (mr_boxes_stack is not None):####################################################修改！！！
                Lm = mr_logits_stack.shape[0]
                for li in range(Lm):
                    mr_logits = mr_logits_stack[li]   # [B,Q,C]
                    mr_boxes  = mr_boxes_stack[li]    # [B,Q,4]

                    # 1) O2M 匹配（分层单独做）
                    indices_mr = self.build_o2m_indices(
                        mr_logits, mr_boxes, targets,
                        topk=self.o2m_topk, iou_thr=self.o2m_iou_thr, cls_thr=self.o2m_cls_thr
                    )
                    # 在 MR 循环计算完 indices_mr 后（每层）新增：
                    # 统计同一 query 被匹配到的次数，>1 的部分加罚
                    # 2) 归一化项（匹配数量）
                    num_boxes_mr = sum(len(x[0]) for x in indices_mr)
                    num_boxes_mr = torch.as_tensor([num_boxes_mr], dtype=torch.float, device=mr_boxes.device)
                    if is_dist_available_and_initialized():
                        torch.distributed.all_reduce(num_boxes_mr)
                    num_boxes_mr = torch.clamp(num_boxes_mr / get_world_size(), min=1).item()

                    # 重复匹配的轻量惩罚（用 num_boxes_mr 归一化）
                    q_counts = torch.zeros(mr_logits.shape[:2], device=mr_logits.device, dtype=torch.float32)  # [B,Q]
                    for b, (src_idx, _) in enumerate(indices_mr):
                        if src_idx.numel()>0:
                            q_counts[b].index_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float32))
                    dup_penalty = (q_counts.clamp_min(1.0) - 1.0).sum() / max(1.0, num_boxes_mr)
                    losses[f'loss_mr_dedup_L{li}'] = 0.05 * dup_penalty


                    # 3) 组建“伪 outputs”复用现有 loss 实现
                    mr_out = {'pred_logits': mr_logits, 'pred_boxes': mr_boxes}
                    # self.weight_dict.update({
                    #     # 分类：选一种（和主干一致的 focal / vfl / mal）
                    #     'loss_focal_mr': 0.25,   # 或者 0.2
                    #     'loss_vfl_mr'  : 0.25,   # 只开其中一种
                    #     'loss_mal_mr'  : 0.0,    # 另一种置 0

                    #     # 回归
                    #     'loss_bbox_mr' : 0.25,
                    #     'loss_giou_mr' : 0.25,
                    # })

                    # 4) 逐项损失（不含 'local'）
                    if 'boxes' in self.mr_loss_on:
                        meta = self.get_loss_meta_info('boxes', mr_out, targets, indices_mr)
                        l_dict = self.loss_boxes(mr_out, targets, indices_mr, num_boxes_mr, **meta)
                        # 使用 weight_dict 中 *_mr 的权重；若未提供，则复用主干权重
                        w_l1  = self.weight_dict.get('loss_bbox_mr', self.weight_dict.get('loss_bbox', 1.0))
                        w_gIoU= self.weight_dict.get('loss_giou_mr', self.weight_dict.get('loss_giou', 1.0))
                        losses[f'loss_bbox_mr_L{li}'] = l_dict['loss_bbox'] * w_l1
                        losses[f'loss_giou_mr_L{li}'] = l_dict['loss_giou'] * w_gIoU

                    if 'focal' in self.mr_loss_on:
                        l_dict = self.loss_labels_focal(mr_out, targets, indices_mr, num_boxes_mr)
                        w_cls = self.weight_dict.get('loss_focal_mr', self.weight_dict.get('loss_focal', 1.0))
                        losses[f'loss_focal_mr_L{li}'] = l_dict['loss_focal'] * w_cls

                    if 'vfl' in self.mr_loss_on:
                        meta = self.get_loss_meta_info('vfl', mr_out, targets, indices_mr)
                        l_dict = self.loss_labels_vfl(mr_out, targets, indices_mr, num_boxes_mr, **meta)
                        w_cls = self.weight_dict.get('loss_vfl_mr', self.weight_dict.get('loss_vfl', 1.0))
                        losses[f'loss_vfl_mr_L{li}'] = l_dict['loss_vfl'] * w_cls

                    if 'mal' in self.mr_loss_on:
                        meta = self.get_loss_meta_info('mal', mr_out, targets, indices_mr)
                        l_dict = self.loss_labels_mal(mr_out, targets, indices_mr, num_boxes_mr, **meta)
                        w_cls = self.weight_dict.get('loss_mal_mr', self.weight_dict.get('loss_mal', 1.0))
                        losses[f'loss_mal_mr_L{li}'] = l_dict['loss_mal'] * w_cls
        # === Arch 顺序一致性损失（利用现成 indices，对最后一层输出）===
        if ('pred_boxes' in outputs) and ('pred_logits' in outputs) and len(indices) > 0:
            cxcy = outputs['pred_boxes'][..., :2]  # [B,Q,2] 归一化中心
            arch_loss, n_pair = 0.0, 0
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if tgt_idx.numel() == 0: 
                    continue
                # 依据 GT 的 y 将匹配对分上/下颌
                # tgt_boxes = targets[b]['boxes'][tgt_idx]          # [N,4] cxcywh
                # 先把 indices 搬到与 boxes 一致的设备
                dev = targets[b]['boxes'].device
                src_idx = src_idx.to(dev)
                tgt_idx = tgt_idx.to(dev)

                # 统一设备，防止 CPU/CUDA 混用
                dev = targets[b]['boxes'].device
                src_idx = src_idx.to(dev)
                tgt_idx = tgt_idx.to(dev)
                # 依据 GT 的 y 将匹配对分上/下颌
                tgt_boxes = targets[b]['boxes'][tgt_idx]          # [N,4] cxcywh
                is_upper  = (tgt_boxes[:,1] < 0.5)
                # 类别下标 -> FDI 逆映射
                from .utils_arch import fdi_index_map
                _map = fdi_index_map(self.num_classes)  # {FDI -> cls_id}
                id2fdi = {v: k for k, v in _map.items()}  # {cls_id -> FDI}

                for flag in (True, False):
                    sel = (is_upper == flag)
                    if not sel.any():
                        continue
                    p_idx  = src_idx[sel]
                    g_idx  = tgt_idx[sel]
                    p_xy   = cxcy[b, p_idx]                 # [n,2]
                    t_pred = fit_arch_t(p_xy)               # [n]

                    # 取该子集的 GT 类别下标并映射到 FDI
                    cls_ids = targets[b]['labels'][g_idx].tolist()
                    fdis    = [id2fdi.get(int(cid), None) for cid in cls_ids]

                    # 根据是否含乳牙选择序列
                    has_primary = any((f is not None) and (int(f) >= 50) for f in fdis)
                    seq = fdi_sequence_targets(flag, dentition=("primary" if has_primary else "perm"))

                    # 仅保留能映射且在该颌序列中的样本
                    keep = [(f is not None) and (f in seq) for f in fdis]
                    if not any(keep):
                        continue
                    keep_mask = torch.tensor(keep, device=p_xy.device, dtype=torch.bool)
                    p_xy   = p_xy[keep_mask]
                    fdis   = [f for f, k in zip(fdis, keep) if k]

                    # 等距目标：序列位置 → [0,1] 归一
                    pos   = torch.tensor([seq.index(int(f)) for f in fdis], device=p_xy.device, dtype=p_xy.dtype)
                    t_tgt = (pos + 0.5) / float(len(seq))

                    arch_loss = arch_loss + (fit_arch_t(p_xy) - t_tgt).abs().mean()
                    n_pair += 1

            if n_pair > 0:
                losses['loss_arch'] = arch_loss / n_pair * self.weight_dict['loss_arch']

        if self.use_fdi_anatomy_loss:
            anatomy_losses = self.loss_fdi_anatomy_final(outputs, targets, indices, num_boxes)
            for k, v in anatomy_losses.items():
                if k in self.weight_dict:
                    losses[k] = v * self.weight_dict[k]

        if self.use_tooth_size_prior_loss:
            losses['loss_tooth_size_prior'] = (
                self.loss_tooth_size_prior(outputs, targets, indices) *
                self.weight_dict['loss_tooth_size_prior']
            )

        if self.use_dsfed_dual_scale_kd:
            losses.update(self.loss_dsfed_dual_scale_kd(outputs))

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
