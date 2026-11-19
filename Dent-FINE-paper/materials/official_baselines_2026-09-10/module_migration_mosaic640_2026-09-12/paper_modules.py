"""Minimal paper-module ports on top of the unmodified official DEIM scaffold.

Only CMFA, factor/prototype supervision, SpatialShape and DualOpt are included.
There is no runtime import from /home/wuxi/Dent-FINE. Native DEIM decoder layers,
nearest-neighbour FPN upsampling and sinusoidal encoder positions are retained.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from engine.core import register
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.postprocessor import PostProcessor


@register()
class DualOptAdamW(torch.optim.AdamW):
    """Paper's parameter-enumeration-rank regularizers after native AdamW."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, rollback_weight=0.0,
                 realtime_decay=0.0, layer_decay_power=1.5,
                 layer_decay_min=0.5, layer_decay_max=1.5):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self.rollback_weight = float(rollback_weight)
        self.realtime_decay = float(realtime_decay)
        self.layer_decay_power = float(layer_decay_power)
        self.layer_decay_min = float(layer_decay_min)
        self.layer_decay_max = float(layer_decay_max)
        self._dualopt_rank, self._dualopt_anchor = {}, {}
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        total = max(len(params) - 1, 1)
        for i, p in enumerate(params):
            self._dualopt_rank[p] = i / total
            if self.rollback_weight > 0:
                self._dualopt_anchor[p] = p.detach().clone(memory_format=torch.preserve_format)

    def _rank_scale(self, p):
        rank = float(self._dualopt_rank.get(p, 1.0))
        scale = self.layer_decay_min + (self.layer_decay_max - self.layer_decay_min) * rank**self.layer_decay_power
        return max(scale, 0.0)

    @torch.no_grad()
    def _dualopt_post_step(self):
        for group in self.param_groups:
            lr = float(group.get('lr', 0.0))
            if lr <= 0:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                scale = self._rank_scale(p)
                if self.realtime_decay > 0:
                    p.mul_(max(0.0, 1.0 - lr * self.realtime_decay * scale))
                if self.rollback_weight > 0:
                    anchor = self._dualopt_anchor.get(p)
                    if anchor is not None and anchor.shape == p.shape:
                        p.lerp_(anchor.to(device=p.device, dtype=p.dtype),
                                min(1.0, lr * self.rollback_weight * scale))

    def step(self, closure=None):
        loss = super().step(closure=closure)
        self._dualopt_post_step()
        return loss

    def state_dict(self):
        # Persist initialization anchors for faithful checkpoint/RAM restoration.
        # This does not change the original uninterrupted optimization updates.
        state = super().state_dict()
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        state['paper_dualopt'] = {
            'anchors': [self._dualopt_anchor.get(p) for p in params],
            'ranks': [self._dualopt_rank[p] for p in params],
            'settings': {k: getattr(self, k) for k in ('rollback_weight', 'realtime_decay',
                        'layer_decay_power', 'layer_decay_min', 'layer_decay_max')},
        }
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        extra = state.pop('paper_dualopt')
        result = super().load_state_dict(state)
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        assert len(params) == len(extra['anchors']) == len(extra['ranks'])
        for key, value in extra['settings'].items():
            setattr(self, key, value)
        for p, anchor, rank in zip(params, extra['anchors'], extra['ranks']):
            self._dualopt_rank[p] = rank
            if anchor is not None:
                assert p.shape == anchor.shape
                self._dualopt_anchor[p] = anchor.detach().to(device=p.device, dtype=p.dtype).clone()
        return result


class AdaptiveInterleaveFusion(nn.Module):
    """CMFA: the paper's equal-channel, concatenation-output path only."""
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        hidden = max((channels * 2) // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.SiLU(), nn.Conv2d(hidden, channels * 2, 1, bias=False), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, feat_a, feat_b):
        assert feat_a.shape == feat_b.shape
        gate_a, gate_b = self.channel_gate(torch.cat([feat_a, feat_b], 1)).split(self.channels, 1)
        fused = torch.cat([feat_a + feat_b * gate_b, feat_b + feat_a * gate_a], 1)
        spatial = torch.cat([fused.mean(1, keepdim=True), fused.max(1, keepdim=True).values], 1)
        return fused * self.spatial_gate(spatial)


class CMFAHybridEncoder(HybridEncoder):
    """Native encoder forward with only four concatenations replaced by CMFA."""
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            site = len(self.in_channels) - 1 - idx
            feat_heigh = self.lateral_convs[site](inner_outs[0])
            feat_low = proj_feats[idx - 1]
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            fused = self.top_down_fusions[site](upsample_feat, feat_low)
            inner_outs.insert(0, self.fpn_blocks[site](fused))
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsample_convs[idx](outs[-1])
            fused = self.bottom_up_fusions[idx](downsample_feat, inner_outs[idx + 1])
            outs.append(self.pan_blocks[idx](fused))
        return outs


class StructuredDFINETransformer(DFINETransformer):
    """Supervision taps; native query computation and main logits are unchanged."""
    def forward(self, feats, targets=None):
        captured = []
        handles = [layer.register_forward_hook(lambda _m, _i, out: captured.append(out))
                   for layer in self.decoder.layers]
        try:
            out = super().forward(feats, targets)
        finally:
            for handle in handles:
                handle.remove()
        main_count = out['pred_logits'].shape[1]
        dn_count = captured[-1].shape[1] - main_count
        assert dn_count >= 0
        main_feats = [x[:, dn_count:] for x in captured]
        last_index = len(captured) - 1

        def add_factors(item, feature, i):
            item.update(pred_quad_logits=self.dec_quad_head[i](feature),
                        pred_tooth_logits=self.dec_toothpos_head[i](feature),
                        pred_dent_logits=self.dec_dent_head[i](feature))

        add_factors(out, main_feats[-1], last_index)
        out['query_feats'] = main_feats[-1]
        features = F.normalize(main_feats[-1], dim=-1)
        prototypes = F.normalize(self.tooth_prototypes, dim=-1)
        similarity = torch.einsum('bqd,ckd->bqck', features, prototypes)
        out['pred_proto_logits'] = torch.logsumexp(similarity / self.prototype_tau, dim=-1)
        # No factor or prototype score fusion, graph/query rewriting or ROI path.
        for i, item in enumerate(out.get('aux_outputs', [])):
            add_factors(item, main_feats[i], i)
        for i, item in enumerate(out.get('dn_outputs', [])):
            assert dn_count > 0
            add_factors(item, captured[i][:, :dn_count], i)
        return out


class StructuredCriterion(DEIMCriterion):
    """Native matching/loss assembly plus the four paper auxiliary CE losses."""
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        keys = {'quad': 'pred_quad_logits', 'tooth': 'pred_tooth_logits',
                'dent': 'pred_dent_logits', 'proto': 'pred_proto_logits'}
        if loss not in keys:
            return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        logits = outputs.get(keys[loss])
        if logits is None:
            value = labels.float().sum() * 0.0
        elif labels.numel() == 0:
            value = logits.sum() * 0.0
        else:
            target = labels
            if loss != 'proto':
                fdi = self.fdi_codes[labels]
                target = {'quad': (fdi // 10 - 1).clamp(0, 7),
                          'tooth': (fdi % 10 - 1).clamp(0, 7),
                          'dent': (fdi < 50).long()}[loss]
            value = F.cross_entropy(logits[self._get_src_permutation_idx(indices)], target, reduction='mean')
        return {f'loss_{loss}': value}


class SpatialShapePostProcessor(PostProcessor):
    """Training-derived center/shape priors, followed by untouched native top-k."""
    @staticmethod
    def _soft_prior(score):
        return (score - score.amax(dim=-1, keepdim=True)).clamp(min=-6.0, max=0.0)

    def _score(self, coords, prefix):
        mu = getattr(self, prefix + '_mu').to(coords)
        inv_cov = getattr(self, prefix + '_inv_cov').to(coords)
        diff = coords[:, :, None, :] - mu[None, None, :, :]
        score = (-0.5 * torch.einsum('bqci,cij,bqcj->bqc', diff, inv_cov, diff)).clamp(-20., 0.)
        valid = getattr(self, prefix + '_valid')
        return torch.where(valid.view(1, 1, -1), score, torch.zeros_like(score))

    def forward(self, outputs, orig_target_sizes):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        spatial = self._score(boxes[..., :2], 'spatial')
        shape = self._score(boxes[..., 2:].clamp_min(1e-5).log(), 'shape')
        logits = logits.clone() + self.spatial_prior_weight * self._soft_prior(spatial)
        logits = logits + self.shape_prior_weight * self._soft_prior(shape)
        logits = logits + self.view_prior_weight * self.view_log_prior.to(logits).view(1, 1, -1)
        return super().forward({**outputs, 'pred_logits': logits}, orig_target_sizes)


def install_modules(cfg, specification):
    """Install only the frozen recipe, before EMA/optimizer construction."""
    model = cfg.model
    if specification['cmfa']:
        encoder = model.encoder
        assert type(encoder) is HybridEncoder
        encoder.__class__ = CMFAHybridEncoder
        count = len(encoder.in_channels) - 1
        encoder.top_down_fusions = nn.ModuleList([AdaptiveInterleaveFusion(encoder.hidden_dim) for _ in range(count)])
        encoder.bottom_up_fusions = nn.ModuleList([AdaptiveInterleaveFusion(encoder.hidden_dim) for _ in range(count)])
    if specification['structured']:
        decoder = model.decoder
        assert type(decoder) is DFINETransformer
        decoder.__class__ = StructuredDFINETransformer
        widths = [head.in_features for head in decoder.dec_score_head]
        decoder.dec_quad_head = nn.ModuleList([nn.Linear(d, 8) for d in widths])
        decoder.dec_toothpos_head = nn.ModuleList([nn.Linear(d, 8) for d in widths])
        decoder.dec_dent_head = nn.ModuleList([nn.Linear(d, 2) for d in widths])
        decoder.prototype_tau = specification['prototype_tau']
        decoder.tooth_prototypes = nn.Parameter(torch.empty(decoder.num_classes, specification['prototype_count'], widths[0]))
        nn.init.normal_(decoder.tooth_prototypes, mean=0.0, std=0.02)
        criterion = cfg.criterion
        criterion.__class__ = StructuredCriterion
        criterion.register_buffer('fdi_codes', torch.tensor(specification['fdi_codes'], dtype=torch.long))
    if specification['spatial_shape']:
        processor = cfg.postprocessor
        assert type(processor) is PostProcessor
        processor.__class__ = SpatialShapePostProcessor
        prior = json.loads(Path(specification['prior_file']).read_text())
        assert prior['fdi_label_order'] == specification['fdi_codes']
        assert prior['num_classes'] == processor.num_classes
        for key in ('spatial_mu', 'spatial_inv_cov', 'spatial_valid', 'shape_mu', 'shape_inv_cov', 'shape_valid'):
            processor.register_buffer(key, torch.tensor(prior[key], dtype=torch.bool if key.endswith('valid') else torch.float32))
        # Every permanent class is visible in the configured panorama view.
        assert specification['cohort'] == 'dentex' and specification['view_type'] == 'panorama'
        processor.register_buffer('view_log_prior', torch.zeros(processor.num_classes))
        processor.spatial_prior_weight = specification['spatial_prior_weight']
        processor.shape_prior_weight = specification['shape_prior_weight']
        processor.view_prior_weight = specification['view_prior_weight']
    return model


def audit_model(model, base_schema, specification):
    schema = {n: {'shape': list(p.shape), 'numel': p.numel(), 'trainable': p.requires_grad}
              for n, p in model.named_parameters()}
    assert all(schema.get(k) == v for k, v in base_schema.items()), 'Native parameter removed or changed'
    extra = {k: v for k, v in schema.items() if k not in base_schema}
    allowed = []
    if specification['cmfa']:
        allowed += ['encoder.top_down_fusions.', 'encoder.bottom_up_fusions.']
    if specification['structured']:
        allowed += ['decoder.dec_quad_head.', 'decoder.dec_toothpos_head.', 'decoder.dec_dent_head.', 'decoder.tooth_prototypes']
    assert all(any(k.startswith(prefix) for prefix in allowed) for k in extra), extra
    forbidden = ('queryfusion', 'query_fusion', 'dysample', 'row_embed', 'col_embed', 'mi_heads', 'querygraphprior')
    for name, module in model.named_modules():
        assert not any(term in (name + ' ' + type(module).__name__).lower() for term in forbidden)
    return {'native_parameters_preserved': True, 'unlisted_parameter_additions': [],
            'forbidden_modules_absent': True, 'additional_parameters': sum(v['numel'] for v in extra.values()),
            'params_training_total': sum(v['numel'] for v in schema.values()),
            'additional_parameter_schema': extra, 'parameter_schema': schema}
