"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ['AdamW', 'DualOptAdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


@register()
class DualOptAdamW(optim.AdamW):
    """
    DualOpt/OLOR-inspired AdamW variant for fine-tuning ablations.

    It keeps AdamW as the base optimizer and adds two lightweight post-step
    regularizers: real-time layer-wise decay and a small rollback toward the
    initialization snapshot. This mirrors the useful fine-tuning idea without
    changing the solver loop or requiring a separate optimizer schedule.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        rollback_weight=0.0,
        realtime_decay=0.0,
        layer_decay_power=1.5,
        layer_decay_min=0.5,
        layer_decay_max=1.5,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.rollback_weight = float(rollback_weight)
        self.realtime_decay = float(realtime_decay)
        self.layer_decay_power = float(layer_decay_power)
        self.layer_decay_min = float(layer_decay_min)
        self.layer_decay_max = float(layer_decay_max)
        self._dualopt_rank = {}
        self._dualopt_anchor = {}
        self._init_dualopt_state()

    @torch.no_grad()
    def _init_dualopt_state(self):
        params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        total = max(len(params) - 1, 1)
        for idx, p in enumerate(params):
            self._dualopt_rank[p] = idx / total
            if self.rollback_weight > 0:
                self._dualopt_anchor[p] = p.detach().clone(memory_format=torch.preserve_format)

    def _rank_scale(self, p):
        rank = float(self._dualopt_rank.get(p, 1.0))
        scale = self.layer_decay_min + (self.layer_decay_max - self.layer_decay_min) * (rank ** self.layer_decay_power)
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
                        p.lerp_(anchor.to(device=p.device, dtype=p.dtype), min(1.0, lr * self.rollback_weight * scale))

    def step(self, closure=None):
        loss = super().step(closure=closure)
        self._dualopt_post_step()
        return loss


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
