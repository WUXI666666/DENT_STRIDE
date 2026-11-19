#!/usr/bin/env python3
"""Code/gradient equivalence checks, no formal training and no source mutation."""
from __future__ import annotations
import ast
import copy
import gc
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prepare_migration import HERE, ROOT, LOCAL, sha256, write_json
sys.path.insert(0, str(ROOT / 'upstream/DEIM'))
from engine.core import YAMLConfig
from engine.deim.hybrid_encoder import get_activation, ConvNormLayer_fuse
from paper_modules import DualOptAdamW, AdaptiveInterleaveFusion, install_modules, audit_model


def source_definition(file, name):
    # Execute just a selected definition in an isolated test namespace, never
    # import the old project and never bring its backbone/decoder into a run.
    tree = ast.parse(Path(file).read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    node.decorator_list = []
    namespace = {'torch': torch, 'optim': torch.optim, 'nn': nn, 'F': F, 'Dict': dict,
                 'get_activation': get_activation, 'ConvNormLayer_fuse': ConvNormLayer_fuse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), 'exec'), namespace)
    return namespace[name]


def assert_outputs_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f'Tensor mismatch: {a.shape}'
    elif isinstance(a, dict):
        for key, value in a.items():
            assert_outputs_equal(value, b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_outputs_equal(x, y)
    else:
        assert a == b


def main():
    torch.set_num_threads(4)
    torch.manual_seed(91)
    results = {}
    original_opt = source_definition(LOCAL / 'engine/optim/optim.py', 'DualOptAdamW')
    params1 = [nn.Parameter(torch.randn(3, 4, dtype=torch.float64)) for _ in range(4)]
    params2 = [nn.Parameter(x.detach().clone()) for x in params1]
    kwargs = dict(lr=0.0005, weight_decay=0.0001, rollback_weight=0.005, realtime_decay=0.01,
                  layer_decay_power=1.5, layer_decay_min=0.5, layer_decay_max=1.5)
    old_opt, new_opt = original_opt(params1, **kwargs), DualOptAdamW(params2, **kwargs)
    for step in range(8):
        for i, (a, b) in enumerate(zip(params1, params2)):
            grad = None if i == step % 4 else torch.randn_like(a)
            a.grad = grad
            b.grad = None if grad is None else grad.clone()
        old_opt.step(); new_opt.step()
        assert all(torch.equal(a, b) for a, b in zip(params1, params2))
    state = copy.deepcopy(new_opt.state_dict())
    resumed_params = [nn.Parameter(x.detach().clone()) for x in params2]
    resumed = DualOptAdamW(resumed_params, **kwargs)
    resumed.load_state_dict(state)
    for a, b in zip(params2, resumed_params):
        a.grad = torch.randn_like(a); b.grad = a.grad.clone()
    new_opt.step(); resumed.step()
    assert all(torch.equal(a, b) for a, b in zip(params2, resumed_params))
    results['DualOpt'] = '8 updates bitwise equal to original; anchor-aware restart next update bitwise equal'
    original_cmfa = source_definition(LOCAL / 'engine/deim/hybrid_encoder.py', 'AdaptiveInterleaveFusion')
    reference = original_cmfa(16, fuse_op='cat').double()
    port = AdaptiveInterleaveFusion(16).double()
    port.load_state_dict(reference.state_dict(), strict=True)
    inputs1 = [torch.randn(2, 16, 8, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    inputs2 = [x.detach().clone().requires_grad_() for x in inputs1]
    a, b = reference(*inputs1), port(*inputs2)
    assert torch.equal(a, b)
    a.square().sum().backward(); b.square().sum().backward()
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(inputs1, inputs2))
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(reference.parameters(), port.parameters()))
    results['CMFA'] = 'Forward, input gradients and parameter gradients bitwise equal to original cat path'
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for task, frozen in protocol['runs'].items():
        spec = frozen['modules']
        torch.manual_seed(0)
        cfg = YAMLConfig(frozen['config'], HGNetv2={'pretrained': False})
        base = cfg.model
        native = copy.deepcopy(base)
        before = {k: v.detach().clone() for k, v in base.state_dict().items()}
        model = install_modules(cfg, spec)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        profile = json.loads((ROOT / 'profiles' / f"deim_s_{spec['cohort']}.json").read_text())
        audit = audit_model(model, profile['parameter_schema'], spec)
        write_json(HERE / 'validation' / f'{task}_architecture.json', audit)
        native.eval(); model.eval()
        images = torch.randn(1, 3, 640, 640)
        if spec['cmfa']:
            class CatOnly(nn.Module):
                def forward(self, a, b):
                    return torch.cat([a, b], 1)
            bypass = copy.deepcopy(model)
            bypass.encoder.top_down_fusions = nn.ModuleList([CatOnly(), CatOnly()])
            bypass.encoder.bottom_up_fusions = nn.ModuleList([CatOnly(), CatOnly()])
            with torch.no_grad():
                assert_outputs_equal(native(images), bypass(images))
            del bypass
        else:
            with torch.no_grad():
                assert_outputs_equal(native(images), model(images))
        # Native class-agnostic/hierarchical loss assembly and DN split handling.
        model.train(); native.train()
        images = torch.randn(2, 3, 160, 160)
        targets = [{'labels': torch.tensor([0, 9, 20]),
                    'boxes': torch.tensor([[.2,.3,.1,.2],[.5,.4,.12,.15],[.7,.65,.15,.22]])} for _ in range(2)]
        torch.manual_seed(44)
        native_out = native(images, targets)
        torch.manual_seed(44)
        out = model(images, targets)
        if not spec['cmfa']:
            assert_outputs_equal(native_out, out)
        criterion = cfg.criterion
        losses = criterion(out, targets)
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
        extras = audit['additional_parameter_schema']
        active = {k: p for k, p in model.named_parameters() if k in extras}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in active.values())
        if spec['structured']:
            assert len(out['dn_outputs']) == 3
            assert all(out['dn_outputs'][i]['pred_quad_logits'].shape[1] == out['dn_meta']['dn_num_split'][0] for i in range(3))
            assert all(losses[k] > 0 for k in ['loss_quad', 'loss_tooth', 'loss_dent', 'loss_proto'])
        if spec['spatial_shape']:
            original = source_definition(LOCAL / 'engine/deim/dental_prior.py', 'spatial_template_score')
            original_shape = source_definition(LOCAL / 'engine/deim/dental_prior.py', 'shape_template_score')
            processor = cfg.postprocessor
            prior = {k: getattr(processor, k) for k in ('spatial_mu', 'spatial_inv_cov', 'spatial_valid', 'shape_mu', 'shape_inv_cov', 'shape_valid')}
            boxes = out['pred_boxes'].detach()
            assert torch.equal(original(boxes, prior), processor._score(boxes[..., :2], 'spatial'))
            assert torch.equal(original_shape(boxes, prior), processor._score(boxes[..., 2:].clamp_min(1e-5).log(), 'shape'))
            logits = out['pred_logits'].detach()
            adjusted = logits.clone() + .2 * processor._soft_prior(original(boxes, prior))
            adjusted = adjusted + .05 * processor._soft_prior(original_shape(boxes, prior))
            from engine.deim.postprocessor import PostProcessor
            native_processor = PostProcessor(num_classes=32, num_top_queries=300)
            sizes = torch.tensor([[160,160],[160,160]])
            assert_outputs_equal(processor({'pred_boxes': boxes, 'pred_logits': logits}, sizes),
                                 native_processor({'pred_boxes': boxes, 'pred_logits': adjusted}, sizes))
        results[task] = {'native_initial_parameters_unchanged': True,
                         'native_outputs_equal_without_active_CMFA': True,
                         'all_added_parameters_have_finite_nonzero_gradients': True,
                         'params_total': audit['params_training_total'], 'additional_params': audit['additional_parameters']}
        del native, model, cfg, out, native_out, losses, total, before
        gc.collect()
    write_json(HERE / 'unit_validation.json', {'passed': True, 'checks': results})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
