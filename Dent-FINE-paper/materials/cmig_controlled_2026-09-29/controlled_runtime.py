"""Scoped training-time ablations. Original project and frozen source remain untouched."""
import torch
import torch.nn.functional as F


def install_controls(settings):
    from engine.deim import hybrid_encoder as he
    from engine.deim import dfine_decoder as dd
    channel = settings.get('cmfa_channel', True)
    spatial = settings.get('cmfa_spatial', True)
    high = settings.get('hooa_highpass', True)
    second = settings.get('hooa_second_order', True)
    infer_fusion = settings.get('factor_inference_fusion', True)

    if not (channel and spatial):
        def cmfa_forward(self, a, b):
            fused = torch.cat([a, b], dim=1)
            if channel:
                ga, gb = self.channel_gate(fused).split(self.channels, dim=1)
                fused = torch.cat([a + b * gb, b + a * ga], dim=1)
            if spatial:
                stat = torch.cat([fused.mean(1, keepdim=True), fused.max(1, keepdim=True).values], 1)
                fused = fused * self.spatial_gate(stat)
            return self.out_proj(fused)
        he.AdaptiveInterleaveFusion.forward = cmfa_forward

    if not (high and second):
        def hooa_forward(self, x):
            local = self.local_branch(x)
            context = self.context_branch(x)
            edge = self.edge_branch(x - F.avg_pool2d(x, 3, 1, 1)) if high else torch.zeros_like(local)
            interaction = local * context.sigmoid() if second else context
            fused = self.out_proj(torch.cat([local, interaction, edge], dim=1))
            stat = torch.cat([fused.mean(1, keepdim=True), fused.max(1, keepdim=True).values], 1)
            fused = fused * self.channel_gate(fused) * self.spatial_gate(stat)
            return x + .30 * self.res_scale.sigmoid() * fused
        he.HighOrderOmniAttention.forward = hooa_forward

    if not infer_fusion:
        original = dd.TransformerDecoder.forward
        def decoder_forward(self, *args, **kwargs):
            # Only the pure inference switch changes; training uses the native
            # factor_fusion_train flag and original alpha weights.
            if self.training:
                return original(self, *args, **kwargs)
            names = ['factor_alpha_quad', 'factor_alpha_tooth', 'factor_alpha_dent', 'fdi_factor_logit_scale']
            old = {n: getattr(self, n) for n in names if hasattr(self, n)}
            try:
                for n in old: setattr(self, n, 0.)
                return original(self, *args, **kwargs)
            finally:
                for n, value in old.items(): setattr(self, n, value)
        dd.TransformerDecoder.forward = decoder_forward


def validate_modules():
    """Check full-setting equivalence and every gate combination with gradients."""
    from engine.deim import hybrid_encoder as he
    original_cmfa = he.AdaptiveInterleaveFusion.forward
    original_hooa = he.HighOrderOmniAttention.forward
    torch.manual_seed(29)
    a, b = torch.randn(2,32,12,12), torch.randn(2,32,12,12)
    cmfa = he.AdaptiveInterleaveFusion(32).eval()
    hooa = he.HighOrderOmniAttention(32).eval()
    reference_c = cmfa(a,b).detach()
    reference_h = hooa(a).detach()
    install_controls({})
    assert torch.equal(reference_c, cmfa(a,b))
    assert torch.equal(reference_h, hooa(a))
    rows = []
    for c,s in [(False,False),(True,False),(False,True),(True,True)]:
        he.AdaptiveInterleaveFusion.forward=original_cmfa
        install_controls({'cmfa_channel':c,'cmfa_spatial':s})
        cmfa.zero_grad();x=a.clone().requires_grad_();y=b.clone().requires_grad_()
        z=cmfa(x,y);assert z.shape==(2,64,12,12)
        z.square().mean().backward();assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
        if not c and not s: assert torch.equal(z.detach(),torch.cat([a,b],1))
        rows.append({'module':'CMFA','channel':c,'spatial':s,'finite_backward':True})
    he.AdaptiveInterleaveFusion.forward=original_cmfa
    for h,o in [(False,False),(True,False),(False,True),(True,True)]:
        he.HighOrderOmniAttention.forward=original_hooa
        install_controls({'hooa_highpass':h,'hooa_second_order':o})
        hooa.zero_grad();x=a.clone().requires_grad_();z=hooa(x)
        assert z.shape==x.shape;z.square().mean().backward();assert torch.isfinite(x.grad).all()
        assert any(p.grad is not None for p in hooa.context_branch.parameters())
        if not h: assert all(p.grad is None for p in hooa.edge_branch.parameters())
        rows.append({'module':'HOOA','highpass':h,'second_order':o,'finite_backward':True})
    he.HighOrderOmniAttention.forward=original_hooa
    return {'default_forward_bitwise_equal':True,'tests':rows}
