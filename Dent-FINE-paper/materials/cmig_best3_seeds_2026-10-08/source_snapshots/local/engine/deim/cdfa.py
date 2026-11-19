import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1, d=1, s=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=p, dilation=d, stride=s, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x

class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0., dilation=1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        k2 = kernel_size * kernel_size
        self.attn_fg = nn.Linear(dim, k2 * k2 * num_heads)
        self.attn_bg = nn.Linear(dim, k2 * k2 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        # keep pool for legacy but we'll prefer adaptive pooling to align with unfold
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(CBR(in_c, dim, 3, 1), CBR(dim, dim, 3, 1))
        self.output_cbr = nn.Sequential(CBR(dim, dim, 3, 1), CBR(dim, dim, 3, 1))

    def _calc_out_hw(self, H, W):
        """
        与 nn.Unfold 对应的输出高宽（floor 公式，支持 dilation）。
        out = floor((in + 2*pad - dilation*(k-1) -1)/stride) + 1
        """
        k = self.kernel_size
        p = self.padding
        s = self.stride
        d = self.dilation
        out_h = (H + 2 * p - d * (k - 1) - 1) // s + 1
        out_w = (W + 2 * p - d * (k - 1) - 1) // s + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Computed unfold output <=0: H={H},W={W},pad={p},k={k},s={s},d={d}, out_h={out_h}, out_w={out_w}")
        return int(out_h), int(out_w)

    # 向后兼容：有些地方调用 _calc_hw，我们把它指向 _calc_out_hw
    def _calc_hw(self, H, W):
        return self._calc_out_hw(H, W)

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        """
        feature_map: [B, H, W, C]
        返回 attn: [B, num_heads, L, k*k, k*k]  (L = out_h*out_w)
        """
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg

        # 1) 用与 unfold 相同公式计算 out_h/out_w
        out_h, out_w = self._calc_out_hw(H, W)
        L = out_h * out_w

        # 2) 使用 adaptive pool 保证与 unfold 对齐
        feat_perm = feature_map.permute(0, 3, 1, 2)  # [B, C, H, W]
        pooled = F.interpolate(feat_perm, size=(out_h, out_w), mode='area')
        feature_map_pooled = pooled.permute(0, 2, 3, 1)  # [B, out_h, out_w, C]

        # 3) 线性投影并 reshape 成 [B, num_heads, L, k*k, k*k]
        attn = attn_layer(feature_map_pooled).reshape(
            B, out_h * out_w, self.num_heads,
            self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size
        ).permute(0, 2, 1, 3, 4)  # -> [B, num_heads, L, k*k, k*k]

        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        """
        attn: [B, num_heads, L, k*k, k*k]
        v:    [B, num_heads, L, k*k, head_dim]  (注意这里 v 必须先按 unfold 得到)
        返回: folded 特征 [B, dim, H, W] (随后在 caller 中 permute-> [B,H,W,C] 再 proj)
        """
        # 1) matmul => [B, num_heads, L, k*k, head_dim]
        x_weighted = (attn @ v)

        # 2) permute -> [B, num_heads, head_dim, k*k, L]
        x_weighted = x_weighted.permute(0, 1, 4, 3, 2)

        # 3) reshape -> [B, dim * k*k, L]
        x_weighted = x_weighted.reshape(B, self.dim * self.kernel_size * self.kernel_size, -1)

        # 4) fold back to [B, dim, H, W]
        out = F.fold(x_weighted, output_size=(H, W),
                     kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)

        # 5) proj expects [B, H, W, C] -> do permute in caller
        out = self.proj(out.permute(0, 2, 3, 1))
        out = self.proj_drop(out)
        return out

    def forward(self, x, fg, bg):
        # x: [B, C_in, H, W] expected
        x = self.input_cbr(x).permute(0, 2, 3, 1)  # -> [B, H, W, C]
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        k, p, s = self.kernel_size, self.padding, self.stride
        h, w = self._calc_hw(H, W)
        L_expected = h * w

        # value tokens
        v = self.v(x).permute(0, 3, 1, 2)  # [B, C(dim), H, W]
        v_unf = self.unfold(v)            # [B, dim*k2, L_unf]
        # explicit L from unfold
        L_unf = v_unf.shape[-1]
        # 检查 unfold 的 channel 数与预期是否一致
        expected_ch = self.dim * (self.kernel_size * self.kernel_size)
        assert v_unf.shape[1] == expected_ch, f"[CDFA] unfold channels mismatch: expected {expected_ch}, got {v_unf.shape[1]}"

        # -> [B, num_heads, head_dim, k*k, L_unf] -> permute to [B, num_heads, L_unf, k*k, head_dim]
        v_unf = v_unf.view(B, self.num_heads, self.head_dim, k * k, L_unf).permute(0, 1, 4, 3, 2).contiguous()

        # 断言 L 对齐
        assert L_unf == L_expected, f"[CDFA] unfold L={L_unf} != expected {L_expected} (H={H},W={W},k={k},p={p},s={s})"

        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        # 再次断言，确保 attn 的 L 与 v_unf 一致
        assert attn_fg.shape[2] == v_unf.shape[2], f"[CDFA] attn_fg L={attn_fg.shape[2]} vs v_unf L={v_unf.shape[2]}"

        x_fg = self.apply_attention(attn_fg, v_unf, B, H, W, C)

        # stage 2: unfold the x_fg (currently x_fg is after apply_attention: it returned [B, H, W, dim] due to proj)
        # caller's apply_attention returns [B, H, W, dim] (proj output), but in our implementation apply_attention returns proj result already.
        # so ensure consistent shape: x_fg expected as [B, H, W, dim] -> to unfold need [B, dim, H, W]
        x_fg_unfold_input = x_fg.permute(0, 3, 1, 2)  # [B, dim, H, W]
        v2_unf = self.unfold(x_fg_unfold_input)  # [B, dim*k2, L2]
        L2 = v2_unf.shape[-1]
        v2 = v2_unf.view(B, self.num_heads, self.head_dim, k * k, L2).permute(0, 1, 4, 3, 2).contiguous()
        assert v2.shape[2] == L_expected, f"[CDFA] v2 L={v2.shape[2]} != expected {L_expected}"

        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        assert attn_bg.shape[2] == v2.shape[2], f"[CDFA] attn_bg L={attn_bg.shape[2]} vs v2 L={v2.shape[2]}"

        x_bg = self.apply_attention(attn_bg, v2, B, H, W, C)
        out = self.output_cbr(x_bg.permute(0, 3, 1, 2))  # -> [B, dim, H, W]
        return out
