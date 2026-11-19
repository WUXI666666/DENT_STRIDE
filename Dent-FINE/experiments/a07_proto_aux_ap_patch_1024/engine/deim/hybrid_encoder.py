"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation

from ..core import register

__all__ = ['HybridEncoder']

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    _WAVELETS_AVAILABLE = True
except ImportError:
    DWTForward = DWTInverse = None
    _WAVELETS_AVAILABLE = False


def log_shape(name, x, verbose=False):
    if verbose:
        print(f"{name}: {tuple(x.shape)}")


class _ConvDWT(nn.Module):
    def __init__(self, wave='haar', mode='zero'):
        super().__init__()
        if not _WAVELETS_AVAILABLE:
            raise RuntimeError(
                "pytorch_wavelets 未安装，无法使用 WavEnhance。请先安装该依赖。"
            )
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            xf = x.float() if x.dtype != torch.float32 else x
            yl, yh = self.dwt(xf)
        b, c, h, w = x.shape
        yh = yh[0].transpose(1, 2).reshape(b, -1, yh[0].shape[3], yh[0].shape[4])
        out = torch.cat([yl, yh], dim=1)
        return F.interpolate(out, size=(h // 2, w // 2), mode='bilinear', align_corners=False)


class _ConvIDWT(nn.Module):
    def __init__(self, wave='haar', mode='zero'):
        super().__init__()
        if not _WAVELETS_AVAILABLE:
            raise RuntimeError("pytorch_wavelets 未安装，无法使用 WavEnhance。")
        self.idwt = DWTInverse(wave=wave, mode=mode)

    def forward(self, ll, yh):
        b, c, h, w = ll.shape
        yh = yh.reshape(b, c, 3, h, w)
        device_type = 'cuda' if ll.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            rec = self.idwt((ll.float(), [yh.float()]))
        return F.interpolate(rec, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class WavEnhance(nn.Module):
    def __init__(self, in_channels, wave='haar', mode='symmetric',
                 residual_alpha=0.15, detail_alpha=0.2):
        super().__init__()
        self.dwt = _ConvDWT(wave=wave, mode=mode)
        self.idwt = _ConvIDWT(wave=wave, mode=mode)
        self.attention = _SpatialAttention(kernel_size=7)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_alpha)))
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_alpha)))
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape
        dwt_out = self.dwt(x)
        ll, yh = dwt_out[:, :c], dwt_out[:, c:]
        att = self.attention(ll)

        detail_scale = self.detail_scale.clamp(0.0, 0.5)
        yh_refined = yh * (1.0 - detail_scale + detail_scale * att)

        threshold = self.threshold.abs()
        yh_refined = torch.sign(yh_refined) * F.relu(yh_refined.abs() - threshold)

        enhanced = self.idwt(ll, yh_refined)
        if enhanced.shape[2:] != (h, w):
            enhanced = F.interpolate(enhanced, size=(h, w), mode='bilinear', align_corners=False)

        residual_scale = self.residual_scale.clamp(0.0, 0.3)
        return x + residual_scale * (enhanced - x)


class StatisticalGuidedDenoising(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.stat_pool = nn.AdaptiveAvgPool2d(1)
        self.stat_net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        stats = self.stat_pool(x)
        stats_weight = self.stat_net(stats)
        noise = self.noise_estimator(x)
        denoised = x - noise * (1 - stats_weight)
        fused = torch.cat([x, denoised], dim=1)
        alpha = 0.25 * torch.sigmoid(self.res_scale)
        return x + alpha * self.fusion(fused)


class MultiPathFeatureFusion(nn.Module):
    def __init__(self, in_channels=256, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.spatial_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.channel_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.context_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        spatial = self.spatial_path(x)
        channel_feat = x * self.channel_path(x)
        context = self.context_path(x)
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * spatial + weights[1] * channel_feat + weights[2] * context
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * fused


class DynamicReceptiveField(nn.Module):
    def __init__(self, in_channels=256, num_scales=3, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        hidden = max(8, in_channels // reduction_ratio)
        dilations = [1, 2, 4][:num_scales]

        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3,
                    padding=dilation, dilation=dilation,
                    groups=max(1, in_channels // 32), bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),
            )
            for dilation in dilations
        ])
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * num_scales, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_scales, 1),
            nn.Softmax(dim=1),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid(),
        )
        self.fusion_weight = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        b, _, _, _ = x.shape
        multi_scale_features = [conv(x) for conv in self.multi_scale_convs]
        concat_features = torch.cat(multi_scale_features, dim=1)
        scale_weights = self.scale_attention(concat_features).view(b, self.num_scales, 1, 1, 1)
        stacked_features = torch.stack(multi_scale_features, dim=1)
        fused = (stacked_features * scale_weights).sum(dim=1)
        fused = fused * self.channel_attention(fused)
        alpha = 0.35 * torch.sigmoid(self.fusion_weight)
        return x + alpha * fused

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        assert in_channels == out_channels, \
            "For HFSA we expect in_channels == out_channels"
        self.groups = groups
        # 频域卷积：处理 [B, 2C, H, W/2+1]
        self.conv_layer = nn.Conv2d(
            in_channels * 2,
            out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        orig_dtype = x.dtype

        # 所有 FFT/复数操作用 FP32，禁用 AMP（区分 CPU / CUDA）
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            ffted = torch.fft.rfft2(x32, norm='ortho')  # complex64, [B,C,H,W/2+1]

            ffted_ri = torch.view_as_real(ffted)                      # [B,C,H,W/2+1,2]
            ffted_ri = ffted_ri.permute(0, 1, 4, 2, 3).contiguous()   # [B,C,2,H,W/2+1]
            ffted_ri = ffted_ri.view(B, 2 * C, H, ffted_ri.size(-1))  # [B,2C,H,W/2+1]

            ffted_ri = self.conv_layer(ffted_ri)
            ffted_ri = self.relu(self.bn(ffted_ri))

            ffted_ri = ffted_ri.view(B, C, 2, H, ffted_ri.size(-1)) \
                               .permute(0, 1, 3, 4, 2).contiguous()
            ffted_c  = torch.view_as_complex(ffted_ri)                 # complex64
            out32    = torch.fft.irfft2(ffted_c, s=(H, W), norm='ortho')  # fp32

        return out32.to(orig_dtype)

class FBMLite(nn.Module):
    """
    FBM-lite: 简化版 FrequencyBandModulation
    - 多个环形低通掩码，把频域分成多个频带
    - 每个频带的高频残差 high_part 用一个 depthwise conv 生成的 spatial weight 做调制
    - 最后把所有频带 + 最低频残差加起来
    说明：
    - 为了简化，这里不做预缓存 mask，而是根据 H,W 动态生成频率网格
    - 所有 FFT 相关操作强制用 float32，并禁用 AMP
    """
    def __init__(
        self,
        in_channels: int,
        k_list=(2, 4),
        spatial_kernel: int = 3,
        spatial_group: int = 1,
        lowfreq_att: bool = False,
        act: str = "sigmoid",
    ):
        super().__init__()
        assert in_channels % spatial_group == 0, \
            f"in_channels={in_channels} 不能被 spatial_group={spatial_group} 整除"
        self.in_channels = in_channels
        self.k_list = sorted(list(k_list))
        self.spatial_kernel = spatial_kernel
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        self.act = act

        n_convs = len(self.k_list) + (1 if self.lowfreq_att else 0)
        self.freq_weight_convs = nn.ModuleList()
        for _ in range(n_convs):
            conv = nn.Conv2d(
                in_channels,
                spatial_group,
                kernel_size=spatial_kernel,
                stride=1,
                padding=spatial_kernel // 2,
                groups=spatial_group,
                bias=True,
            )
            nn.init.normal_(conv.weight, std=1e-6)
            if conv.bias is not None:
                conv.bias.data.zero_()
            self.freq_weight_convs.append(conv)

    def _sp_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "sigmoid":
            return torch.sigmoid(x) * 2
        elif self.act == "tanh":
            return 1.0 + torch.tanh(x)
        elif self.act == "softmax":
            # 沿着 group 维度做 softmax（注意这里只是一个可选模式）
            return F.softmax(x, dim=1) * x.shape[1]
        else:
            return x

    def _build_radius(self, H: int, Wf: int, device, dtype):
        """
        构建频率半径矩阵 radius[h, w_f] ~ sqrt(f_h^2 + f_w^2) in [0, 0.5]
        H: 空间高度
        Wf: 频域宽度 (W//2 + 1)
        """
        # 这里用 fftfreq / rfftfreq 来和 FFT 对齐
        freq_h = torch.fft.fftfreq(H, device=device)
        freq_w = torch.fft.rfftfreq((Wf - 1) * 2, device=device)  # W = 2*(Wf-1)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.sqrt(grid_h ** 2 + grid_w ** 2).to(dtype)
        return radius  # [H, Wf]

    def forward(self, x: torch.Tensor, att_feat: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B,C,H,W]
        att_feat: 默认用 x，本质是生成 freq_weight 的特征
        """
        if att_feat is None:
            att_feat = x

        B, C, H, W = x.shape
        orig_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"

        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            att32 = att_feat.float()

            # FFT 到频域
            x_fft = torch.fft.rfft2(x32, norm="ortho")  # [B,C,H,Wf], complex
            _, _, _, Wf = x_fft.shape

            radius = self._build_radius(H, Wf, x32.device, x32.dtype)  # [H, Wf]

            pre_x = x32  # 用来做多频带残差
            out_list = []

            g = self.spatial_group
            cg = C // g

            # 逐频带处理高频残差
            for idx, k in enumerate(self.k_list):
                thr = 0.5 / float(k) + 1e-8
                mask = (radius < thr).to(x32.dtype)              # [H, Wf]
                mask = mask.view(1, 1, H, Wf)                   # [1,1,H,Wf]

                low = torch.fft.irfft2(x_fft * mask, s=(H, W), norm="ortho")   # [B,C,H,W]
                high = pre_x - low
                pre_x = low

                # 空间调制权重
                freq_weight = self.freq_weight_convs[idx](att32)  # [B,G,H,W]
                freq_weight = self._sp_act(freq_weight)

                # [B,G,1,H,W] * [B,G,C/G,H,W] -> [B,C,H,W]
                fw = freq_weight.view(B, g, 1, H, W)
                hp = high.view(B, g, cg, H, W)
                tmp = (fw * hp).view(B, C, H, W)
                out_list.append(tmp)

            # 最低频部分：要么也加注意力，要么直接加
            if self.lowfreq_att:
                idx = len(self.k_list)
                freq_weight = self.freq_weight_convs[idx](att32)  # [B,G,H,W]
                freq_weight = self._sp_act(freq_weight)
                fw = freq_weight.view(B, g, 1, H, W)
                lp = pre_x.view(B, g, cg, H, W)
                tmp = (fw * lp).view(B, C, H, W)
                out_list.append(tmp)
            else:
                out_list.append(pre_x)

            out32 = sum(out_list)  # [B,C,H,W]

        return out32.to(orig_dtype)


class HighPassFreqBranch(nn.Module):
    """
    高通残差分支（模仿 FreqFusion 里的高频增强部分）：
    - 先用 depthwise blur 得到低频
    - high = x - low
    - 用一个轻量 depthwise + 1x1 处理 high
    - 最后：x + hf_gain * high
    """
    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.k = k
        # 简单的 depthwise blur（可以初始化成平均卷积，也可以学）
        self.lowpass = nn.Conv2d(
            dim, dim, kernel_size=k, padding=k // 2,
            groups=dim, bias=False
        )
        # 初始化为均值滤波（可选：也可以改成 xavier）
        with torch.no_grad():
            self.lowpass.weight.data.fill_(1.0 / (k * k))

        # 对 high 做一点非线性处理
        self.proc = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        # 高频增益，初始为 0：一开始几乎不动 backbone
        self.hf_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里不需要关 AMP / 复数，都是实数卷积
        low = self.lowpass(x)
        high = x - low                   # 明确的高频残差信号
        high = self.proc(high)
        return x + self.hf_gain * high   # 类似 FreqFusion 的设计


class StripBlock(nn.Module):
    """
    针对牙齿排列优化的条形卷积模块。
    牙齿是水平排列的，且单颗牙齿是垂直长条状的。
    """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        # 水平分支：捕捉牙列走向 (Horizontal Dental Arch)
        self.conv_h = nn.Sequential(
            nn.Conv2d(dim, dim, (1, kernel_size), padding=(0, padding), groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        # 垂直分支：捕捉牙根/牙体形状 (Vertical Tooth Shape)
        self.conv_v = nn.Sequential(
            nn.Conv2d(dim, dim, (kernel_size, 1), padding=(padding, 0), groups=dim, bias=False),
            nn.BatchNorm2d(dim)
        )
        # 1x1 融合
        self.conv_mix = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        x_h = self.conv_h(x)
        x_v = self.conv_v(x)
        return self.conv_mix(x_h + x_v)
class DySample(nn.Module):
    """
    DySample: Learning to Upsample by Learning to Sample
    超轻量动态上采样，比 Nearest 清晰得多，比 Deconv 便宜。
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= groups * scale**2
            in_channels = in_channels // scale**2
        out_channels = 2 * groups * scale**2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.offset.weight, std=1e-6)
        nn.init.constant_(self.offset.bias, 0)

    def forward(self, x):
        import torch.nn.functional as F
        B, C, H, W = x.shape
        offset = self.offset(x)
        if self.style == 'pl':
            x = F.pixel_shuffle(x, self.scale)
            offset = F.pixel_shuffle(offset, self.scale)
        elif self.style == 'lp':
            offset = F.pixel_shuffle(offset, self.scale)
        
        # Grid sample logic
        norm = torch.tensor([[[[W, H]]]], dtype=x.dtype, device=x.device)
        grid = torch.meshgrid(torch.arange(H*self.scale, device=x.device), 
                              torch.arange(W*self.scale, device=x.device), indexing='ij')
        grid = torch.stack(grid, dim=-1).unsqueeze(0).expand(B, -1, -1, -1).float() # [B, H*s, W*s, 2]
        
        # Adding offset
        offset = offset.permute(0, 2, 3, 1).reshape(B, H*self.scale, W*self.scale, self.groups, 2)
        grid = grid.unsqueeze(3) + offset # [B, H, W, g, 2]
        grid = grid.mean(dim=3) # Average groups to get final sampling pos
        
        # Normalize to [-1, 1]
        grid = 2.0 * grid / (norm * self.scale - 1) - 1.0
        
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

class HFSA(nn.Module):
    """
    HFSA v4: FBMLite (Frequency) + Strip Conv (Spatial) + SK Fusion
    专为牙科全景图设计。
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        
        # === 1. 频域分支 (保持你原本优秀的 FBMLite) ===
        self.freq_branch = FBMLite(
            in_channels=dim,
            k_list=(2, 4), 
            spatial_group=dim, 
            lowfreq_att=True
        )
        
        # === 2. [改进] 条形卷积空间分支 ===
        # 替换掉了原来的 Dilated Conv
        self.spatial_branch = StripBlock(dim, kernel_size=11) # 11的感受野足够覆盖3-4颗牙
        
        # === 3. 融合模块 (不变) ===
        self.fusion_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2 * dim, 1, bias=False)
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        feat_freq = self.freq_branch(x)         
        feat_spatial = self.spatial_branch(x)
        
        # SK-Fusion
        feat_sum = feat_freq + feat_spatial
        attn_weights = self.fusion_fc(feat_sum)
        attn_weights = attn_weights.reshape(B, 2, C, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=1) 
        
        x_fused = feat_freq * attn_weights[:, 0] + feat_spatial * attn_weights[:, 1]
        
        out = self.out_proj(x_fused)
        return residual + torch.tanh(self.res_scale) * out


class FreqAdapter(nn.Module):
    """
    优化后的频域适配器
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        # 简化为两个频带：低频保持，高频调制
        # 这样网络更容易学习，不需要手动调那么多阈值
        self.threshold = 0.5 # 归一化频率半径
        
        # 用轻量级注意力调制高频部分
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局上下文辅助
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 简单的深度卷积处理频率图转换回来的特征
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. FFT
        x_float = x.float()
        x_fft = torch.fft.rfft2(x_float, norm='ortho')
        
        freq_h = torch.fft.fftfreq(H, device=x.device)
        freq_w = torch.fft.rfftfreq(W, device=x.device)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        radius = torch.sqrt(grid_h**2 + grid_w**2)
        
        # 2. 软 Mask (比硬阈值更容易反向传播梯度)
        # Sigmoid 使得过渡更平滑
        mask_high = torch.sigmoid((radius - self.threshold) * 10).unsqueeze(0).unsqueeze(0) # [1,1,H,W//2+1]
        mask_low = 1.0 - mask_high
        
        # 3. 分离
        x_low_fft = x_fft * mask_low
        x_high_fft = x_fft * mask_high
        
        # 4. 逆变换
        x_low = torch.fft.irfft2(x_low_fft, s=(H, W), norm='ortho').type_as(x)
        x_high = torch.fft.irfft2(x_high_fft, s=(H, W), norm='ortho').type_as(x)
        
        # 5. 调制高频 (Texture Enhancement)
        # 使用通道注意力来决定增强哪些通道的高频信息
        mod_weight = self.modulator(x)
        x_high_enhanced = x_high * mod_weight
        
        # 6. 融合
        out = x_low + x_high_enhanced
        return self.post_process(out)
# class HFSA(nn.Module):
#     """
#     统一的频域-空域混合注意力（借鉴FDConv思路）
#     - 用频域自适应卷积替代复杂的FBM+高通滤波
#     - 保留通道注意力和位置注意力
#     """
#     def __init__(self, dim, reduction=8):
#         super().__init__()
#         self.dim = dim
        
#         # === 频域分支（参考FDConv简化版）===
#         self.freq_adapter = FreqAdapter(
#             in_channels=dim,
#             kernel_num=3,  # 低、中、高频三个自适应核
#             reduction=reduction
#         )
        
#         # === 空域分支（保持原逻辑）===
#         self.spatial_conv3 = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
#             nn.GELU()
#         )
#         self.spatial_conv5 = nn.Sequential(
#             nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
#             nn.GELU()
#         )
        
#         # === 通道注意力 ===
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // reduction, 1),
#             nn.GELU(),
#             nn.Conv2d(dim // reduction, dim, 1),
#             nn.Sigmoid()
#         )
        
#         # === 位置注意力（简化版）===
#         self.pos_query = nn.Conv2d(dim, dim // 8, 1)
#         self.pos_key = nn.Conv2d(dim, dim // 8, 1)
#         self.pos_value = nn.Conv2d(dim, dim, 1)
#         self.pos_gamma = nn.Parameter(torch.zeros(1))
        
#         # === 三分支自适应融合 ===
#         self.fusion_gate = nn.Sequential(
#             nn.Conv2d(dim * 3, dim, 1),
#             nn.GELU(),
#             nn.Conv2d(dim, 3, 1),
#             nn.Softmax(dim=1)
#         )
        
#         # === 输出投影 ===
#         self.out_proj = nn.Conv2d(dim, dim, 1)
#         self.out_norm = nn.LayerNorm(dim)
#         self.res_scale = nn.Parameter(torch.zeros(1))
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         residual = x
        
#         # 1) 频域分支（替代原来的FBM+高通滤波）
#         x_freq = self.freq_adapter(x)
        
#         # 2) 空域分支
#         x_sp3 = self.spatial_conv3(x)
#         x_sp5 = self.spatial_conv5(x)
#         x_spatial = x_sp3 + x_sp5
        
#         # 3) 位置注意力
#         q = self.pos_query(x).view(B, -1, H * W).permute(0, 2, 1)
#         k = self.pos_key(x).view(B, -1, H * W)
#         v = self.pos_value(x).view(B, -1, H * W)
        
#         attn = torch.bmm(q, k) / (q.shape[-1] ** 0.5)
#         attn = F.softmax(attn, dim=-1)
        
#         x_pos = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
#         x_pos = self.pos_gamma * x_pos + x
        
#         # 4) 三分支自适应融合
#         fusion_input = torch.cat([x_freq, x_spatial, x_pos], dim=1)
#         gate_weights = self.fusion_gate(fusion_input)
        
#         x_fused = (
#             gate_weights[:, 0:1] * x_freq +
#             gate_weights[:, 1:2] * x_spatial +
#             gate_weights[:, 2:3] * x_pos
#         )
        
#         # 5) 通道注意力
#         ca_weight = self.channel_attn(x_fused)
#         x_fused = x_fused * ca_weight
        
#         # 6) 输出 + 残差
#         out = self.out_proj(x_fused)
#         out = out.permute(0, 2, 3, 1)
#         out = self.out_norm(out)
#         out = out.permute(0, 3, 1, 2)
        
#         scale = torch.tanh(self.res_scale)
#         return residual + scale * out


# class FreqAdapter(nn.Module):
#     """
#     频域自适应卷积核（参考FDConv，简化为无参数版本）
#     - 用预定义的频率band分解输入
#     - 对每个band用可学习的空域权重调制
#     """
#     def __init__(self, in_channels, kernel_num=3, reduction=8):
#         super().__init__()
#         self.kernel_num = kernel_num
#         self.in_channels = in_channels
        
#         # 预定义频率阈值（低、中、高）
#         # 这些阈值表示"保留频率半径 < thr"的分量
#         self.register_buffer(
#             'freq_thresholds',
#             torch.tensor([0.2, 0.4, 0.5])  # 对应k=2.5, 1.25, 1.0
#         )
        
#         # 为每个频带生成空域调制权重
#         mid_channels = max(in_channels // reduction, 4)
#         self.band_modulators = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, mid_channels, 3, padding=1, groups=1),
#                 nn.GELU(),
#                 nn.Conv2d(mid_channels, in_channels, 1),
#                 nn.Sigmoid()
#             )
#             for _ in range(kernel_num)
#         ])
        
#         # 最低频残差的调制
#         self.lowfreq_modulator = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, in_channels, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         device = x.device
        
#         # FFT到频域
#         x_fft = torch.fft.rfft2(x, norm='ortho')
        
#         # 构建频率坐标网格
#         freq_h = torch.fft.fftfreq(H, device=device)
#         freq_w = torch.fft.rfftfreq(W, device=device)
#         grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
#         freq_radius = torch.sqrt(grid_h**2 + grid_w**2)  # [H, W//2+1]
        
#         # 逐频带处理
#         pre_x = x.clone()
#         outputs = []
        
#         for i, thr in enumerate(self.freq_thresholds):
#             # 低通mask
#             mask = (freq_radius < thr).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W//2+1]
            
#             # 提取低频部分并逆变换
#             low_fft = x_fft * mask
#             low_spatial = torch.fft.irfft2(low_fft, s=(H, W), norm='ortho')
            
#             # 高频残差
#             high_spatial = pre_x - low_spatial
#             pre_x = low_spatial
            
#             # 空域调制
#             modulator_weight = self.band_modulators[i](x)  # 用原始特征生成权重
#             high_modulated = high_spatial * modulator_weight
            
#             outputs.append(high_modulated)
        
#         # 最低频部分
#         lowfreq_weight = self.lowfreq_modulator(x)
#         outputs.append(pre_x * lowfreq_weight)
        
#         return sum(outputs)


    
    
class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MultiScalePerceptionBlock(nn.Module):
    def __init__(self, channels, kernels=(3, 5, 7), act='silu'):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act),
                ConvNormLayer_fuse(channels, channels, kernel_size, 1, g=channels, bias=False, act=act),
            )
            for kernel_size in kernels
        ])
        self.project = ConvNormLayer_fuse(channels * (len(kernels) + 1), channels, 1, 1, bias=False, act=act)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        feats = [x] + [branch(x) for branch in self.branches]
        fused = self.project(torch.cat(feats, dim=1))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * fused * self.gate(fused)


class HighFrequencyEnhancer(nn.Module):
    def __init__(self, channels, act='silu'):
        super().__init__()
        self.depthwise = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act)
        self.project = ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = self.project(self.depthwise(x - smooth))
        return x + torch.tanh(self.scale) * high


class ParameterFreeEdgeStructureAttention(nn.Module):
    def __init__(self, cutoff=0.10, blend=0.15, eps=1e-6):
        super().__init__()
        self.cutoff = float(cutoff)
        self.blend = float(blend)
        self.eps = eps

    def _attention(self, x, center=True):
        if center:
            base = x - x.mean(dim=(-2, -1), keepdim=True)
            score = base.pow(2)
        else:
            score = x.pow(2)
        score = score / (score.mean(dim=(-2, -1), keepdim=True) + self.eps)
        score = (score - score.mean(dim=(-2, -1), keepdim=True)) / (
            score.std(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.sigmoid(score)

    def forward(self, x):
        b, c, h, w = x.shape
        orig_dtype = x.dtype
        device_type = 'cuda' if x.is_cuda else 'cpu'

        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            x_fft = torch.fft.rfft2(x32, norm='ortho')
            freq_h = torch.fft.fftfreq(h, device=x.device)
            freq_w = torch.fft.rfftfreq(w, device=x.device)
            grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
            radius2 = grid_h.pow(2) + grid_w.pow(2)
            sigma2 = max(self.cutoff, self.eps) ** 2
            low_mask = torch.exp(-radius2 / (2.0 * sigma2)).view(1, 1, h, w // 2 + 1)

            low = torch.fft.irfft2(x_fft * low_mask, s=(h, w), norm='ortho')
            high = x32 - low
            edge_att = self._attention(high, center=True)
            struct_att = self._attention(low, center=False)
            refined = (
                low * (1.0 + self.blend * (struct_att - 0.5)) +
                high * (1.0 + self.blend * (edge_att - 0.5))
            )
        return refined.to(orig_dtype)


class LargeKernelDistillationBlock(nn.Module):
    def __init__(self, channels, act='silu'):
        super().__init__()
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act)
        self.mid = ConvNormLayer_fuse(channels, channels, 7, 1, g=channels, bias=False, act=act)
        self.large = ConvNormLayer_fuse(channels, channels, 11, 1, g=channels, bias=False, act=act)
        self.mix = ConvNormLayer_fuse(channels * 3, channels, 1, 1, bias=False, act=act)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 16, 8), 1, bias=False),
            get_activation(act),
            nn.Conv2d(max(channels // 16, 8), channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        fused = self.mix(torch.cat([self.local(x), self.mid(x), self.large(x)], dim=1))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * fused * self.gate(fused)


class UncertaintyContextRefiner(nn.Module):
    def __init__(self, channels, act='silu'):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.context_proj = ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act)
        self.detail_proj = nn.Sequential(
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            get_activation(act),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_var = F.avg_pool2d(x.pow(2), kernel_size=5, stride=1, padding=2) - local_mean.pow(2)
        local_var = local_var.relu()
        uncertainty = self.uncertainty(torch.cat([x, local_var], dim=1))
        context = self.context_proj(local_mean)
        detail = self.detail_proj(x - local_mean)
        update = uncertainty * detail + (1.0 - uncertainty) * context
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class RetinexPriorGate(nn.Module):
    def __init__(self, channels, act='silu'):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.prior_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            get_activation(act),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        illumination = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        reflectance = x - illumination
        contrast = reflectance.abs()
        gate = self.prior_fuse(torch.cat([illumination, reflectance, contrast], dim=1))
        alpha = 0.25 * torch.sigmoid(self.res_scale)
        return x + alpha * reflectance * gate


class HighOrderOmniAttention(nn.Module):
    """
    Detector-friendly high-order omni-attention.
    It mixes local texture, larger-context structure, channel attention and
    spatial attention, then injects a small high-order residual.
    """
    def __init__(self, channels, act='silu'):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.local_branch = nn.Sequential(
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.edge_branch = nn.Sequential(
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            get_activation(act),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out_proj = ConvNormLayer_fuse(channels * 3, channels, 1, 1, bias=False, act=act)
        self.res_scale = nn.Parameter(torch.tensor(-2.5))

    def forward(self, x):
        local = self.local_branch(x)
        context = self.context_branch(x)
        high = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = self.edge_branch(high)
        order2 = local * torch.sigmoid(context)
        fused = self.out_proj(torch.cat([local, order2, edge], dim=1))
        spatial_stats = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.max(dim=1, keepdim=True).values],
            dim=1,
        )
        fused = fused * self.channel_gate(fused) * self.spatial_gate(spatial_stats)
        alpha = 0.30 * torch.sigmoid(self.res_scale)
        return x + alpha * fused


class MultiPriorRetinexRefiner(nn.Module):
    """
    Multi-prior Retinex-style refiner for feature maps.
    It uses multi-scale illumination, reflectance, contrast and gradient priors
    without assuming RGB input, so it can sit after the FPN/PAN.
    """
    def __init__(self, channels, act='silu'):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.prior_fuse = nn.Sequential(
            nn.Conv2d(channels * 6, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            get_activation(act),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.detail_proj = nn.Sequential(
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, bias=False, act=act),
            ConvNormLayer_fuse(channels, channels, 1, 1, bias=False, act=act),
        )
        self.res_scale = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _illumination(x, kernel_size):
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        illum_s = self._illumination(x, 3)
        illum_m = self._illumination(x, 7)
        illum_l = self._illumination(x, 11)
        reflectance = x - illum_m
        contrast = reflectance.abs()
        grad_h = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        grad_v = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        gradient = grad_h.abs() + grad_v.abs()
        gate = self.prior_fuse(torch.cat([illum_s, illum_m, illum_l, reflectance, contrast, gradient], dim=1))
        detail = self.detail_proj(reflectance + gradient)
        alpha = 0.25 * torch.sigmoid(self.res_scale)
        return x + alpha * detail * gate


class MSCKChineseKnotRefiner(nn.Module):
    """
    MSCK-Net style multi-scale Chinese-knot convolution for dense small targets.
    Each kernel branch combines strip convolutions in horizontal/vertical routes
    with a local depthwise body branch, then a convolutional GLU stabilizes the
    residual update.
    """
    def __init__(self, channels, kernels=(3, 5, 7), act='silu'):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels or any(k not in (3, 5, 7, 9, 11) for k in kernels):
            raise ValueError("MSCK kernels must be selected from 3, 5, 7, 9, 11.")
        half = max(channels // 2, 1)
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.ModuleDict({
                "body": ConvNormLayer_fuse(channels, half, 3, 1, g=half, act=act),
                "head_h": ConvNormLayer_fuse(channels, half, (1, k), 1, g=half,
                                             padding=(0, k // 2), act=act),
                "head_v": ConvNormLayer_fuse(half, half, (k, 1), 1, g=half,
                                             padding=(k // 2, 0), act=act),
                "expand": nn.Conv2d(half, channels, 1, groups=half, bias=False),
            }))
        fuse_groups = math.gcd(16, math.gcd(len(kernels) * channels, channels))
        self.fuse = nn.Sequential(
            nn.Conv2d(len(kernels) * channels, channels, 1, groups=max(1, fuse_groups), bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )
        hidden = max(int(channels * 2 / 3), 16)
        self.glu_fc1 = nn.Conv2d(channels, hidden * 2, 1)
        self.glu_dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.GELU(),
        )
        self.glu_fc2 = nn.Conv2d(hidden, channels, 1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(channels))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(channels))

    def forward(self, x):
        x_norm = self.norm1(x)
        outs = []
        for branch in self.branches:
            strip = branch["head_v"](branch["head_h"](x_norm))
            body = branch["body"](x_norm)
            outs.append(branch["expand"](strip + body))
        update = self.fuse(torch.cat(outs, dim=1))
        x = x + self.layer_scale_1.view(1, -1, 1, 1) * update

        y, gate = self.glu_fc1(self.norm2(x)).chunk(2, dim=1)
        y = self.glu_fc2(self.glu_dw(y) * gate)
        return x + self.layer_scale_2.view(1, -1, 1, 1) * y


class CSAFSpatialEnergy(nn.Module):
    """Spatial energy attention from CSAFNet/SimAM-style variance statistics."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        max_v = x.amax(dim=(-2, -1), keepdim=True)
        min_v = x.amin(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        score = (x - mean).pow(2) / (max_v - min_v + self.eps) * var
        return torch.sigmoid(score)


class CSAFWaveletDetailEnhancer(nn.Module):
    """CSAFNet wavelet-detail enhancement branch with a convolutional fallback."""
    def __init__(self, channels, act='silu'):
        super().__init__()
        self.depth_value = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.depth_attn = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.detail_proj = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.GELU()
        if _WAVELETS_AVAILABLE:
            self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        else:
            self.dwt = None

    def _wavelet_detail(self, x):
        if self.dwt is None:
            smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            return x - smooth
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            _, yh = self.dwt(x32)
            detail = yh[0][:, :, 0] + yh[0][:, :, 1] + yh[0][:, :, 2]
        return F.interpolate(detail.to(dtype=x.dtype), size=x.shape[2:], mode='nearest')

    def forward(self, x):
        value = self.depth_value(x)
        attn_seed = self.depth_attn(x)
        detail = self._wavelet_detail(attn_seed)
        attn = torch.tanh(self.out_proj(self.act(self.detail_proj(torch.cat([detail, attn_seed], dim=1)))))
        return value * attn


class CSAFAdaptiveFilter(nn.Module):
    """
    CSAFNet-inspired context-aware and semantic-guided adaptive filtering.
    It preserves the local/global adaptive branch, wavelet detail branch, and
    channel reweighting, adapted as a residual feature refiner for FPN maps.
    """
    def __init__(self, channels, scale=4, act='silu'):
        super().__init__()
        self.scale = max(int(scale), 1)
        self.split = nn.Conv2d(channels, channels * 2, 1)
        self.local_proj = nn.Conv2d(channels, channels, 1)
        self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.energy = CSAFSpatialEnergy()
        self.detail = CSAFWaveletDetailEnhancer(channels, act=act)
        hidden = max(channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.out_proj = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.belt = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.res_scale = nn.Parameter(torch.tensor(-2.5))

    def forward(self, x):
        b, c, h, w = x.shape
        y, local = self.split(x).chunk(2, dim=1)
        pooled = F.adaptive_avg_pool2d(local, (max(1, h // self.scale), max(1, w // self.scale)))
        local_context = self.local_dw(pooled * self.alpha)
        local_context = F.interpolate(self.local_proj(F.gelu(local_context)), size=(h, w), mode='nearest')
        local_context = local_context + self.energy(local) * self.belt
        local_filtered = local * local_context
        detail = self.detail(y)
        out = (local_filtered + detail)
        out = self.out_proj(out * self.channel_gate(out))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * out


class CSAFDMSSPRefiner(nn.Module):
    """
    Dense multi-scale spatial semantic perception from CSAFNet, kept as a
    same-resolution residual block for detector features.
    """
    def __init__(self, channels, channel_split=(1, 1, 2), dilations=(18, 12, 6), act='silu'):
        super().__init__()
        total = sum(channel_split)
        if channels % total != 0:
            raise ValueError(f"channels={channels} must be divisible by sum(channel_split)={total}")
        c1 = int(channels * channel_split[2] / total)
        c2 = int(channels * channel_split[1] / total)
        c0 = channels - c1 - c2
        self.splits = (c1, c2, c0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Conv2d(channels, channels, 1)
        self.spatial_energy = CSAFSpatialEnergy()
        self.belt = nn.Parameter(torch.zeros(1, channels, 1, 1))
        def make_atrous(ch, dilation):
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(
                    ch, ch, kernel_size=3, stride=1,
                    padding=dilation, dilation=dilation, bias=False
                )),
                ("norm", nn.BatchNorm2d(ch)),
                ("act", get_activation(act)),
            ]))

        self.atrous = nn.ModuleList([
            make_atrous(c1, dilations[2]),
            make_atrous(c2, dilations[1]),
            make_atrous(c0, dilations[0]),
        ])
        self.mix = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.res_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        h, w = x.shape[2:]
        global_context = F.interpolate(self.global_proj(self.global_pool(x)), size=(h, w), mode='bilinear', align_corners=False)
        spatial_context = self.spatial_energy(x) * x
        x_s = global_context + self.belt * spatial_context
        x1, x2, x0 = torch.split(x, self.splits, dim=1)
        out = torch.cat([self.atrous[0](x1), self.atrous[1](x2), self.atrous[2](x0)], dim=1)
        out = self.mix(out + x_s)
        alpha = 0.30 * torch.sigmoid(self.res_scale)
        return x + alpha * out


class RDVQCodebookRefiner(nn.Module):
    """
    Differentiable vector-quantization feature refiner inspired by RDVQ.
    It uses a soft codebook distribution, exposes an entropy-shaped confidence,
    and injects only the quantization residual that the detector can use.
    """
    def __init__(self, channels, codebook_size=64, temperature=0.7, act='silu'):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.temperature = float(temperature)
        self.pre = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, channels) * 0.02)
        self.post = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 8, 16), channels, 1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(-3.0))
        self.last_entropy = None

    def forward(self, x):
        z = self.pre(x)
        b, c, h, w = z.shape
        flat = z.flatten(2).transpose(1, 2).float()
        codebook = self.codebook.float()
        dist = flat.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=-1).view(1, 1, -1)
        dist = dist - 2.0 * torch.matmul(flat, codebook.t())
        assign = torch.softmax(-dist / max(self.temperature, 1e-4), dim=-1)
        quant = torch.matmul(assign, codebook).transpose(1, 2).reshape(b, c, h, w).to(dtype=x.dtype)
        entropy = -(assign * assign.clamp_min(1e-8).log()).sum(dim=-1)
        confidence = (1.0 - entropy / math.log(float(self.codebook_size))).clamp(0, 1)
        confidence = confidence.reshape(b, 1, h, w).to(dtype=x.dtype)
        self.last_entropy = entropy.detach().mean()
        update = self.post((quant - z).to(dtype=x.dtype)) * confidence * self.channel_gate(z)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class DATPPDeformableRefiner(nn.Module):
    """
    DAT++ style spatially dynamic deformable attention for selected FPN levels.
    It keeps the learned grouped offsets, sampled keys/values, and attention
    over deformed locations while using a residual wrapper for detector stability.
    """
    def __init__(self, channels, n_heads=8, n_groups=4, stride=2,
                 offset_range_factor=2.0, ksize=5, act='silu'):
        super().__init__()
        if channels % n_heads != 0 or channels % n_groups != 0:
            raise ValueError("channels must be divisible by n_heads and n_groups")
        self.channels = channels
        self.n_heads = int(n_heads)
        self.n_groups = int(n_groups)
        self.head_dim = channels // n_heads
        self.group_channels = channels // n_groups
        self.group_heads = n_heads // n_groups
        self.stride = int(stride)
        self.offset_range_factor = float(offset_range_factor)
        pad = ksize // 2
        self.proj_q = nn.Conv2d(channels, channels, 1)
        self.proj_k = nn.Conv2d(channels, channels, 1)
        self.proj_v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels, ksize,
                      stride=self.stride, padding=pad, groups=self.group_channels),
            nn.GroupNorm(1, self.group_channels),
            nn.GELU(),
            nn.Conv2d(self.group_channels, 2, 1, bias=False),
        )
        self.lepe = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.drop = nn.Dropout(0.0)
        self.scale = self.head_dim ** -0.5
        self.res_scale = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _reference_grid(h, w, b, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing='ij',
        )
        grid = torch.stack((x, y), dim=-1)
        return grid.unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.proj_q(x)
        q_group = q.reshape(b, self.n_groups, self.group_channels, h, w).reshape(
            b * self.n_groups, self.group_channels, h, w
        )
        offset = self.conv_offset(q_group)
        hk, wk = offset.shape[-2:]
        offset = offset.permute(0, 2, 3, 1).contiguous()
        range_xy = torch.tensor(
            [2.0 / max(wk - 1, 1), 2.0 / max(hk - 1, 1)],
            device=x.device, dtype=offset.dtype,
        ).view(1, 1, 1, 2)
        offset = offset.tanh() * range_xy * self.offset_range_factor
        ref = self._reference_grid(hk, wk, b * self.n_groups, x.device, offset.dtype)
        grid = (ref + offset).clamp(-1.0, 1.0)

        sampled = F.grid_sample(
            x.reshape(b * self.n_groups, self.group_channels, h, w),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).reshape(b, c, hk, wk)

        ns = hk * wk
        nq = h * w
        qh = q.reshape(b, self.n_heads, self.head_dim, nq).transpose(2, 3)
        kh = self.proj_k(sampled).reshape(b, self.n_heads, self.head_dim, ns)
        vh = self.proj_v(sampled).reshape(b, self.n_heads, self.head_dim, ns).transpose(2, 3)
        attn = torch.matmul(qh, kh) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(self.drop(attn), vh).transpose(2, 3).reshape(b, c, h, w)
        out = self.proj_out(out + self.lepe(q))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * out


class EFRUFGradientTransferRefiner(nn.Module):
    """
    EFRUF-inspired gradient-transfer and unfolded fusion rule refiner.
    For dental images, Sobel-like edge/detail evidence is a useful proxy for
    cusps, roots, and inter-tooth boundaries; the unfolded stages learn how
    much of that detail should be injected into detector features.
    """
    def __init__(self, channels, unfold_steps=2, res_init=-1.9, act='silu'):
        super().__init__()
        self.unfold_steps = max(int(unfold_steps), 1)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(1, max(channels // 4, 16), 3, padding=1, bias=False),
            nn.BatchNorm2d(max(channels // 4, 16)),
            get_activation(act),
            nn.Conv2d(max(channels // 4, 16), channels, 1, bias=True),
        )
        self.rules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels * 2, 1, bias=True),
            )
            for _ in range(self.unfold_steps)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _gradient_detail(self, x):
        gray = x.mean(dim=1, keepdim=True).float()
        gx = F.conv2d(gray, self.sobel_x.to(device=x.device, dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(device=x.device, dtype=gray.dtype), padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-6)
        grad = grad / grad.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-4)
        return self.detail_proj(grad).to(dtype=x.dtype)

    def forward(self, x):
        detail = self._gradient_detail(x)
        state = x
        for rule in self.rules:
            gate, update = rule(torch.cat([state, detail], dim=1)).chunk(2, dim=1)
            state = state + torch.sigmoid(gate) * update
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * (state - x)


class SINFSpatialImplicitFieldRefiner(nn.Module):
    """
    SINF-inspired continuous spatial implicit field refiner for single images.
    The original temporal field is not appropriate for still panoramic images;
    this keeps blind-spot local denoising plus coordinate-conditioned implicit
    residuals to smooth noise while preserving structured tooth edges.
    """
    def __init__(self, channels, num_blocks=2, coord_dim=8, res_init=-2.0, act='silu'):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.masked = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        mask = torch.ones_like(self.masked.weight)
        mask[:, :, 1, 1] = 0.0
        self.register_buffer("center_mask", mask, persistent=False)
        self.point = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(max(int(num_blocks), 1))
        ])
        self.coord_mlp = nn.Sequential(
            nn.Conv2d(self.coord_dim, max(channels // 4, 16), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 16), channels, 1, bias=True),
        )
        self.tail = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _coords(self, h, w, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing='ij',
        )
        coords = [x, y]
        for freq in (1.0, 2.0, 4.0):
            coords.extend([torch.sin(math.pi * freq * x), torch.cos(math.pi * freq * y)])
        return torch.stack(coords, dim=0).unsqueeze(0)

    def forward(self, x):
        weight = self.masked.weight * self.center_mask.to(device=x.device, dtype=self.masked.weight.dtype)
        y = F.conv2d(x, weight, bias=None, padding=1, groups=x.shape[1])
        y = self.point(y)
        for block in self.blocks:
            y = y + block(y)
        coord = self._coords(x.shape[-2], x.shape[-1], x.device, x.dtype).expand(x.shape[0], -1, -1, -1)
        y = self.tail(y + self.coord_mlp(coord))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * y


class SMWGFourierSpectralModulation(nn.Module):
    """Fourier spectral modulation adapted from SMWG-DETR for feature maps."""
    def __init__(self, channels, res_init=-2.0, act='silu'):
        super().__init__()
        self.pre = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.post = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.band_logits = nn.Parameter(torch.zeros(channels, 3))
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 8, 16), channels, 1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _radial_bands(h, w, device, dtype):
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, -1)

        r = torch.sqrt(fx.square() + fy.square())
        low = torch.exp(-(r / 0.16).square())
        mid = torch.exp(-((r - 0.28) / 0.16).square())
        high = (1.0 - torch.exp(-(r / 0.34).square())).clamp(0, 1)

        return torch.stack([low, mid, high], dim=0)

    def forward(self, x):
        z = self.pre(x)
        b, c, h, w = z.shape
        z32 = z.float()
        spec = torch.fft.rfft2(z32, norm='ortho')
        bands = self._radial_bands(h, w, z.device, z32.dtype)
        band_weights = torch.softmax(self.band_logits.float(), dim=-1)
        modulation = torch.einsum('ck,khw->chw', band_weights, bands).unsqueeze(0)
        spec = spec * (1.0 + 0.20 * torch.tanh(modulation))
        filtered = torch.fft.irfft2(spec, s=(h, w), norm='ortho').to(dtype=x.dtype)
        update = self.post(filtered) * self.channel_gate(z)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class SMWGSpectralWaveletRefiner(nn.Module):
    """
    SMWG-DETR-inspired Fourier spectral modulation plus wavelet/detail-guided
    cross-level refinement. The implementation uses detector feature maps
    directly and falls back to a local high-pass detail map when DWT is absent.
    """
    def __init__(self, channels, num_levels=3, res_init=-1.8, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.spectral = nn.ModuleList([
            SMWGFourierSpectralModulation(channels, res_init=res_init, act=act)
            for _ in range(self.num_levels)
        ])
        self.detail_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])
        self.coarse_fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _high_pass(x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - smooth

    def forward(self, feats):
        feats = [self.spectral[i](x) for i, x in enumerate(feats)]
        outs = list(feats)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        for i in range(len(feats) - 1):
            fine_detail = self._high_pass(feats[i])
            fine_detail = F.interpolate(
                fine_detail,
                size=feats[i + 1].shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            guide = torch.sigmoid(self.detail_proj[i](fine_detail))
            update = self.coarse_fuse[i](feats[i + 1] * guide + fine_detail)
            outs[i + 1] = outs[i + 1] + alpha * update
        return outs


class StructuralEntropyFrequencySaliencyRefiner(nn.Module):
    """
    Lightweight detector-native refiner inspired by SEGD, FSDETR, DF2Net,
    SDCoNet, and IDGBR. It keeps the useful parts for tooth detection:
    structural-entropy gating, frequency/spatial detail, cross-level
    differential fusion, saliency focusing, and boundary-biased residuals.
    """
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.low_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.high_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.diff_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                ConvNormLayer_fuse(channels * 3, channels, 1, 1, act=act),
                ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act),
            )
            for _ in range(self.num_levels)
        ])
        self.saliency_head = nn.ModuleList([
            nn.Conv2d(channels * 2, 1, kernel_size=3, padding=1, bias=True)
            for _ in range(self.num_levels)
        ])
        self.entropy_gate = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=1, bias=True)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _high_pass(x, kernel_size=5):
        padding = kernel_size // 2
        low = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        return low, x - low

    @staticmethod
    def _local_entropy_proxy(x):
        energy = x.float().abs().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(energy, kernel_size=5, stride=1, padding=2)
        prob = (energy / (energy + local + 1e-6)).clamp(1e-4, 1.0 - 1e-4)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return entropy / math.log(2.0)

    def forward(self, feats):
        if not feats:
            return feats

        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            resized = [
                feat if feat.shape[2:] == x.shape[2:] else F.interpolate(
                    feat, size=x.shape[2:], mode='bilinear', align_corners=False
                )
                for feat in feats
            ]
            context = torch.stack(resized, dim=0).mean(dim=0)
            low, high = self._high_pass(x)
            diff = x - context

            low_update = self.low_proj[idx](low)
            high_update = self.high_proj[idx](high)
            diff_update = self.diff_proj[idx](diff)
            update = self.fuse[idx](torch.cat([low_update, high_update, diff_update], dim=1))

            saliency = torch.sigmoid(self.saliency_head[idx](torch.cat([x, high], dim=1)))
            entropy = self._local_entropy_proxy(x).to(dtype=x.dtype)
            detail = high.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype)
            entropy_gate = torch.sigmoid(self.entropy_gate[idx](torch.cat([entropy, detail], dim=1)))
            gate = 0.5 * (saliency + entropy_gate)

            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * gate * update)
        return outs


class RotEquivariantDetailRefiner(nn.Module):
    """
    Rot-E ASISR-inspired shared-transform detail refiner. It applies the same
    local detail branch to several geometric views and averages the inverse
    transformed residuals, preserving the useful equivariance bias without
    introducing an image super-resolution head.
    """
    def __init__(self, channels, res_init=-2.0, act='silu', use_rot90=True):
        super().__init__()
        self.use_rot90 = bool(use_rot90)
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 8, 16), channels, 1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _apply_transform(x, mode):
        if mode == 'identity':
            return x
        if mode == 'hflip':
            return torch.flip(x, dims=(-1,))
        if mode == 'vflip':
            return torch.flip(x, dims=(-2,))
        if mode == 'rot90':
            return torch.rot90(x, k=1, dims=(-2, -1))
        raise ValueError(f"Unsupported transform mode {mode!r}")

    @staticmethod
    def _invert_transform(x, mode):
        if mode == 'identity':
            return x
        if mode == 'hflip':
            return torch.flip(x, dims=(-1,))
        if mode == 'vflip':
            return torch.flip(x, dims=(-2,))
        if mode == 'rot90':
            return torch.rot90(x, k=-1, dims=(-2, -1))
        raise ValueError(f"Unsupported transform mode {mode!r}")

    def forward(self, x):
        modes = ['identity', 'hflip', 'vflip']
        if self.use_rot90:
            modes.append('rot90')
        updates = []
        for mode in modes:
            update = self.detail(self._apply_transform(x, mode))
            update = self._invert_transform(update, mode)
            if update.shape[2:] != x.shape[2:]:
                update = F.interpolate(update, size=x.shape[2:], mode='bilinear', align_corners=False)
            updates.append(update)
        update = torch.stack(updates, dim=0).mean(dim=0)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha.to(dtype=x.dtype) * self.channel_gate(x) * update


class HybridEnergyDistancePromptRefiner(nn.Module):
    """
    HEDP-inspired feature prompt gate. It uses learnable domain prompts and a
    hybrid energy-distance score to choose a mild residual update for exposure
    and acquisition-domain shifts in phone-shot dental data.
    """
    def __init__(self, channels, num_levels=3, prompt_count=4, res_init=-2.1, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.prompt_count = max(int(prompt_count), 2)
        self.prompts = nn.Parameter(torch.randn(self.num_levels, self.prompt_count, channels) * 0.02)
        self.local = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(self.num_levels)
        ])
        self.prompt_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2 + 1, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.out_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, feats):
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            local = self.local[idx](x)
            desc = F.adaptive_avg_pool2d(local, 1).flatten(1).float()
            prompts = self.prompts[idx].float()
            dist = (desc[:, None, :] - prompts[None, :, :]).pow(2).mean(dim=-1).sqrt()
            energy = -torch.logsumexp(-dist, dim=-1, keepdim=True)
            weights = F.softmax(-dist, dim=-1)
            prompt = torch.matmul(weights, prompts).to(dtype=x.dtype).view(x.shape[0], -1, 1, 1)
            energy_map = energy.to(dtype=x.dtype).view(x.shape[0], 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
            gate = self.prompt_proj[idx](torch.cat([x, prompt.expand_as(x), energy_map], dim=1))
            update = self.out_proj[idx](local * gate)
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class SAMPriorSSTScanRefiner(nn.Module):
    """
    SAMPG-SST-MCDNet-inspired prior-guided spatial/spectral scan. The original
    method targets hyperspectral bi-temporal change detection with SAM priors;
    here we keep the useful prior-guided long-range scan and spectral gate for
    single-image dental detection.
    """
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.prior_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 1, 7, padding=3, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.spectral_scan = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=True)
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                ConvNormLayer_fuse(channels * 3, channels, 1, 1, act=act),
                ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act),
            )
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _arch_prior(h, w, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing='ij',
        )
        arch = torch.exp(-((y - 0.22 * (x.square() - 0.45)).square()) / 0.20)
        return arch.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _bidirectional_cumsum(x, gate, dim):
        weight = gate.clamp_min(1e-4)
        fwd_num = torch.cumsum(x * weight, dim=dim)
        fwd_den = torch.cumsum(weight, dim=dim).clamp_min(1e-4)
        fwd = fwd_num / fwd_den

        rev_x = torch.flip(x, dims=(dim,))
        rev_w = torch.flip(weight, dims=(dim,))
        bwd_num = torch.cumsum(rev_x * rev_w, dim=dim)
        bwd_den = torch.cumsum(rev_w, dim=dim).clamp_min(1e-4)
        bwd = torch.flip(bwd_num / bwd_den, dims=(dim,))
        return 0.5 * (fwd + bwd)

    def forward(self, feats):
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            mean = x.mean(dim=1, keepdim=True)
            maxv = x.max(dim=1, keepdim=True).values
            arch = self._arch_prior(x.shape[-2], x.shape[-1], x.device, x.dtype).expand(x.shape[0], -1, -1, -1)
            prior = self.prior_head[idx](torch.cat([mean, maxv, arch], dim=1))

            h_scan = self._bidirectional_cumsum(x, prior, dim=-1)
            v_scan = self._bidirectional_cumsum(x, prior, dim=-2)
            spectral_desc = x.mean(dim=(-2, -1), keepdim=False).unsqueeze(1)
            spectral = torch.sigmoid(self.spectral_scan[idx](spectral_desc.float())).to(dtype=x.dtype)
            spectral = spectral.squeeze(1).view(x.shape[0], -1, 1, 1)

            update = self.fuse[idx](torch.cat([x, h_scan, v_scan], dim=1)) * spectral * prior
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class SEGDStructuralEntropyRefiner(nn.Module):
    """SEGD-inspired structural entropy and degradation-evidence feature refiner."""
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.low_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.detail_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.evidence_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _entropy(x):
        energy = x.float().abs().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(energy, kernel_size=5, stride=1, padding=2)
        prob = (energy / (energy + local + 1e-6)).clamp(1e-4, 1.0 - 1e-4)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return entropy / math.log(2.0)

    @staticmethod
    def _norm_map(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.float().var(dim=(-2, -1), keepdim=True, unbiased=False).add(1e-6).sqrt().to(dtype=x.dtype)
        return (x - mean) / std.clamp_min(1e-4)

    def forward(self, feats):
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            detail = x - low
            entropy = self._entropy(x).to(dtype=x.dtype)
            detail_energy = detail.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype)
            contrast = F.avg_pool2d(detail_energy, kernel_size=3, stride=1, padding=1)
            evidence = self.evidence_gate[idx](torch.cat([
                entropy,
                self._norm_map(detail_energy),
                self._norm_map(contrast),
            ], dim=1))
            update = self.fuse[idx](torch.cat([
                self.low_proj[idx](low),
                self.detail_proj[idx](detail * evidence),
            ], dim=1))
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class FSDETRFrequencySpatialRefiner(nn.Module):
    """FSDETR-inspired frequency-spatial feature enhancement for small objects."""
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.frequency = nn.ModuleList([
            SMWGFourierSpectralModulation(channels, res_init=res_init, act=act)
            for _ in range(self.num_levels)
        ])
        self.spatial = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(self.num_levels)
        ])
        self.mix_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, feats):
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            freq_update = self.frequency[idx](x) - x
            spatial_update = self.spatial[idx](x)
            gate = self.mix_gate[idx](torch.cat([freq_update, spatial_update], dim=1))
            update = gate * freq_update + (1.0 - gate) * spatial_update
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class DF2NetDifferentialDynamicFusionRefiner(nn.Module):
    """DF2Net-inspired differential feature dynamic fusion over FPN levels."""
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.context_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.diff_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(self.num_levels)
        ])
        self.dynamic_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 3, channels, 1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, feats):
        if not feats:
            return feats
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            resized = [
                feat if feat.shape[2:] == x.shape[2:] else F.interpolate(
                    feat, size=x.shape[2:], mode='bilinear', align_corners=False
                )
                for feat in feats
            ]
            context = self.context_proj[idx](torch.stack(resized, dim=0).mean(dim=0))
            diff = self.diff_proj[idx](x - context)
            gate = self.dynamic_gate[idx](torch.cat([x, context, diff], dim=1))
            update = self.fuse[idx](torch.cat([x * gate, diff * (1.0 - gate)], dim=1))
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class SDCoNetSaliencyCollaborationRefiner(nn.Module):
    """SDCoNet-inspired saliency-driven shared/specific feature collaboration."""
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.shared = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(self.num_levels)
        ])
        self.specific = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.saliency = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _detail(x):
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            detail = self._detail(x)
            sal = self.saliency[idx](torch.cat([
                x.mean(dim=1, keepdim=True),
                x.max(dim=1, keepdim=True).values,
                detail.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype),
            ], dim=1))
            shared = self.shared[idx](x) * sal
            specific = self.specific[idx](x * sal + detail * (1.0 - sal))
            update = self.fuse[idx](torch.cat([shared, specific], dim=1))
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class IDGBRBoundaryRefinementRefiner(nn.Module):
    """IDGBR-inspired semantic-guided boundary refinement for detector features."""
    def __init__(self, channels, num_levels=3, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.semantic_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.boundary_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(self.num_levels)
        ])
        self.boundary_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.num_levels)
        ])
        self.fuse = nn.ModuleList([
            ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _boundary(x):
        return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, feats):
        if not feats:
            return feats
        context = feats[-1]
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            semantic = context if context.shape[2:] == x.shape[2:] else F.interpolate(
                context, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            semantic = self.semantic_proj[idx](semantic)
            boundary = self.boundary_proj[idx](self._boundary(x))
            gate = self.boundary_gate[idx](torch.cat([
                boundary.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype),
                semantic.float().abs().mean(dim=1, keepdim=True).to(dtype=x.dtype),
            ], dim=1))
            update = self.fuse[idx](torch.cat([semantic * gate, boundary], dim=1))
            outs.append(x + alpha.to(device=x.device, dtype=x.dtype) * update)
        return outs


class BiLaLoRALayerPositionedAdapter(nn.Module):
    """
    BiLaLoRA-inspired low-rank adapter with learnable level positioning.
    Instead of modifying every convolution in the detector, this block searches
    which FPN levels should receive a compact low-rank residual update.
    """
    def __init__(self, channels, num_levels=3, rank=8, topk=2, res_init=-2.1, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.rank = max(int(rank), 1)
        self.topk = max(int(topk), 1)
        self.level_logits = nn.Parameter(torch.zeros(self.num_levels))
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, self.rank, 1, bias=False),
                nn.BatchNorm2d(self.rank),
                get_activation(act),
                nn.Conv2d(self.rank, self.rank, 3, padding=1, groups=self.rank, bias=False),
                nn.BatchNorm2d(self.rank),
                get_activation(act),
                nn.Conv2d(self.rank, channels, 1, bias=True),
            )
            for _ in range(self.num_levels)
        ])
        for adapter in self.adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    def _level_weights(self, n_levels):
        logits = self.level_logits[:n_levels].float()
        if self.topk < n_levels:
            keep = torch.topk(logits, k=self.topk, dim=0).indices
            masked = torch.full_like(logits, -torch.inf)
            masked[keep] = logits[keep]
            logits = masked
        return F.softmax(logits, dim=0)

    def forward(self, feats):
        n_levels = min(len(feats), len(self.adapters))
        weights = self._level_weights(n_levels)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            if i >= n_levels:
                outs.append(x)
                continue
            update = self.adapters[i](x)
            outs.append(x + alpha.to(dtype=x.dtype) * weights[i].to(device=x.device, dtype=x.dtype) * update)
        return outs


class SURMTSemanticStructureBridge(nn.Module):
    """
    SU-RMT-inspired bridge between high-level semantic representation and
    structural detail. It uses the deepest level as semantic context and a
    high-pass branch as tooth-boundary/cusp detail before dynamic fusion.
    """
    def __init__(self, channels, num_levels=3, res_init=-1.9, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.semantic_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.structure_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(self.num_levels)
        ])
        self.fuse_logits = nn.Parameter(torch.zeros(self.num_levels, 2))
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _structure_detail(x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - smooth

    def forward(self, feats):
        if not feats:
            return feats
        context = feats[-1]
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            semantic = context if context.shape[2:] == x.shape[2:] else F.interpolate(
                context, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            semantic = self.semantic_proj[idx](semantic)
            semantic_update = semantic * torch.sigmoid(semantic)
            structure_update = self.structure_proj[idx](self._structure_detail(x))
            weights = F.softmax(self.fuse_logits[idx].float(), dim=0).to(device=x.device, dtype=x.dtype)
            update = weights[0] * semantic_update + weights[1] * structure_update
            outs.append(x + alpha.to(dtype=x.dtype) * update)
        return outs


class MAVSparseInteractionRefiner(nn.Module):
    """
    MAVFusion-inspired sparse interaction adapted to still panoramic X-rays.
    A cross-scale residual map serves as a motion/saliency proxy: high-change
    tooth regions get stronger interaction, while static background uses a
    cheap weak branch.
    """
    def __init__(self, channels, num_levels=3, sparse_ratio=0.25, res_init=-2.0, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.sparse_ratio = float(sparse_ratio)
        self.context_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.strong_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for _ in range(self.num_levels)
        ])
        self.weak_interaction = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(self.num_levels)
        ])
        self.saliency_proj = nn.ModuleList([
            nn.Conv2d(channels, 1, 1, bias=True)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _sparse_mask(self, score):
        b, _, h, w = score.shape
        n = h * w
        if self.sparse_ratio >= 1.0:
            return torch.ones_like(score)
        k = max(1, min(n, int(round(n * max(self.sparse_ratio, 0.01)))))
        flat = score.flatten(2)
        threshold = torch.topk(flat, k=k, dim=-1).values[..., -1:].unsqueeze(-1)
        return (flat.view(b, 1, h, w) >= threshold).to(dtype=score.dtype)

    def forward(self, feats):
        if not feats:
            return feats
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            idx = min(i, self.num_levels - 1)
            resized = [
                f if f.shape[2:] == x.shape[2:] else F.interpolate(
                    f, size=x.shape[2:], mode='bilinear', align_corners=False
                )
                for f in feats
            ]
            context = torch.stack(resized, dim=0).mean(dim=0)
            context = self.context_proj[idx](context)
            residual_score = (x - context).abs().mean(dim=1, keepdim=True)
            learned_score = torch.sigmoid(self.saliency_proj[idx](x.float())).to(dtype=x.dtype)
            score = residual_score + learned_score
            mask = self._sparse_mask(score).detach()
            strong = self.strong_interaction[idx](torch.cat([x, context], dim=1))
            weak = self.weak_interaction[idx](x)
            update = mask * strong + (1.0 - mask) * weak
            outs.append(x + alpha.to(dtype=x.dtype) * update)
        return outs


class CDMLDomainAffineAdapter(nn.Module):
    """
    CDML-inspired first-order domain affine adapter.
    Panoramic radiographs often vary by machine, exposure, and preprocessing.
    This block predicts a small feature-wise affine correction from global
    statistics instead of assuming all domains share the same feature scale.
    """
    def __init__(self, channels, res_init=-2.2):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.stat_proj = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.float().var(dim=(-2, -1), keepdim=True, unbiased=False).add(1e-6).sqrt().to(dtype=x.dtype)
        token = torch.cat([mean.flatten(1), std.flatten(1)], dim=-1)
        gamma, beta = self.stat_proj(token.float()).to(dtype=x.dtype).chunk(2, dim=-1)
        gamma = 0.20 * torch.tanh(gamma).view(x.shape[0], -1, 1, 1)
        beta = 0.20 * torch.tanh(beta).view(x.shape[0], -1, 1, 1)
        normalized = (x - mean) / std.clamp_min(1e-4)
        adapted = normalized * (1.0 + gamma) + beta
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * (adapted - x)


class IQPQualityPriorRefiner(nn.Module):
    """
    IQPIR-inspired no-reference image quality prior for detector features.
    It estimates contrast/sharpness/noise-like statistics from each feature map
    and uses them to choose between common and high-quality codebook priors.
    """
    def __init__(self, channels, codebook_size=16, res_init=-2.0, act='silu'):
        super().__init__()
        self.codebook_size = max(int(codebook_size), 4)
        hidden = max(channels // 4, 16)
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.quality_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels + 1),
        )
        self.common_codebook = nn.Parameter(torch.randn(self.codebook_size, channels) * 0.02)
        self.hq_codebook = nn.Parameter(torch.randn(self.codebook_size, channels) * 0.02)
        self.token_proj = nn.Linear(channels, channels)
        self.out_proj = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _feature_quality_stats(x):
        x32 = x.float()
        contrast = x32.var(dim=(-2, -1), unbiased=False).mean(dim=1).sqrt()
        dx = (x32[..., :, 1:] - x32[..., :, :-1]).abs().mean(dim=(1, 2, 3))
        dy = (x32[..., 1:, :] - x32[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        sharpness = 0.5 * (dx + dy)
        smooth = F.avg_pool2d(x32, 3, stride=1, padding=1)
        noise = (x32 - smooth).abs().mean(dim=(1, 2, 3))
        dynamic = (x32.amax(dim=(-2, -1)) - x32.amin(dim=(-2, -1))).mean(dim=1)
        return torch.stack([contrast, sharpness, noise, dynamic], dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        stats = self._feature_quality_stats(x)
        cond = self.quality_mlp(stats).to(dtype=x.dtype)
        channel_gate, quality_score = cond[:, :c], cond[:, c:]
        channel_gate = torch.sigmoid(channel_gate).view(b, c, 1, 1)
        quality_score = torch.sigmoid(quality_score).view(b, 1, 1)

        token = F.adaptive_avg_pool2d(x, 1).flatten(1)
        token = self.token_proj(token.float())
        common_logits = torch.matmul(token, self.common_codebook.float().t()) / math.sqrt(max(c, 1))
        hq_logits = torch.matmul(token, self.hq_codebook.float().t()) / math.sqrt(max(c, 1))
        common_prior = torch.matmul(torch.softmax(common_logits, dim=-1), self.common_codebook.float())
        hq_prior = torch.matmul(torch.softmax(hq_logits, dim=-1), self.hq_codebook.float())
        prior = common_prior + quality_score.squeeze(-1) * (hq_prior - common_prior)
        prior = prior.to(dtype=x.dtype).view(b, c, 1, 1)

        update = self.out_proj(self.local(x) * channel_gate + prior.expand(-1, -1, h, w))
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class DFormerGeometryPriorRefiner(nn.Module):
    """
    DFormer++/DFormerv2-inspired geometry-prior feature refiner.
    Since dental panoramic input has no depth channel, a lightweight coordinate
    and pseudo-arch prior substitutes for RGB-D geometry attention.
    """
    def __init__(self, channels, res_init=-2.0, act='silu'):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Conv2d(4, max(channels // 8, 16), 1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(channels // 8, 16), channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.row_proj = nn.Conv1d(channels + 1, channels, 1, bias=True)
        self.col_proj = nn.Conv1d(channels + 1, channels, 1, bias=True)
        self.local = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.out_proj = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _coord_prior(h, w, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing='ij',
        )
        upper_arch = 0.36 + 0.10 * (x - 0.5).pow(2) * 4.0
        lower_arch = 0.64 - 0.10 * (x - 0.5).pow(2) * 4.0
        arch = torch.exp(-((y - upper_arch).pow(2)) / 0.018) + torch.exp(-((y - lower_arch).pow(2)) / 0.018)
        jaw = torch.tanh((0.5 - y) / 0.10)
        return torch.stack([x, y, arch.clamp(0, 1), jaw], dim=0).unsqueeze(0)

    def forward(self, x):
        b, c, h, w = x.shape
        coord = self._coord_prior(h, w, x.device, x.dtype).expand(b, -1, -1, -1)
        coord_gate = self.coord_proj(coord)

        row_coord = coord[:, 1:2].mean(dim=3)
        col_coord = coord[:, 0:1].mean(dim=2)
        row_ctx = torch.cat([x.mean(dim=3), row_coord], dim=1)
        col_ctx = torch.cat([x.mean(dim=2), col_coord], dim=1)
        row_gate = torch.sigmoid(self.row_proj(row_ctx)).unsqueeze(-1)
        col_gate = torch.sigmoid(self.col_proj(col_ctx)).unsqueeze(-2)

        update = self.local(x * coord_gate) * row_gate * col_gate
        update = self.out_proj(update)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class DCSAMDualConsistencyRefiner(nn.Module):
    """
    DC-SAM-inspired dual-consistency refiner for multi-scale detector features.
    Positive and negative saliency prompts are extracted per level, then a
    shared cross-level prompt enforces consistency before FPN/PAN fusion.
    """
    def __init__(self, channels, num_levels=3, res_init=-1.8, act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        hidden = max(channels // 4, 16)
        self.saliency = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=True), nn.Sigmoid())
            for _ in range(self.num_levels)
        ])
        self.local = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(self.num_levels)
        ])
        self.prompt_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * 2, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, channels),
            )
            for _ in range(self.num_levels)
        ])
        self.out_proj = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(self.num_levels)
        ])
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _weighted_token(x, weight):
        num = (x * weight).sum(dim=(-2, -1))
        den = weight.sum(dim=(-2, -1)).clamp_min(1e-4)
        return num / den

    def forward(self, feats):
        prompts = []
        saliencies = []
        for i, x in enumerate(feats):
            seed = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
            sal = self.saliency[i](seed)
            pos = self._weighted_token(x, sal)
            neg = self._weighted_token(x, 1.0 - sal)
            prompts.append(pos - neg)
            saliencies.append(sal)

        global_prompt = torch.stack(prompts, dim=0).mean(dim=0)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        outs = []
        for i, x in enumerate(feats):
            gate = self.prompt_proj[i](torch.cat([prompts[i], global_prompt], dim=-1).float()).to(dtype=x.dtype)
            gate = torch.sigmoid(gate).view(x.shape[0], x.shape[1], 1, 1)
            contrast_map = 2.0 * saliencies[i] - 1.0
            update = self.out_proj[i](self.local[i](x) * gate * (1.0 + contrast_map))
            outs.append(x + alpha * update)
        return outs


class CNMContinuousODERefiner(nn.Module):
    """
    CNM-UNet inspired continuous neural-memory ODE block.
    It keeps one shared vector field and updates feature state through a stable
    midpoint ODE solver while a compact global memory is self-updated at each
    step. The residual scale starts small so pretrained detector features are
    not disturbed at initialization.
    """
    def __init__(self, channels, steps=3, step_size=0.35, res_init=-1.4, act='silu'):
        super().__init__()
        self.steps = max(int(steps), 1)
        self.step_size = float(step_size)
        self.state_proj = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
        self.local_field = ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
        self.context_field = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.memory_gate = nn.Sequential(
            nn.Conv2d(channels * 2, max(channels // 4, 16), 1, bias=False),
            get_activation(act),
            nn.Conv2d(max(channels // 4, 16), channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.memory_update = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Tanh(),
        )
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _field(self, state, memory):
        memory_map = memory.expand_as(state)
        fused = self.state_proj(torch.cat([state, memory_map], dim=1))
        update = self.local_field(fused) + self.context_field(fused)
        gate = self.memory_gate(torch.cat([state, memory_map], dim=1))
        return self.out_proj(update * gate)

    def forward(self, x):
        state = x
        memory = self.memory_update(x)
        h = self.step_size / float(self.steps)
        for _ in range(self.steps):
            k1 = self._field(state, memory)
            mid = state + 0.5 * h * k1
            k2 = self._field(mid, memory)
            state = state + h * k2
            target_memory = self.memory_update(state)
            memory = memory + h * (target_memory - memory)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * (state - x)


class CLDyNRequirementSemanticCompensation(nn.Module):
    """
    Detector adaptation of CLDyN RSC/A2SI.
    A basis-vector bank and orthogonal architecture prototypes choose among
    dynamic 1x1 / 3x3 / dilated semantic injections using image and task tokens.
    """
    def __init__(self, channels, embed_dim=64, filter_num=4, set_num=4,
                 vector_num=8, res_init=-1.3, act='silu'):
        super().__init__()
        self.channels = channels
        self.embed_dim = int(embed_dim)
        self.filter_num = max(int(filter_num), 1)
        self.set_num = max(int(set_num), 1)
        self.vector_num = max(int(vector_num), 1)

        self.basis_vector_bank = nn.Parameter(
            torch.randn(self.set_num, self.vector_num, self.embed_dim) * 0.02
        )
        proto = torch.randn(max(self.embed_dim, self.set_num), self.set_num)
        proto, _ = torch.linalg.qr(proto, mode='reduced')
        proto = proto[:self.embed_dim].transpose(0, 1).contiguous()
        self.register_buffer("orthogonal_prototypes", proto, persistent=False)

        hidden = max(channels // 4, 16)
        self.img_embed = nn.Sequential(
            ConvNormLayer_fuse(channels, hidden, 3, 1, act=act),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, self.embed_dim),
        )
        self.task_embed = nn.Sequential(
            nn.Linear(channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.assignment = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.filter_num * self.embed_dim),
        )
        self.basis_filter = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        branch_num = min(self.set_num, 4)
        self.branch_num = max(branch_num, 1)
        self.branch_gate = nn.Linear(self.embed_dim, self.branch_num)
        self.channel_gate = nn.Linear(self.embed_dim, channels)

        self.branches = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act),
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            ),
        ][:self.branch_num])
        self.zero_inject = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.xavier_uniform_(self.zero_inject.weight)
        nn.init.zeros_(self.zero_inject.bias)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, x, task_token=None):
        b = x.shape[0]
        img_token = self.img_embed(x)
        if task_token is None:
            task_token = F.adaptive_avg_pool2d(x, 1).flatten(1)
        task_token = self.task_embed(task_token)

        assign_seed = self.assignment(torch.cat([task_token, img_token], dim=-1))
        assign_seed = assign_seed.view(b, self.filter_num, self.embed_dim)
        proto = F.normalize(self.orthogonal_prototypes, dim=-1)
        logits = torch.matmul(F.normalize(assign_seed, dim=-1), proto.t())
        if self.training:
            set_type = F.gumbel_softmax(logits, tau=0.7, hard=True, dim=-1)
        else:
            set_type = F.softmax(logits, dim=-1)

        bank = self.basis_vector_bank.unsqueeze(0).expand(b, -1, -1, -1)
        selected = torch.einsum('bfs,bsvd->bfvd', set_type, bank)
        selected = selected.mean(dim=2)
        task_img = torch.cat([task_token, img_token], dim=-1).unsqueeze(1).expand(-1, self.filter_num, -1)
        basis = self.basis_filter(torch.cat([selected, task_img], dim=-1))
        basis = basis.mean(dim=1)

        branch_weights = F.softmax(self.branch_gate(basis), dim=-1)
        channel_weights = torch.sigmoid(self.channel_gate(basis)).view(b, self.channels, 1, 1)
        branch_outs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outs, dim=1)
        fused = (stacked * branch_weights.view(b, self.branch_num, 1, 1, 1)).sum(dim=1)
        fused = self.zero_inject(fused * channel_weights)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * fused


class CEDCascadeStage(nn.Module):
    """One CEDNet-style encoder-decoder cascade stage over existing FPN levels."""
    def __init__(self, channels, num_levels=3, expansion=1.0, depth_mult=1.0,
                 res_init=-1.6, version='dfine', act='silu'):
        super().__init__()
        self.num_levels = int(num_levels)
        self.top_laterals = nn.ModuleList()
        self.top_blocks = nn.ModuleList()
        self.top_guides = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.top_laterals.append(ConvNormLayer_fuse(channels, channels, 1, 1, act=act))
            self.top_guides.append(nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid()))
            self.top_blocks.append(
                RepNCSPELAN4(channels * 2, channels, channels * 2,
                             round(expansion * channels // 2), round(3 * depth_mult), act=act)
                if version == 'dfine'
                else CSPLayer(channels * 2, channels, round(3 * depth_mult), act=act,
                              expansion=expansion, bottletype=VGGBlock)
            )

        self.bottom_downs = nn.ModuleList()
        self.bottom_blocks = nn.ModuleList()
        self.bottom_guides = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.bottom_downs.append(ConvNormLayer_fuse(channels, channels, 3, 2, act=act))
            self.bottom_guides.append(nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid()))
            self.bottom_blocks.append(
                RepNCSPELAN4(channels * 2, channels, channels * 2,
                             round(expansion * channels // 2), round(3 * depth_mult), act=act)
                if version == 'dfine'
                else CSPLayer(channels * 2, channels, round(3 * depth_mult), act=act,
                              expansion=expansion, bottletype=VGGBlock)
            )
        self.stage_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, feats):
        inner = [feats[-1]]
        for rev_idx in range(self.num_levels - 1, 0, -1):
            pos = self.num_levels - 1 - rev_idx
            high = self.top_laterals[pos](inner[0])
            low = feats[rev_idx - 1]
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
            guide = self.top_guides[pos](high_up)
            fused = torch.cat([high_up, low * (1.0 + guide)], dim=1)
            inner.insert(0, self.top_blocks[pos](fused))

        outs = [inner[0]]
        for idx in range(self.num_levels - 1):
            low = outs[-1]
            high = inner[idx + 1]
            low_down = self.bottom_downs[idx](low)
            if low_down.shape[2:] != high.shape[2:]:
                low_down = F.interpolate(low_down, size=high.shape[2:], mode='bilinear', align_corners=False)
            guide = self.bottom_guides[idx](high)
            fused = torch.cat([low_down, high * (1.0 + guide)], dim=1)
            outs.append(self.bottom_blocks[idx](fused))

        alpha = 0.35 * torch.sigmoid(self.stage_scale)
        return [old + alpha * (new - old) for old, new in zip(feats, outs)]


class CEDCascadeEncoderDecoder(nn.Module):
    """
    CEDNet-style repeated encoder-decoder feature fusion.
    Later stages receive the fully fused high-level output from earlier stages,
    so high-level semantics guide subsequent low-level learning.
    """
    def __init__(self, channels, num_levels=3, stages=2, expansion=1.0,
                 depth_mult=1.0, res_init=-1.6, version='dfine', act='silu'):
        super().__init__()
        self.stages = nn.ModuleList([
            CEDCascadeStage(
                channels, num_levels=num_levels, expansion=expansion,
                depth_mult=depth_mult, res_init=res_init, version=version, act=act,
            )
            for _ in range(max(int(stages), 1))
        ])

    def forward(self, feats):
        for stage in self.stages:
            feats = stage(feats)
        return feats


class DentalUncertaintyGuidedHOIF(nn.Module):
    """
    DUG-HOIF: Dental Uncertainty-Guided High-Order Interleaved Fusion.

    It supports both single-level enhancement and pair-wise FPN/PAN fusion.
    Ablation flags keep the same module class usable from interleave-only to full
    uncertainty-routed high-order omni-axis recalibration.
    """
    def __init__(
        self,
        channels,
        act='silu',
        reduction=8,
        route_branches=4,
        res_init=-2.0,
        use_interleave=True,
        use_multibranch=True,
        use_uncertainty_routing=True,
        use_high_order=True,
        use_omni_gate=True,
    ):
        super().__init__()
        assert route_branches == 4, "DUG-HOIF uses local/context/edge/high_order branches."

        hidden = max(channels // reduction, 16)
        self.channels = channels
        self.use_interleave = bool(use_interleave)
        self.use_multibranch = bool(use_multibranch)
        self.use_uncertainty_routing = bool(use_uncertainty_routing) and self.use_multibranch
        self.use_high_order = bool(use_high_order) and self.use_multibranch
        self.use_omni_gate = bool(use_omni_gate) and self.use_multibranch

        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            get_activation(act),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
            nn.Sigmoid(),
        ) if self.use_interleave else None
        self.cross_proj = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)

        if self.use_multibranch:
            self.local_branch = nn.Sequential(
                ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act),
                ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
            )

            self.context_branch = nn.Sequential(
                nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                get_activation(act),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )

            self.edge_branch = nn.Sequential(
                ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act),
                ConvNormLayer_fuse(channels, channels, 1, 1, act=act),
            )

        if self.use_high_order:
            self.high_order_proj = ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)
            self.high_order_scale = nn.Parameter(torch.tensor(-2.5))

        if self.use_uncertainty_routing:
            self.uncertainty_router = nn.Sequential(
                nn.Conv2d(channels * 2, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                get_activation(act),
                nn.Conv2d(hidden, route_branches, 1, bias=True),
            )
            nn.init.zeros_(self.uncertainty_router[-1].weight)
            nn.init.zeros_(self.uncertainty_router[-1].bias)

        if self.use_omni_gate:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1, bias=False),
                get_activation(act),
                nn.Conv2d(hidden, channels, 1, bias=True),
                nn.Sigmoid(),
            )

            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )

            self.coord_reduce = nn.Sequential(
                nn.Conv2d(channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                get_activation(act),
            )
            self.coord_h = nn.Conv2d(hidden, channels, 1, bias=True)
            self.coord_w = nn.Conv2d(hidden, channels, 1, bias=True)

        self.out_proj = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _coord_gate(self, x):
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.coord_reduce(y)
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        gate_h = torch.sigmoid(self.coord_h(y_h))
        gate_w = torch.sigmoid(self.coord_w(y_w))
        return gate_h * gate_w

    @staticmethod
    def _local_var(x, kernel_size=5):
        pad = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x.pow(2), kernel_size=kernel_size, stride=1, padding=pad)
        var = (mean_sq - mean.pow(2)).relu()
        return mean, var

    def _interleave_fuse(self, feat_a, feat_b):
        fused0 = torch.cat([feat_a, feat_b], dim=1)
        if self.use_interleave:
            gate_a, gate_b = self.cross_gate(fused0).split(self.channels, dim=1)
            z1 = feat_a + gate_b * feat_b
            z2 = feat_b + gate_a * feat_a
            return self.cross_proj(torch.cat([z1, z2], dim=1))
        return self.cross_proj(fused0)

    @staticmethod
    def _near_identity_gate(gate):
        return 1.0 + 0.5 * (gate - 0.5)

    def forward(self, x, y=None, verbose=False):
        log_shape('DUG-HOIF/x', x, verbose)
        if y is not None:
            log_shape('DUG-HOIF/y', y, verbose)
            z = self._interleave_fuse(x, y)
        else:
            z = x
        log_shape('DUG-HOIF/z', z, verbose)

        if not self.use_multibranch:
            return z

        local_mean, local_var = self._local_var(z, kernel_size=5)
        local = self.local_branch(z)
        context = self.context_branch(z)
        edge = self.edge_branch(z - local_mean)

        candidates = [local, context, edge]
        if self.use_high_order:
            order2 = local * torch.sigmoid(context)
            order3 = order2 * torch.sigmoid(edge)
            high_order = self.high_order_proj(torch.cat([order2, order3], dim=1))
            high_order_alpha = 0.5 * torch.sigmoid(self.high_order_scale)
            candidates.append(local + high_order_alpha * high_order)
            log_shape('DUG-HOIF/high_order', high_order, verbose)

        log_shape('DUG-HOIF/local', local, verbose)
        log_shape('DUG-HOIF/context', context, verbose)
        log_shape('DUG-HOIF/edge', edge, verbose)

        if self.use_uncertainty_routing:
            route_logits = self.uncertainty_router(torch.cat([z, local_var], dim=1))
            route_logits = route_logits[:, :len(candidates)]
            route = F.softmax(route_logits, dim=1)
            stacked = torch.stack(candidates, dim=1)
            fused = (route[:, :, None, :, :] * stacked).sum(dim=1)
        else:
            fused = torch.stack(candidates, dim=0).mean(dim=0)

        log_shape('DUG-HOIF/routed_fused', fused, verbose)

        if self.use_omni_gate:
            channel_gate = self.channel_gate(fused)
            spatial_stats = torch.cat(
                [fused.mean(dim=1, keepdim=True), fused.max(dim=1, keepdim=True).values],
                dim=1,
            )
            spatial_gate = self.spatial_gate(spatial_stats)
            coord_gate = self._coord_gate(fused)
            fused = (
                fused
                * self._near_identity_gate(channel_gate)
                * self._near_identity_gate(spatial_gate)
                * self._near_identity_gate(coord_gate)
            )

        fused = self.out_proj(fused)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        out = z + alpha * fused
        log_shape('DUG-HOIF/out', out, verbose)
        return out


class AdaptiveFourierSpectralModulationV2(nn.Module):
    """Input-adaptive Fourier band modulation for D-SWFM."""
    def __init__(self, channels, res_init=-2.2, act='silu', strength=0.25, dynamic=True):
        super().__init__()
        self.channels = channels
        self.strength = float(strength)
        self.dynamic = bool(dynamic)

        self.pre = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
        self.post = ConvNormLayer_fuse(channels, channels, 1, 1, act=act)

        hidden = max(channels // 8, 16)
        self.base_band_logits = nn.Parameter(torch.zeros(channels, 3))
        self.dynamic_band = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 3, 1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.freq_scale = nn.Parameter(torch.tensor(float(res_init)))
        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _radial_bands(h, w, wf, device, dtype):
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, wf)
        r = torch.sqrt(fx.square() + fy.square())
        low = torch.exp(-(r / 0.16).square())
        mid = torch.exp(-((r - 0.28) / 0.16).square())
        high = (1.0 - torch.exp(-(r / 0.34).square())).clamp(0, 1)
        bands = torch.stack([low, mid, high], dim=0)
        return bands - bands.mean(dim=(1, 2), keepdim=True)

    def forward(self, x):
        z = self.pre(x)
        b, c, h, w = z.shape
        device_type = 'cuda' if z.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            z32 = z.float()
            spec = torch.fft.rfft2(z32, norm='ortho')
            bands = self._radial_bands(h, w, spec.shape[-1], z.device, z32.dtype)
            logits = self.base_band_logits.float().unsqueeze(0)
            if self.dynamic:
                dyn = self.dynamic_band(z32).view(b, c, 3).float()
                logits = logits + 0.25 * torch.tanh(dyn)
            band_weights = torch.softmax(logits, dim=-1)
            modulation = torch.einsum('bck,khw->bchw', band_weights, bands)
            freq_strength = self.strength * torch.sigmoid(self.freq_scale.float())
            gain = 1.0 + freq_strength * torch.tanh(modulation)
            filtered = torch.fft.irfft2(spec * gain, s=(h, w), norm='ortho')

        filtered = filtered.to(dtype=x.dtype)
        delta = filtered - z
        update = self.post(delta) * self.channel_gate(z)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return x + alpha * update


class DentalSpectralWaveletRefinerV2(nn.Module):
    """
    D-SWFM: Dental Spectral-Wavelet Feature Modulation.

    It first applies adaptive Fourier band modulation per feature level, then
    transfers fine detail to coarser levels and optionally sends coarse semantic
    compensation back to finer levels.
    """
    def __init__(
        self,
        channels,
        num_levels=3,
        res_init=-2.0,
        act='silu',
        use_bidirectional=True,
        use_dynamic_fourier=True,
        use_haar_detail=True,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.use_bidirectional = bool(use_bidirectional)
        self.use_haar_detail = bool(use_haar_detail)

        self.spectral = nn.ModuleList([
            AdaptiveFourierSpectralModulationV2(
                channels,
                res_init=res_init,
                act=act,
                dynamic=use_dynamic_fourier,
            )
            for _ in range(self.num_levels)
        ])

        self.detail_guides = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])
        self.coarse_updates = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])

        self.semantic_guides = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 1, 1, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])
        self.fine_updates = nn.ModuleList([
            ConvNormLayer_fuse(channels, channels, 3, 1, g=channels, act=act)
            for _ in range(max(self.num_levels - 1, 0))
        ])

        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    @staticmethod
    def _high_pass(x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - smooth

    @staticmethod
    def _haar_detail_magnitude(x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            x_pad = x

        x00 = x_pad[..., 0::2, 0::2]
        x01 = x_pad[..., 0::2, 1::2]
        x10 = x_pad[..., 1::2, 0::2]
        x11 = x_pad[..., 1::2, 1::2]

        lh = (x00 - x01 + x10 - x11) * 0.25
        hl = (x00 + x01 - x10 - x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        detail = (lh.abs() + hl.abs() + hh.abs()) / 3.0
        return F.interpolate(detail, size=(h, w), mode='bilinear', align_corners=False)

    def forward(self, feats, verbose=False):
        assert len(feats) == self.num_levels

        feats = [self.spectral[i](x) for i, x in enumerate(feats)]
        outs = list(feats)
        alpha = 0.35 * torch.sigmoid(self.res_scale)

        for i in range(self.num_levels - 1):
            fine = outs[i]
            coarse = outs[i + 1]
            signed_detail = self._high_pass(fine)
            if self.use_haar_detail:
                detail_mag = self._haar_detail_magnitude(fine)
                detail = signed_detail * (1.0 + torch.sigmoid(detail_mag))
            else:
                detail = signed_detail
            detail_down = F.interpolate(
                detail,
                size=coarse.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            guide = torch.sigmoid(self.detail_guides[i](detail_down))
            update = self.coarse_updates[i](coarse + guide * detail_down)
            outs[i + 1] = coarse + alpha * (update - coarse)

        if self.use_bidirectional:
            for i in range(self.num_levels - 2, -1, -1):
                fine = outs[i]
                coarse = outs[i + 1]
                coarse_up = F.interpolate(
                    coarse,
                    size=fine.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                sem_gate = torch.sigmoid(self.semantic_guides[i](coarse_up))
                update = self.fine_updates[i](fine + sem_gate * coarse_up)
                outs[i] = fine + alpha * (update - fine)

        if verbose:
            for i, out in enumerate(outs):
                log_shape(f'D-SWFM/out_{i}', out, True)
        return outs


class DentalCEDFusionUnit(nn.Module):
    """
    Stable pair-wise fusion unit for D-CED.

    Base fusion uses positive normalized weights. When enabled, DUG-HOIF adds
    rich dental-aware high-order fusion and a gate mixes base/rich branches.
    """
    def __init__(
        self,
        channels,
        expansion=1.0,
        depth_mult=1.0,
        version='dfine',
        act='silu',
        res_init=-2.0,
        use_rich_fusion=True,
        dug_kwargs=None,
    ):
        super().__init__()
        self.channels = channels
        self.use_rich_fusion = bool(use_rich_fusion)
        self.fusion_weights = nn.Parameter(torch.ones(2, dtype=torch.float32))

        dug_kwargs = {} if dug_kwargs is None else dict(dug_kwargs)
        self.rich_fusion = DentalUncertaintyGuidedHOIF(channels, act=act, **dug_kwargs) \
            if self.use_rich_fusion else None
        self.mix_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid(),
        ) if self.use_rich_fusion else None

        if version == 'dfine':
            self.refine = RepNCSPELAN4(
                channels,
                channels,
                channels * 2,
                round(expansion * channels // 2),
                round(3 * depth_mult),
                act=act,
            )
        else:
            self.refine = CSPLayer(
                channels,
                channels,
                round(3 * depth_mult),
                act=act,
                expansion=expansion,
                bottletype=VGGBlock,
            )

        self.res_scale = nn.Parameter(torch.tensor(float(res_init)))

    def _norm_weights(self):
        w = F.relu(self.fusion_weights)
        return w / (w.sum() + 1e-4)

    def forward(self, a, b, verbose=False):
        w = self._norm_weights()
        base = w[0] * a + w[1] * b
        if self.use_rich_fusion:
            rich = self.rich_fusion(a, b, verbose=verbose)
            gate = self.mix_gate(torch.cat([base, rich], dim=1))
            fused = base + gate * (rich - base)
        else:
            fused = base

        refined = self.refine(fused)
        alpha = 0.35 * torch.sigmoid(self.res_scale)
        return base + alpha * (refined - base)


class DentalCEDCascadeStageV2(nn.Module):
    """Dental-aware CEDNet-style top-down/bottom-up cascade stage."""
    def __init__(
        self,
        channels,
        num_levels=3,
        expansion=1.0,
        depth_mult=1.0,
        res_init=-1.8,
        version='dfine',
        act='silu',
        use_dug_hoif=True,
        dug_kwargs=None,
    ):
        super().__init__()
        self.num_levels = int(num_levels)

        self.top_laterals = nn.ModuleList()
        self.top_fusions = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.top_laterals.append(ConvNormLayer_fuse(channels, channels, 1, 1, act=act))
            self.top_fusions.append(
                DentalCEDFusionUnit(
                    channels,
                    expansion=expansion,
                    depth_mult=depth_mult,
                    version=version,
                    act=act,
                    res_init=res_init,
                    use_rich_fusion=use_dug_hoif,
                    dug_kwargs=dug_kwargs,
                )
            )

        self.bottom_downs = nn.ModuleList()
        self.bottom_fusions = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.bottom_downs.append(ConvNormLayer_fuse(channels, channels, 3, 2, act=act))
            self.bottom_fusions.append(
                DentalCEDFusionUnit(
                    channels,
                    expansion=expansion,
                    depth_mult=depth_mult,
                    version=version,
                    act=act,
                    res_init=res_init,
                    use_rich_fusion=use_dug_hoif,
                    dug_kwargs=dug_kwargs,
                )
            )

        self.stage_scale = nn.Parameter(torch.tensor(float(res_init)))

    def forward(self, feats, verbose=False):
        assert len(feats) == self.num_levels

        inner = [feats[-1]]
        for rev_idx in range(self.num_levels - 1, 0, -1):
            pos = self.num_levels - 1 - rev_idx
            high = self.top_laterals[pos](inner[0])
            low = feats[rev_idx - 1]
            high_up = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
            fused = self.top_fusions[pos](high_up, low, verbose=verbose)
            inner.insert(0, fused)

        outs = [inner[0]]
        for idx in range(self.num_levels - 1):
            low = outs[-1]
            high = inner[idx + 1]
            low_down = self.bottom_downs[idx](low)
            if low_down.shape[2:] != high.shape[2:]:
                low_down = F.interpolate(low_down, size=high.shape[2:], mode='bilinear', align_corners=False)
            fused = self.bottom_fusions[idx](low_down, high, verbose=verbose)
            outs.append(fused)

        alpha = 0.35 * torch.sigmoid(self.stage_scale)
        return [old + alpha * (new - old) for old, new in zip(feats, outs)]


class DentalCEDCascadeEncoderDecoderV2(nn.Module):
    """D-CED: Dental-aware Cascaded Encoder-Decoder Fusion."""
    def __init__(
        self,
        channels,
        num_levels=3,
        stages=2,
        expansion=1.0,
        depth_mult=1.0,
        res_init=-1.8,
        version='dfine',
        act='silu',
        use_dug_hoif=True,
        dug_kwargs=None,
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            DentalCEDCascadeStageV2(
                channels,
                num_levels=num_levels,
                expansion=expansion,
                depth_mult=depth_mult,
                res_init=res_init,
                version=version,
                act=act,
                use_dug_hoif=use_dug_hoif,
                dug_kwargs=dug_kwargs,
            )
            for _ in range(max(int(stages), 1))
        ])

    def forward(self, feats, verbose=False):
        for stage in self.stages:
            feats = stage(feats, verbose=verbose)
        return feats


class AdaptiveInterleaveFusion(nn.Module):
    def __init__(self, channels, fuse_op='cat', reduction=4, act='silu'):
        super().__init__()
        hidden = max((channels * 2) // reduction, 16)
        self.channels = channels
        self.fuse_op = fuse_op
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            get_activation(act),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.out_proj = ConvNormLayer_fuse(channels * 2, channels, 1, 1, bias=False, act=act) \
            if fuse_op == 'sum' else nn.Identity()

    def forward(self, feat_a, feat_b):
        fused = torch.cat([feat_a, feat_b], dim=1)
        gate_a, gate_b = self.channel_gate(fused).split(self.channels, dim=1)
        feat_a_gated = feat_a * gate_a
        feat_b_gated = feat_b * gate_b
        fused = torch.cat([feat_a + feat_b_gated, feat_b + feat_a_gated], dim=1)
        spatial = torch.cat(
            [fused.mean(dim=1, keepdim=True), fused.max(dim=1, keepdim=True).values],
            dim=1
        )
        fused = fused * self.spatial_gate(spatial)
        return self.out_proj(fused)


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.02,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 enc_mi_steps=1,
                 use_hfsa=False,
                 use_stat_denoise=False,
                 use_multipath_fusion=False,
                 use_dynamic_rf=False,
                 use_wav_enhance=False,
                 use_mpc_multiscale_perception=False,
                 use_pqg_high_frequency=False,
                 pqg_high_freq_stages=[0, 1],
                 use_cmfa_fusion=False,
                 use_pfesa=False,
                 pfesa_cutoff=0.10,
                 use_ucan_lkd=False,
                 use_ucm_uncertainty=False,
                 use_retinex_prior=False,
                 use_hooa_attention=False,
                 use_multinex_prior=False,
                 use_msck_refiner=False,
                 msck_kernels=[3, 5, 7],
                 use_csaf_filter=False,
                 csaf_filter_scale=4,
                 use_csaf_dmssp=False,
                 use_rdvq_refiner=False,
                 rdvq_codebook_size=64,
                 rdvq_temperature=0.7,
                 use_datpp_refiner=False,
                 datpp_refiner_stages=[1, 2],
                 datpp_n_heads=8,
                 datpp_n_groups=4,
                 datpp_stride=2,
                 datpp_offset_range_factor=2.0,
                 use_dcsam_dual_consistency=False,
                 use_iqp_quality_prior=False,
                 iqp_codebook_size=16,
                 use_dformer_geometry_prior=False,
                 use_cdml_domain_adapter=False,
                 use_efruf_gradient_fusion=False,
                 efruf_unfold_steps=2,
                 use_sinf_spatial_field=False,
                 sinf_num_blocks=2,
                 use_smwg_spectral_wavelet=False,
                 use_dug_hoif_fusion=False,
                 dug_hoif_use_interleave=True,
                 dug_hoif_use_multibranch=True,
                 dug_hoif_use_uncertainty_routing=True,
                 dug_hoif_use_high_order=True,
                 dug_hoif_use_omni_gate=True,
                 dug_hoif_res_init=-2.0,
                 use_dswfm=False,
                 dswfm_use_dynamic_fourier=True,
                 dswfm_use_haar_detail=True,
                 dswfm_use_bidirectional=True,
                 dswfm_res_init=-2.0,
                 use_dced_cascade=False,
                 dced_stages=1,
                 dced_use_dug_hoif=True,
                 dced_res_init=-1.8,
                 use_structural_entropy_frequency_refiner=False,
                 sefs_res_init=-2.0,
                 use_rot_equivariant_detail_refiner=False,
                 roteq_res_init=-2.0,
                 use_hedp_prompt_refiner=False,
                 hedp_prompt_count=4,
                 hedp_res_init=-2.1,
                 use_sampg_sst_refiner=False,
                 sampg_res_init=-2.0,
                 use_segd_entropy_refiner=False,
                 segd_res_init=-2.0,
                 use_fsdetr_frequency_spatial_refiner=False,
                 fsdetr_res_init=-2.0,
                 use_df2net_dynamic_fusion=False,
                 df2net_res_init=-2.0,
                 use_sdconet_saliency_refiner=False,
                 sdconet_res_init=-2.0,
                 use_idgbr_boundary_refiner=False,
                 idgbr_res_init=-2.0,
                 use_bilalora_layer_adapter=False,
                 bilalora_rank=8,
                 bilalora_topk=2,
                 use_surmt_structure_bridge=False,
                 use_mav_sparse_interaction=False,
                 mav_sparse_ratio=0.25,
                 use_cnm_ode_refiner=False,
                 cnm_ode_steps=3,
                 cnm_ode_step_size=0.35,
                 cnm_ode_res_init=-1.4,
                 use_cldyn_rsc=False,
                 cldyn_filter_num=4,
                 cldyn_set_num=4,
                 cldyn_vector_num=8,
                 cldyn_res_init=-1.3,
                 use_ced_cascade_refiner=False,
                 ced_cascade_stages=1,
                 ced_cascade_expansion=0.5,
                 ced_cascade_res_init=-1.6,
                 paper_refine_apply='proj',
                 apply_extra_modules='proj',
                 apply_denoise_modules=None,
                 apply_refine_modules=None,
                 ):

        super().__init__()
        self.enc_mi_steps = enc_mi_steps
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_hfsa = use_hfsa
        self.use_stat_denoise = use_stat_denoise
        self.use_multipath_fusion = use_multipath_fusion
        self.use_dynamic_rf = use_dynamic_rf
        self.use_wav_enhance = use_wav_enhance
        self.use_cmfa_fusion = use_cmfa_fusion
        self.use_pfesa = use_pfesa
        self.use_ucan_lkd = use_ucan_lkd
        self.use_ucm_uncertainty = use_ucm_uncertainty
        self.use_retinex_prior = use_retinex_prior
        self.use_hooa_attention = use_hooa_attention
        self.use_multinex_prior = use_multinex_prior
        self.use_msck_refiner = bool(use_msck_refiner)
        self.use_csaf_filter = bool(use_csaf_filter)
        self.use_csaf_dmssp = bool(use_csaf_dmssp)
        self.use_rdvq_refiner = bool(use_rdvq_refiner)
        self.use_datpp_refiner = bool(use_datpp_refiner)
        self.use_dcsam_dual_consistency = bool(use_dcsam_dual_consistency)
        self.use_iqp_quality_prior = bool(use_iqp_quality_prior)
        self.use_dformer_geometry_prior = bool(use_dformer_geometry_prior)
        self.use_cdml_domain_adapter = bool(use_cdml_domain_adapter)
        self.use_efruf_gradient_fusion = bool(use_efruf_gradient_fusion)
        self.use_sinf_spatial_field = bool(use_sinf_spatial_field)
        self.use_smwg_spectral_wavelet = bool(use_smwg_spectral_wavelet)
        self.use_dug_hoif_fusion = bool(use_dug_hoif_fusion)
        self.use_dswfm = bool(use_dswfm)
        self.use_dced_cascade = bool(use_dced_cascade)
        self.dced_stages = int(dced_stages)
        self.dced_use_dug_hoif = bool(dced_use_dug_hoif)
        self.dug_hoif_kwargs = {
            "use_interleave": bool(dug_hoif_use_interleave),
            "use_multibranch": bool(dug_hoif_use_multibranch),
            "use_uncertainty_routing": bool(dug_hoif_use_uncertainty_routing),
            "use_high_order": bool(dug_hoif_use_high_order),
            "use_omni_gate": bool(dug_hoif_use_omni_gate),
            "res_init": float(dug_hoif_res_init),
        }
        self.use_structural_entropy_frequency_refiner = bool(use_structural_entropy_frequency_refiner)
        self.use_rot_equivariant_detail_refiner = bool(use_rot_equivariant_detail_refiner)
        self.use_hedp_prompt_refiner = bool(use_hedp_prompt_refiner)
        self.use_sampg_sst_refiner = bool(use_sampg_sst_refiner)
        self.use_segd_entropy_refiner = bool(use_segd_entropy_refiner)
        self.use_fsdetr_frequency_spatial_refiner = bool(use_fsdetr_frequency_spatial_refiner)
        self.use_df2net_dynamic_fusion = bool(use_df2net_dynamic_fusion)
        self.use_sdconet_saliency_refiner = bool(use_sdconet_saliency_refiner)
        self.use_idgbr_boundary_refiner = bool(use_idgbr_boundary_refiner)
        self.use_bilalora_layer_adapter = bool(use_bilalora_layer_adapter)
        self.use_surmt_structure_bridge = bool(use_surmt_structure_bridge)
        self.use_mav_sparse_interaction = bool(use_mav_sparse_interaction)
        self.use_cnm_ode_refiner = bool(use_cnm_ode_refiner)
        self.use_cldyn_rsc = bool(use_cldyn_rsc)
        self.use_ced_cascade_refiner = bool(use_ced_cascade_refiner)
        self.msck_kernels = tuple(msck_kernels or (3, 5, 7))
        self.csaf_filter_scale = int(csaf_filter_scale)
        self.datpp_refiner_stages = set(datpp_refiner_stages or [])
        self.pqg_high_freq_stages = set(pqg_high_freq_stages or [])
        self.apply_extra_modules = apply_extra_modules
        self.apply_denoise_modules = apply_extra_modules if apply_denoise_modules is None else apply_denoise_modules
        self.apply_refine_modules = apply_extra_modules if apply_refine_modules is None else apply_refine_modules
        self.paper_refine_apply = paper_refine_apply

        valid_modes = ('proj', 'outs', 'both', 'none')
        if self.apply_extra_modules not in valid_modes:
            raise ValueError(f"apply_extra_modules must be one of {valid_modes}, got {self.apply_extra_modules!r}")
        if self.apply_denoise_modules not in valid_modes:
            raise ValueError(f"apply_denoise_modules must be one of {valid_modes}, got {self.apply_denoise_modules!r}")
        if self.apply_refine_modules not in valid_modes:
            raise ValueError(f"apply_refine_modules must be one of {valid_modes}, got {self.apply_refine_modules!r}")
        if self.paper_refine_apply not in valid_modes:
            raise ValueError(f"paper_refine_apply must be one of {valid_modes}, got {self.paper_refine_apply!r}")

        num_levels = len(in_channels)
        if self.use_wav_enhance:
            if not _WAVELETS_AVAILABLE:
                raise RuntimeError("use_wav_enhance=True 但 pytorch_wavelets 未安装。")
            self.stat_denoise = nn.ModuleList([WavEnhance(hidden_dim) for _ in range(num_levels)])
        elif self.use_stat_denoise:
            self.stat_denoise = nn.ModuleList(
                [StatisticalGuidedDenoising(channels=hidden_dim) for _ in range(num_levels)]
            )
        else:
            self.stat_denoise = None

        self.multipath_fusion = nn.ModuleList(
            [MultiPathFeatureFusion(in_channels=hidden_dim) for _ in range(num_levels)]
        ) if self.use_multipath_fusion else None
        self.dynamic_rf = nn.ModuleList(
            [DynamicReceptiveField(in_channels=hidden_dim) for _ in range(num_levels)]
        ) if self.use_dynamic_rf else None
        self.pfesa = nn.ModuleList(
            [ParameterFreeEdgeStructureAttention(cutoff=pfesa_cutoff) for _ in range(num_levels)]
        ) if self.use_pfesa else None
        self.ucan_lkd = nn.ModuleList(
            [LargeKernelDistillationBlock(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_ucan_lkd else None
        self.ucm_uncertainty = nn.ModuleList(
            [UncertaintyContextRefiner(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_ucm_uncertainty else None
        self.retinex_prior = nn.ModuleList(
            [RetinexPriorGate(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_retinex_prior else None
        self.hooa_attention = nn.ModuleList(
            [HighOrderOmniAttention(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_hooa_attention else None
        self.multinex_prior = nn.ModuleList(
            [MultiPriorRetinexRefiner(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_multinex_prior else None
        self.msck_refiner = nn.ModuleList(
            [MSCKChineseKnotRefiner(hidden_dim, kernels=self.msck_kernels, act=act) for _ in range(num_levels)]
        ) if self.use_msck_refiner else None
        self.csaf_filter = nn.ModuleList(
            [CSAFAdaptiveFilter(hidden_dim, scale=self.csaf_filter_scale, act=act) for _ in range(num_levels)]
        ) if self.use_csaf_filter else None
        self.csaf_dmssp = nn.ModuleList(
            [CSAFDMSSPRefiner(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_csaf_dmssp else None
        self.rdvq_refiner = nn.ModuleList(
            [
                RDVQCodebookRefiner(
                    hidden_dim,
                    codebook_size=rdvq_codebook_size,
                    temperature=rdvq_temperature,
                    act=act,
                )
                for _ in range(num_levels)
            ]
        ) if self.use_rdvq_refiner else None
        self.datpp_refiner = nn.ModuleList(
            [
                DATPPDeformableRefiner(
                    hidden_dim,
                    n_heads=datpp_n_heads,
                    n_groups=datpp_n_groups,
                    stride=datpp_stride,
                    offset_range_factor=datpp_offset_range_factor,
                    act=act,
                ) if i in self.datpp_refiner_stages else nn.Identity()
                for i in range(num_levels)
            ]
        ) if self.use_datpp_refiner else None
        self.cdml_domain_adapter = nn.ModuleList(
            [CDMLDomainAffineAdapter(hidden_dim) for _ in range(num_levels)]
        ) if self.use_cdml_domain_adapter else None
        self.iqp_quality_prior = nn.ModuleList(
            [IQPQualityPriorRefiner(hidden_dim, codebook_size=iqp_codebook_size, act=act) for _ in range(num_levels)]
        ) if self.use_iqp_quality_prior else None
        self.dformer_geometry_prior = nn.ModuleList(
            [DFormerGeometryPriorRefiner(hidden_dim, act=act) for _ in range(num_levels)]
        ) if self.use_dformer_geometry_prior else None
        self.dcsam_dual_consistency = DCSAMDualConsistencyRefiner(
            hidden_dim,
            num_levels=num_levels,
            act=act,
        ) if self.use_dcsam_dual_consistency else None
        self.efruf_gradient_fusion = nn.ModuleList(
            [
                EFRUFGradientTransferRefiner(
                    hidden_dim,
                    unfold_steps=efruf_unfold_steps,
                    act=act,
                )
                for _ in range(num_levels)
            ]
        ) if self.use_efruf_gradient_fusion else None
        self.sinf_spatial_field = nn.ModuleList(
            [
                SINFSpatialImplicitFieldRefiner(
                    hidden_dim,
                    num_blocks=sinf_num_blocks,
                    act=act,
                )
                for _ in range(num_levels)
            ]
        ) if self.use_sinf_spatial_field else None
        self.smwg_spectral_wavelet = SMWGSpectralWaveletRefiner(
            hidden_dim,
            num_levels=num_levels,
            act=act,
        ) if self.use_smwg_spectral_wavelet else None
        self.dswfm = DentalSpectralWaveletRefinerV2(
            hidden_dim,
            num_levels=num_levels,
            res_init=dswfm_res_init,
            act=act,
            use_bidirectional=dswfm_use_bidirectional,
            use_dynamic_fourier=dswfm_use_dynamic_fourier,
            use_haar_detail=dswfm_use_haar_detail,
        ) if self.use_dswfm else None
        self.dced_cascade = DentalCEDCascadeEncoderDecoderV2(
            hidden_dim,
            num_levels=num_levels,
            stages=dced_stages,
            expansion=expansion,
            depth_mult=depth_mult,
            res_init=dced_res_init,
            version=version,
            act=act,
            use_dug_hoif=dced_use_dug_hoif,
            dug_kwargs=self.dug_hoif_kwargs,
        ) if self.use_dced_cascade else None
        self.structural_entropy_frequency_refiner = StructuralEntropyFrequencySaliencyRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=sefs_res_init,
            act=act,
        ) if self.use_structural_entropy_frequency_refiner else None
        self.rot_equivariant_detail_refiner = nn.ModuleList(
            [RotEquivariantDetailRefiner(hidden_dim, res_init=roteq_res_init, act=act) for _ in range(num_levels)]
        ) if self.use_rot_equivariant_detail_refiner else None
        self.hedp_prompt_refiner = HybridEnergyDistancePromptRefiner(
            hidden_dim,
            num_levels=num_levels,
            prompt_count=hedp_prompt_count,
            res_init=hedp_res_init,
            act=act,
        ) if self.use_hedp_prompt_refiner else None
        self.sampg_sst_refiner = SAMPriorSSTScanRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=sampg_res_init,
            act=act,
        ) if self.use_sampg_sst_refiner else None
        self.segd_entropy_refiner = SEGDStructuralEntropyRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=segd_res_init,
            act=act,
        ) if self.use_segd_entropy_refiner else None
        self.fsdetr_frequency_spatial_refiner = FSDETRFrequencySpatialRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=fsdetr_res_init,
            act=act,
        ) if self.use_fsdetr_frequency_spatial_refiner else None
        self.df2net_dynamic_fusion = DF2NetDifferentialDynamicFusionRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=df2net_res_init,
            act=act,
        ) if self.use_df2net_dynamic_fusion else None
        self.sdconet_saliency_refiner = SDCoNetSaliencyCollaborationRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=sdconet_res_init,
            act=act,
        ) if self.use_sdconet_saliency_refiner else None
        self.idgbr_boundary_refiner = IDGBRBoundaryRefinementRefiner(
            hidden_dim,
            num_levels=num_levels,
            res_init=idgbr_res_init,
            act=act,
        ) if self.use_idgbr_boundary_refiner else None
        self.bilalora_layer_adapter = BiLaLoRALayerPositionedAdapter(
            hidden_dim,
            num_levels=num_levels,
            rank=bilalora_rank,
            topk=bilalora_topk,
            act=act,
        ) if self.use_bilalora_layer_adapter else None
        self.surmt_structure_bridge = SURMTSemanticStructureBridge(
            hidden_dim,
            num_levels=num_levels,
            act=act,
        ) if self.use_surmt_structure_bridge else None
        self.mav_sparse_interaction = MAVSparseInteractionRefiner(
            hidden_dim,
            num_levels=num_levels,
            sparse_ratio=mav_sparse_ratio,
            act=act,
        ) if self.use_mav_sparse_interaction else None
        self.cnm_ode_refiner = nn.ModuleList(
            [
                CNMContinuousODERefiner(
                    hidden_dim,
                    steps=cnm_ode_steps,
                    step_size=cnm_ode_step_size,
                    res_init=cnm_ode_res_init,
                    act=act,
                )
                for _ in range(num_levels)
            ]
        ) if self.use_cnm_ode_refiner else None
        self.cldyn_rsc = nn.ModuleList(
            [
                CLDyNRequirementSemanticCompensation(
                    hidden_dim,
                    filter_num=cldyn_filter_num,
                    set_num=cldyn_set_num,
                    vector_num=cldyn_vector_num,
                    res_init=cldyn_res_init,
                    act=act,
                )
                for _ in range(num_levels)
            ]
        ) if self.use_cldyn_rsc else None
        self.ced_cascade_refiner = CEDCascadeEncoderDecoder(
            hidden_dim,
            num_levels=num_levels,
            stages=ced_cascade_stages,
            expansion=ced_cascade_expansion,
            depth_mult=depth_mult,
            res_init=ced_cascade_res_init,
            version=version,
            act=act,
        ) if self.use_ced_cascade_refiner else None

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        self.perception_blocks = nn.ModuleList([
            MultiScalePerceptionBlock(hidden_dim, act=act) if use_mpc_multiscale_perception else nn.Identity()
            for _ in in_channels
        ])
        self.high_frequency_blocks = nn.ModuleList([
            HighFrequencyEnhancer(hidden_dim, act=act) if use_pqg_high_frequency and i in self.pqg_high_freq_stages
            else nn.Identity()
            for i in range(len(in_channels))
        ])

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        if self.use_hfsa:
            self.hfsas_enc = nn.ModuleList([
                HFSA(dim=hidden_dim, reduction=8) for _ in range(len(use_encoder_idx))
            ])
            self.hfsas_out = nn.ModuleList([
                HFSA(dim=hidden_dim, reduction=8) for _ in range(len(in_channels))
            ])
        # === 优化点 B: DySample 初始化 ===
        # 替代 F.interpolate
        self.upsample_ops = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
             self.upsample_ops.append(DySample(hidden_dim, scale=2, style='lp'))
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.top_down_dug_fusions = nn.ModuleList() if self.use_dug_hoif_fusion else None
        self.top_down_fusions = nn.ModuleList() if (self.use_cmfa_fusion and not self.use_dug_hoif_fusion) else None
        fpn_block_in_channels = hidden_dim if self.use_dug_hoif_fusion else hidden_dim * 2
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(fpn_block_in_channels, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(fpn_block_in_channels, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )
            if self.use_dug_hoif_fusion:
                self.top_down_dug_fusions.append(
                    DentalUncertaintyGuidedHOIF(hidden_dim, act=act, **self.dug_hoif_kwargs)
                )
            elif self.use_cmfa_fusion:
                self.top_down_fusions.append(AdaptiveInterleaveFusion(hidden_dim, fuse_op='cat', act=act))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.bottom_up_dug_fusions = nn.ModuleList() if self.use_dug_hoif_fusion else None
        self.bottom_up_fusions = nn.ModuleList() if (self.use_cmfa_fusion and not self.use_dug_hoif_fusion) else None
        pan_block_in_channels = hidden_dim if self.use_dug_hoif_fusion else hidden_dim * 2
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(pan_block_in_channels, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(pan_block_in_channels, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )
            if self.use_dug_hoif_fusion:
                self.bottom_up_dug_fusions.append(
                    DentalUncertaintyGuidedHOIF(hidden_dim, act=act, **self.dug_hoif_kwargs)
                )
            elif self.use_cmfa_fusion:
                self.bottom_up_fusions.append(AdaptiveInterleaveFusion(hidden_dim, fuse_op='cat', act=act))
        # === 优化点 C: Learnable Position Embedding (初始化) ===
        # 牙齿解剖位置固定，Learnable PE 优于 SinCos
        if self.eval_spatial_size:
            self.row_embed = nn.Embedding(500, hidden_dim // 2) # 假设高度不超过500
            self.col_embed = nn.Embedding(500, hidden_dim // 2) # 假设宽度不超过500
            nn.init.uniform_(self.row_embed.weight)
            nn.init.uniform_(self.col_embed.weight)
        self._reset_parameters()
    # === 优化点 C: Learnable PE (构建函数) ===
    def build_learnable_pos_embed(self, h, w, device):
        x_embed = self.col_embed(torch.arange(w, device=device))
        y_embed = self.row_embed(torch.arange(h, device=device))
        
        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(h, 1, 1),
            y_embed.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
        return pos
    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats = [self.perception_blocks[i](feat) for i, feat in enumerate(proj_feats)]
        proj_feats = [self.high_frequency_blocks[i](feat) for i, feat in enumerate(proj_feats)]

        if self.apply_denoise_modules in ('proj', 'both'):
            proj_feats = self._apply_denoise_stack(proj_feats)
        if self.paper_refine_apply in ('proj', 'both'):
            proj_feats = self._apply_paper_refine_stack(proj_feats)
        if self.apply_refine_modules in ('proj', 'both'):
            proj_feats = self._apply_refine_stack(proj_feats)

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                
                # 使用 Learnable PE 或 SinCos (如果没初始化 Learnable)
                if hasattr(self, 'row_embed'):
                    pos_embed = self.build_learnable_pos_embed(h, w, src_flatten.device)
                    pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
                else:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)

                memory = src_flatten
                for t in range(max(int(getattr(self, "enc_mi_steps", 1)), 1)):
                    memory = self.encoder[i](memory, pos_embed=pos_embed)
                
                x_enc = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                if self.use_hfsa:
                    x_enc = self.hfsas_enc[i](x_enc)
                proj_feats[enc_ind] = x_enc

        # FPN (Top-Down)
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            
            # === 使用 DySample 替代插值 ===
            upsample_feat = self.upsample_ops[len(self.in_channels) - 1 - idx](feat_high)
            # 这里的 concat 维度需要对齐，DySample 输出大小严格为 x2，输入特征图尽量要是偶数
            # 如果尺寸因为 padding 问题不匹配，需要 crop 或 pad (通常 Backbone 输出是对齐的)
            if upsample_feat.shape[2:] != feat_low.shape[2:]:
                upsample_feat = F.interpolate(upsample_feat, size=feat_low.shape[2:], mode='bilinear', align_corners=False)

            fusion_pos = len(self.in_channels) - 1 - idx
            if self.top_down_dug_fusions is not None:
                fused_feat = self.top_down_dug_fusions[fusion_pos](upsample_feat, feat_low)
            elif self.top_down_fusions is not None:
                fused_feat = self.top_down_fusions[fusion_pos](upsample_feat, feat_low)
            else:
                fused_feat = torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # PAN (Bottom-Up)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if downsample_feat.shape[2:] != feat_height.shape[2:]:
                downsample_feat = F.interpolate(
                    downsample_feat,
                    size=feat_height.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            if self.bottom_up_dug_fusions is not None:
                fused_feat = self.bottom_up_dug_fusions[idx](downsample_feat, feat_height)
            elif self.bottom_up_fusions is not None:
                fused_feat = self.bottom_up_fusions[idx](downsample_feat, feat_height)
            else:
                fused_feat = torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)
            
        if self.use_hfsa:
            enhanced_outs = []
            for i, feat in enumerate(outs):
                enhanced_outs.append(self.hfsas_out[i](feat))
            outs = enhanced_outs

        if self.apply_denoise_modules in ('outs', 'both'):
            outs = self._apply_denoise_stack(outs)
        if self.apply_refine_modules in ('outs', 'both'):
            outs = self._apply_refine_stack(outs)
        if self.paper_refine_apply in ('outs', 'both'):
            outs = self._apply_paper_refine_stack(outs)

        return outs

    def _apply_denoise_stack(self, feats):
        if self.stat_denoise is None:
            return feats
        return [self.stat_denoise[i](x) for i, x in enumerate(feats)]

    def _apply_refine_stack(self, feats):
        if self.multipath_fusion is not None:
            feats = [self.multipath_fusion[i](x) for i, x in enumerate(feats)]
        if self.dynamic_rf is not None:
            feats = [self.dynamic_rf[i](x) for i, x in enumerate(feats)]
        if self.msck_refiner is not None:
            feats = [self.msck_refiner[i](x) for i, x in enumerate(feats)]
        if self.csaf_filter is not None:
            feats = [self.csaf_filter[i](x) for i, x in enumerate(feats)]
        if self.csaf_dmssp is not None:
            feats = [self.csaf_dmssp[i](x) for i, x in enumerate(feats)]
        if self.rdvq_refiner is not None:
            feats = [self.rdvq_refiner[i](x) for i, x in enumerate(feats)]
        if self.datpp_refiner is not None:
            feats = [self.datpp_refiner[i](x) for i, x in enumerate(feats)]
        if self.retinex_prior is not None:
            feats = [self.retinex_prior[i](x) for i, x in enumerate(feats)]
        if self.multinex_prior is not None:
            feats = [self.multinex_prior[i](x) for i, x in enumerate(feats)]
        if self.ucan_lkd is not None:
            feats = [self.ucan_lkd[i](x) for i, x in enumerate(feats)]
        if self.ucm_uncertainty is not None:
            feats = [self.ucm_uncertainty[i](x) for i, x in enumerate(feats)]
        if self.pfesa is not None:
            feats = [self.pfesa[i](x) for i, x in enumerate(feats)]
        if self.hooa_attention is not None:
            feats = [self.hooa_attention[i](x) for i, x in enumerate(feats)]
        return feats

    def _apply_paper_refine_stack(self, feats):
        if self.dswfm is not None:
            feats = self.dswfm(feats)
        if self.dced_cascade is not None:
            feats = self.dced_cascade(feats)
        if self.cdml_domain_adapter is not None:
            feats = [self.cdml_domain_adapter[i](x) for i, x in enumerate(feats)]
        if self.efruf_gradient_fusion is not None:
            feats = [self.efruf_gradient_fusion[i](x) for i, x in enumerate(feats)]
        if self.sinf_spatial_field is not None:
            feats = [self.sinf_spatial_field[i](x) for i, x in enumerate(feats)]
        if self.smwg_spectral_wavelet is not None:
            feats = self.smwg_spectral_wavelet(feats)
        if self.structural_entropy_frequency_refiner is not None:
            feats = self.structural_entropy_frequency_refiner(feats)
        if self.rot_equivariant_detail_refiner is not None:
            feats = [self.rot_equivariant_detail_refiner[i](x) for i, x in enumerate(feats)]
        if self.hedp_prompt_refiner is not None:
            feats = self.hedp_prompt_refiner(feats)
        if self.sampg_sst_refiner is not None:
            feats = self.sampg_sst_refiner(feats)
        if self.segd_entropy_refiner is not None:
            feats = self.segd_entropy_refiner(feats)
        if self.fsdetr_frequency_spatial_refiner is not None:
            feats = self.fsdetr_frequency_spatial_refiner(feats)
        if self.df2net_dynamic_fusion is not None:
            feats = self.df2net_dynamic_fusion(feats)
        if self.sdconet_saliency_refiner is not None:
            feats = self.sdconet_saliency_refiner(feats)
        if self.idgbr_boundary_refiner is not None:
            feats = self.idgbr_boundary_refiner(feats)
        if self.bilalora_layer_adapter is not None:
            feats = self.bilalora_layer_adapter(feats)
        if self.surmt_structure_bridge is not None:
            feats = self.surmt_structure_bridge(feats)
        if self.mav_sparse_interaction is not None:
            feats = self.mav_sparse_interaction(feats)
        if self.iqp_quality_prior is not None:
            feats = [self.iqp_quality_prior[i](x) for i, x in enumerate(feats)]
        if self.dformer_geometry_prior is not None:
            feats = [self.dformer_geometry_prior[i](x) for i, x in enumerate(feats)]
        if self.dcsam_dual_consistency is not None:
            feats = self.dcsam_dual_consistency(feats)
        if self.cnm_ode_refiner is not None:
            feats = [self.cnm_ode_refiner[i](x) for i, x in enumerate(feats)]
        if self.cldyn_rsc is not None:
            task_token = torch.stack(
                [F.adaptive_avg_pool2d(x, 1).flatten(1) for x in feats],
                dim=0,
            ).mean(dim=0)
            feats = [self.cldyn_rsc[i](x, task_token=task_token) for i, x in enumerate(feats)]
        if self.ced_cascade_refiner is not None:
            feats = self.ced_cascade_refiner(feats)
        return feats
