import torch
import torch.nn as nn

class CVIM(nn.Module):
    """Cross-View Interaction Module (DDP 优化版)"""
    def __init__(self, c):
        super(CVIM, self).__init__()
        self.scale = c ** -0.5

        # 左视图投影
        self.l_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.l_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.l_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 右视图投影
        self.r_proj1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        )
        self.r_proj3 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 可学习的交互强度
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_l, x_r):
        """
        Args:
            x_l: 左视图特征 [B, C, H, W]
            x_r: 右视图特征 [B, C, H, W]
        Returns:
            out_l, out_r: 交互后的特征 [B, C, H, W]
        """
        # === Query 和 Key 计算 ===
        Q_l = self.l_proj1(x_l)
        Q_l = Q_l.permute(0, 2, 3, 1).contiguous()  # ✅ [B, H, W, C]
        
        Q_r_T = self.r_proj1(x_r)
        Q_r_T = Q_r_T.permute(0, 2, 1, 3).contiguous()  # ✅ [B, H, C, W]

        # === Attention 计算 ===
        attn = torch.matmul(Q_l, Q_r_T) * self.scale  # [B, H, W, W]

        # === Value 计算 ===
        V_l = self.l_proj2(x_l)
        V_l = V_l.permute(0, 2, 3, 1).contiguous()  # ✅ [B, H, W, C]
        
        V_r = self.r_proj2(x_r)
        V_r = V_r.permute(0, 2, 3, 1).contiguous()  # ✅ [B, H, W, C]

        # === 右 -> 左 交互 ===
        attn_r2l = torch.softmax(attn, dim=-1)
        F_r2l = torch.matmul(attn_r2l, V_r)  # [B, H, W, C]
        F_r2l = F_r2l.permute(0, 3, 1, 2).contiguous()  # ✅ [B, C, H, W]
        F_r2l = self.l_proj3(F_r2l)

        # === 左 -> 右 交互 ===
        attn_l2r = attn.permute(0, 1, 3, 2).contiguous()  # ✅ [B, H, W, W]
        attn_l2r = torch.softmax(attn_l2r, dim=-1)
        F_l2r = torch.matmul(attn_l2r, V_l)  # [B, H, W, C]
        F_l2r = F_l2r.permute(0, 3, 1, 2).contiguous()  # ✅ [B, C, H, W]
        F_l2r = self.r_proj3(F_l2r)

        # === 残差更新 ===
        gamma = torch.tanh(self.gamma)  # 限制在 [-1, 1]
        out_l = x_l + gamma * F_r2l
        out_r = x_r + gamma * F_l2r
        
        return out_l, out_r


if __name__ == '__main__':
    # 测试代码
    B, C, H, W = 2, 32, 64, 64
    x_l = torch.randn(B, C, H, W)
    x_r = torch.randn(B, C, H, W)

    cvim = CVIM(C)
    
    # 测试前向传播
    out_l, out_r = cvim(x_l, x_r)
    
    print("=" * 60)
    print("CVIM Module Test")
    print("=" * 60)
    print(f"Input  (left):  {x_l.shape}")
    print(f"Input  (right): {x_r.shape}")
    print(f"Output (left):  {out_l.shape}")
    print(f"Output (right): {out_r.shape}")
    print(f"Gamma value: {cvim.gamma.item():.4f}")
    print("=" * 60)
    
    # 测试内存连续性
    assert out_l.is_contiguous(), "❌ out_l is not contiguous!"
    assert out_r.is_contiguous(), "❌ out_r is not contiguous!"
    print("✅ All outputs are contiguous")
    
    # 测试梯度反向传播
    loss = (out_l.sum() + out_r.sum())
    loss.backward()
    
    print("✅ Backward pass successful")
    print(f"✅ Gamma gradient: {cvim.gamma.grad.item():.6f}")
