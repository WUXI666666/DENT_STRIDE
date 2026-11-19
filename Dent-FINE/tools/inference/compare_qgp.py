"""
步骤1: 单独运行模型并保存注意力热图
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def generate_attention_heatmap(msda_module, image_size=(640, 640), sigma=15):
    """从MSDA模块生成所有query的注意力热图"""
    from scipy.ndimage import gaussian_filter
    
    samp = msda_module.last_sampling_locations
    attn = msda_module.last_attention_weights
    
    if samp is None or attn is None:
        raise RuntimeError("MSDA debug缓存为空")
    
    if isinstance(samp, torch.Tensor):
        samp = samp.detach().cpu().numpy()
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    
    print(f"[DEBUG] sampling_locations shape: {samp.shape}")
    print(f"[DEBUG] attention_weights shape: {attn.shape}")
    
    # 检测格式
    if len(samp.shape) == 5 and samp.shape[-1] == 2:
        B, Q, L, P, _ = samp.shape
        assert B == 1, f"只支持batch_size=1"
        print(f"[INFO] 格式: [B, Q, L, P, 2], Q={Q}, L={L}, P={P}")
    elif len(samp.shape) == 6 and samp.shape[-1] == 2:
        if attn.shape[1:3] != samp.shape[1:3]:
            samp = samp.transpose(0, 2, 1, 3, 4, 5)
            attn = attn.transpose(0, 2, 1, 3, 4)
        B, Q, H_heads, L, P, _ = samp.shape
        assert B == 1
        print(f"[INFO] 格式: [B, Q, H, L, P, 2], Q={Q}, H={H_heads}, L={L}, P={P}")
    else:
        raise ValueError(f"不支持的形状: {samp.shape}")
    
    samp = samp[0]
    attn = attn[0]
    
    H, W = image_size
    heatmaps = np.zeros((Q, H, W), dtype=np.float32)
    
    print(f"[INFO] 生成 {Q} 个query的注意力热图...")
    
    for q in range(Q):
        heatmap = np.zeros((H, W), dtype=np.float32)
        coords = samp[q].reshape(-1, 2)
        weights = attn[q].reshape(-1)
        
        if weights.max() > 0:
            weights = weights / weights.max()
        
        xs = np.clip(coords[:, 0] * W, 0, W-1).astype(int)
        ys = np.clip(coords[:, 1] * H, 0, H-1).astype(int)
        
        for x, y, w in zip(xs, ys, weights):
            heatmap[y, x] += w
        
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmaps[q] = heatmap
        
        if (q + 1) % 50 == 0:
            print(f"  进度: {q+1}/{Q}")
    
    print(f"[INFO] 完成！")
    return heatmaps


def enable_msda_debug_for_decoder(det_model, dec_layer_idx=-1, mi_head_idx=0):
    """为decoder的MSDA模块开启debug"""
    from torch_inf import get_transformer_decoder
    
    decoder = get_transformer_decoder(det_model)
    layers = decoder.layers
    
    if dec_layer_idx < 0:
        dec_layer_idx = len(layers) + dec_layer_idx
    
    layer = layers[dec_layer_idx]
    mi_heads = layer.mi_heads
    assert 0 <= mi_head_idx < len(mi_heads)
    
    msda = mi_heads[mi_head_idx].cross_attn
    msda.enable_debug(True)
    
    print(f"[MSDA] Debug enabled at decoder layer {dec_layer_idx}, mi_head {mi_head_idx}")
    return msda


def load_model(config_path, checkpoint_path, device):
    """加载模型"""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(device)
    model.eval()
    
    return model


def save_attention_results(
    config_path,
    checkpoint_path,
    image_path,
    output_path,
    device='cuda'
):
    """
    运行模型并保存注意力热图和检测结果
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型权重路径
        image_path: 测试图像路径
        output_path: 输出pkl文件路径
        device: 设备
    """
    print("="*80)
    print(f"保存注意力结果到: {output_path}")
    print("="*80)
    
    # 准备图像
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model = load_model(config_path, checkpoint_path, device)
    det_model = model.model
    
    # 开启MSDA debug
    msda = enable_msda_debug_for_decoder(det_model, dec_layer_idx=-1, mi_head_idx=0)
    
    # 前向推理
    print("运行推理...")
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    labels, boxes, scores = output
    
    # 打印检测结果信息
    print(f"检测到 {(scores[0] > 0.3).sum().item()} 个目标 (score > 0.3)")
    print(f"[DEBUG] labels shape: {labels[0].shape}")
    print(f"[DEBUG] boxes shape: {boxes[0].shape}")
    print(f"[DEBUG] scores shape: {scores[0].shape}")
    
    # 生成热图
    print("生成注意力热图...")
    heatmaps = generate_attention_heatmap(msda, image_size=(640, 640), sigma=15)
    
    # 保存结果
    results = {
        'heatmaps': heatmaps,  # [Q, H, W]
        'labels': labels[0].cpu().numpy(),
        'boxes': boxes[0].cpu().numpy(),
        'scores': scores[0].cpu().numpy(),
        'image_path': image_path,
        'checkpoint_path': checkpoint_path,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ 结果已保存到: {output_path}")
    print(f"  - 热图形状: {heatmaps.shape}")
    print(f"  - 标签形状: {results['labels'].shape}")
    print(f"  - 分数形状: {results['scores'].shape}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='保存单个模型的注意力热图')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出pkl文件路径')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='设备 (default: cuda:0)')
    
    args = parser.parse_args()
    
    save_attention_results(
        config_path=args.config,
        checkpoint_path=args.model,
        image_path=args.image,
        output_path=args.output,
        device=args.device
    )
    """
步骤1: 单独运行模型并保存注意力热图
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def generate_attention_heatmap(msda_module, image_size=(640, 640), sigma=15):
    """从MSDA模块生成所有query的注意力热图"""
    from scipy.ndimage import gaussian_filter
    
    samp = msda_module.last_sampling_locations
    attn = msda_module.last_attention_weights
    
    if samp is None or attn is None:
        raise RuntimeError("MSDA debug缓存为空")
    
    if isinstance(samp, torch.Tensor):
        samp = samp.detach().cpu().numpy()
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    
    print(f"[DEBUG] sampling_locations shape: {samp.shape}")
    print(f"[DEBUG] attention_weights shape: {attn.shape}")
    
    # 检测格式
    if len(samp.shape) == 5 and samp.shape[-1] == 2:
        B, Q, L, P, _ = samp.shape
        assert B == 1, f"只支持batch_size=1"
        print(f"[INFO] 格式: [B, Q, L, P, 2], Q={Q}, L={L}, P={P}")
    elif len(samp.shape) == 6 and samp.shape[-1] == 2:
        if attn.shape[1:3] != samp.shape[1:3]:
            samp = samp.transpose(0, 2, 1, 3, 4, 5)
            attn = attn.transpose(0, 2, 1, 3, 4)
        B, Q, H_heads, L, P, _ = samp.shape
        assert B == 1
        print(f"[INFO] 格式: [B, Q, H, L, P, 2], Q={Q}, H={H_heads}, L={L}, P={P}")
    else:
        raise ValueError(f"不支持的形状: {samp.shape}")
    
    samp = samp[0]
    attn = attn[0]
    
    H, W = image_size
    heatmaps = np.zeros((Q, H, W), dtype=np.float32)
    
    print(f"[INFO] 生成 {Q} 个query的注意力热图...")
    
    for q in range(Q):
        heatmap = np.zeros((H, W), dtype=np.float32)
        coords = samp[q].reshape(-1, 2)
        weights = attn[q].reshape(-1)
        
        if weights.max() > 0:
            weights = weights / weights.max()
        
        xs = np.clip(coords[:, 0] * W, 0, W-1).astype(int)
        ys = np.clip(coords[:, 1] * H, 0, H-1).astype(int)
        
        for x, y, w in zip(xs, ys, weights):
            heatmap[y, x] += w
        
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmaps[q] = heatmap
        
        if (q + 1) % 50 == 0:
            print(f"  进度: {q+1}/{Q}")
    
    print(f"[INFO] 完成！")
    return heatmaps


def enable_msda_debug_for_decoder(det_model, dec_layer_idx=-1, mi_head_idx=0):
    """为decoder的MSDA模块开启debug"""
    from torch_inf import get_transformer_decoder
    
    decoder = get_transformer_decoder(det_model)
    layers = decoder.layers
    
    if dec_layer_idx < 0:
        dec_layer_idx = len(layers) + dec_layer_idx
    
    layer = layers[dec_layer_idx]
    mi_heads = layer.mi_heads
    assert 0 <= mi_head_idx < len(mi_heads)
    
    msda = mi_heads[mi_head_idx].cross_attn
    msda.enable_debug(True)
    
    print(f"[MSDA] Debug enabled at decoder layer {dec_layer_idx}, mi_head {mi_head_idx}")
    return msda


def load_model(config_path, checkpoint_path, device):
    """加载模型"""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(device)
    model.eval()
    
    return model


def save_attention_results(
    config_path,
    checkpoint_path,
    image_path,
    output_path,
    device='cuda'
):
    """
    运行模型并保存注意力热图和检测结果
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型权重路径
        image_path: 测试图像路径
        output_path: 输出pkl文件路径
        device: 设备
    """
    print("="*80)
    print(f"保存注意力结果到: {output_path}")
    print("="*80)
    
    # 准备图像
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model = load_model(config_path, checkpoint_path, device)
    det_model = model.model
    
    # 开启MSDA debug
    msda = enable_msda_debug_for_decoder(det_model, dec_layer_idx=-1, mi_head_idx=0)
    
    # 前向推理
    print("运行推理...")
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    labels, boxes, scores = output
    
    # 打印检测结果信息
    print(f"检测到 {(scores[0] > 0.3).sum().item()} 个目标 (score > 0.3)")
    print(f"[DEBUG] labels shape: {labels[0].shape}")
    print(f"[DEBUG] boxes shape: {boxes[0].shape}")
    print(f"[DEBUG] scores shape: {scores[0].shape}")
    
    # 生成热图
    print("生成注意力热图...")
    heatmaps = generate_attention_heatmap(msda, image_size=(640, 640), sigma=15)
    
    # 保存结果
    results = {
        'heatmaps': heatmaps,  # [Q, H, W]
        'labels': labels[0].cpu().numpy(),
        'boxes': boxes[0].cpu().numpy(),
        'scores': scores[0].cpu().numpy(),
        'image_path': image_path,
        'checkpoint_path': checkpoint_path,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ 结果已保存到: {output_path}")
    print(f"  - 热图形状: {heatmaps.shape}")
    print(f"  - 标签形状: {results['labels'].shape}")
    print(f"  - 分数形状: {results['scores'].shape}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='保存单个模型的注意力热图')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出pkl文件路径')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='设备 (default: cuda:0)')
    
    args = parser.parse_args()
    
    save_attention_results(
        config_path=args.config,
        checkpoint_path=args.model,
        image_path=args.image,
        output_path=args.output,
        device=args.device
    )
    