"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw
from compare_qgp import *
import sys
import os
import cv2  # Added for video processing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

import cv2
import numpy as np
import torch
import torch.nn.functional as F
def visualize_unigate_analysis(
    raw_logits,    # [Q, NumClasses] 原始输出
    raw_boxes,     # [Q, 4] 原始输出 (cx, cy, w, h)
    orig_img_pil,  # PIL Image 用于画图
    file_name,     # 当前文件名
    save_dir="vis_unigate",
    score_thresh=0.1,  # 基础过滤阈值
    analysis_thresh=0.2 # 【新增】用于判定"存在类别"的阈值，建议比 score_thresh 稍高
):
    """
    全套 Unigate 分析图生成器
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. 数据准备
    # Logits -> Probabilities
    # 加上 detach() 以防万一，虽然一般 inference 时已经无梯度
    probs = raw_logits.sigmoid().detach().cpu().numpy() # [Q, C]
    Q, C = probs.shape
    
    # 筛选：只保留至少有一个类别分数 > thresh 的 Query
    max_scores_per_query = probs.max(axis=1) # [Q]
    valid_query_mask = max_scores_per_query > score_thresh
    
    if valid_query_mask.sum() == 0:
        print(f"[UniGate Vis] {file_name}: 没有分数 > {score_thresh} 的 Query，跳过。")
        return

    # 提取有效子矩阵
    filtered_probs = probs[valid_query_mask]
    
    # 为了好看，按“最可能的类别ID”对 Query 进行排序
    pred_classes = filtered_probs.argmax(axis=1)
    sort_idx = np.argsort(pred_classes)
    sorted_probs = filtered_probs[sort_idx]
    
    # === 图1: Logit Heatmap (全局稀疏性矩阵) ===
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_probs, cmap="magma", vmin=0, vmax=1, xticklabels=5, yticklabels=False)
    plt.xlabel("Class ID (Tooth Position)")
    plt.ylabel("Active Queries (Sorted by Class)")
    plt.title(f"Query-Class Score Matrix\n(Brighter & Cleaner = Better Uniqueness)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_matrix.png"), dpi=300)
    plt.close()

    # === 图2: Top-K Competition (竞争抑制分析) ===
    # 统计每一类的前两名分差
    # 【修复】使用参数 analysis_thresh 而不是硬编码 0.4
    high_conf_mask = filtered_probs.max(axis=1) > analysis_thresh
    if high_conf_mask.sum() == 0:
         print(f"[UniGate Vis] {file_name}: 没有分数 > {analysis_thresh} 的强类别，跳过 Top-K 图。")
         return

    present_classes = np.unique(pred_classes[high_conf_mask])
    
    if len(present_classes) > 0:
        rows = 2
        # 最多画 5 列，如果类别很少，列数就少一点
        cols = min(5, len(present_classes)) 
        
        # 【关键修复】加上 squeeze=False，无论行列是多少，axes 永远是 2D 数组
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 5), constrained_layout=True, squeeze=False)
        # 统一展平为 1D 数组，避免维度混乱
        axes = axes.flatten() 
        
        # 我们只画前 rows*cols 个存在的类别
        num_plots = min(len(present_classes), rows * cols)
        
        for i in range(num_plots):
            cls_id = present_classes[i]
            
            # 拿到所有 Query 对该 cls_id 的分数 (包括被过滤掉的低分 query，统统拉进来比较)
            cls_scores = probs[:, cls_id]
            
            # 降序排列取前5
            top_k_scores = np.sort(cls_scores)[::-1][:5] 
            
            ax = axes[i]
            x_ranks = ["1st", "2nd", "3rd", "4th", "5th"]
            # 颜色逻辑：第一名绿色，如果第二名分数也很高(>0.5)，给它标橙色警告，否则红色
            colors = ['green'] + ['red'] * 4
            
            bars = ax.bar(x_ranks, top_k_scores, color=colors, alpha=0.7)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"Class {cls_id}")
            if i % cols == 0: ax.set_ylabel("Score") # 只在每行第一个图画 Y 轴标签
            
            # 在柱子上标数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 隐藏多余的子图 (如果有的话)
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Top-5 Query Scores per Tooth (Steep Drop = Good Uniqueness)")
        plt.savefig(os.path.join(save_dir, f"{file_name}_competition.png"), dpi=300)
        plt.close()
        
    print(f"[UniGate Vis] 分析图已保存至 {save_dir}")

class ObjectDetectionGradCAM:
    def __init__(self, model, target_layer, visualize_input=False):
        """
        visualize_input: 
            False (默认) -> 可视化该层的【输出】特征
            True         -> 可视化该层的【输入】特征
        """
        self.model = model
        self.target_layer = target_layer
        self.visualize_input = visualize_input
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # 兼容 tuple/list 输入输出
            if self.visualize_input:
                data = input[0] if isinstance(input, (tuple, list)) else input
            else:
                data = output
            
            # 如果输出是多尺度的 list，取最后一个
            if isinstance(data, (list, tuple)):
                self.activations = data[-1]
            else:
                self.activations = data
            
        def backward_hook(module, grad_input, grad_output):
            if self.visualize_input:
                g = grad_input[0] if isinstance(grad_input, (tuple, list)) else grad_input
            else:
                g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
                
            if isinstance(g, (list, tuple)):
                self.gradients = g[-1]
            else:
                self.gradients = g

        # 注册钩子
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def generate_cam(self, input_tensor):
        # 1. 开启梯度并前向传播
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
            
        outputs = self.model(input_tensor)
        
        # 2. 获取预测分数
        if isinstance(outputs, dict) and 'pred_logits' in outputs:
            pred_logits = outputs['pred_logits']
            probs = pred_logits.sigmoid()[0]
            max_scores, class_ids = probs.max(dim=-1)
            best_idx = max_scores.argmax()
            best_score = max_scores[best_idx]
            
            state = "Input" if self.visualize_input else "Output"
            print(f"[Grad-CAM] [{state}] 解释目标: Index={best_idx}, Score={best_score.item():.4f}")
            
            # 反向传播
            best_score.backward(retain_graph=True)
        else:
            print("[Grad-CAM Error] 模型输出格式不匹配 (需要 pred_logits)")
            return None

        if self.gradients is None or self.activations is None:
            print("[Grad-CAM Error] 未捕获到梯度或特征图 (Hook 未触发)。")
            return None
            
        # 3. 计算 CAM
        grads = self.gradients
        acts = self.activations
        
        # --- 情况 A: 处理 4D 张量 [B, C, H, W] (标准 CNN) ---
        if grads.ndim == 4:
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            return cam.detach().cpu().numpy()[0, 0] # 返回 [H, W]

        # --- 情况 B: 处理 3D 张量 [B, N, C] 或 [B, C, N] (Transformer/Flattened) ---
        elif grads.ndim == 3:
            print(f"[Grad-CAM] 检测到 3D 特征: {grads.shape}，尝试重塑为 2D...")
            # 假设通常是 [B, N, C]，但也可能是 [B, C, N]
            # 启发式判断：哪个维度像 Channels？通常 Channel 较小 (256)，N 较大 (400, 1600)
            dim1, dim2 = grads.shape[1], grads.shape[2]
            
            # 如果 dim2 是 Channel (例如 [B, 400, 256]) -> 转为 [B, 256, 400] 便于计算
            if dim1 > dim2: 
                grads = grads.permute(0, 2, 1)
                acts = acts.permute(0, 2, 1)
            
            # 此时格式统一为 [B, C, N]
            # 对 N (序列维度) 求平均得到权重
            weights = torch.mean(grads, dim=2, keepdim=True) # [B, C, 1]
            cam = torch.sum(weights * acts, dim=1, keepdim=True) # [B, 1, N]
            
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7) # [B, 1, N]
            
            # 尝试还原 N 到 (H, W)
            # 假设输入是 640x640，常见 stride 是 8, 16, 32
            N = cam.shape[-1]
            H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]
            
            found_shape = False
            cam_2d = None
            
            # 尝试匹配常见 Stride
            for stride in [32, 16, 8, 64, 128]:
                h_f = int(np.ceil(H_in / stride))
                w_f = int(np.ceil(W_in / stride))
                if h_f * w_f == N:
                    cam_2d = cam.view(1, 1, h_f, w_f)
                    found_shape = True
                    break
            
            # 如果步长匹配失败，尝试正方形还原
            if not found_shape:
                side = int(np.sqrt(N))
                if side * side == N:
                    cam_2d = cam.view(1, 1, side, side)
                    found_shape = True

            if found_shape:
                return cam_2d.detach().cpu().numpy()[0, 0]
            else:
                print(f"[Grad-CAM Error] 无法将序列长度 {N} 还原为 2D 图像，跳过可视化。")
                return None
        
        else:
            print(f"[Grad-CAM Error] 不支持的特征维度: {grads.shape}")
            return None



def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i].detach().cpu()
        lab = labels[i][scr > thrh].detach().cpu()
        box = boxes[i][scr > thrh].detach().cpu()
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save('torch_results.jpg')
def visualize_hfsa_heatmaps(pil_img, model, save_dir="hfsa_vis"):
    """
    可视化 HFSA 模块前后的特征热图
    """
    import cv2
    import numpy as np
    import os
    
    # 1. 找到 HybridEncoder 模块并提取 buffer
    hybrid_encoder = None
    for m in model.modules():
        if m.__class__.__name__ == 'HybridEncoder':
            hybrid_encoder = m
            break
    
    if hybrid_encoder is None or not hasattr(hybrid_encoder, 'vis_hfsa_buffer'):
        print("[HFSA Vis] 未找到 HybridEncoder 或 buffer 为空。")
        return

    buffer_list = hybrid_encoder.vis_hfsa_buffer
    if len(buffer_list) == 0:
        print("[HFSA Vis] Buffer 为空，可能未开启 record 或该模型未使用 HFSA。")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 准备原图
    img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H, W = img_cv2.shape[:2]

    print(f"[HFSA Vis] 提取到 {len(buffer_list)} 层 HFSA 特征，开始绘制...")

    for item in buffer_list:
        idx = item['layer_idx']
        feat_before = item['before'] # [B, C, h, w]
        feat_after = item['after']   # [B, C, h, w]

        # 取 Batch 0
        fb = feat_before[0]
        fa = feat_after[0]

        # 计算热图：在通道维度求平均 (也可以试 max)
        # 结果 shape: [h, w]
        heatmap_b = torch.mean(fb, dim=0).numpy()
        heatmap_a = torch.mean(fa, dim=0).numpy()

        # 归一化函数
        def normalize_and_colorize(hm, target_h, target_w):
            # 归一化到 0-255
            min_v, max_v = hm.min(), hm.max()
            hm_norm = (hm - min_v) / (max_v - min_v + 1e-8) * 255
            hm_uint8 = hm_norm.astype(np.uint8)
            
            # Resize 到原图大小
            hm_resized = cv2.resize(hm_uint8, (target_w, target_h))
            
            # 应用伪彩色 (JET 也是常用的热图颜色)
            hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            return hm_color

        vis_b = normalize_and_colorize(heatmap_b, H, W)
        vis_a = normalize_and_colorize(heatmap_a, H, W)

        # 叠加到原图 (0.5 原图 + 0.5 热图)
        overlay_b = cv2.addWeighted(img_cv2, 0.5, vis_b, 0.5, 0)
        overlay_a = cv2.addWeighted(img_cv2, 0.5, vis_a, 0.5, 0)

        # 保存
        cv2.imwrite(os.path.join(save_dir, f"layer_{idx}_before_hfsa.jpg"), overlay_b)
        cv2.imwrite(os.path.join(save_dir, f"layer_{idx}_after_hfsa.jpg"), overlay_a)
        
        # 另外保存一个对比图
        combined = np.hstack([overlay_b, overlay_a])
        cv2.imwrite(os.path.join(save_dir, f"layer_{idx}_compare.jpg"), combined)

    print(f"[HFSA Vis] 热图已保存至 {save_dir}/ 目录。")
    
    # 清空 buffer 以防下一次推理累积
    hybrid_encoder.vis_hfsa_buffer = []

def collect_features_from_model(model):
    """
    在整个 model 里找所有有 tsne_buffer / tsne_label_buffer 的模块，
    把特征 [B,Q,C] 拼成 [N,C]，标签 [B,Q] 拼成 [N]。
    这里的标签为 int，-1 表示背景/无效。
    """
    feat_bufs = []
    label_bufs = []

    for m in model.modules():
        if hasattr(m, "tsne_buffer"):
            buf = getattr(m, "tsne_buffer")
            if isinstance(buf, list) and len(buf) > 0:
                feat_bufs.extend(buf)

        if hasattr(m, "tsne_label_buffer"):
            buf2 = getattr(m, "tsne_label_buffer")
            if isinstance(buf2, list) and len(buf2) > 0:
                label_bufs.extend(buf2)

    if len(feat_bufs) == 0:
        raise RuntimeError("[t-SNE] 没有任何模块的 tsne_buffer 有内容，说明 forward 没有触发钩子。")

    # 特征：[B_all, Q, C] -> [N, C]
    feats = torch.cat(feat_bufs, dim=0)     # [B_all, Q, C]
    n_img, n_q, c = feats.shape
    feats = feats.view(n_img * n_q, c)      # [N, C]

    # 标签：[B_all, Q] -> [N]，int，-1 表示背景
    labels = None
    if len(label_bufs) > 0:
        labels = torch.cat(label_bufs, dim=0)   # [B_all, Q]
        labels = labels.view(n_img * n_q)       # [N]
        labels = labels.cpu().numpy()

    return feats.cpu().numpy(), labels


def enable_tsne_collect(model):
    """
    在整个模型里打开所有 tsne_collect，并清空 tsne_buffer / tsne_label_buffer。
    """
    cnt = 0
    for m in model.modules():
        if hasattr(m, "tsne_collect"):
            m.tsne_collect = True
            m.tsne_buffer = []
            if hasattr(m, "tsne_label_buffer"):
                m.tsne_label_buffer = []
            cnt += 1
    print(f"[t-SNE] 已在 {cnt} 个模块上打开 tsne_collect 并清空 tsne_buffer。")


# def run_tsne_and_plot(features,
#                       labels=None,
#                       save_path='tsne_queries.png',
#                       max_points=5000,
#                       perplexity=30):
#     """
#     features: numpy [N, C]
#     labels:  numpy [N]，int 类型：
#              -1 表示背景 / 无效；
#              0 ~ (num_classes-1) 表示各个类别。
#              若为 None，则所有点用同一种颜色。
#     """
#     N = features.shape[0]
#     if labels is not None:
#         assert labels.shape[0] == N

#     # --- 子采样：features 和 labels 同时采样 ---
#     if N > max_points:
#         idx = np.random.choice(N, max_points, replace=False)
#         features = features[idx]
#         if labels is not None:
#             labels = labels[idx]
#         N = max_points

#     perplexity = min(perplexity, max(5, N - 1))

#     tsne = TSNE(
#         n_components=2,
#         init='pca',
#         random_state=0,
#         perplexity=perplexity
#     )
#     X_2d = tsne.fit_transform(features)   # [N, 2]

#     plt.figure(figsize=(6, 6))

#     if labels is None:
#         # 没有标签就全画成一种颜色
#         plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.6)
#     else:
#         labels = np.asarray(labels)

#         # 背景（label < 0）先画成灰色
#         bg_mask = labels < 0
#         if bg_mask.any():
#             plt.scatter(
#                 X_2d[bg_mask, 0], X_2d[bg_mask, 1],
#                 s=4, alpha=0.2, c='lightgray', label='bg/low-conf'
#             )

#         # 真实类别（>=0）的每个类别用一个颜色
#         fg_mask = labels >= 0
#         if fg_mask.any():
#             unique_labels = np.unique(labels[fg_mask])
#             cmap = plt.get_cmap('tab20')   # 最多 20 种颜色，够大部分情况用

#             for i, cls_id in enumerate(unique_labels):
#                 cls_mask = labels == cls_id
#                 color = cmap(i % cmap.N)
#                 plt.scatter(
#                     X_2d[cls_mask, 0], X_2d[cls_mask, 1],
#                     s=6, alpha=0.8, c=[color],
#                     label=f'class {int(cls_id)}'
#                 )

#         plt.legend(markerscale=2, fontsize=8)

#     plt.title('Decoder queries (t-SNE)')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"t-SNE 可视化已保存到: {save_path}")

def run_tsne_and_plot(features,
                      labels=None,
                      save_path='tsne_queries.png',
                      max_points=5000,
                      perplexity=30):
    """
    t-SNE 可视化，图例放到最右侧并缩小尺寸。
    """
    N = features.shape[0]
    if labels is not None:
        assert labels.shape[0] == N

    # --- 子采样：features 和 labels 同时采样 ---
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        features = features[idx]
        if labels is not None:
            labels = labels[idx]
        N = max_points

    perplexity = min(perplexity, max(5, N - 1))

    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        init='pca',
        random_state=0,
        perplexity=perplexity
    )
    X_2d = tsne.fit_transform(features)

    # 图宽加大，让右侧图例不挤压
    plt.figure(figsize=(9, 6))

    if labels is None:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.6)
    else:
        labels = np.asarray(labels)

        # 背景
        bg_mask = labels < 0
        if bg_mask.any():
            plt.scatter(
                X_2d[bg_mask, 0], X_2d[bg_mask, 1],
                s=4, alpha=0.15, c='lightgray', label='bg/low-conf'
            )

        # 前景
        fg_mask = labels >= 0
        if fg_mask.any():
            unique_labels = np.unique(labels[fg_mask])
            cmap = plt.get_cmap('tab20')

            for i, cls_id in enumerate(unique_labels):
                cls_mask = labels == cls_id
                color = cmap(i % cmap.N)
                plt.scatter(
                    X_2d[cls_mask, 0], X_2d[cls_mask, 1],
                    s=6, alpha=0.8, c=[color],
                    label=f'class {int(cls_id)}'
                )

        # ✔ 图例放到右侧，缩小 marker 和字体
        legend = plt.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            markerscale=0.6,      # 缩小点
            borderpad=0.3,
            handletextpad=0.4
        )

    plt.title('Decoder queries (t-SNE)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE 可视化已保存到: {save_path}")


# 修改 process_image 让它返回 output
def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size) # 这里的 output 是 tuple (labels, boxes, scores)
    
    # 画框
    labels, boxes, scores = output
    draw([im_pil], labels, boxes, scores)
    
    # ★★★ 新增：我们需要原始的 ref_points (中心点) 来画连线 ★★★
    # model.forward 返回的是 postprocessor 后的结果。
    # 我们需要 access 内部 decoder 的输出。
    # 简单的 trick：直接用预测框的中心点 (boxes 已经是 [cx, cy, w, h] 格式吗？)
    # D-FINE postprocessor 输出的 boxes 通常是 [x0, y0, x1, y1] 绝对坐标
    # 我们可以直接算中心点
    
    pred_boxes = boxes[0] # [Q, 4]
    cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2 / w # 归一化回 0-1
    cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2 / h
    ref_points_est = torch.stack([cx, cy], dim=-1).unsqueeze(0) # [1, Q, 2]
    
    return ref_points_est, scores[0] 


def visualize_sorted_matrix(
    attn_weights,    # [Q, Q] 原始注意力权重
    ref_points,      # [Q, 2] 坐标
    scores,          # [Q] 分数
    save_path="attn_matrix.png",
    score_thresh=0.4 # 只看分数大于 0.4 的牙齿
):
    """
    科学验证法：将牙齿按 X 轴排序，绘制注意力矩阵。
    QGP 的矩阵应该在对角线附近高亮，Baseline 则会比较散。
    """
    # 1. 筛选：只取分数高的有效牙齿 (剔除背景 Query)
    valid_mask = scores > score_thresh
    valid_mask = valid_mask.cpu().numpy().astype(bool)
    
    if valid_mask.sum() < 3:
        print(f"[Vis Matrix] 有效目标太少 ({valid_mask.sum()})，跳过绘制。")
        return

    # 提取有效子集
    # attn_weights: numpy [Q, Q]
    # ref_points: numpy [Q, 2]
    valid_pts = ref_points[valid_mask]
    
    # 提取子矩阵：先切行，再切列
    sub_attn = attn_weights[valid_mask][:, valid_mask]
    
    # 2. 排序：按 X 坐标 (左右顺序) 对索引进行排序
    # valid_pts[:, 0] 是 x 坐标
    sort_idx = np.argsort(valid_pts[:, 0])
    
    # 对矩阵的行和列同时应用这个排序
    # 结果：第 i 行就是最左边的第 i 颗牙齿
    sorted_attn = sub_attn[sort_idx][:, sort_idx]
    
    # 3. 绘图
    plt.figure(figsize=(8, 7))
    # 使用 'viridis' 或 'hot' 配色，高亮更明显
    plt.imshow(sorted_attn, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    plt.title("Sorted Attention Matrix (Left -> Right)\nDiagnosis: Diagonal=Good, Scattered=Bad")
    plt.xlabel("Key Tooth Index (Sorted)")
    plt.ylabel("Query Tooth Index (Sorted)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Vis Matrix] 矩阵热图已保存到 {save_path}")

def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

def enable_qgp_debug_for_decoder(det_model, only_last: bool = True):
    """
    从 det_model 中取出真正的 TransformerDecoder，
    在所有带 query_graph_prior 的 layer 上打开 debug。
    """
    decoder = get_transformer_decoder(det_model)   # ★ 统一入口
    print("[QGP] decoder type:", type(decoder))

    qgps = []
    for li, layer in enumerate(decoder.layers):
        qgp = getattr(layer, "query_graph_prior", None)
        if qgp is not None:
            qgp.enable_debug(True)
            qgps.append((li, qgp))

    if only_last and len(qgps) > 1:
        for li, qgp in qgps[:-1]:
            qgp.enable_debug(False)
        qgps = [qgps[-1]]

    print(f"[QGP] enabled debug for layers: {[li for li, _ in qgps]}")
    return qgps


def enable_msda_debug(det_model,
                      dec_layer_idx: int = -1,
                      mi_head_idx: int = 0):
    """
    打开 decoder[dec_layer_idx].mi_heads[mi_head_idx].cross_attn 的 debug。
    默认 dec_layer_idx=-1 表示最后一层。
    """
    decoder = get_transformer_decoder(det_model)   # ★ 用工具函数
    print("[MSDA] use decoder:", type(decoder))

    layers = decoder.layers
    if dec_layer_idx < 0:
        dec_layer_idx = len(layers) + dec_layer_idx

    layer = layers[dec_layer_idx]
    mi_heads = layer.mi_heads
    assert 0 <= mi_head_idx < len(mi_heads), "mi_head_idx 越界"

    msda = mi_heads[mi_head_idx].cross_attn
    msda.enable_debug(True)
    print(f"[MSDA] debug enabled at decoder layer {dec_layer_idx}, mi_head {mi_head_idx}")
    return msda, dec_layer_idx, mi_head_idx

def visualize_sa_graph_on_image(
    pil_img,
    sa_weights,      # [B, H, Q, Q] 或 [1, H, Q, Q]
    ref_points,      # [B, Q, 2]
    save_path="sa_graph.png",
    topk=3,          # 每个点只画最强的 K 条连线
):
    """
    可视化普通 Self-Attention 的连接图 (Baseline)
    """
    if sa_weights is None or ref_points is None:
        print("[Vis] SA weights or Ref points missing.")
        return

    # 取 Batch 0, 平均所有 Heads -> [Q, Q]
    # sa_weights 通常已经在 decoder 里 detach 过了，所以这里直接 numpy 没问题
    attn = sa_weights[0].mean(dim=0).float().numpy() 
    
    # ★★★ 修复点：增加 .detach() ★★★
    # ref_points 是刚刚算出来的，带着梯度，必须先 detach
    ref = ref_points[0].detach().cpu().numpy()
    
    Q = ref.shape[0]
    W, H = pil_img.size
    xs = ref[:, 0] * W
    ys = ref[:, 1] * H

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_img)
    plt.axis("off")

    # 画点
    plt.scatter(xs, ys, s=10, c='yellow', edgecolors='black', linewidths=0.5, alpha=0.8)

    # 画连线
    # 对每个 query，找到权重最大的 topk 个邻居 (排除自己)
    np.fill_diagonal(attn, 0) # 排除自己
    
    for q in range(Q):
        scores = attn[q] # [Q]
        neigh_idx = np.argsort(-scores)[:topk]
        
        # 归一化粗细
        s_max = scores[neigh_idx].max()
        
        for j in neigh_idx:
            # 过滤掉权重太小的线，避免画面太乱
            if scores[j] < 0.01: continue 

            x0, y0 = xs[q], ys[q]
            x1, y1 = xs[j], ys[j]
            
            # 线条透明度跟权重挂钩
            alpha = float(scores[j] / (s_max + 1e-6))
            alpha = 0.2 + 0.6 * min(alpha, 1.0)
            
            plt.plot([x0, x1], [y0, y1], color='orange', alpha=alpha, linewidth=0.8)

    plt.title("Standard Self-Attention (No QGP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Vis] Baseline SA 图已保存到 {save_path}")

def enable_sa_debug_for_decoder(det_model, layer_idx=-1):
    """开启指定层 Self-Attention 的 debug"""
    decoder = get_transformer_decoder(det_model)
    if layer_idx < 0: layer_idx = len(decoder.layers) + layer_idx
    
    layer = decoder.layers[layer_idx]
    # 开启该层所有 MI Heads 的 debug
    for head in layer.mi_heads:
        head.enable_debug(True)
    print(f"[Debug] Enabled SA debug for Layer {layer_idx}")
    return layer


def visualize_qgp_graph_on_image(
    pil_img,
    qgp_module,
    save_path="qgp_graph.png",
    topk=10,
    query_indices=None,
    overwrite_coords=None, # <--- 新增参数：传入最终的准确坐标
):
    """
    pil_img: 原图 (PIL.Image)
    qgp_module: 某一层的 QueryGraphPrior（已经跑过 forward，带 last_* 缓存）
    topk: 每个 query 只画前 topk 个注意力最大的邻居
    query_indices: 若只想画少量 query，可传 [q0, q1, ...]；默认画所有
    """
    if qgp_module.last_ref is None or qgp_module.last_attn is None:
        raise RuntimeError("QGP debug 缓存为空...")

    # 1. 获取连线权重 (保持不变，这是 QGP 的核心)
    mask = qgp_module.last_mask[0].cpu().numpy().astype(bool)
    attn = qgp_module.last_attn[0].mean(axis=0).cpu().numpy()
    
    # 2. 获取坐标 (核心修改！)
    if overwrite_coords is not None:
        # 如果传入了准确坐标（ref_points_est），就用传入的
        ref = overwrite_coords[0].cpu().numpy() # 假设传入的是 [1, Q, 2]
    else:
        # 否则才用 QGP 内部存储的（可能是修正前的）坐标
        ref = qgp_module.last_ref[0].cpu().numpy()

    # mask=True 的位置视为 -inf，不参与 topk
    attn_masked = attn.copy()
    attn_masked[mask] = -np.inf

    Q = ref.shape[0]
    if query_indices is None:
        query_indices = np.arange(Q)

    # 图像尺寸
    W, H = pil_img.size
    xs = ref[:, 0] * W
    ys = ref[:, 1] * H

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_img)
    plt.axis("off")

    # 先画所有 query 的点
    plt.scatter(xs, ys, s=10, c='yellow',
                edgecolors='black', linewidths=0.5, alpha=0.8)

    # 再画每个 query 的 topk 邻居连线
    for q in query_indices:
        scores = attn_masked[q]     # [Q]

        # 只看非 -inf 的位置
        valid_mask = ~np.isneginf(scores)
        if not valid_mask.any():
            continue

        valid_scores = scores[valid_mask]
        s_min = valid_scores.min()
        s_max = valid_scores.max()
        denom = s_max - s_min

        # 全部无差别（max==min）时就给个固定 alpha
        use_norm = denom > 1e-6

        # 还是按所有 index 排序，后面再用 valid_mask 过滤
        neigh_idx = np.argsort(-scores)[:topk]
        for j in neigh_idx:
            if not valid_mask[j]:
                continue  # 跳过被 mask 掉的

            x0, y0 = xs[q], ys[q]
            x1, y1 = xs[j], ys[j]

            if use_norm:
                a = float((scores[j] - s_min) / (denom + 1e-6))
            else:
                a = 0.5  # 所有边一样粗细/透明度就行了

            # 保证 alpha 在 [0,1] 范围内
            a = max(0.0, min(1.0, a))
            plt.plot([x0, x1], [y0, y1],
                     color='cyan',
                     alpha=0.15 + 0.6 * a,
                     linewidth=0.8)

    plt.title("Query Graph Prior attention")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[QGP] 可视化已保存到 {save_path}")


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # 获取模型的原始输出和后处理后的输出
    raw_output, post_output = model(im_data, orig_size)
    
    # 从后处理的输出中提取 labels, boxes 和 scores
    labels, boxes, scores = post_output
    
    # 画框
    draw([im_pil], labels, boxes, scores)
    
    # 获取预测框的中心点 (cx, cy)，用于后续的可视化
    pred_boxes = boxes[0]  # [Q, 4]
    cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2 / w  # 归一化回 0-1
    cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2 / h
    ref_points_est = torch.stack([cx, cy], dim=-1).unsqueeze(0)  # [1, Q, 2]
    
    return ref_points_est, scores[0]
def visualize_query_sampling_points(
    pil_img,
    msda_module,
    query_idx: int,
    save_path="query_sampling.png",
    head_agg: str = "mean",      # 现在基本用不到了，保留参数避免其它地方报错
    level_colors=None,
):
    """
    通用版采样点可视化，不再强依赖具体维度排列。

    假设:
        last_sampling_locations: [B, Q, ..., 2]
        last_attention_weights:  [B, Q, ...]   （... 维度与上面除最后一个 2 相同）

    我们只关心 B=0, 给定 query_idx 的那一片，然后全部展平。
    """
    samp = msda_module.last_sampling_locations
    attn = msda_module.last_attention_weights

    if samp is None or attn is None:
        raise RuntimeError("MSDA debug 缓存为空，请确认已经跑过一次 forward。")

    # 转 numpy
    if isinstance(samp, torch.Tensor):
        samp = samp.detach().cpu().numpy()
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()

    if samp.ndim < 3:
        raise RuntimeError(f"last_sampling_locations 维度过低: {samp.shape}")

    B, Q = samp.shape[0], samp.shape[1]
    assert 0 <= query_idx < Q, f"query_idx 越界，Q={Q}"

    # 取 batch=0, 指定 query 的所有采样点: 形状 [..., 2]
    coords = samp[0, query_idx]      # shape: [*, 2]，* 可能是 [H,L,P] 或 [L,P] 等
    if coords.shape[-1] != 2:
        raise RuntimeError(f"采样点最后一维不是 2: {coords.shape}")

    # 展平成 [N, 2]
    coords = coords.reshape(-1, 2)   # [N, 2]

    # 同样取出对应的注意力权重（如果维度能对得上就用，否则退化成全 1）
    weights = None
    if attn is not None and attn.ndim >= 2:
        w = attn[0, query_idx]       # shape: [*]
        w = w.reshape(-1)
        if w.shape[0] == coords.shape[0]:
            weights = w

    if weights is None:
        # 没法对齐就用均匀权重
        weights = np.ones(coords.shape[0], dtype=np.float32)

    # 归一化权重做大小和透明度
    w_min, w_max = float(weights.min()), float(weights.max())
    ws_n = (weights - w_min) / (w_max - w_min + 1e-6)

    # 坐标 [0,1] -> 像素
    W, H_img = pil_img.size
    xs = coords[:, 0] * W
    ys = coords[:, 1] * H_img

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_img)
    plt.axis("off")

    sizes = 10 + 30 * ws_n
    alphas = 0.2 + 0.6 * ws_n

    for x, y, s, a in zip(xs, ys, sizes, alphas):
        plt.scatter([x], [y], s=s, c='red', alpha=a)

    plt.title(f"Sampling points of query #{query_idx}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[MSDA] query #{query_idx} 采样点可视化已保存到 {save_path}")

def process_image_with_sampling(model, det_model, msda, device, file_path, query_idx=0):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # 这里最好也保存一份 resize 后的 PIL 图，用来画采样点
    im_resized = im_pil.resize((640, 640))

    output = model(im_data, orig_size)
    labels, boxes, scores = output
    draw([im_pil], labels, boxes, scores)

    visualize_query_sampling_points(
        im_resized,
        msda_module=msda,
        query_idx=query_idx,
        save_path=f"sampling_q{query_idx}.png"
    )
def get_transformer_decoder(det_model):
    """
    从 det_model 中取到真正的 TransformerDecoder（那个有 .layers 的模块）.

    结构一般是：
        det_model (DFINE)
            .decoder -> DFINETransformer
                .decoder -> TransformerDecoder(有 .layers)

    为了稳妥，做两次 decoder 解析。
    """
    dec = getattr(det_model, "decoder", det_model)   # 先拿到 DFINETransformer 或直接是 decoder
    inner = getattr(dec, "decoder", None)
    if inner is not None:
        dec = inner                                  # 如果里面还有一层 decoder，再深入一层
    return dec

def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    # === 修改 1: 定义一个能返回 raw output 的 Wrapper ===
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            # 获取原始输出 [pred_logits, pred_boxes]
            outputs = self.model(images)
            
            # 获取后处理结果 (NMS后的) 用于画常规框
            post_outputs = self.postprocessor(outputs, orig_target_sizes)
            
            return outputs, post_outputs  # 返回两者

    device = args.device
    model = Model().to(device)

    # 打开 tsne_collect
    # ... (前面的代码不变，直到 model = Model().to(device)) ...
    # 打开 tsne_collect
    det_model = model.model

    # ==========================================
    # === [Grad-CAM 智能设置 (修复版)] ===
    # ==========================================
    grad_cam_target = None
    layer_name_in = "Input Features"
    layer_name_out = "Output Features"

    # 1. 寻找 HybridEncoder
    encoder_module = None
    for m in det_model.modules():
        if m.__class__.__name__ == 'HybridEncoder':
            encoder_module = m
            break

    if encoder_module is not None:
        # 优先级 1: HFSA (针对原始模型)
        if hasattr(encoder_module, 'hfsas_enc') and len(encoder_module.hfsas_enc) > 0:
            grad_cam_target = encoder_module.hfsas_enc[-1]
            print(f"[Grad-CAM] 已锁定目标: HybridEncoder.hfsas_enc[-1] (HFSA)")
            layer_name_in = "Before HFSA"
            layer_name_out = "After HFSA"
            
        # 优先级 2: Encoder 输出投影层 (针对 No-HFSA 模型)
        elif hasattr(encoder_module, 'enc_projs') and len(encoder_module.enc_projs) > 0:
            grad_cam_target = encoder_module.enc_projs[-1]
            print(f"[Grad-CAM] 已锁定目标: HybridEncoder.enc_projs[-1] (Projection)")
            layer_name_in = "Encoder Input"
            layer_name_out = "Encoder Output"
            
        # 优先级 3: Encoder 的最后一个 CSP 块 (通用兜底)
        elif hasattr(encoder_module, 'encoder') and len(encoder_module.encoder) > 0:
            grad_cam_target = encoder_module.encoder[-1]
            print(f"[Grad-CAM] 已锁定目标: HybridEncoder.encoder[-1] (Last CSP Block)")
            layer_name_in = "Block Input"
            layer_name_out = "Encoder Output"
            
        # 优先级 4: 直接 Hook 整个 Encoder 模块
        else:
            grad_cam_target = encoder_module
            print(f"[Grad-CAM] 已锁定目标: Whole HybridEncoder Module")
            layer_name_in = "Backbone Features"
            layer_name_out = "Encoder Output"
    else:
        print("[Grad-CAM] ⚠️ 未找到 HybridEncoder，跳过 Grad-CAM。")

    # ==========================================
    # 2. 运行推理 & Grad-CAM 可视化
    # ==========================================
    if os.path.isfile(args.input):
        import cv2
        orig_image = Image.open(args.input).convert('RGB')
        w, h = orig_image.size
        
        # 预处理
        transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        img_tensor = transform(orig_image).unsqueeze(0).to(device)

        # --- Grad-CAM 生成逻辑 ---
        if grad_cam_target is not None:
            
            # (A) 生成【输出】热图 (最重要的图: Encoder 输出特征)
            print(f"\n--- 正在生成 {layer_name_out} 热图 ---")
            grad_cam_out = ObjectDetectionGradCAM(det_model, grad_cam_target, visualize_input=False)
            
            with torch.enable_grad():
                img_tensor.requires_grad_(True)
                cam_map = grad_cam_out.generate_cam(img_tensor)
            
            if cam_map is not None:
                cam_resized = cv2.resize(cam_map, (w, h))
                cam_uint8 = (cam_resized * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
                img_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
                result = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
                
                save_name = "grad_cam_encoder_output.jpg"
                cv2.imwrite(save_name, result)
                print(f"✓ [Grad-CAM] {layer_name_out} 结果已保存: {save_name}")
            else:
                print(f"[Grad-CAM] {layer_name_out} 生成失败 (无梯度)。")
            grad_cam_out.remove_hooks() # 清理

            # (B) 生成【输入】热图 (可选: Backbone 特征)
            # 如果是 Hook 整个 Encoder，这里的输入就是 Backbone 的输出
            print(f"\n--- 正在生成 {layer_name_in} 热图 ---")
            grad_cam_in = ObjectDetectionGradCAM(det_model, grad_cam_target, visualize_input=True)
            
            with torch.enable_grad():
                img_tensor.grad = None
                img_tensor.requires_grad_(True)
                cam_map = grad_cam_in.generate_cam(img_tensor)
                
            if cam_map is not None:
                cam_resized = cv2.resize(cam_map, (w, h))
                cam_uint8 = (cam_resized * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
                img_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
                result = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
                
                save_name = "grad_cam_encoder_input.jpg"
                cv2.imwrite(save_name, result)
                print(f"✓ [Grad-CAM] {layer_name_in} 结果已保存: {save_name}")
            else:
                print(f"[Grad-CAM] {layer_name_in} 生成失败 (无梯度)。")
            grad_cam_in.remove_hooks() # 清理
            
    # ==========================================
    # 3. HFSA 热图 (对于 No-HFSA 模型，这步会自动跳过)
    # ==========================================
    if os.path.isfile(args.input):
        # 开启记录 (虽然 No-HFSA 模型没用，但为了防止报错还是开一下)
        found_enc = False
        for m in det_model.modules():
            if m.__class__.__name__ == 'HybridEncoder':
                m.vis_hfsa_record = True
                m.vis_hfsa_buffer = []
                found_enc = True
                break
        
        # 正常跑一次流程 (顺便画检测框)
        process_image(model, device, args.input)
        
        # 尝试画 HFSA 热图 (如果是 No-HFSA 模型，函数内部会检测到 Buffer 为空并自动跳过)
        visualize_hfsa_heatmaps(Image.open(args.input), det_model, save_dir="hfsa_results")

    # ... (后面的 MSDA, t-SNE, QGP 代码保持不变) ...

    # msda, dec_idx, mi_idx = enable_msda_debug(det_model, dec_layer_idx=-1, mi_head_idx=0)
    # process_image_with_sampling(model, det_model, msda, device, args.input, query_idx=0)

    # enable_tsne_collect(det_model)
    # # 1. 尝试开启 QGP Debug (针对 Ours 模型)
    # qgp_list = enable_qgp_debug_for_decoder(det_model, only_last=True)
    
    # # 2. 如果没找到 QGP (说明是 Baseline 模型)，则开启 Self-Attention Debug
    # target_layer_sa = None
    # if len(qgp_list) == 0:
    #     print("未检测到 QGP 模块，切换为 Baseline Self-Attention 可视化模式...")
    #     # 开启最后一层的 SA Debug
    #     target_layer_sa = enable_sa_debug_for_decoder(det_model, layer_idx=-1)
    # file_path = args.input
    # def process_and_analyze(file_path):
    #     filename = os.path.splitext(os.path.basename(file_path))[0]
        
    #     im_pil = Image.open(file_path).convert('RGB')
    #     w, h = im_pil.size
    #     orig_size = torch.tensor([[w, h]]).to(device)
        
    #     transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    #     im_data = transforms(im_pil).unsqueeze(0).to(device)

    #     # 推理
    #     raw_output, post_output = model(im_data, orig_size)
        
    #     # 1. 常规画图 (NMS后的结果)
    #     labels, boxes, scores = post_output
    #     draw([im_pil], labels, boxes, scores) # 保存为 torch_results.jpg
    #     # 可以重命名一下防止覆盖
    #     # im_pil.save(f"vis_unigate/{filename}_det.jpg")

    #     # 2. Unigate 可视化 (使用 raw_output)
    #     # raw_output['pred_logits']: [B, Q, C]
    #     # raw_output['pred_boxes']:  [B, Q, 4]
    #     visualize_unigate_analysis(
    #         raw_logits=raw_output['pred_logits'][0], 
    #         raw_boxes=raw_output['pred_boxes'][0],
    #         orig_img_pil=Image.open(file_path).convert('RGB'),
    #         file_name=filename,
    #         save_dir="vis_unigate_results" # 结果保存路径
    #     )
    # # 如果是目录，就遍历目录下所有图片
    # if os.path.isdir(file_path):
    #     exts = ('.jpg', '.jpeg', '.png', '.bmp')
    #     img_paths = [
    #         os.path.join(file_path, f)
    #         for f in os.listdir(file_path)
    #         if os.path.splitext(f)[-1].lower() in exts
    #     ]
    #     img_paths.sort()
    #     print(f"共发现 {len(img_paths)} 张图片，将逐张推理并收集 queries...")
    #     for p in img_paths:
    #         process_image(model, device, p)
    #     print("✓ 所有图片处理完毕。")
    # else:
    #     # 处理单张图片的情况
    #     if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            
    #         # 1. 获取坐标和分数 (注意这里接收两个返回值)
    #         ref_points_est, scores_est = process_image(model, device, file_path) 
            
    #         # 准备数据用于矩阵可视化
    #         final_attn = None
    #         method_name = "unknown"
            
    #         # A. 处理 QGP (Ours)
    #         if len(qgp_list) > 0:
    #             layer_idx, qgp = qgp_list[0]
    #             method_name = "QGP_Ours"
                
    #             # 1. 画原来的连线图 (保留着做参考)
    #             visualize_qgp_graph_on_image(
    #                 Image.open(file_path).convert('RGB'),
    #                 qgp,
    #                 save_path=f"vis_graph_{method_name}.jpg",
    #                 topk=3,
    #                 overwrite_coords=ref_points_est.detach()
    #             )
                
    #             # 提取权重用于画矩阵
    #             # [B, H, Q, Q] -> [Q, Q]
    #             final_attn = qgp.last_attn[0].mean(axis=0).cpu().numpy()

    #         # B. 处理 Baseline (Standard SA)
    #         elif target_layer_sa is not None:
    #             head = target_layer_sa.mi_heads[0]
    #             method_name = "Baseline_SA"
                
    #             if head.last_sa_weights is not None:
    #                 # 1. 画原来的连线图
    #                 visualize_sa_graph_on_image(
    #                     Image.open(file_path).convert('RGB'),
    #                     head.last_sa_weights,
    #                     ref_points_est, 
    #                     save_path=f"vis_graph_{method_name}.jpg",
    #                     topk=3 
    #                 )
                    
    #                 # 提取权重用于画矩阵
    #                 final_attn = head.last_sa_weights[0].mean(dim=0).float().cpu().numpy()

    #         # ★★★ 核心验证：画排序后的注意力矩阵 ★★★
    #         if final_attn is not None:
    #             # 排除自己对自己的注意力 (对角线置0)，这样更能看清它关注了谁
    #             # (可选操作，如果你想看它多关注邻居，就把这行取消注释)
    #             # np.fill_diagonal(final_attn, 0)
                
    #             visualize_sorted_matrix(
    #                 final_attn, 
    #                 ref_points_est[0].detach().cpu().numpy(), # [Q, 2]
    #                 scores_est.detach(),                      # [Q]
    #                 save_path=f"vis_matrix_{method_name}.png",
    #                 score_thresh=0.4
    #             )


    # # 3) 收集特征和标记
    # # 3) 收集特征和类别标签
    # feats, labels = collect_features_from_model(det_model)
    # print(f"✓ [t-SNE] Features shape: {feats.shape}")
    
    # if labels is not None:
    #     labels_np = np.asarray(labels)
    #     n_total = labels_np.shape[0]
    #     n_bg = int((labels_np < 0).sum())
    #     n_fg = n_total - n_bg
    #     unique_fg = np.unique(labels_np[labels_np >= 0])
    #     print(f"✓ [t-SNE] Labels collected: {n_fg}/{n_total} queries with valid classes")
    #     print(f"✓ [t-SNE] Num foreground classes in this run: {len(unique_fg)} -> {unique_fg.tolist()}")
    # else:
    #     print("⚠ [t-SNE] No labels collected (tsne_label_buffer is empty)")
    
    # # 4) t-SNE 可视化
    # run_tsne_and_plot(feats, labels=labels, save_path='tsne_queries.png')

    # 2. MSDA 采样点 Debug
    msda, dec_idx, mi_idx = enable_msda_debug(det_model, dec_layer_idx=-1, mi_head_idx=0)
    
    # 3. t-SNE 数据收集开启
    enable_tsne_collect(det_model)
    
    # 4. QGP 或 Self-Attention Debug 开启
    qgp_list = enable_qgp_debug_for_decoder(det_model, only_last=True)
    target_layer_sa = None
    if len(qgp_list) == 0:
        print("[Info] 未检测到 QGP，切换为 Baseline SA 可视化模式...")
        target_layer_sa = enable_sa_debug_for_decoder(det_model, layer_idx=-1)

    # === 定义单张图片的处理逻辑 (替代原来的 process_image) ===
    def run_inference_and_visualize(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {filename} ...")
        
        # 1. 加载图片
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # 2. 推理 (获取 Raw 和 Post 两份结果)
        raw_output, post_output = model(im_data, orig_size)
        
        # 3. 解包结果
        labels, boxes, scores = post_output  # [B, N], [B, N, 4], [B, N]
        
        # === 可视化 A: 常规画框 (NMS后) ===
        draw([im_pil], labels, boxes, scores)
        if not os.path.exists("vis_det"): 
            os.makedirs("vis_det")
        im_pil.save(f"vis_det/{filename}_detection.jpg")

        # === 可视化 B: UniGate 矩阵分析 (重点!) ===
        # 使用 raw_output 中的 logits，不做 NMS，才能看到重复预测
        # 确保 raw_output 是字典格式
        if isinstance(raw_output, dict) and 'pred_logits' in raw_output:
            visualize_unigate_analysis(
                raw_logits=raw_output['pred_logits'][0],  # [Q, C]
                raw_boxes=raw_output['pred_boxes'][0],    # [Q, 4]
                orig_img_pil=Image.open(file_path).convert('RGB'),
                file_name=filename,
                save_dir="vis_unigate_results",
                score_thresh=0.2
            )
        else:
            print(f"[Warning] raw_output 格式不正确，跳过 UniGate 可视化")
            print(f"raw_output type: {type(raw_output)}")
            if isinstance(raw_output, dict):
                print(f"raw_output keys: {raw_output.keys()}")

        # === 可视化 C: 采样点 (MSDA) ===
        visualize_query_sampling_points(
            im_pil.resize((640, 640)),
            msda_module=msda,
            query_idx=0, # 看第0个query
            save_path=f"vis_sampling_{filename}_q0.png"
        )

        # === 可视化 D: 连线图 & 排序矩阵 (QGP / SA) ===
        # 计算预测框中心点，用于连线图的坐标
        pred_boxes = boxes[0] # [Q, 4]
        cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2 / w 
        cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2 / h
        ref_points_est = torch.stack([cx, cy], dim=-1).unsqueeze(0) # [1, Q, 2]
        scores_est = scores[0]

        final_attn = None
        method_name = "unknown"

        # D.1 如果是 Ours (QGP)
        if len(qgp_list) > 0:
            layer_idx, qgp = qgp_list[0]
            method_name = "QGP"
            # 画连线图
            visualize_qgp_graph_on_image(
                Image.open(file_path).convert('RGB'),
                qgp,
                save_path=f"vis_graph_{filename}_{method_name}.jpg",
                topk=3,
                overwrite_coords=ref_points_est.detach()
            )
            # 提取权重
            if qgp.last_attn is not None:
                final_attn = qgp.last_attn[0].mean(axis=0).cpu().numpy()

        # D.2 如果是 Baseline (Standard SA)
        elif target_layer_sa is not None:
            head = target_layer_sa.mi_heads[0]
            method_name = "BaselineSA"
            # 画连线图
            if head.last_sa_weights is not None:
                visualize_sa_graph_on_image(
                    Image.open(file_path).convert('RGB'),
                    head.last_sa_weights,
                    ref_points_est, 
                    save_path=f"vis_graph_{filename}_{method_name}.jpg",
                    topk=3 
                )
                final_attn = head.last_sa_weights[0].mean(dim=0).float().cpu().numpy()

        # D.3 画排序注意力矩阵 (Sorted Matrix)
        if final_attn is not None:
            visualize_sorted_matrix(
                final_attn, 
                ref_points_est[0].detach().cpu().numpy(),
                scores_est.detach(),
                save_path=f"vis_matrix_{filename}_{method_name}.png",
                score_thresh=0.4
            )

    # === 执行循环 ===
    if os.path.isdir(args.input):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_paths = [
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if os.path.splitext(f)[-1].lower() in exts
        ]
        img_paths.sort()
        print(f"Found {len(img_paths)} images.")
        for p in img_paths:
            run_inference_and_visualize(p)
    else:
        run_inference_and_visualize(args.input)

    # === E: t-SNE (最后汇总画一张) ===
    feats, labels = collect_features_from_model(det_model)
    if labels is not None:
        print(f"✓ [t-SNE] Collected {feats.shape[0]} queries.")
        run_tsne_and_plot(feats, labels=labels, save_path='vis_tsne_queries.png')
    else:
        print("⚠ [t-SNE] No labels collected.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
"python tools/inference/torch_inf.py   -c /home/wuxi/DEIM_origin/configs/deim_dfine/deim_hgnetv2_s_coco.yml   -r /home/wuxi/DEIM_origin/outputs/deim_hgnetv2_s_coco/best_stg1.pth   -i /home/wuxi/DENTEX2023/enumeration32/val2017/train_375.png   -d cuda:1"