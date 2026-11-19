import torch
import torch.nn as nn
import torchvision.transforms as T
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import cv2
import matplotlib.pyplot as plt

# 确保能找到 engine 和 core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# ==========================================
# 1. 核心可视化函数 (你的代码，无需修改，通用)
# ==========================================
def visualize_unigate_analysis(
    raw_logits,    # [Q, NumClasses] 原始输出
    raw_boxes,     # [Q, 4] 原始输出 (cx, cy, w, h)
    orig_img_pil,  # PIL Image 用于画图
    file_name,     # 当前文件名
    save_dir="vis_baseline_analysis", # 修改保存路径以示区分
    score_thresh=0.1,  
    analysis_thresh=0.2 
):
    """
    全套 Unigate 分析图生成器 (适用于 Baseline 模型)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    probs = raw_logits.sigmoid().detach().cpu().numpy() # [Q, C]
    
    # 筛选：只保留至少有一个类别分数 > thresh 的 Query
    max_scores_per_query = probs.max(axis=1) # [Q]
    valid_query_mask = max_scores_per_query > score_thresh
    
    if valid_query_mask.sum() == 0:
        print(f"[Vis] {file_name}: 没有分数 > {score_thresh} 的 Query，跳过。")
        return

    # 提取有效子矩阵
    filtered_probs = probs[valid_query_mask]
    
    # 按“最可能的类别ID”对 Query 进行排序
    pred_classes = filtered_probs.argmax(axis=1)
    sort_idx = np.argsort(pred_classes)
    sorted_probs = filtered_probs[sort_idx]
    
    # === 图1: Logit Heatmap (全局稀疏性矩阵) ===
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_probs, cmap="magma", vmin=0, vmax=1, xticklabels=5, yticklabels=False)
    plt.xlabel("Class ID (Tooth Position)")
    plt.ylabel("Active Queries (Sorted by Class)")
    plt.title(f"Baseline: Query-Class Score Matrix\n(Comparison: Is this messier than UniGate?)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_matrix.png"), dpi=300)
    plt.close()

    # === 图2: Top-K Competition (竞争抑制分析) ===
    high_conf_mask = filtered_probs.max(axis=1) > analysis_thresh
    if high_conf_mask.sum() == 0:
         return

    present_classes = np.unique(pred_classes[high_conf_mask])
    
    if len(present_classes) > 0:
        rows = 2
        cols = min(5, len(present_classes)) 
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 5), constrained_layout=True, squeeze=False)
        axes = axes.flatten() 
        num_plots = min(len(present_classes), rows * cols)
        
        for i in range(num_plots):
            cls_id = present_classes[i]
            cls_scores = probs[:, cls_id]
            top_k_scores = np.sort(cls_scores)[::-1][:5] 
            
            ax = axes[i]
            x_ranks = ["1st", "2nd", "3rd", "4th", "5th"]
            colors = ['green'] + ['red'] * 4
            
            bars = ax.bar(x_ranks, top_k_scores, color=colors, alpha=0.7)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"Class {cls_id}")
            if i % cols == 0: ax.set_ylabel("Score")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Baseline: Top-5 Query Scores per Tooth")
        plt.savefig(os.path.join(save_dir, f"{file_name}_competition.png"), dpi=300)
        plt.close()
    
    print(f"[Vis] Baseline 分析图已保存至 {save_dir}")

# ==========================================
# 2. 基础画框函数
# ==========================================
def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i].detach().cpu()
        lab = labels[i][scr > thrh].detach().cpu()
        box = boxes[i][scr > thrh].detach().cpu()
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red', width=2)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

# ==========================================
# 3. 主程序
# ==========================================
def main(args):
    # 1. 加载配置
    cfg = YAMLConfig(args.config, resume=args.resume)

    # 针对 HGNetv2 的特殊处理
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # 2. 加载权重
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # 3. 加载模型
    # 注意：这里假设你的环境代码已经是原始代码。
    # 如果你的环境代码还是魔改版，load_state_dict 可能会报错。
    # 这里的 strict=False 是为了兼容一些细微差别，但在结构完全不匹配时仍会报错。
    # 既然你说要用"原始代码"，请确保 import 进来的 DFINETransformer 是没改过的。
    print("Loading model state dict...")
    cfg.model.load_state_dict(state, strict=True) # 使用 strict=True 验证是否完全匹配原始权重

    # 4. 构建推理模型 Wrapper
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            # 获取原始输出 [pred_logits, pred_boxes]
            outputs = self.model(images)
            # 获取后处理结果 (NMS后的)
            post_outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs, post_outputs

    device = args.device
    model = Model().to(device)
    model.eval()

    # 5. 定义推理逻辑
    def run_inference(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {filename} ...")

        # 读取图片
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        
        # 预处理
        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            raw_output, post_output = model(im_data, orig_size)

        # === 任务 A: 画检测框 (常规) ===
        labels, boxes, scores = post_output
        im_copy = im_pil.copy()
        draw([im_copy], labels, boxes, scores)
        if not os.path.exists("vis_baseline_det"):
            os.makedirs("vis_baseline_det")
        im_copy.save(f"vis_baseline_det/{filename}_det.jpg")

        # === 任务 B: 矩阵/稀疏性分析 (对比 UniGate 效果) ===
        # 使用 raw_output 中的 pred_logits
        if isinstance(raw_output, dict) and 'pred_logits' in raw_output:
            visualize_unigate_analysis(
                raw_logits=raw_output['pred_logits'][0],  # [Q, C]
                raw_boxes=raw_output['pred_boxes'][0],    # [Q, 4]
                orig_img_pil=im_pil,
                file_name=filename,
                save_dir="vis_baseline_analysis", # 结果保存在这里
                score_thresh=0.2
            )
        else:
            print("[Error] 无法获取 pred_logits，请检查模型输出格式。")

    # 6. 执行遍历
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
            run_inference(p)
    else:
        run_inference(args.input)

    print("Done. Visualization results saved in 'vis_baseline_analysis/'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
