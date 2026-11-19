# """
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# Modified to generate COCO evaluation JSON
# """

# import torch
# import torch.nn as nn
# import torchvision.transforms as T

# import numpy as np
# from PIL import Image
# import json
# import sys
# import os
# from tqdm import tqdm
# from pathlib import Path

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from engine.core import YAMLConfig


# class COCOResultGenerator:
#     """生成COCO格式的评估结果"""
    
#     def __init__(self, annotation_file):
#         """
#         初始化结果生成器
        
#         参数:
#             annotation_file: COCO格式的标注文件路径
#         """
#         with open(annotation_file, 'r') as f:
#             self.coco_data = json.load(f)
        
#         # 构建image_id到file_name的映射
#         self.image_id_to_filename = {
#             img['id']: img['file_name'] 
#             for img in self.coco_data['images']
#         }
        
#         # 构建file_name到image_id的映射
#         self.filename_to_image_id = {
#             img['file_name']: img['id'] 
#             for img in self.coco_data['images']
#         }
        
#         self.results = []
    
#     def add_detections(self, image_id, labels, boxes, scores, threshold=0.3):
#         """
#         添加单张图像的检测结果
        
#         参数:
#             image_id: COCO图像ID
#             labels: 类别标签张量 [N]
#             boxes: 检测框张量 [N, 4] (x1, y1, x2, y2)
#             scores: 置信度分数张量 [N]
#             threshold: 置信度阈值
#         """
#         # 过滤低置信度检测
#         mask = scores > threshold
#         labels = labels[mask]
#         boxes = boxes[mask]
#         scores = scores[mask]
        
#         # 转换为numpy以便处理
#         if isinstance(labels, torch.Tensor):
#             labels = labels.cpu().numpy()
#         if isinstance(boxes, torch.Tensor):
#             boxes = boxes.cpu().numpy()
#         if isinstance(scores, torch.Tensor):
#             scores = scores.cpu().numpy()
        
#         # 将每个检测添加到结果列表
#         for label, box, score in zip(labels, boxes, scores):
#             x1, y1, x2, y2 = box
#             width = float(x2 - x1)
#             height = float(y2 - y1)
            
#             # COCO格式要求bbox为[x, y, width, height]
#             detection = {
#                 'image_id': int(image_id),
#                 'category_id': int(label),
#                 'bbox': [float(x1), float(y1), width, height],
#                 'score': float(score)
#             }
            
#             self.results.append(detection)
    
#     def save_results(self, output_path):
#         """
#         保存结果为JSON文件
        
#         参数:
#             output_path: 输出文件路径
#         """
#         output_dir = Path(output_path).parent
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         with open(output_path, 'w') as f:
#             json.dump(self.results, f, indent=2)
        
#         print(f"COCO评估结果已保存至: {output_path}")
#         print(f"总检测数: {len(self.results)}")
    
#     def get_statistics(self):
#         """获取检测统计信息"""
#         from collections import Counter
        
#         category_counts = Counter([det['category_id'] for det in self.results])
#         image_counts = Counter([det['image_id'] for det in self.results])
        
#         stats = {
#             'total_detections': len(self.results),
#             'total_images': len(image_counts),
#             'avg_detections_per_image': len(self.results) / len(image_counts) if image_counts else 0,
#             'detections_per_category': dict(category_counts)
#         }
        
#         return stats


# def process_dataset(model, device, image_dir, annotation_file, output_path, 
#                    conf_threshold=0.3, batch_size=1):
#     """
#     处理整个数据集并生成COCO评估JSON
    
#     参数:
#         model: 部署模式的模型
#         device: 推理设备
#         image_dir: 图像目录路径
#         annotation_file: COCO标注文件路径
#         output_path: 输出JSON文件路径
#         conf_threshold: 置信度阈值
#         batch_size: 批处理大小（建议设为1以确保准确性）
#     """
#     # 初始化结果生成器
#     result_generator = COCOResultGenerator(annotation_file)
    
#     # 定义图像转换
#     transforms = T.Compose([
#         T.Resize((640, 640)),
#         T.ToTensor(),
#     ])
    
#     # 获取所有需要处理的图像
#     image_list = sorted(result_generator.image_id_to_filename.items())
    
#     print(f"开始处理 {len(image_list)} 张图像...")
    
#     # 处理每张图像
#     model.eval()
#     with torch.no_grad():
#         for image_id, filename in tqdm(image_list):
#             # 构建完整图像路径
#             image_path = os.path.join(image_dir, filename)
            
#             if not os.path.exists(image_path):
#                 print(f"警告: 图像不存在 {image_path}")
#                 continue
            
#             try:
#                 # 加载图像
#                 im_pil = Image.open(image_path).convert('RGB')
#                 w, h = im_pil.size
#                 orig_size = torch.tensor([[w, h]]).to(device)
                
#                 # 转换图像
#                 im_data = transforms(im_pil).unsqueeze(0).to(device)
                
#                 # 模型推理
#                 output = model(im_data, orig_size)
#                 labels, boxes, scores = output
                
#                 # 添加检测结果
#                 result_generator.add_detections(
#                     image_id=image_id,
#                     labels=labels[0],
#                     boxes=boxes[0],
#                     scores=scores[0],
#                     threshold=conf_threshold
#                 )
                
#             except Exception as e:
#                 print(f"处理图像 {filename} 时出错: {str(e)}")
#                 continue
    
#     # 保存结果
#     result_generator.save_results(output_path)
    
#     # 打印统计信息
#     stats = result_generator.get_statistics()
#     print("\n检测统计信息:")
#     print(f"  总检测数: {stats['total_detections']}")
#     print(f"  处理图像数: {stats['total_images']}")
#     print(f"  平均每张图检测数: {stats['avg_detections_per_image']:.2f}")
    
#     # 打印各类别检测数量
#     print("\n各类别检测数量:")
#     for cat_id, count in sorted(stats['detections_per_category'].items()):
#         print(f"  类别 {cat_id}: {count} 次")
    
#     return result_generator.results


# def evaluate_with_coco_api(annotation_file, result_file):
#     """
#     使用COCO API进行评估
    
#     参数:
#         annotation_file: 标注文件路径
#         result_file: 预测结果文件路径
#     """
#     try:
#         from pycocotools.coco import COCO
#         from pycocotools.cocoeval import COCOeval
        
#         print("\n开始COCO评估...")
        
#         # 加载标注和预测结果
#         coco_gt = COCO(annotation_file)
#         coco_dt = coco_gt.loadRes(result_file)
        
#         # 执行评估
#         coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
        
#         # 提取关键指标
#         metrics = {
#             'mAP@0.5:0.95': coco_eval.stats[0],
#             'mAP@0.5': coco_eval.stats[1],
#             'mAP@0.75': coco_eval.stats[2],
#             'mAP_small': coco_eval.stats[3],
#             'mAP_medium': coco_eval.stats[4],
#             'mAP_large': coco_eval.stats[5],
#         }
        
#         return metrics
        
#     except ImportError:
#         print("\n警告: 未安装pycocotools，跳过COCO评估")
#         print("可通过以下命令安装: pip install pycocotools")
#         return None


# def main(args):
#     """主函数"""
#     cfg = YAMLConfig(args.config, resume=args.resume)

#     if 'HGNetv2' in cfg.yaml_cfg:
#         cfg.yaml_cfg['HGNetv2']['pretrained'] = False

#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         if 'ema' in checkpoint:
#             state = checkpoint['ema']['module']
#         else:
#             state = checkpoint['model']
#     else:
#         raise AttributeError('必须提供模型权重文件 (--resume)')

#     # 加载训练模式的state并转换为部署模式
#     cfg.model.load_state_dict(state)

#     class Model(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = cfg.model.deploy()
#             self.postprocessor = cfg.postprocessor.deploy()

#         def forward(self, images, orig_target_sizes):
#             outputs = self.model(images)
#             outputs = self.postprocessor(outputs, orig_target_sizes)
#             return outputs

#     device = args.device
#     model = Model().to(device)
    
#     print(f"使用设备: {device}")
#     print(f"模型权重: {args.resume}")
#     print(f"置信度阈值: {args.conf_threshold}")
    
#     # 处理数据集并生成结果
#     results = process_dataset(
#         model=model,
#         device=device,
#         image_dir=args.image_dir,
#         annotation_file=args.annotation_file,
#         output_path=args.output,
#         conf_threshold=args.conf_threshold,
#         batch_size=args.batch_size
#     )
    
#     # 如果指定了评估选项，则执行COCO评估
#     if args.evaluate:
#         metrics = evaluate_with_coco_api(args.annotation_file, args.output)
#         if metrics:
#             print("\n评估指标:")
#             for key, value in metrics.items():
#                 print(f"  {key}: {value:.4f}")


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='DEIM模型推理并生成COCO评估JSON')
#     parser.add_argument('-c', '--config', type=str, required=True, 
#                        help='模型配置文件路径')
#     parser.add_argument('-r', '--resume', type=str, required=True, 
#                        help='模型权重文件路径')
#     parser.add_argument('-i', '--image_dir', type=str, required=True, 
#                        help='图像目录路径')
#     parser.add_argument('-a', '--annotation_file', type=str, required=True, 
#                        help='COCO格式的标注文件路径')
#     parser.add_argument('-o', '--output', type=str, default='predictions.json', 
#                        help='输出JSON文件路径')
#     parser.add_argument('-d', '--device', type=str, default='cuda', 
#                        help='推理设备 (cuda/cpu)')
#     parser.add_argument('--conf_threshold', type=float, default=0.3, 
#                        help='置信度阈值')
#     parser.add_argument('--batch_size', type=int, default=1, 
#                        help='批处理大小')
#     parser.add_argument('--evaluate', action='store_true', 
#                        help='生成JSON后立即进行COCO评估')
    
#     args = parser.parse_args()
#     main(args)





"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified to generate COCO evaluation JSON with per-category metrics
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


class COCOResultGenerator:
    """生成COCO格式的评估结果"""
    
    def __init__(self, annotation_file):
        """
        初始化结果生成器
        
        参数:
            annotation_file: COCO格式的标注文件路径
        """
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # 构建image_id到file_name的映射
        self.image_id_to_filename = {
            img['id']: img['file_name'] 
            for img in self.coco_data['images']
        }
        
        # 构建file_name到image_id的映射
        self.filename_to_image_id = {
            img['file_name']: img['id'] 
            for img in self.coco_data['images']
        }
        
        # 构建category_id到name的映射
        self.category_id_to_name = {
            cat['id']: cat['name'] 
            for cat in self.coco_data.get('categories', [])
        }
        
        self.results = []
    
    def add_detections(self, image_id, labels, boxes, scores, threshold=0.3):
        """
        添加单张图像的检测结果
        
        参数:
            image_id: COCO图像ID
            labels: 类别标签张量 [N]
            boxes: 检测框张量 [N, 4] (x1, y1, x2, y2)
            scores: 置信度分数张量 [N]
            threshold: 置信度阈值
        """
        # 过滤低置信度检测
        mask = scores > threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        
        # 转换为numpy以便处理
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # 将每个检测添加到结果列表
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            width = float(x2 - x1)
            height = float(y2 - y1)
            
            # COCO格式要求bbox为[x, y, width, height]
            detection = {
                'image_id': int(image_id),
                'category_id': int(label),
                'bbox': [float(x1), float(y1), width, height],
                'score': float(score)
            }
            
            self.results.append(detection)
    
    def save_results(self, output_path):
        """
        保存结果为JSON文件
        
        参数:
            output_path: 输出文件路径
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"COCO评估结果已保存至: {output_path}")
        print(f"总检测数: {len(self.results)}")
    
    def get_statistics(self):
        """获取检测统计信息"""
        from collections import Counter
        
        category_counts = Counter([det['category_id'] for det in self.results])
        image_counts = Counter([det['image_id'] for det in self.results])
        
        stats = {
            'total_detections': len(self.results),
            'total_images': len(image_counts),
            'avg_detections_per_image': len(self.results) / len(image_counts) if image_counts else 0,
            'detections_per_category': dict(category_counts)
        }
        
        return stats


def process_dataset(model, device, image_dir, annotation_file, output_path, 
                   conf_threshold=0.3, batch_size=1):
    """
    处理整个数据集并生成COCO评估JSON
    
    参数:
        model: 部署模式的模型
        device: 推理设备
        image_dir: 图像目录路径
        annotation_file: COCO标注文件路径
        output_path: 输出JSON文件路径
        conf_threshold: 置信度阈值
        batch_size: 批处理大小（建议设为1以确保准确性）
    """
    # 初始化结果生成器
    result_generator = COCOResultGenerator(annotation_file)
    
    # 定义图像转换
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # 获取所有需要处理的图像
    image_list = sorted(result_generator.image_id_to_filename.items())
    
    print(f"开始处理 {len(image_list)} 张图像...")
    
    # 处理每张图像
    model.eval()
    with torch.no_grad():
        for image_id, filename in tqdm(image_list):
            # 构建完整图像路径
            image_path = os.path.join(image_dir, filename)
            
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在 {image_path}")
                continue
            
            try:
                # 加载图像
                im_pil = Image.open(image_path).convert('RGB')
                w, h = im_pil.size
                orig_size = torch.tensor([[w, h]]).to(device)
                
                # 转换图像
                im_data = transforms(im_pil).unsqueeze(0).to(device)
                
                # 模型推理
                output = model(im_data, orig_size)
                labels, boxes, scores = output
                
                # 添加检测结果
                result_generator.add_detections(
                    image_id=image_id,
                    labels=labels[0],
                    boxes=boxes[0],
                    scores=scores[0],
                    threshold=conf_threshold
                )
                
            except Exception as e:
                print(f"处理图像 {filename} 时出错: {str(e)}")
                continue
    
    # 保存结果
    result_generator.save_results(output_path)
    
    # 打印统计信息
    stats = result_generator.get_statistics()
    print("\n检测统计信息:")
    print(f"  总检测数: {stats['total_detections']}")
    print(f"  处理图像数: {stats['total_images']}")
    print(f"  平均每张图检测数: {stats['avg_detections_per_image']:.2f}")
    
    # 打印各类别检测数量（带类别名称）
    print("\n各类别检测数量:")
    for cat_id in sorted(stats['detections_per_category'].keys()):
        count = stats['detections_per_category'][cat_id]
        cat_name = result_generator.category_id_to_name.get(cat_id, f"未知类别{cat_id}")
        print(f"  类别 {cat_id} ({cat_name}): {count} 次")
    
    return result_generator.results


def evaluate_with_coco_api(annotation_file, result_file):
    """
    使用COCO API进行评估，包括每个类别的详细指标
    
    参数:
        annotation_file: 标注文件路径
        result_file: 预测结果文件路径
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        print("\n" + "="*80)
        print("开始COCO评估...")
        print("="*80)
        
        # 加载标注和预测结果
        coco_gt = COCO(annotation_file)
        coco_dt = coco_gt.loadRes(result_file)
        
        # ============= 整体评估 =============
        print("\n【整体评估指标】")
        print("-"*80)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取关键指标
        overall_metrics = {
            'mAP@0.5:0.95': coco_eval.stats[0],
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],
            'AR@10': coco_eval.stats[7],
            'AR@100': coco_eval.stats[8],
            'AR_small': coco_eval.stats[9],
            'AR_medium': coco_eval.stats[10],
            'AR_large': coco_eval.stats[11],
        }
        
        # ============= 每个类别的评估 =============
        print("\n" + "="*80)
        print("【每个类别的详细指标】")
        print("="*80)
        
        # 获取所有类别
        category_ids = sorted(coco_gt.getCatIds())
        category_names = [coco_gt.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
        
        # 存储每个类别的指标
        per_category_metrics = {}
        
        # 为每个类别单独评估
        for cat_id, cat_name in zip(category_ids, category_names):
            # 只评估当前类别
            coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            
            # 提取该类别的AP
            # precision: [TxRxKxAxM] T=IoU阈值, R=recall阈值, K=类别, A=面积, M=maxDets
            precision = coco_eval_cat.eval['precision']
            
            # AP@0.5:0.95 (平均所有IoU阈值)
            ap_50_95 = np.mean(precision[0:10, :, 0, 0, 2])  # IoU 0.5:0.95, all areas, maxDets=100
            
            # AP@0.5
            ap_50 = np.mean(precision[0, :, 0, 0, 2])  # IoU=0.5
            
            # AP@0.75
            ap_75 = np.mean(precision[5, :, 0, 0, 2])  # IoU=0.75
            
            # AR@100 (Average Recall)
            recall = coco_eval_cat.eval['recall']
            ar_100 = np.mean(recall[0:10, 0, 0, 2]) if recall.size > 0 else 0  # all IoUs, all areas, maxDets=100
            
            per_category_metrics[cat_id] = {
                'name': cat_name,
                'AP@0.5:0.95': float(ap_50_95) if not np.isnan(ap_50_95) else 0.0,
                'AP@0.5': float(ap_50) if not np.isnan(ap_50) else 0.0,
                'AP@0.75': float(ap_75) if not np.isnan(ap_75) else 0.0,
                'AR@100': float(ar_100) if not np.isnan(ar_100) else 0.0,
            }
        
        # ============= 打印每个类别的指标 =============
        print(f"\n{'类别ID':<8} {'类别名称':<20} {'AP@0.5:0.95':<12} {'AP@0.5':<10} {'AP@0.75':<10} {'AR@100':<10}")
        print("-"*80)
        
        for cat_id in category_ids:
            metrics = per_category_metrics[cat_id]
            print(f"{cat_id:<8} {metrics['name']:<20} "
                  f"{metrics['AP@0.5:0.95']:<12.4f} "
                  f"{metrics['AP@0.5']:<10.4f} "
                  f"{metrics['AP@0.75']:<10.4f} "
                  f"{metrics['AR@100']:<10.4f}")
        
        # ============= 统计分析 =============
        print("\n" + "="*80)
        print("【统计分析】")
        print("-"*80)
        
        # 计算类别间的统计
        all_ap_50_95 = [m['AP@0.5:0.95'] for m in per_category_metrics.values()]
        all_ap_50 = [m['AP@0.5'] for m in per_category_metrics.values()]
        
        print(f"AP@0.5:0.95 统计:")
        print(f"  平均值: {np.mean(all_ap_50_95):.4f}")
        print(f"  中位数: {np.median(all_ap_50_95):.4f}")
        print(f"  最大值: {np.max(all_ap_50_95):.4f}")
        print(f"  最小值: {np.min(all_ap_50_95):.4f}")
        print(f"  标准差: {np.std(all_ap_50_95):.4f}")
        
        # 找出表现最好和最差的类别
        sorted_by_ap = sorted(per_category_metrics.items(), 
                            key=lambda x: x[1]['AP@0.5:0.95'], 
                            reverse=True)
        
        print(f"\nAP@0.5:0.95 最高的5个类别:")
        for cat_id, metrics in sorted_by_ap[:5]:
            print(f"  {metrics['name']}: {metrics['AP@0.5:0.95']:.4f}")
        
        print(f"\nAP@0.5:0.95 最低的5个类别:")
        for cat_id, metrics in sorted_by_ap[-5:]:
            print(f"  {metrics['name']}: {metrics['AP@0.5:0.95']:.4f}")
        
        # ============= 按象限分组统计（如果是牙齿数据集）=============
        if all(isinstance(name, str) and name.startswith('FDI') for name in category_names[:5]):
            print("\n" + "="*80)
            print("【按象限分组统计】")
            print("-"*80)
            
            quadrant_stats = defaultdict(list)
            for cat_id, metrics in per_category_metrics.items():
                name = metrics['name']
                if 'FDI' in name:
                    # 提取FDI编号（例如 "FDI_11" -> 11）
                    fdi_num = int(name.split('_')[1]) if '_' in name else 0
                    quadrant = fdi_num // 10  # 第一位数字是象限
                    quadrant_stats[quadrant].append(metrics['AP@0.5:0.95'])
            
            for quad in sorted(quadrant_stats.keys()):
                aps = quadrant_stats[quad]
                quadrant_name = {
                    1: "第1象限(右上恒牙)",
                    2: "第2象限(左上恒牙)", 
                    3: "第3象限(左下恒牙)",
                    4: "第4象限(右下恒牙)",
                    5: "第5象限(右上乳牙)",
                    6: "第6象限(左上乳牙)",
                    7: "第7象限(左下乳牙)",
                    8: "第8象限(右下乳牙)"
                }.get(quad, f"象限{quad}")
                
                print(f"{quadrant_name}:")
                print(f"  平均AP: {np.mean(aps):.4f}")
                print(f"  牙齿数: {len(aps)}")
        
        # ============= 保存详细结果到文件 =============
        output_dir = Path(result_file).parent
        detail_file = output_dir / 'evaluation_details.json'
        
        evaluation_results = {
            'overall_metrics': overall_metrics,
            'per_category_metrics': per_category_metrics,
            'statistics': {
                'mean_ap_50_95': float(np.mean(all_ap_50_95)),
                'median_ap_50_95': float(np.median(all_ap_50_95)),
                'std_ap_50_95': float(np.std(all_ap_50_95)),
                'max_ap_50_95': float(np.max(all_ap_50_95)),
                'min_ap_50_95': float(np.min(all_ap_50_95)),
            }
        }
        
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细评估结果已保存至: {detail_file}")
        
        return overall_metrics, per_category_metrics
        
    except ImportError:
        print("\n警告: 未安装pycocotools，跳过COCO评估")
        print("可通过以下命令安装: pip install pycocotools")
        return None, None


def main(args):
    """主函数"""
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
        raise AttributeError('必须提供模型权重文件 (--resume)')

    # 加载训练模式的state并转换为部署模式
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

    device = args.device
    model = Model().to(device)
    
    print(f"使用设备: {device}")
    print(f"模型权重: {args.resume}")
    print(f"置信度阈值: {args.conf_threshold}")
    
    # 处理数据集并生成结果
    results = process_dataset(
        model=model,
        device=device,
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        batch_size=args.batch_size
    )
    
    # 如果指定了评估选项，则执行COCO评估
    if args.evaluate:
        overall_metrics, per_category_metrics = evaluate_with_coco_api(
            args.annotation_file, 
            args.output
        )
        
        if overall_metrics:
            print("\n" + "="*80)
            print("【整体评估指标汇总】")
            print("="*80)
            for key, value in overall_metrics.items():
                print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DEIM模型推理并生成COCO评估JSON')
    parser.add_argument('-c', '--config', type=str, required=True, 
                       help='模型配置文件路径')
    parser.add_argument('-r', '--resume', type=str, required=True, 
                       help='模型权重文件路径')
    parser.add_argument('-i', '--image_dir', type=str, required=True, 
                       help='图像目录路径')
    parser.add_argument('-a', '--annotation_file', type=str, required=True, 
                       help='COCO格式的标注文件路径')
    parser.add_argument('-o', '--output', type=str, default='predictions.json', 
                       help='输出JSON文件路径')
    parser.add_argument('-d', '--device', type=str, default='cuda', 
                       help='推理设备 (cuda/cpu)')
    parser.add_argument('--conf_threshold', type=float, default=0.3, 
                       help='置信度阈值')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='批处理大小')
    parser.add_argument('--evaluate', action='store_true', 
                       help='生成JSON后立即进行COCO评估')
    
    args = parser.parse_args()
    main(args)

"""python inference.py     --config /home/wuxi/DEIM_origin/configs/deim_dfine/deim_hgnetv2_s_coco.yml     --resume /home/wuxi/DEIM_origin/outputs/deim_hgnetv2_s_coco/best_stg1.pth     --image_dir /home/wuxi/coco/val     --annotation_file /home/wuxi/coco/val/annotations_fdi_types.json     --output results/predictions.json     --device cuda     --conf_threshold 0.3     --evaluate"""