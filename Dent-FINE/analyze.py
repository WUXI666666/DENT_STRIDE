import json
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class OfflineToothAnalyzer:
    """使用推理结果json进行离线分析"""
    
    def __init__(self, pred_json_path, gt_json_path, iou_threshold=0.5):
        self.pred_json_path = pred_json_path
        self.gt_json_path = gt_json_path
        self.iou_threshold = iou_threshold
        
        # 加载数据
        with open(pred_json_path, 'r') as f:
            self.predictions = json.load(f)
        with open(gt_json_path, 'r') as f:
            self.ground_truth = json.load(f)
        
        # 初始化统计结构
        self.num_classes = 52
        self.tp_per_class = np.zeros(self.num_classes)
        self.fp_per_class = np.zeros(self.num_classes)
        self.fn_per_class = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        # 额外统计
        self.detection_errors = []  # 记录每个错误检测的详细信息
        
    def compute_iou(self, box1, box2):
        """计算两个框的IoU (box format: [x, y, w, h])"""
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def match_predictions_to_gt(self, pred_boxes, pred_classes, gt_boxes, gt_classes):
        """使用匈牙利算法匹配预测和GT"""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # 构建IoU代价矩阵
        cost_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                cost_matrix[i, j] = -self.compute_iou(pred_box, gt_box)
        
        # 匈牙利匹配
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # 过滤低IoU匹配
        matched_pairs = []
        unmatched_preds = set(range(len(pred_boxes)))
        unmatched_gts = set(range(len(gt_boxes)))
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = -cost_matrix[pred_idx, gt_idx]
            if iou >= self.iou_threshold:
                matched_pairs.append((pred_idx, gt_idx, iou))
                unmatched_preds.discard(pred_idx)
                unmatched_gts.discard(gt_idx)
        
        return matched_pairs, list(unmatched_preds), list(unmatched_gts)
    
    def analyze(self):
        """执行完整分析"""
        # 按图像ID组织预测和GT
        pred_by_image = defaultdict(list)
        for pred in self.predictions:
            pred_by_image[pred['image_id']].append(pred)
        
        gt_by_image = defaultdict(list)
        for ann in self.ground_truth.get('annotations', []):
            gt_by_image[ann['image_id']].append(ann)
        
        # 对每张图像进行分析
        for image_id in gt_by_image.keys():
            preds = pred_by_image.get(image_id, [])
            gts = gt_by_image[image_id]
            
            pred_boxes = [p['bbox'] for p in preds]
            pred_classes = [p['category_id'] for p in preds]
            pred_scores = [p.get('score', 1.0) for p in preds]
            
            gt_boxes = [g['bbox'] for g in gts]
            gt_classes = [g['category_id'] for g in gts]
            
            # 执行匹配
            matches, unmatched_preds, unmatched_gts = self.match_predictions_to_gt(
                pred_boxes, pred_classes, gt_boxes, gt_classes
            )
            
            # 更新统计
            for pred_idx, gt_idx, iou in matches:
                pred_cls = pred_classes[pred_idx]
                gt_cls = gt_classes[gt_idx]
                
                self.confusion_matrix[gt_cls, pred_cls] += 1
                
                if pred_cls == gt_cls:
                    self.tp_per_class[gt_cls] += 1
                else:
                    self.fp_per_class[pred_cls] += 1
                    self.fn_per_class[gt_cls] += 1
                    
                    # 记录错误详情
                    self.detection_errors.append({
                        'image_id': image_id,
                        'gt_class': gt_cls,
                        'pred_class': pred_cls,
                        'iou': iou,
                        'score': pred_scores[pred_idx],
                        'error_type': 'classification'
                    })
            
            # 记录假阳性（未匹配的预测）
            for pred_idx in unmatched_preds:
                pred_cls = pred_classes[pred_idx]
                self.fp_per_class[pred_cls] += 1
                self.detection_errors.append({
                    'image_id': image_id,
                    'pred_class': pred_cls,
                    'score': pred_scores[pred_idx],
                    'error_type': 'false_positive'
                })
            
            # 记录假阴性（未匹配的GT）
            for gt_idx in unmatched_gts:
                gt_cls = gt_classes[gt_idx]
                self.fn_per_class[gt_cls] += 1
                self.detection_errors.append({
                    'image_id': image_id,
                    'gt_class': gt_cls,
                    'error_type': 'false_negative'
                })
    
    def generate_report(self, output_path='offline_analysis_report.txt'):
        """生成详细分析报告"""
        from engine.deim.utils_arch import fdi_index_map
        
        fdi_map = fdi_index_map()
        idx_to_fdi = {v: k for k, v in fdi_map.items()}
        
        precision = self.tp_per_class / (self.tp_per_class + self.fp_per_class + 1e-6)
        recall = self.tp_per_class / (self.tp_per_class + self.fn_per_class + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 牙齿检测离线分析报告 ===\n\n")
            f.write(f"预测文件: {self.pred_json_path}\n")
            f.write(f"真值文件: {self.gt_json_path}\n")
            f.write(f"IoU阈值: {self.iou_threshold}\n\n")
            
            # 总体统计
            total_tp = self.tp_per_class.sum()
            total_fp = self.fp_per_class.sum()
            total_fn = self.fn_per_class.sum()
            overall_precision = total_tp / (total_tp + total_fp + 1e-6)
            overall_recall = total_tp / (total_tp + total_fn + 1e-6)
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
            
            f.write(f"总体性能:\n")
            f.write(f"  Precision: {overall_precision:.4f}\n")
            f.write(f"  Recall: {overall_recall:.4f}\n")
            f.write(f"  F1: {overall_f1:.4f}\n\n")
            
            # 详细类别统计
            f.write("各类别详细性能:\n")
            f.write(f"{'FDI':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
            f.write("-" * 80 + "\n")
            
            for idx in range(self.num_classes):
                if self.tp_per_class[idx] + self.fn_per_class[idx] > 0:  # 只显示有GT的类别
                    fdi = idx_to_fdi.get(idx, f'ID{idx}')
                    f.write(f"{fdi:<8} {int(self.tp_per_class[idx]):<8} {int(self.fp_per_class[idx]):<8} "
                           f"{int(self.fn_per_class[idx]):<8} {precision[idx]:<12.4f} {recall[idx]:<12.4f} "
                           f"{f1[idx]:<12.4f}\n")
            
            # 常见错误模式分析
            f.write("\n\n=== 常见错误模式 ===\n\n")
            
            # 分类错误
            classification_errors = [e for e in self.detection_errors if e['error_type'] == 'classification']
            if classification_errors:
                error_counts = defaultdict(int)
                for err in classification_errors:
                    pair = (err['gt_class'], err['pred_class'])
                    error_counts[pair] += 1
                
                sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
                f.write("最常见的分类错误（真实→预测）:\n")
                for (gt_cls, pred_cls), count in sorted_errors[:15]:
                    gt_fdi = idx_to_fdi.get(gt_cls, f'ID{gt_cls}')
                    pred_fdi = idx_to_fdi.get(pred_cls, f'ID{pred_cls}')
                    f.write(f"  {gt_fdi} → {pred_fdi}: {count}次\n")
            
            # 漏检最严重的类别
            f.write("\n漏检最严重的类别:\n")
            worst_fn_indices = np.argsort(self.fn_per_class)[::-1][:10]
            for idx in worst_fn_indices:
                if self.fn_per_class[idx] > 0:
                    fdi = idx_to_fdi.get(idx, f'ID{idx}')
                    f.write(f"  {fdi}: {int(self.fn_per_class[idx])}次漏检\n")
            
            # 误检最多的类别
            f.write("\n误检最多的类别:\n")
            worst_fp_indices = np.argsort(self.fp_per_class)[::-1][:10]
            for idx in worst_fp_indices:
                if self.fp_per_class[idx] > 0:
                    fdi = idx_to_fdi.get(idx, f'ID{idx}')
                    f.write(f"  {fdi}: {int(self.fp_per_class[idx])}次误检\n")
        
        print(f"离线分析报告已生成: {output_path}")
        return precision, recall, f1

# 使用示例
if __name__ == '__main__':
    analyzer = OfflineToothAnalyzer(
        pred_json_path='results/predictions.json',
        gt_json_path='/home/wuxi/coco/val/annotations_fdi_types.json',
        iou_threshold=0.5
    )
    
    print("开始分析...")
    analyzer.analyze()
    
    print("生成报告...")
    precision, recall, f1 = analyzer.generate_report('tooth_analysis_report.txt')
    
    print(f"\n总体mAP: {f1.mean():.4f}")
    print(f"性能最差的5个类别: {np.argsort(f1)[:5]}")