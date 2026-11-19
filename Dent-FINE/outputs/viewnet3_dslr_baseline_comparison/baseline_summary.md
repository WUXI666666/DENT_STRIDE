# View-Net3 DSLR FDI52 Baseline Comparison

- Split and label space: identical 750-image train / 190-image validation cohort with canonical zero-based FDI52 labels.
- Training budget: 160 epochs, 640×640 input, seed 0, and one GPU. Batch size is 8 except Faster R-CNN (4); each framework retains only its final selected best weight.
- Geometric horizontal/vertical flips are disabled because FDI labels encode left/right tooth position. The initial exploratory YOLO artifacts with default horizontal flip are excluded from this paper-facing table.
- Primary metrics are standard COCO bbox metrics on the same validation JSON. Native Ultralytics mAP is retained separately for traceability.

| Rank | Method | Framework | Status | Epochs | Best epoch | COCO AP | AP50 | AP75 | AR100 | Δ vs best | Params | Best weight |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | YOLOv8-s | Ultralytics | done | 160 | 55 | 71.854 | 88.438 | 83.696 | 76.531 | -0.706 | 11.15M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/yolov8s_fdi52_no_flip/weights/best.pt` |
| 4 | YOLO11-s | Ultralytics | done | 160 | 59 | 71.094 | 88.065 | 81.258 | 75.179 | -1.466 | 9.43M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/yolo11s_fdi52_no_flip/weights/best.pt` |
| 5 | CAF-YOLO-s | CAF-YOLO (Ultralytics 8.0.228) | done | 160 | 0 | 67.460 | 84.845 | 77.979 | 71.386 | -5.100 | 11.50M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/caf_yolo_s_fdi52/weights/best.pt` |
| 6 | Faster R-CNN R50-FPN | Detectron2 | done | 160 | 56 | 63.151 | 82.211 | 75.552 | 73.723 | -9.409 | 41.56M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/faster_rcnn_r50_fpn_fdi52/model_best.pth` |
| 3 | Dome-DETR-S | Dome-DETR | done | 160 | 152 | 71.400 | 86.700 | 81.000 | 85.400 | -1.160 | 13.20M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/dome_detr_s_fdi52/best.pth` |
| 1 | DEIM-s | Dent-FINE | done | 160 | 133 | 72.560 | 88.245 | 83.249 | 81.704 | +0.000 | 10.63M | `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/runs/deim_s_fdi52/best.pth` |

## Reproducibility

- YOLOv8-s / YOLO11-s: Ultralytics detection models initialized from their official pretrained small checkpoints and optimized with AdamW, with FDI-safe flips disabled.
- CAF-YOLO-s: authors' CAFMAttention YOLOv8 fork, small scale, initialized by compatible weights from the official YOLOv8-s checkpoint and trained with the authors' non-AMP setting.
- Faster R-CNN R50-FPN: Detectron2 implementation initialized from the official COCO R50-FPN 3x checkpoint and optimized with SGD.
- Dome-DETR-S: authors' official HGNetv2-B0 small configuration, adapted only for the canonical FDI52 data paths, class count, COCO evaluator, and 640-pixel protocol.
- DEIM-s: HGNetv2-B0 / 3-layer DEIM configuration initialized from the project-local pretrained HGNetv2 weights; all Dent-FINE additions and architecture-prior loss are disabled, with standard AdamW.
- Evaluation artifacts: `runs/<model>/coco_eval/coco_metrics.json`; DEIM also records evaluator results in `runs/deim_s_fdi52/log.txt`.

## Files

- CSV: `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/baseline_summary.csv`
- Markdown: `/home/wuxi/Dent-FINE/outputs/viewnet3_dslr_baseline_comparison/baseline_summary.md`
