#!/usr/bin/env bash
set -euo pipefail

repo=/home/wuxi/Dent-FINE
python_bin=/home/wuxi/miniconda3/envs/deimv2/bin/python
cohort=${1:?usage: queue_final_qualitative_baselines.sh phone|dentex}

wait_for_run() {
    local run_dir=$1
    while [[ ! -f "${run_dir}/COMPLETED" ]]; do
        if [[ -f "${run_dir}/FAILED" ]]; then
            echo "FAILED prerequisite: ${run_dir}" >&2
            exit 1
        fi
        sleep 20
    done
    echo "COMPLETED prerequisite: ${run_dir}"
}

cd "$repo"

if [[ "$cohort" == phone ]]; then
    base=outputs/cross_dataset_baseline_comparison/phone/runs
    wait_for_run "$base/rtdetr_l"
    wait_for_run "$base/caf_yolo_s"
    wait_for_run "$base/faster_rcnn"
    "$python_bin" tools/dental/run_dome_detr_dental_baseline.py \
        --train-ann /home/wuxi/coco/train/annotations_fdi.json \
        --val-ann /home/wuxi/coco/valid/annotations_fdi.json \
        --train-images /home/wuxi/coco/train \
        --val-images /home/wuxi/coco/valid \
        --num-classes 52 \
        --output-dir "$base/dome_detr_s" \
        --gpu 0 --epochs 160 --batch 8 --workers 8
elif [[ "$cohort" == dentex ]]; then
    base=outputs/cross_dataset_baseline_comparison/dentex/runs
    wait_for_run "$base/rtdetr_l"
    wait_for_run "$base/caf_yolo_s"
    wait_for_run "$base/faster_rcnn"
    "$python_bin" tools/dental/run_dome_detr_dental_baseline.py \
        --train-ann /home/wuxi/DENTEX2023/annotations/instances_train2017.json \
        --val-ann /home/wuxi/DENTEX2023/annotations/instances_val2017.json \
        --train-images /home/wuxi/DENTEX2023/train2017 \
        --val-images /home/wuxi/DENTEX2023/val2017 \
        --num-classes 32 \
        --output-dir "$base/dome_detr_s" \
        --gpu 1 --epochs 160 --batch 8 --workers 8
    "$python_bin" tools/dental/run_yolo_family_dental_baselines.py \
        --data outputs/viewnet3_dslr_baseline_comparison/yolo_dataset/viewnet3_dslr_fdi52.yaml \
        --annotation datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json \
        --images /home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val \
        --output-dir outputs/cross_dataset_baseline_comparison/dslr \
        --gpu 1 --epochs 160 --batch 8 --workers 8 --only rtdetr_l
else
    echo "Unknown cohort queue: ${cohort}" >&2
    exit 2
fi
