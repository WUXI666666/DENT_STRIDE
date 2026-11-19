#!/usr/bin/env bash
# Complete the paper-facing View-Net3 DSLR FDI52 baseline suite on GPU 0.
#
# Runs are deliberately serial because the workspace has limited disk space.
# Every completed experiment keeps exactly one selected checkpoint.  The two
# YOLO reruns disable flips: canonical FDI labels carry left/right semantics,
# so the earlier exploratory runs made with Ultralytics' default horizontal
# flip are not suitable for the final comparison table.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
YOLO_BIN="${YOLO_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/yolo}"
D2_PYTHON_BIN="${D2_PYTHON_BIN:-/home/wuxi/miniconda3/envs/mask2former/bin/python}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-160}"
IMGSZ="${IMGSZ:-640}"
YOLO_BATCH_SIZE="${YOLO_BATCH_SIZE:-8}"
YOLO_WORKERS="${YOLO_WORKERS:-8}"
SEED="${SEED:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
START_FROM="${START_FROM:-caf_yolo_s_fdi52}"
MAX_RUNS="${MAX_RUNS:-0}"

BASE_DIR="${ROOT_DIR}/outputs/viewnet3_dslr_baseline_comparison"
RUNS_DIR="${BASE_DIR}/runs"
WEIGHTS_DIR="${BASE_DIR}/weights"
YOLO_DATA_YAML="${BASE_DIR}/yolo_dataset/viewnet3_dslr_fdi52.yaml"
VAL_ANN="${ROOT_DIR}/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json"
VAL_IMAGES="/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_viewnet3_dslr_baselines.py"
YOLO_EVALUATOR="${ROOT_DIR}/tools/dental/evaluate_yolo_coco_detection.py"
CAF_RUNNER="${ROOT_DIR}/tools/dental/run_caf_yolo_viewnet3_dslr_fdi52.py"
FASTER_RCNN_RUNNER="${ROOT_DIR}/tools/dental/train_faster_rcnn_viewnet3_dslr_fdi52.py"
DOME_RUNNER="${ROOT_DIR}/tools/run_dome_detr_viewnet3_dslr_fdi52.sh"

# CAF first validates the requested method promptly; the two no-flip reruns
# replace exploratory values; Faster R-CNN and Dome-DETR then run serially.
ORDER=(
  caf_yolo_s_fdi52
  yolov8s_fdi52_no_flip
  yolo11s_fdi52_no_flip
  faster_rcnn_r50_fpn_fdi52
  dome_detr_s_fdi52
)

CURRENT_EXP_DIR=""
RUN_COUNT=0

summarize() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --base-dir "${BASE_DIR}" --expected-epochs "${EPOCHS}"
}

on_exit() {
  local exit_code="$?"
  if [[ "${exit_code}" -ne 0 && -n "${CURRENT_EXP_DIR}" ]]; then
    rm -f "${CURRENT_EXP_DIR}/RUNNING"
    touch "${CURRENT_EXP_DIR}/FAILED"
  fi
  summarize || true
  exit "${exit_code}"
}
trap on_exit EXIT

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

should_run() {
  local name="$1"
  local start_index=-1
  local current_index=-1
  local index
  for index in "${!ORDER[@]}"; do
    [[ "${ORDER[$index]}" == "${START_FROM}" ]] && start_index="${index}"
    [[ "${ORDER[$index]}" == "${name}" ]] && current_index="${index}"
  done
  if [[ "${start_index}" -lt 0 || "${current_index}" -lt 0 ]]; then
    echo "Invalid START_FROM=${START_FROM}; expected one of: ${ORDER[*]}" >&2
    exit 2
  fi
  [[ "${current_index}" -ge "${start_index}" ]]
}

count_run_and_maybe_stop() {
  RUN_COUNT=$((RUN_COUNT + 1))
  summarize
  if [[ "${MAX_RUNS}" -gt 0 && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}; queue stopped cleanly."
    exit 0
  fi
}

cleanup_yolo_weights() {
  local exp_dir="$1"
  local best_weight="${exp_dir}/weights/best.pt"
  [[ -f "${best_weight}" ]] || { echo "Cannot clean without ${best_weight}" >&2; return 1; }
  find "${exp_dir}/weights" -type f -name '*.pt' ! -path "${best_weight}" -delete
  [[ "$(find "${exp_dir}/weights" -type f -name '*.pt' -print | wc -l | tr -d '[:space:]')" == "1" ]] || {
    echo "Expected one retained YOLO checkpoint in ${exp_dir}" >&2
    return 1
  }
}

run_yolo_no_flip() {
  local name="$1"
  local weight_name="$2"
  if ! should_run "${name}"; then
    # A skipped predecessor is a successful no-op.  With `set -e`, a bare
    # `return` would otherwise inherit should_run's status 1 and abort a
    # START_FROM continuation before it reaches the requested experiment.
    return 0
  fi
  local exp_dir="${RUNS_DIR}/${name}"
  local init_weight="${WEIGHTS_DIR}/${weight_name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/weights/best.pt" ]]; then
    echo "[skip] ${name} is already completed"
    return
  fi
  if [[ -f "${exp_dir}/results.csv" && ! -f "${exp_dir}/weights/last.pt" ]]; then
    echo "Refusing to overwrite an unresumable partial YOLO run: ${exp_dir}" >&2
    return 1
  fi
  require_file "${init_weight}"
  mkdir -p "${exp_dir}"
  rm -f "${exp_dir}/FAILED" "${exp_dir}/COMPLETED"
  touch "${exp_dir}/RUNNING"
  CURRENT_EXP_DIR="${exp_dir}"

  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: ${name} (FDI-safe no-flip rerun)"
  echo "============================================================"
  if [[ -f "${exp_dir}/weights/last.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${YOLO_BIN}" detect train resume="${exp_dir}/weights/last.pt"
  else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${YOLO_BIN}" detect train \
      model="${init_weight}" data="${YOLO_DATA_YAML}" \
      epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${YOLO_BATCH_SIZE}" workers="${YOLO_WORKERS}" \
      patience="${EPOCHS}" \
      device=0 project="${RUNS_DIR}" name="${name}" exist_ok=True \
      pretrained=True optimizer=AdamW lr0=0.0002 momentum=0.9 weight_decay=0.0001 \
      warmup_epochs=3.0 warmup_bias_lr=0.0 close_mosaic=30 cos_lr=False \
      seed="${SEED}" deterministic=True amp=True cache=False rect=False \
      save=True save_period=-1 plots=True flipud=0.0 fliplr=0.0
  fi
  [[ -f "${exp_dir}/weights/best.pt" ]] || { echo "${name} did not emit best.pt" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${YOLO_EVALUATOR}" \
    --weights "${exp_dir}/weights/best.pt" --ann-file "${VAL_ANN}" --image-dir "${VAL_IMAGES}" \
    --output-dir "${exp_dir}/coco_eval" --imgsz "${IMGSZ}" --device 0 --batch "${YOLO_BATCH_SIZE}" \
    --conf 0.001 --iou 0.7 --max-det 300
  cleanup_yolo_weights "${exp_dir}"
  rm -f "${exp_dir}/RUNNING" "${exp_dir}/FAILED"
  touch "${exp_dir}/COMPLETED"
  CURRENT_EXP_DIR=""
  count_run_and_maybe_stop
}

run_caf() {
  local name="caf_yolo_s_fdi52"
  should_run "${name}" || return 0
  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: CAF-YOLO-s"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${CAF_RUNNER}" --device 0
  count_run_and_maybe_stop
}

run_faster_rcnn() {
  local name="faster_rcnn_r50_fpn_fdi52"
  should_run "${name}" || return 0
  local exp_dir="${RUNS_DIR}/${name}"
  local args=()
  if [[ -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/model_best.pth" && "${SKIP_COMPLETED}" == "1" ]]; then
    echo "[skip] ${name} is already completed"
    return
  fi
  if [[ -f "${exp_dir}/last_checkpoint" ]]; then
    args+=(--resume)
  elif [[ -f "${exp_dir}/FAILED" ]]; then
    # A failed run without Detectron2's recovery marker has no model/optimizer
    # state worth preserving.  Explicitly authorize only this known failed
    # directory to be reset; arbitrary non-empty directories remain guarded by
    # the runner itself.
    echo "[restart] ${name} failed before a recoverable checkpoint; restarting from iteration 0"
    args+=(--overwrite)
  fi
  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: Faster R-CNN R50-FPN"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="/home/wuxi/detectron2${PYTHONPATH:+:${PYTHONPATH}}" \
    "${D2_PYTHON_BIN}" "${FASTER_RCNN_RUNNER}" --gpu-id 0 "${args[@]}"
  count_run_and_maybe_stop
}

run_dome() {
  local name="dome_detr_s_fdi52"
  should_run "${name}" || return 0
  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: Dome-DETR-S"
  echo "============================================================"
  GPU_ID="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" "${DOME_RUNNER}"
  count_run_and_maybe_stop
}

main() {
  [[ "${GPU_ID}" == "0" ]] || { echo "This paper queue is pinned to GPU 0 (got GPU_ID=${GPU_ID})" >&2; exit 2; }
  [[ "${EPOCHS}" == "160" ]] || { echo "Comparison is pinned to 160 epochs (got ${EPOCHS})" >&2; exit 2; }
  [[ "${IMGSZ}" == "640" ]] || { echo "Comparison is pinned to 640 px (got ${IMGSZ})" >&2; exit 2; }
  [[ "${SEED}" == "0" ]] || { echo "Comparison is pinned to seed 0 (got ${SEED})" >&2; exit 2; }
  [[ -x "${PYTHON_BIN}" && -x "${YOLO_BIN}" && -x "${D2_PYTHON_BIN}" ]] || { echo "Required interpreter/YOLO executable is missing" >&2; exit 1; }
  require_file "${YOLO_DATA_YAML}"
  require_file "${VAL_ANN}"
  require_file "${YOLO_EVALUATOR}"
  require_file "${CAF_RUNNER}"
  require_file "${FASTER_RCNN_RUNNER}"
  require_file "${DOME_RUNNER}"
  mkdir -p "${RUNS_DIR}"
  summarize
  run_caf
  run_yolo_no_flip yolov8s_fdi52_no_flip yolov8s.pt
  run_yolo_no_flip yolo11s_fdi52_no_flip yolo11s.pt
  run_faster_rcnn
  run_dome
  echo "All View-Net3 DSLR extended baselines completed."
}

main "$@"
