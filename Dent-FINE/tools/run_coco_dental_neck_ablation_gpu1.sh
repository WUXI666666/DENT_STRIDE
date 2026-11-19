#!/usr/bin/env bash
set -euo pipefail

cd /home/wuxi/Dent-FINE

PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_INDEX="${GPU_INDEX:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/coco_dental_neck_ablation_fixed}"
EPOCHS="${EPOCHS:-160}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
NO_AMP="${NO_AMP:-0}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-0}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-0}"
EXPERIMENTS="${EXPERIMENTS:-}"

ARGS=(
  tools/dental/run_coco_neck_ablation.py
  --python "${PYTHON_BIN}"
  --base-config configs/deim_dfine/deim_hgnetv2_s_phone.yml
  --output-root "${OUTPUT_ROOT}"
  --epochs "${EPOCHS}"
  --device cuda:0
  --seed "${SEED}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-train-images "${MAX_TRAIN_IMAGES}"
  --max-val-images "${MAX_VAL_IMAGES}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${NO_AMP}" == "1" ]]; then
  ARGS+=(--no-amp)
fi
if [[ -n "${EXPERIMENTS}" ]]; then
  read -r -a EXPERIMENT_ARRAY <<< "${EXPERIMENTS}"
  ARGS+=(--experiments "${EXPERIMENT_ARRAY[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" "${ARGS[@]}"
