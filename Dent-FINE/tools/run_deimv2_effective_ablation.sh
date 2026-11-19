#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/deim_hgnetv2_s_coco_deimv2_effective_ablation}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
START_FROM="${START_FROM:-abl00_baseline}"

COMMON_ARGS=(
  train.py
  -c "$CONFIG"
  --use-amp
  --seed=0
  -u
  HybridEncoder.use_hfsa=False
  HybridEncoder.use_wav_enhance=False
  HybridEncoder.use_stat_denoise=False
  HybridEncoder.use_multipath_fusion=False
  HybridEncoder.use_dynamic_rf=False
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_qgp=False
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
)

run_exp() {
  local name="$1"
  local exp_output="${BASE_OUTPUT_DIR}/${name}"
  shift
  echo "=============================="
  echo "Running ${name}"
  echo "=============================="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    "$@" \
    --output-dir "${exp_output}"

  if [[ "${KEEP_ONLY_BEST}" == "1" && -d "${exp_output}" ]]; then
    find "${exp_output}" -type f -name '*.pth' \
      ! -name 'best_stg1.pth' \
      ! -name 'best.pth' \
      -delete
  fi
}

should_run() {
  local name="$1"
  local order=(
    abl00_baseline
    abl01_mpc
    abl02_pqg_hf
    abl03_cmfa
    abl04_pgr
    abl05_full_effective
  )
  local start_idx=-1
  local cur_idx=-1
  local i
  for i in "${!order[@]}"; do
    if [[ "${order[$i]}" == "${START_FROM}" ]]; then
      start_idx="$i"
    fi
    if [[ "${order[$i]}" == "${name}" ]]; then
      cur_idx="$i"
    fi
  done
  if [[ "${start_idx}" -lt 0 || "${cur_idx}" -lt 0 ]]; then
    echo "Invalid START_FROM=${START_FROM}" >&2
    exit 1
  fi
  [[ "${cur_idx}" -ge "${start_idx}" ]]
}

cd "${ROOT_DIR}"

if should_run abl00_baseline; then
  run_exp abl00_baseline
fi
if should_run abl01_mpc; then
  run_exp abl01_mpc \
    HybridEncoder.use_mpc_multiscale_perception=True
fi
if should_run abl02_pqg_hf; then
  run_exp abl02_pqg_hf \
    HybridEncoder.use_pqg_high_frequency=True
fi
if should_run abl03_cmfa; then
  run_exp abl03_cmfa \
    HybridEncoder.use_cmfa_fusion=True
fi
if should_run abl04_pgr; then
  run_exp abl04_pgr \
    DFINETransformer.use_pgr_roi_reasoning=True
fi
if should_run abl05_full_effective; then
  run_exp abl05_full_effective \
    HybridEncoder.use_mpc_multiscale_perception=True \
    HybridEncoder.use_pqg_high_frequency=True \
    HybridEncoder.use_cmfa_fusion=True \
    DFINETransformer.use_pgr_roi_reasoning=True
fi
