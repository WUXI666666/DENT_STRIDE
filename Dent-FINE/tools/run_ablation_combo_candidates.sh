#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_all_new_modules}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
START_FROM="${START_FROM:-abl12_cmfa_mpf}"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_ablation_all_new_modules.py"

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
  HybridEncoder.use_mpc_multiscale_perception=False
  HybridEncoder.use_pqg_high_frequency=False
  HybridEncoder.use_cmfa_fusion=False
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_uni_gate_infer=False
  DFINETransformer.uni_gate_apply_last_only=True
  DFINETransformer.use_qgp=False
  DFINETransformer.qgp_layers=[]
  DFINETransformer.use_pgr_roi_reasoning=False
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
)

ORDER=(
  abl12_cmfa_mpf
  abl13_cmfa_pgr
  abl14_mpf_pgr
  abl15_cmfa_mpf_pgr
  abl16_stat_noise_dynamic_rf
  abl17_stat_noise_mpc
)

should_run() {
  local name="$1"
  local start_idx=-1
  local cur_idx=-1
  local i
  for i in "${!ORDER[@]}"; do
    if [[ "${ORDER[$i]}" == "${START_FROM}" ]]; then
      start_idx="$i"
    fi
    if [[ "${ORDER[$i]}" == "${name}" ]]; then
      cur_idx="$i"
    fi
  done
  if [[ "${start_idx}" -lt 0 || "${cur_idx}" -lt 0 ]]; then
    echo "Invalid START_FROM=${START_FROM}" >&2
    exit 1
  fi
  [[ "${cur_idx}" -ge "${start_idx}" ]]
}

refresh_summary() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --base-output-dir "${ROOT_DIR}/${BASE_OUTPUT_DIR}"
}

run_exp() {
  local name="$1"
  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
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

  refresh_summary
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"
refresh_summary

if should_run abl12_cmfa_mpf; then
  run_exp abl12_cmfa_mpf \
    HybridEncoder.use_cmfa_fusion=True \
    HybridEncoder.use_multipath_fusion=True
fi
if should_run abl13_cmfa_pgr; then
  run_exp abl13_cmfa_pgr \
    HybridEncoder.use_cmfa_fusion=True \
    DFINETransformer.use_pgr_roi_reasoning=True
fi
if should_run abl14_mpf_pgr; then
  run_exp abl14_mpf_pgr \
    HybridEncoder.use_multipath_fusion=True \
    DFINETransformer.use_pgr_roi_reasoning=True
fi
if should_run abl15_cmfa_mpf_pgr; then
  run_exp abl15_cmfa_mpf_pgr \
    HybridEncoder.use_cmfa_fusion=True \
    HybridEncoder.use_multipath_fusion=True \
    DFINETransformer.use_pgr_roi_reasoning=True
fi
if should_run abl16_stat_noise_dynamic_rf; then
  run_exp abl16_stat_noise_dynamic_rf \
    HybridEncoder.use_stat_denoise=True \
    HybridEncoder.use_dynamic_rf=True
fi
if should_run abl17_stat_noise_mpc; then
  run_exp abl17_stat_noise_mpc \
    HybridEncoder.use_stat_denoise=True \
    HybridEncoder.use_mpc_multiscale_perception=True
fi
