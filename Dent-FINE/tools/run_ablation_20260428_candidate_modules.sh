#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_all_new_modules}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
START_FROM="${START_FROM:-abl31_hooa_attention}"
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
  HybridEncoder.use_pfesa=False
  HybridEncoder.use_ucan_lkd=False
  HybridEncoder.use_ucm_uncertainty=False
  HybridEncoder.use_retinex_prior=False
  HybridEncoder.use_hooa_attention=False
  HybridEncoder.use_multinex_prior=False
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_uni_gate_infer=False
  DFINETransformer.uni_gate_apply_last_only=True
  DFINETransformer.use_qgp=False
  DFINETransformer.qgp_layers=[]
  DFINETransformer.use_pgr_roi_reasoning=False
  DFINETransformer.roi_init_ratio=0.5
  DFINETransformer.use_fdi_factor_prior=False
  DFINETransformer.fdi_factor_logit_scale=0.0
  DFINETransformer.use_spegc_prompt_graph=False
  DFINETransformer.spegc_num_prompts=16
  DFINETransformer.spegc_graph_topk=8
  DFINETransformer.spegc_ref_bias=0.25
  DFINETransformer.spegc_res_scale=0.20
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
)

ORDER=(
  abl31_hooa_attention
  abl32_multinex_prior
  abl33_spegc_prompt_graph
  abl34_cmfa_hooa
  abl35_ucm_hooa
  abl36_cmfa_ucm_spegc
  abl37_hooa_multinex
  abl38_cmfa_ucm_hooa_spegc
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

if should_run abl31_hooa_attention; then
  run_exp abl31_hooa_attention \
    HybridEncoder.use_hooa_attention=True
fi
if should_run abl32_multinex_prior; then
  run_exp abl32_multinex_prior \
    HybridEncoder.use_multinex_prior=True
fi
if should_run abl33_spegc_prompt_graph; then
  run_exp abl33_spegc_prompt_graph \
    DFINETransformer.use_spegc_prompt_graph=True
fi
if should_run abl34_cmfa_hooa; then
  run_exp abl34_cmfa_hooa \
    HybridEncoder.use_cmfa_fusion=True \
    HybridEncoder.use_hooa_attention=True
fi
if should_run abl35_ucm_hooa; then
  run_exp abl35_ucm_hooa \
    HybridEncoder.use_ucm_uncertainty=True \
    HybridEncoder.use_hooa_attention=True
fi
if should_run abl36_cmfa_ucm_spegc; then
  run_exp abl36_cmfa_ucm_spegc \
    HybridEncoder.use_cmfa_fusion=True \
    HybridEncoder.use_ucm_uncertainty=True \
    DFINETransformer.use_spegc_prompt_graph=True
fi
if should_run abl37_hooa_multinex; then
  run_exp abl37_hooa_multinex \
    HybridEncoder.use_hooa_attention=True \
    HybridEncoder.use_multinex_prior=True
fi
if should_run abl38_cmfa_ucm_hooa_spegc; then
  run_exp abl38_cmfa_ucm_hooa_spegc \
    HybridEncoder.use_cmfa_fusion=True \
    HybridEncoder.use_ucm_uncertainty=True \
    HybridEncoder.use_hooa_attention=True \
    DFINETransformer.use_spegc_prompt_graph=True
fi

refresh_summary
