#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_all_new_modules}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
START_FROM="${START_FROM:-abl39_nwd_loss}"
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
  DFINETransformer.use_fdi_spatial_prior=False
  DFINETransformer.fdi_prior_knn=8
  DFINETransformer.fdi_prior_sigma=0.12
  DFINETransformer.fdi_prior_jaw_scale=0.08
  DFINETransformer.fdi_prior_neighbor_scale=0.12
  DFINETransformer.fdi_prior_unique_scale=0.04
  DFINETransformer.fdi_prior_dent_scale=0.04
  DFINETransformer.use_spegc_prompt_graph=False
  DFINETransformer.spegc_num_prompts=16
  DFINETransformer.spegc_graph_topk=8
  DFINETransformer.spegc_ref_bias=0.25
  DFINETransformer.spegc_res_scale=0.20
  DEIMCriterion.use_fdi_anatomy_loss=False
  DEIMCriterion.nwd_normalizer=0.08
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
)

NWD_ARGS=(
  DEIMCriterion.losses=[mal,boxes,local,nwd]
  DEIMCriterion.weight_dict.loss_nwd=0.50
  DEIMCriterion.nwd_normalizer=0.08
)

ANATOMY_ARGS=(
  DEIMCriterion.use_fdi_anatomy_loss=True
  DEIMCriterion.weight_dict.loss_fdi_factor=0.12
  DEIMCriterion.weight_dict.loss_fdi_unique=0.02
  DEIMCriterion.weight_dict.loss_fdi_dent=0.03
)

BEST_ENCODER_ARGS=(
  HybridEncoder.use_cmfa_fusion=True
  HybridEncoder.use_hooa_attention=True
)

ORDER=(
  abl39_nwd_loss
  abl40_fdi_spatial_prior
  abl41_fdi_anatomy_loss
  abl42_cmfa_hooa_nwd
  abl43_cmfa_hooa_fdi_prior
  abl44_cmfa_hooa_anatomy_loss
  abl45_cmfa_hooa_nwd_fdi_prior
  abl46_cmfa_hooa_all_priors
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

if should_run abl39_nwd_loss; then
  run_exp abl39_nwd_loss \
    "${NWD_ARGS[@]}"
fi
if should_run abl40_fdi_spatial_prior; then
  run_exp abl40_fdi_spatial_prior \
    DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl41_fdi_anatomy_loss; then
  run_exp abl41_fdi_anatomy_loss \
    "${ANATOMY_ARGS[@]}"
fi
if should_run abl42_cmfa_hooa_nwd; then
  run_exp abl42_cmfa_hooa_nwd \
    "${BEST_ENCODER_ARGS[@]}" \
    "${NWD_ARGS[@]}"
fi
if should_run abl43_cmfa_hooa_fdi_prior; then
  run_exp abl43_cmfa_hooa_fdi_prior \
    "${BEST_ENCODER_ARGS[@]}" \
    DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl44_cmfa_hooa_anatomy_loss; then
  run_exp abl44_cmfa_hooa_anatomy_loss \
    "${BEST_ENCODER_ARGS[@]}" \
    "${ANATOMY_ARGS[@]}"
fi
if should_run abl45_cmfa_hooa_nwd_fdi_prior; then
  run_exp abl45_cmfa_hooa_nwd_fdi_prior \
    "${BEST_ENCODER_ARGS[@]}" \
    "${NWD_ARGS[@]}" \
    DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl46_cmfa_hooa_all_priors; then
  run_exp abl46_cmfa_hooa_all_priors \
    "${BEST_ENCODER_ARGS[@]}" \
    "${NWD_ARGS[@]}" \
    "${ANATOMY_ARGS[@]}" \
    DFINETransformer.use_fdi_spatial_prior=True
fi

refresh_summary
