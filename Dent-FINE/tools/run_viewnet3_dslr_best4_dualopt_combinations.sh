#!/usr/bin/env bash
# Serial ablation queue for the View-Net3 DSLR FDI52 tooth-detection cohort.
# Every run uses DualOptAdamW; CMFA, SMWG, UCM, and MPF are varied over all
# 2^4 subsets, including the DualOpt-only control.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_viewnet3_dslr_fdi52.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_viewnet3_dslr_best4_dualopt_combinations}"
EPOCHS="${EPOCHS:-160}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# The solver continuously maintains last.pth for resume. Disable its additional
# every-N-epoch snapshots because they are redundant for this serial queue.
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000000}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
START_FROM="${START_FROM:-viewnet3_dualopt_only}"
MAX_RUNS="${MAX_RUNS:-0}"

PREPARE_SCRIPT="${ROOT_DIR}/tools/dental/prepare_viewnet3_dslr_fdi52.py"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_viewnet3_dslr_best4_dualopt_combinations.py"
CURRENT_EXP_DIR=""

ORDER=(
  viewnet3_dualopt_only
  viewnet3_cmfa_dualopt
  viewnet3_smwg_dualopt
  viewnet3_ucm_dualopt
  viewnet3_mpf_dualopt
  viewnet3_cmfa_smwg_dualopt
  viewnet3_cmfa_ucm_dualopt
  viewnet3_cmfa_mpf_dualopt
  viewnet3_smwg_ucm_dualopt
  viewnet3_smwg_mpf_dualopt
  viewnet3_ucm_mpf_dualopt
  viewnet3_cmfa_smwg_ucm_dualopt
  viewnet3_cmfa_smwg_mpf_dualopt
  viewnet3_cmfa_ucm_mpf_dualopt
  viewnet3_smwg_ucm_mpf_dualopt
  viewnet3_cmfa_smwg_ucm_mpf_dualopt
)

summarize() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --base-dir "${ROOT_DIR}/${BASE_OUTPUT_DIR}" \
    --expected-epochs "${EPOCHS}"
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
    echo "Invalid START_FROM=${START_FROM}" >&2
    echo "Valid experiments: ${ORDER[*]}" >&2
    return 2
  fi
  [[ "${current_index}" -ge "${start_index}" ]]
}

# Keep all switches other than the four requested modules off. This mirrors the
# established Best4 setup while eliminating prior/decoder confounders.
base_args=(
  "HybridEncoder.use_hfsa=False"
  "HybridEncoder.use_wav_enhance=False"
  "HybridEncoder.use_stat_denoise=False"
  "HybridEncoder.use_multipath_fusion=False"
  "HybridEncoder.use_dynamic_rf=False"
  "HybridEncoder.use_mpc_multiscale_perception=False"
  "HybridEncoder.use_pqg_high_frequency=False"
  "HybridEncoder.use_cmfa_fusion=False"
  "HybridEncoder.use_pfesa=False"
  "HybridEncoder.use_ucan_lkd=False"
  "HybridEncoder.use_ucm_uncertainty=False"
  "HybridEncoder.use_retinex_prior=False"
  "HybridEncoder.use_hooa_attention=False"
  "HybridEncoder.use_multinex_prior=False"
  "HybridEncoder.use_dcsam_dual_consistency=False"
  "HybridEncoder.use_iqp_quality_prior=False"
  "HybridEncoder.use_dformer_geometry_prior=False"
  "HybridEncoder.use_cdml_domain_adapter=False"
  "HybridEncoder.use_efruf_gradient_fusion=False"
  "HybridEncoder.use_sinf_spatial_field=False"
  "HybridEncoder.use_smwg_spectral_wavelet=False"
  "HybridEncoder.use_structural_entropy_frequency_refiner=False"
  "HybridEncoder.use_rot_equivariant_detail_refiner=False"
  "HybridEncoder.use_hedp_prompt_refiner=False"
  "HybridEncoder.use_sampg_sst_refiner=False"
  "HybridEncoder.use_segd_entropy_refiner=False"
  "HybridEncoder.use_fsdetr_frequency_spatial_refiner=False"
  "HybridEncoder.use_df2net_dynamic_fusion=False"
  "HybridEncoder.use_sdconet_saliency_refiner=False"
  "HybridEncoder.use_idgbr_boundary_refiner=False"
  "HybridEncoder.use_bilalora_layer_adapter=False"
  "HybridEncoder.use_surmt_structure_bridge=False"
  "HybridEncoder.use_mav_sparse_interaction=False"
  "HybridEncoder.use_cnm_ode_refiner=False"
  "HybridEncoder.use_cldyn_rsc=False"
  "HybridEncoder.use_ced_cascade_refiner=False"
  "HybridEncoder.use_dswfm=False"
  "HybridEncoder.use_dced_cascade=False"
  "HybridEncoder.use_dug_hoif_fusion=False"
  "HybridEncoder.paper_refine_apply=proj"
  "HybridEncoder.apply_extra_modules=proj"
  "HybridEncoder.apply_denoise_modules=proj"
  "HybridEncoder.apply_refine_modules=outs"
  "DFINETransformer.use_factor_heads=False"
  "DFINETransformer.use_tooth_prototype=False"
  "DFINETransformer.use_uni_gate=False"
  "DFINETransformer.use_uni_gate_infer=False"
  "DFINETransformer.uni_gate_apply_last_only=True"
  "DFINETransformer.use_qgp=False"
  "DFINETransformer.qgp_layers=[]"
  "DFINETransformer.use_pgr_roi_reasoning=False"
  "DFINETransformer.roi_init_ratio=0.5"
  "DFINETransformer.use_fdi_factor_prior=False"
  "DFINETransformer.fdi_factor_logit_scale=0.0"
  "DFINETransformer.use_spegc_prompt_graph=False"
  "DFINETransformer.use_cluster_query_refiner=False"
  "DFINETransformer.use_query_recovery_memory=False"
  "DFINETransformer.use_pfrpn_query_adapter=False"
  "DFINETransformer.use_tooth_template_prior=False"
  "DFINETransformer.use_anatomical_region_prior=False"
  "PostProcessor.use_dental_spatial_prior=False"
  "PostProcessor.use_dental_shape_prior=False"
  "PostProcessor.use_dental_view_prior=False"
  "PostProcessor.use_global_assignment=False"
  "DEIMCriterion.losses=[mal,boxes,local]"
  "DEIMCriterion.use_fdi_anatomy_loss=False"
  "DEIMCriterion.use_dsfed_dual_scale_kd=False"
  "DEIMCriterion.use_tooth_size_prior_loss=False"
  "checkpoint_freq=${CHECKPOINT_FREQ}"
  "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
  "val_dataloader.num_workers=${NUM_WORKERS}"
  "epoches=${EPOCHS}"
)

dualopt_args=(
  "optimizer.type=DualOptAdamW"
  "optimizer.rollback_weight=0.005"
  "optimizer.realtime_decay=0.01"
  "optimizer.layer_decay_power=1.5"
  "optimizer.layer_decay_min=0.5"
  "optimizer.layer_decay_max=1.5"
)
cmfa_args=("HybridEncoder.use_cmfa_fusion=True")
smwg_args=("HybridEncoder.use_smwg_spectral_wavelet=True" "HybridEncoder.paper_refine_apply=proj")
ucm_args=("HybridEncoder.use_ucm_uncertainty=True")
mpf_args=("HybridEncoder.use_multipath_fusion=True")

cleanup_completed_weights() {
  local exp_dir="$1"
  # Metrics stay in log.txt and the total table. Once a run has completed,
  # only the canonical best checkpoint is useful; delete only files inside the
  # experiment's own output directory.
  if [[ -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/best.pth" ]]; then
    find "${exp_dir}" -type f -name '*.pth' ! -path "${exp_dir}/best.pth" -delete
  fi
}

RUN_COUNT=0
run_exp() {
  local name="$1"
  shift
  if ! should_run "${name}"; then
    return
  fi

  local exp_dir="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${exp_dir}/COMPLETED" ]]; then
    echo "[skip] ${name} already completed"
    return
  fi
  if [[ -f "${exp_dir}/log.txt" && ! -f "${exp_dir}/last.pth" && ! -f "${exp_dir}/COMPLETED" ]]; then
    echo "Refusing to mix logs for partial run without a resumable last.pth: ${exp_dir}" >&2
    return 1
  fi

  mkdir -p "${exp_dir}"
  rm -f "${exp_dir}/FAILED" "${exp_dir}/COMPLETED"
  touch "${exp_dir}/RUNNING"
  CURRENT_EXP_DIR="${exp_dir}"

  local command=(
    "${PYTHON_BIN}" train.py
    -c "${CONFIG}"
    --use-amp
    --seed=0
    -d cuda
  )
  if [[ -f "${exp_dir}/last.pth" ]]; then
    command+=(--resume "${exp_dir}/last.pth")
  fi
  command+=(
    -u "${base_args[@]}" "${dualopt_args[@]}" "$@"
    --output-dir "${exp_dir}"
  )

  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: ${name}"
  echo "Output: ${exp_dir}"
  printf 'Command: '
  printf '%q ' "${command[@]}"
  printf '\n'
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${command[@]}"

  rm -f "${exp_dir}/RUNNING" "${exp_dir}/FAILED"
  touch "${exp_dir}/COMPLETED"
  cleanup_completed_weights "${exp_dir}"
  CURRENT_EXP_DIR=""
  summarize

  RUN_COUNT=$((RUN_COUNT + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}; queue paused after ${name}."
    exit 0
  fi
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON_BIN}" "${PREPARE_SCRIPT}" --check-images
fi

summarize

run_exp viewnet3_dualopt_only
run_exp viewnet3_cmfa_dualopt "${cmfa_args[@]}"
run_exp viewnet3_smwg_dualopt "${smwg_args[@]}"
run_exp viewnet3_ucm_dualopt "${ucm_args[@]}"
run_exp viewnet3_mpf_dualopt "${mpf_args[@]}"
run_exp viewnet3_cmfa_smwg_dualopt "${cmfa_args[@]}" "${smwg_args[@]}"
run_exp viewnet3_cmfa_ucm_dualopt "${cmfa_args[@]}" "${ucm_args[@]}"
run_exp viewnet3_cmfa_mpf_dualopt "${cmfa_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_smwg_ucm_dualopt "${smwg_args[@]}" "${ucm_args[@]}"
run_exp viewnet3_smwg_mpf_dualopt "${smwg_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_ucm_mpf_dualopt "${ucm_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_cmfa_smwg_ucm_dualopt "${cmfa_args[@]}" "${smwg_args[@]}" "${ucm_args[@]}"
run_exp viewnet3_cmfa_smwg_mpf_dualopt "${cmfa_args[@]}" "${smwg_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_cmfa_ucm_mpf_dualopt "${cmfa_args[@]}" "${ucm_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_smwg_ucm_mpf_dualopt "${smwg_args[@]}" "${ucm_args[@]}" "${mpf_args[@]}"
run_exp viewnet3_cmfa_smwg_ucm_mpf_dualopt "${cmfa_args[@]}" "${smwg_args[@]}" "${ucm_args[@]}" "${mpf_args[@]}"

echo "All View-Net3 DSLR Best4 + DualOpt combination runs completed."
