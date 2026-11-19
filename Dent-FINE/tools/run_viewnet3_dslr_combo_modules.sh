#!/usr/bin/env bash
# Serial queue for the ten View-Net3 DSLR recipes missing from the original
# cross-dataset combination ablation. CMFA+DualOpt and MPF+DualOpt are already
# complete in the Best4 DSLR queue and are reused in the shared total table.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_viewnet3_dslr_fdi52.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_viewnet3_dslr_combo_modules}"
INHERITED_OUTPUT_DIR="${INHERITED_OUTPUT_DIR:-outputs/ablation_viewnet3_dslr_best4_dualopt_combinations}"
PRIOR_PATH="${PRIOR_PATH:-${ROOT_DIR}/outputs/ablation_viewnet3_dslr_combo_modules/priors/viewnet3_dslr_fdi52_tooth_templates.pt}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${ROOT_DIR}/tools/summarize_viewnet3_dslr_combo_modules.py}"

EPOCHS="${EPOCHS:-160}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# last.pth is overwritten each epoch for safe resume; avoid redundant periodic
# snapshots on the nearly-full disk.
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000000}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
START_FROM="${START_FROM:-viewnet3_spatial_shape_dualopt}"
MAX_RUNS="${MAX_RUNS:-0}"

ORDER=(
  viewnet3_spatial_shape_dualopt
  viewnet3_prototype_dualopt
  viewnet3_cmfa_hooa_dualopt
  viewnet3_cmfa_hooa_prototype
  viewnet3_cmfa_hooa_spatial_shape
  viewnet3_cmfa_hooa_spatial_shape_dualopt
  viewnet3_cmfa_hooa_cedv2
  viewnet3_cmfa_hooa_cedv2_dualopt
  viewnet3_cmfa_hooa_cedv2_spatial_shape
  viewnet3_cmfa_hooa_cedv2_spatial_shape_dualopt
)

CURRENT_EXP_DIR=""
RUN_COUNT=0

summarize() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --base-dir "${ROOT_DIR}/${BASE_OUTPUT_DIR}" \
    --inherited-dir "${ROOT_DIR}/${INHERITED_OUTPUT_DIR}" \
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

# Keep every unrelated branch disabled. The remaining switches are supplied by
# each recipe below, reproducing the definitions in the source combo table.
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
  "HybridEncoder.ced_cascade_stages=1"
  "HybridEncoder.ced_cascade_expansion=0.5"
  "HybridEncoder.ced_cascade_res_init=-1.6"
  "HybridEncoder.use_dswfm=False"
  "HybridEncoder.use_dced_cascade=False"
  "HybridEncoder.use_dug_hoif_fusion=False"
  "HybridEncoder.paper_refine_apply=proj"
  "HybridEncoder.apply_extra_modules=proj"
  "HybridEncoder.apply_denoise_modules=proj"
  "HybridEncoder.apply_refine_modules=outs"
  "DFINETransformer.use_factor_heads=False"
  "DFINETransformer.factor_alpha_quad=0.0"
  "DFINETransformer.factor_alpha_tooth=0.0"
  "DFINETransformer.factor_alpha_dent=0.0"
  "DFINETransformer.factor_fusion_train=False"
  "DFINETransformer.use_tooth_prototype=False"
  "DFINETransformer.prototype_count=4"
  "DFINETransformer.prototype_tau=0.20"
  "DFINETransformer.prototype_fusion_weight=0.0"
  "DFINETransformer.prototype_fusion_train=False"
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
  "DFINETransformer.spegc_num_prompts=16"
  "DFINETransformer.spegc_graph_topk=8"
  "DFINETransformer.spegc_ref_bias=0.25"
  "DFINETransformer.spegc_res_scale=0.20"
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
mpf_args=("HybridEncoder.use_multipath_fusion=True")
cmfa_hooa_args=(
  "HybridEncoder.use_cmfa_fusion=True"
  "HybridEncoder.use_hooa_attention=True"
)
cedv2_args=(
  "HybridEncoder.use_ced_cascade_refiner=True"
  "HybridEncoder.ced_cascade_stages=1"
  "HybridEncoder.ced_cascade_expansion=0.5"
  "HybridEncoder.ced_cascade_res_init=-1.6"
  "HybridEncoder.paper_refine_apply=proj"
)
prototype_args=(
  "DFINETransformer.use_factor_heads=True"
  "DFINETransformer.use_tooth_prototype=True"
  "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent,proto]"
)
spatial_shape_args=(
  "${prototype_args[@]}"
  "PostProcessor.use_dental_spatial_prior=True"
  "PostProcessor.spatial_prior_weight=0.2"
  "PostProcessor.use_dental_shape_prior=True"
  "PostProcessor.shape_prior_weight=0.05"
  "PostProcessor.use_dental_view_prior=True"
  "PostProcessor.view_prior_weight=0.2"
)
dslr_prior_args=(
  "fdi_label_order=fdi52"
  "dental_prior_path=${PRIOR_PATH}"
  "PostProcessor.dental_view_type=unknown"
)

cleanup_completed_weights() {
  local exp_dir="$1"
  if [[ -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/best.pth" ]]; then
    find "${exp_dir}" -type f -name '*.pth' ! -path "${exp_dir}/best.pth" -delete
  fi
}

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
    echo "Refusing to mix a partial log without a resumable last.pth: ${exp_dir}" >&2
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
    -u "${base_args[@]}" "$@"
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
if [[ ! -f "${PRIOR_PATH}" ]]; then
  echo "Missing DSLR-specific spatial/shape prior: ${PRIOR_PATH}" >&2
  exit 1
fi

summarize

run_exp viewnet3_spatial_shape_dualopt \
  "${dslr_prior_args[@]}" "${spatial_shape_args[@]}" "${dualopt_args[@]}"
run_exp viewnet3_prototype_dualopt \
  "${dslr_prior_args[@]}" "${prototype_args[@]}" "${dualopt_args[@]}"
run_exp viewnet3_cmfa_hooa_dualopt \
  "${cmfa_hooa_args[@]}" "${dualopt_args[@]}"
run_exp viewnet3_cmfa_hooa_prototype \
  "${dslr_prior_args[@]}" "${cmfa_hooa_args[@]}" "${prototype_args[@]}"
run_exp viewnet3_cmfa_hooa_spatial_shape \
  "${dslr_prior_args[@]}" "${cmfa_hooa_args[@]}" "${spatial_shape_args[@]}"
run_exp viewnet3_cmfa_hooa_spatial_shape_dualopt \
  "${dslr_prior_args[@]}" "${cmfa_hooa_args[@]}" "${spatial_shape_args[@]}" "${dualopt_args[@]}"
run_exp viewnet3_cmfa_hooa_cedv2 \
  "${cmfa_hooa_args[@]}" "${cedv2_args[@]}"
run_exp viewnet3_cmfa_hooa_cedv2_dualopt \
  "${cmfa_hooa_args[@]}" "${cedv2_args[@]}" "${dualopt_args[@]}"
run_exp viewnet3_cmfa_hooa_cedv2_spatial_shape \
  "${dslr_prior_args[@]}" "${cmfa_hooa_args[@]}" "${cedv2_args[@]}" "${spatial_shape_args[@]}"
run_exp viewnet3_cmfa_hooa_cedv2_spatial_shape_dualopt \
  "${dslr_prior_args[@]}" "${cmfa_hooa_args[@]}" "${cedv2_args[@]}" "${spatial_shape_args[@]}" "${dualopt_args[@]}"

echo "All missing View-Net3 DSLR combo-module recipes completed."
