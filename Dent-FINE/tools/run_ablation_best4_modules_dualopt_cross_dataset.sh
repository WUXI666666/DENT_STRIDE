#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"

COCO_CONFIG="${COCO_CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
DENTEX_CONFIG="${DENTEX_CONFIG:-configs/deim_dfine/deim_hgnetv2_s_dentex.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_best4_modules_dualopt_cross_dataset}"
COCO_PRIOR="${COCO_PRIOR:-${ROOT_DIR}/outputs/ablation_all_new_modules/priors/phone_tooth_templates.pt}"
DENTEX_PRIOR="${DENTEX_PRIOR:-${ROOT_DIR}/outputs/ablation_all_new_modules/priors/dentex_tooth_templates.pt}"

KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_RUNS="${MAX_RUNS:-0}"
INCLUDE_DUALOPT_ONLY="${INCLUDE_DUALOPT_ONLY:-1}"
START_FROM="${START_FROM:-dentex_best4_cmfa_smwg_ucm_mpf_dualopt}"

# Picked from outputs/ablation_all_new_modules/ablation_summary.md.
# Excluding DualOpt because it is an optimizer strategy, the four strongest
# individual modules with positive AP delta are:
#   abl08 CMFA                 +0.0049
#   abl87 SMWG-SpectralWavelet +0.0024
#   abl20 UCM-Uncertainty      +0.0021
#   abl04 MPF                  +0.0020

ORDER=(
  dentex_best4_cmfa_smwg_ucm_mpf_dualopt
  dentex_dualopt_only
  dentex_cmfa_hooa
  dentex_single_prototype
  dentex_single_spatial_shape
  dentex_single_cmfa
  dentex_single_smwg
  dentex_single_ucm
  dentex_single_mpf
  coco_best4_cmfa_smwg_ucm_mpf_dualopt
  coco_dualopt_only
  coco_cmfa_hooa
  coco_single_prototype
  coco_single_spatial_shape
  coco_single_cmfa
  coco_single_smwg
  coco_single_ucm
  coco_single_mpf
)

should_run() {
  local name="$1"
  local start_idx=-1
  local cur_idx=-1
  local i
  for i in "${!ORDER[@]}"; do
    [[ "${ORDER[$i]}" == "${START_FROM}" ]] && start_idx="$i"
    [[ "${ORDER[$i]}" == "${name}" ]] && cur_idx="$i"
  done
  if [[ "${start_idx}" -lt 0 || "${cur_idx}" -lt 0 ]]; then
    echo "Invalid START_FROM=${START_FROM}" >&2
    echo "Valid experiments: ${ORDER[*]}" >&2
    exit 1
  fi
  [[ "${cur_idx}" -ge "${start_idx}" ]]
}

RUN_COUNT=0

run_exp() {
  local name="$1"
  local config="$2"
  shift 2

  if [[ "${INCLUDE_DUALOPT_ONLY}" != "1" && "${name}" == *"_dualopt_only" ]]; then
    return
  fi
  if ! should_run "${name}"; then
    return
  fi

  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  local update_args=(
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
    "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
    "train_dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    update_args+=("epoches=${EPOCHS}")
  fi
  update_args+=("$@")

  echo "=============================="
  echo "Running ${name}"
  echo "Config: ${config}"
  echo "Output: ${exp_output}"
  echo "=============================="

  if [[ "${SKIP_EXISTING}" == "1" && -f "${exp_output}/log.txt" ]]; then
    echo "Skip existing ${name}"
    return
  fi
  if [[ "${CLEAN_OUTPUT}" == "1" && -d "${exp_output}" ]]; then
    rm -rf "${exp_output}"
  fi

  local cmd=(
    "${PYTHON_BIN}" train.py
    -c "${config}"
    --use-amp
    --seed=0
    -d cuda
    -u "${update_args[@]}"
    --output-dir "${exp_output}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
    if [[ "${KEEP_ONLY_BEST}" == "1" && -d "${exp_output}" ]]; then
      find "${exp_output}" -type f -name '*.pth' \
        ! -name 'best_stg1.pth' \
        ! -name 'best_stg2.pth' \
        ! -name 'best.pth' \
        -delete
    fi
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}; stop after ${name}."
    exit 0
  fi
}

cmfa_args=(
  "HybridEncoder.use_cmfa_fusion=True"
)

cmfa_hooa_args=(
  "HybridEncoder.use_cmfa_fusion=True"
  "HybridEncoder.use_hooa_attention=True"
)

smwg_args=(
  "HybridEncoder.use_smwg_spectral_wavelet=True"
  "HybridEncoder.paper_refine_apply=proj"
)

ucm_args=(
  "HybridEncoder.use_ucm_uncertainty=True"
)

mpf_args=(
  "HybridEncoder.use_multipath_fusion=True"
)

dualopt_args=(
  "optimizer.type=DualOptAdamW"
  "optimizer.rollback_weight=0.005"
  "optimizer.realtime_decay=0.01"
  "optimizer.layer_decay_power=1.5"
  "optimizer.layer_decay_min=0.5"
  "optimizer.layer_decay_max=1.5"
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

dentex_prior_args=(
  "fdi_label_order=dentex_enumeration32"
  "dental_prior_path=${DENTEX_PRIOR}"
  "PostProcessor.dental_view_type=panorama"
)

coco_prior_args=(
  "fdi_label_order=fdi52"
  "dental_prior_path=${COCO_PRIOR}"
  "PostProcessor.dental_view_type=unknown"
)

best4_args=(
  "${cmfa_args[@]}"
  "${smwg_args[@]}"
  "${ucm_args[@]}"
  "${mpf_args[@]}"
)

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  for prior_path in "${COCO_PRIOR}" "${DENTEX_PRIOR}"; do
    if [[ ! -f "${prior_path}" ]]; then
      echo "Missing prior file: ${prior_path}" >&2
      echo "Set COCO_PRIOR/DENTEX_PRIOR or build priors before running." >&2
      exit 1
    fi
  done
fi

run_exp dentex_best4_cmfa_smwg_ucm_mpf_dualopt "${DENTEX_CONFIG}" \
  "${best4_args[@]}" "${dualopt_args[@]}"
run_exp dentex_dualopt_only "${DENTEX_CONFIG}" \
  "${dualopt_args[@]}"
run_exp dentex_cmfa_hooa "${DENTEX_CONFIG}" \
  "${cmfa_hooa_args[@]}"
run_exp dentex_single_prototype "${DENTEX_CONFIG}" \
  "${dentex_prior_args[@]}" "${prototype_args[@]}"
run_exp dentex_single_spatial_shape "${DENTEX_CONFIG}" \
  "${dentex_prior_args[@]}" "${spatial_shape_args[@]}"
run_exp dentex_single_cmfa "${DENTEX_CONFIG}" \
  "${cmfa_args[@]}"
run_exp dentex_single_smwg "${DENTEX_CONFIG}" \
  "${smwg_args[@]}"
run_exp dentex_single_ucm "${DENTEX_CONFIG}" \
  "${ucm_args[@]}"
run_exp dentex_single_mpf "${DENTEX_CONFIG}" \
  "${mpf_args[@]}"

run_exp coco_best4_cmfa_smwg_ucm_mpf_dualopt "${COCO_CONFIG}" \
  "${best4_args[@]}" "${dualopt_args[@]}"
run_exp coco_dualopt_only "${COCO_CONFIG}" \
  "${dualopt_args[@]}"
run_exp coco_cmfa_hooa "${COCO_CONFIG}" \
  "${cmfa_hooa_args[@]}"
run_exp coco_single_prototype "${COCO_CONFIG}" \
  "${coco_prior_args[@]}" "${prototype_args[@]}"
run_exp coco_single_spatial_shape "${COCO_CONFIG}" \
  "${coco_prior_args[@]}" "${spatial_shape_args[@]}"
run_exp coco_single_cmfa "${COCO_CONFIG}" \
  "${cmfa_args[@]}"
run_exp coco_single_smwg "${COCO_CONFIG}" \
  "${smwg_args[@]}"
run_exp coco_single_ucm "${COCO_CONFIG}" \
  "${ucm_args[@]}"
run_exp coco_single_mpf "${COCO_CONFIG}" \
  "${mpf_args[@]}"

echo "Finished. Outputs are under ${ROOT_DIR}/${BASE_OUTPUT_DIR}"
