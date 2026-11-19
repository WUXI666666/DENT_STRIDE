#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_all_new_modules}"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_ablation_all_new_modules.py"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
START_FROM="${START_FROM:-abl154_coco_rot_equivariant_detail}"
MAX_RUNS="${MAX_RUNS:-0}"

ORDER=(
  abl154_coco_rot_equivariant_detail
  abl155_coco_dualopt_adamw
  abl156_coco_hedp_prompt
  abl157_coco_sampg_sst
  abl158_coco_segd_entropy
  abl159_coco_fsdetr_frequency_spatial
  abl160_coco_df2net_dynamic_fusion
  abl161_coco_sdconet_saliency
  abl162_coco_idgbr_boundary
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
    exit 1
  fi
  [[ "${cur_idx}" -ge "${start_idx}" ]]
}

refresh_summary() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --base-output-dir "${ROOT_DIR}/${BASE_OUTPUT_DIR}"
}

RUN_COUNT=0

run_exp() {
  local name="$1"
  shift
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
    "HybridEncoder.paper_refine_apply=proj"
    "HybridEncoder.apply_extra_modules=proj"
    "HybridEncoder.apply_denoise_modules=proj"
    "HybridEncoder.apply_refine_modules=outs"
    "DFINETransformer.use_factor_heads=False"
    "DFINETransformer.use_tooth_prototype=False"
    "DFINETransformer.use_uni_gate=False"
    "DFINETransformer.use_uni_gate_infer=False"
    "DFINETransformer.use_qgp=False"
    "DFINETransformer.qgp_layers=[]"
    "DFINETransformer.use_pgr_roi_reasoning=False"
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
    "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
    "train_dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    update_args+=(
      "epoches=${EPOCHS}"
      "flat_epoch=0"
      "no_aug_epoch=0"
      "train_dataloader.dataset.transforms.policy.epoch=[100,101,102]"
      "train_dataloader.collate_fn.mixup_epochs=[100,101]"
      "train_dataloader.collate_fn.stop_epoch=100"
    )
  fi
  update_args+=("$@")

  echo "=============================="
  echo "Running ${name}"
  echo "Config: ${CONFIG}"
  echo "Output: ${exp_output}"
  echo "=============================="

  if [[ "${SKIP_EXISTING}" == "1" && -f "${exp_output}/log.txt" ]]; then
    echo "Skip existing ${name}"
    refresh_summary
    return
  fi
  if [[ "${CLEAN_OUTPUT}" == "1" && -d "${exp_output}" ]]; then
    rm -rf "${exp_output}"
  fi

  local cmd=(
    "${PYTHON_BIN}" train.py
    -c "${CONFIG}"
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

  refresh_summary
  RUN_COUNT=$((RUN_COUNT + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}; stop after ${name}."
    echo "Summary refreshed at ${ROOT_DIR}/${BASE_OUTPUT_DIR}/ablation_summary.md"
    exit 0
  fi
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"
refresh_summary

run_exp abl154_coco_rot_equivariant_detail \
  "HybridEncoder.use_rot_equivariant_detail_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl155_coco_dualopt_adamw \
  "optimizer.type=DualOptAdamW" \
  "optimizer.rollback_weight=0.005" \
  "optimizer.realtime_decay=0.01" \
  "optimizer.layer_decay_power=1.5" \
  "optimizer.layer_decay_min=0.5" \
  "optimizer.layer_decay_max=1.5"

run_exp abl156_coco_hedp_prompt \
  "HybridEncoder.use_hedp_prompt_refiner=True" \
  "HybridEncoder.hedp_prompt_count=4" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl157_coco_sampg_sst \
  "HybridEncoder.use_sampg_sst_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl158_coco_segd_entropy \
  "HybridEncoder.use_segd_entropy_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl159_coco_fsdetr_frequency_spatial \
  "HybridEncoder.use_fsdetr_frequency_spatial_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl160_coco_df2net_dynamic_fusion \
  "HybridEncoder.use_df2net_dynamic_fusion=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl161_coco_sdconet_saliency \
  "HybridEncoder.use_sdconet_saliency_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

run_exp abl162_coco_idgbr_boundary \
  "HybridEncoder.use_idgbr_boundary_refiner=True" \
  "HybridEncoder.paper_refine_apply=proj"

refresh_summary
echo "Summary refreshed at ${ROOT_DIR}/${BASE_OUTPUT_DIR}/ablation_summary.md"
