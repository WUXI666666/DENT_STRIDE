#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_WEIGHT="${BASE_WEIGHT:-${ROOT_DIR}/outputs/ablation_all_new_modules/abl34_cmfa_hooa/best_stg1.pth}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_coco_cmfa_hooa_optimized_followup}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-32}"
LR="${LR:-0.00002}"
BACKBONE_LR="${BACKBONE_LR:-0.000005}"
WARMUP_ITER="${WARMUP_ITER:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
START_FROM="${START_FROM:-ft_control_lowlr}"
MAX_RUNS="${MAX_RUNS:-0}"
INCLUDE_DUG_REPLACEMENT="${INCLUDE_DUG_REPLACEMENT:-0}"

ORDER=(
  ft_control_lowlr
  cedv2_lowlr
  dswfm_proj_lowlr
  dswfm_outs_lowlr
  dualopt_lowlr
  cedv2_dualopt_lowlr
)

if [[ "${INCLUDE_DUG_REPLACEMENT}" == "1" ]]; then
  ORDER+=(
    dug_interleave_replacement
    dug_full_replacement
  )
fi

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
  shift

  if ! should_run "${name}"; then
    return
  fi

  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  local update_args=(
    "epoches=${EPOCHS}"
    "warmup_iter=${WARMUP_ITER}"
    "lr_gamma=1.0"
    "flat_epoch=${EPOCHS}"
    "no_aug_epoch=0"
    "optimizer.lr=${LR}"
    "optimizer.params=[{params: '^(?=.*backbone)(?!.*(?:norm|bn)).*$', lr: ${BACKBONE_LR}}, {params: '^(?=.*(?:norm|bn)).*$', lr: ${LR}, weight_decay: 0.0}]"
    "HybridEncoder.use_hfsa=False"
    "HybridEncoder.use_wav_enhance=False"
    "HybridEncoder.use_stat_denoise=False"
    "HybridEncoder.use_multipath_fusion=False"
    "HybridEncoder.use_dynamic_rf=False"
    "HybridEncoder.use_mpc_multiscale_perception=False"
    "HybridEncoder.use_pqg_high_frequency=False"
    "HybridEncoder.use_cmfa_fusion=True"
    "HybridEncoder.use_pfesa=False"
    "HybridEncoder.use_ucan_lkd=False"
    "HybridEncoder.use_ucm_uncertainty=False"
    "HybridEncoder.use_retinex_prior=False"
    "HybridEncoder.use_hooa_attention=True"
    "HybridEncoder.use_multinex_prior=False"
    "HybridEncoder.use_msck_refiner=False"
    "HybridEncoder.use_csaf_filter=False"
    "HybridEncoder.use_csaf_dmssp=False"
    "HybridEncoder.use_rdvq_refiner=False"
    "HybridEncoder.use_datpp_refiner=False"
    "HybridEncoder.use_cdml_domain_adapter=False"
    "HybridEncoder.use_efruf_gradient_fusion=False"
    "HybridEncoder.use_sinf_spatial_field=False"
    "HybridEncoder.use_smwg_spectral_wavelet=False"
    "HybridEncoder.use_structural_entropy_frequency_refiner=False"
    "HybridEncoder.use_dug_hoif_fusion=False"
    "HybridEncoder.use_dswfm=False"
    "HybridEncoder.use_dced_cascade=False"
    "HybridEncoder.use_ced_cascade_refiner=False"
    "HybridEncoder.paper_refine_apply=none"
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
    "train_dataloader.dataset.transforms.policy.epoch=[100,101,102]"
    "train_dataloader.collate_fn.mixup_prob=0.0"
    "train_dataloader.collate_fn.mixup_epochs=[100,101]"
    "train_dataloader.collate_fn.stop_epoch=100"
    "train_dataloader.collate_fn.base_size_repeat=null"
  )
  update_args+=("$@")

  echo "=============================="
  echo "Running ${name}"
  echo "Config: ${CONFIG}"
  echo "Tuning from: ${BASE_WEIGHT}"
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
    -c "${CONFIG}"
    --use-amp
    --seed="${SEED}"
    -d cuda
    -t "${BASE_WEIGHT}"
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

cedv2_args=(
  "HybridEncoder.use_ced_cascade_refiner=True"
  "HybridEncoder.ced_cascade_stages=1"
  "HybridEncoder.ced_cascade_expansion=0.5"
  "HybridEncoder.ced_cascade_res_init=-1.6"
)

dswfm_proj_args=(
  "HybridEncoder.use_dswfm=True"
  "HybridEncoder.dswfm_use_dynamic_fourier=True"
  "HybridEncoder.dswfm_use_haar_detail=True"
  "HybridEncoder.dswfm_use_bidirectional=False"
  "HybridEncoder.dswfm_res_init=-3.0"
  "HybridEncoder.paper_refine_apply=proj"
)

dswfm_outs_args=(
  "HybridEncoder.use_dswfm=True"
  "HybridEncoder.dswfm_use_dynamic_fourier=True"
  "HybridEncoder.dswfm_use_haar_detail=True"
  "HybridEncoder.dswfm_use_bidirectional=False"
  "HybridEncoder.dswfm_res_init=-3.0"
  "HybridEncoder.paper_refine_apply=outs"
)

dualopt_args=(
  "optimizer.type=DualOptAdamW"
  "optimizer.rollback_weight=0.02"
  "optimizer.realtime_decay=0.02"
  "optimizer.layer_decay_power=1.5"
  "optimizer.layer_decay_min=0.5"
  "optimizer.layer_decay_max=1.5"
)

dug_interleave_replacement_args=(
  "HybridEncoder.use_dug_hoif_fusion=True"
  "HybridEncoder.dug_hoif_use_interleave=True"
  "HybridEncoder.dug_hoif_use_multibranch=False"
  "HybridEncoder.dug_hoif_res_init=-3.0"
)

dug_full_replacement_args=(
  "HybridEncoder.use_dug_hoif_fusion=True"
  "HybridEncoder.dug_hoif_use_interleave=True"
  "HybridEncoder.dug_hoif_use_multibranch=True"
  "HybridEncoder.dug_hoif_use_uncertainty_routing=True"
  "HybridEncoder.dug_hoif_use_high_order=True"
  "HybridEncoder.dug_hoif_use_omni_gate=True"
  "HybridEncoder.dug_hoif_res_init=-3.0"
)

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

run_exp ft_control_lowlr
run_exp cedv2_lowlr "${cedv2_args[@]}"
run_exp dswfm_proj_lowlr "${dswfm_proj_args[@]}"
run_exp dswfm_outs_lowlr "${dswfm_outs_args[@]}"
run_exp dualopt_lowlr "${dualopt_args[@]}"
run_exp cedv2_dualopt_lowlr "${cedv2_args[@]}" "${dualopt_args[@]}"

if [[ "${INCLUDE_DUG_REPLACEMENT}" == "1" ]]; then
  echo "WARNING: DUG replacement experiments rebuild the CMFA FPN/PAN fusion shape and are kept optional."
  run_exp dug_interleave_replacement "${dug_interleave_replacement_args[@]}"
  run_exp dug_full_replacement "${dug_full_replacement_args[@]}"
fi

echo "Finished. Outputs are under ${ROOT_DIR}/${BASE_OUTPUT_DIR}"
