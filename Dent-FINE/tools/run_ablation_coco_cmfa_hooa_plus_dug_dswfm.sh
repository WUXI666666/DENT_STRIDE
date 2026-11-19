#!/usr/bin/env bash
set -euo pipefail

# Historical follow-up for checking DUG/D-SWFM on top of the CMFA+HOOA
# checkpoint. Important: HybridEncoder.use_dug_hoif_fusion=True replaces the
# CMFA FPN/PAN fusion path and changes fpn/pan block input shapes, so these DUG
# rows are not additive CMFA+HOOA+DUG tests. Prefer
# tools/run_ablation_coco_cmfa_hooa_optimized_followup.sh for the corrected
# low-risk follow-up plan.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_WEIGHT="${BASE_WEIGHT:-${ROOT_DIR}/outputs/ablation_all_new_modules/abl34_cmfa_hooa/best_stg1.pth}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_coco_cmfa_hooa_plus_dug_dswfm}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-160}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
START_FROM="${START_FROM:-ft_control}"
MAX_RUNS="${MAX_RUNS:-0}"

ORDER=(
  ft_control
  dug_interleave
  dug_high_order
  dug_full
  dswfm
  dug_interleave_dswfm
  dug_high_order_dswfm
  dug_full_dswfm
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
    "HybridEncoder.use_efruf_gradient_fusion=False"
    "HybridEncoder.use_sinf_spatial_field=False"
    "HybridEncoder.use_smwg_spectral_wavelet=False"
    "HybridEncoder.use_dswfm=False"
    "HybridEncoder.use_dug_hoif_fusion=False"
    "HybridEncoder.dug_hoif_use_interleave=True"
    "HybridEncoder.dug_hoif_use_multibranch=True"
    "HybridEncoder.dug_hoif_use_uncertainty_routing=True"
    "HybridEncoder.dug_hoif_use_high_order=True"
    "HybridEncoder.dug_hoif_use_omni_gate=True"
    "HybridEncoder.dug_hoif_res_init=-2.0"
    "HybridEncoder.use_dced_cascade=False"
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
    "PostProcessor.use_dental_spatial_prior=False"
    "PostProcessor.use_dental_shape_prior=False"
    "PostProcessor.use_dental_view_prior=False"
    "PostProcessor.use_global_assignment=False"
    "DEIMCriterion.losses=[mal,boxes,local]"
    "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
    "train_dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
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

dug_interleave_args=(
  "HybridEncoder.use_dug_hoif_fusion=True"
  "HybridEncoder.dug_hoif_use_multibranch=False"
)

dug_high_order_args=(
  "HybridEncoder.use_dug_hoif_fusion=True"
  "HybridEncoder.dug_hoif_use_interleave=True"
  "HybridEncoder.dug_hoif_use_multibranch=True"
  "HybridEncoder.dug_hoif_use_uncertainty_routing=True"
  "HybridEncoder.dug_hoif_use_high_order=True"
  "HybridEncoder.dug_hoif_use_omni_gate=False"
)

dug_full_args=(
  "HybridEncoder.use_dug_hoif_fusion=True"
  "HybridEncoder.dug_hoif_use_interleave=True"
  "HybridEncoder.dug_hoif_use_multibranch=True"
  "HybridEncoder.dug_hoif_use_uncertainty_routing=True"
  "HybridEncoder.dug_hoif_use_high_order=True"
  "HybridEncoder.dug_hoif_use_omni_gate=True"
)

dswfm_args=(
  "HybridEncoder.use_dswfm=True"
  "HybridEncoder.dswfm_use_dynamic_fourier=True"
  "HybridEncoder.dswfm_use_haar_detail=True"
  "HybridEncoder.dswfm_use_bidirectional=True"
  "HybridEncoder.dswfm_res_init=-2.4"
  "HybridEncoder.paper_refine_apply=proj"
)

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

run_exp ft_control
run_exp dug_interleave "${dug_interleave_args[@]}"
run_exp dug_high_order "${dug_high_order_args[@]}"
run_exp dug_full "${dug_full_args[@]}"
run_exp dswfm "${dswfm_args[@]}"
run_exp dug_interleave_dswfm "${dug_interleave_args[@]}" "${dswfm_args[@]}"
run_exp dug_high_order_dswfm "${dug_high_order_args[@]}" "${dswfm_args[@]}"
run_exp dug_full_dswfm "${dug_full_args[@]}" "${dswfm_args[@]}"

echo "Finished. Outputs are under ${ROOT_DIR}/${BASE_OUTPUT_DIR}"
