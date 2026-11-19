#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
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
START_FROM="${START_FROM:-abl106_phone_tooth_prior_baseline}"

PHONE_CONFIG="configs/deim_dfine/deim_hgnetv2_s_phone.yml"
DENTEX_CONFIG="configs/deim_dfine/deim_hgnetv2_s_dentex.yml"
PHONE_PRIOR="${ROOT_DIR}/${BASE_OUTPUT_DIR}/priors/phone_tooth_templates.pt"
DENTEX_PRIOR="${ROOT_DIR}/${BASE_OUTPUT_DIR}/priors/dentex_tooth_templates.pt"

ORDER=(
  abl106_phone_tooth_prior_baseline
  abl107_phone_factor
  abl108_phone_prototype
  abl109_phone_spatial_shape
  abl110_dentex_tooth_prior_baseline
  abl111_dentex_factor
  abl112_dentex_prototype
  abl113_dentex_spatial_shape
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

build_priors() {
  mkdir -p "$(dirname "${PHONE_PRIOR}")"
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/dental/build_tooth_templates.py" \
    --ann-file /home/wuxi/coco/train/annotations_fdi_types.json \
    --output "${PHONE_PRIOR}" \
    --fdi-order from_categories
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/dental/build_tooth_templates.py" \
    --ann-file /home/wuxi/DENTEX2023/annotations/instances_train2017.json \
    --output "${DENTEX_PRIOR}" \
    --fdi-order dentex_enumeration32
}

run_exp() {
  local name="$1"
  local config="$2"
  local prior_path="$3"
  local view_type="$4"
  local fdi_order="$5"
  shift 5

  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  local update_args=(
    "fdi_label_order=${fdi_order}"
    "dental_prior_path=${prior_path}"
    "HybridEncoder.use_cmfa_fusion=True"
    "HybridEncoder.use_hooa_attention=True"
    "HybridEncoder.use_efruf_gradient_fusion=True"
    "HybridEncoder.use_sinf_spatial_field=True"
    "HybridEncoder.use_smwg_spectral_wavelet=True"
    "HybridEncoder.paper_refine_apply=outs"
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
    "PostProcessor.use_dental_spatial_prior=False"
    "PostProcessor.use_dental_shape_prior=False"
    "PostProcessor.use_dental_view_prior=False"
    "PostProcessor.use_global_assignment=False"
    "PostProcessor.dental_view_type=${view_type}"
    "DEIMCriterion.losses=[mal,boxes,local]"
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
    refresh_summary
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

  refresh_summary
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"
build_priors
refresh_summary

if should_run abl106_phone_tooth_prior_baseline; then
  run_exp abl106_phone_tooth_prior_baseline "${PHONE_CONFIG}" "${PHONE_PRIOR}" unknown fdi52
fi
if should_run abl107_phone_factor; then
  run_exp abl107_phone_factor "${PHONE_CONFIG}" "${PHONE_PRIOR}" unknown fdi52 \
    "DFINETransformer.use_factor_heads=True" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent]"
fi
if should_run abl108_phone_prototype; then
  run_exp abl108_phone_prototype "${PHONE_CONFIG}" "${PHONE_PRIOR}" unknown fdi52 \
    "DFINETransformer.use_factor_heads=True" \
    "DFINETransformer.use_tooth_prototype=True" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent,proto]"
fi
if should_run abl109_phone_spatial_shape; then
  run_exp abl109_phone_spatial_shape "${PHONE_CONFIG}" "${PHONE_PRIOR}" unknown fdi52 \
    "DFINETransformer.use_factor_heads=True" \
    "DFINETransformer.use_tooth_prototype=True" \
    "PostProcessor.use_dental_spatial_prior=True" \
    "PostProcessor.spatial_prior_weight=0.2" \
    "PostProcessor.use_dental_shape_prior=True" \
    "PostProcessor.shape_prior_weight=0.05" \
    "PostProcessor.use_dental_view_prior=True" \
    "PostProcessor.view_prior_weight=0.2" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent,proto]"
fi
if should_run abl110_dentex_tooth_prior_baseline; then
  run_exp abl110_dentex_tooth_prior_baseline "${DENTEX_CONFIG}" "${DENTEX_PRIOR}" panorama dentex_enumeration32
fi
if should_run abl111_dentex_factor; then
  run_exp abl111_dentex_factor "${DENTEX_CONFIG}" "${DENTEX_PRIOR}" panorama dentex_enumeration32 \
    "DFINETransformer.use_factor_heads=True" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent]"
fi
if should_run abl112_dentex_prototype; then
  run_exp abl112_dentex_prototype "${DENTEX_CONFIG}" "${DENTEX_PRIOR}" panorama dentex_enumeration32 \
    "DFINETransformer.use_factor_heads=True" \
    "DFINETransformer.use_tooth_prototype=True" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent,proto]"
fi
if should_run abl113_dentex_spatial_shape; then
  run_exp abl113_dentex_spatial_shape "${DENTEX_CONFIG}" "${DENTEX_PRIOR}" panorama dentex_enumeration32 \
    "DFINETransformer.use_factor_heads=True" \
    "DFINETransformer.use_tooth_prototype=True" \
    "PostProcessor.use_dental_spatial_prior=True" \
    "PostProcessor.spatial_prior_weight=0.2" \
    "PostProcessor.use_dental_shape_prior=True" \
    "PostProcessor.shape_prior_weight=0.05" \
    "PostProcessor.use_dental_view_prior=True" \
    "PostProcessor.view_prior_weight=0.2" \
    "DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent,proto]"
fi

refresh_summary
echo "Summary refreshed at ${ROOT_DIR}/${BASE_OUTPUT_DIR}/ablation_summary.md"
