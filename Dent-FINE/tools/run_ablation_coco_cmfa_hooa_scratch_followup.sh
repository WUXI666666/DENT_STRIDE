#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_coco_cmfa_hooa_scratch_followup}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-160}"
LR="${LR:-0.0005}"
BACKBONE_LR="${BACKBONE_LR:-0.00005}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
START_FROM="${START_FROM:-cmfa_hooa_scratch}"
MAX_RUNS="${MAX_RUNS:-0}"

ORDER=(
  cmfa_hooa_scratch
  dswfm_outs_scratch
  dswfm_proj_scratch
  cedv2_scratch
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
  shift

  if ! should_run "${name}"; then
    return
  fi

  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  local update_args=(
    "epoches=${EPOCHS}"
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
  )
  update_args+=("$@")

  echo "=============================="
  echo "Running ${name} from scratch"
  echo "Config: ${CONFIG}"
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

dswfm_outs_args=(
  "HybridEncoder.use_dswfm=True"
  "HybridEncoder.dswfm_use_dynamic_fourier=True"
  "HybridEncoder.dswfm_use_haar_detail=True"
  "HybridEncoder.dswfm_use_bidirectional=False"
  "HybridEncoder.dswfm_res_init=-3.0"
  "HybridEncoder.paper_refine_apply=outs"
)

dswfm_proj_args=(
  "HybridEncoder.use_dswfm=True"
  "HybridEncoder.dswfm_use_dynamic_fourier=True"
  "HybridEncoder.dswfm_use_haar_detail=True"
  "HybridEncoder.dswfm_use_bidirectional=False"
  "HybridEncoder.dswfm_res_init=-3.0"
  "HybridEncoder.paper_refine_apply=proj"
)

cedv2_args=(
  "HybridEncoder.use_ced_cascade_refiner=True"
  "HybridEncoder.ced_cascade_stages=1"
  "HybridEncoder.ced_cascade_expansion=0.5"
  "HybridEncoder.ced_cascade_res_init=-1.6"
)

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"

run_exp cmfa_hooa_scratch
run_exp dswfm_outs_scratch "${dswfm_outs_args[@]}"
run_exp dswfm_proj_scratch "${dswfm_proj_args[@]}"
run_exp cedv2_scratch "${cedv2_args[@]}"

echo "Finished. Outputs are under ${ROOT_DIR}/${BASE_OUTPUT_DIR}"
