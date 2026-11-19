#!/usr/bin/env bash
# Train evaluator-aligned YOLOv8-s, YOLO11-s, and pure DEIM-s baselines on the
# canonical View-Net3 DSLR FDI52 split. Output is intentionally isolated from
# all Dent-FINE ablations and completed runs retain one selected best weight.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
YOLO_BIN="${YOLO_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/yolo}"
GPU_ID="${GPU_ID:-0}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/viewnet3_dslr_baseline_comparison}"
BASE_DIR="${ROOT_DIR}/${BASE_OUTPUT_DIR}"
RUNS_DIR="${BASE_DIR}/runs"
WEIGHTS_DIR="${BASE_DIR}/weights"
YOLO_DATA_DIR="${BASE_DIR}/yolo_dataset"
YOLO_DATA_YAML="${YOLO_DATA_DIR}/viewnet3_dslr_fdi52.yaml"
DEIM_CONFIG="${DEIM_CONFIG:-configs/deim_dfine/deim_hgnetv2_s_viewnet3_dslr_fdi52.yml}"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_viewnet3_dslr_baselines.py"
EXPORT_SCRIPT="${ROOT_DIR}/tools/dental/export_viewnet3_fdi52_to_yolo.py"
EVALUATE_YOLO_SCRIPT="${ROOT_DIR}/tools/dental/evaluate_yolo_coco_detection.py"

TRAIN_ANN="${ROOT_DIR}/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_train_fdi52.json"
VAL_ANN="${ROOT_DIR}/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json"
TRAIN_IMAGES="/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/train"
VAL_IMAGES="/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val"

EPOCHS="${EPOCHS:-160}"
IMGSZ="${IMGSZ:-640}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
# DEIM maintains resumable last.pth each epoch, so periodic snapshots are not
# needed and would waste scarce disk capacity.
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000000}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
START_FROM="${START_FROM:-yolov8s_fdi52}"
MAX_RUNS="${MAX_RUNS:-0}"

ORDER=(yolov8s_fdi52 yolo11s_fdi52 deim_s_fdi52)
CURRENT_EXP_DIR=""
RUN_COUNT=0

summarize() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --base-dir "${BASE_DIR}" --expected-epochs "${EPOCHS}"
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

download_pretrained_weight() {
  local filename="$1"
  local url="$2"
  local target="${WEIGHTS_DIR}/${filename}"
  if [[ -s "${target}" ]]; then
    return
  fi
  mkdir -p "${WEIGHTS_DIR}"
  echo "Downloading ${filename} to ${target}"
  curl --fail --location --http1.1 --retry 4 --retry-all-errors --retry-delay 2 \
    --output "${target}.download" "${url}"
  mv "${target}.download" "${target}"
  if [[ ! -s "${target}" ]]; then
    echo "Downloaded weight is empty: ${target}" >&2
    return 1
  fi
}

cleanup_yolo_weights() {
  local exp_dir="$1"
  if [[ -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/weights/best.pt" ]]; then
    find "${exp_dir}/weights" -type f -name '*.pt' ! -path "${exp_dir}/weights/best.pt" -delete
  fi
}

cleanup_deim_weights() {
  local exp_dir="$1"
  if [[ -f "${exp_dir}/COMPLETED" && -f "${exp_dir}/best.pth" ]]; then
    find "${exp_dir}" -type f -name '*.pth' ! -path "${exp_dir}/best.pth" -delete
  fi
}

begin_experiment() {
  local exp_dir="$1"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${exp_dir}/COMPLETED" ]]; then
    return 1
  fi
  mkdir -p "${exp_dir}"
  rm -f "${exp_dir}/FAILED" "${exp_dir}/COMPLETED"
  touch "${exp_dir}/RUNNING"
  CURRENT_EXP_DIR="${exp_dir}"
  return 0
}

finish_experiment() {
  local exp_dir="$1"
  rm -f "${exp_dir}/RUNNING" "${exp_dir}/FAILED"
  touch "${exp_dir}/COMPLETED"
  CURRENT_EXP_DIR=""
  summarize
  RUN_COUNT=$((RUN_COUNT + 1))
  if [[ "${MAX_RUNS}" -gt 0 && "${RUN_COUNT}" -ge "${MAX_RUNS}" ]]; then
    echo "Reached MAX_RUNS=${MAX_RUNS}; queue paused."
    exit 0
  fi
}

run_yolo() {
  local name="$1"
  local filename="$2"
  local url="$3"
  if ! should_run "${name}"; then
    return
  fi
  local exp_dir="${RUNS_DIR}/${name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${exp_dir}/COMPLETED" ]]; then
    echo "[skip] ${name} already completed"
    return
  fi
  download_pretrained_weight "${filename}" "${url}"
  if [[ -f "${exp_dir}/results.csv" && ! -f "${exp_dir}/weights/last.pt" && ! -f "${exp_dir}/COMPLETED" ]]; then
    echo "Refusing to mix a partial YOLO run without resumable last.pt: ${exp_dir}" >&2
    return 1
  fi
  begin_experiment "${exp_dir}" || return

  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: ${name}"
  echo "Output: ${exp_dir}"
  echo "============================================================"
  if [[ -f "${exp_dir}/weights/last.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${YOLO_BIN}" detect train resume="${exp_dir}/weights/last.pt"
  else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${YOLO_BIN}" detect train \
      model="${WEIGHTS_DIR}/${filename}" \
      data="${YOLO_DATA_YAML}" \
      epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${TRAIN_BATCH_SIZE}" workers="${NUM_WORKERS}" \
      device=0 project="${RUNS_DIR}" name="${name}" exist_ok=True \
      pretrained=True optimizer=AdamW lr0=0.0002 momentum=0.9 weight_decay=0.0001 \
      warmup_epochs=3.0 warmup_bias_lr=0.0 close_mosaic=30 cos_lr=False \
      seed="${SEED}" deterministic=True amp=True cache=False rect=False \
      save=True save_period=-1 plots=True
  fi

  if [[ ! -f "${exp_dir}/weights/best.pt" ]]; then
    echo "YOLO training returned without best.pt: ${exp_dir}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${EVALUATE_YOLO_SCRIPT}" \
    --weights "${exp_dir}/weights/best.pt" \
    --ann-file "${VAL_ANN}" \
    --image-dir "${VAL_IMAGES}" \
    --output-dir "${exp_dir}/coco_eval" \
    --imgsz "${IMGSZ}" --device 0 --batch "${TRAIN_BATCH_SIZE}" --conf 0.001 --iou 0.7 --max-det 300
  finish_experiment "${exp_dir}"
  cleanup_yolo_weights "${exp_dir}"
}

# A strict DEIM-s baseline: preserve the standard AdamW recipe, keep all
# project-specific enhancement branches off, and zero the fork-only arch loss.
deim_base_args=(
  "HGNetv2.pretrained=True"
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
  "DFINETransformer.use_mr=False"
  "DFINETransformer.use_tooth_query_embed=False"
  "DFINETransformer.use_uni_gate=False"
  "DFINETransformer.use_uni_gate_infer=False"
  "DFINETransformer.use_qgp=False"
  "DFINETransformer.use_pgr_roi_reasoning=False"
  "DFINETransformer.use_fdi_factor_prior=False"
  "DFINETransformer.use_fdi_spatial_prior=False"
  "DFINETransformer.use_spegc_prompt_graph=False"
  "DFINETransformer.use_cluster_query_refiner=False"
  "DFINETransformer.use_query_recovery_memory=False"
  "DFINETransformer.use_pfrpn_query_adapter=False"
  "DFINETransformer.use_tooth_template_prior=False"
  "DFINETransformer.use_anatomical_region_prior=False"
  "PostProcessor.use_dental_spatial_prior=False"
  "PostProcessor.use_dental_shape_prior=False"
  "PostProcessor.use_dental_appearance_prior=False"
  "PostProcessor.use_dental_view_prior=False"
  "PostProcessor.use_global_assignment=False"
  "DEIMCriterion.use_fdi_anatomy_loss=False"
  "DEIMCriterion.use_dsfed_dual_scale_kd=False"
  "DEIMCriterion.use_tooth_size_prior_loss=False"
  "DEIMCriterion.weight_dict.loss_arch=0.0"
  "checkpoint_freq=${CHECKPOINT_FREQ}"
  "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "val_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}"
  "val_dataloader.num_workers=${NUM_WORKERS}"
  "epoches=${EPOCHS}"
)

run_deim() {
  local name="deim_s_fdi52"
  if ! should_run "${name}"; then
    return
  fi
  local exp_dir="${RUNS_DIR}/${name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${exp_dir}/COMPLETED" ]]; then
    echo "[skip] ${name} already completed"
    return
  fi
  if [[ -f "${exp_dir}/log.txt" && ! -f "${exp_dir}/last.pth" && ! -f "${exp_dir}/COMPLETED" ]]; then
    echo "Refusing to mix a partial DEIM run without resumable last.pth: ${exp_dir}" >&2
    return 1
  fi
  begin_experiment "${exp_dir}" || return

  local command=("${PYTHON_BIN}" train.py -c "${DEIM_CONFIG}" --use-amp --seed="${SEED}" -d cuda)
  if [[ -f "${exp_dir}/last.pth" ]]; then
    command+=(--resume "${exp_dir}/last.pth")
  fi
  command+=(-u "${deim_base_args[@]}" --output-dir "${exp_dir}")
  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: ${name}"
  echo "Output: ${exp_dir}"
  printf 'Command: '
  printf '%q ' "${command[@]}"
  printf '\n'
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${command[@]}"

  if [[ ! -f "${exp_dir}/best.pth" ]]; then
    echo "DEIM training returned without best.pth: ${exp_dir}" >&2
    return 1
  fi
  finish_experiment "${exp_dir}"
  cleanup_deim_weights "${exp_dir}"
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_DIR}" "${RUNS_DIR}" "${WEIGHTS_DIR}"
if [[ ! -x "${YOLO_BIN}" ]]; then
  echo "Missing Ultralytics yolo executable: ${YOLO_BIN}" >&2
  exit 1
fi
for path in "${TRAIN_ANN}" "${VAL_ANN}"; do
  [[ -f "${path}" ]] || { echo "Missing annotation file: ${path}" >&2; exit 1; }
done

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON_BIN}" "${EXPORT_SCRIPT}" \
    --train-ann "${TRAIN_ANN}" --val-ann "${VAL_ANN}" \
    --train-images "${TRAIN_IMAGES}" --val-images "${VAL_IMAGES}" \
    --output-dir "${YOLO_DATA_DIR}"
fi
[[ -f "${YOLO_DATA_YAML}" ]] || { echo "Missing generated YOLO data YAML: ${YOLO_DATA_YAML}" >&2; exit 1; }

summarize
run_yolo yolov8s_fdi52 yolov8s.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt
run_yolo yolo11s_fdi52 yolo11s.pt https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt
run_deim

echo "All View-Net3 DSLR baseline comparison runs completed."
