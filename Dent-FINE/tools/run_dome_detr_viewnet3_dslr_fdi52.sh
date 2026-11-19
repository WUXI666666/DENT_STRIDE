#!/usr/bin/env bash
# Train the official Dome-DETR-S implementation on the canonical View-Net3
# DSLR FDI52 split.  This runner is intentionally isolated from Dent-FINE
# outputs and keeps all resumable state only while a run is in progress.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOME_DIR="${ROOT_DIR}/third_party/dome_detr"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-160}"
IMGSZ="${IMGSZ:-640}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1000000}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

CONFIG_REL="configs/dome/Dome-S-ViewNet3-DSLR-FDI52.yml"
CONFIG_FILE="${DOME_DIR}/${CONFIG_REL}"
BASE_DIR="${ROOT_DIR}/outputs/viewnet3_dslr_baseline_comparison"
EXP_DIR="${BASE_DIR}/runs/dome_detr_s_fdi52"
METRIC_COLLECTOR="${ROOT_DIR}/tools/dental/collect_dome_detr_coco_metrics.py"

CURRENT_EXP_DIR=""

fail() {
  echo "[Dome-DETR runner] $*" >&2
  return 1
}

on_exit() {
  local status="$?"
  if [[ "${status}" -ne 0 && -n "${CURRENT_EXP_DIR}" ]]; then
    rm -f "${CURRENT_EXP_DIR}/RUNNING"
    touch "${CURRENT_EXP_DIR}/FAILED"
  fi
  exit "${status}"
}
trap on_exit EXIT

require_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

completed_epoch_present() {
  local official_log="$1"
  "${PYTHON_BIN}" - "${official_log}" "${EPOCHS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_epoch = int(sys.argv[2]) - 1
if not path.is_file():
    raise SystemExit(1)
for line in reversed(path.read_text(errors="replace").splitlines()):
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if int(item.get("epoch", -1)) >= target_epoch:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

select_candidate() {
  # The official solver saves best_stg2 only when the fine-tuning phase beats
  # the phase-1 best AP.  Therefore best_stg2 is preferred; otherwise the
  # phase-1 best is the official selected model.
  if [[ -f "${EXP_DIR}/best.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/best.pth"
  elif [[ -f "${EXP_DIR}/best_stg2.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/best_stg2.pth"
  elif [[ -f "${EXP_DIR}/best_stg1.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/best_stg1.pth"
  elif [[ -f "${EXP_DIR}/last.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/last.pth"
  else
    return 1
  fi
}

checkpoint_last_epoch() {
  # Dome checkpoints are complete solver states, not model-only weights.  Read
  # the saved epoch on CPU so stage-2 recovery can prefer a newer checkpoint
  # without allocating a GPU or changing the active training job.
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu")
epoch = state.get("last_epoch") if isinstance(state, dict) else None
if not isinstance(epoch, int):
    raise SystemExit(1)
print(epoch)
PY
}

select_resume_checkpoint() {
  # The official solver writes last.pth only before stop_epoch (108), while
  # each improving stage-2 validation saves a full best_stg2.pth state.  For a
  # stage-2 interruption, resuming that newer state preserves its optimizer,
  # scheduler, EMA, and last_epoch instead of replaying the whole stage.
  #
  # If best_stg2 was saved at the final epoch but the process was interrupted
  # before the corresponding log line was flushed, retain the existing
  # last.pth fallback.  That fallback safely replays stage 2 and avoids a
  # zero-iteration resume followed by an incomplete-log failure.
  local stage2_epoch
  if [[ -f "${EXP_DIR}/best_stg2.pth" ]]; then
    stage2_epoch="$(checkpoint_last_epoch "${EXP_DIR}/best_stg2.pth" 2>/dev/null || true)"
    if [[ "${stage2_epoch}" =~ ^[0-9]+$ ]] && (( stage2_epoch < EPOCHS - 1 )); then
      printf '%s\n' "${EXP_DIR}/best_stg2.pth"
      return 0
    fi
  fi
  if [[ -f "${EXP_DIR}/last.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/last.pth"
  elif [[ -f "${EXP_DIR}/best_stg1.pth" ]]; then
    printf '%s\n' "${EXP_DIR}/best_stg1.pth"
  elif [[ -f "${EXP_DIR}/best_stg2.pth" ]]; then
    # No phase-1 recovery state remains; a valid stage-2 best is still the
    # only safe state from which the official solver can resume.
    printf '%s\n' "${EXP_DIR}/best_stg2.pth"
  else
    return 1
  fi
}

cleanup_final_checkpoints() {
  local best_path="${EXP_DIR}/best.pth"
  [[ -f "${best_path}" ]] || fail "Cannot clean without ${best_path}"
  find "${EXP_DIR}" -type f -name '*.pth' ! -path "${best_path}" -delete
  local count
  count="$(find "${EXP_DIR}" -type f -name '*.pth' -print | wc -l | tr -d '[:space:]')"
  [[ "${count}" == "1" ]] || fail "Expected exactly one final .pth checkpoint, found ${count}"
}

finalize_completed_training() {
  local candidate
  candidate="$(select_candidate)" || fail "Training finished but no Dome checkpoint was found in ${EXP_DIR}"
  local eval_log="${EXP_DIR}/best_eval.stdout.log"

  echo "[$(date -Is)] Evaluating selected Dome checkpoint: ${candidate}"
  (
    cd "${DOME_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
      -c "${CONFIG_FILE}" --use-amp --seed="${SEED}" -d cuda \
      --output-dir "${EXP_DIR}" --test-only -r "${candidate}" \
      -u \
        "checkpoint_freq=${CHECKPOINT_FREQ}" \
        "epoches=${EPOCHS}" \
        "eval_spatial_size=[${IMGSZ},${IMGSZ}]" \
        "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}" \
        "train_dataloader.num_workers=${NUM_WORKERS}" \
        "train_dataloader.collate_fn.base_size=${IMGSZ}" \
        "train_dataloader.collate_fn.base_size_repeat=null" \
        "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}" \
        "val_dataloader.num_workers=${NUM_WORKERS}"
  ) 2>&1 | tee "${eval_log}"

  # Keep the original official checkpoint through the evaluation so `-r` has a
  # valid artifact.  Then rename it to the cross-method stable convention.
  if [[ "${candidate}" != "${EXP_DIR}/best.pth" ]]; then
    mv -f "${candidate}" "${EXP_DIR}/best.pth"
  fi
  "${PYTHON_BIN}" "${METRIC_COLLECTOR}" \
    --log "${eval_log}" --train-log "${EXP_DIR}/log.txt" \
    --checkpoint "${EXP_DIR}/best.pth" \
    --output "${EXP_DIR}/coco_eval/coco_metrics.json" \
    --method "Dome-DETR-S"
  # Add lightweight provenance expected by the common baseline table without
  # retaining another checkpoint or prediction artifact.
  "${PYTHON_BIN}" - "${EXP_DIR}/coco_eval/coco_metrics.json" "${EXP_DIR}/log.txt" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
payload = json.loads(metrics_path.read_text())
rows = []
for line in log_path.read_text(errors="replace").splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    values = row.get("test_coco_eval_bbox")
    epoch = row.get("epoch")
    if isinstance(epoch, int) and isinstance(values, list) and values:
        rows.append((epoch, float(values[0])))
if rows:
    best_epoch, _ = max(rows, key=lambda row: row[1])
    payload["training"] = {
        "epochs_target": 160,
        "epochs_logged": len(rows),
        "last_epoch": rows[-1][0],
        "best_epoch": best_epoch,
        "optimizer": "AdamW",
        "seed": 0,
        "horizontal_flip": False,
    }
payload["framework"] = "Dome-DETR"
payload["architecture"] = "Dome-DETR-S (HGNetv2-B0)"
metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
  cleanup_final_checkpoints
  rm -f "${EXP_DIR}/RUNNING" "${EXP_DIR}/FAILED"
  touch "${EXP_DIR}/COMPLETED"
  CURRENT_EXP_DIR=""
  echo "[$(date -Is)] Dome-DETR-S complete: ${EXP_DIR}/best.pth"
}

train_or_resume() {
  local resume_args=()
  local resume_checkpoint=""
  resume_checkpoint="$(select_resume_checkpoint || true)"
  if [[ -n "${resume_checkpoint}" ]]; then
    resume_args=(-r "${resume_checkpoint}")
    echo "[$(date -Is)] Resuming official Dome state: ${resume_checkpoint}"
  elif [[ -f "${EXP_DIR}/log.txt" ]]; then
    # A partial log with no official state cannot be resumed safely.  Do not
    # silently mix it with a new initialization.
    fail "Partial Dome run lacks a resumable solver checkpoint: ${EXP_DIR}"
  fi

  echo "============================================================"
  echo "[$(date -Is)] GPU ${GPU_ID}: Dome-DETR-S DSLR FDI52"
  echo "Config: ${CONFIG_FILE}"
  echo "Output: ${EXP_DIR}"
  echo "Epochs=${EPOCHS}, image=${IMGSZ}, global batch=${TRAIN_BATCH_SIZE}, seed=${SEED}"
  echo "============================================================"
  (
    cd "${DOME_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
      -c "${CONFIG_FILE}" --use-amp --seed="${SEED}" -d cuda \
      --output-dir "${EXP_DIR}" "${resume_args[@]}" \
      -u \
        "checkpoint_freq=${CHECKPOINT_FREQ}" \
        "epoches=${EPOCHS}" \
        "eval_spatial_size=[${IMGSZ},${IMGSZ}]" \
        "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}" \
        "train_dataloader.num_workers=${NUM_WORKERS}" \
        "train_dataloader.collate_fn.base_size=${IMGSZ}" \
        "train_dataloader.collate_fn.base_size_repeat=null" \
        "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}" \
        "val_dataloader.num_workers=${NUM_WORKERS}"
  ) 2>&1 | tee -a "${EXP_DIR}/train.stdout.log"
}

main() {
  require_file "${CONFIG_FILE}"
  require_file "${METRIC_COLLECTOR}"
  [[ -x "${PYTHON_BIN}" ]] || fail "Python interpreter is not executable: ${PYTHON_BIN}"
  [[ "${EPOCHS}" == "160" ]] || fail "Dome comparison is pinned to 160 epochs (got ${EPOCHS})"
  [[ "${IMGSZ}" == "640" ]] || fail "Dome comparison is pinned to 640 px (got ${IMGSZ})"
  [[ "${TRAIN_BATCH_SIZE}" == "8" ]] || fail "Dome comparison is pinned to global batch 8 (got ${TRAIN_BATCH_SIZE})"
  [[ "${SEED}" == "0" ]] || fail "Dome comparison is pinned to seed 0 (got ${SEED})"

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${EXP_DIR}/COMPLETED" && -f "${EXP_DIR}/best.pth" ]]; then
    echo "[skip] Dome-DETR-S already completed: ${EXP_DIR}"
    return 0
  fi

  mkdir -p "${EXP_DIR}"
  rm -f "${EXP_DIR}/FAILED" "${EXP_DIR}/COMPLETED"
  touch "${EXP_DIR}/RUNNING"
  CURRENT_EXP_DIR="${EXP_DIR}"

  # This condition also makes the post-training evaluation/finalization
  # resumable: an interruption after epoch 159 does not retrain the model.
  if ! completed_epoch_present "${EXP_DIR}/log.txt"; then
    train_or_resume
  else
    echo "[$(date -Is)] Found all ${EPOCHS} official epochs; finalizing only."
  fi
  completed_epoch_present "${EXP_DIR}/log.txt" || fail "Dome training returned before epoch $((EPOCHS - 1))"
  finalize_completed_training
}

main "$@"
