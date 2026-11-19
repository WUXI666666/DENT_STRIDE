#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_dentex.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablation_dentex_dualopt_adamw/abl155_dentex_dualopt_adamw}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

CMD=(
  "${PYTHON_BIN}" train.py
  -c "${CONFIG}"
  --use-amp
  --seed=0
  -d cuda
  -u
  HybridEncoder.use_hfsa=False
  HybridEncoder.use_wav_enhance=False
  HybridEncoder.use_stat_denoise=False
  HybridEncoder.use_multipath_fusion=False
  HybridEncoder.use_dynamic_rf=False
  HybridEncoder.use_mpc_multiscale_perception=False
  HybridEncoder.use_pqg_high_frequency=False
  HybridEncoder.use_cmfa_fusion=False
  HybridEncoder.use_pfesa=False
  HybridEncoder.use_ucan_lkd=False
  HybridEncoder.use_ucm_uncertainty=False
  HybridEncoder.use_retinex_prior=False
  HybridEncoder.use_hooa_attention=False
  HybridEncoder.use_multinex_prior=False
  HybridEncoder.use_dcsam_dual_consistency=False
  HybridEncoder.use_iqp_quality_prior=False
  HybridEncoder.use_dformer_geometry_prior=False
  HybridEncoder.use_cdml_domain_adapter=False
  HybridEncoder.use_efruf_gradient_fusion=False
  HybridEncoder.use_sinf_spatial_field=False
  HybridEncoder.use_smwg_spectral_wavelet=False
  HybridEncoder.use_structural_entropy_frequency_refiner=False
  HybridEncoder.use_rot_equivariant_detail_refiner=False
  HybridEncoder.use_hedp_prompt_refiner=False
  HybridEncoder.use_sampg_sst_refiner=False
  HybridEncoder.use_segd_entropy_refiner=False
  HybridEncoder.use_fsdetr_frequency_spatial_refiner=False
  HybridEncoder.use_df2net_dynamic_fusion=False
  HybridEncoder.use_sdconet_saliency_refiner=False
  HybridEncoder.use_idgbr_boundary_refiner=False
  HybridEncoder.use_bilalora_layer_adapter=False
  HybridEncoder.use_surmt_structure_bridge=False
  HybridEncoder.use_mav_sparse_interaction=False
  HybridEncoder.use_cnm_ode_refiner=False
  HybridEncoder.use_cldyn_rsc=False
  HybridEncoder.use_ced_cascade_refiner=False
  HybridEncoder.paper_refine_apply=proj
  HybridEncoder.apply_extra_modules=proj
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_factor_heads=False
  DFINETransformer.use_tooth_prototype=False
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_uni_gate_infer=False
  DFINETransformer.use_qgp=False
  DFINETransformer.qgp_layers=[]
  DFINETransformer.use_pgr_roi_reasoning=False
  DFINETransformer.use_spegc_prompt_graph=False
  DFINETransformer.use_cluster_query_refiner=False
  DFINETransformer.use_query_recovery_memory=False
  DFINETransformer.use_pfrpn_query_adapter=False
  DFINETransformer.use_tooth_template_prior=False
  DFINETransformer.use_anatomical_region_prior=False
  PostProcessor.use_dental_spatial_prior=False
  PostProcessor.use_dental_shape_prior=False
  PostProcessor.use_dental_view_prior=False
  PostProcessor.use_global_assignment=False
  DEIMCriterion.losses=[mal,boxes,local]
  DEIMCriterion.use_fdi_anatomy_loss=False
  DEIMCriterion.use_dsfed_dual_scale_kd=False
  DEIMCriterion.use_tooth_size_prior_loss=False
  optimizer.type=DualOptAdamW
  optimizer.rollback_weight=0.005
  optimizer.realtime_decay=0.01
  optimizer.layer_decay_power=1.5
  optimizer.layer_decay_min=0.5
  optimizer.layer_decay_max=1.5
  train_dataloader.total_batch_size="${TRAIN_BATCH_SIZE}"
  train_dataloader.num_workers="${NUM_WORKERS}"
  val_dataloader.total_batch_size="${VAL_BATCH_SIZE}"
  val_dataloader.num_workers="${NUM_WORKERS}"
  --output-dir "${ROOT_DIR}/${OUTPUT_DIR}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
  if [[ "${KEEP_ONLY_BEST}" == "1" && -d "${ROOT_DIR}/${OUTPUT_DIR}" ]]; then
    find "${ROOT_DIR}/${OUTPUT_DIR}" -type f -name '*.pth' \
      ! -name 'best_stg1.pth' \
      ! -name 'best_stg2.pth' \
      ! -name 'best.pth' \
      -delete
  fi
fi
