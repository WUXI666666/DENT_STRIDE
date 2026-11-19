#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ablation_coco_cmfa_hooa_dswfm_outs_abl34_recipe}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

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
  HybridEncoder.use_cmfa_fusion=True
  HybridEncoder.use_pfesa=False
  HybridEncoder.use_ucan_lkd=False
  HybridEncoder.use_ucm_uncertainty=False
  HybridEncoder.use_retinex_prior=False
  HybridEncoder.use_hooa_attention=True
  HybridEncoder.use_multinex_prior=False
  HybridEncoder.use_msck_refiner=False
  HybridEncoder.use_csaf_filter=False
  HybridEncoder.use_csaf_dmssp=False
  HybridEncoder.use_rdvq_refiner=False
  HybridEncoder.use_datpp_refiner=False
  HybridEncoder.use_cdml_domain_adapter=False
  HybridEncoder.use_efruf_gradient_fusion=False
  HybridEncoder.use_sinf_spatial_field=False
  HybridEncoder.use_smwg_spectral_wavelet=False
  HybridEncoder.use_structural_entropy_frequency_refiner=False
  HybridEncoder.use_dug_hoif_fusion=False
  HybridEncoder.use_dced_cascade=False
  HybridEncoder.use_ced_cascade_refiner=False
  HybridEncoder.use_dswfm=True
  HybridEncoder.dswfm_use_dynamic_fourier=True
  HybridEncoder.dswfm_use_haar_detail=True
  HybridEncoder.dswfm_use_bidirectional=False
  HybridEncoder.dswfm_res_init=-3.0
  HybridEncoder.paper_refine_apply=outs
  HybridEncoder.apply_extra_modules=proj
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_factor_heads=False
  DFINETransformer.use_tooth_prototype=False
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_uni_gate_infer=False
  DFINETransformer.uni_gate_apply_last_only=True
  DFINETransformer.use_qgp=False
  DFINETransformer.qgp_layers=[]
  DFINETransformer.use_pgr_roi_reasoning=False
  DFINETransformer.roi_init_ratio=0.5
  DFINETransformer.use_fdi_factor_prior=False
  DFINETransformer.fdi_factor_logit_scale=0.0
  DFINETransformer.use_spegc_prompt_graph=False
  DFINETransformer.spegc_num_prompts=16
  DFINETransformer.spegc_graph_topk=8
  DFINETransformer.spegc_ref_bias=0.25
  DFINETransformer.spegc_res_scale=0.20
  PostProcessor.use_dental_spatial_prior=False
  PostProcessor.use_dental_shape_prior=False
  PostProcessor.use_dental_view_prior=False
  PostProcessor.use_global_assignment=False
  DEIMCriterion.losses=[mal,boxes,local]
  DEIMCriterion.use_fdi_anatomy_loss=False
  DEIMCriterion.use_dsfed_dual_scale_kd=False
  DEIMCriterion.use_tooth_size_prior_loss=False
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
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
