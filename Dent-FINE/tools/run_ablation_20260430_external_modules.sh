#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wuxi/miniconda3/envs/deimv2/bin/python}"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-configs/deim_dfine/deim_hgnetv2_s_coco.yml}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/ablation_all_new_modules}"
KEEP_ONLY_BEST="${KEEP_ONLY_BEST:-1}"
START_FROM="${START_FROM:-abl67_cnm_ode_refiner_v2}"
ONLY_EXPS="${ONLY_EXPS:-}"
EPOCHES="${EPOCHES:-}"
SMOKE_DATASET="${SMOKE_DATASET:-0}"
RERUN_POLICY="${RERUN_POLICY:-archive}"
DETERMINISTIC="${DETERMINISTIC:-0}"
SUMMARY_RUN_SCOPE="${SUMMARY_RUN_SCOPE:-latest}"
SUMMARY_SCRIPT="${ROOT_DIR}/tools/summarize_ablation_all_new_modules.py"

COMMON_ARGS=(
  train.py
  -c "$CONFIG"
  --use-amp
  --seed=0
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
  HybridEncoder.use_msck_refiner=False
  HybridEncoder.msck_kernels=[3,5,7]
  HybridEncoder.use_csaf_filter=False
  HybridEncoder.csaf_filter_scale=4
  HybridEncoder.use_csaf_dmssp=False
  HybridEncoder.use_rdvq_refiner=False
  HybridEncoder.rdvq_codebook_size=64
  HybridEncoder.rdvq_temperature=0.7
  HybridEncoder.use_datpp_refiner=False
  HybridEncoder.datpp_refiner_stages=[1,2]
  HybridEncoder.datpp_n_heads=8
  HybridEncoder.datpp_n_groups=4
  HybridEncoder.datpp_stride=2
  HybridEncoder.datpp_offset_range_factor=2.0
  HybridEncoder.use_dcsam_dual_consistency=False
  HybridEncoder.use_iqp_quality_prior=False
  HybridEncoder.iqp_codebook_size=16
  HybridEncoder.use_dformer_geometry_prior=False
  HybridEncoder.use_cdml_domain_adapter=False
  HybridEncoder.use_efruf_gradient_fusion=False
  HybridEncoder.efruf_unfold_steps=2
  HybridEncoder.use_sinf_spatial_field=False
  HybridEncoder.sinf_num_blocks=2
  HybridEncoder.use_smwg_spectral_wavelet=False
  HybridEncoder.use_bilalora_layer_adapter=False
  HybridEncoder.bilalora_rank=8
  HybridEncoder.bilalora_topk=2
  HybridEncoder.use_surmt_structure_bridge=False
  HybridEncoder.use_mav_sparse_interaction=False
  HybridEncoder.mav_sparse_ratio=0.25
  HybridEncoder.use_cnm_ode_refiner=False
  HybridEncoder.cnm_ode_steps=3
  HybridEncoder.cnm_ode_step_size=0.35
  HybridEncoder.cnm_ode_res_init=-1.4
  HybridEncoder.use_cldyn_rsc=False
  HybridEncoder.cldyn_filter_num=4
  HybridEncoder.cldyn_set_num=4
  HybridEncoder.cldyn_vector_num=8
  HybridEncoder.cldyn_res_init=-1.3
  HybridEncoder.use_ced_cascade_refiner=False
  HybridEncoder.ced_cascade_stages=1
  HybridEncoder.ced_cascade_expansion=0.5
  HybridEncoder.ced_cascade_res_init=-1.6
  HybridEncoder.paper_refine_apply=proj
  HybridEncoder.apply_denoise_modules=proj
  HybridEncoder.apply_refine_modules=outs
  DFINETransformer.use_uni_gate=False
  DFINETransformer.use_uni_gate_infer=False
  DFINETransformer.uni_gate_apply_last_only=True
  DFINETransformer.use_qgp=False
  DFINETransformer.qgp_layers=[]
  DFINETransformer.use_pgr_roi_reasoning=False
  DFINETransformer.roi_init_ratio=0.5
  DFINETransformer.use_fdi_factor_prior=False
  DFINETransformer.fdi_factor_logit_scale=0.0
  DFINETransformer.use_fdi_spatial_prior=False
  DFINETransformer.fdi_prior_knn=8
  DFINETransformer.fdi_prior_sigma=0.12
  DFINETransformer.fdi_prior_jaw_scale=0.08
  DFINETransformer.fdi_prior_neighbor_scale=0.12
  DFINETransformer.fdi_prior_unique_scale=0.04
  DFINETransformer.fdi_prior_dent_scale=0.04
  DFINETransformer.use_spegc_prompt_graph=False
  DFINETransformer.spegc_num_prompts=16
  DFINETransformer.spegc_graph_topk=8
  DFINETransformer.spegc_ref_bias=0.25
  DFINETransformer.spegc_res_scale=0.20
  DFINETransformer.use_cluster_query_refiner=False
  DFINETransformer.cluster_num=16
  DFINETransformer.cluster_graph_topk=6
  DFINETransformer.cluster_ref_bias=0.25
  DFINETransformer.cluster_res_scale=0.20
  DFINETransformer.use_query_recovery_memory=False
  DFINETransformer.query_recovery_scale=0.15
  DFINETransformer.use_pfrpn_query_adapter=False
  DFINETransformer.pfrpn_pool_size=4
  DFINETransformer.pfrpn_cascade_steps=2
  DFINETransformer.pfrpn_res_scale=0.20
  DFINETransformer.use_tooth_template_prior=False
  DFINETransformer.tooth_template_count=4
  DFINETransformer.tooth_template_scale=0.08
  DFINETransformer.tooth_template_spatial_scale=0.05
  DFINETransformer.tooth_template_sigma=0.18
  DFINETransformer.use_anatomical_region_prior=False
  DFINETransformer.anatomical_region_scale=0.08
  DFINETransformer.anatomical_region_sigma=0.18
  DEIMCriterion.use_fdi_anatomy_loss=False
  DEIMCriterion.use_dsfed_dual_scale_kd=False
  DEIMCriterion.dsfed_kd_temperature=2.0
  DEIMCriterion.dsfed_kd_weight=0.05
  DEIMCriterion.dsfed_box_kd_weight=0.02
  DEIMCriterion.dsfed_kd_topk_ratio=0.20
  DEIMCriterion.dsfed_kd_min_conf=0.10
  DEIMCriterion.dsfed_final_kd_weight=0.02
  DEIMCriterion.use_tooth_size_prior_loss=False
  DEIMCriterion.weight_dict.loss_tooth_size_prior=0.03
  DEIMCriterion.nwd_normalizer=0.08
  train_dataloader.total_batch_size=8
  val_dataloader.total_batch_size=16
)

if [[ -n "${EPOCHES}" ]]; then
  COMMON_ARGS+=(
    epoches="${EPOCHES}"
    flat_epoch=0
    no_aug_epoch=0
    train_dataloader.dataset.transforms.policy.epoch=[100,101,102]
    train_dataloader.collate_fn.mixup_epochs=[100,101]
    train_dataloader.collate_fn.stop_epoch=100
  )
fi

if [[ "${SMOKE_DATASET}" == "1" ]]; then
  COMMON_ARGS+=(
    train_dataloader.dataset.ann_file=outputs/smoke_dataset/train_smoke.json
    val_dataloader.dataset.ann_file=outputs/smoke_dataset/valid_smoke.json
    train_dataloader.num_workers=2
    val_dataloader.num_workers=2
    train_dataloader.total_batch_size=2
    val_dataloader.total_batch_size=2
  )
fi

NWD_ARGS=(
  DEIMCriterion.losses=[mal,boxes,local,nwd]
  DEIMCriterion.weight_dict.loss_nwd=0.50
  DEIMCriterion.nwd_normalizer=0.08
)

ANATOMY_ARGS=(
  DEIMCriterion.use_fdi_anatomy_loss=True
  DEIMCriterion.weight_dict.loss_fdi_factor=0.12
  DEIMCriterion.weight_dict.loss_fdi_unique=0.02
  DEIMCriterion.weight_dict.loss_fdi_dent=0.03
)

BEST_ENCODER_ARGS=(
  HybridEncoder.use_cmfa_fusion=True
  HybridEncoder.use_hooa_attention=True
)

MSCK_ARGS=(HybridEncoder.use_msck_refiner=True)
CSAF_FILTER_ARGS=(HybridEncoder.use_csaf_filter=True)
CSAF_DMSSP_ARGS=(HybridEncoder.use_csaf_dmssp=True)
RDVQ_ARGS=(HybridEncoder.use_rdvq_refiner=True)
DATPP_ARGS=(HybridEncoder.use_datpp_refiner=True)
CLUSTER_ARGS=(DFINETransformer.use_cluster_query_refiner=True)
RECOVERY_ARGS=(DFINETransformer.use_query_recovery_memory=True)
DCSAM_ARGS=(HybridEncoder.use_dcsam_dual_consistency=True)
IQP_ARGS=(HybridEncoder.use_iqp_quality_prior=True)
DFORMER_ARGS=(HybridEncoder.use_dformer_geometry_prior=True)
CDML_ARGS=(HybridEncoder.use_cdml_domain_adapter=True)
EFRUF_ARGS=(HybridEncoder.use_efruf_gradient_fusion=True)
SINF_ARGS=(HybridEncoder.use_sinf_spatial_field=True)
SMWG_ARGS=(HybridEncoder.use_smwg_spectral_wavelet=True)
BILALORA_ARGS=(HybridEncoder.use_bilalora_layer_adapter=True)
SURMT_ARGS=(HybridEncoder.use_surmt_structure_bridge=True)
MAV_ARGS=(HybridEncoder.use_mav_sparse_interaction=True)
PFRPN_ARGS=(DFINETransformer.use_pfrpn_query_adapter=True)
TEMPLATE_ARGS=(DFINETransformer.use_tooth_template_prior=True)
AREGION_ARGS=(DFINETransformer.use_anatomical_region_prior=True)
TOOTH_SIZE_ARGS=(DEIMCriterion.use_tooth_size_prior_loss=True)
CNM_ARGS=(HybridEncoder.use_cnm_ode_refiner=True)
CLDYN_ARGS=(HybridEncoder.use_cldyn_rsc=True)
CED_ARGS=(HybridEncoder.use_ced_cascade_refiner=True)
DSFED_ARGS=(DEIMCriterion.use_dsfed_dual_scale_kd=True)
SERIAL_OUTS_ARGS=(HybridEncoder.paper_refine_apply=outs)

ORDER=(
  abl39_nwd_loss
  abl40_fdi_spatial_prior
  abl41_fdi_anatomy_loss
  abl42_cmfa_hooa_nwd
  abl43_cmfa_hooa_fdi_prior
  abl44_cmfa_hooa_anatomy_loss
  abl45_cmfa_hooa_nwd_fdi_prior
  abl46_cmfa_hooa_all_priors
  abl47_msck_refiner
  abl48_csaf_filter
  abl49_csaf_dmssp
  abl50_rdvq_refiner
  abl51_datpp_refiner
  abl52_cluster_query_refiner
  abl53_query_recovery_memory
  abl54_cmfa_hooa_msck_csaf_filter
  abl55_cmfa_hooa_csaf_dmssp_rdvq
  abl56_cmfa_hooa_datpp_cluster_recovery
  abl57_cmfa_hooa_all_external
  abl58_cmfa_hooa_external_fdi_priors
  abl59_cnm_ode_refiner
  abl60_cldyn_rsc
  abl61_ced_cascade_refiner
  abl62_dsfed_dual_scale_kd
  abl63_cmfa_hooa_cnm_ced
  abl64_cmfa_hooa_cldyn_dsfed
  abl65_cmfa_hooa_cnm_cldyn_ced
  abl66_cmfa_hooa_paper_modules_dsfed_priors
  abl67_cnm_ode_refiner_v2
  abl68_cldyn_rsc_v2
  abl69_ced_cascade_refiner_v2
  abl70_dsfed_dual_scale_kd_v2
  abl71_cmfa_hooa_cnm_cldyn_proj
  abl72_cmfa_hooa_ced_proj
  abl73_cmfa_hooa_paper_modules_proj
  abl74_cmfa_hooa_paper_modules_dsfed_v2
  abl75_dcsam_dual_consistency
  abl76_iqp_quality_prior
  abl77_dformer_geometry_prior
  abl78_cdml_domain_adapter
  abl79_pfrpn_query_adapter
  abl80_anatomical_region_prior
  abl81_tooth_size_prior_loss
  abl82_cmfa_hooa_dental_priors_v3
  abl83_cmfa_hooa_external_priors_v3
  abl84_cmfa_hooa_paper_modules_dsfed_priors_v3
  abl85_efruf_gradient_fusion
  abl86_sinf_spatial_field
  abl87_smwg_spectral_wavelet
  abl88_tooth_template_prior
  abl89_cmfa_hooa_frequency_field_v4
  abl90_cmfa_hooa_template_dental_v4
  abl91_cmfa_hooa_all_new_20260512
  abl92_cmfa_hooa_efruf_serial
  abl93_cmfa_hooa_efruf_sinf_serial
  abl94_cmfa_hooa_efruf_sinf_smwg_serial
  abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial
  abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial
  abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial
  abl98_cmfa_hooa_sinf_serial
  abl99_cmfa_hooa_smwg_serial
  abl100_cmfa_hooa_surmt_serial
  abl101_cmfa_hooa_sinf_smwg_serial
  abl102_cmfa_hooa_sinf_surmt_serial
  abl103_cmfa_hooa_smwg_surmt_serial
  abl104_cmfa_hooa_sinf_smwg_surmt_serial
  abl105_cmfa_hooa_sinf_smwg_cedv2_serial
)

should_run() {
  local name="$1"
  if [[ -n "${ONLY_EXPS}" ]]; then
    local normalized=" ${ONLY_EXPS//,/ } "
    [[ "${normalized}" == *" ${name} "* ]]
    return
  fi

  local start_idx=-1
  local cur_idx=-1
  local i
  for i in "${!ORDER[@]}"; do
    if [[ "${ORDER[$i]}" == "${START_FROM}" ]]; then
      start_idx="$i"
    fi
    if [[ "${ORDER[$i]}" == "${name}" ]]; then
      cur_idx="$i"
    fi
  done
  if [[ "${start_idx}" -lt 0 || "${cur_idx}" -lt 0 ]]; then
    echo "Invalid START_FROM=${START_FROM}" >&2
    exit 1
  fi
  [[ "${cur_idx}" -ge "${start_idx}" ]]
}

refresh_summary() {
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --base-output-dir "${ROOT_DIR}/${BASE_OUTPUT_DIR}" --run-scope "${SUMMARY_RUN_SCOPE}"
}

run_exp() {
  local name="$1"
  local exp_output="${ROOT_DIR}/${BASE_OUTPUT_DIR}/${name}"
  shift
  echo "=============================="
  echo "Running ${name}"
  echo "=============================="

  if [[ -e "${exp_output}/log.txt" || -e "${exp_output}/best_stg1.pth" || -e "${exp_output}/best_stg2.pth" || -e "${exp_output}/best.pth" ]]; then
    if [[ "${RERUN_POLICY}" == "archive" ]]; then
      local stamp
      stamp="$(date +%Y%m%d_%H%M%S)"
      local archive_dir="${ROOT_DIR}/${BASE_OUTPUT_DIR}/_rerun_backups/${stamp}/${name}"
      mkdir -p "$(dirname "${archive_dir}")"
      mv "${exp_output}" "${archive_dir}"
      echo "Archived previous ${name} output to ${archive_dir}"
    elif [[ "${RERUN_POLICY}" == "overwrite" ]]; then
      rm -rf "${exp_output}"
      echo "Removed previous ${name} output"
    elif [[ "${RERUN_POLICY}" == "append" ]]; then
      echo "Appending to existing ${name} output"
    else
      echo "Invalid RERUN_POLICY=${RERUN_POLICY}; use archive, overwrite, or append" >&2
      exit 1
    fi
  fi

  PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
  DETERMINISTIC="${DETERMINISTIC}" \
  CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    "$@" \
    --output-dir "${exp_output}"

  if [[ "${KEEP_ONLY_BEST}" == "1" && -d "${exp_output}" ]]; then
    find "${exp_output}" -type f -name '*.pth' \
      ! -name 'best_stg1.pth' \
      ! -name 'best_stg2.pth' \
      ! -name 'best.pth' \
      -delete
  fi

  refresh_summary
}

cd "${ROOT_DIR}"
mkdir -p "${BASE_OUTPUT_DIR}"
refresh_summary

if should_run abl39_nwd_loss; then
  run_exp abl39_nwd_loss "${NWD_ARGS[@]}"
fi
if should_run abl40_fdi_spatial_prior; then
  run_exp abl40_fdi_spatial_prior DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl41_fdi_anatomy_loss; then
  run_exp abl41_fdi_anatomy_loss "${ANATOMY_ARGS[@]}"
fi
if should_run abl42_cmfa_hooa_nwd; then
  run_exp abl42_cmfa_hooa_nwd "${BEST_ENCODER_ARGS[@]}" "${NWD_ARGS[@]}"
fi
if should_run abl43_cmfa_hooa_fdi_prior; then
  run_exp abl43_cmfa_hooa_fdi_prior "${BEST_ENCODER_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl44_cmfa_hooa_anatomy_loss; then
  run_exp abl44_cmfa_hooa_anatomy_loss "${BEST_ENCODER_ARGS[@]}" "${ANATOMY_ARGS[@]}"
fi
if should_run abl45_cmfa_hooa_nwd_fdi_prior; then
  run_exp abl45_cmfa_hooa_nwd_fdi_prior "${BEST_ENCODER_ARGS[@]}" "${NWD_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl46_cmfa_hooa_all_priors; then
  run_exp abl46_cmfa_hooa_all_priors "${BEST_ENCODER_ARGS[@]}" "${NWD_ARGS[@]}" "${ANATOMY_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True
fi

if should_run abl47_msck_refiner; then
  run_exp abl47_msck_refiner "${MSCK_ARGS[@]}"
fi
if should_run abl48_csaf_filter; then
  run_exp abl48_csaf_filter "${CSAF_FILTER_ARGS[@]}"
fi
if should_run abl49_csaf_dmssp; then
  run_exp abl49_csaf_dmssp "${CSAF_DMSSP_ARGS[@]}"
fi
if should_run abl50_rdvq_refiner; then
  run_exp abl50_rdvq_refiner "${RDVQ_ARGS[@]}"
fi
if should_run abl51_datpp_refiner; then
  run_exp abl51_datpp_refiner "${DATPP_ARGS[@]}"
fi
if should_run abl52_cluster_query_refiner; then
  run_exp abl52_cluster_query_refiner "${CLUSTER_ARGS[@]}"
fi
if should_run abl53_query_recovery_memory; then
  run_exp abl53_query_recovery_memory "${RECOVERY_ARGS[@]}"
fi
if should_run abl54_cmfa_hooa_msck_csaf_filter; then
  run_exp abl54_cmfa_hooa_msck_csaf_filter "${BEST_ENCODER_ARGS[@]}" "${MSCK_ARGS[@]}" "${CSAF_FILTER_ARGS[@]}"
fi
if should_run abl55_cmfa_hooa_csaf_dmssp_rdvq; then
  run_exp abl55_cmfa_hooa_csaf_dmssp_rdvq "${BEST_ENCODER_ARGS[@]}" "${CSAF_DMSSP_ARGS[@]}" "${RDVQ_ARGS[@]}"
fi
if should_run abl56_cmfa_hooa_datpp_cluster_recovery; then
  run_exp abl56_cmfa_hooa_datpp_cluster_recovery "${BEST_ENCODER_ARGS[@]}" "${DATPP_ARGS[@]}" "${CLUSTER_ARGS[@]}" "${RECOVERY_ARGS[@]}"
fi
if should_run abl57_cmfa_hooa_all_external; then
  run_exp abl57_cmfa_hooa_all_external "${BEST_ENCODER_ARGS[@]}" "${MSCK_ARGS[@]}" "${CSAF_FILTER_ARGS[@]}" "${CSAF_DMSSP_ARGS[@]}" "${RDVQ_ARGS[@]}" "${DATPP_ARGS[@]}" "${CLUSTER_ARGS[@]}" "${RECOVERY_ARGS[@]}"
fi
if should_run abl58_cmfa_hooa_external_fdi_priors; then
  run_exp abl58_cmfa_hooa_external_fdi_priors "${BEST_ENCODER_ARGS[@]}" "${MSCK_ARGS[@]}" "${CSAF_FILTER_ARGS[@]}" "${CSAF_DMSSP_ARGS[@]}" "${RDVQ_ARGS[@]}" "${DATPP_ARGS[@]}" "${CLUSTER_ARGS[@]}" "${RECOVERY_ARGS[@]}" "${NWD_ARGS[@]}" "${ANATOMY_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl59_cnm_ode_refiner; then
  run_exp abl59_cnm_ode_refiner "${CNM_ARGS[@]}"
fi
if should_run abl60_cldyn_rsc; then
  run_exp abl60_cldyn_rsc "${CLDYN_ARGS[@]}"
fi
if should_run abl61_ced_cascade_refiner; then
  run_exp abl61_ced_cascade_refiner "${CED_ARGS[@]}"
fi
if should_run abl62_dsfed_dual_scale_kd; then
  run_exp abl62_dsfed_dual_scale_kd "${DSFED_ARGS[@]}"
fi
if should_run abl63_cmfa_hooa_cnm_ced; then
  run_exp abl63_cmfa_hooa_cnm_ced "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CED_ARGS[@]}"
fi
if should_run abl64_cmfa_hooa_cldyn_dsfed; then
  run_exp abl64_cmfa_hooa_cldyn_dsfed "${BEST_ENCODER_ARGS[@]}" "${CLDYN_ARGS[@]}" "${DSFED_ARGS[@]}"
fi
if should_run abl65_cmfa_hooa_cnm_cldyn_ced; then
  run_exp abl65_cmfa_hooa_cnm_cldyn_ced "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}"
fi
if should_run abl66_cmfa_hooa_paper_modules_dsfed_priors; then
  run_exp abl66_cmfa_hooa_paper_modules_dsfed_priors "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}" "${DSFED_ARGS[@]}" "${NWD_ARGS[@]}" "${ANATOMY_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True
fi
if should_run abl67_cnm_ode_refiner_v2; then
  run_exp abl67_cnm_ode_refiner_v2 "${CNM_ARGS[@]}"
fi
if should_run abl68_cldyn_rsc_v2; then
  run_exp abl68_cldyn_rsc_v2 "${CLDYN_ARGS[@]}"
fi
if should_run abl69_ced_cascade_refiner_v2; then
  run_exp abl69_ced_cascade_refiner_v2 "${CED_ARGS[@]}"
fi
if should_run abl70_dsfed_dual_scale_kd_v2; then
  run_exp abl70_dsfed_dual_scale_kd_v2 "${DSFED_ARGS[@]}"
fi
if should_run abl71_cmfa_hooa_cnm_cldyn_proj; then
  run_exp abl71_cmfa_hooa_cnm_cldyn_proj "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}"
fi
if should_run abl72_cmfa_hooa_ced_proj; then
  run_exp abl72_cmfa_hooa_ced_proj "${BEST_ENCODER_ARGS[@]}" "${CED_ARGS[@]}"
fi
if should_run abl73_cmfa_hooa_paper_modules_proj; then
  run_exp abl73_cmfa_hooa_paper_modules_proj "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}"
fi
if should_run abl74_cmfa_hooa_paper_modules_dsfed_v2; then
  run_exp abl74_cmfa_hooa_paper_modules_dsfed_v2 "${BEST_ENCODER_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}" "${DSFED_ARGS[@]}"
fi
if should_run abl75_dcsam_dual_consistency; then
  run_exp abl75_dcsam_dual_consistency "${DCSAM_ARGS[@]}"
fi
if should_run abl76_iqp_quality_prior; then
  run_exp abl76_iqp_quality_prior "${IQP_ARGS[@]}"
fi
if should_run abl77_dformer_geometry_prior; then
  run_exp abl77_dformer_geometry_prior "${DFORMER_ARGS[@]}"
fi
if should_run abl78_cdml_domain_adapter; then
  run_exp abl78_cdml_domain_adapter "${CDML_ARGS[@]}"
fi
if should_run abl79_pfrpn_query_adapter; then
  run_exp abl79_pfrpn_query_adapter "${PFRPN_ARGS[@]}"
fi
if should_run abl80_anatomical_region_prior; then
  run_exp abl80_anatomical_region_prior "${AREGION_ARGS[@]}"
fi
if should_run abl81_tooth_size_prior_loss; then
  run_exp abl81_tooth_size_prior_loss "${TOOTH_SIZE_ARGS[@]}"
fi
if should_run abl82_cmfa_hooa_dental_priors_v3; then
  run_exp abl82_cmfa_hooa_dental_priors_v3 "${BEST_ENCODER_ARGS[@]}" "${AREGION_ARGS[@]}" "${TOOTH_SIZE_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True "${ANATOMY_ARGS[@]}"
fi
if should_run abl83_cmfa_hooa_external_priors_v3; then
  run_exp abl83_cmfa_hooa_external_priors_v3 "${BEST_ENCODER_ARGS[@]}" "${DCSAM_ARGS[@]}" "${IQP_ARGS[@]}" "${DFORMER_ARGS[@]}" "${CDML_ARGS[@]}" "${PFRPN_ARGS[@]}"
fi
if should_run abl84_cmfa_hooa_paper_modules_dsfed_priors_v3; then
  run_exp abl84_cmfa_hooa_paper_modules_dsfed_priors_v3 "${BEST_ENCODER_ARGS[@]}" "${DCSAM_ARGS[@]}" "${IQP_ARGS[@]}" "${DFORMER_ARGS[@]}" "${CDML_ARGS[@]}" "${PFRPN_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}" "${DSFED_ARGS[@]}" "${NWD_ARGS[@]}" "${ANATOMY_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True "${AREGION_ARGS[@]}" "${TOOTH_SIZE_ARGS[@]}"
fi
if should_run abl85_efruf_gradient_fusion; then
  run_exp abl85_efruf_gradient_fusion "${EFRUF_ARGS[@]}"
fi
if should_run abl86_sinf_spatial_field; then
  run_exp abl86_sinf_spatial_field "${SINF_ARGS[@]}"
fi
if should_run abl87_smwg_spectral_wavelet; then
  run_exp abl87_smwg_spectral_wavelet "${SMWG_ARGS[@]}"
fi
if should_run abl88_tooth_template_prior; then
  run_exp abl88_tooth_template_prior "${TEMPLATE_ARGS[@]}"
fi
if should_run abl89_cmfa_hooa_frequency_field_v4; then
  run_exp abl89_cmfa_hooa_frequency_field_v4 "${BEST_ENCODER_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}"
fi
if should_run abl90_cmfa_hooa_template_dental_v4; then
  run_exp abl90_cmfa_hooa_template_dental_v4 "${BEST_ENCODER_ARGS[@]}" "${TEMPLATE_ARGS[@]}" "${AREGION_ARGS[@]}" "${TOOTH_SIZE_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True "${ANATOMY_ARGS[@]}"
fi
if should_run abl91_cmfa_hooa_all_new_20260512; then
  run_exp abl91_cmfa_hooa_all_new_20260512 "${BEST_ENCODER_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${TEMPLATE_ARGS[@]}" "${DCSAM_ARGS[@]}" "${IQP_ARGS[@]}" "${DFORMER_ARGS[@]}" "${CDML_ARGS[@]}" "${PFRPN_ARGS[@]}" "${CNM_ARGS[@]}" "${CLDYN_ARGS[@]}" "${CED_ARGS[@]}" "${DSFED_ARGS[@]}" "${NWD_ARGS[@]}" "${ANATOMY_ARGS[@]}" DFINETransformer.use_fdi_spatial_prior=True "${AREGION_ARGS[@]}" "${TOOTH_SIZE_ARGS[@]}"
fi
if should_run abl92_cmfa_hooa_efruf_serial; then
  run_exp abl92_cmfa_hooa_efruf_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}"
fi
if should_run abl93_cmfa_hooa_efruf_sinf_serial; then
  run_exp abl93_cmfa_hooa_efruf_sinf_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}"
fi
if should_run abl94_cmfa_hooa_efruf_sinf_smwg_serial; then
  run_exp abl94_cmfa_hooa_efruf_sinf_smwg_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}"
fi
if should_run abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial; then
  run_exp abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${BILALORA_ARGS[@]}"
fi
if should_run abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial; then
  run_exp abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${BILALORA_ARGS[@]}" "${SURMT_ARGS[@]}"
fi
if should_run abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial; then
  run_exp abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${EFRUF_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${BILALORA_ARGS[@]}" "${SURMT_ARGS[@]}" "${MAV_ARGS[@]}"
fi
if should_run abl98_cmfa_hooa_sinf_serial; then
  run_exp abl98_cmfa_hooa_sinf_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SINF_ARGS[@]}"
fi
if should_run abl99_cmfa_hooa_smwg_serial; then
  run_exp abl99_cmfa_hooa_smwg_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SMWG_ARGS[@]}"
fi
if should_run abl100_cmfa_hooa_surmt_serial; then
  run_exp abl100_cmfa_hooa_surmt_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SURMT_ARGS[@]}"
fi
if should_run abl101_cmfa_hooa_sinf_smwg_serial; then
  run_exp abl101_cmfa_hooa_sinf_smwg_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}"
fi
if should_run abl102_cmfa_hooa_sinf_surmt_serial; then
  run_exp abl102_cmfa_hooa_sinf_surmt_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SINF_ARGS[@]}" "${SURMT_ARGS[@]}"
fi
if should_run abl103_cmfa_hooa_smwg_surmt_serial; then
  run_exp abl103_cmfa_hooa_smwg_surmt_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SMWG_ARGS[@]}" "${SURMT_ARGS[@]}"
fi
if should_run abl104_cmfa_hooa_sinf_smwg_surmt_serial; then
  run_exp abl104_cmfa_hooa_sinf_smwg_surmt_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${SURMT_ARGS[@]}"
fi
if should_run abl105_cmfa_hooa_sinf_smwg_cedv2_serial; then
  run_exp abl105_cmfa_hooa_sinf_smwg_cedv2_serial "${BEST_ENCODER_ARGS[@]}" "${SERIAL_OUTS_ARGS[@]}" "${SINF_ARGS[@]}" "${SMWG_ARGS[@]}" "${CED_ARGS[@]}"
fi

refresh_summary
