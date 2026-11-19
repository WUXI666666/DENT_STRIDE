#!/home/wuxi/miniconda3/envs/deimv2/bin/python
import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    ("abl00_baseline", "Baseline", "No extra module or method"),
    ("abl01_hfsa", "HFSA", "Enable HybridEncoder.use_hfsa"),
    ("abl02_wav", "Wav", "Enable HybridEncoder.use_wav_enhance"),
    ("abl03_stat_noise", "StatNoise", "Enable HybridEncoder.use_stat_denoise"),
    ("abl04_mpf", "MPF", "Enable HybridEncoder.use_multipath_fusion"),
    ("abl05_dynamic_rf", "DynamicRF", "Enable HybridEncoder.use_dynamic_rf"),
    ("abl06_mpc", "MPC", "Enable HybridEncoder.use_mpc_multiscale_perception"),
    ("abl07_pqg_hf", "PQG-HF", "Enable HybridEncoder.use_pqg_high_frequency"),
    ("abl08_cmfa", "CMFA", "Enable HybridEncoder.use_cmfa_fusion"),
    ("abl09_uni_gate", "UniGate", "Enable DFINETransformer.use_uni_gate + infer"),
    ("abl10_qgp", "QGP", "Enable DFINETransformer.use_qgp with qgp_layers=[1,2]"),
    ("abl11_pgr", "PGR", "Enable DFINETransformer.use_pgr_roi_reasoning"),
    ("abl12_cmfa_mpf", "CMFA+MPF", "Enable HybridEncoder.use_cmfa_fusion + use_multipath_fusion"),
    ("abl13_cmfa_pgr", "CMFA+PGR", "Enable HybridEncoder.use_cmfa_fusion + DFINETransformer.use_pgr_roi_reasoning"),
    ("abl14_mpf_pgr", "MPF+PGR", "Enable HybridEncoder.use_multipath_fusion + DFINETransformer.use_pgr_roi_reasoning"),
    ("abl15_cmfa_mpf_pgr", "CMFA+MPF+PGR", "Enable HybridEncoder.use_cmfa_fusion + use_multipath_fusion + DFINETransformer.use_pgr_roi_reasoning"),
    ("abl16_stat_noise_dynamic_rf", "StatNoise+DynamicRF", "Enable HybridEncoder.use_stat_denoise + use_dynamic_rf"),
    ("abl17_stat_noise_mpc", "StatNoise+MPC", "Enable HybridEncoder.use_stat_denoise + use_mpc_multiscale_perception"),
    ("abl18_pfesa", "PFESA", "Enable HybridEncoder.use_pfesa"),
    ("abl19_ucan_lkd", "UCAN-LKD", "Enable HybridEncoder.use_ucan_lkd"),
    ("abl20_ucm_uncertainty", "UCM-Uncertainty", "Enable HybridEncoder.use_ucm_uncertainty"),
    ("abl21_retinex_prior", "RetinexPrior", "Enable HybridEncoder.use_retinex_prior"),
    ("abl22_cmfa_pfesa", "CMFA+PFESA", "Enable HybridEncoder.use_cmfa_fusion + use_pfesa"),
    ("abl23_mpf_pfesa", "MPF+PFESA", "Enable HybridEncoder.use_multipath_fusion + use_pfesa"),
    ("abl24_cmfa_ucan_lkd", "CMFA+UCAN-LKD", "Enable HybridEncoder.use_cmfa_fusion + use_ucan_lkd"),
    ("abl25_pfesa_ucm_uncertainty", "PFESA+UCM-Uncertainty", "Enable HybridEncoder.use_pfesa + use_ucm_uncertainty"),
    ("abl26_cmfa_pfesa_ucan_lkd", "CMFA+PFESA+UCAN-LKD", "Enable HybridEncoder.use_cmfa_fusion + use_pfesa + use_ucan_lkd"),
    ("abl27_cmfa_pfesa_ucm_uncertainty", "CMFA+PFESA+UCM-Uncertainty", "Enable HybridEncoder.use_cmfa_fusion + use_pfesa + use_ucm_uncertainty"),
    ("abl28_fdi_factor_prior", "FDI-FactorPrior", "Enable trainable quadrant/tooth-position/dentition factor heads"),
    ("abl29_cmfa_ucm_uncertainty", "CMFA+UCM-Uncertainty", "Enable HybridEncoder.use_cmfa_fusion + use_ucm_uncertainty"),
    ("abl30_cmfa_ucm_fdi_factor", "CMFA+UCM+FDI-FactorPrior", "Enable CMFA + UCM-Uncertainty + trainable FDI factor prior"),
    ("abl31_hooa_attention", "HOOA", "Enable HybridEncoder.use_hooa_attention"),
    ("abl32_multinex_prior", "MultiNeX-Prior", "Enable HybridEncoder.use_multinex_prior"),
    ("abl33_spegc_prompt_graph", "SPEGC-PromptGraph", "Enable DFINETransformer.use_spegc_prompt_graph"),
    ("abl34_cmfa_hooa", "CMFA+HOOA", "Enable CMFA fusion + HOOA"),
    ("abl35_ucm_hooa", "UCM+HOOA", "Enable UCM uncertainty + HOOA"),
    ("abl36_cmfa_ucm_spegc", "CMFA+UCM+SPEGC", "Enable CMFA + UCM uncertainty + SPEGC prompt graph"),
    ("abl37_hooa_multinex", "HOOA+MultiNeX", "Enable HOOA + MultiNeX prior"),
    ("abl38_cmfa_ucm_hooa_spegc", "CMFA+UCM+HOOA+SPEGC", "Enable current best encoder pair plus HOOA and SPEGC prompt graph"),
    ("abl39_nwd_loss", "NWD-Loss", "Add normalized Gaussian Wasserstein distance box loss"),
    ("abl40_fdi_spatial_prior", "FDI-SpatialPrior", "Enable FDI jaw/neighbor/unique logit prior in decoder"),
    ("abl41_fdi_anatomy_loss", "FDI-AnatomyLoss", "Enable final-layer FDI factor and uniqueness anatomy losses"),
    ("abl42_cmfa_hooa_nwd", "CMFA+HOOA+NWD", "Add NWD loss to the current strongest CMFA+HOOA branch"),
    ("abl43_cmfa_hooa_fdi_prior", "CMFA+HOOA+FDI-SpatialPrior", "Add FDI spatial logit prior to CMFA+HOOA"),
    ("abl44_cmfa_hooa_anatomy_loss", "CMFA+HOOA+FDI-AnatomyLoss", "Add final FDI anatomy losses to CMFA+HOOA"),
    ("abl45_cmfa_hooa_nwd_fdi_prior", "CMFA+HOOA+NWD+FDI-SpatialPrior", "Combine NWD loss with FDI spatial logit prior"),
    ("abl46_cmfa_hooa_all_priors", "CMFA+HOOA+AllPriors", "Combine NWD, FDI spatial prior, and final anatomy losses"),
    ("abl47_msck_refiner", "MSCK-Refiner", "Add MSCK-Net inspired Chinese-knot multi-scale strip convolution refiner"),
    ("abl48_csaf_filter", "CSAF-Filter", "Add CSAFNet context-aware semantic adaptive filtering with wavelet-detail fallback"),
    ("abl49_csaf_dmssp", "CSAF-DMSSP", "Add CSAFNet dense multi-scale spatial semantic perception refiner"),
    ("abl50_rdvq_refiner", "RDVQ-Codebook", "Add differentiable vector-quantization codebook feature refiner"),
    ("abl51_datpp_refiner", "DAT++-Refiner", "Add DAT++ style grouped deformable attention on selected FPN levels"),
    ("abl52_cluster_query_refiner", "ClusterQueryGraph", "Add cluster-wise graph transformer over decoder main queries"),
    ("abl53_query_recovery_memory", "QueryRecovery", "Add decoder query recovery memory against iterative query drift"),
    ("abl54_cmfa_hooa_msck_csaf_filter", "CMFA+HOOA+MSCK+CSAF-Filter", "Combine strongest encoder branch with MSCK and CSAF adaptive filtering"),
    ("abl55_cmfa_hooa_csaf_dmssp_rdvq", "CMFA+HOOA+DMSSP+RDVQ", "Combine strongest encoder branch with CSAF-DMSSP and RDVQ codebook refinement"),
    ("abl56_cmfa_hooa_datpp_cluster_recovery", "CMFA+HOOA+DAT+++Cluster+Recovery", "Combine deformable feature refinement with cluster query graph and recovery memory"),
    ("abl57_cmfa_hooa_all_external", "CMFA+HOOA+AllExternal", "Combine all externally inspired modules without FDI/NWD priors"),
    ("abl58_cmfa_hooa_external_fdi_priors", "CMFA+HOOA+External+AllPriors", "Combine all external modules with NWD, FDI spatial prior, and anatomy losses"),
    ("abl59_cnm_ode_refiner", "CNM-ODE", "Add CNM-UNet inspired continuous neural-memory ODE feature refiner"),
    ("abl60_cldyn_rsc", "CLDyN-RSC", "Add CLDyN RSC/A2SI basis-bank semantic compensation refiner"),
    ("abl61_ced_cascade_refiner", "CED-Cascade", "Add CEDNet cascade encoder-decoder multi-scale feature refiner"),
    ("abl62_dsfed_dual_scale_kd", "DSFed-DualScaleKD", "Add DSFedMed inspired dual-scale distillation from final decoder to lightweight heads"),
    ("abl63_cmfa_hooa_cnm_ced", "CMFA+HOOA+CNM+CED", "Combine current strongest encoder branch with CNM-ODE and CED cascade refiners"),
    ("abl64_cmfa_hooa_cldyn_dsfed", "CMFA+HOOA+CLDyN+DSFed", "Combine task-aware semantic compensation with dual-scale distillation"),
    ("abl65_cmfa_hooa_cnm_cldyn_ced", "CMFA+HOOA+CNM+CLDyN+CED", "Combine the three paper-inspired encoder refiners"),
    ("abl66_cmfa_hooa_paper_modules_dsfed_priors", "CMFA+HOOA+PaperModules+KD+Priors", "Combine CNM-ODE, CLDyN-RSC, CED cascade, DSFed KD, NWD, FDI spatial prior, and anatomy losses"),
    ("abl67_cnm_ode_refiner_v2", "CNM-ODE-v2", "CNM-ODE moved to projected features with non-zero ODE projection and stronger residual scale"),
    ("abl68_cldyn_rsc_v2", "CLDyN-RSC-v2", "CLDyN RSC moved to projected features with active zero-conv replacement and stronger semantic residual"),
    ("abl69_ced_cascade_refiner_v2", "CED-Cascade-v2", "CED cascade moved before FPN/PAN with one bottlenecked cascade stage to reduce overfitting"),
    ("abl70_dsfed_dual_scale_kd_v2", "DSFed-DualScaleKD-v2", "DSFed KD changed to sigmoid reliable-query dual-scale consistency with a light final-head consensus term"),
    ("abl71_cmfa_hooa_cnm_cldyn_proj", "CMFA+HOOA+CNM+CLDyN-v2", "Combine strongest encoder branch with corrected CNM and CLDyN projected-feature refiners"),
    ("abl72_cmfa_hooa_ced_proj", "CMFA+HOOA+CED-v2", "Combine strongest encoder branch with corrected projected-feature CED cascade"),
    ("abl73_cmfa_hooa_paper_modules_proj", "CMFA+HOOA+PaperModules-v2", "Combine corrected CNM, CLDyN and CED projected-feature refiners"),
    ("abl74_cmfa_hooa_paper_modules_dsfed_v2", "CMFA+HOOA+PaperModules+KD-v2", "Combine corrected paper-inspired encoder refiners with corrected DSFed dual-scale KD"),
    ("abl75_dcsam_dual_consistency", "DC-SAM-DualConsistency", "Add DC-SAM inspired positive/negative prompt and cross-level dual-consistency feature refinement on projected features"),
    ("abl76_iqp_quality_prior", "IQP-QualityPrior", "Add IQPIR inspired no-reference feature quality prior with common/HQ codebook conditioning"),
    ("abl77_dformer_geometry_prior", "DFormer-GeometryPrior", "Add DFormer++ inspired pseudo-arch geometry attention for panoramic dental features"),
    ("abl78_cdml_domain_adapter", "CDML-DomainAffine", "Add CDML inspired first-order domain affine adapter for scanner/exposure shift robustness"),
    ("abl79_pfrpn_query_adapter", "PF-RPN-QueryAdapter", "Add PF-RPN inspired sparse image-aware query adapter with cascade self-prompt and centerness gating"),
    ("abl80_anatomical_region_prior", "ARegion-AnatomicalPrior", "Add anatomical region-guided contrastive logit prior adapted from medical VLM region decoding"),
    ("abl81_tooth_size_prior_loss", "ToothSizePriorLoss", "Add weak pairwise tooth-size ordering prior from dental anatomy"),
    ("abl82_cmfa_hooa_dental_priors_v3", "CMFA+HOOA+DentalPriors-v3", "Combine FDI spatial prior, anatomy loss, anatomical region prior, and tooth-size prior on CMFA+HOOA"),
    ("abl83_cmfa_hooa_external_priors_v3", "CMFA+HOOA+ExternalPriors-v3", "Combine DC-SAM, IQP, DFormer geometry, CDML affine, and PF-RPN query adapter with CMFA+HOOA"),
    ("abl84_cmfa_hooa_paper_modules_dsfed_priors_v3", "CMFA+HOOA+AllPaper+KD+Priors-v3", "Combine v2 paper modules, new external priors, DSFed KD, NWD, and dental anatomy priors"),
    ("abl85_efruf_gradient_fusion", "EFRUF-GradientFusion", "Add EFRUF inspired Sobel gradient-transfer and unfolded feature fusion on projected features"),
    ("abl86_sinf_spatial_field", "SINF-SpatialField", "Add SINF inspired blind-spot continuous spatial implicit field refinement for single-image denoising"),
    ("abl87_smwg_spectral_wavelet", "SMWG-SpectralWavelet", "Add SMWG-DETR inspired Fourier spectral modulation and wavelet/detail-guided multi-scale feature refinement"),
    ("abl88_tooth_template_prior", "ToothTemplatePrior", "Add multi-template tooth-position matching prior over decoder query features and FDI arch locations"),
    ("abl89_cmfa_hooa_frequency_field_v4", "CMFA+HOOA+FreqField-v4", "Combine EFRUF gradient fusion, SINF spatial field, and SMWG spectral-wavelet refinement with CMFA+HOOA"),
    ("abl90_cmfa_hooa_template_dental_v4", "CMFA+HOOA+TemplateDental-v4", "Combine tooth template matching with FDI spatial, anatomical region, tooth-size, and anatomy priors"),
    ("abl91_cmfa_hooa_all_new_20260512", "CMFA+HOOA+AllNew-20260512", "Combine v2 paper modules, abl75-84 modules, EFRUF, SINF, SMWG, template matching, KD, NWD, and dental priors"),
    ("abl92_cmfa_hooa_efruf_serial", "CMFA+HOOA+EFRUF-Serial", "Serial ablation step 1: add EFRUF gradient-transfer fusion after the best CMFA+HOOA encoder"),
    ("abl93_cmfa_hooa_efruf_sinf_serial", "CMFA+HOOA+EFRUF+SINF-Serial", "Serial ablation step 2: add SINF spatial implicit field on top of EFRUF"),
    ("abl94_cmfa_hooa_efruf_sinf_smwg_serial", "CMFA+HOOA+EFRUF+SINF+SMWG-Serial", "Serial ablation step 3: add SMWG Fourier spectral modulation and wavelet-guided refinement"),
    ("abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial", "CMFA+HOOA+Freq+BiLaLoRA-Serial", "Serial ablation step 4: add BiLaLoRA-inspired low-rank layer-positioned adapter"),
    ("abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial", "CMFA+HOOA+Freq+BiLaLoRA+SURMT-Serial", "Serial ablation step 5: add SU-RMT semantic-structure bridge for dental boundary detail"),
    ("abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial", "CMFA+HOOA+All6-Serial", "Serial ablation step 6: add MAVFusion-inspired sparse cross-scale interaction"),
    ("abl98_cmfa_hooa_sinf_serial", "CMFA+HOOA+SINF-Serial", "Follow-up subset ablation: add SINF spatial implicit field to CMFA+HOOA on output features"),
    ("abl99_cmfa_hooa_smwg_serial", "CMFA+HOOA+SMWG-Serial", "Follow-up subset ablation: add SMWG Fourier spectral modulation and wavelet-guided refinement to CMFA+HOOA"),
    ("abl100_cmfa_hooa_surmt_serial", "CMFA+HOOA+SURMT-Serial", "Follow-up subset ablation: add SU-RMT semantic-structure bridge to CMFA+HOOA"),
    ("abl101_cmfa_hooa_sinf_smwg_serial", "CMFA+HOOA+SINF+SMWG-Serial", "Follow-up subset ablation: combine SINF and SMWG on top of CMFA+HOOA"),
    ("abl102_cmfa_hooa_sinf_surmt_serial", "CMFA+HOOA+SINF+SURMT-Serial", "Follow-up subset ablation: combine SINF and SU-RMT on top of CMFA+HOOA"),
    ("abl103_cmfa_hooa_smwg_surmt_serial", "CMFA+HOOA+SMWG+SURMT-Serial", "Follow-up subset ablation: combine SMWG and SU-RMT on top of CMFA+HOOA"),
    ("abl104_cmfa_hooa_sinf_smwg_surmt_serial", "CMFA+HOOA+SINF+SMWG+SURMT-Serial", "Follow-up subset ablation: combine SINF, SMWG, and SU-RMT on top of CMFA+HOOA"),
    ("abl105_cmfa_hooa_sinf_smwg_cedv2_serial", "CMFA+HOOA+SINF+SMWG+CED-v2-Serial", "Follow-up subset ablation: add corrected CED cascade to CMFA+HOOA+SINF+SMWG"),
    ("abl106_phone_tooth_prior_baseline", "Phone-Baseline", "Phone /home/wuxi/coco baseline for factor-prototype-spatial_shape serial ablation"),
    ("abl107_phone_factor", "Phone-Factor", "Phone serial ablation: add factor heads"),
    ("abl108_phone_prototype", "Phone-Prototype", "Phone serial ablation: add learnable tooth appearance prototypes"),
    ("abl109_phone_spatial_shape", "Phone-SpatialShape", "Phone serial ablation: add external spatial and bbox-shape templates"),
    ("abl110_dentex_tooth_prior_baseline", "DENTEX-Baseline", "DENTEX2023 baseline for factor-prototype-spatial_shape serial ablation"),
    ("abl111_dentex_factor", "DENTEX-Factor", "DENTEX serial ablation: add factor heads"),
    ("abl112_dentex_prototype", "DENTEX-Prototype", "DENTEX serial ablation: add learnable tooth appearance prototypes"),
    ("abl113_dentex_spatial_shape", "DENTEX-SpatialShape", "DENTEX serial ablation: add external spatial and bbox-shape templates"),
]


def _add_tooth_prior_branch_rows(start_id, branch_key, branch_label, branch_note):
    stage_specs = [
        ("baseline", "Baseline", "baseline detector"),
        ("factor", "Factor", "add factor heads"),
        ("prototype", "Prototype", "add factor heads and learnable tooth appearance prototypes"),
        ("spatial_shape", "SpatialShape", "add factor/prototype plus spatial, shape, and view priors"),
    ]
    dataset_specs = [
        ("phone", "Phone", "/home/wuxi/coco"),
        ("dentex", "DENTEX", "/home/wuxi/DENTEX2023"),
    ]
    idx = int(start_id)
    for dataset_key, dataset_label, dataset_path in dataset_specs:
        for stage_key, stage_label, stage_note in stage_specs:
            EXPERIMENTS.append((
                f"abl{idx}_{dataset_key}_{branch_key}_{stage_key}",
                f"{dataset_label}-{branch_label}-{stage_label}",
                f"{dataset_label} {dataset_path}: {branch_note}; {stage_note}",
            ))
            idx += 1


_add_tooth_prior_branch_rows(
    114,
    "cmfa_hooa_cedv2",
    "CMFA+HOOA+CED-v2",
    "best previous branch with CMFA, HOOA, and projected CED cascade",
)
_add_tooth_prior_branch_rows(
    122,
    "cmfa_hooa",
    "CMFA+HOOA",
    "best previous encoder pair",
)
_add_tooth_prior_branch_rows(
    130,
    "cmfa",
    "CMFA",
    "standalone cross-modal feature aggregation branch",
)
_add_tooth_prior_branch_rows(
    138,
    "smwg",
    "SMWG-SpectralWavelet",
    "standalone spectral-wavelet frequency/detail branch",
)
_add_tooth_prior_branch_rows(
    146,
    "sefs",
    "SEFS",
    "new structural-entropy frequency-saliency refiner from the 2026 paper/code review",
)

EXPERIMENTS.extend([
    (
        "abl154_coco_rot_equivariant_detail",
        "RotE-EquivariantDetail",
        "Phone /home/wuxi/coco single-module ablation: Rot-E ASISR-inspired transform-equivariant detail refiner",
    ),
    (
        "abl155_coco_dualopt_adamw",
        "DualOpt-AdamW",
        "Phone /home/wuxi/coco single-module ablation: OLOR/DualOpt-inspired real-time decay and initialization rollback optimizer",
    ),
    (
        "abl156_coco_hedp_prompt",
        "HEDP-Prompt",
        "Phone /home/wuxi/coco single-module ablation: hybrid energy-distance prompt feature refiner",
    ),
    (
        "abl157_coco_sampg_sst",
        "SAMPG-SSTScan",
        "Phone /home/wuxi/coco single-module ablation: SAM-prior-guided spatial/spectral scan refiner",
    ),
    (
        "abl158_coco_segd_entropy",
        "SEGD-StructuralEntropy",
        "Phone /home/wuxi/coco single-module ablation: SEGD-inspired structural entropy and degradation-evidence refiner",
    ),
    (
        "abl159_coco_fsdetr_frequency_spatial",
        "FSDETR-FrequencySpatial",
        "Phone /home/wuxi/coco single-module ablation: FSDETR-inspired frequency-spatial small-object feature enhancer",
    ),
    (
        "abl160_coco_df2net_dynamic_fusion",
        "DF2Net-DynamicFusion",
        "Phone /home/wuxi/coco single-module ablation: DF2Net-inspired differential dynamic FPN fusion",
    ),
    (
        "abl161_coco_sdconet_saliency",
        "SDCoNet-Saliency",
        "Phone /home/wuxi/coco single-module ablation: SDCoNet-inspired saliency-driven collaborative feature refiner",
    ),
    (
        "abl162_coco_idgbr_boundary",
        "IDGBR-Boundary",
        "Phone /home/wuxi/coco single-module ablation: IDGBR-inspired semantic-guided boundary refinement",
    ),
])


def fmt_float(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value):
    if value is None:
        return "-"
    return f"{value:+.4f}"


def fmt_params(value):
    if value is None:
        return "-"
    return f"{value / 1_000_000:.2f}M"


def find_best_record(log_path: Path, run_scope: str = "latest"):
    records = []
    if not log_path.exists():
        return None

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            metrics = obj.get("test_coco_eval_bbox")
            if not metrics:
                continue

            record = {
                "epoch": obj.get("epoch"),
                "ap": float(metrics[0]),
                "ap50": float(metrics[1]),
                "ap75": float(metrics[2]),
                "params": int(obj.get("n_parameters") or 0),
            }
            records.append(record)

    if not records:
        return None

    if run_scope == "latest":
        latest = []
        prev_epoch = None
        for record in records:
            epoch = record["epoch"]
            if prev_epoch is not None and epoch is not None and epoch <= prev_epoch:
                latest = []
            latest.append(record)
            prev_epoch = epoch
        records = latest
    elif run_scope != "all":
        raise ValueError(f"run_scope must be 'latest' or 'all', got {run_scope!r}")

    return max(records, key=lambda record: record["ap"])


def infer_best_weight(exp_dir: Path, best):
    if best is None:
        return str(exp_dir / "best.pth")

    candidates = []
    if (exp_dir / "best.pth").exists():
        candidates.append(exp_dir / "best.pth")
    if best["epoch"] is not None and best["epoch"] >= 50:
        candidates.extend([exp_dir / "best_stg2.pth", exp_dir / "best_stg1.pth"])
    else:
        candidates.extend([exp_dir / "best_stg1.pth", exp_dir / "best_stg2.pth"])

    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def tooth_prior_group(exp_name):
    parts = exp_name.split("_")
    if len(parts) < 3 or parts[1] not in ("phone", "dentex"):
        return None

    dataset = parts[1]
    tail = parts[2:]
    if len(tail) >= 2 and tail[-2:] == ["spatial", "shape"]:
        stage = "spatial_shape"
        branch_parts = tail[:-2]
    elif tail and tail[-1] in ("baseline", "factor", "prototype"):
        stage = tail[-1]
        branch_parts = tail[:-1]
    else:
        return None

    if not branch_parts or branch_parts == ["tooth", "prior"]:
        branch = "tooth_prior"
    else:
        branch = "_".join(branch_parts)
    return f"{dataset}:{branch}", stage == "baseline"


def build_rows(base_output_dir: Path, run_scope: str = "latest"):
    rows = []
    baseline_ap = None
    tooth_prior_baselines = {}

    for exp_name, label, note in EXPERIMENTS:
        exp_dir = base_output_dir / exp_name
        log_path = exp_dir / "log.txt"
        best = find_best_record(log_path, run_scope=run_scope)

        row = {
            "experiment": exp_name,
            "label": label,
            "note": note,
            "status": "done" if best else "pending",
            "output_dir": str(exp_dir),
            "log_path": str(log_path),
            "best_weight": infer_best_weight(exp_dir, best),
            "best_epoch": best["epoch"] if best else None,
            "ap": best["ap"] if best else None,
            "ap50": best["ap50"] if best else None,
            "ap75": best["ap75"] if best else None,
            "params": best["params"] if best else None,
        }
        if exp_name == "abl00_baseline" and best:
            baseline_ap = best["ap"]
        group = tooth_prior_group(exp_name)
        if group and group[1] and best:
            tooth_prior_baselines[group[0]] = best["ap"]
        rows.append(row)

    for row in rows:
        ref_ap = baseline_ap
        group = tooth_prior_group(row["experiment"])
        if group and group[0] in tooth_prior_baselines:
            ref_ap = tooth_prior_baselines[group[0]]
        row["delta_ap"] = None if row["ap"] is None or ref_ap is None else row["ap"] - ref_ap

    return rows


def write_json(rows, output_path: Path):
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows, output_path: Path):
    fieldnames = [
        "experiment",
        "label",
        "status",
        "best_epoch",
        "ap",
        "ap50",
        "ap75",
        "delta_ap",
        "params",
        "note",
        "output_dir",
        "log_path",
        "best_weight",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, output_path: Path):
    lines = []
    lines.append("| Experiment | Module/Method | Status | Best AP | AP50 | AP75 | Delta | Best Epoch | Params |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {experiment} | {label} | {status} | {ap} | {ap50} | {ap75} | {delta} | {epoch} | {params} |".format(
                experiment=row["experiment"],
                label=row["label"],
                status=row["status"],
                ap=fmt_float(row["ap"]),
                ap50=fmt_float(row["ap50"]),
                ap75=fmt_float(row["ap75"]),
                delta=fmt_delta(row["delta_ap"]),
                epoch=row["best_epoch"] if row["best_epoch"] is not None else "-",
                params=fmt_params(row["params"]),
            )
        )
    lines.append("")
    lines.append("Notes:")
    for row in rows:
        lines.append(f"- `{row['experiment']}`: {row['note']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-output-dir",
        default="/home/wuxi/Dent-FINE/outputs/ablation_all_new_modules",
    )
    parser.add_argument(
        "--run-scope",
        choices=["latest", "all"],
        default="latest",
        help="Use only the latest run in an appended log, or the best record across all historical runs.",
    )
    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(base_output_dir, run_scope=args.run_scope)
    write_json(rows, base_output_dir / "ablation_summary.json")
    write_csv(rows, base_output_dir / "ablation_summary.csv")
    write_markdown(rows, base_output_dir / "ablation_summary.md")

    done = sum(1 for row in rows if row["status"] == "done")
    print(f"Wrote summary for {done}/{len(rows)} experiments to {base_output_dir}")


if __name__ == "__main__":
    main()
