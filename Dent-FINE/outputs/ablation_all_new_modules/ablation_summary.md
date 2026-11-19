| Experiment | Module/Method | Status | Best AP | AP50 | AP75 | Delta | Best Epoch | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| abl00_baseline | Baseline | done | 0.5343 | 0.8348 | 0.5869 | +0.0000 | 151 | 11.17M |
| abl01_hfsa | HFSA | done | 0.5329 | 0.8370 | 0.5680 | -0.0015 | 128 | 11.85M |
| abl02_wav | Wav | done | 0.5225 | 0.8233 | 0.5783 | -0.0119 | 124 | 11.17M |
| abl03_stat_noise | StatNoise | done | 0.5283 | 0.8317 | 0.5680 | -0.0060 | 121 | 15.13M |
| abl04_mpf | MPF | done | 0.5364 | 0.8345 | 0.5831 | +0.0020 | 144 | 12.40M |
| abl05_dynamic_rf | DynamicRF | done | 0.5215 | 0.8220 | 0.5709 | -0.0129 | 152 | 11.90M |
| abl06_mpc | MPC | done | 0.5283 | 0.8227 | 0.5922 | -0.0060 | 132 | 12.82M |
| abl07_pqg_hf | PQG-HF | done | 0.5272 | 0.8273 | 0.5870 | -0.0072 | 118 | 11.31M |
| abl08_cmfa | CMFA | done | 0.5393 | 0.8534 | 0.5751 | +0.0049 | 154 | 11.17M |
| abl09_uni_gate | UniGate | done | 0.5233 | 0.8186 | 0.5949 | -0.0110 | 154 | 11.17M |
| abl10_qgp | QGP | done | 0.5261 | 0.8316 | 0.5697 | -0.0082 | 134 | 11.70M |
| abl11_pgr | PGR | done | 0.5314 | 0.8312 | 0.5834 | -0.0030 | 142 | 11.63M |
| abl12_cmfa_mpf | CMFA+MPF | done | 0.5354 | 0.8406 | 0.5929 | +0.0011 | 158 | 12.40M |
| abl13_cmfa_pgr | CMFA+PGR | done | 0.5331 | 0.8403 | 0.5756 | -0.0012 | 148 | 11.63M |
| abl14_mpf_pgr | MPF+PGR | done | 0.5244 | 0.8257 | 0.5744 | -0.0099 | 145 | 12.86M |
| abl15_cmfa_mpf_pgr | CMFA+MPF+PGR | done | 0.5321 | 0.8404 | 0.5782 | -0.0023 | 135 | 12.86M |
| abl16_stat_noise_dynamic_rf | StatNoise+DynamicRF | done | 0.5283 | 0.8277 | 0.5787 | -0.0061 | 122 | 15.86M |
| abl17_stat_noise_mpc | StatNoise+MPC | done | 0.5253 | 0.8105 | 0.5939 | -0.0090 | 151 | 16.78M |
| abl18_pfesa | PFESA | done | 0.5324 | 0.8295 | 0.5916 | -0.0019 | 118 | 11.17M |
| abl19_ucan_lkd | UCAN-LKD | done | 0.5331 | 0.8377 | 0.5874 | -0.0012 | 152 | 11.93M |
| abl20_ucm_uncertainty | UCM-Uncertainty | done | 0.5365 | 0.8416 | 0.6000 | +0.0021 | 138 | 11.62M |
| abl21_retinex_prior | RetinexPrior | done | 0.5266 | 0.8129 | 0.5992 | -0.0077 | 139 | 11.27M |
| abl22_cmfa_pfesa | CMFA+PFESA | done | 0.5366 | 0.8393 | 0.5950 | +0.0022 | 125 | 11.17M |
| abl23_mpf_pfesa | MPF+PFESA | done | 0.5310 | 0.8318 | 0.5785 | -0.0033 | 132 | 12.40M |
| abl24_cmfa_ucan_lkd | CMFA+UCAN-LKD | done | 0.5313 | 0.8338 | 0.5903 | -0.0031 | 101 | 11.93M |
| abl25_pfesa_ucm_uncertainty | PFESA+UCM-Uncertainty | done | 0.5351 | 0.8321 | 0.5932 | +0.0008 | 142 | 11.62M |
| abl26_cmfa_pfesa_ucan_lkd | CMFA+PFESA+UCAN-LKD | done | 0.5308 | 0.8404 | 0.5851 | -0.0035 | 146 | 11.93M |
| abl27_cmfa_pfesa_ucm_uncertainty | CMFA+PFESA+UCM-Uncertainty | done | 0.5133 | 0.8150 | 0.5709 | -0.0210 | 56 | 11.62M |
| abl28_fdi_factor_prior | FDI-FactorPrior | done | 0.4993 | 0.8090 | 0.5350 | -0.0350 | 139 | 10.64M |
| abl29_cmfa_ucm_uncertainty | CMFA+UCM-Uncertainty | done | 0.5377 | 0.8331 | 0.6018 | +0.0033 | 148 | 11.61M |
| abl30_cmfa_ucm_fdi_factor | CMFA+UCM+FDI-FactorPrior | done | 0.4993 | 0.8016 | 0.5542 | -0.0350 | 159 | 11.62M |
| abl31_hooa_attention | HOOA | done | 0.5268 | 0.8317 | 0.5696 | -0.0075 | 152 | 11.92M |
| abl32_multinex_prior | MultiNeX-Prior | done | 0.5266 | 0.8255 | 0.5674 | -0.0077 | 133 | 11.01M |
| abl33_spegc_prompt_graph | SPEGC-PromptGraph | done | 0.5293 | 0.8316 | 0.5804 | -0.0051 | 159 | 11.44M |
| abl34_cmfa_hooa | CMFA+HOOA | done | 0.5426 | 0.8455 | 0.5958 | +0.0083 | 143 | 12.45M |
| abl35_ucm_hooa | UCM+HOOA | done | 0.5339 | 0.8243 | 0.6090 | -0.0005 | 155 | 12.38M |
| abl36_cmfa_ucm_spegc | CMFA+UCM+SPEGC | done | 0.5286 | 0.8333 | 0.5779 | -0.0058 | 156 | 12.42M |
| abl37_hooa_multinex | HOOA+MultiNeX | done | 0.5310 | 0.8322 | 0.5833 | -0.0033 | 116 | 12.30M |
| abl38_cmfa_ucm_hooa_spegc | CMFA+UCM+HOOA+SPEGC | done | 0.5243 | 0.8104 | 0.5822 | -0.0100 | 132 | 13.71M |
| abl39_nwd_loss | NWD-Loss | done | 0.5354 | 0.8269 | 0.6126 | +0.0011 | 151 | 10.63M |
| abl40_fdi_spatial_prior | FDI-SpatialPrior | done | 0.5310 | 0.8385 | 0.5726 | -0.0034 | 133 | 10.63M |
| abl41_fdi_anatomy_loss | FDI-AnatomyLoss | done | 0.5362 | 0.8347 | 0.6111 | +0.0019 | 135 | 10.63M |
| abl42_cmfa_hooa_nwd | CMFA+HOOA+NWD | done | 0.5267 | 0.8270 | 0.5834 | -0.0077 | 126 | 12.45M |
| abl43_cmfa_hooa_fdi_prior | CMFA+HOOA+FDI-SpatialPrior | done | 0.5347 | 0.8327 | 0.5995 | +0.0003 | 156 | 12.45M |
| abl44_cmfa_hooa_anatomy_loss | CMFA+HOOA+FDI-AnatomyLoss | done | 0.5296 | 0.8290 | 0.5790 | -0.0048 | 155 | 12.45M |
| abl45_cmfa_hooa_nwd_fdi_prior | CMFA+HOOA+NWD+FDI-SpatialPrior | done | 0.5268 | 0.8217 | 0.5761 | -0.0075 | 158 | 12.45M |
| abl46_cmfa_hooa_all_priors | CMFA+HOOA+AllPriors | done | 0.5347 | 0.8285 | 0.6039 | +0.0003 | 140 | 12.45M |
| abl47_msck_refiner | MSCK-Refiner | done | 0.5303 | 0.8305 | 0.5802 | -0.0040 | 139 | 11.12M |
| abl48_csaf_filter | CSAF-Filter | done | 0.5299 | 0.8287 | 0.5895 | -0.0045 | 95 | 12.14M |
| abl49_csaf_dmssp | CSAF-DMSSP | done | 0.5281 | 0.8275 | 0.5717 | -0.0063 | 143 | 11.69M |
| abl50_rdvq_refiner | RDVQ-Codebook | done | 0.5305 | 0.8249 | 0.6047 | -0.0038 | 156 | 11.12M |
| abl51_datpp_refiner | DAT++-Refiner | done | 0.5287 | 0.8322 | 0.5861 | -0.0056 | 144 | 11.16M |
| abl52_cluster_query_refiner | ClusterQueryGraph | done | 0.5246 | 0.8231 | 0.5759 | -0.0097 | 153 | 12.03M |
| abl53_query_recovery_memory | QueryRecovery | done | 0.5277 | 0.8383 | 0.5688 | -0.0066 | 152 | 11.09M |
| abl54_cmfa_hooa_msck_csaf_filter | CMFA+HOOA+MSCK+CSAF-Filter | done | 0.5388 | 0.8364 | 0.6056 | +0.0044 | 152 | 14.44M |
| abl55_cmfa_hooa_csaf_dmssp_rdvq | CMFA+HOOA+DMSSP+RDVQ | done | 0.5289 | 0.8274 | 0.5907 | -0.0055 | 152 | 14.00M |
| abl56_cmfa_hooa_datpp_cluster_recovery | CMFA+HOOA+DAT+++Cluster+Recovery | done | 0.5354 | 0.8290 | 0.5888 | +0.0011 | 147 | 14.84M |
| abl57_cmfa_hooa_all_external | CMFA+HOOA+AllExternal | done | 0.5348 | 0.8323 | 0.5884 | +0.0004 | 151 | 18.39M |
| abl58_cmfa_hooa_external_fdi_priors | CMFA+HOOA+External+AllPriors | done | 0.5314 | 0.8359 | 0.5765 | -0.0030 | 159 | 18.39M |
| abl59_cnm_ode_refiner | CNM-ODE | done | 0.5262 | 0.8218 | 0.5839 | -0.0082 | 153 | 11.78M |
| abl60_cldyn_rsc | CLDyN-RSC | done | 0.5337 | 0.8310 | 0.6046 | -0.0007 | 134 | 12.13M |
| abl61_ced_cascade_refiner | CED-Cascade | done | 0.5253 | 0.8383 | 0.5839 | -0.0090 | 102 | 18.78M |
| abl62_dsfed_dual_scale_kd | DSFed-DualScaleKD | done | 0.5265 | 0.8207 | 0.5866 | -0.0078 | 148 | 10.63M |
| abl63_cmfa_hooa_cnm_ced | CMFA+HOOA+CNM+CED | done | 0.5319 | 0.8347 | 0.5869 | -0.0025 | 144 | 21.75M |
| abl64_cmfa_hooa_cldyn_dsfed | CMFA+HOOA+CLDyN+DSFed | done | 0.5296 | 0.8313 | 0.5727 | -0.0047 | 158 | 13.95M |
| abl65_cmfa_hooa_cnm_cldyn_ced | CMFA+HOOA+CNM+CLDyN+CED | done | 0.5334 | 0.8407 | 0.5770 | -0.0010 | 154 | 23.26M |
| abl66_cmfa_hooa_paper_modules_dsfed_priors | CMFA+HOOA+PaperModules+KD+Priors | done | 0.5237 | 0.8201 | 0.5724 | -0.0107 | 128 | 23.26M |
| abl67_cnm_ode_refiner_v2 | CNM-ODE-v2 | done | 0.5259 | 0.8188 | 0.6039 | -0.0085 | 147 | 11.78M |
| abl68_cldyn_rsc_v2 | CLDyN-RSC-v2 | done | 0.5270 | 0.8202 | 0.5813 | -0.0073 | 152 | 12.13M |
| abl69_ced_cascade_refiner_v2 | CED-Cascade-v2 | done | 0.5271 | 0.8268 | 0.5901 | -0.0072 | 146 | 14.71M |
| abl70_dsfed_dual_scale_kd_v2 | DSFed-DualScaleKD-v2 | done | 0.5286 | 0.8264 | 0.5846 | -0.0058 | 122 | 10.63M |
| abl71_cmfa_hooa_cnm_cldyn_proj | CMFA+HOOA+CNM+CLDyN-v2 | done | 0.5358 | 0.8351 | 0.6049 | +0.0014 | 128 | 15.10M |
| abl72_cmfa_hooa_ced_proj | CMFA+HOOA+CED-v2 | done | 0.5423 | 0.8393 | 0.6171 | +0.0080 | 145 | 16.52M |
| abl73_cmfa_hooa_paper_modules_proj | CMFA+HOOA+PaperModules-v2 | done | 0.5311 | 0.8358 | 0.5716 | -0.0033 | 156 | 19.18M |
| abl74_cmfa_hooa_paper_modules_dsfed_v2 | CMFA+HOOA+PaperModules+KD-v2 | done | 0.5351 | 0.8311 | 0.6094 | +0.0008 | 148 | 19.18M |
| abl75_dcsam_dual_consistency | DC-SAM-DualConsistency | done | 0.5261 | 0.8324 | 0.5661 | -0.0082 | 153 | 10.98M |
| abl76_iqp_quality_prior | IQP-QualityPrior | done | 0.5220 | 0.8208 | 0.5712 | -0.0123 | 149 | 11.11M |
| abl77_dformer_geometry_prior | DFormer-GeometryPrior | done | 0.5310 | 0.8389 | 0.5987 | -0.0034 | 130 | 11.26M |
| abl78_cdml_domain_adapter | CDML-DomainAffine | done | 0.5291 | 0.8299 | 0.5985 | -0.0053 | 108 | 10.83M |
| abl79_pfrpn_query_adapter | PF-RPN-QueryAdapter | done | 0.5283 | 0.8308 | 0.5757 | -0.0061 | 148 | 11.26M |
| abl80_anatomical_region_prior | ARegion-AnatomicalPrior | done | 0.5298 | 0.8289 | 0.5872 | -0.0045 | 139 | 10.63M |
| abl81_tooth_size_prior_loss | ToothSizePriorLoss | done | 0.5278 | 0.8241 | 0.5807 | -0.0065 | 121 | 10.63M |
| abl82_cmfa_hooa_dental_priors_v3 | CMFA+HOOA+DentalPriors-v3 | done | 0.5246 | 0.8266 | 0.5808 | -0.0097 | 139 | 12.45M |
| abl83_cmfa_hooa_external_priors_v3 | CMFA+HOOA+ExternalPriors-v3 | done | 0.5361 | 0.8441 | 0.6003 | +0.0017 | 153 | 14.74M |
| abl84_cmfa_hooa_paper_modules_dsfed_priors_v3 | CMFA+HOOA+AllPaper+KD+Priors-v3 | done | 0.5299 | 0.8319 | 0.5783 | -0.0044 | 151 | 21.47M |
| abl85_efruf_gradient_fusion | EFRUF-GradientFusion | done | 0.5257 | 0.8218 | 0.5775 | -0.0086 | 145 | 12.28M |
| abl86_sinf_spatial_field | SINF-SpatialField | done | 0.5277 | 0.8205 | 0.5803 | -0.0066 | 127 | 11.50M |
| abl87_smwg_spectral_wavelet | SMWG-SpectralWavelet | done | 0.5368 | 0.8381 | 0.5928 | +0.0024 | 101 | 11.22M |
| abl88_tooth_template_prior | ToothTemplatePrior | done | 0.5291 | 0.8272 | 0.5885 | -0.0053 | 124 | 10.83M |
| abl89_cmfa_hooa_frequency_field_v4 | CMFA+HOOA+FreqField-v4 | done | 0.5274 | 0.8319 | 0.5931 | -0.0069 | 122 | 15.55M |
| abl90_cmfa_hooa_template_dental_v4 | CMFA+HOOA+TemplateDental-v4 | done | 0.5337 | 0.8326 | 0.5914 | -0.0006 | 149 | 12.65M |
| abl91_cmfa_hooa_all_new_20260512 | CMFA+HOOA+AllNew-20260512 | done | 0.5229 | 0.8259 | 0.5724 | -0.0114 | 142 | 24.77M |
| abl92_cmfa_hooa_efruf_serial | CMFA+HOOA+EFRUF-Serial | done | 0.5271 | 0.8273 | 0.5850 | -0.0072 | 139 | 14.09M |
| abl93_cmfa_hooa_efruf_sinf_serial | CMFA+HOOA+EFRUF+SINF-Serial | done | 0.5310 | 0.8290 | 0.5871 | -0.0034 | 148 | 14.96M |
| abl94_cmfa_hooa_efruf_sinf_smwg_serial | CMFA+HOOA+EFRUF+SINF+SMWG-Serial | done | 0.5363 | 0.8436 | 0.5893 | +0.0020 | 115 | 15.55M |
| abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial | CMFA+HOOA+Freq+BiLaLoRA-Serial | done | 0.5220 | 0.8202 | 0.5692 | -0.0124 | 110 | 15.56M |
| abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial | CMFA+HOOA+Freq+BiLaLoRA+SURMT-Serial | done | 0.5353 | 0.8409 | 0.5901 | +0.0009 | 144 | 15.97M |
| abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial | CMFA+HOOA+All6-Serial | done | 0.5239 | 0.8188 | 0.5759 | -0.0104 | 129 | 16.78M |
| abl98_cmfa_hooa_sinf_serial | CMFA+HOOA+SINF-Serial | done | 0.5309 | 0.8332 | 0.5763 | -0.0035 | 144 | 13.31M |
| abl99_cmfa_hooa_smwg_serial | CMFA+HOOA+SMWG-Serial | done | 0.5297 | 0.8276 | 0.5814 | -0.0047 | 121 | 13.03M |
| abl100_cmfa_hooa_surmt_serial | CMFA+HOOA+SURMT-Serial | done | 0.5289 | 0.8271 | 0.5926 | -0.0055 | 128 | 12.85M |
| abl101_cmfa_hooa_sinf_smwg_serial | CMFA+HOOA+SINF+SMWG-Serial | done | 0.5315 | 0.8327 | 0.5773 | -0.0029 | 154 | 13.90M |
| abl102_cmfa_hooa_sinf_surmt_serial | CMFA+HOOA+SINF+SURMT-Serial | done | 0.5271 | 0.8204 | 0.5799 | -0.0072 | 123 | 13.72M |
| abl103_cmfa_hooa_smwg_surmt_serial | CMFA+HOOA+SMWG+SURMT-Serial | done | 0.5345 | 0.8398 | 0.5899 | +0.0002 | 156 | 13.44M |
| abl104_cmfa_hooa_sinf_smwg_surmt_serial | CMFA+HOOA+SINF+SMWG+SURMT-Serial | done | 0.5363 | 0.8364 | 0.5863 | +0.0019 | 108 | 14.31M |
| abl105_cmfa_hooa_sinf_smwg_cedv2_serial | CMFA+HOOA+SINF+SMWG+CED-v2-Serial | done | 0.5260 | 0.8223 | 0.5843 | -0.0083 | 139 | 17.98M |
| abl106_phone_tooth_prior_baseline | Phone-Baseline | done | 0.5355 | 0.8317 | 0.6031 | +0.0000 | 130 | 15.55M |
| abl107_phone_factor | Phone-Factor | done | 0.5360 | 0.8398 | 0.5878 | +0.0005 | 109 | 15.56M |
| abl108_phone_prototype | Phone-Prototype | done | 0.5424 | 0.8400 | 0.6178 | +0.0070 | 149 | 15.62M |
| abl109_phone_spatial_shape | Phone-SpatialShape | done | 0.5292 | 0.8311 | 0.5857 | -0.0062 | 122 | 15.62M |
| abl110_dentex_tooth_prior_baseline | DENTEX-Baseline | done | 0.5359 | 0.9394 | 0.5319 | +0.0000 | 66 | 15.52M |
| abl111_dentex_factor | DENTEX-Factor | done | 0.5357 | 0.9315 | 0.5604 | -0.0002 | 50 | 15.54M |
| abl112_dentex_prototype | DENTEX-Prototype | done | 0.5375 | 0.9392 | 0.5500 | +0.0016 | 133 | 15.57M |
| abl113_dentex_spatial_shape | DENTEX-SpatialShape | done | 0.5423 | 0.9430 | 0.5579 | +0.0064 | 76 | 15.57M |
| abl114_phone_cmfa_hooa_cedv2_baseline | Phone-CMFA+HOOA+CED-v2-Baseline | done | 0.5406 | 0.8344 | 0.6239 | +0.0000 | 138 | 16.52M |
| abl115_phone_cmfa_hooa_cedv2_factor | Phone-CMFA+HOOA+CED-v2-Factor | done | 0.5352 | 0.8412 | 0.5924 | -0.0054 | 132 | 16.54M |
| abl116_phone_cmfa_hooa_cedv2_prototype | Phone-CMFA+HOOA+CED-v2-Prototype | done | 0.5309 | 0.8331 | 0.5770 | -0.0096 | 158 | 16.59M |
| abl117_phone_cmfa_hooa_cedv2_spatial_shape | Phone-CMFA+HOOA+CED-v2-SpatialShape | done | 0.5284 | 0.8274 | 0.5776 | -0.0122 | 157 | 16.59M |
| abl118_dentex_cmfa_hooa_cedv2_baseline | DENTEX-CMFA+HOOA+CED-v2-Baseline | done | 0.5357 | 0.9410 | 0.5398 | +0.0000 | 157 | 16.50M |
| abl119_dentex_cmfa_hooa_cedv2_factor | DENTEX-CMFA+HOOA+CED-v2-Factor | done | 0.5344 | 0.9372 | 0.5490 | -0.0013 | 70 | 16.51M |
| abl120_dentex_cmfa_hooa_cedv2_prototype | DENTEX-CMFA+HOOA+CED-v2-Prototype | done | 0.5357 | 0.9409 | 0.5560 | +0.0001 | 90 | 16.54M |
| abl121_dentex_cmfa_hooa_cedv2_spatial_shape | DENTEX-CMFA+HOOA+CED-v2-SpatialShape | done | 0.5445 | 0.9478 | 0.5504 | +0.0088 | 159 | 16.54M |
| abl122_phone_cmfa_hooa_baseline | Phone-CMFA+HOOA-Baseline | done | 0.5362 | 0.8373 | 0.5877 | +0.0000 | 135 | 12.45M |
| abl123_phone_cmfa_hooa_factor | Phone-CMFA+HOOA-Factor | done | 0.5291 | 0.8288 | 0.5951 | -0.0072 | 144 | 12.46M |
| abl124_phone_cmfa_hooa_prototype | Phone-CMFA+HOOA-Prototype | done | 0.5347 | 0.8282 | 0.6057 | -0.0015 | 143 | 12.51M |
| abl125_phone_cmfa_hooa_spatial_shape | Phone-CMFA+HOOA-SpatialShape | done | 0.5310 | 0.8280 | 0.5847 | -0.0052 | 143 | 12.51M |
| abl126_dentex_cmfa_hooa_baseline | DENTEX-CMFA+HOOA-Baseline | done | 0.5334 | 0.9324 | 0.5322 | +0.0000 | 51 | 12.42M |
| abl127_dentex_cmfa_hooa_factor | DENTEX-CMFA+HOOA-Factor | done | 0.5364 | 0.9400 | 0.5418 | +0.0031 | 76 | 12.43M |
| abl128_dentex_cmfa_hooa_prototype | DENTEX-CMFA+HOOA-Prototype | done | 0.5318 | 0.9359 | 0.5315 | -0.0016 | 70 | 12.47M |
| abl129_dentex_cmfa_hooa_spatial_shape | DENTEX-CMFA+HOOA-SpatialShape | done | 0.5415 | 0.9428 | 0.5536 | +0.0082 | 159 | 12.47M |
| abl130_phone_cmfa_baseline | Phone-CMFA-Baseline | done | 0.5323 | 0.8352 | 0.5732 | +0.0000 | 147 | 11.15M |
| abl131_phone_cmfa_factor | Phone-CMFA-Factor | done | 0.5313 | 0.8381 | 0.5851 | -0.0009 | 135 | 11.17M |
| abl132_phone_cmfa_prototype | Phone-CMFA-Prototype | done | 0.5300 | 0.8297 | 0.5797 | -0.0023 | 159 | 11.22M |
| abl133_phone_cmfa_spatial_shape | Phone-CMFA-SpatialShape | done | 0.5308 | 0.8370 | 0.5800 | -0.0015 | 127 | 11.22M |
| abl134_dentex_cmfa_baseline | DENTEX-CMFA-Baseline | done | 0.5341 | 0.9413 | 0.5419 | +0.0000 | 130 | 11.13M |
| abl135_dentex_cmfa_factor | DENTEX-CMFA-Factor | done | 0.5337 | 0.9370 | 0.5519 | -0.0004 | 48 | 11.14M |
| abl136_dentex_cmfa_prototype | DENTEX-CMFA-Prototype | done | 0.5351 | 0.9378 | 0.5497 | +0.0011 | 64 | 11.17M |
| abl137_dentex_cmfa_spatial_shape | DENTEX-CMFA-SpatialShape | done | 0.5331 | 0.9402 | 0.5411 | -0.0009 | 147 | 11.17M |
| abl138_phone_smwg_baseline | Phone-SMWG-SpectralWavelet-Baseline | done | 0.5283 | 0.8258 | 0.5828 | +0.0000 | 152 | 11.22M |
| abl139_phone_smwg_factor | Phone-SMWG-SpectralWavelet-Factor | done | 0.5342 | 0.8374 | 0.5892 | +0.0059 | 157 | 11.23M |
| abl140_phone_smwg_prototype | Phone-SMWG-SpectralWavelet-Prototype | done | 0.5350 | 0.8348 | 0.5853 | +0.0067 | 158 | 11.28M |
| abl141_phone_smwg_spatial_shape | Phone-SMWG-SpectralWavelet-SpatialShape | done | 0.5327 | 0.8370 | 0.5893 | +0.0044 | 153 | 11.28M |
| abl142_dentex_smwg_baseline | DENTEX-SMWG-SpectralWavelet-Baseline | done | 0.5329 | 0.9332 | 0.5433 | +0.0000 | 68 | 11.19M |
| abl143_dentex_smwg_factor | DENTEX-SMWG-SpectralWavelet-Factor | done | 0.5368 | 0.9441 | 0.5443 | +0.0039 | 137 | 11.20M |
| abl144_dentex_smwg_prototype | DENTEX-SMWG-SpectralWavelet-Prototype | done | 0.5327 | 0.9358 | 0.5304 | -0.0001 | 156 | 11.24M |
| abl145_dentex_smwg_spatial_shape | DENTEX-SMWG-SpectralWavelet-SpatialShape | done | 0.5343 | 0.9348 | 0.5437 | +0.0014 | 71 | 11.24M |
| abl146_phone_sefs_baseline | Phone-SEFS-Baseline | done | 0.5288 | 0.8281 | 0.5865 | +0.0000 | 152 | 11.84M |
| abl147_phone_sefs_factor | Phone-SEFS-Factor | done | 0.5299 | 0.8247 | 0.5845 | +0.0010 | 154 | 11.85M |
| abl148_phone_sefs_prototype | Phone-SEFS-Prototype | done | 0.5341 | 0.8261 | 0.6040 | +0.0053 | 153 | 11.90M |
| abl149_phone_sefs_spatial_shape | Phone-SEFS-SpatialShape | done | 0.5243 | 0.8156 | 0.5880 | -0.0045 | 149 | 11.90M |
| abl150_dentex_sefs_baseline | DENTEX-SEFS-Baseline | done | 0.5385 | 0.9378 | 0.5411 | +0.0000 | 151 | 11.81M |
| abl151_dentex_sefs_factor | DENTEX-SEFS-Factor | done | 0.5341 | 0.9358 | 0.5532 | -0.0044 | 73 | 11.83M |
| abl152_dentex_sefs_prototype | DENTEX-SEFS-Prototype | done | 0.5318 | 0.9393 | 0.5227 | -0.0067 | 72 | 11.86M |
| abl153_dentex_sefs_spatial_shape | DENTEX-SEFS-SpatialShape | done | 0.5410 | 0.9390 | 0.5575 | +0.0024 | 79 | 11.86M |
| abl154_coco_rot_equivariant_detail | RotE-EquivariantDetail | done | 0.5336 | 0.8376 | 0.5902 | -0.0008 | 145 | 10.89M |
| abl155_coco_dualopt_adamw | DualOpt-AdamW | done | 0.5412 | 0.8437 | 0.6120 | +0.0068 | 149 | 10.63M |
| abl156_coco_hedp_prompt | HEDP-Prompt | done | 0.5252 | 0.8285 | 0.5724 | -0.0092 | 150 | 11.43M |
| abl157_coco_sampg_sst | SAMPG-SSTScan | done | 0.5265 | 0.8344 | 0.5688 | -0.0078 | 141 | 11.23M |
| abl158_coco_segd_entropy | SEGD-StructuralEntropy | done | 0.5233 | 0.8199 | 0.5656 | -0.0111 | 144 | 11.42M |
| abl159_coco_fsdetr_frequency_spatial | FSDETR-FrequencySpatial | done | 0.5253 | 0.8229 | 0.5759 | -0.0090 | 159 | 11.69M |
| abl160_coco_df2net_dynamic_fusion | DF2Net-DynamicFusion | done | 0.5319 | 0.8196 | 0.6022 | -0.0025 | 111 | 11.82M |
| abl161_coco_sdconet_saliency | SDCoNet-Saliency | done | 0.5250 | 0.8235 | 0.5846 | -0.0094 | 143 | 11.23M |
| abl162_coco_idgbr_boundary | IDGBR-Boundary | done | 0.5213 | 0.8272 | 0.5748 | -0.0131 | 132 | 11.43M |

Notes:
- `abl00_baseline`: No extra module or method
- `abl01_hfsa`: Enable HybridEncoder.use_hfsa
- `abl02_wav`: Enable HybridEncoder.use_wav_enhance
- `abl03_stat_noise`: Enable HybridEncoder.use_stat_denoise
- `abl04_mpf`: Enable HybridEncoder.use_multipath_fusion
- `abl05_dynamic_rf`: Enable HybridEncoder.use_dynamic_rf
- `abl06_mpc`: Enable HybridEncoder.use_mpc_multiscale_perception
- `abl07_pqg_hf`: Enable HybridEncoder.use_pqg_high_frequency
- `abl08_cmfa`: Enable HybridEncoder.use_cmfa_fusion
- `abl09_uni_gate`: Enable DFINETransformer.use_uni_gate + infer
- `abl10_qgp`: Enable DFINETransformer.use_qgp with qgp_layers=[1,2]
- `abl11_pgr`: Enable DFINETransformer.use_pgr_roi_reasoning
- `abl12_cmfa_mpf`: Enable HybridEncoder.use_cmfa_fusion + use_multipath_fusion
- `abl13_cmfa_pgr`: Enable HybridEncoder.use_cmfa_fusion + DFINETransformer.use_pgr_roi_reasoning
- `abl14_mpf_pgr`: Enable HybridEncoder.use_multipath_fusion + DFINETransformer.use_pgr_roi_reasoning
- `abl15_cmfa_mpf_pgr`: Enable HybridEncoder.use_cmfa_fusion + use_multipath_fusion + DFINETransformer.use_pgr_roi_reasoning
- `abl16_stat_noise_dynamic_rf`: Enable HybridEncoder.use_stat_denoise + use_dynamic_rf
- `abl17_stat_noise_mpc`: Enable HybridEncoder.use_stat_denoise + use_mpc_multiscale_perception
- `abl18_pfesa`: Enable HybridEncoder.use_pfesa
- `abl19_ucan_lkd`: Enable HybridEncoder.use_ucan_lkd
- `abl20_ucm_uncertainty`: Enable HybridEncoder.use_ucm_uncertainty
- `abl21_retinex_prior`: Enable HybridEncoder.use_retinex_prior
- `abl22_cmfa_pfesa`: Enable HybridEncoder.use_cmfa_fusion + use_pfesa
- `abl23_mpf_pfesa`: Enable HybridEncoder.use_multipath_fusion + use_pfesa
- `abl24_cmfa_ucan_lkd`: Enable HybridEncoder.use_cmfa_fusion + use_ucan_lkd
- `abl25_pfesa_ucm_uncertainty`: Enable HybridEncoder.use_pfesa + use_ucm_uncertainty
- `abl26_cmfa_pfesa_ucan_lkd`: Enable HybridEncoder.use_cmfa_fusion + use_pfesa + use_ucan_lkd
- `abl27_cmfa_pfesa_ucm_uncertainty`: Enable HybridEncoder.use_cmfa_fusion + use_pfesa + use_ucm_uncertainty
- `abl28_fdi_factor_prior`: Enable trainable quadrant/tooth-position/dentition factor heads
- `abl29_cmfa_ucm_uncertainty`: Enable HybridEncoder.use_cmfa_fusion + use_ucm_uncertainty
- `abl30_cmfa_ucm_fdi_factor`: Enable CMFA + UCM-Uncertainty + trainable FDI factor prior
- `abl31_hooa_attention`: Enable HybridEncoder.use_hooa_attention
- `abl32_multinex_prior`: Enable HybridEncoder.use_multinex_prior
- `abl33_spegc_prompt_graph`: Enable DFINETransformer.use_spegc_prompt_graph
- `abl34_cmfa_hooa`: Enable CMFA fusion + HOOA
- `abl35_ucm_hooa`: Enable UCM uncertainty + HOOA
- `abl36_cmfa_ucm_spegc`: Enable CMFA + UCM uncertainty + SPEGC prompt graph
- `abl37_hooa_multinex`: Enable HOOA + MultiNeX prior
- `abl38_cmfa_ucm_hooa_spegc`: Enable current best encoder pair plus HOOA and SPEGC prompt graph
- `abl39_nwd_loss`: Add normalized Gaussian Wasserstein distance box loss
- `abl40_fdi_spatial_prior`: Enable FDI jaw/neighbor/unique logit prior in decoder
- `abl41_fdi_anatomy_loss`: Enable final-layer FDI factor and uniqueness anatomy losses
- `abl42_cmfa_hooa_nwd`: Add NWD loss to the current strongest CMFA+HOOA branch
- `abl43_cmfa_hooa_fdi_prior`: Add FDI spatial logit prior to CMFA+HOOA
- `abl44_cmfa_hooa_anatomy_loss`: Add final FDI anatomy losses to CMFA+HOOA
- `abl45_cmfa_hooa_nwd_fdi_prior`: Combine NWD loss with FDI spatial logit prior
- `abl46_cmfa_hooa_all_priors`: Combine NWD, FDI spatial prior, and final anatomy losses
- `abl47_msck_refiner`: Add MSCK-Net inspired Chinese-knot multi-scale strip convolution refiner
- `abl48_csaf_filter`: Add CSAFNet context-aware semantic adaptive filtering with wavelet-detail fallback
- `abl49_csaf_dmssp`: Add CSAFNet dense multi-scale spatial semantic perception refiner
- `abl50_rdvq_refiner`: Add differentiable vector-quantization codebook feature refiner
- `abl51_datpp_refiner`: Add DAT++ style grouped deformable attention on selected FPN levels
- `abl52_cluster_query_refiner`: Add cluster-wise graph transformer over decoder main queries
- `abl53_query_recovery_memory`: Add decoder query recovery memory against iterative query drift
- `abl54_cmfa_hooa_msck_csaf_filter`: Combine strongest encoder branch with MSCK and CSAF adaptive filtering
- `abl55_cmfa_hooa_csaf_dmssp_rdvq`: Combine strongest encoder branch with CSAF-DMSSP and RDVQ codebook refinement
- `abl56_cmfa_hooa_datpp_cluster_recovery`: Combine deformable feature refinement with cluster query graph and recovery memory
- `abl57_cmfa_hooa_all_external`: Combine all externally inspired modules without FDI/NWD priors
- `abl58_cmfa_hooa_external_fdi_priors`: Combine all external modules with NWD, FDI spatial prior, and anatomy losses
- `abl59_cnm_ode_refiner`: Add CNM-UNet inspired continuous neural-memory ODE feature refiner
- `abl60_cldyn_rsc`: Add CLDyN RSC/A2SI basis-bank semantic compensation refiner
- `abl61_ced_cascade_refiner`: Add CEDNet cascade encoder-decoder multi-scale feature refiner
- `abl62_dsfed_dual_scale_kd`: Add DSFedMed inspired dual-scale distillation from final decoder to lightweight heads
- `abl63_cmfa_hooa_cnm_ced`: Combine current strongest encoder branch with CNM-ODE and CED cascade refiners
- `abl64_cmfa_hooa_cldyn_dsfed`: Combine task-aware semantic compensation with dual-scale distillation
- `abl65_cmfa_hooa_cnm_cldyn_ced`: Combine the three paper-inspired encoder refiners
- `abl66_cmfa_hooa_paper_modules_dsfed_priors`: Combine CNM-ODE, CLDyN-RSC, CED cascade, DSFed KD, NWD, FDI spatial prior, and anatomy losses
- `abl67_cnm_ode_refiner_v2`: CNM-ODE moved to projected features with non-zero ODE projection and stronger residual scale
- `abl68_cldyn_rsc_v2`: CLDyN RSC moved to projected features with active zero-conv replacement and stronger semantic residual
- `abl69_ced_cascade_refiner_v2`: CED cascade moved before FPN/PAN with one bottlenecked cascade stage to reduce overfitting
- `abl70_dsfed_dual_scale_kd_v2`: DSFed KD changed to sigmoid reliable-query dual-scale consistency with a light final-head consensus term
- `abl71_cmfa_hooa_cnm_cldyn_proj`: Combine strongest encoder branch with corrected CNM and CLDyN projected-feature refiners
- `abl72_cmfa_hooa_ced_proj`: Combine strongest encoder branch with corrected projected-feature CED cascade
- `abl73_cmfa_hooa_paper_modules_proj`: Combine corrected CNM, CLDyN and CED projected-feature refiners
- `abl74_cmfa_hooa_paper_modules_dsfed_v2`: Combine corrected paper-inspired encoder refiners with corrected DSFed dual-scale KD
- `abl75_dcsam_dual_consistency`: Add DC-SAM inspired positive/negative prompt and cross-level dual-consistency feature refinement on projected features
- `abl76_iqp_quality_prior`: Add IQPIR inspired no-reference feature quality prior with common/HQ codebook conditioning
- `abl77_dformer_geometry_prior`: Add DFormer++ inspired pseudo-arch geometry attention for panoramic dental features
- `abl78_cdml_domain_adapter`: Add CDML inspired first-order domain affine adapter for scanner/exposure shift robustness
- `abl79_pfrpn_query_adapter`: Add PF-RPN inspired sparse image-aware query adapter with cascade self-prompt and centerness gating
- `abl80_anatomical_region_prior`: Add anatomical region-guided contrastive logit prior adapted from medical VLM region decoding
- `abl81_tooth_size_prior_loss`: Add weak pairwise tooth-size ordering prior from dental anatomy
- `abl82_cmfa_hooa_dental_priors_v3`: Combine FDI spatial prior, anatomy loss, anatomical region prior, and tooth-size prior on CMFA+HOOA
- `abl83_cmfa_hooa_external_priors_v3`: Combine DC-SAM, IQP, DFormer geometry, CDML affine, and PF-RPN query adapter with CMFA+HOOA
- `abl84_cmfa_hooa_paper_modules_dsfed_priors_v3`: Combine v2 paper modules, new external priors, DSFed KD, NWD, and dental anatomy priors
- `abl85_efruf_gradient_fusion`: Add EFRUF inspired Sobel gradient-transfer and unfolded feature fusion on projected features
- `abl86_sinf_spatial_field`: Add SINF inspired blind-spot continuous spatial implicit field refinement for single-image denoising
- `abl87_smwg_spectral_wavelet`: Add SMWG-DETR inspired Fourier spectral modulation and wavelet/detail-guided multi-scale feature refinement
- `abl88_tooth_template_prior`: Add multi-template tooth-position matching prior over decoder query features and FDI arch locations
- `abl89_cmfa_hooa_frequency_field_v4`: Combine EFRUF gradient fusion, SINF spatial field, and SMWG spectral-wavelet refinement with CMFA+HOOA
- `abl90_cmfa_hooa_template_dental_v4`: Combine tooth template matching with FDI spatial, anatomical region, tooth-size, and anatomy priors
- `abl91_cmfa_hooa_all_new_20260512`: Combine v2 paper modules, abl75-84 modules, EFRUF, SINF, SMWG, template matching, KD, NWD, and dental priors
- `abl92_cmfa_hooa_efruf_serial`: Serial ablation step 1: add EFRUF gradient-transfer fusion after the best CMFA+HOOA encoder
- `abl93_cmfa_hooa_efruf_sinf_serial`: Serial ablation step 2: add SINF spatial implicit field on top of EFRUF
- `abl94_cmfa_hooa_efruf_sinf_smwg_serial`: Serial ablation step 3: add SMWG Fourier spectral modulation and wavelet-guided refinement
- `abl95_cmfa_hooa_efruf_sinf_smwg_bilalora_serial`: Serial ablation step 4: add BiLaLoRA-inspired low-rank layer-positioned adapter
- `abl96_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_serial`: Serial ablation step 5: add SU-RMT semantic-structure bridge for dental boundary detail
- `abl97_cmfa_hooa_efruf_sinf_smwg_bilalora_surmt_mav_serial`: Serial ablation step 6: add MAVFusion-inspired sparse cross-scale interaction
- `abl98_cmfa_hooa_sinf_serial`: Follow-up subset ablation: add SINF spatial implicit field to CMFA+HOOA on output features
- `abl99_cmfa_hooa_smwg_serial`: Follow-up subset ablation: add SMWG Fourier spectral modulation and wavelet-guided refinement to CMFA+HOOA
- `abl100_cmfa_hooa_surmt_serial`: Follow-up subset ablation: add SU-RMT semantic-structure bridge to CMFA+HOOA
- `abl101_cmfa_hooa_sinf_smwg_serial`: Follow-up subset ablation: combine SINF and SMWG on top of CMFA+HOOA
- `abl102_cmfa_hooa_sinf_surmt_serial`: Follow-up subset ablation: combine SINF and SU-RMT on top of CMFA+HOOA
- `abl103_cmfa_hooa_smwg_surmt_serial`: Follow-up subset ablation: combine SMWG and SU-RMT on top of CMFA+HOOA
- `abl104_cmfa_hooa_sinf_smwg_surmt_serial`: Follow-up subset ablation: combine SINF, SMWG, and SU-RMT on top of CMFA+HOOA
- `abl105_cmfa_hooa_sinf_smwg_cedv2_serial`: Follow-up subset ablation: add corrected CED cascade to CMFA+HOOA+SINF+SMWG
- `abl106_phone_tooth_prior_baseline`: Phone /home/wuxi/coco baseline for factor-prototype-spatial_shape serial ablation
- `abl107_phone_factor`: Phone serial ablation: add factor heads
- `abl108_phone_prototype`: Phone serial ablation: add learnable tooth appearance prototypes
- `abl109_phone_spatial_shape`: Phone serial ablation: add external spatial and bbox-shape templates
- `abl110_dentex_tooth_prior_baseline`: DENTEX2023 baseline for factor-prototype-spatial_shape serial ablation
- `abl111_dentex_factor`: DENTEX serial ablation: add factor heads
- `abl112_dentex_prototype`: DENTEX serial ablation: add learnable tooth appearance prototypes
- `abl113_dentex_spatial_shape`: DENTEX serial ablation: add external spatial and bbox-shape templates
- `abl114_phone_cmfa_hooa_cedv2_baseline`: Phone /home/wuxi/coco: best previous branch with CMFA, HOOA, and projected CED cascade; baseline detector
- `abl115_phone_cmfa_hooa_cedv2_factor`: Phone /home/wuxi/coco: best previous branch with CMFA, HOOA, and projected CED cascade; add factor heads
- `abl116_phone_cmfa_hooa_cedv2_prototype`: Phone /home/wuxi/coco: best previous branch with CMFA, HOOA, and projected CED cascade; add factor heads and learnable tooth appearance prototypes
- `abl117_phone_cmfa_hooa_cedv2_spatial_shape`: Phone /home/wuxi/coco: best previous branch with CMFA, HOOA, and projected CED cascade; add factor/prototype plus spatial, shape, and view priors
- `abl118_dentex_cmfa_hooa_cedv2_baseline`: DENTEX /home/wuxi/DENTEX2023: best previous branch with CMFA, HOOA, and projected CED cascade; baseline detector
- `abl119_dentex_cmfa_hooa_cedv2_factor`: DENTEX /home/wuxi/DENTEX2023: best previous branch with CMFA, HOOA, and projected CED cascade; add factor heads
- `abl120_dentex_cmfa_hooa_cedv2_prototype`: DENTEX /home/wuxi/DENTEX2023: best previous branch with CMFA, HOOA, and projected CED cascade; add factor heads and learnable tooth appearance prototypes
- `abl121_dentex_cmfa_hooa_cedv2_spatial_shape`: DENTEX /home/wuxi/DENTEX2023: best previous branch with CMFA, HOOA, and projected CED cascade; add factor/prototype plus spatial, shape, and view priors
- `abl122_phone_cmfa_hooa_baseline`: Phone /home/wuxi/coco: best previous encoder pair; baseline detector
- `abl123_phone_cmfa_hooa_factor`: Phone /home/wuxi/coco: best previous encoder pair; add factor heads
- `abl124_phone_cmfa_hooa_prototype`: Phone /home/wuxi/coco: best previous encoder pair; add factor heads and learnable tooth appearance prototypes
- `abl125_phone_cmfa_hooa_spatial_shape`: Phone /home/wuxi/coco: best previous encoder pair; add factor/prototype plus spatial, shape, and view priors
- `abl126_dentex_cmfa_hooa_baseline`: DENTEX /home/wuxi/DENTEX2023: best previous encoder pair; baseline detector
- `abl127_dentex_cmfa_hooa_factor`: DENTEX /home/wuxi/DENTEX2023: best previous encoder pair; add factor heads
- `abl128_dentex_cmfa_hooa_prototype`: DENTEX /home/wuxi/DENTEX2023: best previous encoder pair; add factor heads and learnable tooth appearance prototypes
- `abl129_dentex_cmfa_hooa_spatial_shape`: DENTEX /home/wuxi/DENTEX2023: best previous encoder pair; add factor/prototype plus spatial, shape, and view priors
- `abl130_phone_cmfa_baseline`: Phone /home/wuxi/coco: standalone cross-modal feature aggregation branch; baseline detector
- `abl131_phone_cmfa_factor`: Phone /home/wuxi/coco: standalone cross-modal feature aggregation branch; add factor heads
- `abl132_phone_cmfa_prototype`: Phone /home/wuxi/coco: standalone cross-modal feature aggregation branch; add factor heads and learnable tooth appearance prototypes
- `abl133_phone_cmfa_spatial_shape`: Phone /home/wuxi/coco: standalone cross-modal feature aggregation branch; add factor/prototype plus spatial, shape, and view priors
- `abl134_dentex_cmfa_baseline`: DENTEX /home/wuxi/DENTEX2023: standalone cross-modal feature aggregation branch; baseline detector
- `abl135_dentex_cmfa_factor`: DENTEX /home/wuxi/DENTEX2023: standalone cross-modal feature aggregation branch; add factor heads
- `abl136_dentex_cmfa_prototype`: DENTEX /home/wuxi/DENTEX2023: standalone cross-modal feature aggregation branch; add factor heads and learnable tooth appearance prototypes
- `abl137_dentex_cmfa_spatial_shape`: DENTEX /home/wuxi/DENTEX2023: standalone cross-modal feature aggregation branch; add factor/prototype plus spatial, shape, and view priors
- `abl138_phone_smwg_baseline`: Phone /home/wuxi/coco: standalone spectral-wavelet frequency/detail branch; baseline detector
- `abl139_phone_smwg_factor`: Phone /home/wuxi/coco: standalone spectral-wavelet frequency/detail branch; add factor heads
- `abl140_phone_smwg_prototype`: Phone /home/wuxi/coco: standalone spectral-wavelet frequency/detail branch; add factor heads and learnable tooth appearance prototypes
- `abl141_phone_smwg_spatial_shape`: Phone /home/wuxi/coco: standalone spectral-wavelet frequency/detail branch; add factor/prototype plus spatial, shape, and view priors
- `abl142_dentex_smwg_baseline`: DENTEX /home/wuxi/DENTEX2023: standalone spectral-wavelet frequency/detail branch; baseline detector
- `abl143_dentex_smwg_factor`: DENTEX /home/wuxi/DENTEX2023: standalone spectral-wavelet frequency/detail branch; add factor heads
- `abl144_dentex_smwg_prototype`: DENTEX /home/wuxi/DENTEX2023: standalone spectral-wavelet frequency/detail branch; add factor heads and learnable tooth appearance prototypes
- `abl145_dentex_smwg_spatial_shape`: DENTEX /home/wuxi/DENTEX2023: standalone spectral-wavelet frequency/detail branch; add factor/prototype plus spatial, shape, and view priors
- `abl146_phone_sefs_baseline`: Phone /home/wuxi/coco: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; baseline detector
- `abl147_phone_sefs_factor`: Phone /home/wuxi/coco: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor heads
- `abl148_phone_sefs_prototype`: Phone /home/wuxi/coco: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor heads and learnable tooth appearance prototypes
- `abl149_phone_sefs_spatial_shape`: Phone /home/wuxi/coco: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor/prototype plus spatial, shape, and view priors
- `abl150_dentex_sefs_baseline`: DENTEX /home/wuxi/DENTEX2023: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; baseline detector
- `abl151_dentex_sefs_factor`: DENTEX /home/wuxi/DENTEX2023: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor heads
- `abl152_dentex_sefs_prototype`: DENTEX /home/wuxi/DENTEX2023: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor heads and learnable tooth appearance prototypes
- `abl153_dentex_sefs_spatial_shape`: DENTEX /home/wuxi/DENTEX2023: new structural-entropy frequency-saliency refiner from the 2026 paper/code review; add factor/prototype plus spatial, shape, and view priors
- `abl154_coco_rot_equivariant_detail`: Phone /home/wuxi/coco single-module ablation: Rot-E ASISR-inspired transform-equivariant detail refiner
- `abl155_coco_dualopt_adamw`: Phone /home/wuxi/coco single-module ablation: OLOR/DualOpt-inspired real-time decay and initialization rollback optimizer
- `abl156_coco_hedp_prompt`: Phone /home/wuxi/coco single-module ablation: hybrid energy-distance prompt feature refiner
- `abl157_coco_sampg_sst`: Phone /home/wuxi/coco single-module ablation: SAM-prior-guided spatial/spectral scan refiner
- `abl158_coco_segd_entropy`: Phone /home/wuxi/coco single-module ablation: SEGD-inspired structural entropy and degradation-evidence refiner
- `abl159_coco_fsdetr_frequency_spatial`: Phone /home/wuxi/coco single-module ablation: FSDETR-inspired frequency-spatial small-object feature enhancer
- `abl160_coco_df2net_dynamic_fusion`: Phone /home/wuxi/coco single-module ablation: DF2Net-inspired differential dynamic FPN fusion
- `abl161_coco_sdconet_saliency`: Phone /home/wuxi/coco single-module ablation: SDCoNet-inspired saliency-driven collaborative feature refiner
- `abl162_coco_idgbr_boundary`: Phone /home/wuxi/coco single-module ablation: IDGBR-inspired semantic-guided boundary refinement
