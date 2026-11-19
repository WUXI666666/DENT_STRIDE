# View-Net3 DSLR FDI52: Original Combination-Module Recipes

- Dataset: 750 training images / 190 validation images; canonical 52-class FDI label space.
- Recipes: the 12 unique module recipes in `ablation_combo_modules_cross_dataset`; its DENTEX and COCO sections repeat this same recipe set.
- CMFA+DualOpt and MPF+DualOpt are reused from the completed Best4 DSLR queue; the other 10 recipes run in this queue.
- The SpatialShape recipes use templates built only from the DSLR training split; `dental_view_type=unknown` avoids imposing a panorama prior on five-view DSLR data.
- Completed runs retain only `best.pth`. AP values are COCO bbox percentage points; ranks include completed rows only.

| Order | Rank | Method | Status | Source | Epochs | Best epoch | AP | AP50 | AP75 | AR100 | Δ vs best completed | Params | Experiment |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | SpatialShape + DualOpt | done | combo-recipe queue | 160 | 153 | 71.533 | 87.085 | 82.554 | 82.804 | -1.898 | 10.70M | `viewnet3_spatial_shape_dualopt` |
| 2 | 10 | Prototype + DualOpt | done | combo-recipe queue | 160 | 148 | 72.292 | 87.958 | 83.116 | 81.785 | -1.139 | 10.70M | `viewnet3_prototype_dualopt` |
| 3 | 1 | CMFA + DualOpt | done | reused from Best4 queue | 160 | 156 | 73.431 | 89.580 | 84.660 | 82.054 | +0.000 | 11.15M | `viewnet3_cmfa_dualopt` |
| 4 | 5 | MPF + DualOpt | done | reused from Best4 queue | 159 | 135 | 72.700 | 88.181 | 83.244 | 81.765 | -0.732 | 11.86M | `viewnet3_mpf_dualopt` |
| 5 | 2 | CMFA + HOOA + DualOpt | done | combo-recipe queue | 160 | 151 | 72.995 | 89.086 | 83.696 | 81.168 | -0.436 | 12.45M | `viewnet3_cmfa_hooa_dualopt` |
| 6 | 6 | CMFA + HOOA + Prototype | done | combo-recipe queue | 160 | 68 | 72.670 | 88.183 | 83.886 | 83.069 | -0.761 | 12.51M | `viewnet3_cmfa_hooa_prototype` |
| 7 | 7 | CMFA + HOOA + SpatialShape | done | combo-recipe queue | 160 | 141 | 72.417 | 88.569 | 83.685 | 83.231 | -1.015 | 12.51M | `viewnet3_cmfa_hooa_spatial_shape` |
| 8 | 11 | CMFA + HOOA + SpatialShape + DualOpt | done | combo-recipe queue | 160 | 145 | 72.266 | 88.127 | 82.935 | 83.645 | -1.165 | 12.51M | `viewnet3_cmfa_hooa_spatial_shape_dualopt` |
| 9 | 9 | CMFA + HOOA + CEDv2 | done | combo-recipe queue | 160 | 148 | 72.353 | 88.056 | 82.285 | 81.783 | -1.079 | 16.52M | `viewnet3_cmfa_hooa_cedv2` |
| 10 | 8 | CMFA + HOOA + CEDv2 + DualOpt | done | combo-recipe queue | 160 | 146 | 72.402 | 88.011 | 83.215 | 82.253 | -1.029 | 16.52M | `viewnet3_cmfa_hooa_cedv2_dualopt` |
| 11 | 4 | CMFA + HOOA + CEDv2 + SpatialShape | done | combo-recipe queue | 160 | 114 | 72.812 | 88.491 | 83.946 | 82.935 | -0.619 | 16.59M | `viewnet3_cmfa_hooa_cedv2_spatial_shape` |
| 12 | 3 | CMFA + HOOA + CEDv2 + SpatialShape + DualOpt | done | combo-recipe queue | 160 | 137 | 72.838 | 88.590 | 83.353 | 84.086 | -0.593 | 16.59M | `viewnet3_cmfa_hooa_cedv2_spatial_shape_dualopt` |

## Files

- CSV: `/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_combo_modules/ablation_summary.csv`
- Markdown: `/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_combo_modules/ablation_summary.md`
