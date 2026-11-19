# View-Net3 DSLR FDI52: Best4 + DualOpt Combination Ablation

- Dataset: 750 training images / 190 validation images; canonical 52-class FDI label space.
- Queue: 16 serial experiments on one GPU; expected length: 160 epochs each.
- Every row uses DualOptAdamW. Only CMFA, SMWG, UCM, and MPF vary.
- AP values are COCO bbox percentage points. Ranks and deltas include completed runs only.

| Order | Rank | Modules | Status | Epochs | Best epoch | AP | AP50 | AP75 | AR100 | Δ vs best completed | Params | Experiment |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | none | done | 160 | 137 | 73.173 | 89.677 | 83.736 | 80.538 | -0.258 | 10.63M | `viewnet3_dualopt_only` |
| 2 | 1 | CMFA | done | 160 | 156 | 73.431 | 89.580 | 84.660 | 82.054 | +0.000 | 11.15M | `viewnet3_cmfa_dualopt` |
| 3 | 15 | SMWG | done | 160 | 124 | 72.092 | 87.901 | 82.995 | 81.149 | -1.340 | 11.22M | `viewnet3_smwg_dualopt` |
| 4 | 5 | UCM | done | 160 | 81 | 73.107 | 88.902 | 84.196 | 81.623 | -0.324 | 11.08M | `viewnet3_ucm_dualopt` |
| 5 | 7 | MPF | done | 159 | 135 | 72.700 | 88.181 | 83.244 | 81.765 | -0.732 | 11.86M | `viewnet3_mpf_dualopt` |
| 6 | 8 | CMFA+SMWG | done | 160 | 151 | 72.640 | 88.407 | 83.458 | 80.992 | -0.792 | 11.74M | `viewnet3_cmfa_smwg_dualopt` |
| 7 | 10 | CMFA+UCM | done | 160 | 149 | 72.551 | 88.484 | 83.706 | 81.069 | -0.880 | 11.61M | `viewnet3_cmfa_ucm_dualopt` |
| 8 | 11 | CMFA+MPF | done | 160 | 118 | 72.528 | 88.260 | 82.936 | 82.250 | -0.903 | 12.38M | `viewnet3_cmfa_mpf_dualopt` |
| 9 | 16 | SMWG+UCM | done | 160 | 137 | 71.999 | 87.329 | 82.592 | 80.347 | -1.432 | 11.67M | `viewnet3_smwg_ucm_dualopt` |
| 10 | 12 | SMWG+MPF | done | 160 | 139 | 72.335 | 88.132 | 83.498 | 81.501 | -1.096 | 12.44M | `viewnet3_smwg_mpf_dualopt` |
| 11 | 6 | UCM+MPF | done | 160 | 153 | 72.759 | 88.616 | 83.451 | 80.904 | -0.672 | 12.31M | `viewnet3_ucm_mpf_dualopt` |
| 12 | 9 | CMFA+SMWG+UCM | done | 160 | 142 | 72.582 | 88.514 | 83.490 | 81.228 | -0.849 | 12.19M | `viewnet3_cmfa_smwg_ucm_dualopt` |
| 13 | 4 | CMFA+SMWG+MPF | done | 160 | 154 | 73.123 | 89.630 | 83.501 | 82.297 | -0.308 | 12.97M | `viewnet3_cmfa_smwg_mpf_dualopt` |
| 14 | 14 | CMFA+UCM+MPF | done | 160 | 140 | 72.231 | 87.686 | 83.315 | 81.489 | -1.200 | 12.84M | `viewnet3_cmfa_ucm_mpf_dualopt` |
| 15 | 3 | SMWG+UCM+MPF | done | 160 | 139 | 73.148 | 89.503 | 83.793 | 81.974 | -0.283 | 12.90M | `viewnet3_smwg_ucm_mpf_dualopt` |
| 16 | 13 | CMFA+SMWG+UCM+MPF | done | 160 | 154 | 72.262 | 87.864 | 82.798 | 81.413 | -1.169 | 13.42M | `viewnet3_cmfa_smwg_ucm_mpf_dualopt` |

## Files

- CSV: `/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_best4_dualopt_combinations/ablation_summary.csv`
- Markdown: `/home/wuxi/Dent-FINE/outputs/ablation_viewnet3_dslr_best4_dualopt_combinations/ablation_summary.md`
