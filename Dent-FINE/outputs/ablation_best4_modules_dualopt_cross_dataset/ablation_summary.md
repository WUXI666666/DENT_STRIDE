# Ablation Summary

Base directory: `/home/wuxi/Dent-FINE/outputs/ablation_best4_modules_dualopt_cross_dataset`

Metrics are COCO bbox metrics. AP values below are shown as percentage points, e.g. 54.321 means AP=0.54321. Undefined scale metrics from COCO evaluator are shown as `-`.

## DENTEX2023

| Rank | Experiment | Method | Status | Best AP | AP50 | AP75 | APs | APm | APl | AR100 | Best Epoch | Last AP | Delta vs Best | Params |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dentex_single_spatial_shape` | SpatialShape | done | 54.082 | 94.011 | 54.845 | - | 52.386 | 54.336 | 70.849 | 63 | 53.612 | +0.000 | 10.65M |
| 2 | `dentex_single_prototype` | Prototype | done | 53.633 | 93.189 | 54.455 | - | 40.462 | 54.052 | 66.512 | 92 | 53.407 | -0.449 | 10.65M |
| 3 | `dentex_cmfa_hooa` | CMFA+HOOA | done | 53.444 | 93.390 | 54.930 | - | 46.883 | 53.743 | 68.884 | 72 | 53.057 | -0.638 | 12.42M |
| 4 | `dentex_best4_cmfa_smwg_ucm_mpf_dualopt` | Best4(CMFA+SMWG+UCM+MPF)+DualOpt | done | 53.201 | 94.007 | 53.777 | - | 41.563 | 53.592 | 68.073 | 75 | 52.928 | -0.881 | 13.40M |
| 5 | `dentex_single_ucm` | UCM-Uncertainty | done | 53.190 | 93.585 | 52.935 | - | 37.121 | 53.611 | 67.156 | 135 | 52.812 | -0.893 | 11.06M |
| 6 | `dentex_dualopt_only` | DualOpt-only | done | 53.180 | 93.297 | 53.739 | - | 40.707 | 53.617 | 67.112 | 145 | 52.520 | -0.902 | 10.60M |
| 7 | `dentex_single_cmfa` | CMFA | done | 53.127 | 93.728 | 52.941 | - | 43.895 | 53.518 | 68.383 | 138 | 52.387 | -0.955 | 11.13M |
| 8 | `dentex_single_smwg` | SMWG-SpectralWavelet | done | 53.013 | 93.261 | 53.491 | - | 44.595 | 53.368 | 68.131 | 69 | 52.465 | -1.070 | 11.19M |
| 9 | `dentex_single_mpf` | MPF | done | 52.808 | 93.316 | 53.993 | - | 42.305 | 53.189 | 67.690 | 66 | 52.468 | -1.274 | 11.83M |

## COCO-/home/wuxi/coco

| Rank | Experiment | Method | Status | Best AP | AP50 | AP75 | APs | APm | APl | AR100 | Best Epoch | Last AP | Delta vs Best | Params |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `coco_dualopt_only` | DualOpt-only | done | 54.090 | 84.165 | 59.737 | 45.000 | 36.740 | 55.855 | 67.684 | 133 | 54.042 | +0.000 | 10.63M |
| 2 | `coco_best4_cmfa_smwg_ucm_mpf_dualopt` | Best4(CMFA+SMWG+UCM+MPF)+DualOpt | done | 53.998 | 83.613 | 60.057 | 40.000 | 36.980 | 55.707 | 68.683 | 155 | 53.734 | -0.092 | 13.42M |
| 3 | `coco_single_mpf` | MPF | done | 53.726 | 83.882 | 59.323 | 42.500 | 39.451 | 55.253 | 67.040 | 140 | 53.630 | -0.364 | 11.86M |
| 4 | `coco_single_cmfa` | CMFA | done | 53.551 | 83.914 | 58.898 | 42.500 | 36.115 | 55.142 | 67.082 | 145 | 53.548 | -0.539 | 11.15M |
| 5 | `coco_single_prototype` | Prototype | done | 53.305 | 83.327 | 57.481 | 42.500 | 37.864 | 55.055 | 68.276 | 116 | 53.196 | -0.785 | 10.70M |
| 6 | `coco_single_smwg` | SMWG-SpectralWavelet | done | 53.236 | 83.707 | 59.507 | 37.500 | 34.973 | 55.020 | 67.179 | 134 | 53.224 | -0.854 | 11.22M |
| 7 | `coco_single_ucm` | UCM-Uncertainty | done | 52.606 | 82.100 | 58.973 | 37.500 | 34.800 | 54.402 | 67.982 | 138 | 52.582 | -1.484 | 11.08M |
| 8 | `coco_cmfa_hooa` | CMFA+HOOA | done | 52.369 | 82.657 | 56.582 | 41.250 | 35.984 | 53.979 | 66.302 | 149 | 52.223 | -1.721 | 12.45M |
| 9 | `coco_single_spatial_shape` | SpatialShape | done | 52.227 | 82.298 | 57.393 | 42.500 | 35.409 | 54.063 | 66.822 | 149 | 52.205 | -1.863 | 10.70M |

## Files

- CSV: `/home/wuxi/Dent-FINE/outputs/ablation_best4_modules_dualopt_cross_dataset/ablation_summary.csv`
- Markdown: `/home/wuxi/Dent-FINE/outputs/ablation_best4_modules_dualopt_cross_dataset/ablation_summary.md`
