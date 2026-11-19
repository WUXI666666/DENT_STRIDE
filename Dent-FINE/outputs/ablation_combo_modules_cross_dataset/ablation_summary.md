# Combination Ablation Summary

Base directory: `/home/wuxi/Dent-FINE/outputs/ablation_combo_modules_cross_dataset`

Metrics are COCO bbox metrics. AP values are percentage points; undefined scale metrics are shown as `-`.

## DENTEX2023

| Rank | Experiment | Method | Status | Best AP | AP50 | AP75 | APs | APm | APl | AR100 | Best Epoch | Last AP | Delta vs Best | Params |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dentex_cmfa_hooa_cedv2_spatial_shape_dualopt` | CMFA+HOOA+CEDv2+SpatialShape+DualOpt | done | 54.344 | 93.851 | 57.081 | - | 48.247 | 54.750 | 71.272 | 64 | 54.084 | +0.000 | 16.54M |
| 2 | `dentex_cmfa_hooa_cedv2_spatial_shape` | CMFA+HOOA+CEDv2+SpatialShape | done | 54.020 | 93.997 | 55.694 | - | 48.640 | 54.362 | 70.687 | 76 | 53.654 | -0.324 | 16.54M |
| 3 | `dentex_spatial_shape_dualopt` | SpatialShape+DualOpt | done | 53.919 | 93.878 | 54.650 | - | 48.970 | 54.179 | 70.359 | 64 | 53.452 | -0.425 | 10.65M |
| 4 | `dentex_cmfa_hooa_cedv2_dualopt` | CMFA+HOOA+CEDv2+DualOpt | done | 53.779 | 93.715 | 54.785 | - | 42.231 | 54.189 | 67.387 | 137 | 53.495 | -0.565 | 16.50M |
| 5 | `dentex_cmfa_hooa_spatial_shape_dualopt` | CMFA+HOOA+SpatialShape+DualOpt | done | 53.610 | 94.140 | 54.788 | - | 42.881 | 54.025 | 70.011 | 153 | 53.461 | -0.734 | 12.47M |
| 6 | `dentex_cmfa_hooa_spatial_shape` | CMFA+HOOA+SpatialShape | done | 53.522 | 94.399 | 53.300 | - | 47.915 | 53.892 | 69.304 | 150 | 53.176 | -0.822 | 12.47M |
| 7 | `dentex_cmfa_hooa_prototype` | CMFA+HOOA+Prototype | done | 53.384 | 93.454 | 54.285 | - | 45.272 | 53.728 | 67.357 | 64 | 52.861 | -0.960 | 12.47M |
| 8 | `dentex_prototype_dualopt` | Prototype+DualOpt | done | 53.183 | 93.357 | 53.957 | - | 46.594 | 53.550 | 68.074 | 55 | 52.267 | -1.161 | 10.65M |
| 9 | `dentex_cmfa_hooa_dualopt` | CMFA+HOOA+DualOpt | done | 53.181 | 93.277 | 53.976 | - | 42.020 | 53.560 | 67.690 | 143 | 52.901 | -1.163 | 12.42M |
| 10 | `dentex_cmfa_dualopt` | CMFA+DualOpt | done | 53.088 | 93.707 | 53.223 | - | 42.680 | 53.475 | 68.531 | 61 | 52.614 | -1.256 | 11.13M |
| 11 | `dentex_mpf_dualopt` | MPF+DualOpt | done | 53.079 | 93.268 | 53.271 | - | 42.946 | 53.419 | 68.447 | 52 | 52.339 | -1.265 | 11.83M |
| 12 | `dentex_cmfa_hooa_cedv2` | CMFA+HOOA+CEDv2 | done | 53.037 | 93.813 | 53.392 | - | 40.579 | 53.470 | 67.009 | 134 | 52.636 | -1.307 | 16.50M |

## COCO-/home/wuxi/coco

| Rank | Experiment | Method | Status | Best AP | AP50 | AP75 | APs | APm | APl | AR100 | Best Epoch | Last AP | Delta vs Best | Params |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `coco_cmfa_hooa_cedv2_dualopt` | CMFA+HOOA+CEDv2+DualOpt | done | 54.071 | 84.780 | 59.676 | 42.500 | 39.075 | 55.820 | 68.031 | 153 | 53.976 | +0.000 | 16.52M |
| 2 | `coco_cmfa_dualopt` | CMFA+DualOpt | done | 54.057 | 83.957 | 58.477 | 42.500 | 35.514 | 55.804 | 68.133 | 131 | 54.005 | -0.014 | 11.15M |
| 3 | `coco_cmfa_hooa_cedv2` | CMFA+HOOA+CEDv2 | done | 54.009 | 83.952 | 58.928 | 45.000 | 37.166 | 55.679 | 67.751 | 138 | 53.665 | -0.062 | 16.52M |
| 4 | `coco_cmfa_hooa_cedv2_spatial_shape` | CMFA+HOOA+CEDv2+SpatialShape | done | 53.752 | 85.083 | 59.409 | 42.500 | 35.861 | 55.480 | 67.515 | 134 | 53.639 | -0.319 | 16.59M |
| 5 | `coco_cmfa_hooa_prototype` | CMFA+HOOA+Prototype | done | 53.466 | 84.322 | 58.472 | 41.250 | 38.036 | 54.880 | 67.931 | 124 | 53.311 | -0.605 | 12.51M |
| 6 | `coco_cmfa_hooa_dualopt` | CMFA+HOOA+DualOpt | done | 53.447 | 83.481 | 58.466 | 37.500 | 37.117 | 55.202 | 66.664 | 132 | 53.408 | -0.624 | 12.45M |
| 7 | `coco_cmfa_hooa_spatial_shape_dualopt` | CMFA+HOOA+SpatialShape+DualOpt | done | 53.118 | 82.418 | 59.504 | 43.750 | 35.074 | 54.917 | 68.246 | 131 | 52.994 | -0.952 | 12.51M |
| 8 | `coco_mpf_dualopt` | MPF+DualOpt | done | 53.066 | 83.232 | 57.245 | 42.500 | 37.833 | 54.545 | 67.788 | 156 | 52.796 | -1.005 | 11.86M |
| 9 | `coco_cmfa_hooa_cedv2_spatial_shape_dualopt` | CMFA+HOOA+CEDv2+SpatialShape+DualOpt | done | 53.035 | 83.599 | 58.906 | 41.667 | 35.245 | 54.725 | 66.930 | 136 | 52.955 | -1.036 | 16.59M |
| 10 | `coco_spatial_shape_dualopt` | SpatialShape+DualOpt | done | 52.744 | 82.676 | 58.661 | 42.500 | 35.451 | 54.476 | 67.783 | 144 | 52.714 | -1.327 | 10.70M |
| 11 | `coco_cmfa_hooa_spatial_shape` | CMFA+HOOA+SpatialShape | done | 52.490 | 82.289 | 57.385 | 45.000 | 36.068 | 54.094 | 67.550 | 115 | 52.383 | -1.581 | 12.51M |
| 12 | `coco_prototype_dualopt` | Prototype+DualOpt | done | 52.430 | 82.982 | 57.091 | 45.000 | 35.587 | 54.097 | 66.707 | 148 | 52.398 | -1.640 | 10.70M |

## Files

- CSV: `/home/wuxi/Dent-FINE/outputs/ablation_combo_modules_cross_dataset/ablation_summary.csv`
- Markdown: `/home/wuxi/Dent-FINE/outputs/ablation_combo_modules_cross_dataset/ablation_summary.md`
