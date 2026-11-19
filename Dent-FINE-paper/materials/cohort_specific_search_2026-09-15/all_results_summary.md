# Cohort-specific search results (seed 0, 160 epochs)

All AP/AP50/AP75/AR100 values are percentages. New candidates were selected by independently re-evaluated validation COCO AP; the DENTEX row is the preserved AP 54.385 reference and was not retrained.

## Selected/preserved results

| Cohort | Model/job | Input/training recipe | AP | AP50 | AP75 | AR100 | Best epoch | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Smartphone | phone_dualopt_C05 | fixed1024; M=0.0; MixUp=0.0; crop=0.8; zoom=0.5 | 54.907 | 85.415 | 59.801 | 68.740 | 148 | 10,629,283 |
| DENTEX2023 | R01_fixed640_mosaic0_mixup0p5 | fixed640; M=0.0; MixUp=0.5; crop=0.8; zoom=0.5 | 54.385 | 93.744 | 56.481 | 69.765 | 156 | 10,650,249 |
| DSLR | dslr_cmfa_dualopt_C02 | legacy640_multiscale; M=0.0; MixUp=0.0; crop=0.8; zoom=0.5 | 73.115 | 89.149 | 84.341 | 83.880 | 150 | 11,153,963 |
| Smartphone | phone_a07_proto_aux_C05 | fixed1024; M=0.0; MixUp=0.0; crop=0.8; zoom=0.5 | 55.701 | 84.441 | 64.019 | 70.071 | 139 | 12,513,162 |

## All runs

| Cohort | Job | Phase | Profile | M | MixUp | Crop | Zoom | AP | AP50 | AP75 | AR100 | Best epoch |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DENTEX2023 | R01_fixed640_mosaic0_mixup0p5 | preserved_reference | fixed640 | 0.0 | 0.5 | 0.8 | 0.5 | 54.385 | 93.744 | 56.481 | 69.765 | 156 |
| Smartphone | FINAL_phone_dualopt | final_main | fixed640 | 0.0 | 0.5 | 0.8 | 0.5 | 52.193 | 80.878 | 58.082 | 67.798 | 160 |
| DSLR | FINAL_dslr_cmfa_dualopt | final_main | fixed640 | 0.0 | 0.5 | 0.8 | 0.5 | 72.708 | 88.815 | 83.076 | 81.898 | 145 |
| Smartphone | FINAL_phone_a07_proto_aux | final_a07 | fixed640 | 0.0 | 0.5 | 0.8 | 0.5 | 53.708 | 83.258 | 61.464 | 67.602 | 145 |
| Smartphone | phone_dualopt_C01 | core_grid | legacy640_multiscale | 0.0 | 0.5 | 0.8 | 0.5 | 53.064 | 83.088 | 58.527 | 67.527 | 156 |
| DSLR | dslr_cmfa_dualopt_C01 | core_grid | legacy640_multiscale | 0.0 | 0.5 | 0.8 | 0.5 | 72.624 | 88.714 | 83.209 | 81.124 | 155 |
| Smartphone | phone_a07_proto_aux_C01 | core_grid | legacy640_multiscale | 0.0 | 0.0 | 0.8 | 0.5 | 52.339 | 82.632 | 57.354 | 66.421 | 113 |
| Smartphone | phone_dualopt_C02 | core_grid | legacy640_multiscale | 0.0 | 0.0 | 0.8 | 0.5 | 53.652 | 83.104 | 60.640 | 66.942 | 150 |
| DSLR | dslr_cmfa_dualopt_C02 | core_grid | legacy640_multiscale | 0.0 | 0.0 | 0.8 | 0.5 | 73.115 | 89.149 | 84.341 | 83.880 | 150 |
| Smartphone | phone_a07_proto_aux_C02 | core_grid | legacy640_multiscale | 0.0 | 0.5 | 0.8 | 0.5 | 53.991 | 83.802 | 60.141 | 67.698 | 158 |
| Smartphone | phone_dualopt_C03 | core_grid | fixed640 | 0.0 | 0.0 | 0.8 | 0.5 | 52.701 | 82.913 | 58.539 | 66.720 | 148 |
| DSLR | dslr_cmfa_dualopt_C03 | core_grid | fixed640 | 0.0 | 0.0 | 0.8 | 0.5 | 72.945 | 88.824 | 83.752 | 82.784 | 131 |
| Smartphone | phone_a07_proto_aux_C03 | core_grid | fixed640 | 0.0 | 0.0 | 0.8 | 0.5 | 53.520 | 83.399 | 59.194 | 67.513 | 62 |
| Smartphone | phone_dualopt_C04 | core_grid | fixed1024 | 0.0 | 0.5 | 0.8 | 0.5 | 54.678 | 84.506 | 59.045 | 69.105 | 135 |
| DSLR | dslr_cmfa_dualopt_C04 | core_grid | fixed1024 | 0.0 | 0.5 | 0.8 | 0.5 | 73.021 | 88.772 | 82.985 | 81.117 | 160 |
| Smartphone | phone_a07_proto_aux_C04 | core_grid | fixed1024 | 0.0 | 0.5 | 0.8 | 0.5 | 55.101 | 83.656 | 62.695 | 69.727 | 138 |
| Smartphone | phone_dualopt_C05 | core_grid | fixed1024 | 0.0 | 0.0 | 0.8 | 0.5 | 54.907 | 85.415 | 59.801 | 68.740 | 148 |
| DSLR | dslr_cmfa_dualopt_C05 | core_grid | fixed1024 | 0.0 | 0.0 | 0.8 | 0.5 | 72.682 | 88.083 | 83.035 | 81.633 | 95 |
| Smartphone | phone_a07_proto_aux_C05 | core_grid | fixed1024 | 0.0 | 0.0 | 0.8 | 0.5 | 55.701 | 84.441 | 64.019 | 70.071 | 139 |
| Smartphone | phone_dualopt_R01_crop0p5 | single_factor_refinement | fixed1024 | 0.0 | 0.0 | 0.5 | 0.5 | 53.392 | 82.040 | 61.225 | 68.385 | 150 |
| Smartphone | phone_dualopt_R02_zoom0 | single_factor_refinement | fixed1024 | 0.0 | 0.0 | 0.8 | 0.0 | 51.184 | 79.787 | 55.756 | 65.866 | 114 |
| Smartphone | phone_dualopt_R03_mosaic0p25 | single_factor_refinement | fixed1024 | 0.25 | 0.0 | 0.8 | 0.5 | 54.753 | 83.760 | 61.113 | 68.188 | 119 |
| DSLR | dslr_cmfa_dualopt_R01_crop0p5 | single_factor_refinement | legacy640_multiscale | 0.0 | 0.0 | 0.5 | 0.5 | 72.382 | 88.306 | 82.954 | 82.237 | 104 |
| DSLR | dslr_cmfa_dualopt_R02_zoom0 | single_factor_refinement | legacy640_multiscale | 0.0 | 0.0 | 0.8 | 0.0 | 71.064 | 87.648 | 80.852 | 81.469 | 133 |
| DSLR | dslr_cmfa_dualopt_R03_mosaic0p25 | single_factor_refinement | legacy640_multiscale | 0.25 | 0.0 | 0.8 | 0.5 | 72.898 | 89.467 | 83.932 | 82.169 | 143 |
| Smartphone | phone_a07_proto_aux_R01_crop0p5 | single_factor_refinement | fixed1024 | 0.0 | 0.0 | 0.5 | 0.5 | 54.437 | 83.217 | 59.309 | 68.678 | 115 |
| Smartphone | phone_a07_proto_aux_R02_zoom0 | single_factor_refinement | fixed1024 | 0.0 | 0.0 | 0.8 | 0.0 | 50.659 | 78.680 | 55.282 | 66.544 | 80 |
| Smartphone | phone_a07_proto_aux_R03_mosaic0p25 | single_factor_refinement | fixed1024 | 0.25 | 0.0 | 0.8 | 0.5 | 55.158 | 83.953 | 62.457 | 69.672 | 160 |
