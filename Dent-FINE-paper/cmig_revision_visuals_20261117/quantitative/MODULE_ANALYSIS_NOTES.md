# Controlled module analysis: manuscript-ready text and provenance

## Figure and files

- `Figure_Module_controlled_AP.pdf`: 174 × 69 mm, embedded Times New Roman, vector plot; approximately 48 KB.
- `Figure_Module_controlled_AP_preview.png`: preview.
- `paired_ap.csv`: every AP pair shown in the figure.
- `all_primary_secondary_endpoints.csv`: AP, AP50, AP75, and AR100 for every pair, including negative changes in secondary endpoints.
- `recomputed_summary.json`: exact numerical summaries, config differences, and paths/hashes for all 14 runs.
- `recompute_and_plot.py`: verifies source checkpoints/configurations/metrics and regenerates the figure with NumPy, PyYAML, and Matplotlib. No training, model inference, or GPU use.

## Suggested Methods paragraph

We additionally examined controlled module contrasts using archived runs with identical data partitions and matched training seeds. For CMFA, three paired runs (seeds 0, 1, and 2) were available under each of two optimizer settings. Within each pair, the configuration differed only in whether CMFA fusion was enabled. We report the AP difference for every pair and the mean and sample standard deviation of the paired differences. These controlled reference settings were analyzed separately from the final selected configuration. In the DualOpt reference setting, both the extra-decay and initialization-rollback terms were active. Prototype supervision was evaluated in a separate Mobile seed-0 contrast; the enabled branch used four prototypes per class and temperature 0.50, with prototype-score fusion disabled during inference. No statistical significance claim was made from these limited runs.

## Suggested Results paragraph

CMFA increased validation AP in all six matched comparisons. The mean paired AP gain was 0.498 ± 0.145 percentage points under AdamW and 0.513 ± 0.091 under the DualOpt reference setting (three seeds per setting; mean ± sample standard deviation). In the separate Mobile seed-0 contrast, enabling prototype supervision increased AP from 54.918% to 55.683% (+0.765 percentage points). This single-seed comparison supports the prototype-supervision result for that setting but does not estimate its variability across seeds. The secondary endpoints did not improve uniformly across all comparisons.

## Suggested figure caption

Controlled validation-AP comparisons for CMFA and prototype supervision. (A, B) Matched DSLR runs with and without CMFA under AdamW and a DualOpt reference setting. Connected points denote the same seed; all three available pairs are shown in each setting. The reported ΔAP values are the mean ± sample standard deviation of the three paired differences, in percentage points. The DualOpt reference includes extra decay and initialization rollback and is distinct from the final extra-decay-only DSLR configuration. (C) Separate Mobile seed-0 comparison without and with prototype supervision (four prototypes per class, temperature 0.50, no inference-time prototype-score fusion). No variability estimate is available for this single-seed comparison. All values use the original internal validation partitions.

## Exact summaries

| Contrast | Seeds | AP without (mean) | AP with (mean) | Paired ΔAP (mean ± sample SD) |
|---|---:|---:|---:|---:|
| CMFA, AdamW, DSLR | 0, 1, 2 | 72.465502 | 72.963913 | +0.498411 ± 0.145139 |
| CMFA, DualOpt reference, DSLR | 0, 1, 2 | 72.364348 | 72.877842 | +0.513493 ± 0.091020 |
| Prototype, Mobile | 0 | 54.918159 | 55.682899 | +0.764740 (single seed) |

## Interpretation constraints

- The CMFA evidence is confined to the two specified controlled reference settings. It cannot be relabeled as a direct ablation of the final extra-decay-only DSLR recipe.
- The prototype contrast toggles the prototype branch and its auxiliary loss. The stored temperature is 0.20 in the disabled branch and 0.50 in the enabled branch; the former has no active role because that branch is disabled.
- These runs do not demonstrate diversity among the four prototype vectors. The separate prototype audit found high within-class similarity, so claims of four distinct learned tooth morphologies are unsupported.
- AP gains are descriptive. Neither a single seed nor three paired seeds establishes external clinical generalization, and no significance test is used here.
- The global `known_content_overlap_retained` flag does not imply overlap in every cohort. The archived image audit found no train/validation exact-image overlap in DSLR and no patient-ID overlap for its recorded identifiers. It found four Mobile and one Pano validation images with exact training-image duplicates; a separate sensitivity analysis retains this distinction.

## Source scope

CMFA: `materials/effectiveness_evidence_2026-10-30/controlled_evidence.json`, checked against `materials/cmig_controlled_2026-09-29/runs/{dslr_base_adamw,dslr_cmfa_adamw,dslr_base_dualopt,dslr_cmfa_dualopt}/seed_{0,1,2}/metrics_summary.json` and their real configurations/checkpoints.

Prototype: `materials/module_cam_gallery_2026-10-30/Prototype_{before,after}_capture.json`, checked against `materials/cmig_best3_seeds_2026-10-08/runs/internal_mobile_{proto_off,proto_k4_tau0p5}/seed_0/native_coco_metrics.json` and their real configurations/checkpoints.

Dataset audit: `materials/cmig_controlled_2026-09-29/audit/dataset_audit.json`.
