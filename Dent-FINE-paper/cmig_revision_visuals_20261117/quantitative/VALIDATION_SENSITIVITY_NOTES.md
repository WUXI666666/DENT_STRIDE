# Validation seed selection and exact-duplicate sensitivity

## Completed analysis

This is a new, CPU-only analysis of saved predictions. No training, GPU inference, changes to annotations, checkpoint reselection, or changes to the main paper's original metrics were made. All ten available Pano/Mobile runs (five per configuration, seeds 0–4) are retained. It provides an audit of robustness to a small, identified split-overlap issue; it is not an external validation experiment.

`Figure_Validation_sensitivity.pdf` is a 174 × 88 mm vector figure with embedded Times New Roman. Filled markers identify the three seeds in the reported mean; open markers identify the other two completed runs. Both directions of change are shown. The preview is `Figure_Validation_sensitivity_preview.png`.

## Exact source of seed selection

The archived source file is:

`/home/wuxi/Dent-FINE-paper/materials/cmig_selected_seed_revision_2026-10-12/quantitative/quantitative_data.json`

Its `selection_policy` states verbatim:

> User-selected after inspecting five completed seeds; selected-three summaries are exploratory, not prespecified primary evidence.

The accompanying `build_quantitative.py` defines `SPEC` at lines 35–37:

```python
('Pano', 'pano_spatialshape_original', [1,3,4], 3, 'DENTEX2023')
('Mobile', 'mobile_proto_k4_tau0p50', [0,2,3], 2, 'Mobile')
('DSLR', 'dslr_cmfa_extra_decay_only', [0,2,3], 3, 'DSLR')
```

The same script iterates `for seed in range(5)` (line 116), marks the selection flag (line 125), and computes `selected`, `fixed012`, and `all5` summaries (lines 143–147). Its own source-figure caption at line 186 explicitly describes the selected-three summaries as exploratory and not prespecified repeated-seed evidence.

The Pano and Mobile selected means, independently reproduced from the archived predictions, are **54.204700752663086** and **55.45868689603236**, rounding exactly to the Word values **54.205** and **55.459**. The selected three also happen to be the three highest original-split AP values among the archived five for each of these configurations. Calling these three a prespecified random-seed replication would therefore be inaccurate. The saved selection flag was fixed before the new sensitivity analysis; no seeds were selected by the exclusion results.

The archived DSLR selected mean is **72.9077709061321**, which does **not** match the Word value 73.219. This analysis does not replace or relabel the Word DSLR result and does not blend that older configuration into it.

## Original and restricted validation results

All values are percentages, except ΔAP in percentage points; uncertainties are sample standard deviations across seeds (denominator n−1), not confidence intervals.

| Cohort | Subset | Seeds | Original AP | Restricted AP | Paired ΔAP |
|---|---|---|---:|---:|---:|
| Pano | All completed | 0,1,2,3,4 | 53.979 ± 0.337 | 54.558 ± 0.428 | +0.579 ± 0.132 |
| Pano | Reported | 1,3,4 | 54.205 ± 0.144 | 54.854 ± 0.193 | +0.650 ± 0.071 |
| Mobile | All completed | 0,1,2,3,4 | 55.292 ± 0.301 | 55.039 ± 0.319 | −0.253 ± 0.031 |
| Mobile | Reported | 0,2,3 | 55.459 ± 0.275 | 55.215 ± 0.286 | −0.244 ± 0.012 |

| Cohort | Seed | In reported three? | Original AP | Restricted AP | ΔAP |
|---|---:|---|---:|---:|---:|
| Pano | 0 | No | 53.764554 | 54.131478 | +0.366924 |
| Pano | 1 | Yes | 54.095353 | 54.663498 | +0.568145 |
| Pano | 2 | No | 53.517166 | 54.095738 | +0.578572 |
| Pano | 3 | Yes | 54.367697 | 55.048948 | +0.681251 |
| Pano | 4 | Yes | 54.151053 | 54.850620 | +0.699567 |
| Mobile | 0 | Yes | 55.582417 | 55.343713 | −0.238704 |
| Mobile | 1 | No | 55.007941 | 54.703342 | −0.304599 |
| Mobile | 2 | Yes | 55.649635 | 55.413721 | −0.235914 |
| Mobile | 3 | Yes | 55.144009 | 54.886447 | −0.257562 |
| Mobile | 4 | No | 55.073709 | 54.845443 | −0.228266 |

## Verification and exclusion rule

The existing audit identified exact train/validation duplicates by SHA256 of `struct.pack('>II', width, height)` followed by decoded RGB pixel bytes (PIL conversion to RGB; no resizing or EXIF transpose). The current analysis independently rehashed **all 11 involved image files** and confirmed their group hashes. Current train/validation annotation-file hashes were also identical to the audit. It then removed only the affected validation image IDs:

- Pano: ID 238 (`train_412.png`), matching training IDs 327 and 616. Validation changed from 64 to 63 images and 1,861 to 1,832 annotations.
- Mobile: IDs 6, 88, 110, 170, matching training IDs 405, 820, 630, 12 respectively. Validation changed from 220 to 216 images and 2,459 to 2,407 annotations.
- No originally supported category lost all validation support after these exclusions.

For every seed, the current checkpoint and configuration SHA256 values matched the native evaluation metadata, and the native metric-file SHA256 matched the saved selection artifact. Saved `evaluation/predictions_coco.json.gz` files were loaded directly. Standard `pycocotools.COCOeval(..., 'bbox')` was run first over all original image IDs and then over retained image IDs, using unchanged category IDs, COCO IoU thresholds, and `maxDets = [1,10,100]`. Recomputed original AP, AP50, AP75, and AR100 matched native logged values with maximum absolute error **1.4210854715202004e−14 percentage points**. All ten original evaluations passed before the corresponding restricted calculation.

## Suggested Methods text

The reported Pano and Mobile means use three runs selected from five completed seeds (Pano: 1, 3, and 4; Mobile: 0, 2, and 3), and are treated as descriptive selected-run summaries. We additionally report all five runs to show the effect of this selection. A post-hoc sensitivity analysis excluded validation images with exact decoded-pixel duplicates in the corresponding training split: one of 64 Pano images and four of 220 Mobile images. The original AP values were first reproduced from saved predictions for every seed, after which COCO evaluation was repeated on the retained images without retraining or checkpoint reselection. Sample standard deviations were calculated across seeds. This analysis retains the original model-selection history and is not an independent test-set evaluation.

## Suggested Results text

Across all five completed seeds, original-split AP was 53.979 ± 0.337% on Pano and 55.292 ± 0.301% on Mobile, compared with the reported selected-three means of 54.205 ± 0.144% and 55.459 ± 0.275%, respectively. Excluding the identified exact training-image duplicates from validation changed all-five AP to 54.558 ± 0.428% for Pano and 55.039 ± 0.319% for Mobile. The mean paired changes were +0.579 ± 0.132 and −0.253 ± 0.031 percentage points, respectively. Thus the direction of this sensitivity differed by cohort. These results describe the existing checkpoints on restricted validation subsets; they do not estimate performance after retraining on overlap-free partitions.

## Suggested Discussion text

The Pano and Mobile results use internal validation partitions, and their reported three-seed summaries were selected after inspecting five completed runs. Reporting all completed seeds makes this selection explicit but does not remove its bias. Exact train/validation image overlap was found in one Pano and four Mobile validation images. Excluding those images produced a small decrease on Mobile and an increase on Pano in the existing checkpoints; this post-hoc check neither removes the models' prior exposure nor rules out near duplicates or unidentified patient overlap. Evaluation on independently defined, patient-disjoint external data remains necessary.

## Suggested figure caption

Post-hoc sensitivity to exact train/validation image overlap, with all five completed seeds shown. (A) Pano: the original 64-image validation split and the 63-image subset after excluding one validation image with an exact decoded-pixel duplicate in training. (B) Mobile: the original 220-image validation split and the 216-image subset after excluding four such images. Lines connect unchanged checkpoints from the same seed; filled markers identify the seeds included in the reported three-run mean and open markers identify the remaining runs. Text below each panel gives AP mean ± sample standard deviation for all five and for the reported three. ΔAP denotes the mean ± sample standard deviation of the five paired changes, in percentage points. Original metrics were reproduced from saved predictions before exclusion; no retraining or checkpoint reselection was performed. The restricted evaluation remains a post-hoc internal-validation analysis.

## Reproduction and machine-readable files

1. `/home/wuxi/miniconda3/envs/deimv2/bin/python recompute_duplicate_sensitivity.py` (CPU only, pycocotools 2.0.11; no torch import).
2. `python plot_duplicate_sensitivity.py` (Matplotlib, NumPy, local Times New Roman files).

Artifacts:

- `duplicate_sensitivity.json`: source paths/hashes, complete metrics for each seed, selection policy, cohort counts, and all-five/reported-three/fixed-0,1,2 summaries.
- `duplicate_sensitivity_per_seed.csv`: every seed and its AP/AP50/AP75/AR100 before, after, and difference.
- `duplicate_sensitivity_summary.csv`: all-five/reported-three/fixed-0,1,2 summary means and sample SDs.
- `verified_duplicate_members.csv`: eleven source image paths, memberships, and verified pixel hashes.
- `duplicate_sensitivity_evaluation.log`: all twenty full COCO evaluation reports.

The underlying source run directory is `materials/cmig_selected_configs_multiseed_2026-10-11/runs/{pano_spatialshape_original,mobile_proto_k4_tau0p50}/seed_{0,1,2,3,4}/`. Dataset-audit source is `materials/cmig_controlled_2026-09-29/audit/dataset_audit.json`.
