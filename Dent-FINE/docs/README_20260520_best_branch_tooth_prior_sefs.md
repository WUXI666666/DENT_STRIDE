# 2026-05-20 Best Branch Tooth-Prior Ablations And SEFS Integration

## 1. Dataset Check

Previous main ablations `abl00` to `abl105` were run with `tools/run_ablation_all_new_modules.sh`.
That script defaults to `configs/deim_dfine/deim_hgnetv2_s_coco.yml`, which includes
`configs/dataset/coco_detection.yml`. The actual dataset is the phone-shot COCO-format set:

- Train: `/home/wuxi/coco/train`, annotation `/home/wuxi/coco/train/annotations_fdi_types.json`
- Val: `/home/wuxi/coco/valid`, annotation `/home/wuxi/coco/valid/annotations_fdi_types.json`
- Stats: 878 train images, 9837 train boxes, 220 val images, 2459 val boxes, 52 FDI categories

The DENTEX2023 path is now used by the `abl110` to `abl113` tooth-prior group and the new
best-branch matrix:

- Train: `/home/wuxi/DENTEX2023/train2017`, annotation `/home/wuxi/DENTEX2023/annotations/instances_train2017.json`
- Val: `/home/wuxi/DENTEX2023/val2017`, annotation `/home/wuxi/DENTEX2023/annotations/instances_val2017.json`
- Stats: 570 train images, 16234 train boxes, 64 val images, 1861 val boxes, 32 categories

## 2. Paper And Code Review

Sources reviewed:

- SEGD: https://github.com/Kust-lp/SEGD and https://arxiv.org/abs/2604.22886
- FSDETR: https://github.com/YT3DVision/FSDETR and https://arxiv.org/abs/2604.14884
- DF2Net: https://github.com/Mw-yang/DF2Net
- MultiNeX: https://github.com/albrateanu/multinex
- SLURPP: https://tianfuwang.github.io/slurpp and https://arxiv.org/abs/2507.07878
- SDCoNet: https://github.com/qiruo-ya/SDCoNet and https://arxiv.org/abs/2601.12507
- DCL: https://github.com/asdww0721ww/DCL
- IDGBR: https://github.com/KeyanHu-git/IDGBR and https://arxiv.org/abs/2507.01573

Useful ideas kept:

- SEGD: kept the structural-entropy idea as a local entropy proxy gate over detector feature maps.
- FSDETR: kept frequency-spatial enhancement, especially low/high-frequency feature separation and edge/detail preservation.
- DF2Net: its public repository currently says the code will be released later, so no code was copied; kept only the safe idea of differential cross-level fusion.
- SDCoNet: kept saliency-driven focusing as a learned feature gate, without adding an SR branch or gradient-routing dependency.
- IDGBR: kept boundary refinement as a lightweight high-pass/detail branch, without adding latent diffusion.

Not integrated as new heavy dependencies:

- MultiNeX: this project already has a `MultiPriorRetinexRefiner` behind `HybridEncoder.use_multinex_prior`; prior ablation `abl32_multinex_prior` was weaker than the best branches, so I did not force it into the new default matrix.
- SLURPP: underwater latent diffusion restoration is task/domain heavy for dental detection, so only the broader "single-step prior, avoid iterative restoration" lesson was considered.
- DCL: cross-modality cardiac segmentation causality is not directly aligned with single-modality tooth detection; no DCL-specific training objective was added.

## 3. Integrated Module

Added `StructuralEntropyFrequencySaliencyRefiner` in `engine/deim/hybrid_encoder.py`.

Switches:

- `HybridEncoder.use_structural_entropy_frequency_refiner`
- `HybridEncoder.sefs_res_init`

The new SEFS block is residual and disabled by default. It combines:

- local structural-entropy proxy gate
- low/high-frequency split
- cross-level differential context
- saliency gate
- boundary/detail high-pass update

This gives a detector-native version of the useful parts from the reviewed work while avoiding large restoration, segmentation, or diffusion pipelines.

## 4. Ablation Matrix Added

New script:

- `tools/run_ablation_best_branches_tooth_priors.sh`

New summary rows:

- `abl114` to `abl121`: `CMFA+HOOA+CED-v2` plus baseline/factor/prototype/spatial_shape on phone and DENTEX
- `abl122` to `abl129`: `CMFA+HOOA` plus the same stages
- `abl130` to `abl137`: `CMFA` plus the same stages
- `abl138` to `abl145`: `SMWG-SpectralWavelet` plus the same stages
- `abl146` to `abl153`: new `SEFS` plus the same stages

The script refreshes:

- `outputs/ablation_all_new_modules/ablation_summary.md`
- `outputs/ablation_all_new_modules/ablation_summary.csv`
- `outputs/ablation_all_new_modules/ablation_summary.json`

Set `INCLUDE_SEFS=0` if you only want the four branch groups requested originally.

## 5. Files Touched

- `engine/deim/hybrid_encoder.py`: added SEFS module and encoder switch.
- `configs/base/dfine_hgnetv2.yml`: added default-off SEFS config keys.
- `tools/summarize_ablation_all_new_modules.py`: registered `abl114` to `abl153` and generalized per-branch tooth-prior delta AP.
- `tools/run_ablation_best_branches_tooth_priors.sh`: new serial runner for best-branch tooth-prior matrix.
- `outputs/ablation_all_new_modules/ablation_summary.*`: refreshed total summary table.

## 6. Commands

Foreground 1-epoch smoke train on GPU0:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 EPOCHS=1 MAX_RUNS=1 CLEAN_OUTPUT=1 BASE_OUTPUT_DIR=outputs/ablation_best_branch_smoke \
  bash tools/run_ablation_best_branches_tooth_priors.sh
```

Full serial ablation on GPU0, refreshing the total table after every run:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 SKIP_EXISTING=1 bash tools/run_ablation_best_branches_tooth_priors.sh
```

Run only the four requested branch groups, excluding SEFS:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 INCLUDE_SEFS=0 SKIP_EXISTING=1 bash tools/run_ablation_best_branches_tooth_priors.sh
```

## 7. Verification

Completed checks:

- `python -m py_compile engine/deim/hybrid_encoder.py tools/summarize_ablation_all_new_modules.py`
- dry-run of `tools/run_ablation_best_branches_tooth_priors.sh` with `MAX_RUNS=2`
- summary refresh for `outputs/ablation_all_new_modules`
- YAML model instantiation with `HybridEncoder.use_structural_entropy_frequency_refiner=True`
- direct SEFS forward smoke test on three FPN feature levels
