# 2026-05-24 Single-Module Review And COCO Ablation Script

## 1. Are The Previous Papers Only Useful Through SEFS?

No. The current table says SEFS is useful as a conservative feature refiner, but it is not the only useful line and it is not the strongest standalone result on the phone dataset.

Observed from `outputs/ablation_all_new_modules/ablation_summary.md`:

- `CMFA+HOOA` remains the strongest old encoder pair: `abl34` AP 0.5426 on `/home/wuxi/coco`.
- `CMFA+HOOA+CED-v2` is still a strong branch: `abl72` AP 0.5423, and the new DENTEX spatial-shape branch `abl121` reaches AP 0.5445.
- `SMWG-SpectralWavelet` has real signal: `abl87` AP 0.5368 versus `abl00` AP 0.5343. In the tooth-prior matrix, phone `SMWG+prototype` improves over its own baseline by +0.0067.
- `SEFS` is not a universal win: phone `SEFS+prototype` improves over phone SEFS baseline by +0.0053, but pure phone SEFS baseline is AP 0.5288, lower than the original baseline. Its useful parts should stay as an ablation module, not as default.
- `MultiNeX` was tested already and is weak here: `abl32` AP 0.5266 and `abl37` AP 0.5310.
- Heavy restoration/diffusion/cross-modality segmentation ideas from SLURPP and DCL are not a good direct fit for this detector. They add task mismatch and training cost without a clean detection-side supervision target.

So the answer is: SEFS is one useful distilled module, but the stronger evidence still belongs to CMFA/HOOA/CED-v2 and some frequency/detail modules such as SMWG. SEFS should be kept as optional, not treated as the only valid module.

## 2. New Sources Reviewed

- Rot-E ASISR: https://github.com/XieQi2015/Equivariant-ASISR
- DualOpt paper: https://arxiv.org/abs/2604.22838
- OLOR code: https://github.com/qklee-lz/OLOR-AAAI-2024
- HEDP paper: https://arxiv.org/abs/2605.05776
- HEDP code: https://github.com/dannis97500/HEDP
- SAMPG-SSTM-AMHCD actual repo found from GitHub search: https://github.com/Jiahuiqu/SAMPG-SSTM-AMHCD
- User-provided SAMPG alias currently returns 404: https://github.com/Jiahuiqu/SAMPG-SST-MCDNet

## 3. What Was Integrated

Added `RotEquivariantDetailRefiner`.

- Source idea: Rot-E ASISR emphasizes end-to-end rotation equivariance.
- Detector adaptation: shared local-detail branch over identity, horizontal flip, vertical flip, and rot90 views, followed by inverse-transform averaging.
- Why useful: tooth crowns and roots have local shape symmetries; this gives equivariant detail consistency without adding a super-resolution or INR decoder.

Added `DualOptAdamW`.

- Source idea: DualOpt uses real-time layer-wise weight decay for scratch training and rollback for fine-tuning.
- Detector adaptation: keep AdamW and add small post-step real-time decay plus initialization rollback.
- Why useful: D-FINE uses a pretrained HGNetv2 backbone, so a mild rollback can reduce forgetting during small dental fine-tuning.

Added `HybridEnergyDistancePromptRefiner`.

- Source idea: HEDP uses hybrid energy-distance prompts for domain incremental learning.
- Detector adaptation: learn feature prompts per FPN level and gate local features by energy-distance cue.
- Why useful: phone dental images have exposure/device/background shifts, but full domain-incremental replay is unnecessary here.

Added `SAMPriorSSTScanRefiner`.

- Source idea: SAM-prior-guided spatial/spectral/temporal Mamba for affine-misregistered hyperspectral change detection.
- Detector adaptation: no temporal or hyperspectral branch; keep a SAM-like saliency prior, channel spectral gate, and bidirectional spatial scans.
- Why useful: approximates long-range arch context and salient tooth-region scanning without adding Mamba or SAM dependencies.

Added `SEGDStructuralEntropyRefiner`.

- Source idea: SEGD decouples degradation with structural entropy guidance.
- Detector adaptation: local structural entropy, detail energy, and contrast evidence gate a mild low/detail residual update.
- Why useful: phone-shot dental images can have blur, glare, and nonuniform contrast; this tests the degradation-decoupling part directly.

Added `FSDETRFrequencySpatialRefiner`.

- Source idea: FSDETR uses frequency-spatial enhancement for small object detection.
- Detector adaptation: Fourier band modulation plus dilated spatial detail branch with a learned frequency/spatial mixing gate.
- Why useful: individual teeth and boundaries behave like small, high-frequency structures inside a larger jaw image.

Added `DF2NetDifferentialDynamicFusionRefiner`.

- Source idea: DF2Net uses differential-feature dynamic fusion for change detection.
- Detector adaptation: uses cross-level FPN context difference as a pseudo-differential signal and dynamically fuses context/detail.
- Why useful: tooth instances differ subtly from adjacent gum/background and neighboring teeth.

Added `SDCoNetSaliencyCollaborationRefiner`.

- Source idea: SDCoNet uses saliency-driven multi-task collaboration for remote-sensing detection.
- Detector adaptation: learned saliency mask drives shared and tooth-specific feature branches, without adding a new segmentation task.
- Why useful: keeps computation focused on tooth-like regions while preserving context.

Added `IDGBRBoundaryRefinementRefiner`.

- Source idea: IDGBR refines segmentation boundaries by integrating discriminative and generative cues.
- Detector adaptation: semantic context from the deepest FPN level gates boundary high-pass residuals.
- Why useful: detection AP75 is sensitive to tight tooth boundaries.

Not integrated as new modules:

- MultiNeX is already present as `HybridEncoder.use_multinex_prior`; existing `abl32` and `abl37` are weak on this dataset, so I did not duplicate it.
- SLURPP's full latent diffusion restoration path is too heavy and domain-specific for detector training; a direct transplant would likely change the input distribution rather than improve the detector.
- DCL's cross-modality cardiac segmentation objective needs modality labels and segmentation supervision; this repo already has lighter domain-shift adapters, so I did not add another causal-training loss without matching supervision.

## 4. Files Changed

- `engine/deim/hybrid_encoder.py`: added Rot-E, HEDP prompt, SAMPG/SST, SEGD, FSDETR, DF2Net, SDCoNet, and IDGBR refiners plus switches.
- `engine/optim/optim.py`: added `DualOptAdamW`.
- `configs/base/dfine_hgnetv2.yml`: added default-off config keys for the new refiners.
- `tools/summarize_ablation_all_new_modules.py`: registered `abl154` to `abl162`.
- `tools/run_ablation_20260524_single_modules_coco.sh`: new serial single-module ablation runner for `/home/wuxi/coco`.
- `outputs/ablation_all_new_modules/ablation_summary.*`: refreshed, with `abl154` to `abl162` pending until training runs.

## 5. Ablation Rows

- `abl154_coco_rot_equivariant_detail`: Rot-E detail equivariance.
- `abl155_coco_dualopt_adamw`: DualOpt/OLOR-style optimizer.
- `abl156_coco_hedp_prompt`: HEDP energy-distance prompt.
- `abl157_coco_sampg_sst`: SAMPG/SST prior-guided spatial-spectral scan.
- `abl158_coco_segd_entropy`: SEGD structural entropy/degradation evidence.
- `abl159_coco_fsdetr_frequency_spatial`: FSDETR frequency-spatial enhancement.
- `abl160_coco_df2net_dynamic_fusion`: DF2Net differential dynamic fusion.
- `abl161_coco_sdconet_saliency`: SDCoNet saliency collaboration.
- `abl162_coco_idgbr_boundary`: IDGBR semantic boundary refinement.

All four use only `/home/wuxi/coco` through `configs/deim_dfine/deim_hgnetv2_s_coco.yml`.

## 6. Commands

Foreground 1-epoch smoke on GPU0:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 EPOCHS=1 MAX_RUNS=1 CLEAN_OUTPUT=1 BASE_OUTPUT_DIR=outputs/ablation_20260524_smoke \
  bash tools/run_ablation_20260524_single_modules_coco.sh
```

Full serial single-module ablation on GPU0:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 SKIP_EXISTING=1 bash tools/run_ablation_20260524_single_modules_coco.sh
```

Resume from a specific module:

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=0 START_FROM=abl156_coco_hedp_prompt SKIP_EXISTING=1 \
  bash tools/run_ablation_20260524_single_modules_coco.sh
```

## 7. Verification Completed

- `python -m py_compile engine/deim/hybrid_encoder.py engine/optim/optim.py tools/summarize_ablation_all_new_modules.py`
- `bash -n tools/run_ablation_20260524_single_modules_coco.sh`
- dry-run for all nine ablation rows
- YAML model instantiation for Rot-E, HEDP, SAMPG/SST, SEGD, FSDETR, DF2Net, SDCoNet, and IDGBR modules
- `DualOptAdamW` construction and one optimizer step
- direct forward smoke on all eight new encoder refiners
- total summary table refreshed
