# DENT-STRIDE

Code, experiment outputs, publication figures and inference checkpoints associated
with `DENT_STRIDE_CMIG_Visual_20261117.pdf`.

## Contents

- [`Dent-FINE/`](Dent-FINE/): a publication copy of the existing project working
  tree, including model code, configurations, training/evaluation utilities,
  text logs, metrics and archived result tables. Existing model code was not
  edited, and no experiment was retrained for this upload.
- [`Dent-FINE-paper/CMIG_Overleaf_Visual_20261117/`](Dent-FINE-paper/CMIG_Overleaf_Visual_20261117/):
  the exact final LaTeX source, bibliography, template dependencies and all 20
  figure files used by the specified PDF (18 numbered figures; Figure 11 has
  three files).
- [`Dent-FINE-paper/DENT_STRIDE_CMIG_Visual_20261117.pdf`](Dent-FINE-paper/DENT_STRIDE_CMIG_Visual_20261117.pdf):
  the specified manuscript, unchanged.
- [`Dent-FINE-paper/materials/`](Dent-FINE-paper/materials/): recorded experiment
  configurations, software/source snapshots, per-run metrics, predictions,
  pre-top-k candidates, numerical attribution arrays, geometric statistics and
  plotting/evaluation scripts for the released analyses.
- [`data/annotations/`](data/annotations/): COCO annotation and class-map files,
  without image pixels. DSLR annotation files are also retained in
  `Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/`.
- [`checkpoints/`](checkpoints/): selected publication, comparator and component
  inference checkpoints. These are large-file-storage assets, not GitHub Releases.
- [`release_manifest/`](release_manifest/): file hashes, figure mappings and
  checkpoint provenance.

## Download large files

Install Git LFS before cloning, otherwise the large files may appear as small
pointer text files:

```bash
git lfs install
git clone https://github.com/WUXI666666/DENT_STRIDE.git
cd DENT_STRIDE
git lfs pull
```

## Checkpoint format

For native PyTorch `.pth` checkpoints, the released tensors are exactly the EMA
state used by evaluation when available, or the recorded model state otherwise.
Both `model` and `ema.module` refer to that evaluated state. Dtypes and parameter
values are unchanged; optimizer, scaler and scheduler recovery states are omitted.
These checkpoints are for inference/evaluation, not exact training continuation.
Native Ultralytics `.pt` best checkpoints are copied unchanged.
`release_manifest/checkpoints.json` records the source checkpoint hash, selected
state, exported file hash and state-value verification. The original full training
checkpoints on the source server were not changed or deleted.

The released code and config paths retain the original project layout and may
contain original absolute paths. Set dataset and checkpoint locations for your
own machine. Use the matching frozen source snapshot and configuration for each
run; the presence of multiple historical results does not make them interchangeable.
Existing reported manuscript values and archived measurements are preserved,
not recomputed or silently reconciled by the upload process.

## Dataset exclusion and privacy

**No original training-image dataset is included.** Original dataset image
directories, image symlinks, training/mosaic preview photographs and TensorBoard
event files are excluded. Original validation image directories are not included
either. The selected publication figures necessarily contain the illustrative
image regions already shown in the manuscript; numerical heatmaps, predictions,
annotations and geometric templates are included where available.
Credentials, private SSH keys, local Git history, unrelated draft manuscripts,
large prepacked historical bundles and compiled caches are not published.
No chat history or personal Codex configuration is part of this repository.

Dataset access and redistribution conditions continue to apply. Code components
retain the license notices included in their respective source directories;
publishing this collection does not relicense third-party code, datasets or model
weights under one new blanket license.
