# Official-source baseline comparison

This is a new comparison, not a replacement of existing manuscript data.
Original datasets, manuscript documents, prior checkpoints and cloned upstream
source files are not edited by these scripts.

## Prespecified experiments

- Official DEIM-D-FINE-S, HGNetv2-B0: Phone, DSLR and DENTEX, seeds 0, 1, 2.
- Official RT-DETRv2-S, ResNet18-vd: the same three cohorts and seeds.
- 160 epochs; training batch 8; validation batch 16; AMP; validation at 640x640.
- Select the maximum validation COCO AP; use the earliest epoch on exact ties.
- Preserve official ImageNet backbone initialization, not full-detector COCO fine-tuning.
- Retain the local Plain DEIM controls and the larger RT-DETR-l comparison separately.

RT-DETRv2-S is the official small v2 model, not an invented approximately
11-million-parameter “RT-DETR-S”. The original v1 small model is RT-DETR-R18.
The selected v2 model is smaller than RT-DETR-l but not equal-sized to DEIM-S.

## Scope of matching

Dataset membership, zero-based category IDs, original annotations, number of
epochs, train/validation batches, base/evaluation resolution, COCO evaluator
environment and AP-based checkpoint selection are fixed. No new NMS, confidence
cutoff, or dental prior is added to either baseline.

DEIM's architecture, criterion, matching and native epoch loop come from the
pristine official clone. Its dataset-adapted optimizer, schedule, no-flip/no-mosaic
transforms, multi-scale collator and MixUp schedule match the existing local
Plain DEIM controls. This is an upstream implementation with a matched dental
recipe, not an unchanged COCO configuration.

RT-DETRv2-S retains official model-specific losses, optimizer/warmup and fixed
640/no-MixUp collator. Its horizontal flip is disabled, and its augmentation-stop
epoch is aligned to 130. These native recipe differences must be disclosed;
this external model is not an all-hyperparameter-matched DEIM ablation.

DEIM training uses native multi-scale sizes around base 640 until epoch 130.
Do not describe it as fixed-640 training. Both models evaluate at fixed 640.
The existing development splits are retained; repeated seeds do not create an
independent or patient-disjoint test set.

RT-DETRv2 additionally uses the process-local Pillow large-image setting already
used by official DEIM (`Image.MAX_IMAGE_PIXELS=None`). Its original augmentation
can create crops above the default pixel guard for these trusted photographs.
This compatibility adapter changes no pixels or training operations. The first
infrastructure-failed Phone seed-0 attempt is recorded under `failed_attempts`,
excluded from results, and restarted from initialization with the same seed.

## Storage policy

Each run retains one `best.pth` containing the AP-selected model, EMA and native
optimizer state. There are no `last.pth`, periodic model weights or stage-local
weight files on disk. DEIM's required stage-1 rollback state is serialized in
RAM, preserving its native training logic. A temporary file is used only for
atomic replacement of the best checkpoint.

This policy sacrifices recovery of the very latest epoch after interruption.
The runner refuses to overwrite an incomplete run; inspect its best checkpoint
and log before deciding how to resume. Old user experiments are not pruned.
The queue stops if available disk space drops below 3 GiB.

Every completed run also retains config provenance, stdout, native epoch log,
best/final metrics, checkpoint SHA256, best-checkpoint re-evaluation, compressed
COCO predictions and compressed evaluator arrays. These small evidence files
are not model checkpoints and are required for later verification and figures.

## Execution and follow-up

`protocol.json` and `dataset_evidence.json` freeze code revisions, data paths,
annotation hashes and settings. `queue_gpu0.json` and `queue_gpu1.json` expose
queue progress. Each run has a `status.json` and gets `COMPLETED` only after all
160 epochs, AP/checkpoint agreement and re-evaluation checks pass.

The `smoke_runs` directory contains explicitly separate two-epoch pipeline tests;
these are never research results. Full runs are under `runs`.

After training, review paired seed comparisons, model-size/compute comparisons
and same-image qualitative outputs. Do not replace manuscript values or select
new hyperparameters based on interim validation rankings.
