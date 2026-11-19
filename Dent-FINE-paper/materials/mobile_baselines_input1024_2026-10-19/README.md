# Mobile input-1024 framework comparison

Eight new **seed-0**, **160-epoch**, **1024×1024** runs on the existing Mobile
878-image training / 220-image validation split. Original images, COCO
annotations, old checkpoints, source checkouts, and manuscripts are not changed.
The YOLO data view contains image symlinks and mechanically derived label text,
not copied image pixels. `protocol.json` preserves annotation hashes and maps.

Models: official DEIM-s, RT-DETR-L, official RT-DETRv2-S, Faster R-CNN R50-FPN,
CAF-YOLO-s, YOLOv8-s, YOLO11-s, Dome-DETR-S. All use seed 0 (not a repeated-seed
mean). No result-dependent seed or configuration selection is performed here.

## Comparison scope

- Fixed-square 1024 input and evaluation, existing FDI52 map, 160 epochs,
  no geometric flips, no Mosaic, no MixUp.
- DEIM/RTv2/DOME use the same available photometric/crop/zoom recipe; the native
  Compose API differs (DEIM accepts a phase list; RTv2/DOME accept stop epoch 130).
- YOLO-family models retain native color/affine operations and letterboxing;
  the validation data loader is forced to a square canvas. Faster R-CNN uses
  exact-square Resize in both train/test mappers, batch 4 and its native SGD.
  DOME uses batch 4 after batch 8 exhausted GPU0 memory in a forward-only
  compatibility preflight; its learning-rate settings are unchanged. Other
  models use batch 8. Batch/optimizer-step counts are not fully matched. CAF preserves the source author's FP32 setting;
  other transformer/YOLO training uses AMP. Pretraining, native losses,
  optimizers, and postprocessing remain framework-specific.
- DEIM/RTv2/DOME start from ImageNet backbone pretraining, the other systems
  from their established native COCO detector initialization. This is NOT a
  fully hyperparameter-matched architecture or isolated-module ablation.
- The old Dome phone `tuned301` comparison was a 40-epoch tuning run initialized
  from a DSLR dental checkpoint. It is not reused. The new DOME run has 160
  epochs, ImageNet HGNetv2-B0 initialization and its 300..1500 native query cap.
  DOME's MWAS window is set to 16 at 1024, maintaining the original 8×8 window
  grid of window=10 at 640; native code requires exact divisibility. This
  necessary resolution adaptation is declared, not hidden.
- All estimates remain validation estimates, not independent-test results.

## Selection, metrics, and storage

Best means maximum canonical COCO bbox AP at validation; earliest exact tie.
Official native solvers keep their phase rollback snapshots serialized in RAM
and save only one global `best.pth`. YOLO-family trainers save only FP32 EMA
`weights/best.pt`; `last.pt` and periodic files are never written. Faster R-CNN
retains only `model_best.pth`. An atomic temporary file can exist during best
replacement. Best-only storage cannot resume the latest unsaved epoch.

Ultralytics validation predictions are mapped back to the canonical COCO image
IDs before AP calculation; native fitness mixtures are not the selector.
Inference score floor=0.001 is retained for YOLO, CAF and Faster R-CNN. YOLO/CAF
use class-aware NMS IoU=0.70; Faster R-CNN retains its native ROI NMS IoU=0.50.
DETR models use top-300 candidates without
added NMS. AP selection uses FP32 validation; native losses/optimization remain
unchanged. Prediction exports preserve numeric precision.
Native DOME returns as many sorted query-class candidates as its dynamic query
count (and ignores `num_top_queries`). Its native score ranking is followed by
an explicit 300-output cap in this comparison; its internal dynamic query count
and training losses are not reduced. This adapter is outside the source tree.

Each completed model has `metrics_summary.json`, full log/config, selected epoch,
best checkpoint SHA256, compressed predictions and COCO arrays. Fixed endpoints
use score≥0.30 and Hungarian localization matching IoU≥0.50, with the existing
denominators for precision/recall/F1, FDI accuracy, coverage and duplicate rate.

`compute_profile.json` uses calflops 0.3.2, batch 1, FP32 native unfused forward
at 1024. Postprocessing outside the model and unsupported operations are not
counted. Dynamic DOME queries/Faster proposals make input values relevant; the
profiling/latency input is the first canonical validation photograph (sorted by
image ID), resized to 1024 and held on the device, not a cohort mean. Its path
and hash are saved. RGB/BGR ranges follow the framework input convention.
`inference_benchmark.json` uses GPU0 RTX4090, 30 warmups and 100 CUDA-event trials;
it reports mean/SD/P50/P95, images/s and peak inference allocation. The inherited
Detectron2 environment has a different PyTorch/CUDA version, recorded explicitly.

## Queue

GPU1 belongs to another active user job and is left untouched. The detached
GPU0 queue runs each model once and executes post-training compute/metric
finalization automatically. `queue_gpu0.json` exposes active/completed/failed
jobs. Individual failures are retained for inspection; other independent models
can continue. No automatic restarts or overwriting of incomplete runs occur.
Disk free space is checked between jobs/epochs; stop below 3 GiB without deleting
old data. Small logs and evidence are retained; only redundant model weights
are suppressed. `results_summary.csv`/`.json` are updated as models complete.

Preflight files are input/forward/backward compatibility checks, never research
results. The queue is launched only after all eight preflights succeed.

## Corrected reruns (20 October 2026)

Two original attempts are excluded via `repaired_runs.json`: RT-DETR-L had an
incorrect original-image coordinate export, invalidating AP-based checkpoint
selection; RT-DETRv2-S failed before training because its solver has no `stats`
symbol. The RT-DETR adapter now follows native width/height scaling, while the
optional profiling hook is installed only when that symbol exists.

Both models are retrained from their original pretrained initialization with
the same seed-0 / 160-epoch / fixed-1024 protocol. New outputs use
`runs/<model>/seed_0_repaired_20261020`. The original failed attempts are retained
as excluded records; their AP values are not included in the active summary.
The other six completed runs and their checkpoints are unchanged.

Regression tests include exact synthetic COCO geometry, native exporter
equivalence, full-cohort coordinate replay of the excluded RT-DETR checkpoint
(diagnostic only), and two-batch native training/validation/checkpoint tests.
`repairs_20261020` holds these records, the detached GPU0 queue, and its frozen
launch manifest. The main `results_summary.csv`/`.json` automatically use only
the repaired paths for the two affected models.
