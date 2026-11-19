# Dent-FINE — Tooth Position Detection Framework

This repository is an engineering adaptation of `DEIM` for tooth position (tooth location/numbering) detection tasks. It includes training, inference, evaluation, export, and visualization tools.

## Quick Start

- Clone / change directory:

```bash
cd Dent-FINE
```

- Recommended: create a Conda environment and install dependencies:

```bash
conda create -n deim_dental python=3.11 -y
conda activate deim_dental
pip install -r requirements.txt
```

## Project Overview
- `train.py` — training entrypoint.
- `inference.py` — inference entrypoint.
- `analyze.py` — result analysis script.
- `configs/` — model and dataset configurations (global runtime settings in `configs/runtime.yml`).
- `deim/` — module implementation, including tooth-specific losses and post-processing logic.
- `engine/` — training/evaluation engine and data handling.
- `tools/` — export and deployment scripts (ONNX/YOLO/NMS/RT, etc.).

## Dataset

Recommended layout (COCO-style):

```
dataset/
	images/train/...
	images/val/...
	annotations/instances_train.json
	annotations/instances_val.json
```

Annotation suggestions:
- Annotate each tooth with a bounding box. You may use a single class (`tooth`) or multiple classes representing tooth positions (e.g., FDI codes).

## Configuration and Training Examples
1. Choose an appropriate config (examples):

```bash
configs/deim_dfine/deim_hgnetv2_s_coco.yml
configs/deim_dfine/deim_hgnetv2_m_coco.yml
```

2. Training command examples:

Single-GPU training:

```bash
python train.py --config configs/deim_dfine/deim_hgnetv2_s_coco.yml
```

Distributed / explicit torchrun example:

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/deim_hgnetv2_s_coco.yml --use-amp --seed=0
```

Common arguments:
- `--config`: path to the config file.
- `--resume` / `--pretrained`: specify checkpoint to resume training or fine-tune.

## Inference and Evaluation Examples

Single-image inference example:

```bash
python inference.py --config configs/deim_dfine/deim_hgnetv2_s_coco.yml \
	--checkpoint runs/exp/checkpoint_best.pth \
	--input dataset/images/test.jpg \
	--output results/vis.jpg
```

If COCO evaluation tools are available in `engine` or `tools`, use them to compute mAP and other metrics.

## Visualization and Analysis
- Use `tools/inference/torch_inf_vis.py`, `vis_det/`, or other visualization scripts in the repository to overlay and save predictions.
- Use `analyze.py` to compute statistics and analyze detection results.

## Debugging Tips
- OOM / insufficient memory: reduce `batch_size` or use a smaller model (e.g., `hgnetv2_s`).
- Unstable training: check data and label quality, learning rate, and augmentations.
- Class imbalance: consider resampling or augmentation strategies.

## License & Citation
- Please cite the original DEIM repository and the key papers/methods used in this project.
