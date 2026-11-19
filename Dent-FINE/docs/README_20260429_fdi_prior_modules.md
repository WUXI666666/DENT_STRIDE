# 2026-04-29 FDI 先验与小目标检测模块说明

## 本轮判断

上一轮消融已完成，当前最佳主线是：

- `abl34_cmfa_hooa`: AP `0.5426`，比 baseline `+0.0083`
- `abl35_ucm_hooa`: AP75 `0.6090`，定位精度有潜力但整体 AP 未超过 `abl34`

因此本轮新模块不再堆大 backbone/encoder，而是围绕牙位检测最强的结构信息做轻量先验：

- 遥感/自然图像小目标检测：加入 NWD box loss，对密集小框的中心/尺度误差更平滑。
- CRISP-SAM2 / SegVol / AFLoc：不直接搬大模型，抽取 prompt、语义对齐、空间定位思想，落到 FDI class prior 和 query logit prior。
- 牙科先验：同颌邻牙连续、同编号唯一、上下颌不乱跳、混合牙列的恒/乳牙列一致性。

## 新增模块

### 1. FDI-SpatialPrior

文件：`engine/deim/dfine_decoder.py`

作用位置：

- `FDISpatialLogitPrior`: decoder 分类 logits 后、LQE 前。
- 只作用 main queries，DN queries 不参与。

编码的先验：

- jaw prior：根据 query 中心 y 坐标给上/下颌类别轻量 bias。
- neighbor prior：对同颌空间邻近 query 聚合类别分布，再通过 FDI 邻接矩阵鼓励相邻牙位。
- unique prior：同一口腔同一 FDI 编号最多一个，按 query 维度做轻量重复惩罚。
- dentition prior：52 类混合牙列时，邻域恒/乳倾向会影响对应类别；32 类 DENTEX 恒牙时自动弱化。

默认配置在 `configs/base/dfine_hgnetv2.yml`：

```yaml
DFINETransformer:
  use_fdi_spatial_prior: False
  fdi_prior_knn: 8
  fdi_prior_sigma: 0.12
  fdi_prior_jaw_scale: 0.08
  fdi_prior_neighbor_scale: 0.12
  fdi_prior_unique_scale: 0.04
  fdi_prior_dent_scale: 0.04
```

### 2. NWD-Loss

文件：`engine/deim/deim_criterion.py`

作用位置：

- `DEIMCriterion.loss_nwd`
- 加到 `DEIMCriterion.losses=[mal,boxes,local,nwd]`

目的：

- 对牙齿这类密集、小尺度、框相近目标，NWD 比纯 IoU 对小偏移更稳定。

默认配置：

```yaml
DEIMCriterion:
  nwd_normalizer: 0.08
```

### 3. FDI-AnatomyLoss

文件：`engine/deim/deim_criterion.py`

作用位置：

- final layer only，不进入 DN/aux 循环。

包含：

- `loss_fdi_factor`: 直接从主分类 logits 聚合象限、牙位序号概率，不再新增因子头。
- `loss_fdi_unique`: 每张图内每个类别最多出现一次，同时鼓励 GT 类别至少有一个 query 响应。
- `loss_fdi_dent`: 52 类混合牙列时约束恒牙/乳牙列；32 类全恒牙时自动为 0。

默认配置：

```yaml
DEIMCriterion:
  use_fdi_anatomy_loss: False
```

## 修改文件

1. `engine/deim/dfine_decoder.py`
   - 新增 `FDISpatialLogitPrior`
   - `TransformerDecoder.__init__` 新增 FDI spatial prior 参数
   - decoder forward 在 `score_head` 后应用 FDI spatial prior
   - `DFINETransformer.__init__` 透传配置到 decoder

2. `engine/deim/deim_criterion.py`
   - 新增 `loss_nwd`
   - 新增 final-only `loss_fdi_anatomy_final`
   - 新增 `use_fdi_anatomy_loss`、`nwd_normalizer`
   - 新增权重键：`loss_nwd`、`loss_fdi_factor`、`loss_fdi_unique`、`loss_fdi_dent`

3. `configs/base/dfine_hgnetv2.yml`
   - 新增 FDI spatial prior 默认配置
   - 新增 NWD/anatomy loss 默认配置

4. `tools/summarize_ablation_all_new_modules.py`
   - 总表加入 `abl39` 到 `abl46`

5. `tools/run_ablation_20260429_fdi_prior_modules.sh`
   - 新增串行消融脚本

## 备份

```text
backups/20260429_fdi_prior_modules/
```

包含：

- `dfine_decoder.py.bak`
- `deim_criterion.py.bak`
- `dfine_hgnetv2.yml.bak`
- `summarize_ablation_all_new_modules.py.bak`

## 已完成验证

编译：

```bash
/home/wuxi/miniconda3/envs/deimv2/bin/python -m py_compile engine/deim/dfine_decoder.py engine/deim/deim_criterion.py
```

最小 forward/loss smoke：

- `num_classes=32`
- `DFINETransformer.use_fdi_spatial_prior=True`
- `DEIMCriterion.losses=[mal,boxes,local,nwd]`
- `DEIMCriterion.use_fdi_anatomy_loss=True`
- 已成功产生 `loss_nwd`、`loss_fdi_factor`、`loss_fdi_unique`

## 新增消融分支

| 实验 | 模块 |
|---|---|
| `abl39_nwd_loss` | 单独 NWD loss |
| `abl40_fdi_spatial_prior` | 单独 FDI spatial logit prior |
| `abl41_fdi_anatomy_loss` | 单独 FDI anatomy loss |
| `abl42_cmfa_hooa_nwd` | 当前最佳 `CMFA+HOOA` + NWD |
| `abl43_cmfa_hooa_fdi_prior` | 当前最佳 `CMFA+HOOA` + FDI spatial prior |
| `abl44_cmfa_hooa_anatomy_loss` | 当前最佳 `CMFA+HOOA` + FDI anatomy loss |
| `abl45_cmfa_hooa_nwd_fdi_prior` | 当前最佳 `CMFA+HOOA` + NWD + FDI spatial prior |
| `abl46_cmfa_hooa_all_priors` | 当前最佳 `CMFA+HOOA` + NWD + FDI spatial prior + FDI anatomy loss |

## 运行命令

完整串行消融：

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=1 BASE_OUTPUT_DIR=outputs/ablation_all_new_modules START_FROM=abl39_nwd_loss bash tools/run_ablation_20260429_fdi_prior_modules.sh
```

断点续跑示例：

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=1 BASE_OUTPUT_DIR=outputs/ablation_all_new_modules START_FROM=abl43_cmfa_hooa_fdi_prior bash tools/run_ablation_20260429_fdi_prior_modules.sh
```

只刷新总表：

```bash
cd /home/wuxi/Dent-FINE
/home/wuxi/miniconda3/envs/deimv2/bin/python tools/summarize_ablation_all_new_modules.py --base-output-dir /home/wuxi/Dent-FINE/outputs/ablation_all_new_modules
```
