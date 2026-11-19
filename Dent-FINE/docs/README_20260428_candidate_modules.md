# 2026-04-28 牙位检测候选模块接入说明

## 外部模块判断

本轮只把适合目标检测、且能以轻量残差方式接入 Dent-FINE 的思路落到代码里；不直接拷贝外部项目的大网络。

- SPEGC: 官方仓库 `https://github.com/Jwei-Z/SPEGC-for-MIS` 当前 README 说明代码仍在整理、尚未释放完整实现。因此没有直接移植 CTTA 训练流程，只抽取“语义提示 + 图聚类”的思想，做成 decoder main query 的 `SemanticPromptGraphRefiner`。
- High-Order Omni-Attention: 原任务是遥感图像融合，和牙位检测相差较大，但“空间、通道、高阶交互共同建模”适合牙列边界和邻牙关系，因此接入为 encoder 输出特征的 `HighOrderOmniAttention`。
- PFESA: 仓库中已有 `ParameterFreeEdgeStructureAttention`，表中已有 `abl18/22/23/25/26/27`，本轮不重复新增。
- SMFNet: 论文/代码强调语义引导、图推理和多分支融合；Dent-FINE 已有 CMFA/MPF/QGP，本轮把最贴近检测的语义图部分放到 SPEGC-style query refiner 中。
- UCAN: 仓库中已有 `LargeKernelDistillationBlock` 和 `use_ucan_lkd`，本轮不重复新增。
- MultiNeX: 现有 `RetinexPriorGate` 单先验收益偏弱，本轮新增更保守的 `MultiPriorRetinexRefiner`，融合多尺度 illumination、reflectance、contrast、gradient prior。
- UCMNet: 仓库中已有 `UncertaintyContextRefiner` 和 `use_ucm_uncertainty`，表中显示 UCM 对 AP75 有潜力，本轮组合实验继续保留它。

## 修改文件与行号

1. `engine/deim/hybrid_encoder.py`
   - `1226-1278`: 新增 `HighOrderOmniAttention`，融合 local/context/edge 分支，并用 channel/spatial gate 输出轻量残差。
   - `1280-1322`: 新增 `MultiPriorRetinexRefiner`，使用 3/7/11 多尺度 illumination、reflectance、contrast、gradient prior。
   - `1390-1419`: `HybridEncoder.__init__` 新增 `use_hooa_attention`、`use_multinex_prior` 开关。
   - `1463-1468`: 根据开关构建 `self.hooa_attention` 和 `self.multinex_prior`。
   - `1677-1693`: 在 `_apply_refine_stack` 中串接 MultiNeX 和 HOOA，默认关闭，不影响 baseline。

2. `engine/deim/dfine_decoder.py`
   - `233-313`: 新增 `SemanticPromptGraphRefiner`，用 learnable semantic prompts 生成软聚类，再按 cluster similarity/top-k graph 聚合 query 上下文。
   - `832-883`: `TransformerDecoderLayer` 新增 SPEGC prompt graph 配置，并按开关实例化模块。
   - `932-941`: 只对 main queries 应用 semantic prompt graph，DN queries 不参与，避免污染 denoising 监督。
   - `1231`: decoder layer 调用改为传入真实 `main_slice`，让 SPEGC refiner 能拿到 main query 范围。
   - `1505-1588`: `DFINETransformer` 新增并透传 `use_spegc_prompt_graph`、`spegc_num_prompts`、`spegc_graph_topk`、`spegc_ref_bias`、`spegc_res_scale`。

3. `configs/base/dfine_hgnetv2.yml`
   - `62-63`: 新增 `HybridEncoder.use_hooa_attention`、`HybridEncoder.use_multinex_prior`，默认 `False`。
   - `107-111`: 新增 SPEGC prompt graph 配置，默认 `False`。

4. `tools/summarize_ablation_all_new_modules.py`
   - `40-47`: 总表加入 `abl31` 到 `abl38` 的实验定义。

5. `tools/run_ablation_20260428_candidate_modules.sh`
   - `1-154`: 新增串行消融脚本。默认写入 `outputs/ablation_all_new_modules`，每个分支完成后自动刷新 `ablation_summary.json/csv/md`。

6. `outputs/ablation_all_new_modules/ablation_summary.md`
   - `34-41`: 已写入新增 `abl31` 到 `abl38`，当前状态为 `pending`。

## 新增消融分支

| 实验 | 开关 | 目的 |
|---|---|---|
| `abl31_hooa_attention` | `HybridEncoder.use_hooa_attention=True` | 单独验证高阶全注意力是否改善牙列边缘和邻牙上下文 |
| `abl32_multinex_prior` | `HybridEncoder.use_multinex_prior=True` | 单独验证多先验 Retinex 对全景片亮度/对比差异是否有效 |
| `abl33_spegc_prompt_graph` | `DFINETransformer.use_spegc_prompt_graph=True` | 单独验证语义提示图聚类 query refiner |
| `abl34_cmfa_hooa` | CMFA + HOOA | 在已有 AP50 较好的 CMFA 上叠加高阶注意力 |
| `abl35_ucm_hooa` | UCM + HOOA | 在已有 AP75 较好的 UCM 上叠加高阶注意力 |
| `abl36_cmfa_ucm_spegc` | CMFA + UCM + SPEGC | 验证最佳现有组合和 query 结构先验的互补性 |
| `abl37_hooa_multinex` | HOOA + MultiNeX | 验证两种 encoder refine 是否互补 |
| `abl38_cmfa_ucm_hooa_spegc` | CMFA + UCM + HOOA + SPEGC | 候选强组合，风险是参数/FLOPs 增大 |

## 已完成验证

1. 编译检查：

```bash
/home/wuxi/miniconda3/envs/deimv2/bin/python -m py_compile engine/deim/hybrid_encoder.py engine/deim/dfine_decoder.py tools/summarize_ablation_all_new_modules.py
```

2. 随机张量 forward 检查：
   - encoder 输出通过 HOOA/MultiNeX/CMFA。
   - decoder 训练态带 DN queries 通过 SPEGC prompt graph。

3. GPU1 上 1 epoch smoke train：
   - 输出目录：`outputs/smoke_20260428_new_modules`
   - 组合：CMFA + UCM + HOOA + MultiNeX + SPEGC
   - 训练/验证均完成，`log.txt` 写入 epoch 0 结果。
   - smoke AP: `0.0276`，只代表 1 epoch 连通性，不代表完整性能。

## 备份

修改前备份在：

```text
backups/20260428_spegc_hooa_multinex/
```

包含：

- `hybrid_encoder.py.bak`
- `dfine_decoder.py.bak`
- `dfine_hgnetv2.yml.bak`
- `summarize_ablation_all_new_modules.py.bak`

## 运行命令

1 epoch smoke train 命令：

```bash
cd /home/wuxi/Dent-FINE
CUDA_VISIBLE_DEVICES=1 /home/wuxi/miniconda3/envs/deimv2/bin/python train.py -c configs/deim_dfine/deim_hgnetv2_s_coco.yml --use-amp --seed=0 -u epoches=1 checkpoint_freq=1 flat_epoch=1 no_aug_epoch=0 HybridEncoder.use_cmfa_fusion=True HybridEncoder.use_ucm_uncertainty=True HybridEncoder.use_hooa_attention=True HybridEncoder.use_multinex_prior=True HybridEncoder.apply_denoise_modules=proj HybridEncoder.apply_refine_modules=outs DFINETransformer.use_spegc_prompt_graph=True DFINETransformer.spegc_num_prompts=16 DFINETransformer.spegc_graph_topk=8 train_dataloader.total_batch_size=8 val_dataloader.total_batch_size=16 --output-dir outputs/smoke_20260428_new_modules
```

完整串行消融命令：

```bash
cd /home/wuxi/Dent-FINE
GPU_ID=1 BASE_OUTPUT_DIR=outputs/ablation_all_new_modules START_FROM=abl31_hooa_attention bash tools/run_ablation_20260428_candidate_modules.sh
```

只刷新总表：

```bash
cd /home/wuxi/Dent-FINE
/home/wuxi/miniconda3/envs/deimv2/bin/python tools/summarize_ablation_all_new_modules.py --base-output-dir /home/wuxi/Dent-FINE/outputs/ablation_all_new_modules
```
