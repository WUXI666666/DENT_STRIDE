# Dent-FINE 新增模块检查与修改说明

日期：2026-04-25

## 备份位置

本次修改前已备份下列文件到 `backups/20260425_dentfine_ablation_fix/`：

- `hybrid_encoder.py.bak`
- `dfine_decoder.py.bak`
- `deim_criterion.py.bak`
- `dfine_hgnetv2.yml.bak`
- `run_ablation_next_feature_modules.sh.bak`
- `summarize_ablation_all_new_modules.py.bak`

## 主要发现

1. `GPU_ID=1 bash tools/run_ablation_next_feature_modules.sh` 默认从 `abl18_pfesa` 开始，只跑后半段新增模块，不会重新跑 baseline。若前面 baseline 是旧代码或旧配置得到的，后续 delta AP 会不完全公平。
2. 当前 `abl00_baseline` 不是纯原始 D-FINE：DySample、Learnable PE、ROI query 初始化融合、arch 顺序损失等默认已经在基线上生效。新增模块是在强 baseline 上叠加，单模块收益自然会变小。
3. `CMFA` 关闭时仍实例化了 `AdaptiveInterleaveFusion`，导致 baseline 参数量包含未使用 CMFA 参数；且旧融合逻辑会把两路特征变成近似重复的两半通道，削弱跨层信息差异。
4. 牙位因子头 `quad/tooth/dent` 虽然被创建，但主分类回灌系数为 `0,0,0`，配置里也没有启用 `quad/tooth/dent` 损失，因此对牙位 AP 基本没有有效贡献。
5. 多数新增模块使用很小的残差初值，例如 `sigmoid(-2/-3)` 后只有约 1% 到 4% 的更新幅度；在 160 epoch 的强基线上可能表现为“稳定但提升小”。

## 新增模块接入位置和作用

| 模块 | 接入位置 | 当前作用 |
|---|---|---|
| `HFSA` | `HybridEncoder` 编码层后和 FPN/PAN 输出后 | 频域/条带注意力增强牙弓结构 |
| `WavEnhance` | `input_proj` 后，encoder 前 | 小波高频细节增强，但当前结果为负收益 |
| `StatNoise` | `input_proj` 后，encoder 前 | 统计引导去噪，参数多且当前收益偏负 |
| `MPF` | 默认 FPN/PAN 输出后 | 多路径空间、通道、上下文融合，小幅正收益 |
| `DynamicRF` | 默认 FPN/PAN 输出后 | 动态感受野，当前单独负收益 |
| `MPC` | `input_proj` 后，encoder 前 | 多尺度卷积感知，当前 AP 负但 AP75 有一定迹象 |
| `PQG-HF` | `input_proj` 后的指定 stage | 高频残差增强，当前负收益 |
| `CMFA` | FPN top-down 和 PAN bottom-up 融合点 | 跨层融合，当前单项最有效，AP50 提升明显 |
| `PFESA` | 默认 FPN/PAN 输出后 | 无参频域边缘/结构注意力，单项基本持平 |
| `UCAN-LKD` | 默认 FPN/PAN 输出后 | 大核蒸馏式上下文，单项基本持平 |
| `UCM-Uncertainty` | 默认 FPN/PAN 输出后 | 不确定性上下文修正，AP75 提升最明显 |
| `RetinexPrior` | 默认 FPN/PAN 输出后 | 反射/照明先验，当前 AP 负收益 |
| `PGR` | decoder top-k query 初始化后 | 全局 ROI token 引导 query，残差太小，收益不明显 |
| `FDI-FactorPrior` | decoder 分类头旁路，本次修复后可选 | 用象限、牙位序号、恒/乳牙列结构约束主分类 logits |

## 本次修改

1. `engine/deim/hybrid_encoder.py`
   - 修复 `CMFA`：只有 `use_cmfa_fusion=True` 时才实例化融合模块，baseline 不再携带未使用 CMFA 参数。
   - 修改 `AdaptiveInterleaveFusion`：保留每一路原始特征，同时注入对侧 gated 特征，避免旧实现中两半通道近似重复。

2. `engine/deim/dfine_decoder.py`
   - 新增显式配置 `roi_init_ratio`，当前默认保持 `0.5`，后续可用 `DFINETransformer.roi_init_ratio=0.0` 做纯 top-k query 对照。
   - 新增 `use_fdi_factor_prior` 和 `fdi_factor_logit_scale`。默认关闭；开启后才创建 `quad/tooth/dent` 因子头，并把标准化后的 FDI 结构 log-prob 回灌到主分类 logits。
   - 修复因子头默认“建了但无效”的问题，避免无意义参数和计算。
   - 2026-04-27 补充修复：FDI factor prior 在训练态带 DN queries 时，将 `gather` 索引扩展到 `[B, Q, C]`，避免 main query 段切片维度为 0 导致训练中断。

3. `engine/deim/deim_criterion.py`
   - 将默认新增 loss 权重改成 `setdefault`，允许命令行覆盖 `loss_quad/loss_tooth/loss_dent`。
   - 修复因子标签范围，保证象限和牙位索引在 `[0, 7]`。
   - 因子头关闭但误配损失时返回安全零损失，不再触发 `None.sum()`。

4. `configs/base/dfine_hgnetv2.yml`
   - 增加 `roi_init_ratio`、`use_fdi_factor_prior`、`fdi_factor_logit_scale` 默认配置。

5. `tools/run_ablation_next_feature_modules.sh`
   - 增加后续候选：
     - `abl28_fdi_factor_prior`
     - `abl29_cmfa_ucm_uncertainty`
     - `abl30_cmfa_ucm_fdi_factor`
   - 候选中启用 `DEIMCriterion.losses=[mal,boxes,local,quad,tooth,dent]`，让牙位因子头真正受监督。

6. `tools/summarize_ablation_all_new_modules.py`
   - 汇总表加入 `abl28` 到 `abl30`。

## 建议的下一轮实验

为了避免旧 baseline 干扰，建议新建输出目录先重跑 baseline 和已有单模块对照：

```bash
GPU_ID=1 BASE_OUTPUT_DIR=outputs/ablation_after_fix START_FROM=abl00_baseline bash tools/run_ablation_all_new_modules.sh
```

然后在同一输出目录里跑本次新增优化候选：

```bash
GPU_ID=1 BASE_OUTPUT_DIR=outputs/ablation_after_fix START_FROM=abl28_fdi_factor_prior bash tools/run_ablation_next_feature_modules.sh
```

优先关注：

- `abl28_fdi_factor_prior`：验证牙位结构先验是否改善牙位分类 AP。
- `abl29_cmfa_ucm_uncertainty`：结合当前最强的 CMFA 和 AP75 表现较好的 UCM。
- `abl30_cmfa_ucm_fdi_factor`：检测结构先验与特征融合是否互补。

## 后续 AP 提升方向

1. 做一个真正干净的 baseline：关闭 ROI 初始化、DySample、Learnable PE 后与当前 baseline 对照，分清“基础改动”和“新增模块”的贡献。
2. 针对牙位类别引入结构监督：保留本次 FDI factor head，并尝试 `fdi_factor_logit_scale=0.10/0.15/0.20`。
3. 针对 AP75：优先测试 `CMFA+UCM`，再考虑轻量 bbox refinement 或提高后期无增强阶段 epoch。
4. 针对 AP50：CMFA 已经明显提高 AP50，可继续优化融合位置，只在 top-down 或 bottom-up 单独启用做拆分实验。
5. 谨慎使用左右翻转：FDI 牙位标签有左右含义，若翻转不 remap label，会直接伤害牙位分类。
