# 官方 DEIM-S 主配置：Mosaic-640 新配方

本目录是独立新实验版本。原 `/home/wuxi/Dent-FINE`、官方 clone、旧实验、原始数据和论文均不修改。

## 默认训练范围

GPU0 串行执行，全部 seed 0、160 epochs，从同一官方 ImageNet HGNetv2-B0 骨干初始化，不从已训练检测器继续微调。

| 顺序 | 任务 | 启用模块 | 配置 |
|---|---|---|---|
| 1 | Phone | DualOpt | `configs/phone_dualopt.yml` |
| 2 | DENTEX2023 | Factor heads + Prototype + SpatialShape | `configs/dentex_spatial_shape.yml` |
| 3 | DSLR | 四处 CMFA + DualOpt | `configs/dslr_cmfa_dualopt.yml` |

三个同配方 Plain 对照配置也已生成，但默认不额外训练。若用户确认六项，可显式为队列指定各队列的 Plain→selected 任务顺序；不能把“配置已生成”写成“实验已完成”。

本轮没有引入 QueryFusion、DySample、可学习行列位置编码、HOOA、CEDv2、SMWG、MPF、UCM 或其他未选模块。模块代码与上一轮最小迁移版本逐字节一致；本轮不通过临时堆叠模块、择优种子或改变验证数据获取结果。

## 明确的配置调整

用户指定的两项算子配置均原样落实：

```yaml
- {type: RandomIoUCrop, p: 0.5}
- {type: Mosaic, output_size: 640, rotation_range: 10,
   translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5],
   probability: 1.0, fill_value: 0, use_cache: False,
   max_cached_images: 50, random_pop: True}
```

另同步调整或核查：

- `Compose.mosaic_prob: 1.0`：旧值为 0，只修改算子 probability 并不能开启 Mosaic。
- 训练/验证 Resize、`eval_spatial_size` 和 collator base_size 均为 640。
- `base_size_repeat: null`：关闭原 480–800 多尺度，本轮实际网络输入固定 640×640。
- Mosaic 的 output_size 是每张源图的短边尺寸，不是最终网络输入；拼接后的较大画布仍需最后的 Resize。
- 保留原 `[4,70,130]` 增强边界；`no_aug_epoch: 30` 与最后 30 轮无强增强阶段对齐。原值 25 对应 LR 到最小值的时间晚 5 轮。
- 保留 MixUp 概率 0.5、有效区间 `[4,80)`；不新增水平/垂直翻转，避免改变 FDI 左右/上下颌含义。
- 有效调度是 flatcosine：`warmup_iter: 50`、`flat_epoch: 70`。配置中继承的 MultiStep/LinearWarmup 参数不是有效的 flatcosine 日程，不能据此解释本轮学习率。

零基 epoch 的真实增强策略：

| epoch | 实际行为 |
|---|---|
| 0–3 | 关闭 policy 所列的强增强 |
| 4–69 | 每个样本进入 Mosaic；原生策略跳过同一样本的 ZoomOut/IoUCrop |
| 70–129 | Mosaic 关闭；IoUCrop 的条件执行概率为 0.5 |
| 130–159 | 关闭 policy 所列的强增强 |

所以 `RandomIoUCrop.p=0.5` 不意味着全部 160 轮中每张图都有 50% 概率裁剪；不能把原生互斥分支描述成 Mosaic 后总是继续裁剪。

## 代码核查与隔离修复

新 `augmentation_runtime.py` 注册两个仅用于训练的类：

- `AlignedSanitizeBoundingBoxes`：沿用原生完全相同的筛框/标签规则，同步筛选原有 area/iscrowd，并根据框重算 area。
- `AlignedBatchImageCollateFunction`：沿用原生 MixUp 的图像、框、标签和随机数行为，补齐相同来源顺序的 iscrowd，并重算 area。

三队列 CPU 对照检查包括实际裁剪删框、Mosaic、MixUp 和普通样本；修复前后图像/框/标签和 Python/Torch RNG 保持一致。原生验证路径未改。Mosaic 合成的多源 image_id/orig_size 仅作为训练元数据保留，不伪装成真实临床病例。

模块核查未发现 CMFA、结构化 DN/main 索引或 DualOpt 更新中的阻断性错误。Factor/Prototype 仅为训练辅助，推理融合系数为 0；DENTEX prior 继续使用未增强训练标注估计的模板，不使用验证标注或 Mosaic 坐标重新拟合。DENTEX 只有恒牙类别，因此其 dentition head 不能证明乳恒牙识别能力。

上游 EMA 的 decay_fn 捕获构造时 decay，后续修改 `.decay` 并不等同于修改实际函数。本轮保留上游 EMA 与验证驱动的阶段回退，不声称修复或改良了 EMA，也不将其作为选定模型独享的技巧。

## 检查、运行与保存

- `config_audit.json`：三个 Plain/selected 配置对严格核验。除声明的模块、辅助损失、DualOpt 和推理先验外，配方完全相同。
- `augmentation_validation.json`：CPU 图像/框/标签/RNG 等价及目标元数据修复检查。
- `queue_validation.json`：队列资源、进程归属、信号和结果验证的 mock 单元测试。
- `probes/*/probe_result.json`：真实 batch=8、固定640、强增强/裁剪/无强增强的 AMP 前向反向和 EMA，以及 batch=16 的 FP32 验证；DENTEX 另测试272个目标的密集上界。
- `smoke_runs/`：三个独立3轮 solver 测试，覆盖 Mosaic、crop、RAM stage-1 回退、最佳 checkpoint 重载、COCO复评及预测导出，不作论文结果。
- `preflight_validation.json`：全部检查通过后才能启动正式训练。

正式结果目录：

`../runs/deim_s_paper_modules_mosaic640/<task>/seed_0/`

每个正式实验仅保留全160轮验证 COCO AP 最优的 `best.pth`；同分保留最早 epoch。保留完整训练日志、配置关联、最佳/最终指标、最佳权重复评、COCO预测和校验信息。需要的原生 stage-1 状态只放RAM，不保存 last、周期或额外阶段权重。

中断后不把 `best.pth` 冒充最后一轮或等价断点。未完成目录默认拒绝覆盖，异常立即停止后续队列。

`queue_state.json` 记录实际任务、进程和等待原因；只管理本队列创建的进程组。GPU0若有其他账户任务，不停止它们；显存不足则等待。每个训练进程的PyTorch显存上限为9GiB，启动余量由完整batch预检决定。共享GPU的时延不可当作论文效率基准。

## 结果解释

本轮属于新增强配方的 seed-0 内部验证，不能保证 AP 提升或领先。相同新增强的 Plain 完成前，不能把新旧实验之差单独归因于模块。结果无论正负均保留；不自动改论文或选择性省略实验。
