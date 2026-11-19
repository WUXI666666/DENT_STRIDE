# 官方 DEIM-S 主实验：固定输入 1024×1024

本轮按用户要求将网络输入从 640 改为 1024，使用独立实验目录，从同一官方 ImageNet HGNetv2-B0 骨干重新初始化。旧 640 队列已于 2026-09-12 22:45 停止：Phone 完成 24 轮（零基 epoch 0–23），DENTEX、DSLR 尚未启动。保留旧 Phone 日志及唯一 best.pth，不续训其检测器权重，也不将其作为完整实验结果；不修改原 `/home/wuxi/Dent-FINE`、官方 clone、原始数据或论文。

## 唯一配方变更

相对 `../module_migration_mosaic640_2026-09-12/` 的六份冻结配置，仅训练/验证 `Resize.size`、`eval_spatial_size`、训练 collator 的 `base_size` 从 640 改为 1024，另隔离输出目录和字节不变的 DENTEX prior 路径。`base_size_repeat: null` 保持不变，网络输入固定 1024×1024。

Mosaic 算子完整参数保持原样，尤其 `output_size: 640` 不改：

```yaml
- {type: RandomIoUCrop, p: 0.5}
- {type: Mosaic, output_size: 640, rotation_range: 10,
   translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5],
   probability: 1.0, fill_value: 0, use_cache: False,
   max_cached_images: 50, random_pop: True}
```

这里的 Mosaic `output_size` 是源图缩放尺度，不是最终网络输入；拼接结果最终 Resize 到 1024×1024。train batch=8、validation batch=16 不调整；显存不足时停止或等待，不自动减小 batch、分辨率或改变算法。

仍为 seed 0、160 epochs、AMP 训练、原生 FP32 验证、EMA、100 个 DN 分组目标、flatcosine 与原参数。保留 `Compose.mosaic_prob: 1.0`、增强边界 `[4,70,130]`、`no_aug_epoch: 30` 和 MixUp 概率 0.5 / `[4,80)`。不新增翻转，不修改数据划分、类别、标注或先验估计。Mosaic 阶段原生 Compose 跳过同一样本的 ZoomOut/IoUCrop；`RandomIoUCrop.p=0.5` 不是所有 epoch 的无条件裁剪概率。

## 默认训练范围

GPU0 串行执行以下三个已选主配置：

| 顺序 | 任务 | 启用模块 |
|---|---|---|
| 1 | `phone_dualopt` | DualOpt |
| 2 | `dentex_spatial_shape` | Factor heads + Prototype + SpatialShape |
| 3 | `dslr_cmfa_dualopt` | 四处 CMFA + DualOpt |

三个同配方 Plain 配置最初仅备好、不训练。2026-09-13 三个 selected 主实验全部完成后，用户明确授权在 GPU0 继续串行训练 `phone_plain → dentex_plain → dslr_plain`，通过队列的 `--tasks` 显式选择任务；不重新冻结协议、不修改配置、不重复训练 selected。原主实验队列完成记录保存在 `queue_state_selected_completed_2026-09-13.json`；`queue_state.json` 记录最新队列状态。不得将配置已生成、检查通过或 smoke 测试写成正式实验完成。没有新增 QueryFusion、DySample、可学习位置编码或任何未选模块。

`paper_modules.py` 和 `augmentation_runtime.py` 与 640 版本逐字节一致。两个训练期 aligned 类保留相同图像、框、标签和 RNG 行为，只修复逐实例元数据对齐；验证路径不改。Factor/Prototype 仍仅为训练辅助，推理融合系数为 0；DENTEX SpatialShape prior 复用原训练标注模板，不使用验证数据重拟合。DENTEX 只有恒牙类别，不能据此宣称乳恒牙识别能力。

## 冻结、核验与运行

`prepare_migration.py` 在全部新 Python 源文件完成后运行一次，拒绝覆盖已有冻结。它记录来源 640 协议与六份配置的 SHA-256，严格核验全部非分辨率参数不变，并核验每个队列的 Plain/selected 只有声明的模块差异。原始代码、旧配置、旧协议和论文受前后哈希保护。

新版本必须重新完成 CPU 增强等价检查、队列 mock 测试、真实 batch=8 的 1024 AMP 前后向/EMA 与 batch=16 的 1024 FP32 验证显存检查，以及独立三轮 solver smoke 测试。640 的预检结果仅作来源证据，不能替代本轮验证。只有本目录新的 `preflight_validation.json` 全部通过后才允许正式训练。

正式结果保存到：

`../runs/deim_s_paper_modules_input1024/<task>/seed_0/`

每项仅保留全 160 轮验证 COCO AP 最优的 `best.pth`，同分保留最早 epoch；原生 stage-1 回退状态只放 RAM。保留完整训练日志、最佳/最终指标、最佳权重复评、COCO 预测与校验信息，不保存 last、周期或额外阶段权重。中断后不将 best 权重冒充等价断点，异常停止后续队列，未完成目录默认拒绝覆盖。

队列只管理自身创建的进程组，不停止其他账户任务。当前预检与训练使用 10752 MiB（10.5 GiB）的 PyTorch 显存上限；启动余量至少为上限加 768 MiB，并服从预检推荐值。显存不足不会自动降低 batch。共享 GPU 时延不用于论文效率结论。

## 结果解释

本轮是固定输入 1024 的 seed-0 内部验证，不保证 AP 提升。与 640 版本的对比包含输入分辨率变化，不能纯归因于模块；同分辨率、同增强 Plain 未完成前，也不能将新旧结果差异认定为模块收益。正负结果均保留，不自动改论文或择优省略实验。
