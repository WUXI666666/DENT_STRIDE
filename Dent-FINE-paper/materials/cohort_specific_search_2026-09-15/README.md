# 分模型配置搜索（2026-09-15）

本轮只对 Phone DualOpt、DSLR CMFA+DualOpt、Phone A07 Proto Aux 分别选配置。DENTEX AP 54.385458 已单独归档，不重训。论文、模型 Python 源码、既有配置、数据集及旧实验均不改动。

每个模型比较 9 个候选：复用 1 个已完成且复测一致的 fixed640/MixUp=0.5 结果，新增 5 个核心候选及 3 个自适应单变量候选，共新增 24 次训练。

| 阶段 | 配置 |
|---|---|
| 核心网格 | legacy640 多尺度、固定 640、固定 1024 × MixUp 0 或 0.5；Mosaic=0、IoUCrop=0.8、ZoomOut=0.5 |
| 单变量验证 | 分别在该模型核心网格胜出配置上，仅改 IoUCrop=0.5、ZoomOut=0、或 Mosaic=0.25 |

三项单变量改动不自动叠加，不能把未训练的组合称为最优。legacy640 在训练前期使用原生 480–800 多尺度，验证仍为 640×640；不能把它写成固定 640 训练。低概率 Mosaic 使用外层 0.25、内层 1.0，避免双重概率相乘；原生启停时间及与 IoUCrop/ZoomOut 的互斥行为保留。

Phone/DSLR 的 C01 恢复历史多尺度、MixUp=0.5；A07 的 C01 恢复历史多尺度、MixUp=0。主模型保留历史 Arch loss 权重 0.25；A07 保留自己的 0.1 及历史 best-only 行为。模型、损失、优化器及其他模块不在本轮搜索范围内。

每次 seed 0、160 轮、从原初始化重新训练，batch 8 / 验证 batch 16；按验证 COCO AP 选 checkpoint，结束后独立复测 AP/AP50/AP75/AR100 与最佳日志核对。每个候选保留配置、日志、指标及 best.pth；只有复测通过才删除该新运行的 last/stage 临时权重。

GPU0 串行。其他 GPU0 任务不受干预；优先执行当前显存可容纳的候选，资源不足等待，不更换 GPU、不改变 batch。少于 5 GiB 磁盘空间会停止队列并保留现场。遇到失败不静默跳过、不覆写或自动续训部分结果。

- 配置：`/home/wuxi/Dent-FINE/configs/paper_cohort_search_20260915/`
- 运行：`/home/wuxi/Dent-FINE/outputs/paper_cohort_search_20260915/`
- 汇总：本目录 `results_summary.csv`、`results_summary.json`
- 状态：本目录 `queue_state.json`、`queue.log`
- 选优：本目录各模型的 `*_selected_result.json`
- 归档：`/home/wuxi/Dent-FINE-paper/materials/best_model_bundle_2026-09-15/`

`starting_models_verified.zip` 已冻结用户指定的四项现有结果；全部搜索结束后生成另一个 `selected_models_after_search.zip`，包含保留的 DENTEX 与三个模型各自本轮最佳配置/权重。若新候选没有超过复用结果，则保留复用结果为本轮最佳。

这是有限候选集内的内部验证选优，并非全局最优、锁定测试或统计显著性结论；所有低分候选同样保留在结果表中。
