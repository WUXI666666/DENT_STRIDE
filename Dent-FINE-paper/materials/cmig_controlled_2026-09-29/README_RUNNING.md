# 双 GPU 受控消融训练

本轮由用户明确授权沿用原训练/验证划分，不修改数据集。Mobile 4 组、Pano 1 组跨划分重复仍保留；所有结果属于内部开发验证，没有独立测试集。

## 队列

- `execution_manifest.json`：最终 62 个训练任务、冻结配置和代码/数据校验值；此文件取代先前的未冻结计划。
- GPU0、GPU1 各 31 个任务，各卡串行。首任务分别是 Mobile 完整 1024 seed0 与 Pano Factor+Prototype seed0（无因子融合/几何先验选模）。
- 主配置 14 × seeds 0/1/2 = 42 次；内部消融 seed0 额外 20 次，当前没有复用历史训练。
- 配置在 `configs/`，任务在 `jobs/`，结果在 `runs/<configuration>/seed_<seed>/`。
- `gpu0_state.json`、`gpu1_state.json` 和各 GPU queue.log 由队列自身记录状态；助手启动确认后不持续监视。

## 控制与保存

所有主任务160轮、训练batch8、验证batch16、AMP训练、FP32原生验证、EMA、最大验证AP选模；并列时保留首次达到的epoch。域内来源/增强/背景损失一致，除明确消融量外不变。学习型内部消融全部从同一预训练骨干重新训练，非推理置零。

模型使用 `source_snapshots/` 的隔离源码，`controlled_runtime.py` 提供受控开关。CMFA关闭通道门时移除门控交叉残差，关闭空间门时使用恒等映射，保持2C输出；HOOA关闭高通时去掉edge贡献，关闭二阶交互时用直接context替代乘性交互，保持local/context与3C投影。全开设置与原forward相同，已做前后向测试。

Factor去监督实验的训练/推理融合均关闭；融合本身用两个训练状态分别做推理on/off重放。Prototype的K指每类原型个数。DualOpt消融保持普通AdamW衰减不变。Mobile背景损失单独做2×2比较。

每次任务执行独立、可丢弃的真实训练batch及验证batch预检，然后才正式训练。最终保存 `best.pth`、完整配置、日志、选中epoch、COCO指标、固定阈值指标、COCO预测.gz、完整pre-top-k候选.npz及文件SHA256。

临时stage1 checkpoint若被原生训练器读取则在训练期间保留，避免改变阶段算法；复测成功后删除本次任务的冗余临时权重，只保留best.pth。原项目、原数据、原权重、最新两份Word均不改。

Pano Factor+Prototype三个种子完成后，自动在同权重、同pre-top-k候选上评估center/shape/both，与默认无先验结果构成四种先验对照。Mobile full seed0与train-fusion-off seed0会自动完成相反推理开关的重放。

每个任务成功后更新 `results_summary.csv`；三个种子均完成的主配置才写入 `three_seed_mean_sd.json`，采用样本SD。失败、输入哈希变化或磁盘不足8 GiB时对应队列停止，记录原因，不自动换配置或覆盖半成品。

跨设备、完整效率评估、配对CI、最终病例图片与Word更新不由本训练队列自动宣称完成；应在最终权重和主要训练结果齐备后继续完成。已有独立测试集缺失与伦理/标注资料缺口见前置核查说明。
