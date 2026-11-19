# 官方 DEIM-S：论文主配置最小迁移实验

用户选择：只跑三个主配置，每项 seed 0、160 epochs。没有增加消融搜索，
不修改论文，不修改 `/home/wuxi/Dent-FINE`，不修改官方 clone 或已有实验。

| GPU 队列 | 配置 | 实际迁移的模块 | 正式结果相对路径 |
|---|---|---|---|
| GPU 0 | Smartphone | DualOptAdamW | `../runs/deim_s_paper_modules/phone_dualopt/seed_0` |
| GPU 1，第 1 项 | DSLR | 四个 CMFA + DualOptAdamW | `../runs/deim_s_paper_modules/dslr_cmfa_dualopt/seed_0` |
| GPU 1，第 2 项 | DENTEX2023 | Factor heads + Prototype + SpatialShape | `../runs/deim_s_paper_modules/dentex_spatial_shape/seed_0` |

## 官方基底与排除项

基底是独立 clone 的官方 DEIM-D-FINE-S（HGNetv2-B0），commit
`09d35d53d39ee3145a1e61e3a989b28b9468d1dd`。`paper_modules.py` 只包含以上
选定模块及必要的接入、核查逻辑；训练时没有从旧 Dent-FINE 项目导入代码。

保留官方 decoder 层、最近邻上采样、二维正弦/余弦位置编码、匹配与主损失。
不迁移 QueryFusion、DySample、可学习行列位置编码、query/graph/ROI refiner，
也不加入本轮未选用的 HOOA、CEDv2、SMWG、MPF、UCM 或全栈组合。

CMFA 在两个 FPN、两个 PAN 的拼接位置使用相同的互门控公式。
DualOpt 使用论文相同的参数枚举 rank、衰减和初始化回退公式；新增的仅是
初始化锚点的 checkpoint 序列化，便于保存和恢复，不改变不中断训练的更新。

结构化辅助头通过原始 decoder 层输出接入，对 main、aux、DN 分支按原配方
提供 quadrant/position/dentition CE 监督；每类四个原型、温度 0.20，Prototype
CE 只用于最终主查询。四个损失权重为 0.10/0.10/0.05/0.05。
因子头和原型不融合入主分类分数，不引入额外查询修改。

SpatialShape 模板仅从原 DENTEX **训练**标注重新估计；与此前模板张量逐值
相等。center/shape 权重为 0.20/0.05，view 权重 0.20。panorama 下 32 个
永久牙类别全部可见，因此 view 项为零，不声称有额外视图判别效果。
保留原生 top-300；无额外 score 截断、NMS、唯一性分配或外观先验。

## 比较协议

除明确加入的模块/损失/优化器/先验，各项配置均来自已完成的对应 cohort
官方 DEIM baseline。沿用原数据划分、标注、label map、ImageNet 骨干初始化、
增强、学习率/调度、batch 8/16、AMP、EMA、评估器和 AP checkpoint 选择规则。
不从已训练 detector 的最佳权重微调。

训练在 epoch 130 前保留官方比较配置的 480–800 多尺度采样；验证固定为
640×640。保留本次官方配置的 `num_denoising=100`，不带入旧项目的 80。
最佳模型是 160 轮中验证 COCO AP 最大者，完全相同时选择最早 epoch。

本轮为新的 **seed-0 迁移实验**，其直接对照是同 cohort 已完成的官方
DEIM seed 0；不能与旧项目三种子结果合成均值，不构成独立测试或 SOTA 确证。
既有数据划分及其内部验证限制不因模块迁移而改变。

## 校验与保存

`unit_validation.json` 检查 CMFA 前向/梯度、DualOpt 更新和状态恢复、官方
基础参数未变、无隐藏模块、新增参数梯度、原生主输出一致性与 DN 分支。
`preflight_validation.json` 另记录三个两轮 AMP/EMA 训练、stage-1 RAM 回退、
最佳模型复评、只留最佳权重、原代码/论文校验值和计算量检查。
测试用权重在预检成功后清理，测试日志和校验值保留；没有清理历史研究权重。

正式实验每项只保留 `best.pth`。原生 stage-1 rollback 快照保存在 RAM；
不保存 last、周期权重或 stage 权重。意外中断时不能保证恢复到最后一轮。
结果目录已存在未完成日志时程序拒绝覆盖；磁盘不足 3 GiB 时安全停止。

每项训练完成后自动导出：

- `metrics_summary.json`：最佳/最终 AP、AP50、AP75、AR100、epoch、参数量、
  GMACs，以及相对官方 seed-0 对照的差值。
- `verified_coco_metrics.json`：重载最佳权重后的 COCO 复评和 SHA256。
- `best_predictions.coco.json.gz` 与 `best_coco_arrays.npz`：后续可视化和类别分析。
- `log.txt`、`training_stdout.log`、`run_protocol.json`、`architecture_audit.json`。
- `status.json` 与复评成功后才写入的 `COMPLETED`。

`queue_gpu0.json`、`queue_gpu1.json` 是有限训练队列状态；GPU 1 自动在 DSLR
完成后运行 DENTEX。队列不会自行扩展新配置、改动数据或挑选超参数。

模型计算量使用独立 CPU 进程测量，不消耗正式训练 RNG；calflops、FP32、
batch 1、640×640、deploy 后 model forward，**不含 SpatialShape 等后处理**。

后续加载新权重时需使用本目录冻结的配置和 `install_modules` 接口，不能将
其加载到旧 Dent-FINE 架构。所有原代码和旧权重继续保留。
