# 原始 DEIM 与项目内 Plain DEIM：参数与代码核对

本次为独立的官方来源比较。此前 Plain DEIM 应理解为项目内消融对照，不能等同于未经修改的官方 DEIM。

| 52 类配置 | 训练结构总参数 | deploy 后总参数 |
|---|---:|---:|
| 官方 DEIM-D-FINE-S | 10,285,925 | 10,215,931 |
| 项目内 Plain DEIM | 10,629,288 | 10,559,294 |

训练结构总参数相差 343,363，按规范化后的参数名可分解为：

- decoder_QueryFusion: 198,912
- encoder_DySample: 16,448
- encoder_learned_position_embeddings: 128,000
- frozen_graph_gates: 3

代码检查确认 QueryFusion、DySample 和可学习位置编码仍参与旧 Plain 的 forward。关闭本轮模块开关并不会自动移除这些历史基础改动。

因此，既有三种子配对差值衡量的是在项目内部基础结构上增加 DualOpt/CMFA/SpatialShape 配置的收益；新官方重训将另行衡量相对官方 DEIM 的总体差值。两类证据均保留，不互相替换。

历史 10.2 M 与官方 deploy 后四舍五入数值相容，但仅凭数值相同不能证明历史记录的真实来源。训练结构与 deploy 结构的统计口径也需明确区分。

该检查定位了参数差异及三个实际执行路径，不代表所有同名组件已被证明数值等价。完整来源、参数名和代码位置见 architecture_audit.json。
