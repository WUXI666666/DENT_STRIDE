DSLR分视角结构化身份增量实验

本轮按用户确认：逐项加入模块，各组独立训练，seed0，160轮；不是从现有检测权重继续微调。
不修改原始项目、历史实验权重、数据集或划分。新代码和结果仅在本目录。

三组实验（均保留原DSLR CMFA与AdamW额外衰减设置）
 D01_cmfa_factor：加入象限、牙位、乳恒牙三种因子辅助监督。
 D02_cmfa_factor_viewproto：D01加分视角原型辅助监督。
 D03_cmfa_hooa_factor_viewproto：D02再加HOOA。

共同训练条件
 验证640×640，训练仍为原640验证协议对应的legacy多尺度计划。
 Mosaic=0，MixUp=0，RandomIoUCrop=0.8，RandomZoomOut=0.5。
 batch训练8、验证16；AMP、EMA；同一ImageNet预训练HGNetv2-B0。
 原AdamW基础weight_decay=0.0001、额外衰减0.01，rollback=0。
 保持原检测损失mal/boxes/local，不额外加入NWD。
 实际启动日志核实：原DSLR基线本来就有非零loss_arch（首轮0.042497），本轮沿用同一继承行为和原weight_dict.loss_arch=0.25，不把它当作新模块收益。
 额外因子权重与Mobile实际配置一致：quad0.08、tooth0.08、dent0.03。
 因子训练/推理得分融合都关闭，以分离辅助监督收益；不把Mobile的推理融合收益混入。
 原型K=4、温度0.50；loss_proto实际权重0.03（不是无效属性prototype_weight=0.5）。
 原型推理融合为0，属于训练辅助监督。

分视角原型
 原型张量[5,52,4,256]，五个视角各有独立银行。
 0 frontal；1 upper_occlusal；2 lower_occlusal；3 left_buccal；4 right_buccal。
 依据COCO image metadata的view_id路由，不用GT牙位去推断视角。
 原型与query向量归一化，cosine/temperature后对K个原型LogSumExp，匹配查询CE。
 未知视角明确跳过监督，不能默认为frontal；数据视角冲突会报错。
 Mosaic/MixUp保持关闭，以防一张混合视角图错误进入一个原型银行。

几何先验
 只用现有训练标注10760个框建立每视角每FDI中心、log宽高二维均值和完整协方差。
 协方差加1e-3I，非负方差/数值稳定性检查；缺样本类别无证据、不做人工硬遮罩。
 固定center权重0.20、shape权重0.05；在sigmoid和最终top300前加入有界相对得分。
 同一最佳权重分别计算none/center/shape/both四种结果，不再重训四次。
 先验是训练集统计得到的推理评分校正，不是几何训练损失；避免把未变换的原始坐标先验用于随机裁切后的框。
 最佳checkpoint按无先验验证COCO AP选择，四种先验均复测同一个checkpoint，不为不同先验挑不同权重。

数据核对
 train750图、150患者、10760框；val190图、38患者、2742框。
 每患者五视角齐全，患者ID和文件名在两侧无交集；未改变任何数据。
 具体统计与稀疏视角类别见priors/data_checks.json；不把ID/文件名检查称为完整内容重复审计。

运行与保留
 双GPU后台独立运行；先GPU0启动D01，GPU1启动D02，第三组由先空闲的worker领取。
 启动前完成所有组真实训练batch前向/反向、每视角梯度、验证batch和EMA检查；这些更新不进入正式训练。
 训练保持原生选模和阶段回退。best_stg1.pth在训练中临时保留，复测成功后移除；最终每组只保留best.pth。
 失败不会自动复用半截训练或删除已有结果，会记录FAILED.json并停止该worker，防止污染结果。
 新增代码/source/input哈希冻结于execution_manifest.json，启动后不能随意修改。

结果文件
 runs/<配置>/seed_0/console.log与log.txt：完整训练记录。
 runs/<配置>/seed_0/metrics_summary.json：AP/AP50/AP75/AR100、参数量、最佳轮次等。
 runs/<配置>/seed_0/prior_results.json与csv：同权重四种先验对照。
 runs/<配置>/seed_0/view_evaluation：完整pre-topk候选、视角、压缩预测、固定阈值指标和来源。
 results_summary.csv：完成后自动汇总。
 gpu0_state.json、gpu1_state.json：后台worker状态；LAUNCHED.json：PID。

不预先保证AP提高；这些独立对照用于检验各项新增机制是否有效。
