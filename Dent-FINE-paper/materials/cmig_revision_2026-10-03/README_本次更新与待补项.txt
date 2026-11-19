本次论文更新（2026-10-03）

一、交付文件
主文：/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_main_controlled_results_20261003.docx
补充材料：/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_supplement_controlled_results_20261003.docx
原 main_visual.docx 与 supplement_visual.docx 未被覆盖。

二、结果范围
本次冻结结果记录时间为 2026-10-03 22:25:14（北京时间）。
已完成 42/42 个主实验（14配置×3种子）及16/20个内部训练消融。
仅汇总训练完成、选中权重及评估文件可核对的结果。未采用未完成日志中的最佳AP。
完整配置、选中epoch及逐种子结果在补充材料；结果来源与校验文件在 evidence/。
主文原有51个Word数学公式对象保留，不转换为下划线文本或截图。

三、主要更新
1. 主文Table 3与Figure 6：三域模块组合和Mobile分辨率的14组受控三种子结果。
2. Table 4：与对应受控参照配对的AP差值及Student-t 95%区间。
3. Table 5与Figure 7：官方DEIM-s、RT-DETRv2-S及外部框架对比；保留RT-DETR-L。
4. Table 6与Figures 9–10：同权重、top-k前的无/中心/形状/联合先验对照，含有利与不利案例。
5. Table 7与Figure 11：门控、分支、因子监督与融合、原型设计、DualOpt和背景损失内部消融。
6. Table 8：640/1024受控对照；Table 9：P/R/F1、条件FDI准确率、覆盖率、原始重复编号率。
7. Figures 8与12A–C：实际seed-0权重、同图同局部区域、相同颜色/阈值规则的比较。
8. Figures 13–15：实际原型、辅助头混淆矩阵、真实CMFA/HOOA forward张量；不用Grad-CAM代替门控。
9. 补充Figures S5–S11：真实三种子曲线、其余内部消融、检测/阈值曲线、逐类原型内聚度、完整FDI混淆矩阵及类别训练/验证支持数。
10. 总体图已从原生矢量重导并修正预测头/因子连线和并行损失；SMWG矢量图修正P5尺寸并增加底部留白。

四、贡献及解释
核心贡献保留结构化身份机制、独立的训练来源解剖先验，以及受控的域配置框架评估。
SMWG、MPF、UCM、CEDv2未启用在主配置中，作为补充比较保留，不列为主要贡献。
不将“所有模块都有效”“全域SOTA”“临床自动牙位记录”写成已证实结论。
DSLR CMFA与Pano中心先验表现出较一致的正向效应；联合先验没有超过中心单独使用的均值。
Mobile较高分辨率的AP变化大于完整模块组合的AP变化；若干内部简化在单种子中高于完整配置。
三种子区间反映固定验证集上的训练变异，不是患者抽样区间或独立测试结论。

五、待真实结果补齐的格子
主文Table 7：HOOA去二阶、HOOA高通与二阶都去、Prototype K=2/温度0.20、K=4/温度0.50。
主文Table 10：本轮受控权重的双向跨设备评估。
主文Table 11：本轮受控权重的统一计算量、时延、吞吐与峰值显存。
这些位置统一使用 —，不是0，不作结论依据。论文仍为可继续补数据的版本，并非无缺项投稿终稿。

六、图片与复核
publication_figures/内图文件名与新版Word图号一致；Figure_index.csv记录对应图注与真实源文件。
新增图使用Times New Roman、白底、克制配色；图内不含Main Figure、实验计划或协议脚注。
主要图有600dpi PNG及原生SVG/PDF；SMWG为1000dpi。对无矢量源的旧图不伪造矢量或上采样宣称合规。
保留的CMFA/HOOA/CEDv2/MPF–UCM原始栅格图仍存在印刷清晰度限制；投稿前宜补同内容可编辑原图。
数值与跨引用复核：independent_main_QA.json、supplement_document_checks.json。
检查用PDF：render_main/、render_supplement/。
作者需补伦理、发表许可、采集资料与逐作者单位映射：submission_missing_information.txt。

七、CMIG写作与绘图风格参考
仅参考图文组织、并列病例布局、必要标签、清楚的对照和克制配色；没有复制文章图像或研究结论。
Mathai et al., 2024. Detection of abdominopelvic lymph nodes in multi-parametric MRI.
Computerized Medical Imaging and Graphics 114, 102363.
DOI: https://doi.org/10.1016/j.compmedimag.2024.102363
作者网页PDF: https://www.moreisdifferent.com/assets/my_papers/B_AI_medical_imaging/2024_Mathai.pdf
Arsa et al., 2024. Efficient multi-stage feedback attention for diverse lesion in cancer image segmentation.
Computerized Medical Imaging and Graphics 116, 102417.
DOI: https://doi.org/10.1016/j.compmedimag.2024.102417
作者机构PDF: https://robot.jbnu.ac.kr/bbs/robot/3812/304348/download.do
两篇已下载到references/用于本地查看。
