from pathlib import Path
import csv
import json
from collections import Counter
from hashlib import sha256

ROOT=Path(__file__).resolve().parent

def output_csv(name,rows):
    with (ROOT/'audit'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

audit=json.loads((ROOT/'audit/dataset_audit.json').read_text())
images=list(csv.DictReader((ROOT/'audit/image_hash_manifest.csv').open(encoding='utf-8-sig')))
duplicates=[]
for cohort,value in audit['cohorts'].items():
    info=value['content_duplicates']['decoded_rgb_sha256']
    for scope,groups in [('train_validation',info['cross_split_groups'])]+list(info['within_split_groups'].items()):
        for i,g in enumerate(groups,1):
            for r in g['members']:
                duplicates.append({'cohort':cohort,'scope':scope,'group':i,'pixel_sha256':g['hash'],'annotations_identical_to_6_decimals':g['identical_box_annotations_6_decimals'],**r})
output_csv('duplicate_members.csv',duplicates)
dimensions=[r for r in images if not r['error'] and (r['actual_width'],r['actual_height'])!=(r['annotation_width'],r['annotation_height'])]
output_csv('dimension_mismatches.csv',dimensions)
sources=json.loads((ROOT/'audit/source_manifest.json').read_text())
for item in sources:
    assert sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']

text='''# CMIG 补实验前置核查：2026-09-29

状态：仅完成数据核查、源码快照与实验设计准备；没有启动训练，没有修改数据划分，没有修改两份最新 Word。

## 用户已确认

- 没有独立测试集；本轮为受控内部验证。
- 暂不改变划分，先报告核查结果。
- 14 个主要配置使用种子 0/1/2，内部学习消融先使用种子 0。
- 沿用 GPU0 串行，结束后每个实验保留最佳权重；保留训练算法所依赖的必要临时状态，不以节省磁盘为由暗改训练流程。

## 一、当前数据的实际核查

| 数据域 | 训练/验证图像 | 训练/验证牙齿 | 跨训练—验证相同内容组 | 患者信息 |
|---|---:|---:|---:|---|
| Mobile | 878/220 | 9,837/2,459 | 4 组，涉及训练 4 张、验证 4 张 | 两侧均无患者 ID |
| Pano | 570/64 | 16,234/1,861 | 1 组，涉及训练 2 张、验证 1 张 | 两侧均无患者 ID |
| DSLR | 750/190 | 10,760/2,742 | 0 组 | 150/38 名患者，ID 无重叠；每位均有五个不同视图 |

2,672 张图像均可读取；COCO 图像/标注 ID 未重复，无孤立标注或未知类别。检查了所有图片文件 SHA256 与完整解码 RGB 像素 SHA256，两种方法得到相同的跨划分重叠组。三域间没有检测到完全相同的图像。未实施近重复感知检索；内容哈希不能排除相同患者的不同照片。

额外发现：

- Mobile 训练集内部有 3 个重复组（4 个额外副本）；验证集内部未发现重复。
- Pano 训练集内部有 8 个重复组（8 个额外副本，其中一组也涉及验证集）；验证集内部有 1 个重复组，共 3 张图像（2 个额外副本）。因此 Pano 的 64 条验证图像记录对应 62 个不同的精确图像内容。
- 相同图像的标注框集合并不完全一致（类别/坐标集合以六位小数比较）。这不自动证明哪份标注错误，但意味着不能不检查标注就任意合并或保留副本。
- DSLR 有 5 张训练图的实际宽或高比 JSON 记录少 1 像素，其框未超出实际图像范围。验证图尺寸均一致；这属于轻微元数据问题，不等于患者泄漏。原数据未改。
- Mobile 训练集 FDI 18、28、38、48 无样本。DSLR 训练集 FDI 15、17、18、23、27、28、37、38、47、48 无样本。已保存 32/52 类完整计数，不能将空槽位解释为有学习证据的类别。

跨划分重复的具体文件：

| 域 | 训练侧 | 验证侧 |
|---|---|---|
| Mobile | JPEGImages/836.jpg | JPEGImages/201.jpg |
| Mobile | JPEGImages/797.jpg | JPEGImages/190.jpg |
| Mobile | JPEGImages/796.jpg | JPEGImages/189.jpg |
| Mobile | JPEGImages/157.jpg | JPEGImages/34.jpg |
| Pano | train_612.png、train_179.png | train_412.png |

## 二、建议的处理顺序（本次未执行）

1. 在新标注文件中处理 Mobile/Pano 跨划分重复；不覆盖原标注和原图。若优先保留验证图像内容，至少需从训练侧移除 Mobile 4 条、Pano 2 条记录。
2. Pano 验证集的三份重复内容以及训练集内部重复应同时处理；先核查标注差异，再确定保留/融合规则，不直接按文件名挑一份。
3. 保存最终图像/患者分组、FDI 映射及 SHA256，冻结后再训练。即使完成图像去重，Mobile/Pano 缺少患者 ID，仍不能声称患者独立。
4. 没有未接触测试集时，不将现有验证集重新命名为测试集。三种子 SD/CI 只能补充内部稳定性证据，不能修复划分泄漏或选模偏差。

如果决定保留现用划分，所有新实验必须统一使用并明确为存在上述重叠的内部开发验证；不能在论文中写“历史重叠已解决”。

## 三、已准备的实验设计

`primary_experiments.csv` 含 14 个配置、42 个种子任务；对应 YAML 位于 `planned_configs_unfrozen/`，目前不可视为已批准可启动的冻结配置。训练/验证数据仍指向原划分。

| 域 | 主要训练配置 | 种子任务数 |
|---|---|---:|
| Mobile | 1024 基础/仅特征/仅身份/完整；640 基础/完整 | 18 |
| Pano | 基础/仅 Factor/仅 Prototype/两者 | 12 |
| DSLR | 基础/CMFA × AdamW/DualOpt | 12 |

Mobile 各主对照都保留相同的 NWD=0.50、normalizer=0.08 和 arch=0.10，因此基础行应称“受控基础模型”，不能冒充无这些背景损失的官方 DEIM。Pano/DSLR 保留各自原生 arch=0.25，NWD 不激活。

Pano 四组的训练内验证及选模统一禁用 factor 推理融合与 SpatialShape；训练结束后再对同一个 checkpoint 做无/中心/形状/联合先验重放。旧 Pano 最佳模型依据 prior-on AP 选择，不能直接声称满足这次 prior-off 的选模协议。

`internal_ablation_design.csv` 给出内部消融的控制量、复用关系与定义。复用本轮已经完成的相同 seed-0 基准后，计划额外约 20 次训练，总计约 62 次完整训练；不含重试或新增外部模型训练。该数量不是已启动任务数。

内部 CMFA/HOOA 开关还需在隔离源码中实现并做全开等价、形状及梯度测试后才能训练。HOOA 的 context 分支原本通过二阶交互进入输出，关闭二阶交互不能不加说明地同时删除 context；当前设计建议改为直接 context 通路，维持其余分支和投影尺寸。Factor 辅助监督的去头损失对照统一关训练/推理融合，避免把未监督头的推理噪声误算成监督效应；训练融合×推理融合另列。

代码检查还发现：Mobile 独立实现的 `save_best_only` 会影响阶段恢复流程，不能把它当成纯粹的磁盘选项在不同组之间切换。正式运行前要冻结每个域的训练器版本和阶段行为；主干代码未改。

最终 AP/AP50/AP75/AR100、固定阈值 P/R/F1、匹配 FDI、覆盖率、重复率、跨设备及效率协议在 `protocol_preparation.json`。RT-DETR-L 保留。外部对照先核对图像内容、标注语义和类别映射，不能仅因 JSON 排列不同就判不一致，也不能仅因图像数量相同就判一致。

## 四、尚需作者提供、不能从训练推断的内容

- 各队列数据来源/许可、采集机构与时间、纳排标准；Mobile/Pano 若可回溯，需要匿名患者映射。
- 标注人员资质、标注流程、复核与争议处理依据；重复图像不同标注的保留依据。
- 真实伦理委员会名称、批准号、书面知情同意或豁免依据、图片发表授权。最新主文目前只有一般性伦理描述，不能据此补造具体批准信息。
- 作者与 1–4 号单位的对应关系；主文指向独立 title page，应同步核对。通讯作者信息沿用作者已确认资料。

## 五、文件入口

- `audit/dataset_audit.json`：完整核查和重复组。
- `audit/duplicate_members.csv`：重复记录逐文件清单。
- `audit/image_hash_manifest.csv`：2,672 张图像哈希与元数据。
- `audit/per_class_counts.csv`：每域、每划分、每 FDI 类计数。
- `audit/dimension_mismatches.csv`：5 张 DSLR 尺寸差异图像。
- `audit/source_manifest.json`：原标注、最佳配置、最新两份论文的哈希。
- `source_snapshots/` 和 `source_snapshot_manifest.json`：独立源码快照，不含权重。

本次没有生成新实验数值、置信区间或预测图，也未把旧权重图片改注为新实验。按你的要求，先交付划分核查结果，等待数据处理决定后再启动正式队列。
'''
(ROOT/'PREFLIGHT_核查与实验准备.md').write_text(text)
print(ROOT/'PREFLIGHT_核查与实验准备.md')
