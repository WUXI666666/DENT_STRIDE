from copy import deepcopy
from pathlib import Path
import json
import math

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement


ROOT = Path('/home/wuxi/Dent-FINE')
WORK = ROOT / 'outputs/ppt_review_1030'
OUT = ROOT / 'wx_1030_补充完善版.pptx'
prs = Presentation(ROOT / 'wx_1030.pptx')
old = list(prs.slides)
RED, INK, GRAY = 'B92917', '171717', '5B5B5B'
LIGHT, LINE = 'FBF2F0', 'E8D1CC'
FONT = '微软雅黑'


def rgb(s):
    return RGBColor.from_string(s)


def fmt_font(font, size, bold=False, color=INK):
    font.name = FONT
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = rgb(color)
    props = font._rPr
    for tag in ['a:ea', 'a:cs']:
        el = props.find(tag, props.nsmap)
        if el is None:
            el = OxmlElement(tag)
            props.append(el)
        el.set('typeface', FONT)


def write(shape, value, size=17, bold=False, color=INK, spacing=1.13,
          align=PP_ALIGN.LEFT, after=3):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(after)
        p.space_before = Pt(0)
        fmt_font(p.font, size, bold, color)
        r = p.add_run()
        r.text = line
        fmt_font(r.font, size, bold, color)
    return shape


def pos(shape, x, y, w, h):
    shape.left, shape.top, shape.width, shape.height = map(Inches, (x, y, w, h))
    return shape


def text(slide, x, y, w, h, value, size=17, bold=False, color=INK,
         name=None, **kwargs):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        sh.name = name
    return write(sh, value, size, bold, color, **kwargs)


def rect(slide, x, y, w, h, fill=LIGHT, stroke=LINE, width=.6):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if stroke:
        sh.line.color.rgb = rgb(stroke)
        sh.line.width = Pt(width)
    else:
        sh.line.fill.background()
    return sh


def remove(shape):
    shape._element.getparent().remove(shape._element)


def named(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def notes(slide, value, append=False):
    tf = slide.notes_slide.notes_text_frame
    if append and tf.text:
        value = tf.text + '\n\n补充说明：\n' + value
    tf.text = value


def title(slide, value):
    try:
        sh = named(slide, 'slide-title')
    except StopIteration:
        sh = text(slide, 1.73, .365, 8.85, .57, '', name='slide-title')
    pos(sh, 1.73, .365, 8.85, .57)
    write(sh, value, 24, True)


def new_slide(value):
    s = prs.slides.add_slide(old[1].slide_layout)
    for sh in list(s.shapes):
        remove(sh)
    title(s, value)
    return s


def table(slide, x, y, widths, rows, heights=None, size=15.5,
          highlight=None, header_size=16):
    height = sum(heights) if heights else .52 * len(rows)
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(height))
    t = shape.table
    for c, w in zip(t.columns, widths):
        c.width = Inches(w)
    if heights:
        for r, h in zip(t.rows, heights):
            r.height = Inches(h)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = Inches(.12)
            c.margin_right = Inches(.10)
            c.margin_top = Inches(.04)
            c.margin_bottom = Inches(.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(RED if i == 0 else LIGHT if i == highlight else
                                       'FFFFFF' if i % 2 else 'F7F7F7')
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                p.line_spacing = 1.05
                p.space_after = Pt(0)
                fmt_font(p.font, header_size if i == 0 else size,
                         i == 0 or i == highlight,
                         'FFFFFF' if i == 0 else RED if i == highlight else INK)
                for r in p.runs:
                    fmt_font(r.font, header_size if i == 0 else size,
                             i == 0 or i == highlight,
                             'FFFFFF' if i == 0 else RED if i == highlight else INK)
            tcpr = c._tc.get_or_add_tcPr()
            for side in ['lnL', 'lnR', 'lnT', 'lnB']:
                e = OxmlElement('a:' + side)
                e.set('w', '5715')
                fill = OxmlElement('a:solidFill')
                co = OxmlElement('a:srgbClr')
                co.set('val', LINE)
                fill.append(co)
                e.append(fill)
                tcpr.append(e)
    return shape


def source_link(slide, x, y, w, label, url):
    sh = text(slide, x, y, w, .22, label, 9.3, False, GRAY, after=0)
    sh.text_frame.paragraphs[0].runs[0].hyperlink.address = url
    return sh


# New page 1: concrete research scope and a short narrative for the talk.
topic = new_slide('研究主题：牙齿定位与 FDI 编号联合建模')
text(topic, .55, 1.34, 12.1, .5,
     '面向不同采集条件，将“牙齿在哪里”与“是哪一颗牙”放入同一检测框架。',
     21, True, RED)
cards = [
    (.55, '研究对象', 'Pano｜全景 X 射线\nMobile｜手机口内照片\nDSLR｜单反口内照片',
     '三域分别配置、独立训练'),
    (4.82, '任务定义', '输入：完整牙科图像\n输出：牙框、FDI、置信度\nFDI 编码：象限 + 牙位',
     '以定位与编号的联合正确为目标'),
    (9.09, '核心问题', '外观相似：相邻牙位容易混淆\n局部困难：遮挡、模糊与弱边界\n结构约束：逐牙正确仍可能重复',
     '特征判别与结构一致性共同评估'),
]
for x, head, body, foot in cards:
    rect(topic, x, 2.08, 3.70, 2.65, 'FFFFFF')
    rect(topic, x, 2.08, 3.70, .08, RED, None)
    text(topic, x+.17, 2.36, 3.37, .38, head, 20, True, RED)
    text(topic, x+.17, 2.95, 3.37, 1.10, body, 16.8, spacing=1.2)
    text(topic, x+.17, 4.25, 3.37, .31, foot, 13.5, False, GRAY)
rect(topic, .55, 5.04, 12.24, 1.44)
text(topic, .74, 5.23, 2.0, .36, '本研究的切入点', 18.5, True, RED)
text(topic, 3.03, 5.20, 9.44, .95,
     '特征层：跨尺度融合与局部细化；查询层：结构化身份监督；评分层：牙位几何先验。\n在共同框架中分别验证三类设计，进一步分析编号一致性与泛化问题。',
     17.2, spacing=1.13, after=4)
source_link(topic, .55, 6.78, 7.5, '任务背景参考：Tuzoff et al., 2019；DENTEX, 2023',
            'https://arxiv.org/abs/2305.19112v1')
notes(topic, '本页先说明研究主题。研究对象是全景X射线、手机口内照片与单反口内照片。'
      '目标是在同一个检测查询中给出框、FDI类别和置信度。需要区分三个层次：框是否定位正确、'
      '定位成功的牙是否编号正确、同一图像的编号是否一致。最后一项目前仍是待解决的问题。\n'
      '三域模型独立训练，本页不表示已实现单模型跨域泛化。\n'
      '来源：原稿第1、3、4、15、16页及其演讲备注。外部背景：'
      'Tuzoff et al., 2019, https://pubmed.ncbi.nlm.nih.gov/30835551/ ；'
      'DENTEX 2023, https://arxiv.org/abs/2305.19112v1 。')


# Existing review page: retain the three-column comparison, make the prior work explicit.
s = old[1]
title(s, '已有工作回顾与研究定位')
remove(named(s, 'related-work'))
table(s, .47, 1.48, [2.15, 4.7, 5.52], [
    ['已有思路', '已有工作的处理方式', '本研究的对应选择'],
    ['先定位、再编号', '先检测牙齿，再对局部区域分类\n代表：Tuzoff 等，2019', '共享检测查询，直接输出框与 FDI'],
    ['层级身份表示', '利用象限、牙位等分层标签\n代表：DENTEX，2023', '将象限、牙位与乳恒牙监督附加到同一查询'],
    ['解剖规则 / 先验', '利用牙齿排列、位置等信息辅助编号', '由训练框估计中心与形状分布，独立修正类别评分'],
    ['多尺度特征处理', '融合不同尺度并选择相关特征', 'CMFA 负责跨尺度交换；HOOA 负责局部细化'],
], heights=[.56, .80, .80, .80, .80], size=16, header_size=16.5)
sh = named(s, 'contribution')
pos(sh, .55, 5.56, 12.20, .98)
write(sh, '已有积累：完成三域数据配置、基线训练和模块筛选。\n本阶段重点：用受控实验区分特征增强、身份监督与几何评分的作用。',
      18.0, True, RED, after=5)
source_link(s, .55, 6.74, 6.2, 'Tuzoff et al., 2019｜doi:10.1259/dmfr.20180051',
            'https://pubmed.ncbi.nlm.nih.gov/30835551/')
source_link(s, 6.90, 6.74, 5.6, 'DENTEX, 2023｜arXiv:2305.19112v1',
            'https://arxiv.org/abs/2305.19112v1')
notes(s, '本页分别回顾外部研究路线和自己的前期积累。Tuzoff等使用检测后局部分类的路线；'
      'DENTEX提供层级标注，可作为结构化身份信息的参考。这里不把DENTEX说成与本方法相同的训练算法。'
      '本研究在DEIM/D-FINE检测基础上，将身份辅助监督接到共享查询，并单独检验训练几何统计的推理评分作用。'
      '前期工程基础见本地README、configs、outputs/ablation_*中的配置与实验汇总。\n'
      '文献来源：https://pubmed.ncbi.nlm.nih.gov/30835551/ ；'
      'https://arxiv.org/abs/2305.19112v1 。')


# New page 2: stage progress, intentionally before method details.
progress = new_slide('现阶段进展：模型、实验与问题诊断')
text(progress, .55, 1.32, 12.1, .55,
     '已形成三域实验链路，当前重点从“模型搭建”推进到“贡献解释与论文完善”。',
     20.5, True, RED)
table(progress, .55, 2.00, [2.17, 7.73, 2.32], [
    ['工作环节', '现阶段已完成内容', '阶段状态'],
    ['数据与基线', '建立三域数据配置及 FDI 映射；完成多类检测框架对照', '已完成'],
    ['模型实现', '实现 CMFA、HOOA、因子 / 原型监督与几何评分的按域配置', '已完成'],
    ['实验验证', '三域主结果均完成 3 种子评估；开展分支、门控与同权重先验对照', '已完成'],
    ['问题诊断', '分析相邻牙位混淆、原型重合与重复 FDI 输出', '已形成结论'],
    ['成果凝练', '补充结果解释、统一实验表述，继续完善论文与后续实验', '正在推进'],
], heights=[.55, .65, .65, .65, .65, .65], size=16, header_size=16.5)
for x, width, head, body in [
    (.55, 5.98, '已经得到的认识', '局部细化与牙位监督有条件收益；\nPano 中心先验贡献主要几何增益。'),
    (6.81, 5.98, '仍需解决的问题', '逐牙编号准确仍伴随图级重复输出；\n单模型跨域泛化与全局一致性尚待验证。'),
]:
    rect(progress, x, 5.87, width, .95)
    text(progress, x+.15, 5.98, 1.8, .30, head, 15.5, True, RED)
    text(progress, x+2.0, 5.96, width-2.15, .68, body, 13.6, spacing=1.12, after=1)
notes(progress, '这一页用来先回答“目前做到哪一步”。已完成数据配置、基线、模块实现和三域主结果；'
      '各类受控对照并非全部三种子：HOOA、因子、CMFA门控消融为seed-0重训，几何先验为同权重配对切换。'
      '当前实验支持局部细化和身份目标在参考配置下的作用，但不支持把所有模块视作跨域通用最优解。'
      '已有分析同时揭示原型重合、相邻牙位混淆与重复编号，为后续工作提供明确目标。\n'
      '来源：原稿第4、5、9、10、11、13、14、15、16、17页；本地README与outputs实验汇总。')


# Method pages: small textual additions that connect mechanism to purpose.
s = old[2]
write(named(s, 'slide-title'), 'DENT-STRIDE 总体框架', 24, True)
sh = s.shapes[8]
pos(sh, 9.38, 6.12, 3.47, .60)
write(sh, '三域独立训练\n输出：牙齿框 + FDI + 置信度', 16.7, True, RED, after=2)

s = old[3]
sh = named(s, 'config-conclusion')
pos(sh, .55, 6.12, 12.2, .42)
write(sh, '配置依据：Pano 利用几何布局，Mobile 强化局部细节，DSLR 保持结构简洁。', 17.1, True, RED)
text(s, .55, 6.66, 12.2, .32, '各域独立训练，配置在推理前固定；尚未验证单一模型权重的跨域泛化。',
     13.8, False, GRAY)

s = old[4]
write(s.shapes[4], '同一查询接受三个交叉熵目标。\nMobile：辅助监督 + 小幅分数修正。\nPano：只使用训练辅助监督。', 17.0, after=5)
write(s.shapes[7], '当前作用是查询表征正则化。\n同类原型接近重合，尚不能解释为\n4 种不同的牙齿形态。', 16.5, True, RED, after=3)
notes(s, '原型数量不等于已发现的形态亚型。多原型重合提示需要进一步检查温度、损失权重及多样性约束。', append=True)

s = old[5]
sh = s.shapes[9]
pos(sh, .55, 6.48, 12.2, .39)
write(sh, '作用：兼顾局部纹理与跨尺度上下文；Mobile / DSLR 在 4 处 FPN/PAN 融合位置启用。', 16.5, True, RED)

s = old[6]
text(s, .55, 6.65, 12.2, .33, '设计目标：强化弱边界与相邻牙差异；各分支贡献由后续消融检验。', 16.0, True, RED)


# Rewrite the crowded SpatialShape page without changing its technical scope.
s = old[7]
for sh in list(s.shapes)[2:]:
    remove(sh)
text(s, .55, 1.37, 8.05, .40, '按 FDI 类别估计中心分布', 18.75, True, RED)
s.shapes.add_picture(str(WORK / 'assets/slide8_3.png'), Inches(.55), Inches(2.04),
                     width=Inches(7.98), height=Inches(3.52))
text(s, .55, 5.83, 8.00, .93,
     '32 个恒牙 FDI 类别；坐标相对整幅全景片归一化。\n红十字表示中心均值；蓝色越深，位置与该类分布越相容。\n统计来源：570 张训练全景片，16,234 个牙齿框。',
     15.0, after=4)
text(s, 8.97, 1.37, 3.84, .40, '仅 Pano：从训练集估计', 18.25, True, RED)
text(s, 8.97, 2.02, 3.84, 3.23,
     '训练阶段\n按类别统计中心与对数宽高，\n估计均值与二维协方差。\n\n推理阶段\n计算预测框与类别分布的相容性；\n中心 / 形状权重为 0.20 / 0.05；\n修正类别 logits 后选择 top-300。',
     16.0, after=4)
rect(s, 8.82, 5.57, 3.98, 1.13)
text(s, 9.00, 5.74, 3.60, .87,
     '作用：调整类别评分与排序。\n边界：不移动预测框坐标，\n尚未施加 FDI 编号唯一性约束。', 15.8, True, RED, after=3)
notes(s, '32个面板对应32个恒牙FDI类。图中展示的是训练框中心的几何统计，不是可学习的原型向量。'
      '蓝色相容性图不表示对患者疾病的概率估计。由训练集框估计先验，验证集不参与统计。'
      '推理仅修正类别logits及其排序，不修改预测框坐标。中心/形状权重分别为0.20/0.05。\n'
      '来源：原稿第8页、其备注及第13页同权重先验对照。')


# Main results: preserve all reported values and add the evaluation definition.
s = old[8]
sh = named(s, 'result-limit')
pos(sh, .55, 6.38, 12.2, .38)
write(sh, '三域主结果均完成 3 种子评估；± 为样本标准差，描述训练波动。', 16.6, True, RED)
text(s, .55, 6.80, 12.2, .23,
     'AP 为 COCO bbox AP（IoU = 0.50:0.95），以百分点展示；各域的样本、类别与配置不同。', 12.2, False, GRAY)


# Recreate the screenshot as an editable table and reconcile Mobile AP with the main-results page.
s = old[9]
for sh in list(s.shapes)[2:]:
    remove(sh)
title(s, '完整框架比较：三域 AP 与基线增益')
rows = [
    ['方法', '种子数', 'Pano AP', 'Mobile AP', 'DSLR AP'],
    ['Official DEIM-s', '3', '52.827 ± 0.425', '53.912 ± 0.099', '72.405 ± 0.202'],
    ['RT-DETR-L', '1', '51.978', '54.718', '73.019'],
    ['RT-DETRv2-S', '3', '50.775 ± 0.213', '50.562 ± 0.486', '70.949 ± 0.506'],
    ['Faster R-CNN R50-FPN', '1', '47.695', '42.353', '63.151'],
    ['CAF-YOLO-s', '1', '48.058', '46.444', '67.460'],
    ['YOLOv8-s', '1', '51.431', '48.795', '71.854'],
    ['YOLO11-s', '1', '50.644', '49.601', '71.094'],
    ['Dome-DETR-S', '1', '51.711', '50.964', '71.363'],
    ['DENT-STRIDE', '3', '54.205 ± 0.144', '55.459 ± 0.275', '72.908 ± 0.429'],
]
table(s, .47, 1.40, [3.00, 1.00, 2.79, 2.79, 2.79], rows,
      heights=[.48]+[.43]*9, size=14.0, highlight=9, header_size=16.0)
text(s, .55, 5.91, 12.20, .37,
     '相对 Official DEIM-s：Pano +1.378，Mobile +1.547，DSLR +0.503 AP。', 18.0, True, RED)
text(s, .55, 6.43, 12.20, .54,
     '系统级比较，预训练、策略与后处理未完全统一；部分方法为单种子。\nDSLR 中 RT-DETR-L 的单种子结果为 73.019，高于本法三种子均值。',
     13.4, False, GRAY, after=2)
notes(s, '本页依据原稿完整比较截图重建为可编辑表格，清除截图中的拼写检查线与换行符。'
      'DENT-STRIDE的Mobile主结果以原稿第9页及其备注为准，统一为55.459±0.275；'
      '不使用截图中不一致的53.459。其余AP与种子数保持原截图。'
      'Pano和Mobile的AP在所列方法中最高；DSLR单种子RT-DETR-L为73.019，不能概括为三域均排名第一。'
      '相对DEIM-s增量分别为1.378、1.547、0.503 AP点，属于完整配置差异，不是单个模块的独立贡献。\n'
      '为便于阅读，本页将复杂度明细保留在备注。参数量(M)，按Pano/Mobile/DSLR顺序：'
      'DEIM-s 10.260/10.286/10.286；RT-DETR-L 32.872/32.913/32.913；'
      'RT-DETRv2-S 20.123/20.149/20.149；Faster R-CNN 41.454/41.556/41.556；'
      'CAF-YOLO-s 11.501/11.508/11.508；YOLOv8-s 11.148/11.156/11.156；'
      'YOLO11-s 9.440/9.448/9.448；Dome-DETR-S 13.266/13.291/13.291；'
      'DENT-STRIDE 10.650/12.513/11.154。\n'
      'DENT-STRIDE GFLOPs，Pano/Mobile/DSLR：25.316/78.353/25.394。'
      '其余复杂度明细请见保留的原始PPT第10页。未重新运行训练或评价。')


# Ablation photo: crop in the slide, retain original photos, omit the inconsistent historical AP panel.
s = old[11]
for sh in list(s.shapes):
    remove(sh)
title(s, 'HOOA 可视化：完整分支缓解牙位混淆')
text(s, .55, 1.42, 8.30, .42, '同一局部图像：比较删减分支与完整 HOOA 的编号', 18.3, True, RED)
pic = s.shapes.add_picture(str(WORK / 'assets/slide12_1.png'), Inches(.55), Inches(2.07),
                           width=Inches(8.40), height=Inches(3.48))
pic.crop_top = 180/504
pic.crop_bottom = 20/504
text(s, 9.38, 1.76, 3.40, .39, '示例一｜真实牙位 81', 18.0, True, RED)
text(s, 9.38, 2.35, 3.40, 1.16,
     '删减配置中输出为 41；\n完整 HOOA 输出为 81，\n与真实标注一致。', 16.5, after=5)
text(s, 9.38, 4.07, 3.40, .39, '示例二｜真实牙位 41', 18.0, True, RED)
text(s, 9.38, 4.65, 3.40, 1.16,
     '删减配置中输出为 42；\n完整 HOOA 输出为 41，\n与真实标注一致。', 16.5, after=5)
text(s, .55, 5.87, 8.40, .52,
     '列顺序：真实标注 / 两分支均去除 / 去高通 / 去二阶交互 / 完整 HOOA。',
     14.5, False, GRAY)
text(s, .55, 6.51, 12.20, .43,
     '这些示例展示可能的纠错方式；总体增益以同配置消融表为准，不能据此推断错误发生率。',
     16.1, True, RED)
notes(s, '原图片上方AP点图的Full HOOA=54.910，与原稿受控消融表的55.422不一致。'
      '本次仅在PPT版面上裁去该历史数值面板，保留原始b/c两组定性照片。'
      '两组照片均按真实标注、Neither、No high-pass、No second-order、Full HOOA排列。'
      '例一GT=81，三种删减配置为41，完整配置为81；例二GT=41，三种删减配置为42，完整配置为41。'
      '它们是选定的定性示例，不据此估计总体增益或类别错误率。对应总体定量结论采用前页消融表。')

s = old[12]
text(s, .55, 6.51, 12.20, .43,
     '研究启示：中心先验值得优先保留，形状项的适用条件仍需进一步检验。', 17.0, True, RED)

s = old[13]
notes(s, '这些权重对应当前实际评价配置，不表示跨域通用最优值。CMFA门控消融：'
      '完整55.422，仅通道54.810，仅空间54.428，均关闭54.383；与本页表格保持一致。'
      '该消融保持输出维度，为seed-0重训对照。\n'
      '来源：原稿第14页可编辑表格。旧备注中无门值54.783与表格不符，本次统一按表格54.383表述。')

s = old[14]
sh = named(s, 'slide-title')
write(sh, 'Mobile 定性比较：改善与失败并存', 24, True)
foot = s.shapes[-1]
pos(foot, .55, 6.57, 12.20, .47)
write(foot, '相邻牙位混淆仍然存在。score ≥ 0.30；示例仅展示可能发生的纠正和错误，不代表总体频率。',
      13.7, False, GRAY)

s = old[15]
remove(s.shapes[7])
sh = named(s, 'consistency-lead')
write(sh, '定位与编号需要联合评价：逐牙准确率较高，图级重复输出仍然存在。', 20.0, True)
sh = named(s, 'denominator')
pos(sh, .55, 5.24, 12.2, 1.00)
write(sh, '匹配牙 FDI 准确率：仅以定位匹配成功的牙齿为分母。\n重复标签图像率：至少出现一次相同 FDI 重复输出的图像占比。\nF1：正确检测同时要求框定位与 FDI 类别正确。', 15.7, after=4)
sh = named(s, 'clinical-scope')
pos(sh, .55, 6.51, 12.20, .42)
write(sh, '下一步重点：加入全局编号约束；重复图像率不能直接等同于临床病历错误率。', 16.5, True, RED)


# New page 3: user-confirmed paper outcome, with no invented venue or patent count.
achievements = new_slide('目前成果情况：论文与研究积累')
rect(achievements, .55, 1.43, 12.24, 2.65, 'FFFFFF')
rect(achievements, .55, 1.43, .09, 2.65, RED, None)
text(achievements, .82, 1.66, 6.6, .38, '论文成果  /  CCF-C', 21, True, RED)
text(achievements, 9.10, 1.69, 3.36, .35, '已录用 / 已发表', 17.5, True, RED, align=PP_ALIGN.RIGHT)
text(achievements, .82, 2.32, 11.68, 1.35,
     'Hybrid Frequency–Spatial Attention and\nArch-Aware Priors for Tooth Detection and FDI\nNumbering in Dental Images',
     22.0, True, INK, spacing=1.12, after=4)
text(achievements, .84, 3.74, 11.60, .26,
     '研究主题：频率–空间注意力与牙弓感知先验，面向牙齿检测和 FDI 编号。',
     14.8, False, GRAY)
for x, head, body in [
    (.55, '模型与工程积累', '形成训练、推理、评估与可视化工具；\n实现按域配置的特征增强、身份监督\n与几何先验模块。'),
    (6.81, '实验与分析积累', '完成三域主结果、多种基线与关键消融；\n形成相邻牙混淆、重复编号和原型行为\n的阶段性分析。'),
]:
    rect(achievements, x, 4.45, 5.98, 1.75)
    text(achievements, x+.19, 4.65, 5.58, .38, head, 18.5, True, RED)
    text(achievements, x+.19, 5.17, 5.58, .85, body, 16.0, after=3)
text(achievements, .55, 6.53, 12.20, .43,
     '成果延伸：继续完善期刊论文并推进 CMIG 投稿，拓展牙位相关龋病分割研究。',
     17.0, True, RED)
notes(achievements, '论文完整英文题目由用户提供；CCF-C分类依用户提供信息列示。用户已确认状态为'
      '“已发表或已录用”。由于尚未提供具体会议、年份、作者序和DOI，本页不虚构这些信息。'
      '本页不将PPT全部实验数值自动等同于该论文已发表的最终结果，也不将题名推断为具体实现细节。'
      '题目下中文文字仅概括题名聚焦的主题。\n'
      '未提供专利/软著名称、编号或受理/授权证明，因此未列数量或声称已申请、已授权。'
      '模型与实验资产来自本地README、代码配置、outputs汇总及本PPT前页。'
      'CMIG投稿与龋病分割为原稿第17页既有下一步计划，不作为已经完成的成果。')


# Expand the sparse next-step page into concrete, reviewable work packages.
s = old[16]
for sh in list(s.shapes)[2:]:
    remove(sh)
work_items = [
    ('01', '论文完善与期刊投稿',
     '补齐实验口径、消融解释与失败例分析；继续优化稿件，推进 CMIG 投稿。',
     '预期产出：完整稿件、可复现实验与补充材料。'),
    ('02', '编号一致性与泛化验证',
     '研究牙弓顺序与唯一性约束；按缺牙、混合牙列及采集条件分层评估。',
     '验证重点：重复标签率、图级一致性与检测 AP 之间的权衡。'),
    ('03', '推进牙位龋病分割',
     '衔接牙齿检测与龋病区域分割，梳理标注和评价方案，继续完成模型验证。',
     '预期产出：牙位关联的病灶分割流程与基线结果。'),
]
for i, (num, head, body, target) in enumerate(work_items):
    y = 1.45 + i*1.77
    rect(s, .55, y, 12.24, 1.48, 'FFFFFF')
    rect(s, .55, y, .73, 1.48, RED, None)
    text(s, .64, y+.51, .54, .42, num, 23, True, 'FFFFFF', align=PP_ALIGN.CENTER)
    text(s, 1.54, y+.14, 10.95, .37, head, 19.0, True, RED)
    text(s, 1.54, y+.65, 10.95, .34, body, 16.2)
    text(s, 1.54, y+1.09, 10.95, .27, target, 13.8, False, GRAY)
notes(s, '后续工作围绕三个方向推进。首先完善论文，把参考配置、种子数和受控比较讲清楚，继续推进CMIG投稿；'
      '其次围绕已暴露的重复编号问题研究全局约束，并检查缺牙、混合牙列及不同采集条件下的收益与副作用；'
      '第三继续推进牙位龋病分割，先把检测输出、病灶标注与评价口径衔接起来，再形成可比较的基线。'
      '以上均为后续安排，不表示相应实验或投稿已经完成。\n来源：原稿第16、17页及其备注。')


# Reorder all slides: +3, original cover and closing page preserved.
ordered = [old[0], topic, old[1], progress] + old[2:16] + [achievements, old[16], old[17]]
id_by_part = {prs.part.related_slide(e.rId).part: e for e in list(prs.slides._sldIdLst)}
ids = [id_by_part[s.part] for s in ordered]
for el in list(prs.slides._sldIdLst):
    prs.slides._sldIdLst.remove(el)
for el in ids:
    prs.slides._sldIdLst.append(el)

for i, s in enumerate(prs.slides, 1):
    if i in (1, len(prs.slides)):
        continue
    nums = [sh for sh in s.shapes if sh.name.startswith('page-')]
    if nums:
        sh = nums[0]
    else:
        sh = text(s, 12.34, 7.01, .52, .25, '', name='page-number')
    pos(sh, 12.34, 7.01, .52, .25)
    write(sh, f'{i:02d}', 11.25, False, GRAY, align=PP_ALIGN.RIGHT, after=0)
    if i <= 4:
        label = '研究主题与工作回顾'
    elif i <= 18:
        label = '现阶段进展 · 方法与实验'
    else:
        label = '成果情况与下一步工作'
    if i == 5:
        existing = s.shapes[1]
        write(existing, '现阶段进展 · 方法与实验｜总体框架示意', 9.5, False, GRAY, after=0)
    else:
        text(s, .48, 7.03, 9.0, .20, label, 9.5, False, GRAY, after=0,
             name='section-label')

# Update cover/closing remarks so that the talk no longer refers to deleted appendices.
notes(old[0], '本次汇报围绕四部分展开：研究主题、已有工作回顾、现阶段进展和目前成果。'
      '重点介绍三域模型配置、主要结果、受控消融与当前仍存在的编号一致性问题。'
      '论文成果中列入用户确认已录用或发表的CCF-C论文，最后说明期刊投稿与龋病分割计划。')
notes(old[17], '以上汇报了研究主题、前期工作、现阶段进展和论文成果。'
      '当前已完成三域模型和实验链路，主要开放问题是全局编号一致性与泛化验证。'
      '欢迎老师和同学就后续实验与论文完善提出建议。')

prs.core_properties.title = 'DENT-STRIDE 研究进展与成果汇报'
prs.core_properties.subject = '研究主题、已有工作回顾、现阶段进展、目前成果与下一步安排'
prs.core_properties.author = '吴熹'
prs.core_properties.comments = '在原18页PPT基础上补充完善，新增3页；保留原文件。'
prs.save(OUT)

# A compact machine-readable audit is kept with working assets.
audit = []
for i, s in enumerate(prs.slides, 1):
    title_text = next((sh.text for sh in s.shapes if sh.name == 'slide-title'), '')
    audit.append({'page': i, 'title': title_text,
                  'shape_count': len(s.shapes), 'notes': s.notes_slide.notes_text_frame.text})
(WORK/'revised/slide_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(OUT)
print('slides:', len(prs.slides))
