#!/usr/bin/env python3
"""Reproducible, editable diagrams of the executed CMFA and HOOA operators."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

OUT = Path(__file__).resolve().parent
FONTS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman')
for path in FONTS.glob('*.TTF'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family':'Times New Roman', 'font.size':18,
                     'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.unicode_minus':False})
INK='#252C33'; BLUE='#456D8A'; PALE='#E8EFF4'; GRAY='#E9ECEF'; MID='#9AA6B0'; WHITE='#FFFFFF'

class Diagram:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.items=[]
    def text(self, x,y,text,size=18,bold=False,color=INK,align='center'):
        text=text.replace('⊙','×').replace('⊕','+').replace('AvgPool₃','AvgPool3')
        self.items.append(('text',dict(x=x,y=y,text=text,size=size,bold=bold,color=color,align=align)))
    def rect(self,x,y,w,h,label='',fill=PALE,stroke=BLUE,size=18,bold=False,lw=1.25):
        self.items.append(('rect',dict(x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,lw=lw)))
        if label: self.text(x+w/2,y+h/2,label,size,bold)
    def panel(self,x,y,w,h,label):
        self.rect(x,y,w,h,fill=WHITE,stroke='#CCD4DB',lw=.85)
        self.text(x+14,y+25,label,20,True,align='left')
    def arrow(self,points,color=INK,dashed=False,lw=1.35):
        self.items.append(('line',dict(points=points,color=color,dashed=dashed,lw=lw,arrow=True)))
    def line(self,points,color=INK,dashed=False,lw=1.35):
        self.items.append(('line',dict(points=points,color=color,dashed=dashed,lw=lw,arrow=False)))
    def op(self,x,y,symbol,r=20):
        self.items.append(('circle',dict(x=x,y=y,r=r,fill=WHITE,stroke=INK,lw=1.25)))
        self.text(x,y,symbol,23)
    def tensor(self,x,y,w=55,h=68,label='',channels=None):
        for off in [12,6,0]:
            self.rect(x+off,y-off,w,h,fill=PALE,stroke=BLUE,lw=.9)
        if label:self.text(x+w/2+6,y-30,label,20,True)
        if channels:self.text(x+w/2+6,y+h+28,channels,16)
    def vector(self,x,y,w=80,h=18,label=None):
        self.rect(x,y,w,h,fill=WHITE,stroke=BLUE,lw=.8)
        for i in range(8):
            fill=[PALE,GRAY,'#B5C9D8','#6C8DA6'][i%4]
            self.rect(x+i*w/8,y,w/8,h,fill=fill,stroke=WHITE,lw=.6)
        if label:self.text(x+w/2,y-21,label,20)
    def junction(self,x,y):
        self.items.append(('circle',dict(x=x,y=y,r=3,fill=INK,stroke=INK,lw=.5)))
    def render(self,stem):
        fig,ax=plt.subplots(figsize=(self.width/100,self.height/100))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.set(xlim=(0,self.width),ylim=(self.height,0)); ax.axis('off')
        for typ,k in self.items:
            if typ=='text':
                ax.text(k['x'],k['y'],k['text'],fontsize=k['size'],weight='bold' if k['bold'] else 'normal',
                        ha=k['align'],va='center',color=k['color'],linespacing=1.13)
            elif typ=='rect':
                ax.add_patch(Rectangle((k['x'],k['y']),k['w'],k['h'],facecolor=k['fill'],edgecolor=k['stroke'],lw=k['lw']))
            elif typ=='circle':
                ax.add_patch(Circle((k['x'],k['y']),k['r'],facecolor=k['fill'],edgecolor=k['stroke'],lw=k['lw']))
            elif typ=='line':
                points=k['points']; xs,ys=zip(*points)
                ax.plot(xs,ys,color=k['color'],lw=k['lw'],ls=(0,(4,3)) if k['dashed'] else '-',solid_capstyle='butt')
                if k['arrow']:
                    p,q=points[-2:]
                    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=16,lw=k['lw'],
                                                color=k['color'],shrinkA=0,shrinkB=0))
        for ext in ['svg','pdf','png']:
            fig.savefig(OUT/f'{stem}.{ext}',dpi=600,facecolor=WHITE)
        fig.savefig(OUT/f'{stem}_preview.png',dpi=145,facecolor=WHITE)
        plt.close(fig)
        return self
    def ppt(self,pres):
        from pptx.oxml.xmlchemy import OxmlElement
        slide=pres.slides.add_slide(pres.slide_layouts[6])
        scale=min(12/self.width,8.9/self.height)
        left=(12-self.width*scale)/2
        top=(8.9-self.height*scale)/2
        def xy(x,y):return Inches(left+x*scale),Inches(top+y*scale)
        def col(c):return RGBColor.from_string(c.strip('#'))
        for typ,k in self.items:
            if typ=='text':
                width=max(150,len(max(k['text'].split('\n'),key=len))*k['size']*.69)
                height=max(60,34*len(k['text'].split('\n')))
                x=k['x'] if k['align']=='left' else k['x']-width/2
                a,b=xy(x,k['y']-height/2)
                sh=slide.shapes.add_textbox(a,b,Inches(width*scale),Inches(height*scale))
                tf=sh.text_frame; tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
                tf.vertical_anchor=MSO_ANCHOR.MIDDLE
                for i,txt in enumerate(k['text'].split('\n')):
                    p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                    p.alignment=PP_ALIGN.LEFT if k['align']=='left' else PP_ALIGN.CENTER
                    r=p.add_run();r.text=txt;r.font.name='Times New Roman'
                    r.font.size=Pt(k['size']*scale*100)
                    r.font.bold=k['bold'];r.font.color.rgb=col(k['color'])
            elif typ in ['rect','circle']:
                if typ=='rect': x,y,w,h=k['x'],k['y'],k['w'],k['h'];shape=MSO_SHAPE.RECTANGLE
                else:x,y,w,h=k['x']-k['r'],k['y']-k['r'],2*k['r'],2*k['r'];shape=MSO_SHAPE.OVAL
                a,b=xy(x,y); sh=slide.shapes.add_shape(shape,a,b,Inches(w*scale),Inches(h*scale))
                sh.fill.solid();sh.fill.fore_color.rgb=col(k['fill']);sh.line.color.rgb=col(k['stroke']);sh.line.width=Pt(k['lw'])
                sh._element.spPr.append(OxmlElement('a:effectLst'))
            elif typ=='line':
                for i,(p,q) in enumerate(zip(k['points'],k['points'][1:])):
                    a,b=xy(*p);c,d=xy(*q)
                    sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,a,b,c,d)
                    sh.line.color.rgb=col(k['color']);sh.line.width=Pt(k['lw'])
                    sh._element.spPr.append(OxmlElement('a:effectLst'))
                    ln=sh.line._get_or_add_ln()
                    if k['dashed']:
                        dash=OxmlElement('a:prstDash');dash.set('val','dash');ln.append(dash)
                    if k['arrow'] and i==len(k['points'])-2:
                        end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','sm');end.set('len','sm');ln.append(end)
        return slide

def cmfa():
    d=Diagram(1200,680)
    d.panel(18,18,1164,176,'A  Channel gating')
    d.tensor(45,70,42,36); d.tensor(45,137,42,36)
    d.text(66,88,'A',17,True);d.text(66,155,'B',17,True)
    d.arrow([(100,88),(125,88),(125,115),(155,115)])
    d.arrow([(100,155),(125,155),(125,115)])
    d.rect(155,89,84,52,'Cat',GRAY,INK)
    for x,label,w in [(280,'GAP',90),(411,'1×1',90),(542,'SiLU',90),(673,'1×1',90)]:
        d.rect(x,89,w,52,label)
    d.arrow([(239,115),(280,115)]);d.arrow([(370,115),(411,115)])
    d.arrow([(501,115),(542,115)]);d.arrow([(632,115),(673,115)])
    d.op(812,115,'σ',21);d.arrow([(763,115),(791,115)])
    d.arrow([(833,115),(870,115),(870,86),(925,86)])
    d.arrow([(870,115),(870,146),(1040,146)])
    d.vector(925,76,88,20,'gA');d.vector(1040,136,88,20,'gB')
    d.text(980,170,'Split into C-channel gates',16)
    d.panel(18,216,790,444,'B  Exchange')
    d.tensor(45,296,56,68,'A','C = 256');d.tensor(45,510,56,68,'B','C = 256')
    d.arrow([(115,330),(284,330)]);d.arrow([(115,544),(284,544)])
    d.op(305,330,'⊙');d.op(305,544,'⊙')
    d.vector(265,264,80,18,'gA');d.arrow([(305,282),(305,310)],BLUE,True)
    d.vector(265,609,80,18);d.text(374,618,'gB',20)
    d.arrow([(305,609),(305,564)],BLUE,True)
    # Identity streams bypass their own gated streams.
    d.arrow([(175,330),(175,287),(587,287),(587,309)])
    d.arrow([(175,544),(175,637),(587,637),(587,565)])
    d.junction(175,330);d.junction(175,544)
    d.op(587,330,'⊕');d.op(587,544,'⊕')
    # Gated streams cross; the crossing is not a tensor junction.
    d.arrow([(325,330),(390,330),(518,505),(566,544)])
    d.arrow([(325,544),(390,544),(518,369),(566,330)])
    d.arrow([(607,330),(680,330),(680,415),(709,415)])
    d.arrow([(607,544),(680,544),(680,459),(709,459)])
    d.rect(709,400,81,75,'Cat',GRAY,INK)
    d.text(748,501,'Z',20,True);d.text(748,532,'2C = 512',16)
    d.panel(832,216,350,444,'C  Spatial gate')
    d.arrow([(790,437),(863,437),(1060,437)])
    d.junction(875,437)
    d.arrow([(875,437),(875,288),(902,288)])
    d.rect(902,260,144,55,'Mean / Max',size=17)
    d.arrow([(974,315),(974,340)])
    d.rect(902,340,144,55,'7×7 conv',size=17)
    d.arrow([(1046,368),(1080,368)])
    d.op(1100,368,'σ',20)
    d.arrow([(1100,388),(1100,416)],BLUE,True)
    d.text(1139,402,'Gs',18)
    d.op(1100,437,'⊙')
    d.arrow([(1120,437),(1141,437),(1141,508)])
    d.tensor(1113,520,43,50)
    d.text(1071,541,'Y',20,True);d.text(1042,609,'2C × H × W',16)
    return d

def hooa():
    d=Diagram(1200,910)
    d.panel(18,18,1164,535,'A  Multi-order feature construction')
    d.tensor(40,263,57,70,'X','C × H × W')
    d.arrow([(109,298),(156,298),(156,143),(214,143)])
    d.arrow([(156,298),(214,298)])
    d.arrow([(156,298),(156,456),(214,456)])
    d.junction(156,298)
    d.text(268,93,'Local',19,True)
    d.rect(214,112,130,62,'DW 3×3');d.rect(387,112,130,62,'PW 1×1')
    d.arrow([(344,143),(387,143)]);d.arrow([(517,143),(583,143)])
    d.text(601,143,'L',21,True)
    d.arrow([(620,143),(818,143),(818,256),(862,256)])
    d.arrow([(645,143),(645,278)],BLUE)
    d.junction(645,143)
    d.text(279,237,'Context',19,True)
    d.rect(214,267,130,62,'DW 7×7\nd = 3',size=17)
    d.rect(387,267,130,62,'PW 1×1')
    d.arrow([(344,298),(387,298)]);d.arrow([(517,298),(544,298)])
    d.op(565,298,'σ');d.arrow([(585,298),(625,298)])
    d.op(645,298,'⊙');d.arrow([(665,298),(862,298)])
    d.text(754,272,'L ⊙ σ(C)',18)
    d.text(289,399,'High-pass',19,True)
    d.rect(214,425,155,62,'X − AvgPool₃',size=17)
    d.rect(412,425,130,62,'DW 3×3');d.rect(585,425,130,62,'PW 1×1')
    d.arrow([(369,456),(412,456)]);d.arrow([(542,456),(585,456)])
    d.arrow([(715,456),(818,456),(818,340),(862,340)])
    d.text(753,433,'E',20,True)
    d.rect(862,237,91,123,'Cat',GRAY,INK)
    d.arrow([(953,298),(988,298)])
    d.rect(988,267,116,62,'PW 1×1',size=17)
    d.arrow([(1104,298),(1125,298)])
    d.tensor(1126,265,34,66,'F')
    d.text(436,522,'DW: depthwise convolution     PW: pointwise convolution',16)
    d.panel(18,577,1164,311,'B  Channel–spatial gating and bounded residual')
    d.tensor(45,704,52,60,'F')
    d.arrow([(109,734),(628,734)])
    d.junction(150,734)
    d.arrow([(150,734),(150,652),(194,652)])
    d.rect(194,626,93,52,'GAP',size=17)
    d.rect(327,626,93,52,'1×1',size=17)
    d.rect(460,626,93,52,'SiLU',size=17)
    d.rect(593,626,93,52,'1×1',size=17)
    d.arrow([(287,652),(327,652)]);d.arrow([(420,652),(460,652)])
    d.arrow([(553,652),(593,652)]);d.arrow([(686,652),(714,652)])
    d.op(734,652,'σ');d.arrow([(754,652),(781,652),(781,692),(648,692),(648,714)],BLUE,True)
    d.text(831,652,'Gc',19)
    d.arrow([(150,734),(150,822),(194,822)])
    d.rect(194,796,180,52,'Mean / Max',size=17)
    d.rect(415,796,146,52,'7×7 conv',size=17)
    d.arrow([(374,822),(415,822)]);d.arrow([(561,822),(590,822)])
    d.op(611,822,'σ');d.arrow([(631,822),(648,822),(648,754)],BLUE,True)
    d.text(688,822,'Gs',19)
    d.op(648,734,'⊙');d.arrow([(668,734),(793,734)])
    d.rect(793,703,173,62,'α = 0.30 σ(s)',size=18)
    d.arrow([(966,734),(1016,734)])
    d.op(1036,734,'⊕')
    d.text(1036,639,'X',21,True)
    d.arrow([(1036,666),(1036,714)])
    d.arrow([(1056,734),(1108,734)])
    d.tensor(1109,704,48,60,'Y')
    d.text(1065,814,'C × H × W',16)
    return d

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    one=cmfa().render('Figure_4_CMFA')
    two=hooa().render('Figure_5_HOOA')
    pres=Presentation();pres.slide_width=Inches(12);pres.slide_height=Inches(8.9)
    one.ppt(pres);two.ppt(pres);pres.save(OUT/'Figures_4_5_editable.pptx')
    record={
      'font':'Times New Roman (licensed local font files; embedded in PDFs)',
      'raster_dpi':600,
      'source_file':'/home/wuxi/Dent-FINE-paper/materials/cmig_best3_seeds_2026-10-08/source_snapshots/mobile/engine/deim/hybrid_encoder.py',
      'CMFA_source_lines':'3585–3621',
      'HOOA_source_lines':'1232–1289',
      'references':[
        {'title':'Efficient multi-stage feedback attention for diverse lesion in cancer image segmentation',
         'doi':'https://doi.org/10.1016/j.compmedimag.2024.102417','figure':'Fig. 3',
         'traits':'Parallel operator pathways, tensor glyphs, multiplication/addition nodes, short convolution labels'},
        {'title':'CRAD: Cognitive Aware Feature Refinement with Missing Modalities for Early Alzheimer’s Progression Prediction',
         'doi':'https://doi.org/10.1016/j.compmedimag.2025.102664',
         'author_pdf':'https://oars.uos.ac.uk/5227/1/CRAD_published_paper.pdf','figure':'Fig. 1',
         'traits':'Aligned branches, compact tensor stacks, pale fills, module-level grouping and serif typography'}],
      'captions':{
       'Figure 4':'Cross-scale mutual feature attention (CMFA). Spatially aligned inputs A and B have equal projected width C = 256. (A) Their concatenation supplies a global-average-pooling and two-layer pointwise channel gate, which is split into g_A and g_B. (B) The gated inputs are exchanged through A + B ⊙ g_B and B + A ⊙ g_A, then concatenated. (C) Channel mean and maximum statistics of the concatenated feature produce a 7 × 7 sigmoid spatial gate. The output retains 2C = 512 channels. Cat denotes channel concatenation; circular multiplication and addition nodes operate elementwise.',
       'Figure 5':'High-order omni-attention (HOOA). (A) Local and dilated-context branches generate L and C; the high-pass input X − AvgPool_3(X) supplies an edge branch E. The second-order response L ⊙ sigmoid(C) is concatenated with L and E and projected to F. (B) Channel and spatial gates are both computed from F before gating. Their product modulates F, and the update is added to X with bounded scale α = 0.30 sigmoid(s). DW and PW denote depthwise and pointwise convolutions. Local and edge convolutions include batch normalization and SiLU; the context branch uses SiLU after its depthwise convolution and batch normalization after its final pointwise convolution. Spatial statistics are channel-wise mean and maximum. Tensor glyphs illustrate feature shape, not measured response maps.'}
    }
    (OUT/'figure_design_and_caption.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'files':[str(p) for p in sorted(OUT.glob('Figure*'))],'captions':record['captions']},indent=2))

if __name__=='__main__': main()
