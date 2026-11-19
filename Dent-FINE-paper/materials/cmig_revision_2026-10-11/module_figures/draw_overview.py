#!/usr/bin/env python3
"""Acquisition-configured overview of the three evaluated final models."""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches
from draw_module_figures import Diagram, OUT, INK, BLUE, PALE, GRAY, WHITE


def overview():
    d=Diagram(1400,1030)
    d.panel(18,18,1364,324,'A  Multi-scale image-to-query processing')
    d.tensor(47,161,74,89,'Image')
    d.arrow([(135,205),(179,205)])
    d.rect(179,140,158,130,'HGNetv2-B0',fill=GRAY,stroke=INK,size=20)
    d.arrow([(337,205),(375,205),(375,119),(410,119)])
    d.arrow([(375,205),(410,205)])
    d.arrow([(375,205),(375,291),(410,291)])
    d.junction(375,205)
    d.text(453,77,'C = 256',17)
    for y,label in [(96,'S5'),(182,'S4'),(268,'S3')]:
        d.rect(410,y,84,46,'1×1',size=19)
        d.text(512,y+23,label,17)
    d.rect(552,72,392,252,fill=WHITE,stroke='#AAB6C0',lw=1)
    d.text(696,91,'Hybrid encoder',18,True)
    d.rect(565,108,141,44,'Transformer',fill=GRAY,stroke=INK,size=18)
    d.arrow([(525,119),(565,130)])
    d.text(755,166,'FPN',18,True);d.text(868,166,'PAN',18,True)
    # Each fusion site includes the base fusion block; CMFA precedes that block.
    for x in [710,823]:
        for y in [191,266]:
            d.rect(x,y,91,48,'Fusion',size=17)
    d.arrow([(706,130),(755,130),(755,191)])
    d.arrow([(525,205),(710,215)])
    d.arrow([(525,291),(710,290)])
    d.arrow([(755,239),(755,266)])
    d.arrow([(801,290),(823,290)])
    # P4 skip joins the lower PAN site from its top; downsampled P3 joins
    # its left edge. Separate entry edges prevent the skip being read as P3.
    d.arrow([(801,215),(813,215),(813,254),(846,254),(846,266)])
    d.arrow([(868,266),(868,239)])
    d.arrow([(706,130),(720,130),(720,108),(868,108),(868,191)])
    # CMFA is a conditional replacement at all four fusion sites, not a fifth stage.
    d.text(867,91,'CMFA ×4',16,color=BLUE)
    d.text(1040,91,'Mobile',16,color=BLUE)
    for y in [127,202,277]:
        d.rect(993,y,95,45,'HOOA',size=17)
    d.arrow([(914,215),(961,215),(961,150),(993,150)])
    d.arrow([(914,290),(972,290),(972,225),(993,225)])
    d.arrow([(801,303),(967,303),(993,300)])
    # The domain-specific bypass uses the same base feature outputs.
    d.arrow([(1088,150),(1111,150),(1111,203),(1139,203)])
    d.arrow([(1088,225),(1111,225),(1111,203)])
    d.arrow([(1088,300),(1111,300),(1111,225)])
    d.rect(1139,149,161,118,'D-FINE\ndecoder',fill=GRAY,stroke=INK,size=21)
    d.arrow([(1300,208),(1333,208)])
    d.tensor(1334,185,27,43,'q')
    d.text(1219,294,'Object queries',17)
    # B is the query-to-output stage, with the same q as panel A.
    d.panel(18,365,1364,487,'B  Detection, structured identity, and anatomical scoring')
    d.tensor(47,512,60,80,'q')
    d.arrow([(120,552),(168,552),(168,438),(229,438)])
    d.arrow([(168,552),(168,518),(229,518)])
    d.arrow([(168,552),(168,637),(229,637)])
    d.arrow([(168,637),(168,778),(229,778)],BLUE,True)
    d.junction(168,552);d.junction(168,637)
    d.rect(229,408,181,60,'FDI head',fill=GRAY,stroke=INK,size=20)
    d.rect(229,488,181,60,'Box head',fill=GRAY,stroke=INK,size=20)
    d.arrow([(410,438),(648,438)])
    d.text(496,414,'FDI logits',18)
    d.op(670,438,'⊕',22)
    d.arrow([(692,438),(780,438)])
    d.rect(780,403,178,70,'SpatialShape',size=19)
    d.text(869,387,'Pano',17,color=BLUE)
    d.arrow([(958,438),(997,438)])
    d.op(1019,438,'σ',22)
    d.arrow([(1041,438),(1071,438)])
    d.rect(1071,408,126,60,'Top 300',fill=GRAY,stroke=INK,size=19)
    d.arrow([(410,518),(868,518),(868,473)])
    d.junction(868,518)
    d.text(496,542,'Candidate boxes',18)
    d.arrow([(868,518),(1118,518),(1118,468)])
    d.rect(1239,408,106,60,'FDI map',fill=GRAY,stroke=INK,size=18)
    d.arrow([(1197,438),(1239,438)])
    d.arrow([(1292,468),(1292,586)])
    d.rect(1154,586,192,101,'Tooth boxes\nFDI codes',fill=WHITE,stroke=INK,size=20)
    d.arrow([(1164,468),(1164,586)])
    d.text(1250,720,'32 / 52 classes',17)
    d.rect(229,601,259,73,'Factor heads',size=21)
    d.text(359,698,'Quadrant · Position · Dentition',16)
    d.arrow([(488,637),(670,637),(670,460)])
    d.text(613,612,'Mobile',18,color=BLUE)
    d.arrow([(488,649),(544,649),(544,760),(643,760)],BLUE,True)
    d.rect(229,742,259,72,'Prototype bank\nK × 4',size=20)
    d.arrow([(488,778),(643,778)],BLUE,True)
    d.rect(643,733,153,77,'Auxiliary\nlosses',fill=GRAY,stroke=INK,size=20)
    d.rect(844,622,221,77,'Training-box\ncenter / shape',fill=GRAY,stroke=INK,size=19)
    d.arrow([(955,622),(955,565),(923,565),(923,473)],BLUE,True)
    d.text(987,580,'Fit priors',17,color=BLUE)
    d.text(719,833,'Mobile / Pano',17)
    d.text(955,727,'Pano',17)
    # C specifies switches; the schematic is a framework, not an all-module detector.
    for x,label,line1,line2 in [
      (18,'DENT-STRIDE-Pano','Factor + Prototype','SpatialShape'),
      (482,'DENT-STRIDE-Mobile','CMFA + HOOA','Factor + Prototype'),
      (946,'DENT-STRIDE-DSLR','CMFA','DualOpt (training)')]:
        d.rect(x,878,436,132,fill=WHITE,stroke='#CCD4DB',lw=.85)
        d.text(x+218,901,label,20,True)
        d.text(x+218,945,line1,19)
        d.text(x+218,981,line2,19)
    d.text(32,901,'C',20,True,align='left')
    return d


def main():
    d=overview().render('Figure_2_overview')
    p=Presentation();p.slide_width=Inches(12);p.slide_height=Inches(8.9)
    d.ppt(p);p.save(OUT/'Figure_2_overview_editable.pptx')
    info={
      'caption':'Acquisition-configured DENT-STRIDE framework. (A) Backbone features at strides 8, 16, and 32 are projected to 256 channels; the highest-stride feature passes through the transformer before top-down FPN and bottom-up PAN processing. Mobile and DSLR configurations use CMFA at two FPN and two PAN fusion sites; each site includes the corresponding base fusion block. HOOA processes the three output levels in Mobile only. (B) Decoder queries supply box and FDI heads. Mobile and Pano use factor supervision and a four-prototype-per-class auxiliary bank (K = 52 and 32, respectively). Mobile factor fusion uses weights 0.10, 0.05, and 0.03; factor fusion is disabled in Pano. Prototype learning supplies training supervision only. Pano center and shape statistics modify logits before sigmoid scoring and top-300 selection; labels are decoded through the cohort FDI map. (C) The three configurations share the backbone, base hybrid encoder, and decoder; DSLR uses DualOpt during training. Solid arrows show prediction data flow; dashed arrows indicate training supervision or anatomical-prior estimation. Modules are used only in the domains specified in the configuration cards, not simultaneously in a universal detector.',
      'source_checks':[
       'Frozen configs/mobile_1024_full/seed_0.yml: CMFA and HOOA enabled; factor heads and prototypes enabled; factor fusion weights 0.10/0.05/0.03; prototype fusion zero; AdamW.',
       'Frozen configs/pano_historical_spatialshape/seed_0.yml: CMFA/HOOA disabled; factor heads/prototypes enabled with factor fusion zero; center/shape prior weights 0.20/0.05; 32 categories; AdamW.',
       'Frozen configs/dslr_cmfa_dualopt/seed_0.yml: CMFA enabled; HOOA/factor/prototype/prior disabled; DualOptAdamW.',
       'hybrid_encoder.py forward: only encoder index 2 is transformed; two top-down and two bottom-up CMFA sites; HOOA after the pyramid outputs.',
       'dfine_decoder.py: normalized query/prototypes and LogSumExp cosine scoring; prototype_count=4 and prototype_fusion_weight=0.',
       'postprocessor.py: center/shape terms added to logits before sigmoid and flat top-300 selection.'
      ],
      'omitted_modules':['CEDv2','SMWG','MPF','UCM'],
      'font':'Times New Roman',
      'raster_dpi':600
    }
    (OUT/'Figure_2_overview_caption_and_checks.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
