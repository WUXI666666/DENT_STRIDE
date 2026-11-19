from pathlib import Path
import json

ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-11')
def read(p):return json.loads(Path(p).read_text())
data={}
overview=ROOT/'module_figures/Figure_2_overview.png'
if overview.is_file():data['2']=str(overview)
for number,name in [('6','Figure_6_three_seed_AP'),('7','Figure_7_Mobile_framework_AP'),
                    ('9','Figure_9_Pano_prior_comparison'),('11','Figure_11_HOOA_Factor_learning_effects'),
                    ('S6','Figure_S6_component_sensitivity'),('S12','Figure_S12_prior_paired_AP_intervals'),
                    ('S5','Figure_S5_learning_curves')]:
    data[number]=str(ROOT/'quantitative_figures'/(name+'.png'))
for number,name in [('4','Figure_4_CMFA'),('5','Figure_5_HOOA')]:
    data[number]=str(ROOT/'module_figures'/(name+'.png'))
    data[number+'_caption']='Fig. '+number+'. '+read(ROOT/'module_figures/figure_design_and_caption.json')['captions']['Figure '+number]
data['4_caption']=data['4_caption'].replace('Cross-scale mutual feature attention','Cross-scale mutual feature aggregation')
data['4_caption']=data['4_caption'].replace('g_A','gA').replace('g_B','gB')
data['5_caption']=data['5_caption'].replace('AvgPool_3','AvgPool₃')
caps=read(ROOT/'prediction_figures/figure_captions.json')
for number,name in [('8','Figure_8_HOOA_same_image'),('10','Figure_10_Pano_same_weight_prior'),
                    ('12A','Figure_12A_Mobile_same_image'),('12B','Figure_12B_Pano_same_image'),
                    ('12C','Figure_12C_DSLR_same_image')]:
    data[number]=str(ROOT/'prediction_figures'/(name+'.png'))
    data[number+'_caption']=caps[name].replace('Same-image pano detector','Same-image panoramic detector').replace('Same-image dslr detector','Same-image DSLR detector')
for number,path in data.items():
    if not number.endswith('_caption'):assert Path(path).is_file(),path
(ROOT/'asset_index.json').write_text(json.dumps(data,indent=2))
