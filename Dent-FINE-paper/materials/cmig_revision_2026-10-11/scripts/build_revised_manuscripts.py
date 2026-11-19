"""Create scholarly manuscript copies bound to the completed three-seed results.

Original manuscripts and experiment outputs are never modified. Mathematical
OMML equations, affiliations/declarations, and references are preserved.
"""
from pathlib import Path
from copy import deepcopy
import json
import re
import zipfile
import hashlib
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PAPER=Path('/home/wuxi/Dent-FINE-paper')
ROOT=PAPER/'materials/cmig_revision_2026-10-11'
DATA=PAPER/'materials/cmig_best3_seeds_2026-10-08/manuscript_data'
TRAIN=DATA.parent
OLD=PAPER/'materials/cmig_final_2026-10-06'
MAIN_SOURCE=PAPER/'DENT_STRIDE_CMIG_main_final_20261006.docx'
SUPP_SOURCE=PAPER/'DENT_STRIDE_CMIG_supplement_final_20261006.docx'
MAIN_OUT=PAPER/'DENT_STRIDE_CMIG_main_updated_20261011.docx'
SUPP_OUT=PAPER/'DENT_STRIDE_CMIG_supplement_updated_20261011.docx'
ROOT.mkdir(parents=True,exist_ok=True)

def read(p):return json.loads(Path(p).read_text())
def fmt(x):return f'{float(x):.3f}'
def pm(r,k):return f"{r[k+'_mean']:.3f} ± {r[k+'_SD']:.3f}"
def signed(x):return f'{x:+.3f}'.replace('-', '−')
def source_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

SUMMARY={r['cohort']:r for r in read(DATA/'Table3_primary_three_seed.json')}
SEEDS=read(DATA/'primary_seed_results.json')
PRIORS=read(DATA/'Table6_SpatialShape_three_seed.json')
INTERNAL=read(DATA/'Table7_internal_seed0.json')
FUSION=read(DATA/'factor_fusion_2x2_seed0.json')
PRIOR_SEEDS=read(DATA/'spatial_shape_per_seed.json')
COHORTS=['Pano','Mobile','DSLR']
NAMES={'Pano':'DENT-STRIDE-Pano','Mobile':'DENT-STRIDE-Mobile','DSLR':'DENT-STRIDE-DSLR'}
CONFIGS={'Pano':'pano_historical_spatialshape','Mobile':'mobile_1024_full','DSLR':'dslr_cmfa_dualopt'}
PRIOR_LABELS={'none':'None','center':'Center only','shape':'Shape only','both':'Center + shape'}
MANIFEST={'source_documents':{str(p):source_sha(p) for p in [MAIN_SOURCE,SUPP_SOURCE]},
          'numerical_sources':str(DATA),'tables':[],'figures':[]}

def font(run,size=11,bold=None):
    run.font.name='Times New Roman';run.font.size=Pt(size)
    run.font.color.rgb=RGBColor(0,0,0)
    if bold is not None:run.font.bold=bold
    pr=run._r.get_or_add_rPr();f=pr.find(qn('w:rFonts'))
    if f is None:f=OxmlElement('w:rFonts');pr.insert(0,f)
    for key in ['ascii','hAnsi','eastAsia','cs']:f.set(qn('w:'+key),'Times New Roman')

def set_text(p,text):
    p.clear();r=p.add_run(text);font(r,10 if p.style.name=='Caption' else 11)
    if p.style.name=='Caption':p.paragraph_format.keep_together=True

class Writer:
    def __init__(self,doc,anchor=None):self.doc,self.anchor=doc,anchor
    def place(self,element):
        if self.anchor is not None:self.anchor.addprevious(element)
    def p(self,text='',style='Normal'):
        p=self.doc.add_paragraph(style=style)
        if text:font(p.add_run(text),10 if style=='Caption' else 11,style.startswith('Heading'))
        p.paragraph_format.space_after=Pt(7 if style=='Caption' else 6)
        p.paragraph_format.widow_control=True
        if style=='Caption':p.paragraph_format.keep_together=True
        if style.startswith('Heading'):p.paragraph_format.keep_with_next=True
        self.place(p._element);return p
    def heading(self,text,level=2):return self.p(text,'Heading '+str(level))
    def table(self,caption,headers,rows,widths=None,size=9,numeric_from=1):
        cap=self.p(caption,'Caption');cap.paragraph_format.keep_with_next=True
        t=self.doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        if widths is None:widths=[6.45/len(headers)]*len(headers)
        widths=[v*6.45/sum(widths) for v in widths]
        for col,w in zip(t.columns,widths):col.width=Inches(w)
        for c,text in zip(t.rows[0].cells,headers):c.text=str(text)
        for row in rows:
            for c,text in zip(t.add_row().cells,row):c.text=str(text)
        for ri,row in enumerate(t.rows):
            pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
            if ri==0:pr.append(OxmlElement('w:tblHeader'))
            for j,c in enumerate(row.cells):
                c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in c.paragraphs:
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j<numeric_from else WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(2)
                    p.paragraph_format.line_spacing=1.05
                    for r in p.runs:font(r,size,ri==0)
                cp=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
                for side in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+side);e.set(qn('w:val'),'nil');b.append(e)
                for side in (['top','bottom'] if ri==0 else ['bottom'] if ri==len(t.rows)-1 else []):
                    e=b.find(qn('w:'+side));e.set(qn('w:val'),'single');e.set(qn('w:sz'),'6');e.set(qn('w:color'),'000000')
                cp.append(b)
        self.place(t._tbl)
        MANIFEST['tables'].append({'caption':caption,'headers':headers,'rows':rows})
        return t
    def figure(self,number,path,caption):
        path=Path(path);assert path.is_file(),path
        p=self.p();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(path),width=Inches(6.45))
        self.p('Fig. '+str(number)+'. '+caption,'Caption')
        MANIFEST['figures'].append({'number':str(number),'path':str(path),'sha256':source_sha(path),'caption':caption})

def replace_picture(p,path):
    p.clear();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(6.45))

def remove_unreferenced_images(doc):
    """Do not retain obsolete, invisible figure payloads from the source package."""
    used=set()
    namespace='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    for node in doc._element.iter():
        for key,value in node.attrib.items():
            if key.startswith('{'+namespace+'}'):used.add(value)
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    for rid,rel in list(doc.part.rels.items()):
        if rel.reltype==RT.IMAGE and rid not in used:doc.part.drop_rel(rid)

def get_assets():
    """All figure paths and exact captions are bound by a separate local index."""
    return read(ROOT/'asset_index.json')

def insert_main_experiments(doc,original,assets):
    start,stop=original[77]._element,original[162]._element
    node=start
    while node is not stop:
        following=node.getnext();node.getparent().remove(node);node=following
    w=Writer(doc,stop)
    w.heading('4.2. Implementation and experimental design')
    w.p('The three acquisition-specific configurations were each trained with seeds 0, 1, and 2 for 160 epochs. The image partitions, clinical FDI mapping, model structure, initialization source, augmentation policy, losses, and checkpoint-selection rule were held fixed within each configuration. The HGNetv2-B0 backbone was initialized with ImageNet-pretrained weights. In addition, 21 component-ablation models were trained with seed 0, changing only the designated component relative to its group reference. Pure inference switches reuse the corresponding trained weights.')
    w.p('Training used automatic mixed precision, training and validation batch sizes of 8 and 16, gradient clipping at 0.10, and exponential moving averages with decay 0.9999. AdamW used learning rates of 5 × 10⁻⁴ for the detector and 2.5 × 10⁻⁴ for the backbone, betas (0.9, 0.999), and base weight decay 10⁻⁴; normalization parameters were excluded from base decay. The warm-up, flat, cosine-decay, and terminal low-learning-rate phases were retained. Geometric flips were disabled. The checkpoint with maximum validation COCO AP under each configured inference policy was selected; earlier checkpoints were retained in ties. DENT-STRIDE-Pano was selected with center and shape scoring enabled, and its subsequent prior ablation does not reselect weights.')
    w.table('Table 2. Acquisition-specific configurations and component roles. Loss coefficients and augmentation probabilities refer to the evaluated models.',
       ['Setting','Pano','Mobile','DSLR'],[
       ['Training/evaluation input','Fixed 640 × 640','Fixed 1024 × 1024','Multiscale training; 640 × 640 evaluation'],
       ['Feature processing','Base hybrid encoder','CMFA + HOOA','CMFA'],
       ['Factor heads','Auxiliary supervision','Supervision and score fusion','Inactive'],
       ['Prototype bank','Auxiliary supervision','Auxiliary supervision','Inactive'],
       ['SpatialShape prior','Center + shape','Inactive','Inactive'],
       ['Optimizer','AdamW','AdamW','DualOptAdamW'],
       ['Mosaic / MixUp probabilities','0 / 0.50','0 / 0','0 / 0'],
       ['IoU crop / zoom-out probabilities','0.80 / 0.50','0.80 / 0.50','0.80 / 0.50'],
       ['Arch-position loss weight','0.25','0.10','0.25'],
       ['NWD loss weight','Inactive','0.50; scale 0.08','Inactive'],
       ['Prototype loss weight','0.05','0.03','Inactive']],widths=[1.45,1.60,1.70,1.70],size=9)
    w.p('Within-domain component ablations preserve the full configuration’s resolution, data, backbone initialization, augmentation, background losses, training budget, evaluator, and selection policy, except for the stated change. CMFA gate switches preserve output dimensions; HOOA branch switches preserve local/context processing and the output projection. Factor-supervision comparisons disable factor fusion during both training and inference. The main configurations are not replaced by a single-seed ablation variant.')
    w.heading('4.3. Evaluation endpoints and statistical analysis')
    w.p('The primary endpoint is FDI-labelled COCO bounding-box AP over IoU thresholds 0.50:0.05:0.95. AP50, AP75, and AR100 are secondary endpoints. Classes without validation ground truth are excluded from category averages. DEIM-family models retain the native top-300 predictions without added non-maximum suppression or an FDI-uniqueness constraint. Selected exponential-moving-average checkpoints are evaluated in FP32. Parameter counts include instantiated auxiliary heads.')
    w.p('At score ≥0.30, class-agnostic one-to-one Hungarian assignment maximizes total localization IoU; only pairs with IoU ≥0.50 are retained as localization matches. A correct detection additionally requires exact FDI agreement. Precision uses retained predictions, recall uses reference teeth, matched-tooth FDI accuracy uses localization matches, and localization coverage uses reference teeth as denominators. F1 is computed from pooled TP, FP, and FN within each validation cohort. Raw-query duplicate-image rate is the fraction of images containing a repeated FDI label among thresholded native outputs; it is not a clinical charting-error rate.')
    w.p('Training variability is reported as mean ± sample standard deviation over seeds 0, 1, and 2. Same-checkpoint prior effects are paired within each seed. Their exploratory two-sided Student-t 95% intervals use the three AP differences and two degrees of freedom, without multiplicity adjustment; they describe training-seed variability on a fixed validation cohort rather than patient-level sampling uncertainty. Component-learning ablations use one seed and have no uncertainty bars. Cross-framework values are descriptive system comparisons, not paired architecture effects.')
    w.heading('4.4. Three-seed performance of the acquisition-specific configurations')
    w.p('DENT-STRIDE-Pano, DENT-STRIDE-Mobile, and DENT-STRIDE-DSLR achieved mean AP of 53.897 ± 0.430, 55.219 ± 0.327, and 72.293 ± 0.651, respectively (Table 3; Fig. 6). Pano includes the combined center–shape prior, while Mobile uses structured identity supervision and factor-score fusion. The corresponding AP75 values were 54.436 ± 0.925, 61.440 ± 1.336, and 83.000 ± 0.435. All three configurations were evaluated over the same three seeds, rather than reporting only their highest-performing run.')
    w.table('Table 3. Acquisition-specific DENT-STRIDE performance, mean ± sample SD over seeds 0, 1, and 2. Detection metrics are percentages; parameters are millions.',
       ['Cohort','Configuration','AP','AP50','AP75','AR100','Params (M)'],
       [[c,NAMES[c],*[pm(SUMMARY[c],k) for k in ['AP','AP50','AP75','AR100']],fmt(SUMMARY[c]['params_M'])] for c in COHORTS],
       widths=[.6,1.35,.9,.9,.9,.9,.65],size=8.5,numeric_from=2)
    w.figure(6,assets['6'],'Three-seed performance of the acquisition-specific configurations. Small circles show the actual outcomes for seeds 0, 1, and 2; larger markers and bars show mean ± sample SD. Each cohort has its own labelled AP axis. Pano includes combined center–shape scoring. No between-model improvement or statistical superiority is inferred from these within-configuration error bars.')
    w.heading('4.5. Training-seed variability')
    w.p('AP ranged from 53.521 to 54.365 for Pano, 54.910 to 55.562 for Mobile, and 71.548 to 72.750 for DSLR (Table 4). These ranges and standard deviations characterize repeatability under fixed domain configurations. Individual-run secondary metrics and checkpoint epochs are provided in Supplementary Table S1; learning trajectories are shown in Supplementary Fig. S8.')
    w.table('Table 4. Individual-seed AP and variability of the three final configurations. Values are percentage points; SD is the sample standard deviation.',
       ['Configuration','Seed 0','Seed 1','Seed 2','Mean ± SD','Range'],
       [[NAMES[c],*[fmt(next(r for r in SEEDS if r['cohort']==c and r['seed']==s)['AP']) for s in [0,1,2]],pm(SUMMARY[c],'AP'),fmt(max(r['AP'] for r in SEEDS if r['cohort']==c)-min(r['AP'] for r in SEEDS if r['cohort']==c))] for c in COHORTS],
       widths=[1.75,.72,.72,.72,1.12,.65])
    w.heading('4.6. Framework-level detector comparisons')
    w.p('Table 5 compares the acquisition-specific configurations with official DEIM-s, RT-DETRv2-S, and conventional detector implementations on the same cohort annotations and clinical label maps. Official DEIM-s and RT-DETRv2-S entries summarize three separately trained seeds; other external rows are single-seed results. The frameworks retain their native pretraining, optimization, augmentation, and post-processing. Mobile DENT-STRIDE uses 1024 × 1024 inputs, whereas the external Mobile references use 640 × 640. Consequently, these comparisons evaluate complete detector systems and do not isolate the effect of a module or image resolution.')
    original_table=Document(MAIN_SOURCE).tables[4]
    rows=[]
    for row in original_table.rows[1:]:
        values=[c.text for c in row.cells]
        if values[0]=='Controlled reference':continue
        if values[0]=='DENT-STRIDE':values=['DENT-STRIDE','3',*[pm(SUMMARY[c],'AP') for c in COHORTS]]
        rows.append(values)
    w.table('Table 5. Framework-level validation AP comparison. Three-seed values are mean ± sample SD; single-seed rows have no dispersion estimate. Framework-specific input and training settings are retained.',
       ['Method','Seeds','Pano AP','Mobile AP','DSLR AP'],rows,widths=[2.1,.5,1.25,1.25,1.25])
    w.p('DENT-STRIDE-Mobile has the highest mean AP among the listed Mobile systems, at 55.219, compared with 54.751 for the single-seed RT-DETR-L reference (Fig. 7). Pano mean AP is 53.897, compared with 53.827 for official DEIM-s under its separate training protocol. On DSLR images, RT-DETR-L reaches 74.019 AP and official DEIM-s reaches 72.405 ± 0.202, both above the 72.293 ± 0.651 DENT-STRIDE mean. These descriptive rankings do not establish statistical superiority or a universal domain-independent optimum.')
    w.figure(7,assets['7'],'Mobile framework-level AP comparison. Labels identify each detector and its AP. Error bars show sample SD only for the three-seed DENT-STRIDE-Mobile, official DEIM-s, and RT-DETRv2-S entries; other methods are single-seed observations. Input resolution, pretraining, and native training/output settings differ across frameworks, as specified in Section 4.6. The plot describes system performance rather than an isolated identity-module effect.')
    w.heading('4.7. Branch-level evidence for structured identity and local refinement')
    w.p('The HOOA branch ablation supports complementary local-detail and context interaction within the Mobile configuration. Starting from 54.910 AP, removing the high-pass branch, second-order interaction, or both reduced AP to 54.511, 53.943, and 53.561, respectively. With factor fusion disabled in both phases, removing quadrant, tooth-position, or dentition supervision reduced AP from 55.537 to 54.884, 54.263, and 55.293. These retrained seed-0 contrasts support the tested branches under a common training protocol; they do not quantify repeated-seed module effects. Fig. 8 provides same-image examples, and Table 7 reports the aggregate comparisons.')
    w.figure(8,assets['8'],assets['8_caption'])
    w.heading('4.8. Same-checkpoint anatomical-prior comparison')
    w.p('Each Pano checkpoint selected under combined center–shape scoring was evaluated with no prior, center only, shape only, or both, without reselection of weights (Table 6; Fig. 9). All 300 ordinary-query boxes and their full class logits were retained before top-k selection. Factor fusion, appearance scoring, and global assignment were disabled. The panoramic view term is class-constant and has no numerical effect. The comparison changes class-score ranking, not predicted box coordinates.')
    w.table('Table 6. Anatomical scoring at the same Pano checkpoints. Values are mean ± sample SD across three seeds. ΔAP and its exploratory 95% interval are paired with no prior; checkpoint selection used the combined prior.',
       ['Prior','AP','AP50','AP75','AR100','ΔAP','ΔAP 95% CI'],
       [[PRIOR_LABELS[r['prior']],*[pm(r,k) for k in ['AP','AP50','AP75','AR100']],signed(r['AP_delta_vs_none_mean']) if r['prior']!='none' else '—',f"[{r['AP_delta_vs_none_CI95_low']:.3f}, {r['AP_delta_vs_none_CI95_high']:.3f}]" if r['prior']!='none' else '—'] for r in PRIORS],
       widths=[1.03,.95,.95,.95,.95,.57,1.10],size=8.5)
    w.p('Center scoring increased mean AP from 53.413 to 53.906 (+0.494; 95% interval [0.163, 0.824]); combined scoring reached 53.897 (+0.484; [0.230, 0.739]). Shape-only scoring changed AP by −0.008 ([−0.076, 0.059]). Center-only and combined scoring increased AR100 to 69.249 and 69.387, respectively, from 66.893 without a prior. Center scoring had the highest mean AP, whereas combined scoring had the highest AP75 and AR100. Fig. 10 illustrates a favourable and an adverse identity change from the same checkpoint. These post-selection comparisons isolate inference-score effects, not learned appearance gains.')
    w.figure(9,assets['9'],'Four-way anatomical scoring at identical Pano weights. (A) AP and (B) AR100. Small circles show the actual outcomes of the three seed-specific checkpoints under no prior, center, shape, and combined scoring. Larger markers and bars show mean ± sample SD. All scores are recomputed before native top-k selection; weights and predicted box coordinates are unchanged.')
    w.figure(10,assets['10'],assets['10_caption'])
    w.heading('4.9. Component and optimization sensitivity')
    core=[r for r in INTERNAL if r['component'] in ['CMFA gates','HOOA branches','Factor supervision']]
    w.table('Table 7. Feature-gating, HOOA-branch, and factor-supervision ablations at seed 0. ΔAP subtracts the stated group reference. Factor-supervision models disable training and inference fusion. All settings are separately trained except reused intact references.',
       ['Component','Setting','AP','AP50','AP75','AR100','ΔAP'],
       [[r['component'],r['setting'],*[fmt(r[k]) for k in ['AP','AP50','AP75','AR100']],signed(r['delta_AP_vs_reference'])] for r in core],
       widths=[1.05,1.55,.63,.63,.63,.63,.63],size=8.5,numeric_from=2)
    w.p('Fig. 11 summarizes the HOOA and factor-supervision contrasts. CMFA channel-only gating reached 55.422 AP, compared with 54.910 for both gates, indicating that more gates are not uniformly beneficial. Prototype-number/temperature, optimization, background-loss, and factor-fusion sensitivity results are provided in Supplementary Tables S2 and S5. They are treated as configuration-dependent findings rather than separate contributions or substitutes for the three-seed models.')
    w.figure(11,assets['11'],'Retrained HOOA and factor-supervision ablations. Bars report AP differences after each branch or supervision-target removal. HOOA uses the full Mobile seed-0 configuration (54.910 AP); factor supervision uses all three targets with training and inference fusion disabled (55.537 AP). Every tested removal is shown. These single-seed results have no uncertainty bars.')
    w.heading('4.10. Tooth identity, detection, and numbering consistency')
    w.table('Table 8. Fixed-threshold validation endpoints, mean ± sample SD over three seeds, at score ≥0.30 and localization IoU ≥0.50. All values are percentages. FDI accuracy is conditional on localization matches; recall and coverage use all reference teeth; duplicates use all images.',
       ['Cohort','Precision','Recall','F1','Matched FDI','Coverage','Duplicate images'],
       [[c,*[pm(SUMMARY[c],k) for k in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']]] for c in COHORTS],
       widths=[.56,.95,.95,.95,.95,.95,1.15],size=8.5)
    w.p('The Pano, Mobile, and DSLR configurations achieved mean F1 of 88.877%, 81.461%, and 91.869%, with matched-tooth FDI accuracy of 92.883%, 92.083%, and 97.164%, respectively. Their raw-query duplicate-image rates were 54.688%, 32.879%, and 27.018%. High conditional tooth-identity accuracy therefore coexists with repeated labels at image level. These endpoints distinguish detection correctness from a globally constrained dental chart.')
    w.p('Fig. 12 shows the same images across official DEIM-s, an external detector, and the seed-0 domain configuration, with consistent display thresholds and shared local enlargements. The examples illustrate corrections and failures; aggregate performance uses every validation image. Detection PR, threshold–F1, threshold–duplicate curves, complete FDI outcome matrices, and class support are provided in Supplementary Figs. S4–S7.')
    for letter in ['A','B','C']:w.figure('12'+letter,assets['12'+letter],assets['12'+letter+'_caption'])
    w.heading('4.11. Internal feature responses and computational cost')
    w.p('Fig. 13 displays genuine CMFA and HOOA tensors from the Mobile seed-0 forward pass for the improvement example in Fig. 8. CMFA gA and gB are channel gates, not anatomical spatial maps. HOOA local, context, high-pass, and second-order responses summarize channel-wise root-mean-square magnitude. Spatial and channel gates retain their [0,1] scale, while response maps are normalized independently for display. These visualizations explain executed operations; the retrained comparisons in Table 7, rather than heatmap appearance, support their performance effects.')
    w.figure(13,ROOT/'figures/mechanism/actual_CMFA_HOOA_image_155.png','Actual CMFA/HOOA forward-pass responses for the same Mobile image shown in Fig. 8A. CMFA channel gates gA/gB are arranged as 16 × 16 channel-index grids; Gs is the spatial gate. HOOA panels show local, context, high-pass, second-order, channel/spatial gates, input, and output. Sigmoid gates use their actual [0,1] values. Response RMS maps are divided by their respective 99th percentile and clipped to [0,1]. These are native internal tensors, not Grad-CAM or causal evidence from visualization alone.')
    complexity=[]
    for c in COHORTS:
        r=read(ROOT/'model_measurements/complexity'/(CONFIGS[c]+'.json'))
        complexity.append([c,NAMES[c],str(r['source_inference_resolution'][0])+' × '+str(r['source_inference_resolution'][1]),fmt(r['parameters_total']/1e6),fmt(r['GMACs']),fmt(r['GFLOPs'])])
    w.table('Table 9. Model size and tool-counted native-forward arithmetic of the final configurations. Measurements use calflops 0.3.2, batch 1, FP32, and evaluation mode; auxiliary outputs are included. Post-processing and unsupported operations are excluded.',
       ['Cohort','Configuration','Evaluation input','Params (M)','GMACs','GFLOPs'],complexity,widths=[.6,1.65,1.15,.9,.85,.85],numeric_from=2)
    w.p('The Pano, Mobile, and DSLR native forwards require approximately 12.587, 38.938, and 12.633 tool-counted GMACs, respectively, at their stated evaluation resolutions. Parameters include instantiated auxiliary heads. Prototype tensor contractions are counted in the profiler’s FLOP total but not its MAC total; consequently, FLOPs are not exactly twice MACs. These counts exclude post-processing and uninstrumented operations, including deformable-attention grid sampling, and are not latency or memory benchmarks.')
    w.heading('5. Discussion',1)
    w.heading('5.1. Principal findings')
    w.p('DENT-STRIDE separates query-level identity learning, feature refinement, and anatomical class-score correction within an acquisition-configured detector. Three-seed results quantify repeatability of the complete configurations, while branch ablations and same-checkpoint scoring comparisons identify specific mechanisms. Removing high-pass/second-order HOOA responses or individual factor targets reduced Mobile AP in the seed-0 comparisons. Center-based scoring improved Pano AP for all three checkpoints; shape-only scoring did not provide a corresponding mean AP gain. The evidence therefore supports these specific mechanisms rather than the premise that every additional module improves detection.')
    w.p('Earlier label-tree and prior-knowledge systems already encode tooth identity and anatomy (Zhang et al., 2018; Ali et al., 2023). DENT-STRIDE places factor-score fusion and training-derived geometry within a common query detector, with their training and inference roles stated separately. Center and shape templates do not change the learned query representations or predicted box coordinates. Prototype supervision and optimization choices are retained as precisely defined configuration components, without attributing an independent accuracy benefit that is not supported by their ablations.')
    w.heading('5.2. Acquisition dependence and clinical interpretation')
    w.p('Panoramic framing provides a stable reference for the training-derived center templates in Fig. 3. In photographic images, local appearance, occlusion, similar neighbouring teeth, and mixed dentition motivate structured identity targets and multiscale feature processing. Domain-specific input, augmentation, background losses, and inference settings form part of the evaluated system, rather than interchangeable implementation details. The Mobile framework comparison therefore cannot attribute its AP difference exclusively to identity learning.')
    w.p('Conditional FDI accuracy does not ensure globally unique numbering. The duplicate-image endpoint and selected failure examples expose the remaining separation between tooth-level detection and chart consistency. Clinical interpretation also requires accounting for missing teeth, overlapping views, mixed dentition, and uncertainty. The reported detector endpoints do not measure the accuracy or utility of a complete clinical charting workflow.')
    w.heading('5.3. Limitations')
    w.p('The three-seed estimates describe training variability on validation partitions used for checkpoint selection, not independent-test generalization. DENTEX and smartphone content overlap and unavailable patient identifiers limit independence; the DSLR partitions are patient-disjoint but still serve model selection. Seed-based intervals do not quantify patient-level sampling uncertainty. Component-learning ablations use a single seed, and the anatomical scoring comparisons are conditional on checkpoints selected with combined scoring.')
    w.p('External detectors retain framework-specific pretraining, input policies, optimization, and post-processing. The descriptive AP rankings therefore do not establish a fully matched architectural effect or statistical state-of-the-art superiority. RT-DETR-L and official DEIM-s exceed the mean DSLR AP of DENT-STRIDE under their respective protocols. Prototype slots without training support cannot be interpreted as learned class evidence. No clinician reader study or confidence-calibration study was performed.')
    w.heading('6. Conclusion',1)
    w.p('DENT-STRIDE combines explicit FDI identity supervision, acquisition-specific feature processing, and separate training-derived anatomical scoring in a query detector. Three-seed AP is 53.897 ± 0.430 on panoramic radiographs, 55.219 ± 0.327 on smartphone photographs, and 72.293 ± 0.651 on DSLR photographs. Retrained HOOA/factor ablations and same-checkpoint center-prior comparisons provide mechanism-specific support. Persistent duplicate labels despite high matched-tooth accuracy show that tooth-level detection and image-level numbering consistency require separate evaluation.')

def build_main(assets):
    d=Document(MAIN_SOURCE);p=list(d.paragraphs)
    original_math=len(d._element.xpath('.//m:oMath'))
    replacements={
      4:'Automatic tooth detection with Fédération Dentaire Internationale (FDI) numbering must accommodate panoramic radiographs and intraoral photographs while distinguishing localization, identity, and numbering consistency. We present DENT-STRIDE, an acquisition-configured DEIM–D-FINE framework with factorized identity supervision, multiscale feature processing, and a separate training-derived anatomical prior. Three domain configurations were each trained with seeds 0, 1, and 2 for 160 epochs. Mean validation AP was 53.897 ± 0.430 for panoramic radiographs, 55.219 ± 0.327 for smartphone photographs, and 72.293 ± 0.651 for DSLR photographs. Same-checkpoint center scoring increased panoramic AP by 0.494 points on average. Retrained Mobile branch ablations reduced AP by 0.399–1.349 points when removing HOOA responses and by 0.244–1.274 points when removing factor-supervision targets. Other auxiliary and optimization settings showed configuration-dependent effects rather than uniform gains. Fixed-threshold F1 was 88.877%, 81.461%, and 91.869%, respectively. High matched-tooth identity accuracy coexisted with repeated raw FDI labels. These results support acquisition-specific tooth detection and distinguish mechanism-level evidence from complete-system performance; they do not establish autonomous clinical charting.',
      12:'DENT-STRIDE configures a common DEIM–D-FINE query detector for each acquisition cohort. Factorized heads provide explicit FDI identity targets; SpatialShape supplies training-derived class-specific center and box-shape scores; and CMFA/HOOA process cross-scale and local contextual features. Prototype supervision and optimization settings are defined as parts of the evaluated configurations, rather than assumed sources of additional accuracy. Configuration is fixed for each acquisition domain before inference.',
      13:'The contributions are threefold. First, we formulate explicit query-level FDI identity learning through quadrant, tooth-position, and dentition supervision, distinguishing auxiliary supervision from inference score fusion. Second, we introduce a separate tooth-specific anatomical scoring model derived from training-box center and shape statistics and isolate its effect through same-checkpoint comparisons. Third, we evaluate acquisition-specific detector configurations across radiographic and photographic cohorts, reporting three-seed performance and retrained branch ablations that distinguish supported mechanisms from configuration sensitivity.',
      23:'The common scaffold comprises an HGNetv2-B0 backbone, three 256-channel projected feature levels, a hybrid encoder, and a three-layer D-FINE decoder with 300 ordinary queries. DEIM supplies matching and detection losses (Huang et al., 2025; Peng et al., 2025). The domain configurations are DENT-STRIDE-Pano (Factor + Prototype with SpatialShape), DENT-STRIDE-Mobile (CMFA + HOOA with Factor + Prototype), and DENT-STRIDE-DSLR (CMFA + DualOpt). Pano uses factor/prototype auxiliary supervision and anatomical scoring; Mobile additionally uses factor-score fusion; DSLR uses no identity auxiliary branches. Fig. 2 shows the shared scaffold and acquisition-specific components.',
      25:'Fig. 2. Shared detector and acquisition-specific components. CMFA replaces four adjacent-scale fusion sites for Mobile/DSLR; HOOA refines Mobile feature levels. Factor heads provide supervision for Mobile/Pano and score fusion only for Mobile. Their quadrant, position, and dentition fusion weights are 0.10, 0.05, and 0.03. The K-class, four-prototype-per-class bank supplies auxiliary training supervision only. Pano uses fixed training-derived center and shape scores, while DSLR uses DualOpt during optimization. The dataset label map decodes outputs into clinical FDI codes. Solid arrows denote prediction flow; dashed arrows denote auxiliary supervision or prior fitting. Component roles are summarized in Table 2.',
      54:'Fig. 3 visualizes the training-derived center templates; Supplementary Fig. S1 shows the corresponding width/height distributions. These fixed geometric statistics are distinct from the learnable identity bank. The four-way same-checkpoint comparison in Section 4.8 isolates center and shape scoring effects.',
      56:'Fig. 3. Training-derived center templates for the 32 permanent DENTEX FDI classes. Each panel shows the class-specific center distribution in normalized image coordinates, based on training-box means and full regularized covariances. The templates are fixed at inference and are separate from the learnable appearance prototypes; shape statistics are provided in Supplementary Fig. S1.',
      60:assets['4_caption'],
      61:'Mobile uses HOOA after encoder aggregation (Fig. 5). Local, dilated-context, high-pass, and multiplicative local–context responses are projected, channel-gated, and spatially gated before a residual update bounded by 0.30 × sigmoid(s). HOOA is absent from the DSLR and Pano configurations. Removing the high-pass branch replaces its response with zero; removing the second-order interaction replaces the multiplicative response with the direct context response, preserving the context path and projection dimensions.',
      63:assets['5_caption'],
      65:'Training combines native detection losses with configuration-specific factor/prototype supervision and an arch-position term; Mobile additionally uses a normalized Wasserstein-distance box term. With inactive terms set to zero, the objective is',
      69:'For Mobile, the box-distance term uses normalized center c and width–height vector s. For each matched prediction and target, d = √(‖c − c*‖² + ‖s − s*‖²/4), and LNWD is the mean of 1 − exp(−d/0.08), weighted by 0.50. Pano and DSLR do not use this term. Factor cross-entropy weights for quadrant/position/dentition are 0.10/0.10/0.05 in Pano and 0.08/0.08/0.03 in Mobile.',
      70:'The DSLR configuration uses DualOptAdamW: an AdamW step followed by parameter-rank-scaled additional decay and rollback toward initialization (Ning et al., 2026). Rollback weight is 0.005, additional decay is 0.01, rank exponent is 1.5, and scaling bounds are 0.5–1.5. These operations change training optimization, not the inference architecture. Their separate sensitivity controls retain the same ordinary AdamW base weight decay.',
      74:'Table 1. Dataset composition of the training and validation partitions.',
      75:'DENTEX2023 is a panoramic radiography cohort (Hamamci et al., 2023). DSLR training/validation patient counts are 150/38.',
      76:'The image-level partitions contain one DENTEX and four smartphone training–validation content-overlap groups, plus a within-validation DENTEX duplicate group. Patient-level independence cannot be established for these two cohorts. DSLR training and validation patients are disjoint. All reported results use validation partitions for checkpoint selection.'
    }
    for index,text in replacements.items():set_text(p[index],text)
    if '2' in assets:replace_picture(p[24],assets['2'])
    replace_picture(p[59],assets['4']);replace_picture(p[62],assets['5'])
    insert_main_experiments(d,p,assets)
    assert len(d._element.xpath('.//m:oMath'))==original_math,'Native Word equations changed'
    remove_unreferenced_images(d)
    d.core_properties.subject='Acquisition-configured tooth detection and FDI numbering'
    d.core_properties.comments=''
    d.save(MAIN_OUT)
    return d

def blank_like(source):
    d=Document(source)
    body=d._element.body
    for node in list(body):
        if node.tag!=qn('w:sectPr'):body.remove(node)
    return d

def build_supp(assets):
    d=blank_like(SUPP_SOURCE);w=Writer(d)
    w.p('Supplementary material','Title')
    w.p('DENT-STRIDE: Domain-Configured Structured FDI Tooth Detection Across Panoramic Radiographs and Intraoral Photographs')
    w.p('Detailed repeated-seed results, component ablations, anatomical scoring comparisons, class-wise outcomes, and reproducibility settings for the acquisition-specific configurations.')
    w.heading('S1. Individual-seed performance',1)
    w.p('Each final domain configuration was trained for 160 epochs with seeds 0, 1, and 2. The selected exponential-moving-average checkpoint maximizes validation COCO AP under the specified inference policy. Pano selection includes center and shape scoring; Mobile includes factor fusion; DSLR uses neither. All metrics are evaluated on the original training/validation partitions. The three-seed means and sample standard deviations in the main article use these nine training runs.')
    w.table('Table S1. Individual-seed detection metrics for the final configurations. Metrics are percentage points; checkpoint epochs are one-based.',
       ['Cohort','Seed','AP','AP50','AP75','AR100','Best epoch','Params (M)'],
       [[c,r['seed'],*[fmt(r[k]) for k in ['AP','AP50','AP75','AR100']],r['best_epoch_one_based'],fmt(r['params_M'])] for c in COHORTS for r in sorted([r for r in SEEDS if r['cohort']==c],key=lambda x:x['seed'])],
       widths=[.68,.43,.76,.76,.76,.76,.72,.78],size=9)
    w.p('The fixed Pano geometric templates are estimated only from training boxes. Their normalized width and height distributions are shown in Fig. S1; inference uses the joint quadratic score of log-width and log-height defined in the main article, rather than independent empirical histograms.')
    w.figure('S1',OLD/'publication_figures/Figure_S4.png','Normalized width and height distributions of the 16,234 DENTEX training boxes, grouped by the 32 permanent FDI classes. The distributions describe training geometry; the SpatialShape shape term uses the joint regularized quadratic score of log-width and log-height. No validation boxes are used to fit these templates.')
    w.heading('S2. Complete component-ablation results',1)
    w.p('Twenty-one additional models were trained with seed 0. Each comparison changes only its designated component relative to the specified group reference; the full Mobile, full DSLR, and factor-fusion-off reference checkpoints are reused when appropriate. CMFA switches preserve the 512-channel output. The HOOA high-pass removal sets that response to zero, whereas interaction removal substitutes the direct context response. Factor-supervision removals retain all heads and disable both training and inference fusion. Prototype and optimization controls preserve the remaining model and ordinary AdamW base decay. Background controls change only NWD/arch-position terms.')
    ablations=[r for r in INTERNAL if r['component']!='Factor fusion']
    w.table('Table S2. Complete seed-0 component and background-loss sensitivity. Positive ΔAP means the displayed alternative exceeds its specified reference.',
       ['Component','Setting','AP','AP50','AP75','AR100','ΔAP'],
       [[r['component'],r['setting'],*[fmt(r[k]) for k in ['AP','AP50','AP75','AR100']],signed(r['delta_AP_vs_reference'])] for r in ablations],
       widths=[1.05,1.6,.65,.65,.65,.65,.65],size=8.5,numeric_from=2)
    w.p('The Mobile full-model reference is 54.910 AP for gates, HOOA, prototypes, and background losses. Factor supervision uses the all-factor fusion-off model at 55.537 AP. Optimization uses DSLR CMFA with ordinary AdamW at 72.892 AP. Channel-only CMFA reached 55.422 AP; four prototypes at temperature 0.50 reached 55.683; additional decay alone reached 73.351 on DSLR; NWD alone reached 55.396 on Mobile. Removing prototype supervision changed AP by only +0.008. These single-seed findings are not estimates of stable superiority and do not replace the reported three-seed configurations.')
    w.figure('S2',assets['S6'],'Complete component-sensitivity plots for settings not emphasized in main Fig. 11. All displayed points are seed-0 observations, without uncertainty bars. ΔAP uses the group-specific reference in Table S2 or the factor-fusion-off reference in Table S5. Positive values indicate an alternative that exceeds the reference; they are not gains attributed to the intact module.')
    w.heading('S3. Same-checkpoint anatomical scoring',1)
    w.table('Table S3. Pano prior settings for each seed. Weights and box coordinates are fixed within each seed; all candidate logits are rescored before native top-k selection.',
       ['Seed','Prior','AP','AP50','AP75','AR100','ΔAP'],
       [[r['seed'],PRIOR_LABELS[r['prior']],*[fmt(r[k]) for k in ['AP','AP50','AP75','AR100']],signed(r['delta_AP_vs_none'])] for r in PRIOR_SEEDS],
       widths=[.4,1.5,.9,.9,.9,.9,.8],numeric_from=2)
    w.p('The three checkpoints were selected using the combined center–shape prior; prior-off, center-only, and shape-only outputs are evaluations of those same selected weights, without additional selection. Factor fusion, appearance scoring, and global assignment remain disabled. The panoramic view term is constant across supported classes. Center/log-shape weights are 0.20/0.05. These post-selection comparisons isolate score correction rather than training an alternative detector.')
    prior_index={(r['seed'],r['prior']):r for r in PRIOR_SEEDS}
    w.table('Table S4. Same-checkpoint paired AP differences relative to no prior over three seeds. Student-t intervals describe training-seed variability on the fixed validation split (df = 2).',
       ['Prior','Seed 0','Seed 1','Seed 2','Mean ± SD','95% CI'],
       [[PRIOR_LABELS[r['prior']],*[signed(prior_index[s,r['prior']]['AP']-prior_index[s,'none']['AP']) for s in [0,1,2]],f"{r['AP_delta_vs_none_mean']:+.3f} ± {r['AP_delta_vs_none_SD']:.3f}",f"[{r['AP_delta_vs_none_CI95_low']:.3f}, {r['AP_delta_vs_none_CI95_high']:.3f}]"] for r in PRIORS if r['prior']!='none'],
       widths=[1.3,.65,.65,.65,1.25,1.35])
    w.figure('S3',assets['S12'],'Same-checkpoint center, shape, and combined prior effects. Points show the mean paired AP difference versus no prior and bars show exploratory Student-t 95% intervals over the three seeds (df = 2). The zero line denotes no AP change. Intervals are conditional on validation-selected weights and are not patient-sampling or independent-test intervals.')
    w.heading('S4. Factor-score fusion',1)
    w.table('Table S5. Training-fusion × inference-fusion comparison at seed 0. Each inference pair uses the same checkpoint within its training state.',
       ['Training fusion','Inference fusion','AP','AP50','AP75','AR100','F1'],
       [[r['training_fusion'],r['inference_fusion'],*[fmt(r[k]) for k in ['AP','AP50','AP75','AR100','F1']]] for r in FUSION],
       widths=[1.12,1.12,.8,.8,.8,.8,.8],numeric_from=2)
    w.p('Both training states retain the same factor and prototype supervision. Quadrant, tooth-position, and dentition inference-fusion weights are 0.10, 0.05, and 0.03. Inference fusion slightly reduces AP in both states, while increasing fixed-threshold F1. The training-state comparison is a single-seed learning contrast; inference on/off is a same-checkpoint score change. F1 uses score ≥0.30 and IoU ≥0.50.')
    w.heading('S5. Fixed-threshold endpoints and complete class outcomes',1)
    w.table('Table S6. Individual-seed fixed-threshold detection and identity endpoints (%).',
       ['Cohort','Seed','P','R','F1','Matched FDI','Coverage','Duplicate images'],
       [[c,r['seed'],*[fmt(r[k]) for k in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']]] for c in COHORTS for r in sorted([r for r in SEEDS if r['cohort']==c],key=lambda x:x['seed'])],
       widths=[.6,.4,.84,.84,.84,1.04,.95,1.06],size=8.5)
    w.p('Native top-300 predictions are retained at score ≥0.30 without added NMS. Class-agnostic Hungarian assignment maximizes total IoU and accepts pairs at IoU ≥0.50. A correct detection requires both localization and exact FDI agreement. P = correct detections / retained predictions; R = correct detections / reference teeth; matched FDI accuracy = correct identities / localization matches; coverage = localization matches / reference teeth. Wrong-number matches contribute a false positive and a false negative. Duplicate-image rate uses all validation images and is not a clinical charting-error rate.')
    w.figure('S4',ROOT/'prediction_diagnostics/Figure_S6_detection_curves.png','Detection and threshold diagnostics for the final seed-0 configurations. Rows show Pano, Mobile, and DSLR. Columns show category-macro-averaged COCO interpolated precision at IoU 0.50 and maxDets = 100, F1, and raw duplicate-image rate. Unsupported categories are omitted from the precision average. Thresholds range from 0.05 to 0.95 in steps of 0.05; dotted lines indicate 0.30. These are cohort-level detection diagnostics, not clinical charting-error estimates.')
    for c,num in [('Pano',7),('Mobile',8),('DSLR',9)]:
        r=read(ROOT/'prediction_diagnostics'/(c+'_complete_FDI_confusion.json'))
        s=r['summary']
        w.figure('S'+str(num-2),r['figure_files']['png'],f'Complete {c} FDI outcome matrix and class support for the final seed-0 configuration. Each reference tooth contributes one predicted FDI or a missed outcome; colors show reference-row percentages. The matrix contains {s["validation_GT"]:,} reference teeth: {s["correct_FDI_matches"]:,} correct identities, {s["wrong_FDI_localized_matches"]:,} wrong-FDI localization matches, and {s["missed_GT"]:,} missed teeth. The {s["unmatched_predictions"]:,} unmatched predictions are counted separately, not assigned a reference row. Training support totals {s["train_teeth"]:,} teeth. Gray rows have no validation support and are unavailable, not zero-accuracy classes.')
    w.heading('S6. Upstream and external detector references',1)
    old=Document(SUPP_SOURCE)
    w.table('Table S7. Three-seed official DEIM-s and RT-DETRv2-S AP references on the same validation cohorts. Each framework retains its own training implementation.',
       [c.text for c in old.tables[3].rows[0].cells],[[c.text for c in row.cells] for row in old.tables[3].rows[1:]],
       widths=[.85,1.7,.65,.65,.65,1.15],numeric_from=2)
    w.p('Framework-level references use shared validation labels and clinical FDI mappings but retain native pretraining, augmentation, optimization, and output rules. Official DEIM-s uses its HGNetv2-B0/D-FINE scaffold. RT-DETRv2-S uses the official ResNet18-vd implementation with ImageNet-pretrained backbone weights, 160 epochs, 640 × 640 input, training batch size 8, mixed precision, EMA, and maximum-validation-AP checkpoint selection; no MixUp or geometric flips are used. YOLOv8-s/YOLO11-s use official Ultralytics detector initialization; Faster R-CNN uses ResNet-50-FPN; Dome-DETR-S uses HGNetv2-B0; CAF-YOLO-s uses its attention-enhanced implementation. Cross-framework comparisons are not all-hyperparameter-matched module ablations.')
    w.table('Table S8. Conventional detector secondary metrics for the panoramic and Mobile validation cohorts. All rows are single-seed framework-level references.',
       [c.text for c in old.tables[4].rows[0].cells],[[c.text for c in row.cells] for row in old.tables[4].rows[1:]],
       widths=[.8,1.6,.9,.9,.9,.9],numeric_from=2)
    w.heading('S7. Reproducibility settings and learning trajectories',1)
    w.p('The common hybrid encoder retains learned positional embeddings and dynamic upsampling in the evaluated implementations. These shared architecture settings are held fixed in the component comparisons and are not claimed as separate contributions.')
    w.table('Table S9. Effective training and inference settings of the three final configurations. Loss coefficients refer to active supervised terms.',
       ['Setting','Pano','Mobile','DSLR'],[
       ['Backbone initialization','ImageNet HGNetv2-B0','ImageNet HGNetv2-B0','ImageNet HGNetv2-B0'],
       ['Epochs; train/validation batch','160; 8/16','160; 8/16','160; 8/16'],
       ['Input policy','Fixed 640','Fixed 1024','480–800 multiscale; eval 640'],
       ['CMFA / HOOA','Off / Off','On / On','On / Off'],
       ['Factor CE q/p/d','0.10/0.10/0.05','0.08/0.08/0.03','Inactive'],
       ['Factor fusion train/inference','Off / Off','On / On','Off / Off'],
       ['Prototype count; temperature; loss','4; 0.20; 0.05','4; 0.20; 0.03','Inactive'],
       ['Prototype inference score','Excluded','Excluded','Excluded'],
       ['NWD / arch-position','Inactive / 0.25','0.50 / 0.10','Inactive / 0.25'],
       ['Mosaic / MixUp','0 / 0.50','0 / 0','0 / 0'],
       ['IoU crop / zoom-out','0.80 / 0.50','0.80 / 0.50','0.80 / 0.50'],
       ['Photometric probability / flips','0.50 / disabled','0.50 / disabled','0.50 / disabled'],
       ['Optimizer','AdamW','AdamW','DualOptAdamW'],
       ['Center / shape score weights','0.20 / 0.05','Inactive','Inactive']],widths=[2,1.5,1.5,1.5])
    w.p('Ordinary AdamW uses detector/backbone learning rates of 5 × 10⁻⁴ / 2.5 × 10⁻⁴, betas (0.9, 0.999), and base weight decay 10⁻⁴, with normalization-parameter decay exclusions. DualOpt adds rollback 0.005, rank-dependent decay 0.01, rank exponent 1.5, and multiplier range [0.5, 1.5]. The common schedule uses warm-up, flat/cosine phases, gradient clipping at 0.10, EMA decay 0.9999, and terminal augmentation shutdown. Pano MixUp is active only within its scheduled interval. At ties in validation AP, the earlier checkpoint is retained. Acquisition-specific settings and model-selection rules are preserved across seeds.')
    w.figure('S8',assets['S5'],'Validation AP trajectories over the full 160-epoch training budget for the three acquisition-specific configurations. Thin lines show actual seeds 0, 1, and 2; thick lines and bands show their epoch-wise mean ± sample SD. Curves show the recorded trajectory without smoothing or synthetic interpolation, and do not replace selected-checkpoint endpoints.')
    w.heading('S8. Structured representations and visualization methods',1)
    mechanism=read(ROOT/'figures/mechanism/mechanism_provenance.json')
    factor=read(ROOT/'figures/mechanism/factor_confusion_summary.json')
    n=mechanism['captured_queries_total']
    w.p('The final Mobile seed-0 checkpoint supplies all representation and forward-response panels. Prototype centroids are formed by averaging four L2-normalized class prototypes and renormalizing; their cosine similarities summarize learned bank structure. Within-class cohesion is the mean of the six pairwise prototype cosine similarities. FDI slots 18, 28, 38, and 48 have zero Mobile training instances and are masked. Cohesion and similarity are representation diagnostics, not evidence that the prototype branch increases AP.')
    w.figure('S9',ROOT/'figures/mechanism/current_Mobile_prototype_diagnostics.png','Prototype representation diagnostics from the final Mobile seed-0 checkpoint. (A) Cosine similarity between normalized class-centroid prototypes ordered by FDI quadrant. (B) Within-class four-prototype cosine cohesion, grouped by quadrant. Gray unsupported slots are unavailable. The prototype bank supplies training supervision only and is distinct from fixed anatomical templates.')
    w.figure('S10',ROOT/'figures/mechanism/current_Mobile_cohesion_all_FDI.png','Per-class prototype cohesion for permanent and primary FDI slots in the final Mobile seed-0 checkpoint. Cohesion averages the six pairwise cosine similarities among four prototypes. Unsupported classes 18, 28, 38, and 48 are unavailable. No inference prototype score is added to detection logits.')
    w.p(f'Auxiliary factor matrices use {n:,} uniquely matched ordinary queries whose maximum sigmoid class score is at least 0.30 and localization IoU is at least 0.50 under class-agnostic Hungarian matching. This query-level denominator differs from native post-top-k detection outputs, which may retain multiple query–class pairs. Quadrant, position, and dentition accuracies are {factor["quadrant"]["accuracy"]*100:.2f}%, {factor["position"]["accuracy"]*100:.2f}%, and {factor["dentition"]["accuracy"]*100:.2f}%, respectively. Factor targets follow the clinical FDI map.')
    w.figure('S11',ROOT/'figures/mechanism/current_Mobile_factor_confusions.png',f'Row-normalized auxiliary quadrant, tooth-position, and dentition confusion matrices for {n:,} localized unique ordinary queries from the final Mobile seed-0 checkpoint. Gray rows have no reference observations. These branches supervise structured identity, not objectness. Their unique-query matching protocol is distinct from post-top-k detector endpoints.')
    w.p('Main Fig. 13 captures the first top-down CMFA module and the highest-resolution HOOA module during an actual forward pass. Both channel and spatial gates are obtained from native module tensors; no Grad-CAM substitutes are used. Response maps summarize channel RMS and are independently normalized for display. Passive capture hooks reproduce the module outputs and leave model predictions unchanged. Channel-gate grid coordinates index channels, not anatomical position.')
    w.p('Same-image qualitative panels use a common score threshold, clinical FDI decoding, and identical target regions across methods. Component panels use independently trained ablation checkpoints; prior panels switch only scoring terms for the same checkpoint before top-k selection. Examples illustrate prediction changes and do not estimate their population frequencies. Statistical plots show individual seed observations, sample SD, or paired-seed intervals as specified in the captions; single-seed comparisons are shown without uncertainty bars.')
    remove_unreferenced_images(d)
    d.core_properties.subject='DENT-STRIDE supplementary methods and results'
    d.core_properties.comments=''
    d.save(SUPP_OUT)
    return d

def check(d,path):
    text='\n'.join(p.text for p in d.paragraphs)+'\n'+'\n'.join(c.text for t in d.tables for row in t.rows for c in row.cells)
    forbidden=['A07','phone_a07','current snapshot','source-rounded','author-confirmed','Not run','Not defined','AUTHOR COMPLETION','PUBLICATION-SAFETY','Pending confirmatory','present revision','pre-revision','dirty Git','540','XX','54.583','55.026','53.580','72.878','55.872','54.023']
    bad=[s for s in forbidden if s.lower() in text.lower()]
    assert not bad,(path,bad)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    MANIFEST[str(path)]={'paragraphs':len(d.paragraphs),'tables':len(d.tables),'images':len(d.inline_shapes),'equations':len(d._element.xpath('.//m:oMath')),'sha256':source_sha(path),'forbidden_phrases_found':bad}

def main():
    assets=get_assets()
    a=build_main(assets);b=build_supp(assets)
    check(a,MAIN_OUT);check(b,SUPP_OUT)
    for p,h in MANIFEST['source_documents'].items():assert source_sha(p)==h,'Original manuscript overwritten'
    (ROOT/'document_build_manifest.json').write_text(json.dumps(MANIFEST,indent=2,ensure_ascii=False))
    print(MAIN_OUT);print(SUPP_OUT)

if __name__=='__main__':main()
