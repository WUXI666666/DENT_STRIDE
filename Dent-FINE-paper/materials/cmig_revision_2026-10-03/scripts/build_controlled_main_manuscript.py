"""Integrate verified controlled experiments and real-model figures into a new Word manuscript."""
from pathlib import Path
import copy
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

PAPER=Path('/home/wuxi/Dent-FINE-paper')
ROOT=PAPER/'materials/cmig_revision_2026-10-03'
SRC=PAPER/'DENT_STRIDE_CMIG_main_visual.docx'
OUT=PAPER/'DENT_STRIDE_CMIG_main_controlled_results_20261003.docx'
E=json.loads((ROOT/'evidence/controlled_evidence.json').read_text())
SELECT=json.loads((ROOT/'evidence/selected_controlled_comparison.json').read_text())['rows']
SUMMARY={r['configuration']:r for r in E['primary_summary']}
PRIORS={r['prior']:r for r in E['prior_summary']}
INTERNAL={r['configuration']:r for r in E['internal_ablations']}
PERF=ROOT/'figures/performance'
QUAL=ROOT/'figures/qualitative'
MECH=ROOT/'figures/mechanism'


def val(x):return '—' if x is None else f'{x:.3f}'


def pm(x):return f"{x['mean']:.3f} ± {x['SD']:.3f}"


def ci(x):return f"[{x['CI95_low']:.3f}, {x['CI95_high']:.3f}]"


def original_run(name):
    p=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29/runs')/name/'seed_0/metrics_summary.json'
    return json.loads(p.read_text())


def text_format(run, size=11, bold=False):
    run.font.name='Times New Roman';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor(0,0,0)
    rpr=run._element.get_or_add_rPr();fonts=rpr.find(qn('w:rFonts'))
    if fonts is None:fonts=OxmlElement('w:rFonts');rpr.insert(0,fonts)
    for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Times New Roman')


def set_text(p, text):
    p.clear();r=p.add_run(text);text_format(r,10 if p.style.name=='Caption' else 11)


def main():
    doc=Document(SRC)
    original_paragraphs=list(doc.paragraphs)
    original_tables=list(doc.tables)
    # Preserve all original method equations and formal author/declaration sections.
    replacements={
      4:('Automatic tooth detection with Fédération Dentaire Internationale (FDI) numbering must handle panoramic radiographs and intraoral photographs while separating localization, identity, and numbering consistency. We present DENT-STRIDE, a domain-configured DEIM–D-FINE framework with factorized identity heads, prototype auxiliary supervision, cross-scale feature aggregation, and a separate training-derived anatomical prior. Fourteen controlled training configurations were evaluated with seeds 0, 1, and 2 over 160 epochs, using the same data partitions and within-domain training protocol. The Pano, Mobile, and DSLR configurations achieved internal-validation AP of 53.580 ± 0.646, 55.026 ± 0.396, and 72.878 ± 0.404, respectively. Corresponding controlled references achieved 53.022 ± 0.392, 54.730 ± 0.149, and 72.466 ± 0.234. On DSLR images, CMFA with AdamW increased mean AP by 0.498 points. Same-weight panoramic replay increased mean AP by 0.453 with the center prior and 0.414 with the combined center–shape prior. Mobile module gains were smaller than the resolution effect, and several single-seed module simplifications exceeded the intact configuration. Matched-tooth FDI accuracy remained high, but duplicate labels persisted in raw predictions. These results support acquisition-specific configuration and distinguish repeatable component effects from configuration sensitivity; they do not establish a universal optimum or autonomous clinical charting.'),
      8:('Panoramic radiographs provide a comparatively standardized global view, whereas intraoral photographs vary in framing, viewpoint, occlusion, illumination, and reflection. Smartphone acquisition supports remote review but introduces substantial image variability (Alavi et al., 2025; Moharrami et al., 2024). Standardized five-view DSLR photography reduces some variability without eliminating tooth gaps, mixed dentition, or similar neighbouring teeth. Fig. 1 presents representative images and FDI annotations from the three acquisition cohorts.'),
      10:('Fig. 1. Representative acquisition conditions and reference FDI annotations. Panoramic radiographs, smartphone intraoral photographs, and DSLR photographs are shown with their existing bounding-box annotations. The examples illustrate differences in global field of view, tooth scale, framing, and illumination. Visible spaces alone do not establish clinically missing teeth.'),
      12:('DENT-STRIDE configures a common DEIM–D-FINE detector for each acquisition cohort. Factorized heads and a prototype bank provide structured identity supervision; SpatialShape supplies class-specific center and box-shape scores; and cross-scale mutual feature aggregation (CMFA) changes feature exchange. HOOA and optimization choices are evaluated as configuration components. Configuration is fixed for each acquisition domain before inference.'),
      13:('The contributions are threefold. First, we formulate explicit query-level FDI identity learning through factorized quadrant, tooth-position, and dentition supervision, while distinguishing factor-score fusion from training-only prototype supervision. Second, we define a separate tooth-specific anatomical scoring model from training-box center and shape statistics, and isolate its inference effect by replaying the same weights with four prior settings. Third, we evaluate these mechanisms within a common acquisition-configured detector using three-seed module comparisons, controlled resolution experiments, and module-internal ablations. The evaluation identifies reproducible cross-scale and center-prior benefits, while reporting configurations for which additional supervision, gates, or optimization operations do not improve AP.'),
      23:('The common scaffold comprises an HGNetv2-B0 backbone, three 256-channel projected feature levels, a hybrid encoder, and a three-layer D-FINE decoder with 300 ordinary queries. DEIM supplies matching and detection losses (Huang et al., 2025; Peng et al., 2025). The domain configurations are termed DENT-STRIDE-Pano (Factor + Prototype with SpatialShape), DENT-STRIDE-Mobile (CMFA + HOOA with Factor + Prototype), and DENT-STRIDE-DSLR (CMFA + DualOpt). Pano uses factor/prototype auxiliary supervision and anatomical scoring; Mobile additionally uses factor-score fusion; DSLR uses no identity auxiliary branches. Fig. 2 shows the component family, and optional comparison refiners are detailed in Supplementary Section S1.'),
      25:('Fig. 2. DENT-STRIDE component family. The domain settings in Table 2 determine which components are active: CMFA for Mobile/DSLR, HOOA for Mobile, identity supervision for Mobile/Pano, factor-score fusion for Mobile, and SpatialShape for Pano. Other refiners are supplementary comparisons. Prediction and factor heads branch from decoded query features; the prototype bank supplies training supervision only. The lower panel shows parallel native loss terms combined before optimization; configuration-dependent auxiliary terms are defined in Section 3.5.'),
      54:('Fig. 3 visualizes the training-derived center templates; Supplementary Fig. S4 shows the corresponding width/height distributions. These fixed geometric statistics are distinct from the learnable identity bank. The four-way same-weight replay in Section 4.8 isolates center and shape scoring effects.'),
      58:('CMFA replaces four adjacent-scale fusion sites: two top-down FPN and two bottom-up PAN sites (Fig. 4). Aligned inputs A and B each have 256 channels. Their concatenation is globally pooled and passed through 1×1 Conv–SiLU–1×1 Conv–Sigmoid to obtain channel gates g_A and g_B. Cross-enhanced features A′ = A + g_B⊙B and B′ = B + g_A⊙A are concatenated. Channel-wise mean/max maps then drive a 7×7 convolution and sigmoid spatial gate G_s. The gated 512-channel output feeds the next fusion block. In gate ablations, the subsequent network and output dimensions are held fixed; disabling a gate removes its modulation without replacing the feature projection.'),
      61:('Mobile also uses HOOA after encoder aggregation (Fig. 5). Local, dilated-context, high-pass, and multiplicative local–context responses are projected, channel-gated, and spatially gated before a residual update bounded by 0.30 × sigmoid(s). HOOA is absent from the DSLR and Pano configurations. Disabling the high-pass branch replaces its output with zero; disabling the second-order interaction replaces the multiplicative response with the direct context response, preserving the context branch and projection dimensions. CEDv2, SMWG, MPF, and UCM are supplementary comparison refiners, rather than required components of the evaluated primary configurations.'),
      70:('DualOptAdamW performs an AdamW step followed by parameter-rank-scaled post-step decay and rollback toward initialization (Ning et al., 2026). The DSLR configuration uses rollback 0.005, additional decay 0.01, rank-scaling power 1.5, and scaling bounds 0.5–1.5. Ordinary AdamW retains the same base weight decay in every optimization control. These operations are investigated as training alternatives; they do not change the inference architecture and are not assumed to provide a universal improvement.'),
      130:('The controlled experiments distinguish three contributors to acquisition-specific performance. CMFA increased DSLR AP consistently when compared under the same optimizer. Center-based anatomical scoring increased panoramic AP at fixed weights, whereas the shape term contributed little on average and the combined prior did not exceed the center prior alone. Mobile resolution changes produced a larger AP difference than adding the complete feature and identity configuration. Thus, the strongest evidence concerns specific mechanisms under specified acquisition conditions, rather than the number of modules enabled.'),
      131:('Earlier label-tree and prior-knowledge systems already encode identity and anatomy (Zhang et al., 2018; Ali et al., 2023). DENT-STRIDE brings factor-score fusion, prototype supervision, and fixed geometric scores into a common query detector with distinct training and inference roles. The same-checkpoint replay isolates geometric score correction, while retrained ablations test learning effects. Prototype and gate visualizations explain how the executed model represents information, but their appearance does not establish an AP gain or the necessity of every branch.'),
      133:('Training-center patterns in Fig. 3 provide an interpretable basis for panoramic scoring. Center-prior AP differences were positive in all three matched checkpoints, but adding shape scoring did not improve the average over center scoring alone. Smartphone framing is less standardized, and the controlled feature–identity combinations showed interactions rather than independent additive benefits. Domain configuration should therefore be evaluated as a defined protocol, with resolution, background losses, and inference rules stated explicitly.'),
      134:('The same-image comparisons and threshold analyses distinguish localization, FDI identity, and image-level numbering. High conditional identity accuracy coexists with repeated raw labels, including a higher DSLR duplicate-image rate for the combined model than for its controlled reference. Clinical deployment would require handling of tooth gaps, mixed dentition, overlapping views, duplicate labels, and uncertainty, followed by evaluation of the complete charting workflow.'),
      136:('The three-seed results quantify training variability on the existing validation partitions. They remain validation-selected estimates without an independent test cohort. DENTEX and smartphone train–validation content overlap may increase estimated performance, and absent patient identifiers prevent verification of patient-level independence. Seed-based intervals do not quantify patient sampling uncertainty.'),
      137:('The primary within-domain comparisons control resolution, augmentation, background losses, pretraining, budget, and checkpoint selection; the three acquisition domains retain different input policies. Module-internal learning ablations use one seed and therefore cannot establish stable superiority of a simplified or intact setting. External detectors retain native pretraining and post-processing. Multiple exploratory contrasts are reported without multiplicity adjustment, and positive interval bounds should not be interpreted as definitive clinical or state-of-the-art evidence.'),
      138:('RT-DETR-L retains higher DSLR AP than the DENT-STRIDE DSLR configuration, while several Mobile simplifications improve single-seed AP. Unsupported FDI classes constrain prototype interpretation. Qualitative examples and forward-pass responses provide mechanistic descriptions rather than clinical validation. No clinician reader study or confidence-calibration study was performed.'),
      140:('DENT-STRIDE separates structured FDI identity learning, training-derived anatomical scoring, and cross-scale feature exchange within a common query detector. Controlled three-seed configurations achieve 53.580 ± 0.646 AP on panoramic radiographs with the combined prior, 55.026 ± 0.396 on smartphone images, and 72.878 ± 0.404 on DSLR images. CMFA and center-prior comparisons provide the clearest consistent component effects; Mobile resolution, module interactions, and single-seed simplifications limit stronger claims about the complete configuration. Persistent duplicate labels despite high matched-tooth accuracy show that detector performance and chart consistency require separate evaluation.'),
      154:('During manuscript preparation, the authors used OpenAI ChatGPT for language refinement, organization, consistency checking, and assistance with reproducible analysis and plotting code. Clinical images were taken from the existing datasets; detection overlays and internal-response plots were generated from actual model predictions and tensors. No clinical image or experimental result was generated using a generative image model. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the publication.')
    }
    for i,text in replacements.items():set_text(original_paragraphs[i],text)
    # The acquisition panel no longer contains historical model outputs.
    original_paragraphs[9].clear()
    original_paragraphs[9].alignment=WD_ALIGN_PARAGRAPH.CENTER
    original_paragraphs[9].add_run().add_picture(str(QUAL/'Acquisition_cohort_montage.png'),width=Inches(6.49))
    original_paragraphs[24].clear()
    original_paragraphs[24].alignment=WD_ALIGN_PARAGRAPH.CENTER
    original_paragraphs[24].add_run().add_picture(str(ROOT/'figures/methods/DENT_STRIDE_overview_native_parallel_losses.png'),width=Inches(6.49))
    start=original_paragraphs[77]._element
    stop=original_paragraphs[128]._element
    node=start
    while node is not stop:
        following=node.getnext();node.getparent().remove(node);node=following
    anchor=stop
    insertion_records=[]

    def p(text='',style='Normal'):
        pp=doc.add_paragraph(style=style)
        if text:text_format(pp.add_run(text),10 if style=='Caption' else 11,bold=style.startswith('Heading'))
        if style=='Normal':pp.paragraph_format.space_after=Pt(6)
        if style=='Caption':pp.paragraph_format.space_after=Pt(7)
        anchor.addprevious(pp._element)
        return pp

    def heading(text,level=2):return p(text,'Heading '+str(level))

    def table(caption,headers,rows,widths=None,font=9):
        cap=p(caption,'Caption');cap.paragraph_format.keep_with_next=True
        tbl=doc.add_table(rows=1,cols=len(headers));tbl.alignment=WD_TABLE_ALIGNMENT.CENTER;tbl.autofit=False
        for k,t in enumerate(headers):tbl.rows[0].cells[k].text=t
        for row in rows:
            cells=tbl.add_row().cells
            for k,t in enumerate(row):cells[k].text=str(t)
        for ri,row in enumerate(tbl.rows):
            trpr=row._tr.get_or_add_trPr()
            if ri==0:
                repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
            keep=OxmlElement('w:cantSplit');trpr.append(keep)
            for k,cell in enumerate(row.cells):
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if widths:cell.width=Inches(widths[k]*6.49/sum(widths))
                for pp in cell.paragraphs:
                    pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if k in [0,1] and len(headers)>4 else WD_ALIGN_PARAGRAPH.CENTER
                    pp.paragraph_format.space_before=Pt(2);pp.paragraph_format.space_after=Pt(2)
                    for run in pp.runs:text_format(run,font,bold=ri==0)
                tcpr=cell._tc.get_or_add_tcPr()
                borders=OxmlElement('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    el=OxmlElement('w:'+edge);el.set(qn('w:val'),'nil');borders.append(el)
                tcpr.append(borders)
                if ri==0:
                    for edge in ['top','bottom']:
                        el=borders.find(qn('w:'+edge));el.set(qn('w:val'),'single');el.set(qn('w:sz'),'8' if edge=='top' else '5');el.set(qn('w:color'),'000000')
                if ri==len(tbl.rows)-1:
                    el=borders.find(qn('w:bottom'));el.set(qn('w:val'),'single');el.set(qn('w:sz'),'8');el.set(qn('w:color'),'000000')
        anchor.addprevious(tbl._element)
        insertion_records.append({'caption':caption,'headers':headers,'rows':rows})
        return tbl

    def figure(number,path,caption):
        pp=p();pp.alignment=WD_ALIGN_PARAGRAPH.CENTER;pp.paragraph_format.keep_with_next=True
        pp.add_run().add_picture(str(path),width=Inches(6.49))
        cap=p(f'Fig. {number}. '+caption,'Caption')
        cap.paragraph_format.keep_together=True
        insertion_records.append({'figure':number,'asset':str(path),'caption':caption})

    heading('4.2. Implementation and controlled training design')
    p('The primary experiment matrix contains 14 training configurations, each repeated with seeds 0, 1, and 2 for 160 epochs. Within each acquisition cohort, the controlled reference and module variants use the same frozen source implementation, ImageNet-pretrained HGNetv2-B0 backbone, dataset partitions, FDI mapping, augmentation, background losses, evaluator, and maximum-validation-AP checkpoint rule. The controlled reference removes the tested modules while preserving the cohort-specific background objective; it is therefore distinct from the upstream official DEIM-s implementation. The Pano and DSLR comparisons share the local detector source; Mobile uses its fixed structured-identity-capable implementation in every control.')
    p('Training uses mixed precision, batches of 8 for training and 16 for validation, and exponential moving averages. AdamW uses learning rates 5 × 10⁻⁴ for the base model and 2.5 × 10⁻⁴ for the backbone, betas (0.9, 0.999), and weight decay 10⁻⁴. The common within-cohort schedule retains warm-up, a flat phase, cosine decay, and a terminal low-learning-rate/no-augmentation phase. Geometric flips are disabled. The retained cohort settings are listed in Table 2; exact configurations and selected epochs are provided in Supplementary Tables S7 and S13.')
    table('Table 2. Cohort configurations and the roles of their components. Controls preserve these domain-level settings except for the indicated experimental variable.', ['Setting','Pano','Mobile','DSLR'],[
        ['Training/evaluation input','Fixed 640 × 640','Fixed 1024 × 1024','Multiscale training; 640 × 640 evaluation'],
        ['Feature components','Native encoder','CMFA + HOOA','CMFA'],
        ['Factor heads','Training supervision only','Training supervision and score fusion','Inactive'],
        ['Prototype bank','Training supervision only','Training supervision only','Inactive'],
        ['Anatomical scoring','Center + shape at inference','Inactive','Inactive'],
        ['Optimizer','AdamW','AdamW','DualOptAdamW'],
        ['Mosaic / MixUp probabilities','0 / 0.50','0 / 0','0 / 0'],
        ['IoU crop / zoom-out probabilities','0.80 / 0.50','0.80 / 0.50','0.80 / 0.50'],
        ['Arch-position loss weight','0.25','0.10','0.25'],
        ['NWD loss weight','Inactive','0.50; scale 0.08','Inactive'],
        ['Prototype loss weight','0.05','0.03','Inactive'],
    ],widths=[1.65,1.55,1.55,1.90],font=9)
    p('Mobile feature processing is CMFA + HOOA; structured identity is Factor + Prototype. Their four combinations are trained at 1024, and the base and combined configurations are additionally trained at 640 with the otherwise fixed Mobile protocol. The four Pano training combinations enable neither auxiliary mechanism, Factor only, Prototype only, or both; factor inference fusion and geometric scoring are disabled throughout model selection. The DSLR matrix crosses CMFA off/on with AdamW/DualOpt. Anatomical-prior and pure inference switches reuse frozen weights; all learning ablations are retrained. No single-seed internal simplification is substituted post hoc for the predefined combined models.')

    heading('4.3. Evaluation endpoints and statistical analysis')
    p('The primary endpoint is FDI-labelled COCO bounding-box AP over IoU thresholds 0.50:0.05:0.95. AP50, AP75, and AR100 are secondary endpoints. Category averages follow the COCO convention: classes without validation ground truth are excluded rather than assigned zero AP. Detector outputs retain native top-300 predictions without added NMS or FDI-uniqueness constraints in the DEIM family; external detectors retain native output processing. All selected weights are replayed in FP32 with the same cohort evaluator. Parameter counts include instantiated auxiliary heads.')
    p('At score ≥0.30, class-agnostic one-to-one Hungarian assignment maximizes total localization IoU; a matched pair is retained when IoU ≥0.50. A detection is correct only when the pair also agrees in FDI identity. Precision uses retained predictions as its denominator, recall uses reference teeth, matched-tooth FDI accuracy uses qualifying localization matches, and localization coverage uses reference teeth. F1 is micro-averaged from TP, FP, and FN. Duplicate-image rate is the proportion of images containing a repeated FDI label in thresholded native outputs. These quantities evaluate distinct aspects of detection and are not clinical charting-error rates.')
    p('Training variability is reported as mean ± sample standard deviation over seeds 0, 1, and 2. Paired AP contrasts subtract the corresponding seed-specific reference result. Student-t 95% intervals use three paired differences and two degrees of freedom; they describe variability across training seeds on the fixed validation cohort. The intervals are exploratory and unadjusted for multiple comparisons. Patient-level sampling intervals and independent test estimates are not available. Single-seed module-internal comparisons are presented without error bars. An em dash (—) denotes an unavailable result and is never treated as zero.')

    heading('4.4. Controlled three-seed primary comparisons')
    p('Table 3 reports all 14 controlled training configurations. The Mobile combined model achieved 55.026 ± 0.396 AP at 1024, compared with 54.730 ± 0.149 for its same-resolution controlled reference. The Pano Factor + Prototype model achieved 53.166 ± 0.526 AP before geometric scoring, compared with 53.022 ± 0.392 for the reference. The DSLR CMFA + DualOpt configuration achieved 72.878 ± 0.404 AP, compared with 72.466 ± 0.234 for the reference with AdamW. The geometric contribution is evaluated separately in Section 4.8.')
    order=[('mobile_1024_base','Mobile 1024','Base'),('mobile_1024_features','Mobile 1024','Features'),('mobile_1024_identity','Mobile 1024','Identity'),('mobile_1024_full','Mobile 1024','Combined'),('mobile_640_base','Mobile 640','Base'),('mobile_640_full','Mobile 640','Combined'),('pano_base','Pano','Base'),('pano_factor','Pano','Factor'),('pano_prototype','Pano','Prototype'),('pano_factor_prototype','Pano','Factor + Prototype'),('dslr_base_adamw','DSLR','Base + AdamW'),('dslr_cmfa_adamw','DSLR','CMFA + AdamW'),('dslr_base_dualopt','DSLR','Base + DualOpt'),('dslr_cmfa_dualopt','DSLR','CMFA + DualOpt')]
    rows=[]
    for name,cohort,label in order:
        r=SUMMARY[name];rows.append([cohort,label]+[pm(r['metrics'][m]) for m in ['AP','AP50','AP75','AR100']]+[f"{r['params_M']:.3f}"])
    table('Table 3. Controlled primary configurations, mean ± SD over three seeds. Pano values exclude geometric scoring and factor inference fusion. All metrics are percentages.', ['Cohort','Configuration','AP','AP50','AP75','AR100','Params (M)'],rows,widths=[.83,1.28,1.0,1.0,1.0,1.0,.54],font=8.5)
    figure(6,PERF/'controlled_primary_three_seed_AP.png','Controlled module and resolution comparisons. (A) Mobile at 1024: base, feature processing, structured identity, and their combination. (B) Mobile at 640. (C) Pano auxiliary supervision with factor inference fusion and geometric scoring disabled. (D) DSLR CMFA × optimizer comparison. Open symbols are the actual seeds; diamonds and bars are mean ± sample SD (n = 3). Thin lines connect corresponding base and combined-model seeds. Panels A/B use the same AP range; other domain ranges differ.')

    heading('4.5. Paired effects of the predefined cohort configurations')
    p('With center and shape scoring enabled on the three Pano joint-supervision checkpoints, DENT-STRIDE-Pano achieved 53.580 ± 0.646 AP. Relative to the corresponding controlled reference, mean AP differences were +0.558 for Pano, +0.296 for Mobile, and +0.412 for DSLR (Table 4). All three complete-configuration paired intervals included zero. By contrast, CMFA under AdamW increased AP in all three DSLR seeds, with a mean difference of +0.498 and an exploratory interval of [0.138, 0.859]. This component comparison provides clearer evidence than assuming that the selected optimizer and all modules are jointly optimal.')
    contrast_rows=[]
    for r in SELECT:
        contrast_rows.append([r['model'],f"{r['AP_mean']:.3f} ± {r['AP_SD']:.3f}",f"{r['AP_delta_mean']:+.3f} ± {r['AP_delta_SD']:.3f}",f"[{r['AP_delta_CI95_low']:.3f}, {r['AP_delta_CI95_high']:.3f}]"])
    cmfa=next(r for r in E['paired_contrasts'] if r['comparison']=='DSLR: CMFA+AdamW minus base+AdamW')['metrics']['AP']
    contrast_rows.append(['DSLR: CMFA + AdamW',pm(SUMMARY['dslr_cmfa_adamw']['metrics']['AP']),f"{cmfa['mean']:+.3f} ± {cmfa['SD']:.3f}",ci(cmfa)])
    table('Table 4. Paired AP comparisons with the corresponding controlled reference. Pano uses the combined prior; all comparisons pair seeds 0/1/2. Intervals use Student-t (df = 2), without multiplicity adjustment.', ['Configuration','AP mean ± SD','Paired ΔAP mean ± SD','Paired ΔAP 95% CI'],contrast_rows,widths=[1.82,1.50,1.80,1.53],font=9)

    heading('4.6. Framework-level detector comparisons')
    p('Table 5 places the cohort configurations alongside the available upstream and external detectors. Official DEIM-s and RT-DETRv2-S values summarize their separately trained official implementations; conventional-detector rows report available single-seed runs. The evaluation annotations and clinical label maps are shared, whereas resolution, pretraining, native optimization, and output rules differ across frameworks. These comparisons describe complete detector systems rather than isolated module effects. Protocol details and secondary metrics are provided in Supplementary Sections S4–S5.')
    table('Table 5. Framework-level AP comparison. Three-seed entries are mean ± SD; single-seed rows have no dispersion estimate. Mobile DENT-STRIDE uses 1024 inputs and the external references use 640. Pano DENT-STRIDE includes combined anatomical scoring. AP values are internal-validation estimates.', ['Method','Seeds','Pano AP','Mobile AP','DSLR AP'],[
        ['Controlled reference',3,pm(SUMMARY['pano_base']['metrics']['AP']),pm(SUMMARY['mobile_1024_base']['metrics']['AP']),pm(SUMMARY['dslr_base_adamw']['metrics']['AP'])],
        ['Official DEIM-s',3,'53.827 ± 0.425','53.553 ± 0.099','72.405 ± 0.202'],
        ['RT-DETRv2-S',3,'50.775 ± 0.213','50.562 ± 0.486','70.949 ± 0.506'],
        ['Faster R-CNN R50-FPN',1,'47.695','41.182','63.151'],
        ['CAF-YOLO-s',1,'48.058','43.771','67.460'],
        ['YOLOv8-s',1,'51.431','46.937','71.854'],
        ['YOLO11-s',1,'50.644','47.064','71.094'],
        ['RT-DETR-L',1,'51.978','54.751','74.019'],
        ['Dome-DETR-S',1,'51.711','51.828','71.363'],
        ['DENT-STRIDE',3,*[f"{r['AP_mean']:.3f} ± {r['AP_SD']:.3f}" for cohort in ['Pano','Mobile','DSLR'] for r in SELECT if r['cohort']==cohort]],
    ],widths=[2.0,.50,1.38,1.38,1.39],font=9)
    p('On DSLR images, RT-DETR-L reached 74.019 AP with 32.091 M parameters, exceeding the mean DENT-STRIDE DSLR AP of 72.878 with 11.154 M parameters. CMFA + AdamW reached 72.964 ± 0.098 AP at the same parameter count. The size comparison in Fig. 7 therefore supports a compact operating point, rather than unrestricted AP leadership. Parameter count is not a substitute for measured latency.')
    figure(7,PERF/'dslr_framework_accuracy_size_updated.png','DSLR accuracy versus model size, with a compact-model enlargement. Model names are attached to the points; RT-DETR-L is retained. Error bars show sample SD across seeds 0, 1, and 2 for the controlled reference, DENT-STRIDE-DSLR, CMFA + AdamW, Official DEIM-s, and RT-DETRv2-S. Remaining comparators retain single-seed values without error bars. No Pareto-frontier connector is imposed. Input and training differences follow the framework protocols described in Section 4.6.')

    heading('4.7. Inter-module interactions')
    p('In Mobile, feature-only and identity-only configurations achieved 54.405 ± 0.657 and 54.351 ± 0.245 AP, respectively, below the 54.730 ± 0.149 controlled reference. Combining them produced 55.026 ± 0.396 AP. The paired feature–identity interaction estimate was +1.001 points, but its interval [−2.092, 4.094] was broad; these data do not establish independent additive benefits. Fig. 8 shows both correction and deterioration in the same-image four-configuration comparison.')
    qual_caps=json.loads((QUAL/'figure_captions.json').read_text())
    def qcap(stem):
        x=qual_caps[stem]
        caption=x if isinstance(x,str) else x.get('caption',str(x))
        caption=re.sub(r'\(A\) Image \d+:', '(A)', caption)
        caption=re.sub(r'\(B\) Image \d+:', '(B)', caption)
        caption=caption.replace('Panel A uses image 111 and panel B image 56. ', '')
        caption=caption.replace('same-image mobile','same-image smartphone').replace('Same-image mobile','Same-image smartphone')
        caption=caption.replace('Same-image pano','Same-image panoramic').replace('Same-image dslr','Same-image DSLR')
        return caption
    figure(8,QUAL/'Mobile_inter_module_comparison.png',qcap('Mobile_inter_module_comparison'))
    p('In Pano, Factor only, Prototype only, and their combination achieved 53.282 ± 0.158, 53.036 ± 0.210, and 53.166 ± 0.526 AP with both inference corrections disabled. Factor-only supervision had the highest mean of these four training settings. In DSLR, CMFA increased mean AP by +0.498 under AdamW and +0.513 under DualOpt, whereas adding DualOpt to CMFA changed mean AP by −0.086. Thus, neither the identity pair nor the optimizer combination is uniformly superior. The separate supplementary historical feature family is retained as context, not pooled with these controlled estimates.')

    heading('4.8. Same-weight anatomical-prior ablation')
    p('Each of the three frozen Pano Factor + Prototype checkpoints was replayed with no prior, center only, shape only, or both (Table 6; Fig. 9). All 300 native query candidates and their complete class scores were retained before the top-k selection, so the comparison includes ranking changes that would be lost by reweighting already truncated predictions. Factor inference fusion remained disabled, and box coordinates were unchanged.')
    priorrows=[]
    for mode,label in [('none','None'),('center','Center only'),('shape','Shape only'),('both','Center + shape')]:
        r=PRIORS[mode];d=r['paired_AP_difference_vs_none'];priorrows.append([label]+[pm(r['metrics'][m]) for m in ['AP','AP50','AP75','AR100']]+[f"{d['mean']:+.3f}",ci(d)])
    table('Table 6. Anatomical scoring at the same Pano weights. Values are mean ± SD across the same three checkpoints; ΔAP and its 95% interval are paired with no prior.', ['Prior','AP','AP50','AP75','AR100','ΔAP','ΔAP 95% CI'],priorrows,widths=[1.0,1.02,1.02,1.02,1.02,.55,1.02],font=8.5)
    p('Center scoring increased mean AP from 53.166 to 53.619 (+0.453; interval [0.097, 0.809]); the combined prior reached 53.580 (+0.414; [0.021, 0.806]). Shape-only scoring changed AP by +0.028 [−0.151, 0.207]. The combined setting did not exceed center-only AP, although both center-based settings improved AR100. Fig. 10 demonstrates suppression of additional candidates as well as an adverse FDI change. These effects concern score ranking and thresholded output, rather than improved localization or learned appearance representations.')
    figure(9,PERF/'controlled_spatialshape_same_weight.png','Four-way Pano prior replay at identical weights. (A) COCO AP and (B) AR100. Each line connects one of the three checkpoints across no prior, center-only, shape-only, and combined scoring; filled diamonds and bars show mean ± sample SD. Scores are recomputed before native top-k selection. Factor inference fusion is disabled in all four settings.')
    figure(10,QUAL/'Pano_same_checkpoint_prior_comparison.png',qcap('Pano_same_checkpoint_prior_comparison'))

    heading('4.9. Module-internal and background-loss ablations')
    p('Table 7 reports seed-0 learning ablations and pure inference replays using the specified reference in each group. The primary Mobile background weights are held fixed when testing feature and identity modules. Factor-loss removals disable both training and inference fusion to isolate supervision; their reference is therefore the 54.023 AP fusion-off model. CMFA gate controls preserve the 512-channel concatenation, and HOOA controls preserve the context path and projection dimensions. Ordinary AdamW weight decay is unchanged in all DualOpt variants.')
    base=original_run('mobile_1024_full');base_ap=base['metrics_percent']['AP']
    factor_base=original_run('internal_mobile_factor_train_off');fa=factor_base['metrics_percent']['AP']
    dslr0=original_run('dslr_cmfa_adamw');da=dslr0['metrics_percent']['AP']
    groups=[('CMFA gates',base_ap,[('Both gates',None,base_ap),('Channel only','internal_mobile_cmfa_channel_only',None),('Spatial only','internal_mobile_cmfa_spatial_only',None),('Neither','internal_mobile_cmfa_no_gates',None)]),('HOOA branches',base_ap,[('Full module',None,base_ap),('Without high-pass','internal_mobile_hooa_no_highpass',None),('Without second-order','internal_mobile_hooa_no_second_order',None),('Without both','internal_mobile_hooa_neither',None)]),('Factor supervision',fa,[('All three factors',None,fa),('Without quadrant','internal_mobile_factor_no_quadrant',None),('Without position','internal_mobile_factor_no_position',None),('Without dentition','internal_mobile_factor_no_dentition',None)]),('Prototype design',base_ap,[('Four prototypes; τ = 0.20',None,base_ap),('No prototype','internal_mobile_proto_off',None),('One prototype; τ = 0.20','internal_mobile_proto_k1_tau0p2',None),('Two prototypes; τ = 0.20','internal_mobile_proto_k2_tau0p2',None),('Four prototypes; τ = 0.10','internal_mobile_proto_k4_tau0p1',None),('Four prototypes; τ = 0.50','internal_mobile_proto_k4_tau0p5',None)]),('DualOpt operations',da,[('Neither; base AdamW',None,da),('Rollback only','internal_dslr_rollback_only',None),('Extra decay only','internal_dslr_extra_decay_only',None),('Both operations',None,original_run('dslr_cmfa_dualopt')['metrics_percent']['AP'])]),('Mobile background losses',base_ap,[('NWD + arch-position',None,base_ap),('Neither','internal_mobile_background_neither',None),('NWD only','internal_mobile_background_nwd_only',None),('Arch-position only','internal_mobile_background_arch_only',None)])]
    internalrows=[]
    for group,reference,items in groups:
        for idx,(label,key,known) in enumerate(items):
            r=INTERNAL.get(key) if key else None
            ap=known if key is None else r.get('AP') if r and r['status']=='complete' else None
            internalrows.append([group if idx==0 else '',label,val(ap),'—' if ap is None else f'{ap-reference:+.3f}'])
    for idx,r in enumerate(E['factor_fusion_2x2']):
        name=('On' if r['training_fusion'] else 'Off')+' / '+('On' if r['inference_fusion'] else 'Off')
        internalrows.append(['Factor fusion' if idx==0 else '',name,val(r['AP']),f"{r['AP']-fa:+.3f}"])
    table('Table 7. Module-internal sensitivity on seed 0. ΔAP uses the intact Mobile model for CMFA/HOOA/prototype/background groups, the fusion-off all-factor model for factor supervision/fusion, and DSLR CMFA + AdamW for optimization. Fusion labels are training / inference. — indicates an unavailable result.', ['Component','Setting','AP','ΔAP vs group reference'],internalrows,widths=[1.57,2.83,.8,1.45],font=9)
    p('The channel-only CMFA setting reached 55.872 AP, compared with 54.583 for both gates. Removing prototype supervision reached 55.649 AP, and one prototype per class reached 55.386. Removing the HOOA high-pass branch reached 55.228; using only the arch-position background loss reached 55.249. Removing any one factor-loss term also exceeded the fusion-off all-factor reference. These outcomes indicate sensitivity and possible over-regularization under this seed, rather than evidence that every intact branch is beneficial (Fig. 11).')
    p('Factor fusion during training and inference yielded 54.583 AP, while disabling only inference fusion yielded 54.632 at the same weights. The corresponding fusion-off training checkpoint yielded 54.023 without inference fusion and 53.859 with it. Inference correction did not improve AP in either training state. The contrast between training states remains a single-seed learning comparison. Detailed secondary metrics and the remaining internal panels are provided in Supplementary Tables S8 and S11 and Fig. S6.')
    figure(11,PERF/'controlled_internal_ablation_AP_deltas_part1.png','Internal sensitivity of CMFA gates, factor supervision, prototype number, and DualOpt operations. Positive ΔAP means that the displayed alternative exceeds the stated group reference; it is not an improvement attributed to the intact module. Factor supervision uses fusion off in both phases. DualOpt controls retain ordinary AdamW base decay. All points are seed 0, without uncertainty bars; — has no numeric mark. Supplementary Fig. S6 provides HOOA, temperature, background-loss, and fusion panels.')

    heading('4.10. Controlled resolution comparison')
    resolutionrows=[]
    for name,label,size in [('mobile_640_base','Base',640),('mobile_640_full','Combined',640),('mobile_1024_base','Base',1024),('mobile_1024_full','Combined',1024)]:
        r=SUMMARY[name]['metrics'];resolutionrows.append([label,str(size)]+[pm(r[m]) for m in ['AP','AP75','F1','raw_duplicate_image_rate']])
    table('Table 8. Mobile resolution comparison with the same within-architecture training protocol. Values are three-seed mean ± SD. F1 uses score ≥0.30 and localization IoU ≥0.50; duplicate-image rates use all native predictions retained at score ≥0.30.', ['Architecture','Input','AP','AP75','F1','Duplicate images (%)'],resolutionrows,widths=[1.12,.55,1.18,1.18,1.18,1.44],font=9)
    p('Increasing Mobile input from 640 to 1024 changed mean AP by +1.247 for the base and +1.274 for the combined architecture. At a fixed resolution, the combined-model AP differences were +0.269 at 640 and +0.296 at 1024. Resolution therefore contributed more AP than the complete module combination. The 640 combined model nevertheless improved mean fixed-threshold F1 by +2.095 points relative to its 640 reference, showing that AP and one operating point need not rank configurations identically. The same-resolution primary comparisons avoid attributing the larger resolution effect to identity learning.')

    heading('4.11. Prediction-level identity and consistency')
    fixedrows=[]
    for cohort,name in [('Pano','pano_base'),('Mobile','mobile_1024_base'),('DSLR','dslr_base_adamw')]:
        base_metrics=SUMMARY[name]['metrics'];selected=next(r for r in SELECT if r['cohort']==cohort)
        fixedrows.append([cohort,'Base']+[pm(base_metrics[m]) for m in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']])
        fixedrows.append([cohort,'DENT-STRIDE']+[f"{selected[m+'_mean']:.3f} ± {selected[m+'_SD']:.3f}" for m in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']])
    table('Table 9. Fixed-threshold endpoints, three-seed mean ± SD, at score ≥0.30 and localization IoU ≥0.50. FDI accuracy is conditional on localized matches; recall and coverage use all reference teeth. Duplicates use all validation images. All entries are percentages.', ['Cohort','Model','Precision','Recall','F1','FDI accuracy','Coverage','Duplicate images'],fixedrows,widths=[.56,.9,.84,.84,.84,.9,.88,.89],font=8)
    p('For Pano, Mobile, and DSLR, the DENT-STRIDE configurations had mean F1 values of 88.078, 80.502, and 92.374%, respectively, and matched-tooth FDI accuracies of 93.100, 91.565, and 97.544%. Their raw-output duplicate-image rates were 74.479, 35.758, and 25.965%. The corresponding reference rates were 76.562, 36.364, and 23.684%. Duplicates therefore persisted across all domains, and the DSLR combined model did not improve this endpoint. These values demonstrate why conditional tooth identity and complete image-level numbering require separate evaluation.')
    p('Fig. 12 compares the same images across the controlled reference, an external detector, and the predefined cohort configuration. The displayed seed is fixed at 0, rather than selected for the largest aggregate AP. Each cohort includes an improvement and an adverse example; shared target enlargements retain the same image region and label rule. Quantitative conclusions use all validation images. Detection PR and threshold–F1/duplicate curves are provided in Supplementary Fig. S7.')
    p('Supplementary Figs. S9–S11 show complete class-wise FDI confusion matrices and training/validation support for the same seed-0 cohort configurations. Their missed-detection columns retain unlocalized reference teeth; unsupported validation rows are unavailable rather than assigned zero accuracy. Additional unmatched predictions are recorded separately because a reference-tooth confusion matrix does not represent their false-positive burden.')
    for stem,letter in [('Mobile_same_image_comparison','A'),('Pano_same_image_comparison','B'),('DSLR_same_image_comparison','C')]:
        figure('12'+letter,QUAL/(stem+'.png'),qcap(stem))

    heading('4.12. Learned identity structure and real internal responses')
    p('The Mobile checkpoint provides the prototype diagnostics in Fig. 13 and factor-head matrices in Fig. 14. Prototype centroids show class-dependent similarity, while within-class cohesion measures the mean pairwise cosine similarity of the four prototypes. The 18, 28, 38, and 48 slots have no Mobile training instances and are excluded from interpretation. High prototype cohesion can indicate redundant prototypes; it is not evidence that four prototypes improve detection. The retrained no-prototype result in Table 7 remains the relevant learning-effect comparison.')
    figure(13,MECH/'current_Mobile_prototype_diagnostics.png','Prototype diagnostics from the seed-0 Mobile EMA checkpoint used in the qualitative comparison. (A) Cosine similarity between normalized class-centroid prototypes, with classes ordered by FDI quadrant Q1–Q8. Grey unsupported slots correspond to 18, 28, 38, and 48. (B) Each open point is a supported FDI class and the short bar is the mean within its quadrant; cohesion is mean pairwise cosine similarity among four normalized prototypes. The bank is training-only supervision. Per-class cohesion is supplied in Supplementary Fig. S8.')
    confusion=json.loads((MECH/'factor_confusion_summary.json').read_text());matched=json.loads((MECH/'mechanism_provenance.json').read_text())['captured_queries_total']
    p(f'The factor analysis uses {matched:,} uniquely matched ordinary queries at maximum sigmoid score ≥0.30 and IoU ≥0.50. It does not duplicate query–class pairs from post-top-k predictions. Quadrant, position, and dentition accuracies are {100*confusion["quadrant"]["accuracy"]:.2f}, {100*confusion["position"]["accuracy"]:.2f}, and {100*confusion["dentition"]["accuracy"]:.2f}%, respectively. This separate representation-analysis denominator is not substituted for Table 9 localization counts.')
    figure(14,MECH/'current_Mobile_factor_confusions.png',f'Auxiliary quadrant, position, and dentition predictions for {matched:,} localized unique ordinary queries from the same Mobile seed-0 checkpoint. Rows are reference factors and columns are predictions; colours and annotations show row-normalized percentages. Grey rows with a dash have no reference observations. Quadrants retain the FDI first digit, position uses the second digit, and dentition is primary/permanent. These are structured identity heads, not objectness heads. The unique-query capture protocol differs from the native post-top-k detection endpoints.')
    p('Fig. 15 displays genuine tensors from the first top-down CMFA fusion and the highest-resolution HOOA branch on a Mobile improvement example. The channel-gate grids display all 256 channels; they are not anatomical maps. Spatial gates are shown on their actual [0,1] scale. Branch responses are channel root-mean-square maps, each normalized for display. The captured tensors reconstruct the executed module outputs, and adding the capture hooks left the evaluated predictions unchanged. Such responses describe the operations, while retrained ablations determine their performance effects.')
    figure(15,MECH/'actual_CMFA_HOOA_image_111.png','Actual Mobile forward-pass responses for the improvement example in Fig. 12A. The first row shows the image, CMFA channel gates g_A/g_B, and its spatial gate G_s; the second shows HOOA local, context, high-pass, and consumed second-order responses; the third shows HOOA channel/spatial gates and its input/output. Channel gates are arranged as 16 × 16 channel grids. Gates retain raw [0,1] values; response RMS maps are independently scaled by their 99th percentile and clipped to [0,1]. The maps are not Grad-CAM or a causal test of AP improvement.')

    heading('4.13. Cross-device and computational comparisons')
    p('Table 10 specifies bidirectional photographic comparisons of the controlled base and combined models using the common FDI52 mapping, frozen source weights, and no target fine-tuning. Numerical entries are reserved for evaluation of these exact checkpoints, separately from within-domain AP. Table 11 records instantiated parameter counts and reserves standardized runtime measurements for the same models. Unavailable measurements are denoted by — and are excluded from interpretation.')
    table('Table 10. Bidirectional evaluation of controlled photographic checkpoints. The source input policy and common FDI52 mapping are preserved. — indicates an unavailable measurement; no transfer conclusion is inferred.', ['Source model','Target cohort','AP','AP50','AP75','AR100'],[['Mobile base','DSLR','—','—','—','—'],['DENT-STRIDE-Mobile','DSLR','—','—','—','—'],['DSLR base','Mobile','—','—','—','—'],['DENT-STRIDE-DSLR','Mobile','—','—','—','—']],widths=[2.0,1.05,.9,.9,.9,.9],font=9)
    efficiency=[]
    for cohort,base_name,full_name in [('Pano','pano_base','pano_factor_prototype'),('Mobile','mobile_1024_base','mobile_1024_full'),('DSLR','dslr_base_adamw','dslr_cmfa_dualopt')]:
        for label,name in [('Base',base_name),('DENT-STRIDE',full_name)]:efficiency.append([cohort,label,f"{SUMMARY[name]['params_M']:.3f}",'—','—','—','—'])
    table('Table 11. Computational measurements for the controlled cohort configurations. Parameter counts include auxiliary training heads. Standardized batch-1 forward MACs, latency, throughput, and peak inference memory refer to the same GPU, precision, warm-up, and timing scope; — indicates an unavailable measurement.', ['Cohort','Model','Params (M)','GMACs','Latency (ms)','Images/s','Peak VRAM'],efficiency,widths=[.63,1.25,.85,.75,1.1,.95,1.12],font=9)
    # Keep headings and old unaffected clinical/reference content consistent in typeface.
    for pp in doc.paragraphs:
        if pp.style.name.startswith('Heading'):
            for run in pp.runs:text_format(run,12 if pp.style.name!='Heading 1' else 13,bold=True)
        # Use actual Word subscript formatting for inline gate symbols.
        pattern=r'(g_A|g_B|G_s|L_(?:det|MAL|L1|GIoU|FGL|DDF|NWD))'
        if re.search(pattern,pp.text) and not pp._element.xpath('.//m:oMath'):
            txt=pp.text;style=pp.style;pp.clear()
            for fragment in re.split(pattern,txt):
                if re.fullmatch(pattern,fragment):
                    lead,sub=fragment.split('_')
                    text_format(pp.add_run(lead),10 if style.name=='Caption' else 11)
                    run=pp.add_run(sub);text_format(run,10 if style.name=='Caption' else 11);run.font.subscript=True
                elif fragment:text_format(pp.add_run(fragment),10 if style.name=='Caption' else 11)
    # Clear obsolete image relationships as well as visible historical paragraphs.
    relation_namespace='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    used={value for element in doc._element.iter() for name,value in element.attrib.items() if name.startswith(relation_namespace)}
    for rid,relationship in list(doc.part.rels.items()):
        if relationship.reltype==RT.IMAGE and rid not in used:del doc.part.rels[rid]
    doc.save(OUT)
    receipt={'output':str(OUT),'source':str(SRC),'evidence_snapshot':E['snapshot_time_Asia_Shanghai'],'generated_time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'inserted_tables_and_figures':insertion_records,'preserved_method_equations':True,'pending_training_configurations':[r['configuration'] for r in E['pending_jobs']],'cross_device_and_full_efficiency':'Unavailable for new controlled checkpoints; em-dash rows are explicit placeholders.'}
    (ROOT/'main_manuscript_change_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('MAIN_MANUSCRIPT_WRITTEN',OUT)


if __name__=='__main__':main()
