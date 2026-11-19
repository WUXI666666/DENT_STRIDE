"""Create publication copies from the delivered manuscripts and verified results.

Selected seeds are explicitly post-hoc; complete five-seed observations and the
fixed 0/1/2 reference are retained. Source documents, PPT and results are immutable.
The SVGs are embedded with Office SVG relationships and high-resolution fallbacks.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re
import statistics
import zipfile

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.part import Part
from docx.opc.packuri import PackURI
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.text.paragraph import Paragraph
from lxml import etree

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_selected_seed_revision_2026-10-12'
QUANT = ROOT / 'quantitative'
PRED = ROOT / 'prediction'
ARCH = ROOT / 'architecture'
SOURCES = {k: PAPER / f'DENT_STRIDE_CMIG_{k}_reorganized_20261011.docx'
           for k in ['main', 'supplement']}
OUTPUTS = {k: PAPER / f'DENT_STRIDE_CMIG_{k}_selected_seed_revision_20261012.docx'
           for k in SOURCES}
METRICS = ['AP', 'AP50', 'AP75', 'AR100']
ENDPOINTS = ['Precision', 'Recall', 'F1', 'matched_tooth_FDI_accuracy',
             'localization_coverage', 'raw_duplicate_image_rate']
COHORTS = ['Pano', 'Mobile', 'DSLR']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def font(run, size=11, bold=False):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.bold = bold
    pr = run._r.get_or_add_rPr()
    rf = pr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        pr.insert(0, rf)
    for key in ['ascii', 'hAnsi', 'cs', 'eastAsia']:
        rf.set(qn('w:' + key), 'Times New Roman')


def set_text(p, text):
    assert not p._element.xpath('.//m:oMath'), 'Do not flatten native equations'
    p.clear()
    font(p.add_run(text), 10 if p.style.name == 'Caption' else 11,
         p.style.name.startswith('Heading'))
    p.paragraph_format.widow_control = True


def fmt(value, signed=False):
    return f'{value:+.3f}' if signed else f'{value:.3f}'


def ms(row, key, signed=False):
    return fmt(row[key + '_mean'], signed) + ' ± ' + fmt(row[key + '_SD'])


def map_by_cohort(rows):
    return {r['cohort']: r for r in rows}


def table_data(table):
    return [[c.text for c in r.cells] for r in table.rows]


def rewrite_table(table, rows, size=9):
    """Preserve placement but rebuild cells with consistent publication formatting."""
    for n in list(table._tbl):
        if n.tag == qn('w:tr'):
            table._tbl.remove(n)
    grid = table._tbl.tblGrid
    for n in list(grid):
        grid.remove(n)
    for _ in rows[0]:
        col = OxmlElement('w:gridCol')
        col.set(qn('w:w'), str(int(9400 / len(rows[0]))))
        grid.append(col)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, texts in enumerate(rows):
        r = table.add_row()
        if i == 0:
            h = OxmlElement('w:tblHeader')
            r._tr.get_or_add_trPr().append(h)
        no_split = OxmlElement('w:cantSplit')
        r._tr.get_or_add_trPr().append(no_split)
        for j, txt in enumerate(texts):
            c = r.cells[j]
            c.text = str(txt)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_together = True
                p.paragraph_format.keep_with_next = (i < len(rows) - 1) if len(rows) <= 5 else (i == 0)
                for run in p.runs:
                    font(run, size, i == 0)


def caption(doc, label):
    result = [p for p in doc.paragraphs if p.text.startswith(label + '. ')]
    assert len(result) == 1, (label, len(result))
    return result[0]


def strip_label(text):
    return re.sub(r'^(?:Figure|Fig\.)\s+[S\d]+[ABC]?\.\s*', '', text)


def replace_picture(doc, label, stem, new_caption):
    new_caption = new_caption.replace('with K=4 and temperature', 'with four prototypes per class and temperature')
    new_caption = new_caption.replace('(K = 4, τ = 0.50)', '(four prototypes per class; τ = 0.50)')
    p = caption(doc, label)
    element = p._element.getprevious()
    while element is not None and not element.xpath('.//w:drawing'):
        element = element.getprevious()
    assert element is not None, label
    ip = Paragraph(element, doc._body)
    ip.clear()
    ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ip.paragraph_format.keep_with_next = True
    ip.paragraph_format.space_before = Pt(4)
    ip.paragraph_format.space_after = Pt(3)
    png, svg = stem.with_suffix('.png'), stem.with_suffix('.svg')
    assert png.exists(), png
    shape = ip.add_run().add_picture(str(png), width=Inches(6.45))
    # Clinical SVGs contain large duplicate raster payloads. Their lossless
    # 600-dpi PNG is used in Word; standalone SVGs remain in the artwork package.
    embed_svg = svg.exists() and not (label == 'Fig. 10' or label.startswith('Fig. 12'))
    if embed_svg:
        # Office 2016+ chooses this SVG; legacy applications retain the PNG.
        part_name = PackURI('/word/media/' + stem.name + '.svg')
        vector_part = Part(part_name, 'image/svg+xml', svg.read_bytes(), doc.part.package)
        svg_rid = doc.part.relate_to(vector_part, RT.IMAGE)
        blip = shape._inline.graphic.graphicData.pic.blipFill.blip
        ext_list = OxmlElement('a:extLst')
        ext = OxmlElement('a:ext')
        ext.set('uri', '{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
        sblip = etree.SubElement(ext,
                    '{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip',
                    nsmap={'asvg': 'http://schemas.microsoft.com/office/drawing/2016/SVG/main'})
        sblip.set(qn('r:embed'), svg_rid)
        ext_list.append(ext)
        blip.append(ext_list)
    set_text(p, label + '. ' + strip_label(new_caption))
    p.paragraph_format.keep_with_next = False
    p.paragraph_format.keep_together = True
    return {'caption': label, 'png': str(png), 'png_sha256': sha(png),
            'svg': str(svg) if svg.exists() else None,
            'svg_embedded_in_word': embed_svg,
            'svg_sha256': sha(svg) if svg.exists() else None}


def drop_orphan_images(doc):
    used = set()
    ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    for n in doc._element.iter():
        for key, value in n.attrib.items():
            if key.startswith(ns):
                used.add(value)
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype == RT.IMAGE and rid not in used:
            doc.part.drop_rel(rid)


def publication_table_rules(table):
    """Three horizontal rules, without a heavy grid or shaded cells."""
    pr = table._tbl.tblPr
    old = pr.find(qn('w:tblBorders'))
    if old is not None:
        pr.remove(old)
    borders = OxmlElement('w:tblBorders')
    for name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement('w:' + name)
        border.set(qn('w:val'), 'single' if name in ['top', 'bottom'] else 'nil')
        border.set(qn('w:sz'), '6')
        border.set(qn('w:color'), '333333')
        borders.append(border)
    pr.append(borders)
    if len(table.rows) <= 16:
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = i < len(table.rows) - 1
    for cell in table.rows[0].cells:
        cp = cell._tc.get_or_add_tcPr()
        previous = cp.find(qn('w:tcBorders'))
        if previous is not None:
            cp.remove(previous)
        cb = OxmlElement('w:tcBorders')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '4')
        bottom.set(qn('w:color'), '333333')
        cb.append(bottom)
        cp.append(cb)


def change_tau(doc):
    p = doc.paragraphs[49]
    nodes = p._element.xpath('.//m:oMath/m:r')
    assert [mathtext(n) for n in nodes] == ['τ', '=', '0.20']
    for n in nodes[1:]:
        n.getparent().remove(n)


def mathtext(n):
    return ''.join(t.text or '' for t in n.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}t'))


def verify_arithmetic(data):
    for scope, rows in data['summaries'].items():
        for row in rows:
            raw = [r for r in data['per_seed'] if r['cohort'] == row['cohort']
                   and r['seed'] in row['seeds']]
            for m in METRICS:
                assert abs(statistics.mean(r[m] for r in raw) - row[m + '_mean']) < 1e-10
                assert abs(statistics.stdev(r[m] for r in raw) - row[m + '_SD']) < 1e-10
    for r in data['per_seed']:
        source = load(r['native_metric_path'])
        assert sha(r['native_metric_path']) == r['native_metric_sha256']
        assert r['epochs_completed'] == 160
    for r in data['prior_per_seed']:
        assert sha(r['native_metric_path']) == r['native_metric_sha256']


def update_main(doc, data):
    ps = doc.paragraphs
    s = map_by_cohort(data['summaries']['selected'])
    a = map_by_cohort(data['summaries']['all5'])
    e = map_by_cohort(data['fixed_threshold_summaries']['selected'])
    priors = {r['prior']: r for r in data['prior_summaries']['selected']}
    all_priors = {r['prior']: r for r in data['prior_summaries']['all5']}
    selected_ap = ', '.join(ms(s[c], 'AP') for c in COHORTS)
    texts = {
        4: 'Automatic tooth detection with Fédération Dentaire Internationale (FDI) numbering must accommodate panoramic radiographs and intraoral photographs while distinguishing localization, identity, and numbering consistency. We present DENT-STRIDE, an acquisition-configured DEIM–D-FINE framework with factorized identity supervision, multiscale feature processing, and a separate training-derived anatomical prior. Each domain configuration was trained for 160 epochs with five seeds. Exploratory three-seed subsets chosen after observing the results yielded validation AP of ' + selected_ap + ' for panoramic, smartphone, and DSLR images, respectively; all-five results are also reported. Across all five panoramic checkpoints, same-weight center scoring increased AP by ' + fmt(all_priors['center']['AP_delta_mean']) + ' points on average. In separately trained reference-configuration ablations, removing HOOA responses or factor-supervision targets reduced Mobile AP. Other auxiliary and optimization settings showed configuration-dependent effects. The selected-subset fixed-threshold F1 scores were ' + ', '.join(fmt(e[c]['F1_mean']) + '%' for c in COHORTS) + ', respectively. High matched-tooth identity accuracy coexisted with repeated raw FDI labels. These descriptive validation results distinguish tooth-level detection from complete clinical charting; post-hoc seed selection does not establish statistical superiority.',
        13: 'The contributions are threefold. First, we formulate explicit query-level FDI identity learning through quadrant, tooth-position, and dentition supervision, distinguishing auxiliary supervision from inference score fusion. Second, we introduce a separate tooth-specific anatomical scoring model derived from training-box center and shape statistics and isolate its effect through same-checkpoint comparisons. Third, we evaluate acquisition-specific detector configurations across radiographic and photographic cohorts, reporting all five training seeds, transparent exploratory seed-subset summaries, and retrained reference-configuration branch ablations that distinguish supported mechanisms from configuration sensitivity.',
        37: 'The configurations are DENT-STRIDE-Pano (Factor + Prototype with SpatialShape), DENT-STRIDE-Mobile (CMFA + HOOA with Factor + Prototype), and DENT-STRIDE-DSLR (CMFA with additional rank-dependent decay). Pano uses factor/prototype auxiliary supervision without factor-score fusion; Mobile uses both auxiliary supervision and factor-score fusion; DSLR uses no identity auxiliary branches or initialization rollback. The training and inference roles are summarized in Table 2 and defined below.',
        53: 'Both identity configurations use four prototypes per class. The temperature is 0.50 in Mobile and 0.20 in Pano; prototype-loss weights are 0.03 and 0.05, respectively. The loss averages over matched final-layer ordinary queries and is zero for an empty match set. Queries and prototypes both receive gradients, but prototype scores do not alter inference logits. Unsupported smartphone classes 18, 28, 38, and 48 are excluded from prototype interpretation. Reference-configuration Mobile component ablations use temperature 0.20 unless temperature itself is varied.',
        84: 'The DSLR configuration uses the additional-decay operation of DualOpt after the ordinary AdamW step (Ning et al., 2026). The extra-decay coefficient is 0.01, the parameter-enumeration rank exponent is 1.5, and multiplier bounds are 0.5–1.5; initialization rollback is disabled. This operation changes training optimization, not the inference architecture. Separate sensitivity controls compare neither operation, rollback only, extra decay only, and both, while retaining the same ordinary AdamW base weight decay.',
        93: 'Each acquisition-specific configuration was trained with seeds 0–4 for 160 epochs. Image partitions, clinical FDI mapping, model structure, initialization source, augmentation policy, losses, and checkpoint selection were held fixed within a configuration. HGNetv2-B0 used ImageNet-pretrained backbone weights. Three-seed summaries use Pano seeds 1/3/4 and Mobile/DSLR seeds 0/2/3, chosen after the five outcomes were observed; they are exploratory selected subsets, not prespecified repetitions. All five runs and the fixed 0/1/2 reference are reported. Twenty-one additional seed-0 component-learning models use reference configurations: Mobile temperature 0.20 and, for the intact DSLR optimization reference, both additional decay and rollback. Pure inference switches reuse their corresponding weights.',
        96: 'Within-domain component ablations preserve their group reference’s resolution, data, backbone initialization, augmentation, background losses, training budget, evaluator, and selection policy except for the stated variable. Mobile reference ablations use prototype temperature 0.20, whereas the domain configuration evaluated across five seeds uses 0.50. CMFA gate switches preserve output dimensions; HOOA branch switches preserve local/context processing and output projection. Factor-supervision contrasts disable factor fusion during training and inference. These controlled estimates apply to their stated reference configurations, not to every acquisition-specific configuration.',
        100: 'Training variability is described by mean ± sample standard deviation. Selected-three summaries are post-hoc and are not used for significance testing; complete five-seed results and fixed 0/1/2 summaries are provided for context. Same-checkpoint prior differences are paired within seeds. Exploratory Student-t 95% intervals in the supplement use all five paired AP differences (df = 4), without multiplicity adjustment, and are conditional on validation-selected weights. They describe training-seed variability, not patient-level sampling uncertainty. Single-seed component-learning controls have no uncertainty bars. Cross-framework values are descriptive system comparisons, not paired architecture effects.',
        102: '4.2.1. Detection performance and exploratory seed-subset summaries',
        103: 'The post-hoc selected subsets yielded AP of ' + ms(s['Pano'], 'AP') + ', ' + ms(s['Mobile'], 'AP') + ', and ' + ms(s['DSLR'], 'AP') + ' for Pano, Mobile, and DSLR, respectively (Table 3; Fig. 6). Pano includes combined center–shape scoring, Mobile uses four prototypes at temperature 0.50, and DSLR uses CMFA with extra decay but no rollback. Their AP75 values were ' + ', '.join(ms(s[c], 'AP75') for c in COHORTS) + '. The complete five-seed AP summaries are ' + ', '.join(ms(a[c], 'AP') for c in COHORTS) + ', in the same order. Selected subsets contain different seed indices across cohorts and should not be interpreted as unbiased estimates from prespecified repetitions.',
        104: 'Table 3. Exploratory selected-three DENT-STRIDE validation results, mean ± sample SD. Pano uses seeds 1/3/4; Mobile and DSLR use seeds 0/2/3. These subsets were chosen after observing five completed runs. Metrics are percentages and parameters are millions. All-five AP appears in Table 4; complete four-metric summaries are in Supplementary Table S10.',
        108: 'Across all five runs, AP ranged from 53.517 to 54.368 for Pano, 55.008 to 55.650 for Mobile, and 72.530 to 73.403 for DSLR (Table 4). The ranges describe the observed spread under each fixed configuration, while the selected-three subsets preferentially retain higher outcomes. Individual-run secondary metrics and checkpoint epochs are provided in Supplementary Table S1; complete selected-three, fixed-0/1/2, and all-five summaries are in Supplementary Table S10. Supplementary Fig. S8 shows the actual learning trajectories.',
        109: 'Table 4. Selected-subset and complete five-seed AP summaries. The selected subsets are post-hoc; all-five summaries include every completed run. Values are percentage points and SD uses the n−1 denominator.',
        111: 'Table 5 compares the acquisition-specific configurations with official DEIM-s, RT-DETRv2-S, and conventional detector implementations on the same cohort annotations and clinical label maps. Official DEIM-s and RT-DETRv2-S summarize their fixed seeds 0/1/2; other external rows are single-seed results. DENT-STRIDE selected-three entries are post-hoc, with all-five AP shown alongside. Frameworks retain native pretraining, optimization, augmentation, and post-processing. Mobile DENT-STRIDE uses 1024 × 1024 inputs, whereas external Mobile references use 640 × 640. These are complete-system comparisons, not isolated module or resolution effects. External comparators include CAF-YOLO-s (Chen and Lu, 2025), Dome-DETR-S (Hu et al., 2025), and the Ultralytics YOLO implementations (Jocher et al., 2023).',
        112: 'Table 5. Framework-level validation AP comparison. Repeated-seed entries are mean ± sample SD; single-seed entries have no dispersion estimate. DENT-STRIDE selected-three values are post-hoc, unlike the fixed 0/1/2 official controls. The all-five DENT-STRIDE row reports all completed runs. Framework-specific inputs and training/output settings differ.',
        113: 'DENT-STRIDE-Mobile yielded ' + fmt(s['Mobile']['AP_mean']) + ' AP in the selected subset and ' + fmt(a['Mobile']['AP_mean']) + ' across all five runs, compared with 54.751 for the single-seed RT-DETR-L reference (Fig. 7). Corresponding Pano AP was ' + fmt(s['Pano']['AP_mean']) + ' and ' + fmt(a['Pano']['AP_mean']) + ', versus 53.827 ± 0.425 for official DEIM-s. DSLR yielded ' + fmt(s['DSLR']['AP_mean']) + ' selected-subset AP and ' + fmt(a['DSLR']['AP_mean']) + ' all-five AP, above the official DEIM-s mean of 72.405 but below the single-seed RT-DETR-L value of 74.019. These descriptive rankings, including post-hoc subset selection, do not establish statistical superiority or a universal domain-independent optimum.',
        118: 'Reference-configuration HOOA ablations test local-detail and context interaction in Mobile with prototype temperature 0.20. From 54.910 AP, removing the high-pass branch, second-order interaction, or both reduced AP to 54.511, 53.943, and 53.561. With factor fusion disabled in both phases, removing quadrant, tooth-position, or dentition supervision reduced AP from 55.537 to 54.884, 54.263, and 55.293. These retrained seed-0 contrasts support the tested branches under their controlled reference protocol; they do not quantify repeated-seed branch effects at temperature 0.50. Fig. 8 provides corresponding same-image examples, and Table 7 reports the aggregate comparisons.',
        123: 'Table 6. Same-checkpoint Pano anatomical scoring for post-hoc selected seeds 1/3/4. Metrics and paired AP differences versus no prior are mean ± sample SD. Checkpoints were selected with combined scoring; alternative scoring does not reselect weights. Complete five-seed results are in Supplementary Tables S3–S4.',
        124: 'In the selected subset, center scoring increased AP from ' + fmt(priors['none']['AP_mean']) + ' to ' + fmt(priors['center']['AP_mean']) + ' (' + ms(priors['center'], 'AP_delta', True) + ' points); combined scoring reached ' + fmt(priors['both']['AP_mean']) + ' (' + ms(priors['both'], 'AP_delta', True) + '). Shape-only scoring changed AP by ' + ms(priors['shape'], 'AP_delta', True) + '. Across all five checkpoints, corresponding mean changes were ' + ', '.join(fmt(all_priors[k]['AP_delta_mean'], True) for k in ['center', 'shape', 'both']) + ' points. Selected-subset AR100 was ' + fmt(priors['center']['AR100_mean']) + ' for center scoring and ' + fmt(priors['both']['AR100_mean']) + ' for combined scoring, compared with ' + fmt(priors['none']['AR100_mean']) + ' without a prior. Center-only scoring had the highest mean AP; combined scoring had the highest AR100, while their mean AP75 values were nearly identical. Fig. 10 shows favourable and adverse identity changes. These conditional comparisons concern inference-score effects rather than learned appearance gains.',
        130: 'Table 7. Reference-configuration seed-0 feature-gating, HOOA-branch, and factor-supervision ablations. Mobile uses prototype temperature 0.20. ΔAP subtracts the stated group reference; factor-supervision models disable training and inference fusion. Learning controls are separately trained except reused intact references. These are not ablations of the temperature-0.50 five-seed configuration.',
        131: 'Fig. 11 summarizes reference-configuration HOOA and factor-supervision contrasts. Channel-only CMFA reached 55.422 AP, compared with 54.910 for both gates, so additional gates were not uniformly beneficial. Supplementary Tables S2 and S5 report prototype design, optimization, background-loss, and factor-fusion controls. Four prototypes at temperature 0.50 and DSLR extra decay without rollback were carried forward from these single-seed sensitivity comparisons into the five-seed domain configurations; their later complete-system results do not establish isolated component gains.',
        136: 'Table 8. Fixed-threshold endpoints for the post-hoc selected three-seed subsets in Table 3, mean ± sample SD, at score ≥0.30 and localization IoU ≥0.50. Values are percentages. FDI accuracy uses localization matches; recall and coverage use reference teeth; duplicate-image rate uses all images. These descriptive results are not independent-test or clinical charting estimates.',
        137: 'Selected-subset mean F1 was ' + ', '.join(fmt(e[c]['F1_mean']) + '%' for c in COHORTS) + ' for Pano, Mobile, and DSLR, respectively. Matched-tooth FDI accuracy was ' + ', '.join(fmt(e[c]['matched_tooth_FDI_accuracy_mean']) + '%' for c in COHORTS) + ', whereas raw-query duplicate-image rates were ' + ', '.join(fmt(e[c]['raw_duplicate_image_rate_mean']) + '%' for c in COHORTS) + '. High conditional identity accuracy therefore coexists with repeated labels at image level. The selected subsets optimize neither this fixed-threshold F1 nor global chart consistency; higher AP does not imply improvement in every secondary endpoint.',
        138: 'Fig. 12 compares the same images across official DEIM-s, RT-DETR-L, and DENT-STRIDE with consistent thresholds and shared local enlargements. The representative DENT-STRIDE weights maximize validation AP within the stated subsets: Pano seed 3, Mobile seed 2, and DSLR seed 3. Each domain includes improvement and adverse examples; these selected cases do not estimate their frequencies. Aggregate endpoints use every validation image. Detection PR, threshold–F1, threshold–duplicate curves, complete FDI outcomes, and class support are in Supplementary Figs. S4–S7.',
        146: 'Fig. 13 displays native CMFA and HOOA tensors from an FP32 CPU forward pass of the Mobile seed-2 checkpoint for the improvement image in Fig. 12A. CMFA gA/gB are channel gates, not anatomical spatial maps. Local, context, high-pass, and second-order HOOA responses summarize channel-wise root-mean-square magnitude. Gates retain their [0,1] scale; response maps are independently normalized for display. These maps describe executed operations, not causal performance evidence. Retrained reference-configuration comparisons in Table 7 provide separate branch-level evidence. Prototype and factor diagnostics from the same Mobile weights are in Supplementary Figs. S9–S11.',
        149: 'Table 9. Model size and tool-counted native-forward arithmetic for the domain configurations. Measurements use calflops 0.3.2, batch 1, FP32, and evaluation mode, including auxiliary outputs. The temperature change and removal of training-only rollback do not change these forward graphs. Post-processing and unsupported operations are excluded.',
        153: 'DENT-STRIDE separates identity learning, feature refinement, and anatomical score correction within an acquisition-configured detector. Five completed runs characterize each configuration’s observed seed variability; the selected-three subsets are explicitly exploratory. Separately trained temperature-0.20 reference controls support HOOA high-pass/second-order responses and factor targets, rather than attributing their effects to the updated temperature-0.50 configuration. Center scoring improved Pano AP at all five checkpoints; shape-only effects were small and positive, while joint scoring had the highest AR100. These findings support particular mechanisms and configuration choices, not a universal benefit from every additional module.',
        154: 'Earlier label-tree and prior-knowledge systems encode tooth identity and anatomy (Zhang et al., 2018; Ali et al., 2023). DENT-STRIDE places factor-score fusion and training-derived geometry within a common query detector and separates their training and inference roles. Center and shape templates do not change learned query representations or predicted boxes. In the representative Mobile bank, the four within-class prototypes are almost coincident; their high cosine cohesion is therefore not evidence of four distinct morphological modes. Prototype supervision and optimization choices are defined configuration components, without an independent accuracy benefit beyond what the corresponding controls support.',
        159: 'All estimates use validation partitions for checkpoint selection rather than independent-test generalization. Three-seed subsets were chosen after observing five outcomes, which can raise their means and narrow their observed spread; all-five and fixed-0/1/2 results are retained to expose this selection. DENTEX and smartphone content overlap and unavailable patient identifiers limit independence; DSLR partitions are patient-disjoint but still serve model selection. Seed-based intervals do not quantify patient-level uncertainty. Component-learning ablations use one seed at their reference configurations, and prior comparisons are conditional on weights selected with combined scoring.',
        160: 'External systems retain framework-specific pretraining, input policies, optimization, and post-processing. Their descriptive rankings do not establish fully matched architectural effects or statistical state-of-the-art superiority. RT-DETR-L exceeds both the selected-three and all-five DSLR AP means under its own protocol. Unsupported prototype slots cannot be interpreted as learned class evidence, and nearly coincident same-class prototypes do not establish multimodal representation. No clinician reader or confidence-calibration study was performed.',
        162: 'DENT-STRIDE combines explicit FDI identity supervision, acquisition-specific feature processing, and separate training-derived anatomical scoring. Exploratory selected-subset AP is ' + selected_ap + ' on panoramic, smartphone, and DSLR images, respectively; complete five-seed AP is ' + ', '.join(ms(a[c], 'AP') for c in COHORTS) + '. Reference-configuration HOOA/factor controls and same-weight center-prior comparisons provide mechanism-specific evidence. Post-hoc subset selection and validation-based model selection constrain generalization claims, while persistent duplicate labels show that tooth-level detection and image-level numbering consistency require separate evaluation.',
    }
    for i, text in texts.items():
        set_text(ps[i], text)
    change_tau(doc)
    # Retain the controlled historical prediction and ablation plots truthfully.
    set_text(ps[120], ps[120].text.replace('HOOA branch comparison on the same smartphone images.',
             'HOOA branch comparison on the same smartphone images under the temperature-0.20 Mobile reference configuration.'))
    set_text(ps[133], ps[133].text.replace('Retrained HOOA and factor-supervision ablations.',
             'Retrained HOOA and factor-supervision ablations under the temperature-0.20 Mobile reference configuration.').replace('full Mobile seed-0 configuration', 'intact Mobile seed-0 reference'))

    rows = table_data(doc.tables[1])
    rows[6][3] = 'AdamW + rank-dependent extra decay; no rollback'
    rows.append(['Prototype temperature', '0.20', '0.50', 'Inactive'])
    rewrite_table(doc.tables[1], rows, 9)
    rewrite_table(doc.tables[2], [['Cohort', 'Configuration', *METRICS, 'Params (M)']] +
                  [[c, 'DENT-STRIDE-' + c, *[ms(s[c], m) for m in METRICS],
                    fmt(s[c]['params_trainable'] / 1e6)] for c in COHORTS], 8.5)
    rows = [['Configuration', 'Selected seeds', 'Selected AP ± SD', 'All-five AP ± SD', 'All-five AP range']]
    for c in COHORTS:
        vs = [r['AP'] for r in data['per_seed'] if r['cohort'] == c]
        rows.append(['DENT-STRIDE-' + c, '/'.join(map(str, s[c]['seeds'])), ms(s[c], 'AP'),
                     ms(a[c], 'AP'), fmt(min(vs)) + '–' + fmt(max(vs))])
    rewrite_table(doc.tables[3], rows, 9)
    rows = table_data(doc.tables[4])
    rows[-1] = ['DENT-STRIDE, selected subset', '3 selected', *[ms(s[c], 'AP') for c in COHORTS]]
    rows.append(['DENT-STRIDE, all runs', '5', *[ms(a[c], 'AP') for c in COHORTS]])
    rewrite_table(doc.tables[4], rows, 9)
    rows = [['Prior', *METRICS, 'Paired ΔAP ± SD']]
    names = {'none': 'None', 'center': 'Center only', 'shape': 'Shape only', 'both': 'Center + shape'}
    for key, row in priors.items():
        rows.append([names[key], *[ms(row, m) for m in METRICS],
                     '—' if key == 'none' else ms(row, 'AP_delta', True)])
    rewrite_table(doc.tables[5], rows, 8.5)
    rewrite_table(doc.tables[7], [['Cohort', 'Precision', 'Recall', 'F1', 'Matched FDI', 'Coverage', 'Duplicate images']]
                  + [[c, *[ms(e[c], m) for m in ENDPOINTS]] for c in COHORTS], 8.5)

    caps = load(QUANT / 'captions.json')
    pcaps = load(PRED / 'figure_captions.json')
    replacements = [
        ('Fig. 2', ARCH / 'Figure_2_DENT_STRIDE_architecture', (ARCH / 'Figure_2_caption.txt').read_text().strip()
         + ' Vector-layout preparation used OpenAI ChatGPT/Codex assistance (OpenAI; October 2026).'),
        ('Fig. 6', QUANT / 'Figure_6_training_seed_variation', caps['Figure_6_training_seed_variation']),
        ('Fig. 7', QUANT / 'Figure_7_Mobile_framework_AP', caps['Figure_7_Mobile_framework_AP']),
        ('Fig. 9', QUANT / 'Figure_9_Pano_prior_comparison', caps['Figure_9_Pano_prior_comparison']),
        ('Fig. 10', PRED / 'Figure_10_Pano_same_weight_prior', pcaps['Figure_10_Pano_same_weight_prior']),
        ('Fig. 12A', PRED / 'Figure_12A_Mobile_same_image', pcaps['Figure_12A_Mobile_same_image']),
        ('Fig. 12B', PRED / 'Figure_12B_Pano_same_image', pcaps['Figure_12B_Pano_same_image']),
        ('Fig. 12C', PRED / 'Figure_12C_DSLR_same_image', pcaps['Figure_12C_DSLR_same_image']),
        ('Fig. 13', PRED / 'actual_CMFA_HOOA_image_182', pcaps['actual_CMFA_HOOA_image_182']),
    ]
    manifest = [replace_picture(doc, *r) for r in replacements]
    for number in ['4', '5']:
        p = caption(doc, 'Fig. ' + number)
        set_text(p, p.text + ' Vector-layout preparation used OpenAI ChatGPT/Codex assistance (OpenAI; October 2026).')
    # Reproducible plotting assistance is disclosed without altering clinical data.
    plot_p = doc.add_paragraph(style='Normal')
    font(plot_p.add_run('Quantitative figures are generated from recorded metrics and native predictions using reproducible plotting scripts; clinical panels overlay annotations on the original image pixels, and forward-response maps use captured module tensors. OpenAI ChatGPT/Codex (OpenAI; October 2026 session) assisted analysis-script and vector-layout preparation. Outputs were checked against raw results, and no generative image model produced or altered clinical images or experimental data.'))
    plot_p.paragraph_format.widow_control = True
    ps[100]._element.addnext(plot_p._element)
    return manifest


def update_supplement(doc, data):
    ps = doc.paragraphs
    texts = {
        4: 'Each domain configuration was trained for 160 epochs with seeds 0–4. The selected EMA checkpoint maximizes validation COCO AP under its configured inference policy. Pano selection includes center and shape scoring; Mobile includes factor fusion; DSLR uses neither score correction. Original image partitions are unchanged. Main selected-three summaries use Pano 1/3/4 and Mobile/DSLR 0/2/3, chosen after observing all five outcomes. All 15 runs are listed below; fixed-0/1/2 and all-five summaries appear in Table S10. The selected subsets are exploratory, not prespecified repetition sets.',
        5: 'Table S1. All 15 domain-configuration runs, including selected-subset membership. Metrics are percentage points and epochs are one-based. Params are 10.650 M (Pano), 12.513 M (Mobile), and 11.154 M (DSLR).',
        10: 'Twenty-one additional models were trained with seed 0 under component-group reference configurations. Mobile controls use four prototypes at temperature 0.20 unless count or temperature is varied; the intact DSLR optimization reference uses both rollback and extra decay. Each learning contrast changes only its stated component. CMFA switches retain 512 output channels. High-pass removal zeros that response; interaction removal substitutes direct context. Factor-supervision removals retain all heads and disable fusion in training and inference. Ordinary AdamW base decay remains fixed. Background controls change NWD/arch-position terms. These controls are distinct from the temperature-0.50 Mobile and extra-decay-only DSLR five-seed configurations.',
        11: 'Table S2. Complete seed-0 component sensitivity under the specified reference configurations. Positive ΔAP means an alternative exceeds its reference. These learning effects apply to the stated group references.',
        12: 'The Mobile reference is 54.910 AP for gates, HOOA, prototypes, and background losses; factor targets use the all-factor fusion-off reference at 55.537. DSLR optimization uses CMFA plus ordinary AdamW at 72.892. Channel-only CMFA reached 55.422 AP; four prototypes at temperature 0.50 reached 55.683; additional decay alone reached 73.351 on DSLR; NWD alone reached 55.396 on Mobile. Removing prototype supervision changed AP by +0.008. The temperature-0.50 and extra-decay-only choices were carried forward into five-seed complete-system evaluation, but these single-seed sensitivity contrasts do not establish stable isolated-module superiority.',
        16: 'Table S3. All five Pano checkpoints under four anatomical-score settings. Weights and box coordinates remain fixed within each seed; full candidates are rescored before native top-k selection. ΔAP is paired against no prior.',
        17: 'The five Pano checkpoints were selected under combined center–shape scoring. No-prior, center-only, and shape-only outputs evaluate those same weights without reselection. Factor fusion, appearance scoring, and global assignment are disabled. The panoramic view term is constant over supported classes. Center/log-shape weights are 0.20/0.05. Main selected-subset summaries are conditional on post-hoc seed selection; this table retains all five paired comparisons.',
        18: 'Table S4. Complete five-seed paired AP differences versus no prior. Mean ± sample SD uses seeds 0–4. Exploratory Student-t 95% intervals use df = 4 and are conditional on validation-selected weights; they are not selected-subset or patient-sampling intervals.',
        22: 'Table S5. Reference Mobile (temperature 0.20) training-fusion × inference-fusion comparison at seed 0. Each inference pair reuses its training-state checkpoint.',
        23: 'These reference-configuration training states retain factor and prototype supervision at temperature 0.20. Quadrant, tooth-position, and dentition score-fusion weights are 0.10, 0.05, and 0.03. Fusion slightly reduces AP in both training states while increasing fixed-threshold F1. Training-state differences require separate training; inference on/off is a same-weight score change. F1 uses score ≥0.30 and IoU ≥0.50. These results are not a fusion ablation at temperature 0.50.',
        25: 'Table S6. Fixed-threshold endpoints for all 15 domain-configuration runs (%). Main Table 8 summarizes only the declared post-hoc subsets.',
        42: 'AdamW uses detector/backbone learning rates of 5 × 10⁻⁴ / 2.5 × 10⁻⁴, betas (0.9, 0.999), and base decay 10⁻⁴, with normalization-parameter decay exclusions. Final DSLR training adds rank-dependent decay 0.01, exponent 1.5, and multipliers [0.5, 1.5], with rollback disabled. The separate optimization sensitivity controls also evaluate rollback 0.005 and both operations. The common schedule uses warm-up, flat/cosine phases, clipping 0.10, EMA decay 0.9999, and terminal augmentation shutdown. Pano MixUp is active only in its scheduled interval. Earlier checkpoints win validation-AP ties; settings remain fixed across each configuration’s five seeds.',
        46: 'The representative Mobile seed-2 checkpoint supplies the prototype, auxiliary-head, and native-response panels. Centroid similarity averages four normalized class prototypes and renormalizes. Within-class cohesion is the mean of their six pairwise cosine similarities. Unsupported slots 18, 28, 38, and 48 are masked. The within-class values are almost one, indicating nearly coincident prototypes rather than four distinct morphological subtypes. These are representation diagnostics, not evidence of an independent AP gain.',
        55: 'Same-image panels share thresholds, clinical FDI decoding, and target enlargements. Representative weights maximize validation AP within the declared selected subsets: Pano 3, Mobile 2, and DSLR 3. Component panels retain their separately trained temperature-0.20 reference weights; prior panels switch scoring terms for the same Pano seed-3 checkpoint before top-k selection. Each domain includes an improvement and an adverse case. All validation images determine aggregate metrics; selected examples do not estimate event frequencies. Seed plots retain all five observations and show selected-subset/all-five sample SD as labelled. Single-run component contrasts have no artificial uncertainty bars.',
    }
    for i, text in texts.items():
        set_text(ps[i], text)
    set_text(ps[14], ps[14].text + ' Mobile component references use temperature 0.20; DSLR optimization references distinguish rollback from extra decay.')
    rows = [['Cohort', 'Seed', 'Selected', *METRICS, 'Best epoch']]
    for r in data['per_seed']:
        rows.append([r['cohort'], r['seed'], 'Yes' if r['selected'] else 'No',
                     *[fmt(r[m]) for m in METRICS], r['best_epoch_one_based']])
    rewrite_table(doc.tables[0], rows, 8.5)
    names = {'none': 'None', 'center': 'Center only', 'shape': 'Shape only', 'both': 'Center + shape'}
    rows = [['Seed', 'Prior', *METRICS, 'ΔAP']]
    prior_off = {r['seed']: r['AP'] for r in data['prior_per_seed'] if r['prior'] == 'none'}
    for r in data['prior_per_seed']:
        rows.append([r['seed'], names[r['prior']], *[fmt(r[m]) for m in METRICS], fmt(r['AP'] - prior_off[r['seed']], True)])
    rewrite_table(doc.tables[2], rows, 8.5)
    rows = [['Prior', 'Seed 0', 'Seed 1', 'Seed 2', 'Seed 3', 'Seed 4', 'Mean ± SD', '95% CI']]
    for r in data['prior_summaries']['all5']:
        if r['prior'] == 'none':
            continue
        rows.append([names[r['prior']], *[fmt(v, True) for v in r['AP_paired_delta_vs_none']],
                     ms(r, 'AP_delta', True), '[' + fmt(r['AP_delta_nominal_CI95_low']) + ', '
                     + fmt(r['AP_delta_nominal_CI95_high']) + ']'])
    rewrite_table(doc.tables[3], rows, 8)
    rows = [['Cohort', 'Seed', 'P', 'R', 'F1', 'Matched FDI', 'Coverage', 'Duplicate images']]
    for r in data['per_seed']:
        rows.append([r['cohort'], r['seed'], *[fmt(r['fixed_threshold'][m] * 100) for m in ENDPOINTS]])
    rewrite_table(doc.tables[5], rows, 8.5)
    rows = table_data(doc.tables[8])
    rows[7][2] = '4; 0.50; 0.03'
    rows[13][3] = 'AdamW + rank-dependent extra decay; no rollback'
    rewrite_table(doc.tables[8], rows, 9)

    caps = load(QUANT / 'captions.json')
    pcaps = load(PRED / 'figure_captions.json')
    entries = [
        ('Fig. S3', QUANT / 'Figure_S3_prior_paired_AP', caps['Figure_S3_prior_paired_AP']),
        ('Fig. S4', QUANT / 'Figure_S4_detection_curves', caps['Figure_S4_detection_curves']),
        ('Fig. S5', QUANT / 'Fig_S5_Pano_complete_FDI_confusion', caps['Fig_S5_Pano_complete_FDI_confusion']),
        ('Fig. S6', QUANT / 'Fig_S6_Mobile_complete_FDI_confusion', caps['Fig_S6_Mobile_complete_FDI_confusion']),
        ('Fig. S7', QUANT / 'Fig_S7_DSLR_complete_FDI_confusion', caps['Fig_S7_DSLR_complete_FDI_confusion']),
        ('Fig. S8', QUANT / 'Figure_S8_learning_curves', caps['Figure_S8_learning_curves']),
        ('Fig. S9', PRED / 'current_Mobile_prototype_diagnostics', pcaps['current_Mobile_prototype_diagnostics']),
        ('Fig. S10', PRED / 'current_Mobile_cohesion_all_FDI', pcaps['current_Mobile_cohesion_all_FDI']),
        ('Fig. S11', PRED / 'current_Mobile_factor_confusions', pcaps['current_Mobile_factor_confusions']),
    ]
    manifest = [replace_picture(doc, *r) for r in entries]
    # Factor-query denominators come from a fresh native CPU forward pass.
    factor = load(PRED / 'factor_matched_queries.json')
    accuracy = [100 * sum(r[k + '_true'] == r[k + '_pred'] for r in factor) / len(factor)
                for k in ['quadrant', 'position', 'dentition']]
    set_text(ps[51], 'Auxiliary matrices use ' + f'{len(factor):,}' + ' uniquely matched ordinary queries across all 220 validation images. Eligibility requires maximum sigmoid class score ≥0.30 and class-agnostic Hungarian localization IoU ≥0.50. This denominator differs from post-top-300 detector outputs, which may retain multiple query–class pairs. Quadrant, tooth-position, and dentition accuracies are ' + ', '.join(f'{v:.2f}%' for v in accuracy) + ', respectively. The tensors come from CPU FP32 forward passes of the representative Mobile seed-2 checkpoint; targets follow the clinical FDI map.')
    p = doc.add_paragraph('S9. Complete seed-set summaries', style='Heading 1')
    for r in p.runs:
        font(r, 12, True)
    p.paragraph_format.keep_with_next = True
    p = doc.add_paragraph(style='Caption')
    set_text(p, 'Table S10. Four-metric summaries for all five runs, fixed seeds 0/1/2, and the post-hoc selected subsets. Values are mean ± sample SD in percentage points. Selected subsets are Pano 1/3/4 and Mobile/DSLR 0/2/3; they are not prespecified evidence.')
    p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=6)
    t.style = doc.tables[0].style
    rows = [['Cohort', 'Seed set', *METRICS]]
    for scope, label in [('all5', 'All 0–4'), ('fixed012', 'Fixed 0/1/2'), ('selected', 'Post-hoc selected')]:
        for r in data['summaries'][scope]:
            rows.append([r['cohort'], label, *[ms(r, m) for m in METRICS]])
    rewrite_table(t, rows, 8.5)
    return manifest


def image_hashes(doc):
    return [hashlib.sha256(doc.part.related_parts[
        x._inline.graphic.graphicData.pic.blipFill.blip.embed].blob).hexdigest()
        for x in doc.inline_shapes]


def validate(original, doc, kind):
    oldmath, newmath = original._element.xpath('.//m:oMath'), doc._element.xpath('.//m:oMath')
    assert len(oldmath) == len(newmath) == (51 if kind == 'main' else 0)
    changed = []
    for i, (a, b) in enumerate(zip(oldmath, newmath)):
        if etree.tostring(a, method='c14n') != etree.tostring(b, method='c14n'):
            changed.append(i)
            assert mathtext(a) == 'τ=0.20'
            assert mathtext(b) == 'τ'
    assert len(changed) == (1 if kind == 'main' else 0)
    if kind == 'main':
        for i in [0, 6, 8]:
            assert table_data(original.tables[i]) == table_data(doc.tables[i]), ('preserved table', i)
        unchanged = [0, 2, 3, 4, 7, 10]
    else:
        for i in [1, 4, 6, 7]:
            assert table_data(original.tables[i]) == table_data(doc.tables[i]), ('preserved table', i)
        unchanged = [0, 1]
    a, b = image_hashes(original), image_hashes(doc)
    assert len(a) == len(b)
    for i in range(len(a)):
        assert (a[i] == b[i]) if i in unchanged else (a[i] != b[i]), ('image', kind, i)
    text = '\n'.join(p.text for p in doc.paragraphs)
    for forbidden in ['AUTHOR COMPLETION', 'PUBLICATION-SAFETY', 'A07', 'Main Figure',
                      'current snapshot', 'dirty Git', 'present revision', 'Not run',
                      'Not defined', 'cmig_todo', 'all three configurations were evaluated over the same three seeds']:
        assert forbidden.casefold() not in text.casefold(), forbidden
    assert 'post-hoc' in text and 'all five' in text
    assert not doc._element.xpath('.//w:ins|.//w:del'), 'Unresolved tracked changes'
    return {'paragraphs': len(doc.paragraphs), 'tables': len(doc.tables),
            'native_equations': len(newmath), 'changed_inline_tau_only': changed,
            'unchanged_image_indices': unchanged, 'changed_figures': len(a) - len(unchanged)}


def main():
    data = load(QUANT / 'quantitative_data.json')
    verify_arithmetic(data)
    inputs = {str(p): sha(p) for p in [*SOURCES.values(), PAPER / 'DENT_STRIDE_All_Elements_Editable_v6.pptx']}
    report = {'sources': inputs, 'reference_article': str(PAPER / '1-s2.0-S0895611126001254-main.pdf'),
              'outputs': {}, 'figures': {}, 'selection_policy': data['selection_policy'],
              'author_checks': ['Author-to-affiliation mapping', 'Originating ethics approval or exemption',
                                'Image publication permission including pediatric photographs',
                                'Final author approval of funding, CRediT, AI use and data terms']}
    for kind in SOURCES:
        original = Document(SOURCES[kind])
        doc = Document(SOURCES[kind])
        report['figures'][kind] = update_main(doc, data) if kind == 'main' else update_supplement(doc, data)
        for table in doc.tables:
            publication_table_rules(table)
        for p in doc.paragraphs:
            if p.style.name == 'Title':
                for r in p.runs:
                    font(r, 14, True)
        drop_orphan_images(doc)
        doc.core_properties.comments = ''
        doc.core_properties.subject = 'DENT-STRIDE acquisition-specific validation and exploratory seed-subset analysis'
        report['outputs'][kind] = validate(original, doc, kind)
        doc.save(OUTPUTS[kind])
        with zipfile.ZipFile(OUTPUTS[kind]) as z:
            assert z.testzip() is None
            embedded = [name for name in z.namelist() if name.endswith('.svg')]
            report['outputs'][kind]['embedded_native_SVG'] = embedded
            assert len(embedded) == (5 if kind == 'main' else 9)
            if kind == 'main':
                assert hashlib.sha256(z.read('word/media/Figure_2_DENT_STRIDE_architecture.svg')).hexdigest() == sha(ARCH / 'Figure_2_DENT_STRIDE_architecture.svg')
        report['outputs'][kind].update({'path': str(OUTPUTS[kind]), 'sha256': sha(OUTPUTS[kind])})
    for p, digest in inputs.items():
        assert sha(p) == digest, 'Source was changed'
    (ROOT / 'document_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report['outputs'], indent=2))


if __name__ == '__main__':
    main()
