"""Create a new CMIG supplement from the original document and verified results."""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
import zipfile

import yaml
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_revision_2026-10-03'
EVIDENCE = ROOT / 'evidence'
SOURCE = PAPER / 'DENT_STRIDE_CMIG_supplement_visual.docx'
DEST = PAPER / 'DENT_STRIDE_CMIG_supplement_controlled_results_20261003.docx'
TRAIN = PAPER / 'materials/cmig_controlled_2026-09-29'
EM = '—'


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value, decimals=3):
    return EM if value is None else f'{float(value):.{decimals}f}'


def font_run(run, size=11, bold=None):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts'); rpr.insert(0, fonts)
    for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
        fonts.set(qn('w:' + key), 'Times New Roman')


def set_text(paragraph, text):
    paragraph.text = text
    for run in paragraph.runs:
        font_run(run, 9 if paragraph.style.name == 'Caption' else 11)


def paragraph(document, text, style=None):
    p = document.add_paragraph(text, style=style)
    for r in p.runs:
        font_run(r, 9 if style == 'Caption' else 11)
    p.paragraph_format.space_after = Pt(5 if style != 'Caption' else 8)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.widow_control = True
    return p


def heading(document, text, level=1):
    p = document.add_paragraph(text, style=f'Heading {level}')
    for r in p.runs:
        font_run(r, 12 if level == 1 else 11, True)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p


def border(parent, name, value, size=None):
    e = OxmlElement('w:' + name); e.set(qn('w:val'), value)
    if size is not None:
        e.set(qn('w:sz'), str(size)); e.set(qn('w:color'), '000000')
    parent.append(e)


def format_table(table, widths=None, numeric_from=1):
    table.autofit = False
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    properties = table._tbl.tblPr
    for existing in properties.findall(qn('w:tblBorders')):
        properties.remove(existing)
    borders = OxmlElement('w:tblBorders')
    for key in ['left', 'right', 'insideH', 'insideV']:
        border(borders, key, 'nil')
    border(borders, 'top', 'single', 8); border(borders, 'bottom', 'single', 8)
    properties.append(borders)
    if widths is not None:
        for col, width in zip(table.columns, widths):
            col.width = Inches(width)
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    for i, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        if i == 0 and trpr.find(qn('w:tblHeader')) is None:
            trpr.append(OxmlElement('w:tblHeader'))
        if trpr.find(qn('w:cantSplit')) is None:
            trpr.append(OxmlElement('w:cantSplit'))
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            for item in tcpr.findall(qn('w:tcBorders')):
                tcpr.remove(item)
            cb = OxmlElement('w:tcBorders')
            for key in ['left', 'right', 'top', 'bottom']:
                border(cb, key, 'nil')
            if i == 0:
                # Replace only the header bottom rule; no body grid.
                cb.remove(cb.find(qn('w:bottom')))
                border(cb, 'bottom', 'single', 5)
            tcpr.append(cb)
            margins = OxmlElement('w:tcMar')
            for key, value in [('top', 45), ('bottom', 45), ('left', 45), ('right', 45)]:
                m = OxmlElement('w:' + key); m.set(qn('w:w'), str(value)); m.set(qn('w:type'), 'dxa'); margins.append(m)
            tcpr.append(margins)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < numeric_from else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                p.paragraph_format.keep_with_next = i == 0
                for r in p.runs:
                    font_run(r, 9, i == 0)


def table(document, caption, headers, rows, widths, note=None, numeric_from=1):
    p = paragraph(document, caption, 'Caption'); p.paragraph_format.keep_with_next = True
    t = document.add_table(rows=1, cols=len(headers))
    for c, text in zip(t.rows[0].cells, headers):
        c.text = text
    for values in rows:
        cells = t.add_row().cells
        for c, text in zip(cells, values):
            c.text = str(text)
    format_table(t, widths, numeric_from)
    if note:
        paragraph(document, note, 'Caption')
    return t


def picture(document, path, caption, new_page=False):
    path = Path(path)
    assert path.is_file(), path
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.page_break_before = new_page
    p.add_run().add_picture(str(path), width=Inches(6.35))
    paragraph(document, caption, 'Caption')


CONFIG_LABELS = {
    'mobile_1024_base': 'Base, 1024', 'mobile_1024_features': 'Features, 1024',
    'mobile_1024_identity': 'Identity, 1024', 'mobile_1024_full': 'Combined, 1024',
    'mobile_640_base': 'Base, 640', 'mobile_640_full': 'Combined, 640',
    'pano_base': 'Base', 'pano_factor': 'Factor', 'pano_prototype': 'Prototype',
    'pano_factor_prototype': 'Factor + Prototype',
    'dslr_base_adamw': 'Base + AdamW', 'dslr_cmfa_adamw': 'CMFA + AdamW',
    'dslr_base_dualopt': 'Base + DualOpt', 'dslr_cmfa_dualopt': 'CMFA + DualOpt',
}
ORDER = ['mobile_1024_base', 'mobile_1024_features', 'mobile_1024_identity', 'mobile_1024_full',
         'mobile_640_base', 'mobile_640_full', 'pano_base', 'pano_factor', 'pano_prototype',
         'pano_factor_prototype', 'dslr_base_adamw', 'dslr_cmfa_adamw', 'dslr_base_dualopt', 'dslr_cmfa_dualopt']


def add_current_results(document, evidence, runs):
    indexed = {(r['configuration'], r['seed']): r for r in runs}
    headings = 'S8. Controlled experiments: detailed results'
    heading(document, headings)
    paragraph(document, 'The controlled study comprises 14 training configurations, each evaluated with seeds 0, 1, and 2. All use the original cohort partitions, 160 recorded epochs, a training batch size of 8, a validation batch size of 16, mixed-precision training, and exponential moving average weights. The checkpoint with maximum validation COCO AP is frozen and replayed in FP32. Within each cohort, data, initialization, common losses, augmentation, optimization schedules, and checkpoint selection are held fixed except for the stated variable. Mobile resolution is an explicit experimental factor. Pano training comparisons disable factor-score fusion and all anatomical priors; the prior comparison is a subsequent same-weight replay. These are internal-validation estimates.')
    paragraph(document, 'Tables S7–S13 provide run-level endpoints, controlled sensitivity analyses, seed-paired differences, and effective training settings. Means and sample standard deviations in the main article use exactly three completed training seeds. “—” denotes a training condition without a completed 160-epoch checkpoint replay; its partial validation maximum is excluded. The internal ablations use seed 0, and reused intact configurations are identified in the table notes.')

    rows = []
    for name in ORDER:
        for seed in [0, 1, 2]:
            r = indexed[name, seed]
            rows.append([r['cohort'], CONFIG_LABELS[name], seed, *[fmt(r[k]) for k in ['AP', 'AP50', 'AP75', 'AR100']],
                         fmt(r['params_M']), r['best_epoch']])
    table(document, 'Table S7. Individual-seed results for all 14 controlled training configurations.',
          ['Cohort', 'Configuration', 'Seed', 'AP', 'AP50', 'AP75', 'AR100', 'Params\n(M)', 'Best\nepoch'],
          rows, [.52, 1.55, .36, .58, .58, .58, .58, .58, .62],
          'Metrics are percentage points; epoch is one-based. Features denote CMFA + HOOA; identity denotes Factor + Prototype. Pano rows exclude inference priors and factor fusion. Parameter counts include all executable parameters, including auxiliary heads; only the appropriate branches are used during inference. These controlled base models are separate from the official DEIM-s references in Table S4.', numeric_from=2)

    heading(document, 'S8.1. Module-internal ablations', 2)
    paragraph(document, 'Internal comparisons retain the Mobile combined architecture and its background losses except for the named change. CMFA channel and spatial switches preserve the concatenated 2C output and downstream network. Removing the HOOA high-pass branch sets only its response to zero; removing the multiplicative interaction substitutes the direct context response, preserving local and context branches and the same projection. Factor-loss removal retains all heads and disables both training and inference fusion, isolating the selected auxiliary loss. Prototype supervision is removed by disabling the branch and its loss; prototype-number and temperature comparisons otherwise preserve the training setup. DSLR DualOpt comparisons retain ordinary AdamW weight decay and toggle only rollback or additional rank-dependent decay.')
    families = [
        ('CMFA', [('Neither gate', 'internal_mobile_cmfa_no_gates'), ('Channel only', 'internal_mobile_cmfa_channel_only'),
                  ('Spatial only', 'internal_mobile_cmfa_spatial_only'), ('Both gates', 'mobile_1024_full')], 'mobile_1024_full'),
        ('HOOA', [('Intact', 'mobile_1024_full'), ('No high-pass', 'internal_mobile_hooa_no_highpass'),
                  ('No interaction', 'internal_mobile_hooa_no_second_order'), ('Neither branch', 'internal_mobile_hooa_neither')], 'mobile_1024_full'),
        ('Factor losses', [('All three', 'internal_mobile_factor_train_off'), ('No quadrant CE', 'internal_mobile_factor_no_quadrant'),
                          ('No position CE', 'internal_mobile_factor_no_position'), ('No dentition CE', 'internal_mobile_factor_no_dentition')], 'internal_mobile_factor_train_off'),
        ('Prototype', [('No supervision', 'internal_mobile_proto_off'), ('K = 1, τ = 0.20', 'internal_mobile_proto_k1_tau0p2'),
                       ('K = 2, τ = 0.20', 'internal_mobile_proto_k2_tau0p2'), ('K = 4, τ = 0.20', 'mobile_1024_full'),
                       ('K = 4, τ = 0.10', 'internal_mobile_proto_k4_tau0p1'), ('K = 4, τ = 0.50', 'internal_mobile_proto_k4_tau0p5')], 'mobile_1024_full'),
        ('DualOpt', [('Neither operation', 'dslr_cmfa_adamw'), ('Rollback only', 'internal_dslr_rollback_only'),
                     ('Additional decay only', 'internal_dslr_extra_decay_only'), ('Both operations', 'dslr_cmfa_dualopt')], 'dslr_cmfa_adamw'),
        ('Background', [('Neither loss', 'internal_mobile_background_neither'), ('NWD only', 'internal_mobile_background_nwd_only'),
                        ('Arch only', 'internal_mobile_background_arch_only'), ('Both losses', 'mobile_1024_full')], 'mobile_1024_full'),
    ]
    rows = []
    for family, settings, reference in families:
        reference_AP = indexed[reference, 0]['AP']
        for label, name in settings:
            r = indexed.get((name, 0))
            rows.append([family, label, *[fmt(r[k]) if r else EM for k in ['AP', 'AP50', 'AP75', 'AR100']],
                         f"{r['AP'] - reference_AP:+.3f}" if r else EM])
    table(document, 'Table S8. Module-internal and background-loss sensitivity at seed 0.',
          ['Component', 'Setting', 'AP', 'AP50', 'AP75', 'AR100', 'ΔAP'], rows,
          [.85, 1.82, .66, .66, .66, .66, .64],
          'ΔAP uses the intact setting in each family: both CMFA gates; intact HOOA; all factor losses with fusion disabled; K = 4 and τ = 0.20; CMFA + ordinary AdamW for DualOpt; both background losses. Positive ΔAP means the displayed alternative exceeds its reference, not that the intact module improves AP. The intact Mobile and DSLR configurations reuse their controlled seed-0 runs. “—” denotes an incomplete training/replay condition. No uncertainty bars are assigned to these single-seed measurements.', numeric_from=2)
    paragraph(document, 'The internal results identify non-monotonic responses: channel-only CMFA, removal of the HOOA high-pass response, and removal of prototype supervision exceed the intact Mobile seed-0 configuration. These comparisons motivate reporting the full sensitivity range rather than treating every enabled branch as necessary. They are not used to replace the predefined combined models with a retrospectively selected single-seed alternative.')

    heading(document, 'S8.2. Paired training-seed differences', 2)
    labels = {
        'Mobile: full minus base, 1024': 'Mobile1024: combined − base',
        'Mobile: features minus base, 1024': 'Mobile1024: features − base',
        'Mobile: identity minus base, 1024': 'Mobile1024: identity − base',
        'Mobile: full minus features, 1024': 'Mobile1024: combined − features',
        'Mobile: full minus identity, 1024': 'Mobile1024: combined − identity',
        'Mobile: full minus base, 640': 'Mobile640: combined − base',
        'Mobile: base 1024 minus base 640': 'Mobile base: 1024 − 640',
        'Mobile: full 1024 minus full 640': 'Mobile combined: 1024 − 640',
        'Pano: factor minus base': 'Pano: Factor − base',
        'Pano: prototype minus base': 'Pano: Prototype − base',
        'Pano: factor+prototype minus base': 'Pano: Factor + Prototype − base',
        'Pano: factor+prototype minus factor': 'Pano: Factor + Prototype − Factor',
        'DSLR: CMFA+AdamW minus base+AdamW': 'DSLR: CMFA − base, AdamW',
        'DSLR: base+DualOpt minus base+AdamW': 'DSLR base: DualOpt − AdamW',
        'DSLR: CMFA+DualOpt minus base+AdamW': 'DSLR: CMFA + DualOpt − base + AdamW',
        'DSLR: CMFA+DualOpt minus CMFA+AdamW': 'DSLR CMFA: DualOpt − AdamW',
        'DSLR: CMFA+DualOpt minus base+DualOpt': 'DSLR: CMFA − base, DualOpt',
        'Mobile feature-identity AP interaction': 'Mobile: feature–identity interaction',
        'Pano factor-prototype AP interaction': 'Pano: factor–prototype interaction',
        'DSLR CMFA-DualOpt AP interaction': 'DSLR: CMFA–DualOpt interaction',
    }
    rows = []
    for r in evidence['paired_contrasts']:
        a = r['metrics']['AP']
        rows.append([labels[r['comparison']], *[f'{v:+.3f}' for v in a['values']],
                     f"{a['mean']:+.3f} ± {a['SD']:.3f}", f"[{a['CI95_low']:.3f}, {a['CI95_high']:.3f}]"])
    table(document, 'Table S9. Paired AP differences across seeds 0, 1, and 2.',
          ['Comparison', 'Seed 0', 'Seed 1', 'Seed 2', 'Mean ± SD', '95% CI'], rows,
          [2.39, .53, .53, .53, 1.0, 1.05],
          'All values are AP percentage points. Intervals are two-sided Student-t intervals for the mean of the three within-seed differences (df = 2). Interaction = combined − component A − component B + base. Intervals describe training-seed variation on the same internal-validation partition; they are not patient-sampling or independent-test intervals. The multiple exploratory comparisons are not adjusted for multiplicity.', numeric_from=1)

    heading(document, 'S8.3. Same-weight anatomical priors and factor fusion', 2)
    prior = list(csv.DictReader((EVIDENCE / 'spatial_shape_per_seed.csv').open(encoding='utf-8-sig')))
    rows = [[r['seed'], {'none': 'None', 'center': 'Center', 'shape': 'Shape', 'both': 'Center + shape'}[r['prior']],
             *[fmt(r[k]) for k in ['AP', 'AP50', 'AP75', 'AR100']], f"{float(r['delta_AP_vs_none']):+.3f}"] for r in prior]
    table(document, 'Table S10. Pano same-checkpoint prior replays for each training seed.',
          ['Seed', 'Prior', 'AP', 'AP50', 'AP75', 'AR100', 'ΔAP'], rows,
          [.48, 1.6, .76, .76, .76, .76, .70],
          'All settings within a seed use the same frozen Factor + Prototype checkpoint selected without anatomical priors. Class scores are recomputed from every ordinary query and class before native top-k selection. Factor fusion, appearance scoring, view scoring, and global assignment are disabled. Center and log-shape weights are 0.20 and 0.05; ΔAP is relative to no prior within the same seed.', numeric_from=2)
    rows = [[('On' if r['training_fusion'] else 'Off'), ('On' if r['inference_fusion'] else 'Off'),
             *[fmt(r[k]) for k in ['AP', 'AP50', 'AP75', 'AR100']], fmt(r['F1'], 2)] for r in evidence['factor_fusion_2x2']]
    table(document, 'Table S11. Mobile training-fusion × inference-fusion comparison at seed 0.',
          ['Training\nfusion', 'Inference\nfusion', 'AP', 'AP50', 'AP75', 'AR100', 'F1'], rows,
          [.79, .79, .89, .89, .89, .89, .70],
          'The two training states retain all factor and prototype supervision. Each frozen checkpoint is replayed with inference fusion on and off. Weights for quadrant, within-quadrant position, and dentition fusion are 0.10, 0.05, and 0.03. Training-state checkpoints are selected by validation AP under their specified inference setting; the inference replays do not change weights or model selection. F1 uses score ≥0.30 and IoU ≥0.50.', numeric_from=2)

    heading(document, 'S8.4. Fixed-threshold detection and identity endpoints', 2)
    rows = []
    prior_lookup = {(r['prior'], int(r['seed'])): r for r in prior}
    for cohort, reference, combined in [('Mobile', 'mobile_1024_base', 'mobile_1024_full'),
                                       ('Pano', 'pano_base', 'pano_factor_prototype'),
                                       ('DSLR', 'dslr_base_adamw', 'dslr_cmfa_dualopt')]:
        for kind, name in [('Base', reference), ('Combined', combined)]:
            for seed in [0, 1, 2]:
                r = prior_lookup['both', seed] if cohort == 'Pano' and kind == 'Combined' else indexed[name, seed]
                rows.append([cohort, kind, seed, *[fmt(r[k], 2) for k in ['Precision', 'Recall', 'F1',
                           'matched_tooth_FDI_accuracy', 'localization_coverage', 'raw_duplicate_image_rate']]])
    table(document, 'Table S12. Individual-seed fixed-threshold endpoints for controlled base and domain-configured models.',
          ['Cohort', 'Model', 'Seed', 'P', 'R', 'F1', 'Matched\nFDI', 'Coverage', 'Duplicate\nimages'], rows,
          [.53, .78, .35, .63, .63, .63, .80, .77, .91],
          'All metrics are percentages. Mobile uses 1024 × 1024; Pano combined includes the joint center/shape prior; DSLR combined uses CMFA + DualOpt. Native top-300 predictions are retained at score ≥0.30, without added NMS. Class-agnostic Hungarian matching maximizes total IoU and accepts matches at IoU ≥0.50; a detection is correct only when FDI also agrees. P = correct detections / retained predictions; R = correct detections / GT teeth; matched FDI = correct identities / localized matches; coverage = localized matches / GT teeth. Duplicate-image rate counts images containing repeated raw predicted FDI labels and is not a clinical charting-error rate.', numeric_from=2)

    heading(document, 'S8.5. Effective training and inference controls', 2)
    paragraph(document, 'Table S13 lists effective settings rather than inactive configuration entries. Mobile base and module comparisons share NWD and arch-position losses; they therefore represent controlled local references, not the unmodified official DEIM-s implementation. Pano and DSLR share an arch-position loss of 0.25 and do not use NWD. Prototype scores are excluded from detection logits in all configurations; prototype supervision is weighted only through its criterion loss. Pano inference priors are disabled for every training-time validation and are evaluated separately in Table S10.')
    roles = read(EVIDENCE / 'active_configuration_roles.json')
    jobs = read(TRAIN / 'execution_manifest.json')['jobs']
    rows = []
    for name in ORDER:
        role = roles[name + '_seed0']
        job = next(j for j in jobs if j['configuration'] == name and j['seed'] == 0)
        cfg = yaml.safe_load(Path(job['config']).read_text())
        weight = cfg['DEIMCriterion']['weight_dict']; loss_names = cfg['DEIMCriterion']['losses']
        f = role['Factor_heads']; p = role['Prototype_head']
        fusion = 'On / On' if any(role['factor_alpha']) and role['factor_fusion_train'] else 'Off / Off'
        aux = '/'.join(fmt(weight['loss_' + key], 2) if key in loss_names else EM for key in ['quad', 'tooth', 'dent'])
        aux += '; ' + (fmt(weight['loss_proto'], 2) if 'proto' in loss_names else EM)
        bkg = '/'.join([fmt(weight['loss_nwd'], 2) if role['logged_loss_activity']['nwd']['ever_nonzero'] else EM,
                        fmt(weight['loss_arch'], 2) if role['logged_loss_activity']['arch']['ever_nonzero'] else EM])
        features = 'CMFA + HOOA' if role['CMFA'] and role['HOOA'] else 'CMFA' if role['CMFA'] else EM
        bank = '4 / 0.20' if p else EM
        scale = '640 multi' if role['multiscale_training'] else str(role['eval_size'][0])
        rows.append([role['eval_size'] and next(r['cohort'] for r in runs if r['configuration'] == name),
                     CONFIG_LABELS[name], features, 'Factor + Proto' if f and p else 'Factor' if f else 'Proto' if p else EM,
                     fusion, aux, bkg, 'DualOpt' if role['optimizer'] == 'DualOptAdamW' else 'AdamW'])
    table(document, 'Table S13. Effective component, loss, and optimization settings for the controlled primary study.',
          ['Cohort', 'Configuration', 'Features', 'Identity', 'Factor\ntrain / infer', 'Auxiliary\nq/p/d; proto', 'NWD /\narch', 'Optimizer'], rows,
          [.49, 1.18, .83, .74, .65, 1.13, .59, .65],
          'Auxiliary entries are cross-entropy coefficients for quadrant (q), position (p), dentition (d), and prototypes; “—” means inactive. Every active bank contains four 256-dimensional prototypes per class at τ = 0.20; prototype inference fusion is zero. Mobile factor coefficients are 0.10/0.05/0.03. Ordinary AdamW uses learning rate 5 × 10⁻⁴, backbone learning rate 2.5 × 10⁻⁴, β = (0.9, 0.999), and weight decay 10⁻⁴; normalization-parameter decay exclusions follow the common detector. DualOpt retains this base decay and adds rollback 0.005, rank-dependent decay 0.01, rank exponent 1.5, and multiplier range [0.5, 1.5].', numeric_from=4)
    paragraph(document, 'All Mobile conditions use fixed square training and evaluation at their stated 640 or 1024 resolution, no Mosaic, and no MixUp. Pano uses fixed 640, no Mosaic, and MixUp probability 0.50 during its scheduled training interval. DSLR uses the common 640 evaluation resolution and a 480–800 multiscale training schedule, no Mosaic, and no MixUp. The common photometric distortion, random zoom-out probability 0.50, random IoU-crop probability 0.80, augmentation shutdown, and native stage schedule are identical within each cohort. Geometric flips are disabled. Automatic mixed precision, gradient clipping at 0.10, EMA decay 0.9999, ImageNet backbone initialization, and the validation-AP checkpoint rule are shared. At best-AP ties, the earlier checkpoint is retained.')


def add_complete_fdi_figures(document):
    h = heading(document, 'S9. Complete FDI errors and class support')
    h.paragraph_format.page_break_before = True
    paragraph(document, 'Figs. S9–S11 include every FDI category for the three domain-configured seed-0 checkpoints. Native top-300 detections use score ≥0.30 and class-agnostic Hungarian assignment maximizing total IoU, accepting pairs at IoU ≥0.50. Each reference tooth contributes one predicted FDI outcome or “Missed”. Color shows its reference-row percentage; gray rows have no validation support and are unavailable, not zero-accuracy classes. Training and validation tooth counts appear beside the corresponding row. Unmatched predictions are counted separately because they have no reference row; wrong-FDI localized matches remain off diagonal. Figure data include raw counts, per-class COCO AP/AP50/AP75/AR100, and both support counts; unavailable class metrics are left missing.')
    captions = {
        'Pano': 'Fig. S9. Complete Pano FDI outcome matrix and class support. The Factor + Prototype checkpoint includes the joint SpatialShape prior. All 32 permanent classes are shown in clinical FDI order after decoding the native enumeration. The matrix contains 1,861 reference teeth: 1,710 correct identities, 128 wrong-FDI localized matches, and 23 missed teeth. Its 183 unmatched predictions are excluded from the reference-row matrix. The training support totals 16,234 teeth.',
        'Mobile': 'Fig. S10. Complete Mobile FDI outcome matrix and class support for the combined 1024-resolution model. All 52 clinical FDI slots are shown. The matrix contains 2,459 reference teeth: 2,134 correct identities, 204 wrong-FDI localized matches, and 121 missed teeth. Its 479 unmatched predictions are excluded from the reference-row matrix. Training support totals 9,837 teeth; FDI 18, 28, 38, and 48 have no training instances. Gray rows indicate absent validation support.',
        'DSLR': 'Fig. S11. Complete DSLR FDI outcome matrix and class support for CMFA + DualOpt. All 52 clinical FDI slots are shown. The matrix contains 2,742 reference teeth: 2,617 correct identities, 57 wrong-FDI localized matches, and 68 missed teeth. Its 240 unmatched predictions are excluded from the reference-row matrix. Training support totals 10,760 teeth. Gray rows indicate absent validation support; zero training support is separately visible in the count column.',
    }
    for cohort, number in [('Pano', 9), ('Mobile', 10), ('DSLR', 11)]:
        data = read(ROOT / f'figures/errors/{cohort}_complete_FDI_confusion.json')
        assert data['verification']['totals_equal_frozen_fixed_threshold_metrics']
        assert data['verification']['macro_per_class_COCO_equal_saved_metrics']
        picture(document, ROOT / f'figures/errors/Fig_S{number}_{cohort}_complete_FDI_confusion.png',
                captions[cohort], new_page=number > 9)


def main():
    document = Document(SOURCE)
    original_table_values = [[[c.text for c in r.cells] for r in t.rows] for t in document.tables[:5]]
    original_math = len(document._element.xpath('.//m:oMath'))
    paragraphs = document.paragraphs
    replacements = {
        2: 'Supplementary component descriptions, historical experimental families, and detailed controlled internal-validation results. Each numerical family is identified by its training and inference protocol; historical results are kept separate from the current three-seed controlled study.',
        4: 'The main article shows HOOA in the Mobile configuration (Fig. 5). Figs. S1–S3 describe CEDv2, SMWG, MPF, and UCM, which were evaluated in earlier feature-processing families and are not active in the three domain-configured models of the controlled study.',
        7: 'Fig. S1. CEDv2 comparison architecture. Guided top-down and bottom-up operations refine the projected pyramid before the main encoder. CEDv2 is inactive in the domain-configured models. ↑ denotes resizing/upsampling except the labelled learned stride-2 operation; C denotes concatenation, σ sigmoid, and ⊙ element-wise multiplication.',
        14: 'S2. Historical recipe and feature-combination results',
        15: 'Table S1 reports earlier cohort-specific configurations, and Table S2 contains a separate 16-way feature-combination family. These individual-run results describe those experimental families; they do not constitute the factorial ablations of the current domain-configured models. The current controlled comparisons are reported in the main article and detailed in Tables S7–S13.',
        17: 'Numerical precision is retained from the original result records. “Prototype” denotes the earlier auxiliary-supervision configuration, not prototype-score fusion. “SpatialShape” denotes a prior-enabled complete configuration rather than a same-checkpoint switch.',
        18: 'In Table S2, DualOpt is active in every row, with the same seed, input policy, training budget, and checkpoint-selection rule within that family. All four-feature combinations are included. Its 73.431 AP CMFA + DualOpt result belongs to this historical feature family and is separate from the three-seed DSLR results of the controlled study.',
        20: '+/− indicate enabled/disabled CMFA, SMWG, UCM, and MPF. Metrics are percentage points and parameters are millions. All entries are individual-run results; no across-seed error bars are inferred for this historical family.',
        21: 'S3. Historical progressive identity-supervision study',
        24: 'The bank uses four prototypes per class and temperature 0.20. The approximately 15.5 M-parameter reference is a distinct historical feature stack. This family does not isolate center versus shape, final-Mobile factor fusion, or the current controlled base models.',
        26: 'S4. Historical upstream baseline repeats and module migration',
        27: 'Table S4 reports prior training runs of the official DEIM-s and RT-DETRv2-S implementations with seeds 0, 1, and 2. Mean and sample standard deviation are calculated across each implementation’s three validation-selected checkpoints. These values belong to the upstream comparator family; current DENT-STRIDE training-seed results are independently provided in Table S7.',
        29: 'All entries are internal-validation estimates selected by maximum validation AP. Sample standard deviations describe variability among the three corresponding training runs and are not transferred to another architecture or configuration.',
        30: 'Prior seed-0 module-migration experiments yielded different effects in the official scaffold: Pano Factor + Prototype + SpatialShape changed AP from 53.349 to 54.151; Mobile DualOpt from 53.605 to 52.967; and DSLR CMFA + DualOpt from 72.557 to 72.095. These separate implementation comparisons illustrate configuration dependence and are not substitutes for the current controlled models.',
        32: 'Table S5 records earlier Pano and Mobile framework comparators using the same image-level validation cohorts and FDI category definitions. Framework-native pretraining, optimization, training, and post-processing are retained. Detection-relevant image, category, and box annotations are equivalent for the smartphone annotation variants, which differ only in additional tooth-type metadata. The current same-image comparisons use these stored external predictions evaluated against the controlled study’s validation annotations. Training resolution and other framework settings remain different, so these are system-level references.',
        34: 'All values are single-run framework references. They do not inherit standard deviations from the controlled detector family. The stronger RT-DETR-L reference is retained alongside the smaller conventional and transformer models.',
        35: 'S6. Training geometry and current controlled visual evidence',
        36: 'Main Fig. 3 shows the training-derived Pano center templates. Fig. S4 complements these with width/height distributions; inference uses the joint quadratic log-shape score. Figs. S5–S8 present learning curves, internal sensitivity, threshold diagnostics, and class-wise prototype evidence from the completed controlled runs.',
        38: 'Fig. S4. Normalized width and height of the 16,234 DENTEX training boxes by permanent FDI class. These distributions describe training geometry; the SpatialShape inference term is a joint quadratic score in log-width and log-height.',
    }
    for index, text in replacements.items():
        set_text(paragraphs[index], text)
    # Replace only the optional SMWG picture with a native, code-aligned redraw.
    # Historical CEDv2/MPF/UCM sources are retained, not artificially upsampled.
    smwg = ROOT / 'figures/methods/SMWG_native_code_aligned.png'
    assert smwg.is_file(), smwg
    smwg_paragraph = paragraphs[9]
    smwg_paragraph.clear()
    smwg_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    smwg_paragraph.paragraph_format.keep_with_next = True
    smwg_paragraph.add_run().add_picture(str(smwg), width=Inches(6.35))
    # Remove only the obsolete visual/evaluation tail; the existing shape figure is retained.
    first = paragraphs[39]._p
    body = document._element.body
    start = list(body).index(first)
    for child in list(body)[start:]:
        if child.tag != qn('w:sectPr'):
            body.remove(child)

    # Retained historical tables keep every original numeric cell unchanged.
    old_widths = [[.65, 2.35, .66, .66, .66, .66, .66],
                  [.39, .39, .39, .39, .77, .77, .77, .77, .84],
                  [.72, 2.9, .81, .81, .81, .75],
                  [.78, 2.04, .65, .65, .65, 1.6],
                  [.75, 2.39, .8, .8, .8, .8]]
    for t, widths in zip(document.tables, old_widths):
        format_table(t, widths, numeric_from=2 if len(t.columns) != 9 else 0)

    picture(document, ROOT / 'figures/performance/controlled_three_seed_learning_curves.png',
            'Fig. S5. Validation learning curves of controlled base and combined models across the three acquisition domains. Lines show the mean COCO AP at each actual epoch; shaded bands show the sample standard deviation across seeds 0, 1, and 2. All 160 recorded epochs are displayed without smoothing. Pano curves exclude factor inference fusion and anatomical priors. Curves represent online training-validation records; endpoint tables use frozen-checkpoint FP32 replay.', new_page=True)
    picture(document, ROOT / 'figures/performance/controlled_internal_ablation_AP_deltas_part2.png',
            'Fig. S6. Internal branch, temperature, and background-loss sensitivity at seed 0. (A) HOOA branches relative to the intact Mobile combined model. (B) Prototype temperature at four prototypes per class, relative to τ = 0.20. (C) NWD and arch-position losses relative to their joint use. (D) Training/inference factor fusion relative to fusion disabled in both phases. Each inference pair uses the same frozen weight within its training state. Positive values mean that the displayed alternative exceeds its specified reference. “—” denotes an incomplete training/replay condition and has no numeric point. The remaining internal families are shown in main Fig. 11.', new_page=True)
    picture(document, ROOT / 'figures/performance/controlled_prediction_curves.png',
            'Fig. S7. Detection and threshold diagnostics for seed-0 controlled base and domain-configured models. Rows show Mobile, Pano, and DSLR. Columns show category-macro-averaged COCO interpolated precision at IoU 0.50 and maxDets = 100; F1; and the fraction of validation images containing repeated raw FDI labels. Categories without validation support are omitted from the precision macro-average. Threshold curves use native top-300 outputs without added NMS and score thresholds 0.05–0.95 in steps of 0.05. Class-agnostic Hungarian localization matching at IoU ≥0.50 is followed by exact FDI comparison. Pano includes the combined prior. Dotted lines indicate score 0.30. These single-seed diagnostics do not measure clinical charting-error rates.', new_page=True)
    picture(document, ROOT / 'figures/mechanism/current_Mobile_cohesion_all_FDI.png',
            'Fig. S8. Within-class prototype cohesion for every FDI slot in the Mobile seed-0 checkpoint. (A) Permanent and (B) primary FDI classes. Cohesion is the mean of the six pairwise cosine similarities among the four normalized prototypes of a class. FDI 18, 28, 38, and 48 have no smartphone training instances and are shown as unavailable; their initialized prototype slots are excluded from learned-evidence interpretation. The bank supplies training auxiliary supervision and is distinct from the fixed Pano center and shape templates.', new_page=True)

    heading(document, 'S7. Evaluation scope and visualization methods')
    paragraph(document, 'The current study uses completed frozen checkpoints and their full validation predictions. Historical feature, identity, and upstream comparator families remain identified in Sections S2–S5; their results are not attributed to current weights. Native online metrics characterize learning trajectories, while FP32 checkpoint replay provides the numerical endpoints. Table S6 summarizes the scope of each evidence family.')
    table(document, 'Table S6. Evaluation scope of the experimental families.',
          ['Evidence family', 'Comparison', 'Interpretation'], [
              ['Controlled primary models', '14 configurations × seeds 0/1/2', 'Within-cohort module and resolution comparisons'],
              ['Internal training ablations', 'Named component change; seed 0', 'Single-seed sensitivity; uncompleted cells unavailable'],
              ['Pano prior replays', 'None / center / shape / joint; same weight', 'Score-correction effects without retraining'],
              ['Mobile fusion replays', 'Training fusion × inference fusion', 'Same-weight inference switches within training state'],
              ['Historical feature family', '16 combinations; DualOpt fixed', 'Earlier within-family feature response'],
              ['Historical identity family', 'Progressive supervision / prior', 'Earlier configuration-specific identity response'],
              ['Official upstream repeats', 'DEIM-s and RT-DETRv2-S, three seeds', 'Upstream baseline training variation'],
              ['External detector outputs', 'Framework-native archived predictions', 'System-level references on common validation labels'],
          ], [1.65, 2.25, 2.55], numeric_from=3)
    heading(document, 'S7.1. Predictions and operating-point definitions', 2)
    paragraph(document, 'Validation image identifiers align predictions with the corresponding unchanged images and ground truth. Clinical FDI codes are decoded through each cohort’s label map; zero-based category indices are not displayed as tooth numbers. COCO AP uses the native detector outputs with no additional display score cutoff. Fixed-threshold endpoints use score ≥0.30 and class-agnostic one-to-one Hungarian assignment that maximizes total IoU, accepting only pairs with IoU ≥0.50. Exact FDI agreement defines a correct detection. Wrong-number matches contribute both a false positive and a false negative. The denominators for FDI accuracy, recall, localization coverage, and repeated-label image rate are explicitly distinguished in Table S12.')
    paragraph(document, 'Same-image qualitative panels use controlled seed-0 checkpoints, stored external predictions, the same image, a common display threshold, consistent FDI decoding, and shared target crops. Examples include favorable and adverse changes. Pano prior panels recompute scores from all ordinary candidates before native top-k selection, switching only center and/or shape scoring for the same checkpoint. Selected examples explain local mechanisms and do not estimate population frequencies of benefit or harm.')
    heading(document, 'S7.2. Internal tensors and structured-identity diagnostics', 2)
    paragraph(document, 'Main Figs. 13–15 derive from the controlled Mobile combined seed-0 checkpoint. Four learned prototypes per FDI class are L2-normalized; class centroids are formed by averaging then renormalizing. Cosine similarity is computed between supported class centroids. Within-class cohesion is the mean pairwise cosine similarity of the four prototypes. Unsupported FDI slots are masked, rather than interpreted as learned representations.')
    paragraph(document, 'Auxiliary-head confusion matrices use unique ordinary queries, retaining a query when its maximum sigmoid class score is at least 0.30. Class-agnostic Hungarian matching against ground-truth boxes accepts IoU ≥0.50 and yields 2,335 localized queries. This query-level subset differs from the post-top-k detection denominator in Table S12, because native top-k outputs can retain multiple class scores from one query. Factor labels follow quadrant 1–8, within-quadrant position 1–8, and dentition as primary or permanent. Confusion values are normalized by the reference row; these branches are not objectness heads.')
    paragraph(document, 'CMFA and HOOA tensors are captured during an actual forward pass of the same seed-0 checkpoint. CMFA channel gates gA and gB each contain 256 channel values displayed on a 16 × 16 grid; grid location indexes channels and has no anatomical spatial meaning. The CMFA spatial gate is the actual scalar gate at each feature location. HOOA panels show local, context, high-pass, and multiplicative local–context responses, channel gates, spatial gates, input, and output. Response maps summarize root-mean-square magnitude across channels and are normalized independently for display, while sigmoid gate values retain their [0,1] scale. These maps are internal responses, not Grad-CAM, and do not substitute for retrained causal ablations.')
    heading(document, 'S7.3. Statistical and graphical conventions', 2)
    paragraph(document, 'Open symbols in repeated-seed plots represent actual seed outcomes; mean markers and error bars show the mean and sample standard deviation. Seed-paired differences and df = 2 Student-t intervals are calculated from the three actual within-seed differences. No error bars are added to single-run external or internal results. Missing completed measurements are shown as “—” and are excluded from calculations. No smoothing, interpolated training results, synthetic predictions, or generated clinical image content is used.')
    paragraph(document, 'Figures use a common Times New Roman typeface, restrained colors, and concise subplot labels; descriptions and thresholds are placed in captions. Performance figures are generated from numerical records and prediction arrays, and response/prototype figures from captured tensors. Standalone high-resolution raster, vector PDF, and SVG sources accompany the figure assets. Clinical pixels, detection boxes, identities, and gate values are not retouched.')

    evidence = read(EVIDENCE / 'controlled_evidence.json')
    runs = list(csv.DictReader((EVIDENCE / 'all_completed_runs.csv').open(encoding='utf-8-sig')))
    for r in runs:
        r['seed'] = int(r['seed']); r['best_epoch'] = int(r['best_epoch'])
        for k in ['AP', 'AP50', 'AP75', 'AR100', 'params_M', 'Precision', 'Recall', 'F1',
                  'matched_tooth_FDI_accuracy', 'localization_coverage', 'raw_duplicate_image_rate']:
            r[k] = float(r[k])
    add_current_results(document, evidence, runs)
    add_complete_fdi_figures(document)

    # Make Word styles explicit and preserve every existing math element.
    for style_name, size in [('Normal', 11), ('Heading 1', 12), ('Heading 2', 11), ('Caption', 9)]:
        style = document.styles[style_name]; style.font.name = 'Times New Roman'; style.font.size = Pt(size)
    for p in document.paragraphs:
        size = {'Title': 16, 'Heading 1': 12, 'Heading 2': 11, 'Caption': 9}.get(p.style.name, 11)
        for run in p.runs:
            font_run(run, size, True if p.style.name in ['Title', 'Heading 1', 'Heading 2'] else None)
        p.paragraph_format.widow_control = True
    assert [[[c.text for c in r.cells] for r in t.rows] for t in document.tables[:5]] == original_table_values
    assert len(document._element.xpath('.//m:oMath')) == original_math
    # Drop the obsolete image relationships as well as their visible paragraphs.
    used_relations = {v for node in document._element.iter() for k, v in node.attrib.items()
                      if k.startswith('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}')}
    for relation_id, relation in list(document.part.rels.items()):
        if relation.reltype.endswith('/image') and relation_id not in used_relations:
            document.part.drop_rel(relation_id)
    document.save(DEST)
    saved = Document(DEST)
    assert len(saved.tables) == 13
    assert len(saved.inline_shapes) == 11
    assert sum(c.text == EM for t in saved.tables[7:8] for r in t.rows for c in r.cells) >= 20
    text = '\n'.join(p.text for p in saved.paragraphs) + '\n'.join(c.text for t in saved.tables for r in t.rows for c in r.cells)
    forbidden = ['A07', 'Main Figure', 'repository', 'current snapshot', 'pre-revision', 'rebuild_quantitative_figures.py']
    assert not any(w in text for w in forbidden), [w for w in forbidden if w in text]
    report = {'source': str(SOURCE), 'destination': str(DEST), 'tables': len(saved.tables), 'inline_figures': len(saved.inline_shapes),
              'historical_tables_S1_to_S5_numeric_cells_unchanged': True, 'old_visual_tail_replaced': True,
              'existing_math_count_before': original_math, 'existing_math_count_after': len(saved._element.xpath('.//m:oMath')),
              'primary_run_rows': 42, 'internal_sensitivity_rows': 26, 'pending_internal_training_conditions': 4,
              'new_primary_detail_tables': ['S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'],
              'new_figures': ['S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11'],
              'complete_FDI_confusions_include_Missed_and_support': True,
              'optional_SMWG_native_reexport': str(smwg),
              'snapshot': evidence['snapshot_time_Asia_Shanghai']}
    (ROOT / 'supplement_build_audit.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'submission_missing_information.txt').write_text(
        '作者需补充/核实的正式投稿信息（不嵌入论文）\n\n'
        '1. 伦理：Mobile/DSLR图像来源的真实伦理委员会全名、批准编号、批准日期，实际同意或豁免依据；公开Pano数据的适用伦理/许可条款与原始研究引用。不能凭图像去标识化推定豁免。\n'
        '2. 图片发表许可：口内照片用于论文公开发表的患者/监护人授权或有效豁免；DENTEX/第三方图片重用的许可范围。现有数据使用条件须允许图像和权重共享。\n'
        '3. 作者单位：逐位作者对应1–4单位的准确上标映射；当前单位列表并未明确每人对应哪些单位，不应推定。请核实作者顺序、共同第一作者、共同通讯作者及全部通讯资料。\n'
        '4. 数据描述：Mobile与DSLR的实际采集机构与时间、纳入排除标准、标注人员职责、标注训练、争议处理流程，以及相应来源文献。现有图片/牙齿总数不能替代患者来源资料。\n'
        '5. 结果补齐：4个内部训练消融仍用—：HOOA去二阶、HOOA高通与二阶都去、Prototype K2/τ0.20、K4/τ0.50。当前权重的跨设备与统一效率数据需待真实评估后填入主文相应表，不能沿用其他权重的结果。\n'
        '6. 数据边界：当前全部为原划分内部验证，没有独立测试；Mobile/Pano既有重复内容未移除。不要将验证集改称锁定独立测试，或删除已知边界来扩大泛化/SOTA结论。\n'
        '7. 投稿提交：核实所有作者对正文/补充材料、利益冲突、基金、CRediT、AI辅助声明以及图像授权作最终认可。\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
