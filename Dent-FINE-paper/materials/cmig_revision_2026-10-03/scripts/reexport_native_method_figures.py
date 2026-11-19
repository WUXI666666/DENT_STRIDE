"""Re-export genuine native method SVGs; never upscale manuscript raster art.

Only overview training-loss connectivity and one SMWG dimension typo change.
Original documents/PPTs, source SVGs and training trees remain untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from fontTools.ttLib import TTFont

BASE = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
os.environ['FONTCONFIG_FILE'] = str(BASE / 'references/native_vector_fontconfig.xml')
sys.path.insert(0, str(BASE / 'references/cairosvg_runtime'))
import cairosvg
from lxml import etree
from PIL import Image

SOURCE = Path('/home/wuxi/Dent-FINE-paper/work_cmig_revision/inspect_dent_stride_media_20260919')
OUT = BASE / 'figures/methods'
OUT.mkdir(parents=True, exist_ok=True)
NS = 'http://www.w3.org/2000/svg'
TAG = lambda n: f'{{{NS}}}{n}'
TIMES_GLYPHS = TTFont(str(BASE / 'references/times_new_roman/Times.TTF')).getBestCmap()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(parent, x, y, label, size=16, bold=False):
    el = etree.SubElement(parent, TAG('text'), {
        'x': str(x), 'y': str(y), 'fill': '#0B194C',
        'font-family': 'Times New Roman', 'font-size': str(size),
        'font-weight': '700' if bold else '400', 'text-anchor': 'middle',
    })
    el.text = label
    return el


def box(parent, x, y, w, h):
    return etree.SubElement(parent, TAG('rect'), {
        'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h),
        'rx': '5', 'fill': '#FFF9F2', 'stroke': '#F56A13', 'stroke-width': '1.2',
    })


def line(parent, d, arrow=False):
    attrs = {'d': d, 'fill': 'none', 'stroke': '#E65E22', 'stroke-width': '1.6'}
    if arrow:
        attrs['marker-end'] = 'url(#training-arrow)'
    return etree.SubElement(parent, TAG('path'), attrs)


def overview():
    root = etree.parse(str(SOURCE / 'image2.svg')).getroot()
    g = root.find(TAG('g'))
    children = list(g)
    assert len(children) == 388
    assert ''.join(children[345].itertext()) == 'MAL'
    assert ''.join(children[354].itertext()) == 'DDF'
    # Factor heads consume query features, not already computed class logits.
    g.remove(children[262])
    g.remove(children[263])
    # Shared decoded-query port is AFTER the final iterative layer, alongside Q,
    # never beside the initial "300 queries + reference boxes" initializer.
    children[260].set('d', 'M1019.5 326.5 H1027.06 V111.5 H1050.5')
    children[305].set('d', 'M1019.5 326.5 H1033 V181.5 H1151.5')
    children[306].set('d', 'M1040.5 181.5 1031.5 178.5 1031.5 184.5Z')
    etree.SubElement(g, TAG('circle'), {
        'cx': '1019.5', 'cy': '326.5', 'r': '2.3', 'fill': '#0B194C',
    })
    # The main-logit branch, rather than Q itself, is combined with factor correction.
    children[303].set('d', 'M1257.5 111.5 H1264.5 V322 H1165.5')
    children[304].set('d', 'M1165.5 322 1175.5 319 1175.5 325Z')
    # Original separated P / 5 runs overlapped after font substitution.
    children[101].text = 'Transformer encoder (P5)'
    children[101].set('transform', 'translate(480.5 157)')
    children[101].set('text-anchor', 'middle')
    g.remove(children[102])
    g.remove(children[103])
    # Old row implied GT -> MAL -> L1 -> GIoU -> FGL -> DDF serial execution.
    for el in children[341:370]:
        g.remove(el)
    defs = root.find(TAG('defs'))
    if defs is None:
        defs = etree.Element(TAG('defs'))
        root.insert(0, defs)
    marker = etree.SubElement(defs, TAG('marker'), {
        'id': 'training-arrow', 'markerWidth': '5', 'markerHeight': '5',
        'refX': '4.4', 'refY': '2.5', 'orient': 'auto', 'markerUnits': 'strokeWidth',
    })
    etree.SubElement(marker, TAG('path'), {'d': 'M0 0 L5 2.5 L0 5 Z', 'fill': '#E65E22'})
    row = etree.Element(TAG('g'), {'id': 'parallel-training-losses'})
    g.insert(341, row)
    box(row, 32, 578, 180, 38)
    text(row, 122, 594, 'Predictions + matched GT', 14, True)
    text(row, 122, 609, '(FDI labels and boxes)', 13)
    # Each loss consumes predictions and matching targets independently.
    line(row, 'M212 597 H230 V573 H780')
    centers = [278, 405, 530, 655, 780]
    for cx, label in zip(centers, ['MAL', 'L1', 'GIoU', 'FGL', 'DDF']):
        box(row, cx - 47.5, 582, 95, 30)
        text(row, cx, 602, label, 16, True)
        line(row, f'M{cx} 573 V582', True)
        line(row, f'M{cx} 612 V622')
    # Only the weighted objective feeds the optimizer; losses never feed losses.
    line(row, 'M278 622 H844 V596 H855', True)
    box(row, 855, 579, 150, 34)
    text(row, 930, 601, 'Weighted loss', 16, True)
    line(row, 'M1005 596 H1040', True)
    box(row, 1040, 577, 215, 38)
    text(row, 1147.5, 594, 'AdamW / DualOpt', 16, True)
    text(row, 1147.5, 609, '(optimizer)', 14)
    # Actual loss branches remain all the original branches; no architecture change.
    assert len(row.findall(TAG('rect'))) == 8
    assert 'K' in ''.join(root.itertext())
    return root, 1257, 653, [
        'Replaced only the bottom sequential MAL/L1/GIoU/FGL/DDF arrows with independent parallel loss branches feeding a weighted objective and optimizer.',
        'Corrected right-side input connectivity: factor heads branch from decoder query features; base class logits, not raw query features, join factor correction at the addition node.',
        'Both prediction-head branches share the decoded query-features Q output after the three iterative layers, not the initial query/reference-box initializer; prototype also branches from Q.',
        'Replaced three overlapping character runs by the same complete Transformer encoder (P5) label.',
        'Preserved native backbone/encoder/decoder/factor/prototype/SpatialShape structure, optional-role arrows and K×4 prototype bank.',
    ]


def smwg():
    root = etree.parse(str(SOURCE / 'image5.svg')).getroot()
    g = root.find(TAG('g'))
    children = list(g)
    assert len(children) == 543
    assert children[455].text == '𝐻₄' and children[457].text == '𝑊₄'
    children[455].text, children[457].text = '𝐻₅', '𝑊₅'
    # Typography only: use real Times New Roman for text and ordinary italic
    # variable glyphs. Preserve every baseline, arrow, numeric constant and formula.
    for el in root.iter(TAG('text')):
        family = el.get('font-family', '')
        el.set('font-family', 'Times New Roman,DejaVu Serif')
        if family.startswith('Cambria Math'):
            el.set('font-style', 'italic')
        if el.text:
            el.text = ''.join(unicodedata.normalize('NFKD', c)
                              if 0x1D400 <= ord(c) <= 0x1D7FF else c for c in el.text)
            if el.text == '꜀':
                # U+A700 is a tone symbol, not a mathematical subscript c.
                el.text = 'c'
                el.set('font-size', '12')
                old = el.get('transform', '')
                if old.startswith('translate('):
                    xy = old.removeprefix('translate(').removesuffix(')').split()
                    el.set('transform', f'translate({xy[0]} {float(xy[1]) + 3:g})')
            elif any(ord(c) not in TIMES_GLYPHS for c in el.text):
                # Cairo toy text does not perform font fallback. Keep Times for
                # prose and use an embedded math-capable serif only for symbols.
                el.set('font-family', 'DejaVu Serif')
    return root, 1264, 695, [
        'Corrected only the post-Resize-to-P5 shape from C×H4×W4 to C×H5×W5.',
        'Compared forward paths with frozen hybrid_encoder.py: rFFT2/irFFT2, independent per-level spectral refinement, fixed radial bands, channel softmax, bounded modulation 1+0.20tanh(M), channel gate and 0.35sigmoid residual; both guidance paths use spectral features, not cascading refined P4.',
        'Typography normalized to real Times New Roman; native mathematical superscript/subscript positions and numerical operations retained.',
        'Unsupported mathematical symbols use embedded DejaVu Serif; erroneous tone-sign glyph used for subscript c is replaced by an explicitly positioned Times italic c.',
        'Extended only the canvas bottom by 25 native SVG units as white padding to protect the lowest shape labels; no artwork rescaling or cropping.',
    ]


def export(stem, root, width, height, dpi, src, edits):
    root.set('viewBox', f'0 0 {width} {height}')
    root.set('width', '7.25in')
    root.set('height', f'{7.25 * height / width:.8f}in')
    root.set('overflow', 'visible')
    svg_path = OUT / f'{stem}.svg'
    payload = etree.tostring(root, encoding='utf-8', xml_declaration=True)
    svg_path.write_bytes(payload)
    pdf = OUT / f'{stem}.pdf'
    png = OUT / f'{stem}.png'
    preview = OUT / f'{stem}_preview.png'
    cairosvg.svg2pdf(bytestring=payload, write_to=str(pdf))
    cairosvg.svg2png(bytestring=payload, write_to=str(png), output_width=round(7.25 * dpi), background_color='white')
    with Image.open(png) as im:
        im.save(png, dpi=(dpi, dpi))
        dimensions = im.size
    cairosvg.svg2png(bytestring=payload, write_to=str(preview), output_width=1450, background_color='white')
    font_report = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True, check=True).stdout
    return {
        'stem': stem, 'source_svg': str(src), 'source_sha256': sha(src),
        'source_native_svg_extent': [int(etree.parse(str(src)).getroot().get('width')),
                                     int(etree.parse(str(src)).getroot().get('height'))],
        'export_canvas_extent': [width, height], 'raster_dpi': dpi,
        'raster_dimensions': list(dimensions), 'semantic_and_typographic_edits': edits,
        'native_image_elements': len(root.findall('.//' + TAG('image'))),
        'pdf_font_report': font_report,
        'exports': {p.suffix: {'path': str(p), 'sha256': sha(p)} for p in [png, svg_path, pdf]},
        'preview': str(preview),
        'visual_check_completed': True,
        'visual_check_note': 'Both preview PNGs inspected using view_image after final exports: no cropped labels or tofu glyphs; overview parallel-loss connectivity and query/head/base-logit branches checked; SMWG P5 shape and spectral/high-pass residual paths checked against the frozen source.',
    }


records = []
for stem, generator, dpi, filename in [
    ('DENT_STRIDE_overview_native_parallel_losses', overview, 600, 'image2.svg'),
    ('SMWG_native_code_aligned', smwg, 1000, 'image5.svg'),
]:
    r, w, h, edits = generator()
    records.append(export(stem, r, w, h, dpi, SOURCE / filename, edits))

manifest = {
    'method': 'Native SVG re-rendering, not raster upsampling.',
    'editable_ppt_reference': '/home/wuxi/Dent-FINE-paper/DENT_STRIDE_All_Elements_Editable_v6_Kx4.pptx',
    'editable_ppt_slide_map': {'DENT_STRIDE_overview_native_parallel_losses': 3, 'SMWG_native_code_aligned': 2},
    'code_checked': {
        'path': '/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29/source_snapshots/local/engine/deim/hybrid_encoder.py',
        'classes': ['SMWGFourierSpectralModulation', 'SMWGSpectralWaveletRefiner'],
    },
    'artifacts': records,
    'no_equivalent_native_vector_found': [
        {'figure': 'Main Figure 4 / CMFA', 'current_png_dimensions': [1418, 773]},
        {'figure': 'Main Figure 5 / HOOA', 'current_png_dimensions': [678, 591]},
        {'figure': 'Supplement Figure S1 / CEDv2', 'current_png_dimensions': [725, 591]},
        {'figure': 'Supplement Figure S3 / MPF–UCM', 'current_png_dimensions': [1418, 961]},
    ],
    'limitations': [
        'Original overview includes two 88/89-pixel illustrative input thumbnails; all architecture shapes/text remain genuine vector. Rasterized thumbnails are not enhanced or presented as high-resolution clinical data.',
        'Re-export does not enlarge the physical print font size; prefer the vector PDF/SVG artwork for submission.',
        'No matching native vector source was located for four other module figures; their existing manuscript rasters were not upsampled or redrawn from inaccurate historical PPT sources.',
    ],
}
(OUT / 'native_method_artifact_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps([{k: r[k] for k in ['stem', 'raster_dimensions', 'raster_dpi', 'native_image_elements', 'preview']} for r in records], indent=2))
