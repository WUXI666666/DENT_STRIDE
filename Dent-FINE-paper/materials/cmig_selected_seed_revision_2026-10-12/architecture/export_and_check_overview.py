#!/usr/bin/env python3
"""Export the modified PowerPoint overview to genuine vector publication files."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '.deps'))
from pypdf import PdfReader, PdfWriter
from lxml import etree
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PPT_PDF = ROOT / 'DENT_STRIDE_All_Elements_Editable_v6_selected_20261012.pdf'

def main():
    # Crop the centered architecture drawing out of the original inherited
    # widescreen slide. No shape, label, or arrow is cropped.
    full = PdfReader(PPT_PDF)
    page = full.pages[2]
    # overview scene exactly 1400 x 980, centered through the uniform scene
    # ->12 x8.7 -> inherited13.333 x7.5 mapping used by the editable builder.
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    factor = min(width_pt / (12 * 72), height_pt / (8.7 * 72))
    inner_w = 12 * 72 * factor
    scene_h = (980 * 12 / 1400) * 72 * factor
    left = (width_pt - inner_w) / 2
    top = (height_pt - scene_h) / 2
    # Retain two-point safety margin beyond the scene box.
    page.cropbox.lower_left = (max(0, left - 2), max(0, top - 2))
    page.cropbox.upper_right = (min(width_pt, left + inner_w + 2), min(height_pt, top + scene_h + 2))
    page.mediabox.lower_left = page.cropbox.lower_left
    page.mediabox.upper_right = page.cropbox.upper_right
    out_pdf = ROOT / 'Figure_2_DENT_STRIDE_architecture.pdf'
    single = PdfWriter()
    single.add_page(page)
    with out_pdf.open('wb') as stream:
        single.write(stream)
    # pdftocairo writes path geometry and embedded vector glyph outlines;
    # it never wraps the slide as a single raster image.
    out_svg = ROOT / 'Figure_2_DENT_STRIDE_architecture.svg'
    subprocess.run(['pdftocairo', '-svg', str(out_pdf), str(out_svg)], check=True)
    for dpi, stem in [(600, 'Figure_2_DENT_STRIDE_architecture'),
                      (160, 'Figure_2_DENT_STRIDE_architecture_preview')]:
        subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', str(dpi),
                        str(out_pdf), str(ROOT / stem)], check=True)
    svg = etree.parse(str(out_svg))
    image_nodes = svg.xpath('//*[local-name()="image"]')
    paths = svg.xpath('//*[local-name()="path"]')
    assert len(image_nodes) == 0 and len(paths) > 100
    # Verify all rendered words remain inside the exported PDF's crop.
    bbox_xml = subprocess.check_output(['pdftotext', '-bbox', str(out_pdf), '-'])
    parsed = etree.fromstring(bbox_xml)
    text_outside = []
    paper_page = parsed.xpath('//*[local-name()="page"]')[0]
    rendered_width, rendered_height = float(paper_page.get('width')), float(paper_page.get('height'))
    for word in parsed.xpath('//*[local-name()="word"]'):
        coords = [float(word.get(k)) for k in ['xMin', 'yMin', 'xMax', 'yMax']]
        if coords[0] < 0 or coords[1] < 0 or coords[2] > rendered_width or coords[3] > rendered_height:
            text_outside.append(word.text)
    assert not text_outside, text_outside
    text = subprocess.check_output(['pdftotext', str(out_pdf), '-'], text=True)
    font_xml = subprocess.check_output(['pdftohtml', '-xml', '-fontfullname', '-stdout', str(out_pdf)])
    font_doc = etree.fromstring(font_xml)
    actual_fonts = sorted({s.get('family') for s in font_doc.xpath('//fontspec')})
    assert actual_fonts and all('TimesNewRoman' in s for s in actual_fonts), actual_fonts
    for term in ['0.50', '0.20', 'Extra decay', 'HOOA', 'SpatialShape', 'Top 300', 'FDI map']:
        assert term in text, term
    forbidden = ['SMWG', 'CEDv2', 'MPF', 'UCM', 'QueryFusion', 'rollback']
    assert not any(s in text for s in forbidden)
    manifest = json.loads((ROOT / 'architecture_manifest.json').read_text())
    manifest['export'] = {'svg_from_modified_pptx': str(out_svg),
                          'pdf': str(out_pdf), 'png_dpi': 600,
                          'svg_image_nodes': len(image_nodes), 'svg_vector_paths': len(paths),
                          'text_outside_canvas': text_outside, 'page_count': len(full.pages),
                          'actual_rendered_fonts': actual_fonts}
    (ROOT / 'architecture_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['export'], indent=2))

if __name__ == '__main__':
    main()
