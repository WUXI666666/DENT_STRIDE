#!/usr/bin/env python3
"""Revise the overview on slide 3 of the supplied editable presentation.

Slides 1 and 2 are retained exactly.  Publication artwork is exported from the
new slide through LibreOffice/PDF; the scene SVG is additionally kept as a
text-editable vector source. No source presentation or model code is modified.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import zipfile
from lxml import etree
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_All_Elements_Editable_v6.pptx')
DRAWING = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_style_revision_2026-10-11/figures/draw_publication_diagrams.py')
SNAPSHOTS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    spec = importlib.util.spec_from_file_location('diagram_source', DRAWING)
    drawing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drawing)
    drawing.OUT = ROOT
    scene = drawing.overview()
    # The underlying routes and all operation nodes are retained from the
    # independently code-checked multiscale diagram. Only the selected domain
    # training settings differ from the previous three-seed run.
    replacements = {
        'Factor + Prototype · SpatialShape':
            ('Factor + Prototype (τ = 0.20) · SpatialShape', 14.5),
        'CMFA + HOOA · Factor + Prototype':
            ('CMFA + HOOA · Factor + Prototype (τ = 0.50)', 14.5),
        'CMFA · DualOpt training':
            ('CMFA · Extra decay (training only)', 16),
    }
    for typ, item in scene.items:
        if typ == 'text' and item['s'] in replacements:
            item['s'], item['size'] = replacements[item['s']]
        if typ == 'rect' and item['x'] == 705 and item['y'] == 748 and item['w'] == 270:
            # Give the center/shape labels clear padding above the lower edge.
            item['h'] = 150
    scene.render('Figure_2_architecture_text_editable')

    inherited = Presentation(SOURCE)
    assert len(inherited.slides) == 3
    old_overview = inherited.slides[2]
    inventory = []
    for i, slide in enumerate(inherited.slides):
        inventory.append({'slide': i + 1, 'shape_count': len(slide.shapes),
                          'texts': [s.text for s in slide.shapes if s.has_text_frame]})
    assert any('Backbone and multiscale' in t for t in inventory[2]['texts'])

    # Scene.ppt creates editable shapes on a new slide. Its original uniform
    # transform is scaled to the inherited canvas while preserving the source
    # presentation's size and every untouched slide.
    temporary = Presentation()
    temporary.slide_width = Inches(12)
    temporary.slide_height = Inches(8.7)
    drawn = scene.ppt(temporary)
    old_sp_tree = old_overview.shapes._spTree
    for shape in list(old_overview.shapes):
        old_sp_tree.remove(shape._element)
    # Original scene uses a 12 by 8.7 in transform. Fit it uniformly to this
    # slide's 13 1/3 by 7.5 in canvas, never resizing preserved source slides.
    factor = min(inherited.slide_width / temporary.slide_width,
                 inherited.slide_height / temporary.slide_height)
    dx = (inherited.slide_width - temporary.slide_width * factor) / 2
    dy = (inherited.slide_height - temporary.slide_height * factor) / 2
    for shape in list(drawn.shapes):
        shape.left = int(dx + shape.left * factor)
        shape.top = int(dy + shape.top * factor)
        shape.width = int(shape.width * factor)
        shape.height = int(shape.height * factor)
        if shape.has_text_frame:
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size is not None:
                        r.font.size = int(r.font.size * factor)
        element = shape._element
        element.getparent().remove(element)
        old_sp_tree.insert_element_before(element, 'p:extLst')
    output = ROOT / 'DENT_STRIDE_All_Elements_Editable_v6_selected_20261012.pptx'
    # python-pptx may normalize XML serialization on otherwise untouched
    # slides. Rebuild the package from the original, substituting slide3.xml
    # alone; this preserves all other user slides/assets/package metadata.
    candidate = ROOT / 'overview_candidate.pptx'
    inherited.save(candidate)
    with zipfile.ZipFile(SOURCE) as original, zipfile.ZipFile(candidate) as revised:
        revised_overview = revised.read('ppt/slides/slide3.xml')
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as final:
            for entry in original.infolist():
                data = revised_overview if entry.filename == 'ppt/slides/slide3.xml' else original.read(entry.filename)
                final.writestr(entry, data)
    assert sha(SOURCE) == source_hash
    with zipfile.ZipFile(SOURCE) as old, zipfile.ZipFile(output) as new:
        for i in [1, 2]:
            name = f'ppt/slides/slide{i}.xml'
            assert old.read(name) == new.read(name), f'Preserved slide {i} changed'
    check = Presentation(output)
    assert len(check.slides) == 3
    assert not any(s.shape_type == 13 for s in check.slides[2].shapes)
    for shape in check.slides[2].shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= check.slide_width
        assert shape.top + shape.height <= check.slide_height
        if shape.has_text_frame:
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    assert r.font.name == 'Times New Roman'
    svg = etree.parse(str(ROOT / 'Figure_2_architecture_text_editable.svg'))
    assert not svg.xpath('//*[local-name()="image"]')
    assert len(svg.xpath('//*[local-name()="text"]')) > 50
    caption = (
        'Figure 2. Acquisition-configured DENT-STRIDE architecture. '
        '(A) HGNetv2-B0 features at strides 8, 16, and 32 are projected to 256 channels. '
        'The highest-stride transformer precedes top-down FPN and bottom-up PAN aggregation. '
        'Four fusion sites use CMFA in Mobile and DSLR, and concatenation in Pano; '
        'the base fusion blocks restore 256 channels. HOOA refines the three output levels in Mobile. '
        '(B) Decoder queries supply box and FDI heads and structured identity supervision. '
        'The learned bank contains four prototypes per class (K = 52 in Mobile and 32 in Pano), '
        'using temperatures 0.50 and 0.20, respectively. Prototype scores provide training-only '
        'auxiliary supervision; factor-logit correction is enabled only in Mobile. '
        'The Pano SpatialShape prior modifies candidate logits before sigmoid scoring and top-300 '
        'selection. Cohort-specific label maps decode predictions to clinical FDI codes. '
        'The DSLR configuration applies additional parameter-rank decay during training, '
        'without initialization rollback. Gray paths bypass inactive modules; dashed paths '
        'denote auxiliary supervision or training-derived prior estimation. '
        'Tensor volumes and prior glyphs are schematic.'
    )
    (ROOT / 'Figure_2_caption.txt').write_text(caption + '\n', encoding='utf-8')
    manifest = {
        'source_pptx': str(SOURCE), 'source_sha256': source_hash,
        'modified_pptx': str(output), 'modified_sha256': sha(output),
        'modified_slide': 3, 'preserved_slides_exact_xml': [1, 2],
        'overview_shape_count': len(check.slides[2].shapes),
        'overview_contains_raster_shapes': False,
        'source_inventory': inventory,
        'caption': caption,
        'selected_configs': {
            'Mobile': str(SNAPSHOTS / 'configs/mobile_proto_k4_tau0p50/seed_2.yml'),
            'Pano': str(SNAPSHOTS / 'configs/pano_spatialshape_original/seed_3.yml'),
            'DSLR': str(SNAPSHOTS / 'configs/dslr_cmfa_extra_decay_only/seed_3.yml'),
        },
        'scientific_checks': [
            'P3/P4/P5 are stride 8/16/32 and backbone channels 256/512/1024.',
            'FPN has two projected-channel cross-scale fusions; PAN has two.',
            'The upper PAN downsample originates from the completed lower PAN output.',
            'CMFA inputs C256 and output 2C512 precede base fusion C256.',
            'HOOA applied to Mobile outputs only; Pano and DSLR bypass it.',
            'Factor correction Mobile only; Pano factor heads only supervise training.',
            'Queries and normalized prototype bank separately supply cosine similarity.',
            'Prototype bank Kclasses x 4; tau .50 Mobile/.20 Pano, training auxiliary only.',
            'Pano priors consume box candidates and adjust logits before top-300.',
            'Only DSLR additional decay enabled; rollback weight zero.',
            'Shared base upsampling/positioning is not claimed as a new module.',
        ],
    }
    (ROOT / 'architecture_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'pptx': str(output), 'svg': str(ROOT / 'Figure_2_architecture_text_editable.svg'),
                      'modified_slide': 3, 'shapes': len(check.slides[2].shapes)}, indent=2))

if __name__ == '__main__':
    main()
