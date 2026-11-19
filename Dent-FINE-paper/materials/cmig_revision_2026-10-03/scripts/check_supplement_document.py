"""Check document content, measured table cells, figure references, and rendered text."""
from pathlib import Path
from hashlib import sha256
import csv
import json
import re
import subprocess
from zipfile import ZipFile
from docx import Document
from docx.oxml.ns import qn

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
DOC = Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_supplement_controlled_results_20261003.docx')
PDF = ROOT / 'render_supplement' / (DOC.stem + '.pdf')
source = Document('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_supplement_visual.docx')
document = Document(DOC)
paragraph_text = '\n'.join(p.text for p in document.paragraphs)
full_text = paragraph_text + '\n' + '\n'.join(c.text for t in document.tables for r in t.rows for c in r.cells)
for a, b in zip(source.tables[:5], document.tables[:5]):
    assert [[c.text for c in r.cells] for r in a.rows] == [[c.text for c in r.cells] for r in b.rows]
assert [int(x) for x in re.findall(r'^Fig\. S(\d+)\.', paragraph_text, re.M)] == list(range(1, 12))
assert [int(x) for x in re.findall(r'^Table S(\d+)\.', paragraph_text, re.M)] == list(range(1, 14))
assert len(document.tables[6].rows) == 43
assert len(document.tables[7].rows) == 27
assert len(document.tables[8].rows) == 21
assert len(document.tables[9].rows) == 13
assert len(document.tables[10].rows) == 5
assert len(document.tables[11].rows) == 19
assert len(document.tables[12].rows) == 15
pending = [r for r in document.tables[7].rows[1:] if r.cells[2].text == '—']
assert len(pending) == 4
assert all(all(c.text == '—' for c in r.cells[2:]) for r in pending)
actual_runs = {(r['cohort'], r['configuration'], r['seed']): r for r in csv.DictReader((ROOT / 'evidence/primary_seed_results.csv').open(encoding='utf-8-sig'))}
table_AP_values = [float(r.cells[3].text) for r in document.tables[6].rows[1:]]
source_AP_values = sorted(round(float(r['AP']), 3) for r in actual_runs.values())
assert sorted(table_AP_values) == source_AP_values
for table in document.tables:
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    assert run.font.name == 'Times New Roman' and run.font.size.pt == 9
for word in ['A07', 'Main Figure', 'PUBLICATION-SAFETY', 'AUTHOR COMPLETION', 'source-rounded', 'current snapshot', 'dirty Git']:
    assert word not in full_text
smwg = ROOT / 'figures/methods/SMWG_native_code_aligned.png'
smwg_embed = document.inline_shapes[1]._inline.xpath('.//a:blip')[0].get(qn('r:embed'))
assert sha256(document.part.related_parts[smwg_embed].blob).hexdigest() == sha256(smwg.read_bytes()).hexdigest()
for cohort, shape_index in [('Pano', 8), ('Mobile', 9), ('DSLR', 10)]:
    evidence = json.loads((ROOT / f'figures/errors/{cohort}_complete_FDI_confusion.json').read_text())
    summary = evidence['summary']
    assert sum(map(sum, evidence['raw_counts'])) == summary['validation_GT']
    assert [sum(row) for row in evidence['raw_counts']] == evidence['val_support']
    assert len(evidence['fdi_order']) == (32 if cohort == 'Pano' else 52)
    for values, support in zip(evidence['row_percent'], evidence['val_support']):
        assert (all(value is None for value in values) if not support else abs(sum(values) - 100) < 1e-8)
    shape_rel = document.inline_shapes[shape_index]._inline.xpath('.//a:blip')[0].get(qn('r:embed'))
    plot = Path(evidence['figure_files']['png'])
    assert sha256(document.part.related_parts[shape_rel].blob).hexdigest() == sha256(plot.read_bytes()).hexdigest()
with ZipFile(DOC) as package:
    images = [p for p in package.namelist() if p.startswith('word/media/')]
    assert len(images) == 11
assert PDF.is_file()
rendered = ROOT / 'render_supplement/supplement_rendered.txt'
subprocess.run(['pdftotext', '-layout', str(PDF), str(rendered)], check=True)
pages = [p.strip() for p in rendered.read_text().split('\f') if p.strip()]
assert len(pages) == 20 and all(len(p) > 400 for p in pages)
fonts = subprocess.check_output(['pdffonts', str(PDF)], text=True)
assert 'TimesNewRomanPSMT' in fonts and 'TimesNewRomanPS-BoldMT' in fonts
assert 'LiberationSerif' not in fonts
report = {'docx': str(DOC), 'docx_sha256': sha256(DOC.read_bytes()).hexdigest(),
          'pdf': str(PDF), 'pdf_sha256': sha256(PDF.read_bytes()).hexdigest(), 'pages': len(pages),
          'no_empty_rendered_pages': True, 'real_Times_New_Roman_embedded': True,
          'tables_S1_to_S5_unchanged_numeric_cells': True, 'unique_figures': list(range(1, 12)),
          'unique_tables': list(range(1, 14)), 'primary_rows': 42, 'internal_setting_rows': 26,
          'pending_rows': 4, 'actual_primary_AP_cells_rechecked': True,
          'table_font': 'Times New Roman 9 pt', 'old_unused_images_removed_from_package': True,
          'visual_inspection_pages': [1, 2, 9, 14, 16, 18, 19, 20],
          'FDI_error_figures': 'All 32/52 FDI classes; gray unsupported validation rows; explicit Missed column; train/val support; separate unmatched prediction counts; independent metrics-total reconciliation.',
          'SMWG_embedded_source_hash_verified': True,
          'complete_FDI_embedded_source_hashes_verified': True,
          'known_retained_artwork_limit': 'Historical CEDv2 raster is 725x591. New quantitative/prototype plots have standalone high-resolution PNG and vector PDF/SVG sources.'}
(ROOT / 'supplement_document_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
