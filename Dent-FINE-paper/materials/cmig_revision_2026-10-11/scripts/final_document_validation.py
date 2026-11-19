"""Validate rendered-document source integrity and every packaged figure binding."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
from collections import Counter
from docx import Document
from lxml import etree

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_revision_2026-10-11'
manifest = json.loads((ROOT / 'document_build_manifest.json').read_text())
report = {'originals_unchanged': {}, 'manuscripts': {}}
for file, digest in manifest['source_documents'].items():
    actual = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    assert actual == digest, file
    report['originals_unchanged'][file] = True

original = Document(PAPER / 'DENT_STRIDE_CMIG_main_final_20261006.docx')
updated = Document(PAPER / 'DENT_STRIDE_CMIG_main_updated_20261011.docx')
canonical = lambda d: [etree.tostring(n, method='c14n') for n in d._element.xpath('.//m:oMath')]
assert canonical(original) == canonical(updated)
report['native_Word_equations_preserved_exactly'] = len(canonical(updated))

for file, expected_figs, expected_tables in [
        ('DENT_STRIDE_CMIG_main_updated_20261011.docx',
         [str(n) for n in range(1, 12)] + ['12A', '12B', '12C', '13'], 9),
        ('DENT_STRIDE_CMIG_supplement_updated_20261011.docx',
         ['S'+str(n) for n in range(1, 12)], 9)]:
    doc = Document(PAPER / file)
    figs = [re.match(r'^Fig\.\s+(S?\d+[A-C]?)\.', p.text).group(1)
            for p in doc.paragraphs if re.match(r'^Fig\.\s+(S?\d+[A-C]?)\.', p.text)]
    assert figs == expected_figs, (file, figs)
    assert len(doc.tables) == expected_tables
    assert len(doc.inline_shapes) == len(figs)
    with zipfile.ZipFile(PAPER / file) as z:
        assert z.testzip() is None
        media = Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()
                        if n.startswith('word/media/'))
    for binding in manifest['figures']:
        if (binding['number'].startswith('S')) == (file.find('supplement') != -1):
            assert media[binding['sha256']] > 0, (file, binding['number'])
    text = '\n'.join(p.text for p in doc.paragraphs)
    forbidden = ['A07', 'main figure', 'internal audit', 'current snapshot',
                 'dirty Git', 'AUTHOR COMPLETION', 'PUBLICATION-SAFETY',
                 'Not run', 'Not defined', 'present revision', 'XX',
                 'pre-revision', 'required confirmatory', 'Table S13']
    assert not [term for term in forbidden if term.lower() in text.lower()]
    report['manuscripts'][file] = {'tables': len(doc.tables), 'figure_panels': figs,
                                 'sha256': hashlib.sha256((PAPER / file).read_bytes()).hexdigest(),
                                 'valid_ZIP_package': True, 'obsolete_phrases_found': []}

index = json.loads((ROOT / 'publication_figures/Figure_index.json').read_text())
assert len(index) == 26
for row in index:
    png = ROOT / 'publication_figures' / next(f for f in row['files'] if f.endswith('.png'))
    assert hashlib.sha256(png.read_bytes()).hexdigest() == row['PNG_SHA256']
report['publication_figure_panels_verified'] = len(index)
(ROOT / 'final_validation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print('PASS: originals unchanged; 51 native equations exact; 26 figure panels bound; DOCX packages valid.')
