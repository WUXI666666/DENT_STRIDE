from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
import json
import re

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/manuscript_inventory')
INPUTS = {
    'main': Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_main_visual.docx'),
    'supplement': Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_supplement_visual.docx'),
}
ROOT.mkdir(parents=True, exist_ok=True)

for key, path in INPUTS.items():
    doc = Document(path)
    paragraphs = []
    tables = []
    blocks = []
    preceding = []
    pidx = tidx = 0
    for bidx, child in enumerate(doc.element.body):
        if child.tag == qn('w:p'):
            p = Paragraph(child, doc)
            images = []
            for blip in child.findall('.//' + qn('a:blip')):
                rid = blip.get(qn('r:embed')) or blip.get(qn('r:link'))
                rel = doc.part.rels.get(rid)
                images.append({'relationship': rid, 'target': str(rel.target_ref) if rel else None})
            equations = child.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')
            eq_count = len(equations)
            eq_text = [''.join(x.text or '' for x in e.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}t')) for e in equations]
            item = {'index': pidx, 'body_block': bidx, 'style': p.style.name, 'text': p.text, 'images': images, 'equations': eq_count, 'equation_linear_text': eq_text}
            paragraphs.append(item)
            blocks.append({'type': 'paragraph', **item})
            if p.text.strip():
                preceding.append(p.text)
                preceding = preceding[-3:]
            pidx += 1
        elif child.tag == qn('w:tbl'):
            t = Table(child, doc)
            rows = [[c.text for c in row.cells] for row in t.rows]
            item = {'index': tidx, 'body_block': bidx, 'preceding_paragraphs': preceding, 'rows': rows, 'style': t.style.name if t.style else None}
            tables.append(item)
            blocks.append({'type': 'table', **item})
            tidx += 1
    result = {'path': str(path), 'paragraphs': paragraphs, 'tables': tables, 'blocks': blocks}
    (ROOT / f'{key}_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = []
    for b in blocks:
        if b['type'] == 'paragraph':
            lines.append(f"P{b['index']:03d} [block {b['body_block']} / {b['style']}] {b['text']}")
            if b['images']:
                lines.append('    IMAGES: ' + json.dumps(b['images']))
            if b['equations']:
                lines.append(f"    EQUATIONS: {b['equations']} " + ' ; '.join(b['equation_linear_text']))
        else:
            lines.append(f"TABLE {b['index']:02d} [block {b['body_block']}]\n" + '\n'.join(' | '.join(r) for r in b['rows']))
    (ROOT / f'{key}_inventory.txt').write_text('\n'.join(lines))
    print(f'=== {key}: {pidx} paragraphs, {tidx} tables ===')
    for p in paragraphs:
        if p['style'].startswith('Heading') or re.match(r'^(?:[1-4](?:\.\d+)*\s|Table\s+\d|Figure\s+\d|Abstract|Contributions|Conclusion|Limitations|Code availability)', p['text']):
            print(f"P{p['index']:03d} {p['text']}")
    for t in tables:
        print(f"T{t['index']} block {t['body_block']} rows={len(t['rows'])}: {t['rows'][0]}")
