"""Render review previews and record final DOCX/PDF structural checks."""
from pathlib import Path
import json
import re
import subprocess
from PIL import Image, ImageDraw

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_seed_revision_2026-10-12')
OUT = ROOT / 'page_previews'
OUT.mkdir(exist_ok=True)
report = {}
for kind in ['main', 'supplement']:
    stem = 'DENT_STRIDE_CMIG_' + kind + '_selected_seed_revision_20261012'
    pdf = ROOT / 'rendered' / (stem + '.pdf')
    text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
    pages = text.rstrip('\f\n').split('\f')
    details = []
    previews = []
    for i, page in enumerate(pages, 1):
        name = OUT / (kind + '_' + f'{i:02}')
        subprocess.run(['pdftoppm', '-f', str(i), '-l', str(i), '-scale-to', '1300',
                        '-png', '-singlefile', str(pdf), str(name)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        previews.append(name.with_suffix('.png'))
        labels = [l.strip() for l in page.splitlines()
                  if re.match(r'^(?:Fig\. [\dS]|Table |[3456]\.\d|S\d\.)', l.strip())]
        details.append({'page': i, 'text_characters': len(page.strip()), 'labels': labels})
    for start in range(0, len(previews), 6):
        canvas = Image.new('RGB', (1050, 2310), 'white')
        painter = ImageDraw.Draw(canvas)
        for local, p in enumerate(previews[start:start + 6]):
            im = Image.open(p).convert('RGB')
            im.thumbnail((505, 720))
            x = 10 + (local % 2) * 525
            y = 25 + (local // 2) * 770
            painter.text((x, y - 16), kind + ' page ' + str(start + local + 1), fill='black')
            canvas.paste(im, (x, y))
        canvas.save(OUT / (kind + '_contact_' + str(start // 6 + 1) + '.png'))
    assert all(x['text_characters'] > 30 for x in details), 'Unexpected empty page'
    assert '54.205' in text and '55.459' in text and '72.908' in text
    assert 'post-hoc' in text
    assert not any(x in text for x in ['53.897 ± 0.430', '55.219 ± 0.327', '72.293 ± 0.651'])
    if kind == 'main':
        for n in range(1, 14):
            assert ('Fig. ' + str(n) + ('.' if n != 12 else 'A.')) in text, n
        for n in range(1, 10):
            assert 'Table ' + str(n) + '.' in text, n
    else:
        for n in range(1, 12):
            assert 'Fig. S' + str(n) + '.' in text, n
        for n in range(1, 11):
            assert 'Table S' + str(n) + '.' in text, n
    fonts = subprocess.check_output(['pdffonts', str(pdf)], text=True)
    assert 'TimesNewRomanPSMT' in fonts, 'TNR not actually embedded in review PDF'
    report[kind] = {'pages': len(pages), 'page_details': details, 'embedded_fonts': fonts,
                    'pdf': str(pdf), 'preview_directory': str(OUT)}
(ROOT / 'render_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: {'pages': v['pages'], 'pdf': v['pdf']} for k, v in report.items()}, indent=2))
