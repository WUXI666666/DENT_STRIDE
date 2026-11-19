"""Package only figures actually embedded in the two revised manuscripts."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile
from docx import Document
from docx.oxml.ns import qn

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_revision_2026-10-11'
DEST = ROOT / 'publication_figures'
OLD = PAPER / 'materials/cmig_final_2026-10-06/publication_figures'
DEST.mkdir(parents=True, exist_ok=True)
build = json.loads((ROOT / 'document_build_manifest.json').read_text())
bindings = {r['number']: Path(r['path']) for r in build['figures']}
bindings.update({str(n): OLD / f'Figure_{n}.png' for n in [1, 3]})
bindings['2'] = ROOT / 'module_figures/Figure_2_overview.png'
bindings['4'] = ROOT / 'module_figures/Figure_4_CMFA.png'
bindings['5'] = ROOT / 'module_figures/Figure_5_HOOA.png'

index = []
for manuscript, label in [('DENT_STRIDE_CMIG_main_updated_20261011.docx', 'Main'),
                          ('DENT_STRIDE_CMIG_supplement_updated_20261011.docx', 'Supplement')]:
    doc = Document(PAPER / manuscript)
    image_blob = None
    for p in doc.paragraphs:
        blips = p._element.xpath('.//a:blip')
        if blips:
            assert len(blips) == 1
            image_blob = doc.part.related_parts[blips[0].get(qn('r:embed'))].blob
        match = re.match(r'^Fig\.\s+(S?\d+[A-C]?)\.\s+', p.text)
        if not match:
            continue
        number = match.group(1)
        assert image_blob is not None, ('Missing figure image', number)
        source = bindings[number]
        assert source.read_bytes() == image_blob, ('Wrong packaged raster', number)
        files = []
        for extension in ['.png', '.svg', '.pdf']:
            companion = source.with_suffix(extension)
            if companion.is_file():
                target = DEST / f'Figure_{number}{extension}'
                shutil.copy2(companion, target)
                files.append(target.name)
        index.append({'manuscript': label, 'figure': f'Figure {number}',
                      'files': files, 'caption': p.text,
                      'PNG_SHA256': hashlib.sha256(image_blob).hexdigest()})
        image_blob = None

assert len(index) == 26, len(index)
for name in ['Figures_4_5_editable.pptx', 'Figure_2_overview_editable.pptx']:
    source = ROOT / 'module_figures' / name
    if source.is_file():
        shutil.copy2(source, DEST / name)

(DEST / 'Figure_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2))
with (DEST / 'Figure_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Manuscript', 'Figure', 'Files', 'Caption'])
    for row in index:
        writer.writerow([row['manuscript'], row['figure'], '; '.join(row['files']), row['caption']])

archive = PAPER / 'DENT_STRIDE_CMIG_updated_figures_20261011.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(DEST.iterdir()):
        if p.is_file():
            z.write(p, arcname='DENT_STRIDE_figures/' + p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(archive)
print(f'{len(index)} figure panels; {archive.stat().st_size / 1024**2:.1f} MiB')
