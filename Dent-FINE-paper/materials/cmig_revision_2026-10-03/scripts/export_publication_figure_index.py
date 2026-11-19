"""Export exactly the figures embedded in the two revised manuscripts.

Names follow final manuscript numbering. SVG/PDF companions are included only
when the embedded PNG matches a genuine source export, never by tracing raster.
"""
from pathlib import Path
from docx import Document
import csv
import hashlib
import json
import re
import shutil
from PIL import Image

PAPER=Path('/home/wuxi/Dent-FINE-paper')
ROOT=PAPER/'materials/cmig_revision_2026-10-03'
DEST=ROOT/'publication_figures'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    candidates={sha(p):p for p in (ROOT/'figures').rglob('*.png') if not p.name.endswith('_preview.png')}
    rows=[]
    for kind,name in [('Main','DENT_STRIDE_CMIG_main_controlled_results_20261003.docx'),
                      ('Supplement','DENT_STRIDE_CMIG_supplement_controlled_results_20261003.docx')]:
        doc=Document(PAPER/name)
        captions=[p.text for p in doc.paragraphs if re.match(r'^Fig\. (?:S\d+|\d+[A-C]?)\.',p.text)]
        assert len(captions)==len(doc.inline_shapes),(name,len(captions),len(doc.inline_shapes))
        for shape,caption in zip(doc.inline_shapes,captions):
            label=re.match(r'^Fig\. ((?:S\d+|\d+[A-C]?))\.',caption).group(1)
            rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            blob=doc.part.related_parts[rid].blob
            digest=hashlib.sha256(blob).hexdigest()
            embedded_path=DEST/f'Figure_{label}.png'
            embedded_path.write_bytes(blob)
            source=candidates.get(digest)
            vectors=[]
            if source:
                for suffix in ['.svg','.pdf']:
                    companion=source.with_suffix(suffix)
                    if companion.is_file():
                        target=DEST/f'Figure_{label}{suffix}'
                        shutil.copyfile(companion,target)
                        vectors.append(str(target))
            with Image.open(embedded_path) as image:
                pixels=list(image.size)
            assert sha(embedded_path)==digest
            row={'manuscript':kind,'figure':'Figure '+label,'PNG':str(embedded_path),
                 'source_export':str(source) if source else 'Original manuscript raster',
                 'SVG_PDF':'; '.join(vectors),'pixels':' × '.join(map(str,pixels)),
                 'embedded_width_inches':shape.width/914400,
                 'effective_dpi_at_Word_size':pixels[0]/(shape.width/914400),
                 'PNG_sha256':digest,'caption':caption}
            rows.append(row)
    with (DEST/'Figure_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (DEST/'Figure_index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('FIGURES_EXPORTED',len(rows),'TO',DEST)


if __name__=='__main__':main()
