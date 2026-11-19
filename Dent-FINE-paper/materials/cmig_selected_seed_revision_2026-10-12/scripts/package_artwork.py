"""Bundle new publication artwork and editable source without model weights."""
from pathlib import Path
import hashlib
import json
import zipfile

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_selected_seed_revision_2026-10-12'
REPORT = json.loads((ROOT / 'document_validation.json').read_text())
OUT = PAPER / 'DENT_STRIDE_CMIG_updated_figures_PPT_20261013.zip'
items = []
for kind, rows in REPORT['figures'].items():
    for row in rows:
        label = row['caption'].replace('Fig. ', 'Figure_')
        stem = Path(row['svg']).with_suffix('')
        for ext in ['.svg', '.pdf']:
            p = stem.with_suffix(ext)
            assert p.exists(), p
            items.append((p, 'figures/' + kind + '/' + label + ext))
        items.append((stem.with_suffix('.png'), 'figures/' + kind + '/' + label + '.png'))
editable = ROOT / 'architecture/DENT_STRIDE_All_Elements_Editable_v6_selected_20261012.pptx'
items.append((editable, 'editable_source/' + editable.name))
for filename in ['selected_mean_sd.csv', 'all5_mean_sd.csv', 'fixed012_mean_sd.csv',
                 'results_per_seed.csv', 'prior_per_seed.csv', 'prior_all5_mean_sd.csv']:
    items.append((ROOT / 'quantitative' / filename, 'results/' + filename))
captions = '\n\n'.join(kind.upper() + '\n' + '\n\n'.join(
            (r['caption'] + '\n' + str(Path(r['png']).name)) for r in rows)
            for kind, rows in REPORT['figures'].items())
readme = '''Updated DENT-STRIDE publication artwork

The editable PPT inherits the original v6 source; only the overview on slide 3
is revised. Slides 1 and 2 remain unchanged. Figure 2 SVG is exported from the
actual revised PPT, with vector outlines and Times New Roman glyphs.

Main and supplementary new figures are supplied as SVG, PDF and 600-dpi PNG.
Clinical panels use actual images, native predictions and shared target regions.
Representative checkpoints: Pano seed 3, Mobile seed 2, DSLR seed 3, selected by
validation AP within the declared seed subsets. Internal Mobile tensors and
auxiliary matrices come from fresh native CPU forward passes of seed 2.

Selected subsets: Pano 1/3/4; Mobile and DSLR 0/2/3. Selection was post-hoc.
Complete five-seed and fixed-0/1/2 results are included; selected-three results
are descriptive and must not be presented as prespecified statistical evidence.
Unchanged temperature-0.20 reference-ablation figures remain in the manuscripts.
No model checkpoint or raw dataset collection is included.
'''
checks = '''投稿前需作者确认的资料（不嵌入正文）

1. 每位作者对应的单位编号；当前独立作者页尚未给出作者到单位1–4的明确映射。
2. Smartphone/DSLR来源伦理委员会、批准号或适用豁免依据，以及既有同意文件是否覆盖这些分析和图片发表。
3. 所有病例照片，特别是包含儿童部分面部的图，是否具备必要的图片发表授权。
4. 作者共同核准Funding、CRediT、利益冲突、数据条款及AI辅助声明。

统计说明：本轮三种子为查看五次结果后选择的子集。正文与补充材料已保留此说明、完整五种子和固定0/1/2结果，不支持统计性SOTA结论。
'''
manifest = {}
with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path, name in items:
        z.write(path, name)
        manifest[name] = {'bytes': path.stat().st_size,
                          'SHA256': hashlib.sha256(path.read_bytes()).hexdigest()}
    z.writestr('README.txt', readme)
    z.writestr('FIGURE_FILE_INDEX.txt', captions)
    z.writestr('投稿前需作者确认.txt', checks)
    z.writestr('artwork_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False))
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
(ROOT / 'artwork_package.json').write_text(json.dumps({'zip': str(OUT), 'bytes': OUT.stat().st_size,
    'files': len(items), 'SHA256': hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2))
print(str(OUT), 'MB:', round(OUT.stat().st_size / 1e6, 2))
