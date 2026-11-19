from pathlib import Path
from PIL import Image
import json, hashlib, re, subprocess

root=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
out=root/'figures/qualitative'
stems=['Acquisition_cohort_montage','Mobile_same_image_comparison','Pano_same_image_comparison','DSLR_same_image_comparison','Mobile_inter_module_comparison','Pano_same_checkpoint_prior_comparison']
entries=[]
for stem in stems:
    im=Image.open(out/f'{stem}.png')
    dpi=im.info['dpi'];assert min(dpi)>599
    svg=(out/f'{stem}.svg').read_text()
    text=re.findall(r'<text[^>]*>(.*?)</text>',svg)
    assert not any(any(b in x for b in ['Main Figure','case 1','Evidence key','Fixed score']) for x in text)
    font_output=subprocess.check_output(['pdffonts',str(out/f'{stem}.pdf')],text=True)
    assert 'TimesNewRoman' in font_output and ' yes ' in font_output
    e={'stem':stem,'PNG_pixels':im.size,'PNG_dpi':dpi,'SVG_text_labels':text,'PDF_font_inspection':font_output,'assets':{}}
    for ext in ['png','svg','pdf']:
        p=out/f'{stem}.{ext}';assert p.stat().st_size>1000
        e['assets'][ext]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    entries.append(e)
provenance=json.loads((out/'source_case_selection.json').read_text())
captions=json.loads((out/'figure_captions.json').read_text())
audit=json.loads((root/'evidence/external_prediction_sources.json').read_text())['entries']
assert all(provenance['cohort_comparison'][c]['sources']['external']['method']=='RT-DETR-L' for c in ['Mobile','Pano','DSLR'])
for c in ['Mobile','Pano','DSLR']:
    stem=f'{c}_same_image_comparison'
    svg=(out/f'{stem}.svg').read_text()
    assert '>RT-DETR-L</text>' in svg and '>YOLOv8-s</text>' not in svg
    assert 'RT-DETR-L' in captions[stem] and 'YOLOv8-s' not in captions[stem]
    source=provenance['cohort_comparison'][c]['sources']['external']
    authority=next(a for a in audit if a['cohort']==c and a['model']=='RT-DETR-L')
    assert source['prediction_file']==authority['prediction_path']
    assert source['checkpoint_sha256']==authority['weight_sha256']
    assert source['prediction_sha256']==hashlib.sha256(Path(source['prediction_file']).read_bytes()).hexdigest()
assert [c['image_id'] for c in provenance['cohort_comparison']['Mobile']['cases']]==[111,56]
assert all(len(c['raw_query_verification'])>=2 for c in provenance['pano_prior']['cases'])
assert len({s['checkpoint_sha256'] for s in provenance['pano_prior']['sources'].values()})==1
result={'entries':entries,'actual_tensor_source_scope':'Saved COCO outputs only; no model, dataset, checkpoint or training mutation','rendering_script':str(root/'scripts/prepare_qualitative_figures.py'),'rendering_script_sha256':hashlib.sha256((root/'scripts/prepare_qualitative_figures.py').read_bytes()).hexdigest(),'selection_manifest':str(out/'source_case_selection.json'),'selection_manifest_sha256':hashlib.sha256((out/'source_case_selection.json').read_bytes()).hexdigest(),'captions':str(out/'figure_captions.json'),'captions_sha256':hashlib.sha256((out/'figure_captions.json').read_bytes()).hexdigest(),'external_consistency':'RT-DETR-L SVG label/caption/source/prediction SHA/checkpoint SHA agree for all three cohorts','visual_checks':'All six final preview PNGs opened with view_image; figure headers and focal FDI labels reviewed for clipping/overlap; photos show oral fields without an identifiable full face.'}
(out/'artifact_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('VERIFIED',len(entries),'600-dpi PNGs and SVG/PDF exports, embedded Times New Roman, actual checkpoint/annotation provenance')
