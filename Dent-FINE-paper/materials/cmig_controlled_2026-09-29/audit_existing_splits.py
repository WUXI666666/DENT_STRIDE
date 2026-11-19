"""Read-only audit of the actual selected-run COCO splits; no partition changes."""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from datetime import datetime, timezone
import ast
import csv
import io
import json
import math
import shutil
import struct

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent
REPO = Path('/home/wuxi/Dent-FINE')
BUNDLE = PAPER/'materials/best_model_bundle_2026-09-15/selected_models'
SELECTED = {'Mobile': 'phone_a07_proto_aux_AP55.701', 'Pano': 'dentex_spatial_shape_AP54.385', 'DSLR': 'dslr_cmfa_dualopt_AP73.115'}

def digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def image_record(item):
    cohort, split, directory, meta = item
    path = Path(directory)/meta['file_name']
    out = {'cohort': cohort, 'split': split, 'image_id': meta['id'], 'file_name': meta['file_name'], 'path': str(path), 'patient_id': meta.get('patient_id', meta.get('patient_key', '')), 'view': meta.get('view', ''), 'annotation_width': meta['width'], 'annotation_height': meta['height'], 'bytes_sha256': '', 'decoded_rgb_sha256': '', 'actual_width': '', 'actual_height': '', 'error': ''}
    try:
        out['bytes_sha256'] = digest(path)
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            out['actual_width'], out['actual_height'] = rgb.size
            h = sha256(struct.pack('>II', *rgb.size)); h.update(rgb.tobytes())
            out['decoded_rgb_sha256'] = h.hexdigest()
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    return out

def groups(records, key, cross):
    grouped = defaultdict(list)
    for r in records:
        if r[key]: grouped[r[key]].append(r)
    return [{'hash': h, 'members': [{k: r[k] for k in ('split','image_id','file_name','patient_id','view')} for r in rs]} for h, rs in grouped.items() if len(rs)>1 and (len({r['split'] for r in rs})>1 if cross else len({r['split'] for r in rs})==1)]

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Execute only the pure label-order functions, not the training package.
    source = REPO/'engine/deim/utils_arch.py'
    tree = ast.parse(source.read_text())
    names = {'fdi_permanent_label_order','fdi_mixed_label_order','enumeration32_universal_to_fdi_order'}
    scope = {'List': list}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    all_records, summaries, class_rows, box_issues, source_manifest = [], {}, [], [], []
    for cohort, selected in SELECTED.items():
        cfgpath = BUNDLE/selected/'config.yml'
        cfg = yaml.safe_load(cfgpath.read_text())
        frozen = ROOT/'reference_configs'/f'{cohort}_selected.yml'
        frozen.parent.mkdir(exist_ok=True)
        if frozen.exists(): assert digest(frozen)==digest(cfgpath)
        else: shutil.copy2(cfgpath, frozen)
        source_manifest.append({'kind':'selected_config','cohort':cohort,'path':str(cfgpath),'sha256':digest(cfgpath)})
        fdi = scope['enumeration32_universal_to_fdi_order' if cohort=='Pano' else 'fdi_mixed_label_order']()
        splits, records, categories = {}, [], []
        for split, loader in [('train','train_dataloader'),('validation','val_dataloader')]:
            dataset = cfg[loader]['dataset']; annpath = Path(dataset['ann_file'])
            d = json.loads(annpath.read_text())
            im = {x['id']:x for x in d['images']}; cats = {x['id']:x for x in d['categories']}
            categories.append(d['categories'])
            assert sorted(cats)==list(range(len(fdi)))
            for cid, cat in cats.items():
                if cohort != 'Pano': assert int(cat.get('fdi',cat['name'])) == fdi[cid]
                else: assert int(cat['name']) == cid+1
            counts = Counter(a['category_id'] for a in d['annotations'])
            for cid in sorted(cats):
                class_rows.append({'cohort':cohort,'split':split,'category_id':cid,'category_name':cats[cid]['name'],'clinical_fdi':fdi[cid],'instances':counts[cid]})
            anns = defaultdict(list)
            issue_counts = Counter()
            for a in d['annotations']:
                anns[a['image_id']].append(a)
                reasons=[]
                if a['image_id'] not in im: reasons.append('orphan_image_id')
                if a['category_id'] not in cats: reasons.append('unknown_category')
                b=a.get('bbox',[])
                if len(b)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in b): reasons.append('invalid_bbox')
                elif a['image_id'] in im:
                    x,y,w,h=b; image=im[a['image_id']]
                    if min(w,h)<=0: reasons.append('nonpositive_bbox')
                    if x<0 or y<0 or x+w>image['width']+1e-5 or y+h>image['height']+1e-5: reasons.append('bbox_outside_image')
                for reason in reasons:
                    issue_counts[reason]+=1
                    box_issues.append({'cohort':cohort,'split':split,'annotation_id':a['id'],'image_id':a['image_id'],'reason':reason,'bbox':json.dumps(b)})
            with ThreadPoolExecutor(max_workers=4) as pool:
                rs=list(pool.map(image_record, [(cohort,split,dataset['img_folder'],x) for x in d['images']]))
            for r in rs:
                normalized=sorted((a['category_id'], *[round(v,6) for v in a['bbox']], a.get('iscrowd',0)) for a in anns[r['image_id']] if len(a.get('bbox',[]))==4)
                r['box_annotation_sha256']=sha256(json.dumps(normalized).encode()).hexdigest()
            records.extend(rs)
            patients={r['patient_id'] for r in rs if r['patient_id'] not in ('',None)}
            per_patient=Counter(r['patient_id'] for r in rs if r['patient_id'] not in ('',None))
            by_patient=defaultdict(list)
            for r in rs:
                if r['patient_id']: by_patient[r['patient_id']].append(r['view'])
            splits[split]={'annotation_file':str(annpath),'annotation_sha256':digest(annpath),'images':len(d['images']),'annotations':len(d['annotations']),'classes':len(cats),'duplicate_image_ids':len(d['images'])-len(im),'duplicate_annotation_ids':len(d['annotations'])-len({a['id'] for a in d['annotations']}),'missing_or_unreadable_images':sum(bool(r['error']) for r in rs),'dimension_mismatches':sum((r['actual_width'],r['actual_height'])!=(r['annotation_width'],r['annotation_height']) for r in rs if not r['error']),'patients_with_identifiers':len(patients),'images_without_patient_id':sum(not r['patient_id'] for r in rs),'patient_image_count_distribution':dict(Counter(per_patient.values())),'patients_with_repeated_view_labels':sum(len(v)!=len(set(v)) for v in by_patient.values()),'views':dict(Counter(r['view'] for r in rs)),'annotation_issue_counts':dict(issue_counts),'zero_support_fdi':[fdi[k] for k in sorted(cats) if counts[k]==0]}
            source_manifest.append({'kind':'annotation','cohort':cohort,'split':split,'path':str(annpath),'sha256':digest(annpath)})
            print(cohort,split,len(rs),'images audited',flush=True)
        overlaps={}
        for key in ['bytes_sha256','decoded_rgb_sha256']:
            cross=groups(records,key,True)
            within={s:groups([r for r in records if r['split']==s],key,False) for s in splits}
            overlaps[key]={'cross_split_groups':cross,'within_split_groups':within}
        trainpatients={r['patient_id'] for r in records if r['split']=='train' and r['patient_id']}
        valpatients={r['patient_id'] for r in records if r['split']=='validation' and r['patient_id']}
        lookup={(r['split'],r['image_id']):r for r in records}
        for key,value in overlaps.items():
            for group in value['cross_split_groups'] + sum(value['within_split_groups'].values(),[]):
                hashes={lookup[(r['split'],r['image_id'])]['box_annotation_sha256'] for r in group['members']}
                group['identical_box_annotations_6_decimals']=len(hashes)==1
        summaries[cohort]={'selected_configuration':str(cfgpath),'label_order':cfg['fdi_label_order'],'clinical_fdi_order':fdi,'train_validation_category_definitions_equal':categories[0]==categories[1],'splits':splits,'content_duplicates':overlaps,'patient_id_overlap':sorted(trainpatients & valpatients),'independent_test_identified':False,'patient_independence_verifiable_from_metadata':bool(trainpatients and valpatients)}
        all_records.extend(records)
    # Cross-cohort identical files/pixels are also checked for transfer analyses.
    cross_domain={}
    for key in ['bytes_sha256','decoded_rgb_sha256']:
        g=defaultdict(list)
        for r in all_records:
            if r[key]:g[r[key]].append(r)
        cross_domain[key]=[{'hash':h,'members':[{k:r[k] for k in ['cohort','split','image_id','file_name']} for r in rs]} for h,rs in g.items() if len({r['cohort'] for r in rs})>1]
    for name in ['DENT_STRIDE_CMIG_main_visual.docx','DENT_STRIDE_CMIG_supplement_visual.docx']:
        p=PAPER/name;source_manifest.append({'kind':'manuscript','path':str(p),'sha256':digest(p)})
    report={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Existing selected-run datasets; read-only; no new splits or exclusions applied. No independent test set per user confirmation.','pixel_hash_definition':'PIL convert(RGB), no resizing or EXIF transposition; SHA256(width,height,decoded pixel bytes). Detects exact duplicates only, not all near duplicates or patient overlap.','label_map_source':{'path':str(source),'sha256':digest(source)},'cohorts':summaries,'cross_cohort_duplicates':cross_domain}
    write_json(ROOT/'audit/dataset_audit.json',report)
    write_json(ROOT/'audit/source_manifest.json',source_manifest)
    write_csv(ROOT/'audit/image_hash_manifest.csv',all_records)
    write_csv(ROOT/'audit/per_class_counts.csv',class_rows)
    write_csv(ROOT/'audit/annotation_issues.csv',box_issues)
    compact=[]
    for c,v in summaries.items():
        compact.append({'cohort':c,'train_images':v['splits']['train']['images'],'validation_images':v['splits']['validation']['images'],'byte_overlap_groups':len(v['content_duplicates']['bytes_sha256']['cross_split_groups']),'pixel_overlap_groups':len(v['content_duplicates']['decoded_rgb_sha256']['cross_split_groups']),'within_train_pixel_groups':len(v['content_duplicates']['decoded_rgb_sha256']['within_split_groups']['train']),'within_validation_pixel_groups':len(v['content_duplicates']['decoded_rgb_sha256']['within_split_groups']['validation']),'train_patients':v['splits']['train']['patients_with_identifiers'],'validation_patients':v['splits']['validation']['patients_with_identifiers'],'patient_id_overlap':len(v['patient_id_overlap'])})
    write_csv(ROOT/'audit/summary.csv',compact)
    print(json.dumps(compact,indent=2))

if __name__=='__main__': main()
