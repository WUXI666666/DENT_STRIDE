"""Assemble verified footprint/AP table without modifying the manuscripts."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parent
RECORDS=ROOT/'complexity_records'
PAPER=Path('/home/wuxi/Dent-FINE-paper')
COHORTS=['dentex','phone','dslr']
NAMES=[('Official DEIM-s','deim_s'),('RT-DETR-L','rtdetr_l'),
       ('RT-DETRv2-S','rtdetrv2_s'),('Faster R-CNN R50-FPN','faster_rcnn'),
       ('CAF-YOLO-s','caf_yolo_s'),('YOLOv8-s','yolov8s'),('YOLO11-s','yolo11s'),
       ('Dome-DETR-S','dome_detr_s'),('DENT-STRIDE','dent_stride')]
AP={'Official DEIM-s':['53.827 ± 0.425','53.553 ± 0.099','72.405 ± 0.202'],
    'RT-DETR-L':['51.978','54.751','74.019'],
    'RT-DETRv2-S':['50.775 ± 0.213','50.562 ± 0.486','70.949 ± 0.506'],
    'Faster R-CNN R50-FPN':['47.695','41.182','63.151'],
    'CAF-YOLO-s':['48.058','43.771','67.460'],
    'YOLOv8-s':['51.431','46.937','71.854'],
    'YOLO11-s':['50.644','47.064','71.094'],
    'Dome-DETR-S':['51.711','51.828','71.363'],
    'DENT-STRIDE':['54.205 ± 0.144','55.459 ± 0.275','72.908 ± 0.429']}


def main():
    rows=[]
    hashes={}
    for name,family in NAMES:
        if family=='dent_stride':
            data=[]
            for p in ['pano','mobile','dslr']:
                data.extend(json.loads((RECORDS/f'dent_stride_{p}.json').read_text()))
        else:
            data=json.loads((RECORDS/(family+'.json')).read_text())
        data=sorted(data,key=lambda r:COHORTS.index(r['cohort']))
        assert [r['cohort']for r in data]==COHORTS
        assert all(r['status']=='measured'for r in data)
        row={'Method':name,'Seeds':3 if name in ['Official DEIM-s','RT-DETRv2-S','DENT-STRIDE']else 1,
             'Pano AP':AP[name][0],'Mobile AP':AP[name][1],'DSLR AP':AP[name][2]}
        for c,r in zip(['Pano','Mobile','DSLR'],data):
            row[c+' Params (M)']=r['Params_M']
            row[c+' GFLOPs']=r['GFLOPs']
            row[c+' GMACs']=r['GMACs']
            assert abs(r['Params_M']-r['params_native_total']/1e6)<1e-12
            assert r['input_values'].startswith('explicit all-zero')
            assert r['input'][0]==1 and r['precision']=='FP32'
            for key in ['checkpoint','source_config']:
                if key in r:
                    p=Path(r[key]);digest=hashlib.sha256(p.read_bytes()).hexdigest()
                    assert digest==r['checkpoint_sha256'if key=='checkpoint'else 'config_sha256']
                    hashes[str(p)]=digest
        rows.append(row)
    data=json.loads((PAPER/'materials/cmig_selected_seed_revision_2026-10-12/quantitative/quantitative_data.json').read_text())
    for i,c in enumerate(['Pano','Mobile','DSLR']):
        s=next(r for r in data['summaries']['selected']if r['cohort']==c)
        values=[r['AP']for r in data['per_seed']if r['cohort']==c and r['seed']in s['seeds']]
        assert f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}'==AP['DENT-STRIDE'][i]
    columns=['Method','Seeds','Pano Params (M)','Mobile Params (M)','DSLR Params (M)',
             'Pano GFLOPs','Mobile GFLOPs','DSLR GFLOPs','Pano GMACs','Mobile GMACs','DSLR GMACs',
             'Pano AP','Mobile AP','DSLR AP']
    with (ROOT/'comparison_AP_parameters_GFLOPs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    manifest={'rows':rows,'cohort_order':'Pano / Mobile / DSLR',
       'parameters':'Native model parameters, before fusion/deployment; includes auxiliary heads.',
       'compute':'calflops 0.3.2, batch1, CPU FP32, native evaluation forward, explicit zero image; excludes data loading, external postprocessing and unsupported operations.',
       'resolution':'DENT-STRIDE-Mobile 1024x1024; all others640x640.',
       'Dome_query_counts':[300,300,480],
       'dynamic_compute_note':'Faster R-CNN proposals and Dome-DETR queries are input-dependent; these counts describe the stated profiling input, not a validation-cohort average.',
       'selection':'DENT-STRIDE Pano1/3/4 and Mobile/DSLR0/2/3 are post-hoc selected subsets; all-five results remain supplementary.',
       'sources_unchanged_SHA256':hashes}
    (ROOT/'comparison_AP_parameters_GFLOPs_sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    for r in rows:
        print(r['Method'],r['Seeds'],' / '.join(f"{r[c+' Params (M)']:.3f}"for c in ['Pano','Mobile','DSLR']),
              ' / '.join(f"{r[c+' GFLOPs']:.3f}"for c in ['Pano','Mobile','DSLR']),
              *AP[r['Method']],sep=' | ')
    print('27 domain/model profiles verified; checkpoint/config hashes unchanged. No manuscript edits.')


if __name__=='__main__':main()
