"""Independent final source/mean/SD verification of the authored chart records."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'quantitative_data.json').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
checks=[]
for key in ['per_seed','prior_per_seed']:
    for row in data[key]:
        p=Path(row['native_metric_path']);d=json.loads(p.read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['native_metric_sha256']
        for metric in ['AP','AP50','AP75','AR100']:
            assert row[metric]==d['metrics_percent'][metric]
        assert row['checkpoint_sha256']==d['checkpoint_sha256']
    checks.append({'check':key+'_raw_native_metrics_and_hashes','rows':len(data[key]),'pass':True})
for scope,rows in data['summaries'].items():
    for row in rows:
        seeds=[r for r in data['per_seed'] if r['cohort']==row['cohort'] and r['seed'] in row['seeds']]
        for metric in ['AP','AP50','AP75','AR100']:
            values=[r[metric] for r in seeds]
            assert math.isclose(statistics.mean(values),row[metric+'_mean'],abs_tol=1e-12)
            assert math.isclose(statistics.stdev(values),row[metric+'_SD'],abs_tol=1e-12)
    checks.append({'check':scope+'_Python_statistics_independent_mean_SD','cohorts':len(rows),'pass':True})
for cohort,row in data['representative_endpoints'].items():
    c=row['counts'];n=len(json.loads(Path(row['sources']['validation_annotation']).read_text())['images'])
    direct={'Precision':c['TP']/c['predictions'],'Recall':c['TP']/c['GT'],
            'F1':2*c['TP']/(c['predictions']+c['GT']),
            'matched_tooth_FDI_accuracy':c['correct_FDI_matches']/c['localized_matches'],
            'localization_coverage':c['localized_matches']/c['GT'],
            'raw_duplicate_image_rate':c['duplicate_image']/n}
    assert all(row[k]==v for k,v in direct.items())
    checks.append({'check':cohort+'_direct_endpoint_ratio_recalculation','pass':True})
for name,fig in manifest['figures'].items():
    svg=Path(fig['files']['svg']).read_text()
    assert 'Times New Roman' in svg
    assert 'Main Figure' not in svg and 'Required success' not in svg
    pdf_fonts=subprocess.check_output(['pdffonts',fig['files']['pdf']],text=True)
    assert 'TimesNewRomanPSMT' in pdf_fonts
    assert all(Path(p).stat().st_size>1000 for p in fig['files'].values())
    checks.append({'check':name+'_vector_font_and_export_integrity','pass':True})
report={'checks':checks,'figures':len(manifest['figures']),
        'no_new_training_or_GPU_use':True,
        'selected_results_are_exploratory':True,
        'visual_inspection':'All nine authored figures viewed; no observed clipped labels or missing marks. Complete FDI-grid labels use compact type for 32/52 categories.'}
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'figures':report['figures'],'checks':len(checks),'pass':all(c['pass'] for c in checks)}))
