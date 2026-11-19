"""Read-only, independent numerical and reference review of the new manuscript.

Writes only this review's JSON under the revision directory. Research files,
builder scripts, datasets, training state, and manuscripts are not altered.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from statistics import mean, stdev
from math import sqrt, isclose
import hashlib
import json
import re
from docx import Document

PAPER=Path('/home/wuxi/Dent-FINE-paper')
ROOT=PAPER/'materials/cmig_revision_2026-10-03'
TRAIN=PAPER/'materials/cmig_controlled_2026-09-29'
MAIN=PAPER/'DENT_STRIDE_CMIG_main_controlled_results_20261003.docx'
SUPP=PAPER/'DENT_STRIDE_CMIG_supplement_controlled_results_20261003.docx'
E=json.loads((ROOT/'evidence/controlled_evidence.json').read_text())
S=json.loads((ROOT/'evidence/selected_controlled_comparison.json').read_text())['rows']
report={'reviewed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'manuscript':str(MAIN),'manuscript_sha256':hashlib.sha256(MAIN.read_bytes()).hexdigest(),
        'completed_snapshot':E['completed_runs'],'total_planned_runs':E['jobs_total'],
        'checks':[],'errors':[],'recommendations':[]}
tcrit=sqrt(2*.95**2/(1-.95**2))  # Exact inverse CDF for df=2 at 0.975.


def metrics(path):
    d=json.loads(Path(path).read_text())
    x=dict(d['metrics_percent'])
    x.update({k:100*v for k,v in d['fixed_threshold'].items() if k not in
              ['score_threshold','IoU_threshold','counts','denominators']})
    return d,x


def check(label,ok,detail=None):
    report['checks'].append({'check':label,'passed':bool(ok),'detail':detail})
    if not ok:report['errors'].append({'check':label,'detail':detail})


def derived(values):
    m=mean(values);sd=stdev(values);half=tcrit*sd/sqrt(3)
    return {'mean':m,'SD':sd,'CI95_low':m-half,'CI95_high':m+half}


def summary_check(name,record,values):
    d=derived(values)
    check(name,all(isclose(record[k],v,abs_tol=1e-8) for k,v in d.items()) and
          all(isclose(x,y,abs_tol=1e-9) for x,y in zip(record['values'],values)),d)


def pm(record):return f"{record['mean']:.3f} ± {record['SD']:.3f}"


def run_values():
    sources={};summaries={r['configuration']:r for r in E['primary_summary']}
    for cfg,s in summaries.items():
        runs=[metrics(TRAIN/'runs'/cfg/f'seed_{seed}/metrics_summary.json') for seed in range(3)]
        sources[cfg]=[r[1] for r in runs]
        check(cfg+': three completed seeds',s['seeds']==[0,1,2] and all(
             (TRAIN/'runs'/cfg/f'seed_{seed}/COMPLETED.json').is_file() for seed in range(3)))
        for name,r in s['metrics'].items():summary_check(cfg+': '+name,r,[v[name] for v in sources[cfg]])
    for contrast in E['paired_contrasts']:
        if contrast.get('proposed'):
            for name,r in contrast['metrics'].items():
                values=[a[name]-b[name] for a,b in zip(sources[contrast['proposed']],sources[contrast['reference']])]
                summary_check(contrast['comparison']+': '+name,r,values)
        else:
            names={'Mobile feature-identity AP interaction':('mobile_1024_full','mobile_1024_features','mobile_1024_identity','mobile_1024_base'),
                   'Pano factor-prototype AP interaction':('pano_factor_prototype','pano_factor','pano_prototype','pano_base'),
                   'DSLR CMFA-DualOpt AP interaction':('dslr_cmfa_dualopt','dslr_cmfa_adamw','dslr_base_dualopt','dslr_base_adamw')}
            a,b,c,d=names[contrast['comparison']]
            values=[sources[a][i]['AP']-sources[b][i]['AP']-sources[c][i]['AP']+sources[d][i]['AP'] for i in range(3)]
            summary_check(contrast['comparison'],contrast['metrics']['AP'],values)
    none=[metrics(TRAIN/'runs/pano_factor_prototype'/f'seed_{seed}/evaluation/metrics.json')[1] for seed in range(3)]
    for prior in E['prior_summary']:
        directory='evaluation' if prior['prior']=='none' else 'replay_prior_'+prior['prior']
        runs=[metrics(TRAIN/'runs/pano_factor_prototype'/f'seed_{seed}'/directory/'metrics.json')[1] for seed in range(3)]
        for name,r in prior['metrics'].items():summary_check('Prior '+prior['prior']+': '+name,r,[v[name] for v in runs])
        summary_check('Prior '+prior['prior']+': paired AP',prior['paired_AP_difference_vs_none'],[a['AP']-b['AP'] for a,b in zip(runs,none)])
    for chosen in S:
        cfg=chosen['configuration'];directory='replay_prior_both' if chosen['cohort']=='Pano' else 'evaluation'
        runs=[metrics(TRAIN/'runs'/cfg/f'seed_{seed}'/directory/'metrics.json')[1] for seed in range(3)]
        refs=sources[chosen['reference']]
        for name in ['AP','AP50','AP75','AR100','Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']:
            d=derived([r[name] for r in runs]);delta=derived([a[name]-b[name] for a,b in zip(runs,refs)])
            check(chosen['cohort']+': chosen '+name,all(isclose(chosen[name+'_'+k],d[k],abs_tol=1e-8) for k in ['mean','SD']))
            check(chosen['cohort']+': paired chosen '+name,all(isclose(chosen[name+'_delta_'+k],delta[k],abs_tol=1e-8) for k in ['mean','SD','CI95_low','CI95_high']))
    check('Snapshot completeness',E['completed_runs']==58 and E['primary_completed']==42 and E['internal_completed']==16 and len(E['pending_jobs'])==4)
    return sources,summaries


def document_check(sources,summaries):
    doc=Document(MAIN);tables=[[[c.text for c in row.cells] for row in table.rows] for table in doc.tables]
    check('11 main tables',len(tables)==11)
    order=['mobile_1024_base','mobile_1024_features','mobile_1024_identity','mobile_1024_full',
           'mobile_640_base','mobile_640_full','pano_base','pano_factor','pano_prototype','pano_factor_prototype',
           'dslr_base_adamw','dslr_cmfa_adamw','dslr_base_dualopt','dslr_cmfa_dualopt']
    for row,cfg in zip(tables[2][1:],order):
        expected=[pm(summaries[cfg]['metrics'][m]) for m in ['AP','AP50','AP75','AR100']]
        check('Table3 '+cfg,row[2:6]==expected and row[6]==f"{summaries[cfg]['params_M']:.3f}")
    for row,chosen in zip(tables[3][1:4],S):
        check('Table4 '+chosen['cohort'],row==[chosen['model'],f"{chosen['AP_mean']:.3f} ± {chosen['AP_SD']:.3f}",
             f"{chosen['AP_delta_mean']:+.3f} ± {chosen['AP_delta_SD']:.3f}",
             f"[{chosen['AP_delta_CI95_low']:.3f}, {chosen['AP_delta_CI95_high']:.3f}]"])
    upstream=Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10/runs')
    for table_row,name in [(tables[4][2],'deim_s'),(tables[4][3],'rtdetrv2_s')]:
        for column,cohort in zip([2,3,4],['dentex','phone','dslr']):
            runs=[json.loads((upstream/name/cohort/f'seed_{seed}/metrics_summary.json').read_text()) for seed in range(3)]
            values=[r['best_metrics']['AP'] for r in runs]
            expected=f'{mean(values):.3f} ± {stdev(values):.3f}'
            check('Table5 '+name+'/'+cohort,table_row[column]==expected and all(r['epochs_completed']==160 for r in runs))
    for row,prior in zip(tables[5][1:],E['prior_summary']):
        expected=[pm(prior['metrics'][m]) for m in ['AP','AP50','AP75','AR100']]
        check('Table6 '+prior['prior'],row[1:5]==expected and row[5]==f"{prior['paired_AP_difference_vs_none']['mean']:+.3f}" and
              row[6]==f"[{prior['paired_AP_difference_vs_none']['CI95_low']:.3f}, {prior['paired_AP_difference_vs_none']['CI95_high']:.3f}]")
    # Cross-check all numeric internal rows against the independent plotted CSV.
    import csv
    plotted=list(csv.DictReader((ROOT/'figures/performance/internal_ablation_values.csv').open()))
    available={(f"{float(x['AP']):.3f}",f"{float(x['delta_AP']):+.3f}") for x in plotted if x['AP']}
    for i,row in enumerate(tables[6][1:],1):
        check('Table7 row '+str(i),row[2:4]==['—','—'] or tuple(row[2:4]) in available,row)
    check('Table7 four incomplete training conditions',sum(row[2]=='—' for row in tables[6][1:])==4)
    for row,cfg in zip(tables[7][1:],['mobile_640_base','mobile_640_full','mobile_1024_base','mobile_1024_full']):
        check('Table8 '+cfg,row[2:]==[pm(summaries[cfg]['metrics'][m]) for m in ['AP','AP75','F1','raw_duplicate_image_rate']])
    fixedmetrics=['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']
    for i,(cohort,cfg) in enumerate([('Pano','pano_base'),('Mobile','mobile_1024_base'),('DSLR','dslr_base_adamw')]):
        base=tables[8][1+i*2];chosen=tables[8][2+i*2];selected=next(x for x in S if x['cohort']==cohort)
        check('Table9 '+cohort+' base',base[2:]==[pm(summaries[cfg]['metrics'][m]) for m in fixedmetrics])
        check('Table9 '+cohort+' combined',chosen[2:]==[f"{selected[m+'_mean']:.3f} ± {selected[m+'_SD']:.3f}" for m in fixedmetrics])
    check('Table10 no invented transfer data',all(row[2:]==['—']*4 for row in tables[9][1:]))
    check('Table11 no invented cost metrics',all(row[3:]==['—']*4 for row in tables[10][1:]))
    text='\n'.join(p.text for p in doc.paragraphs)
    if 'F1 and duplicates use score ≥0.30 and IoU ≥0.50 localization matching' in text:
        report['recommendations'].append({'severity':'minor','location':'Table 8 caption',
            'issue':'The wording can imply that duplicate-label rates depend on Hungarian IoU matching; the actual duplicate metric counts all thresholded native predictions.',
            'suggestion':'State: F1 uses score ≥0.30 and localization IoU ≥0.50; duplicate-image rates use all native predictions retained at score ≥0.30.'})
    if 'The controlled cross-device comparison evaluates' in text:
        report['recommendations'].append({'severity':'minor','location':'Section 4.13 / Tables 10–11',
            'issue':'The section introduces cross-device measurements in present tense despite the all-em-dash table.',
            'suggestion':'Describe the retained table slots as intended evaluation entries rather than completed measurements.'})
    captions=[p.text for p in doc.paragraphs if p.style.name=='Caption']
    tablelabels=set(re.findall(r'^Table (\d+)\.',text,re.M))
    figlabels=set(re.findall(r'^Fig\. (\d+[A-C]?)\.',text,re.M))
    for ident in set(re.findall(r'(?<!Supplementary )\bTable (\d+)\b',text)):
        check('Main table ref '+ident,ident in tablelabels)
    for ident in set(re.findall(r'(?<!Supplementary )\bFig\. (\d+[A-C]?)\b',text)):
        check('Main figure ref '+ident,ident in figlabels or ident=='12')
    check('All expected figure captions',{'1','2','3','4','5','6','7','8','9','10','11','12A','12B','12C','13','14','15'}<=figlabels)
    check('No internal run identifiers in manuscript',not any(x in text for x in ['A07','Main Figure','repository flag','current snapshot','pre-revision']))
    original=Document(PAPER/'DENT_STRIDE_CMIG_main_visual.docx')
    check('Editable math preserved',len(doc._element.xpath('.//m:oMath'))==len(original._element.xpath('.//m:oMath')))
    scatter=json.loads((ROOT/'figures/performance/dslr_accuracy_size_snapshot.json').read_text())
    for name,table_row in [('Official DEIM-s',tables[4][2]),('RT-DETRv2-S',tables[4][3])]:
        row=next(x for x in scatter['rows']if x['model']==name)
        check('Figure7 three-seed value agrees with Table5: '+name,
              row['n']==3 and f"{row['AP']:.3f} ± {row['SD']:.3f}"==table_row[4])
    check('Figure7 exactly five real SD-bearing model groups',sum(x['SD'] is not None for x in scatter['rows'])==5)
    fig7=doc.inline_shapes[6]
    relation=fig7._inline.graphic.graphicData.pic.blipFill.blip.embed
    blob=doc.part.related_parts[relation].blob
    check('Figure7 Word embeds latest exported artifact',hashlib.sha256(blob).hexdigest()==
          hashlib.sha256((ROOT/'figures/performance/dslr_framework_accuracy_size_updated.png').read_bytes()).hexdigest())
    check('AI declaration immediately before references',next(i for i,p in enumerate(doc.paragraphs)if p.text=='References')==
          next(i for i,p in enumerate(doc.paragraphs)if p.text.startswith('Declaration of generative AI'))+2)
    if SUPP.is_file():
        sdoc=Document(SUPP);stext='\n'.join(p.text for p in sdoc.paragraphs)
        for ident in set(re.findall(r'Supplementary (?:Tables? )S(\d+)',text)):
            check('Supplementary table reference S'+ident,'Table S'+ident+'.' in stext)
        for ident in set(re.findall(r'Supplementary Fig\. S(\d+)',text)):
            check('Supplementary figure reference S'+ident,'Fig. S'+ident+'.' in stext)
    else:report['recommendations'].append('Supplement DOCX not available at review time; its referenced sections/tables/figures still require final cross-document verification.')
    report['document_summary']={'tables':len(doc.tables),'inline_figures':len(doc.inline_shapes),'editable_math_elements':len(doc._element.xpath('.//m:oMath'))}


def main():
    sources,summaries=run_values();document_check(sources,summaries)
    report['recommendations'] += [
        {'severity':'submission prerequisite','location':'Title page and ethics statement','issue':'Accurate per-author affiliation superscripts, clinical image publication permission, and actual ethical approval/consent details remain author-supplied information.',
         'suggestion':'Keep the missing-information checklist outside the manuscript and obtain the real institutional information before submission.'},
    ]
    report['overall']='No substantive numerical errors identified' if not report['errors'] else 'Requires corrections'
    (ROOT/'independent_main_QA.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'checks':len(report['checks']),'failed':len(report['errors']),
                      'overall':report['overall'],'review_file':str(ROOT/'independent_main_QA.json')},ensure_ascii=False))


if __name__=='__main__':main()
