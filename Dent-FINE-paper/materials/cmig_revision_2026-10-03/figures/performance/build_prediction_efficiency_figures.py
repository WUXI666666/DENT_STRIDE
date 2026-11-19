#!/usr/bin/env python3
"""CPU-only prediction diagnostics and framework AP/size figures.

Uses completed seed-0 native top-300 prediction exports; imports the original
frozen fixed_endpoints function without bytecode writes or model inference.
Scientific results and provenance are captured as JSON/CSV. No original paper,
annotation, source, or training file is modified.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pycocotools.coco import COCO

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')
spec=importlib.util.spec_from_file_location('figure_style',HERE/'build_controlled_figures.py')
style_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(style_module)
BLUE,ORANGE,GREY,BLACK=style_module.BLUE,style_module.ORANGE,style_module.GREY,style_module.BLACK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_function():
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('original_controlled_endpoints',ROOT/'evaluate_controlled.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.fixed_endpoints


def capture_predictions():
    fixed=fixed_function()
    cohorts=[
        ('Mobile','/home/wuxi/coco/valid/annotations_fdi_types.json',
         'mobile_1024_base','mobile_1024_full','evaluation'),
        ('Pano','/home/wuxi/DENTEX2023/annotations/instances_val2017.json',
         'pano_base','pano_factor_prototype','replay_prior_both'),
        ('DSLR','/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json',
         'dslr_base_adamw','dslr_cmfa_dualopt','evaluation'),
    ]
    data={'captured_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
          'seed':0,'curves':{},'sources':[],
          'endpoint_implementation':{'path':str(ROOT/'evaluate_controlled.py'),
                                     'sha256':sha(ROOT/'evaluate_controlled.py')}}
    for cohort,annotation,base,full,full_eval in cohorts:
        gt=COCO(annotation)
        data['sources'].append({'path':annotation,'sha256':sha(annotation)})
        for cfg,label,evaluation in [(base,'Base','evaluation'),(full,'Full',full_eval)]:
            directory=ROOT/'runs'/cfg/'seed_0'/evaluation
            pred_path=directory/'predictions_coco.json.gz';npz_path=directory/'coco_precision_recall.npz'
            metric_path=directory/'metrics.json'
            for path in [pred_path,npz_path,metric_path]:
                data['sources'].append({'path':str(path),'sha256':sha(path)})
            preds=json.load(gzip.open(pred_path,'rt'))
            results=json.loads(metric_path.read_text())
            arrays=np.load(npz_path)
            t=np.where(np.isclose(arrays['IoU_thresholds'],.50))[0]
            assert len(t)==1
            precision=arrays['precision'][t[0],:,:,0,2]  # recall,category; all area,maxDets100
            valid=precision>=0;class_count=valid.sum(axis=1)
            macro=np.divide(np.where(valid,precision,0).sum(axis=1),class_count,
                            out=np.full(precision.shape[0],np.nan),where=class_count>0)
            thresholds=np.round(np.arange(.05,1.0,.05),2)
            endpoints=[]
            for threshold in thresholds:
                metrics,_=fixed(preds,gt,score=float(threshold),iou=.50)
                endpoints.append(metrics)
            at30=endpoints[list(thresholds).index(.30)]
            for name in ['Precision','Recall','F1','matched_tooth_FDI_accuracy',
                         'localization_coverage','raw_duplicate_image_rate']:
                assert np.isclose(at30[name],results['fixed_threshold'][name]),(cohort,label,name)
            assert at30['counts']==results['fixed_threshold']['counts']
            data['curves'][f'{cohort}/{label}']={
                'configuration':cfg,'prediction_directory':str(directory),
                'PR_recall':arrays['recall_thresholds'].tolist(),
                'PR_precision_macro':macro.tolist(),'PR_supported_category_count':class_count.tolist(),
                'score_thresholds':thresholds.tolist(),'fixed_endpoints':endpoints,
                'checkpoint_sha256':results['checkpoint_sha256'],
                'GT_images':len(gt.imgs),'GT_teeth':len(gt.anns)}
            print(f'{cohort}/{label}: threshold0.30 equality verified; 19 thresholds computed',flush=True)
    (HERE/'prediction_curves_snapshot.json').write_text(json.dumps(data,indent=2))
    return data


def prediction_figures(data):
    fig,axs=plt.subplots(3,3,figsize=(11.8,9.4))
    threshold_rows=[];pr_rows=[]
    titles=['Detection PR','Score–F1','Duplicate images']
    for row,cohort in enumerate(['Mobile','Pano','DSLR']):
        for col in range(3):
            ax=axs[row,col];letter=chr(65+row*3+col)
            ax.set_title(f'({letter}) {titles[col]}',loc='left',pad=9)
            ax.set_xlim(0,100 if col==0 else 1);ax.set_ylim(0,100)
            if col==0:
                ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Recall (%)')
                ax.set_ylabel(f'{cohort}\nPrecision (%)')
            else:
                ax.set_xticks([0,.25,.50,.75,1]);ax.set_xlabel('Confidence threshold')
                ax.set_ylabel('F1 (%)' if col==1 else 'Duplicate-image rate (%)')
                ax.axvline(.30,lw=.75,color='#BCC4CA',ls=':',zorder=1)
            style_module.style(ax)
        for label,color,ls in [('Base',BLUE,'--'),('Full',ORANGE,'-')]:
            result=data['curves'][f'{cohort}/{label}']
            recall=np.asarray(result['PR_recall']);precision=np.asarray(result['PR_precision_macro'])
            axs[row,0].plot(recall*100,precision*100,color=color,ls=ls,lw=1.7,label=label)
            score=np.asarray(result['score_thresholds'])
            f1=np.array([r['F1'] for r in result['fixed_endpoints']])
            dup=np.array([r['raw_duplicate_image_rate'] for r in result['fixed_endpoints']])
            axs[row,1].plot(score,f1*100,color=color,ls=ls,lw=1.7)
            axs[row,2].plot(score,dup*100,color=color,ls=ls,lw=1.7)
            for rr,pp,kk in zip(recall,precision,result['PR_supported_category_count']):
                pr_rows.append({'cohort':cohort,'model':label,'configuration':result['configuration'],
                                'seed':0,'IoU':.50,'maxDets':100,'area':'all',
                                'recall':float(rr),'macro_precision':float(pp),
                                'supported_categories':kk})
            for threshold,endpoint in zip(score,result['fixed_endpoints']):
                item={'cohort':cohort,'model':label,'configuration':result['configuration'],
                      'seed':0,'score_threshold':float(threshold),'IoU':.50,
                      'GT_images':result['GT_images'],'GT_teeth':result['GT_teeth']}
                item.update({k:endpoint[k] for k in ['Precision','Recall','F1',
                             'matched_tooth_FDI_accuracy','localization_coverage',
                             'raw_duplicate_image_rate']})
                item.update(endpoint['counts']);threshold_rows.append(item)
    fig.legend(handles=[Line2D([],[],color=BLUE,ls='--',lw=1.7,label='Controlled base'),
                        Line2D([],[],color=ORANGE,ls='-',lw=1.7,label='Domain-configured model')],
               loc='upper center',bbox_to_anchor=(.5,.997),ncol=2,frameon=False)
    fig.subplots_adjust(left=.085,right=.99,bottom=.055,top=.91,wspace=.37,hspace=.60)
    style_module.save(fig,'controlled_prediction_curves')
    style_module.write_csv('controlled_prediction_threshold_values.csv',threshold_rows)
    style_module.write_csv('controlled_detection_PR_values.csv',pr_rows)


def accuracy_snapshot():
    # Original Table 4 is supplied directly by the manuscript, not a web summary.
    source=Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_main_visual.docx')
    # stdlib OOXML extraction avoids requiring python-docx in the training env.
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(source) as archive:
        tree=ET.fromstring(archive.read('word/document.xml'))
    table=tree.findall('.//w:tbl',ns)[3]
    raw=[]
    for row in table.findall('w:tr',ns)[1:]:
        cells=[''.join(t.text or '' for t in cell.findall('.//w:t',ns)) for cell in row.findall('w:tc',ns)]
        raw.append({'model':cells[0],'AP':float(cells[1]),'Params_M':float(cells[5]),'SD':None,'n':1,
                    'source':'Original main manuscript Table 4'})
    snap=json.loads((HERE/'source_snapshot.json').read_text())
    def stats(cfg):
        values=np.array([snap['metrics'][f'{cfg}/{s}']['metrics_percent']['AP'] for s in range(3)])
        params=snap['metrics'][f'{cfg}/0']['params_total']/1e6
        return float(values.mean()),float(values.std(ddof=1)),params
    full=stats('dslr_cmfa_dualopt');base=stats('dslr_base_adamw');cmfa=stats('dslr_cmfa_adamw')
    upstream=Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10/runs')
    upstream_stats={}
    upstream_sources=[]
    for model,configuration in [('Official DEIM-s','deim_s'),('RT-DETRv2-S','rtdetrv2_s')]:
        paths=[upstream/configuration/'dslr'/f'seed_{seed}/metrics_summary.json' for seed in range(3)]
        summaries=[json.loads(path.read_text()) for path in paths]
        assert all(summary['epochs_completed']==160 for summary in summaries)
        values=np.array([summary['best_metrics']['AP'] for summary in summaries])
        upstream_stats[model]=(float(values.mean()),float(values.std(ddof=1)))
        upstream_sources.extend({'path':str(path),'sha256':sha(path)} for path in paths)
    for row in raw:
        if row['model']=='DENT-STRIDE-DSLR':
            row.update(AP=full[0],SD=full[1],Params_M=full[2],n=3,
                       source='Controlled CMFA + DualOpt; seeds0/1/2 FP32 replay')
        if row['model']=='Local DEIM–D-FINE':
            row.update(model='Controlled base',AP=base[0],SD=base[1],Params_M=base[2],n=3,
                       source='Controlled AdamW base; seeds0/1/2 FP32 replay')
        if row['model'] in upstream_stats:
            value,sd=upstream_stats[row['model']]
            row.update(AP=value,SD=sd,n=3,
                       source='Official upstream implementation; completed seeds0/1/2 best-validation checkpoints')
    raw.append({'model':'CMFA + AdamW','AP':cmfa[0],'SD':cmfa[1],'Params_M':cmfa[2],
                'n':3,'source':'Controlled mechanism comparator; seeds0/1/2 FP32 replay'})
    assert len(raw)==11
    record={'captured_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'table_source':{'path':str(source),'sha256':sha(source),'table':4},
            'upstream_metric_sources':upstream_sources,'rows':raw}
    (HERE/'dslr_accuracy_size_snapshot.json').write_text(json.dumps(record,indent=2))
    return record


def mark_model(ax,row):
    name=row['model']
    if name=='DENT-STRIDE-DSLR':color,marker,face=ORANGE,'D',ORANGE
    elif name=='Controlled base':color,marker,face=BLUE,'s',BLUE
    elif name=='CMFA + AdamW':color,marker,face=BLUE,'D','white'
    else:color,marker,face=GREY,'o',GREY
    if row['SD'] is not None:
        ax.errorbar(row['Params_M'],row['AP'],yerr=row['SD'],color=color,
                    capsize=4,elinewidth=1.0,fmt='none',zorder=2)
    ax.scatter(row['Params_M'],row['AP'],s=45 if row['n']==3 else 35,
               edgecolors=color,facecolors=face,marker=marker,lw=1,zorder=4)


def label_model(ax,row,x,y,fontsize=11):
    ax.annotate(row['model'],xy=(row['Params_M'],row['AP']),xytext=(x,y),
                fontsize=fontsize,ha='left',va='center',color=BLACK,
                arrowprops={'arrowstyle':'-','lw':.6,'color':'#ABB4BB'},
                annotation_clip=False)


def accuracy_figure(data):
    rows=data['rows'];lookup={r['model']:r for r in rows}
    fig,axs=plt.subplots(1,2,figsize=(12.2,5.6),gridspec_kw={'width_ratios':[1.12,1]})
    ax=axs[0]
    for row in rows:mark_model(ax,row)
    # Dense 9–14M labels are resolved in the right panel. Every named model is
    # directly labelled in one of the two panels; no frontier is inferred.
    full_labels={'RT-DETR-L':(27.2,75.05),'RT-DETRv2-S':(20.8,70.65),
                 'CAF-YOLO-s':(12.8,66.55),'Faster R-CNN R50-FPN':(23.8,62.15)}
    for name,pos in full_labels.items():label_model(ax,lookup[name],*pos,fontsize=12)
    ax.set_xlim(7,44.5);ax.set_ylim(61.2,75.8)
    ax.set_xlabel('Parameters (M)');ax.set_ylabel('COCO AP (%)')
    ax.set_title('(A) Framework comparison',loc='left',pad=11)
    style_module.style(ax)
    ax=axs[1]
    compact=[row for row in rows if row['Params_M']<=14 and row['AP']>=70]
    for row in compact:mark_model(ax,row)
    labels={
        'Official DEIM-s':(9.05,73.48),'Controlled base':(9.05,73.02),
        'CMFA + AdamW':(11.80,73.63),'DENT-STRIDE-DSLR':(11.80,73.18),
        'YOLOv8-s':(11.80,72.18),'Dome-DETR-S':(12.13,70.97),
        'YOLO11-s':(9.05,70.90)}
    for row in compact:label_model(ax,row,*labels[row['model']],fontsize=12)
    ax.set_xlim(8.9,14.4);ax.set_ylim(70.65,73.90)
    ax.set_xlabel('Parameters (M)');ax.set_ylabel('COCO AP (%)')
    ax.set_title('(B) Compact-model detail',loc='left',pad=11)
    ax.set_xticks([9,10,11,12,13,14]);style_module.style(ax)
    fig.subplots_adjust(left=.07,right=.995,bottom=.12,top=.9,wspace=.24)
    style_module.save(fig,'dslr_framework_accuracy_size_updated')
    style_module.write_csv('dslr_framework_accuracy_size_values.csv',rows)


def append_captions():
    path=HERE/'CAPTIONS.txt';text=path.read_text()
    if '\ncontrolled_prediction_curves\n' not in text:
        text+='''

controlled_prediction_curves
Detection and confidence-threshold diagnostics for the seed-0 controlled base and domain-configured model. Rows correspond to Mobile, Pano, and DSLR. The first column shows category-macro-averaged COCO interpolated precision at IoU 0.50, all object areas, and maxDets = 100; categories without validation support are omitted from the macro average. The second and third columns show F1 and the fraction of validation images with repeated predicted FDI labels, respectively, over score thresholds 0.05–0.95 in steps of 0.05. Native top-300 outputs are used without added NMS. Fixed-threshold matches use class-agnostic one-to-one Hungarian localization matching at IoU ≥0.50, followed by exact FDI comparison; incorrect FDI predictions contribute both a false positive and a false negative. Pano uses the combined center and shape prior on the frozen Factor + Prototype checkpoint. Dotted vertical lines mark score 0.30. These single-seed validation diagnostics do not measure clinical charting-error rates.

dslr_framework_accuracy_size_updated
DSLR detection accuracy versus model size. (A) The complete retained framework comparator set. (B) A detailed view of the 9–14 M-parameter detection models with AP above 70%; this plotting window is not an inclusion criterion or an accuracy-leadership category. DENT-STRIDE-DSLR, the controlled base, and CMFA + AdamW show means and sample standard deviations across seeds 0, 1, and 2; CMFA + AdamW is denoted by an open diamond. External comparators retain their original single-seed manuscript values and have no invented error bars. Framework-native pretraining, optimization, training, and post-processing settings differ across external comparators. The plot does not establish a statistically confirmed rank or show a Pareto-frontier connection.
'''
        path.write_text(text)
    # Update this figure's paragraph only; preserve all other captions.
    import re
    final_caption=('DSLR detection accuracy versus model size. (A) The complete retained framework comparator set. '
      '(B) A detailed view of the 9–14 M-parameter detection models with AP above 70%; this plotting window is not '
      'an inclusion criterion or an accuracy-leadership category. DENT-STRIDE-DSLR, the controlled base, CMFA + AdamW, '
      'Official DEIM-s, and RT-DETRv2-S show their actual means and sample standard deviations across seeds 0, 1, and 2; '
      'CMFA + AdamW is denoted by an open diamond. The remaining comparators retain their single-seed manuscript '
      'values and have no invented error bars. Framework-native pretraining, optimization, training, and post-processing '
      'settings differ across external comparators. The plot does not establish a statistically confirmed rank or show '
      'a Pareto-frontier connection.')
    text=path.read_text()
    text=re.sub(r'(?m)^dslr_framework_accuracy_size_updated\n[^\n]*',
                'dslr_framework_accuracy_size_updated\n'+final_caption,text)
    path.write_text(text)


def main():
    pred_snapshot=HERE/'prediction_curves_snapshot.json'
    pred=json.loads(pred_snapshot.read_text()) if pred_snapshot.is_file() else capture_predictions()
    prediction_figures(pred)
    # Every upstream run is completed and stable; recapture this figure after
    # protocol changes instead of silently reusing its superseded n=1 snapshot.
    accuracy=accuracy_snapshot()
    accuracy_figure(accuracy);append_captions()
    print(json.dumps({'figures':['controlled_prediction_curves','dslr_framework_accuracy_size_updated'],
                      'font':style_module.FONT,'directory':str(HERE)}))


if __name__=='__main__':main()
