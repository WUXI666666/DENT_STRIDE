#!/usr/bin/env python3
"""CPU-only publication plots tied to 15 completed native detector evaluations.

Selected groups were chosen after results were seen. Their summaries are
exploratory; all-five and fixed-0/1/2 summaries are retained without selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile
import xml.etree.ElementTree as etree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
import yaml

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/wuxi/Dent-FINE-paper')
TRAIN = PAPER/'materials/cmig_selected_configs_multiseed_2026-10-11'
OLD = PAPER/'materials/cmig_revision_2026-10-11'
FONTS = PAPER/'materials/cmig_revision_2026-10-03/references/times_new_roman'
SPEC = [('Pano','pano_spatialshape_original',[1,3,4],3,'DENTEX2023'),
        ('Mobile','mobile_proto_k4_tau0p50',[0,2,3],2,'Mobile'),
        ('DSLR','dslr_cmfa_extra_decay_only',[0,2,3],3,'DSLR')]
METRICS = ['AP','AP50','AP75','AR100']
PANO32 = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
          38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]
BLUE='#365E7D'; GREY='#AAB2B9'; BLACK='#252A2E'; PURPLE='#756A8B'; LIGHT='#E5E7E9'
for p in FONTS.glob('*.TTF'): font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.titlesize':14,
                     'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
                     'axes.titleweight':'bold','axes.linewidth':.8,
                     'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white',
                     'figure.facecolor':'white','text.color':BLACK,'axes.labelcolor':BLACK,
                     'xtick.color':BLACK,'ytick.color':BLACK})
bindings={}; captions={}; figures={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def bind(path,role='source'):
    p=Path(path); digest=sha(p)
    bindings[str(p)]={'path':str(p),'sha256':digest,'role':role}
    return digest


def read(path):
    bind(path)
    return json.loads(Path(path).read_text())


def dump(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def style(ax,axis='y'):
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    if axis:ax.grid(axis=axis,color=LIGHT,lw=.6)


def save(fig,name,caption,sources):
    fig.canvas.draw()
    # Tight-bbox export preserves external row values and legends; padding is
    # not a substitute for checking overlapping text in the rendered preview.
    for ext in ['svg','pdf','png']:
        metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(HERE/(name+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.12,metadata=metadata)
    fig.savefig(HERE/(name+'_preview.png'),dpi=140,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    captions[name]=caption
    figures[name]={'caption':caption,'source_paths':list(dict.fromkeys(map(str,sources))),
                   'files':{ext:str(HERE/(name+'.'+ext)) for ext in ['svg','pdf','png']}}


def summarize(rows,seeds):
    chosen=sorted((r for r in rows if r['seed'] in seeds),key=lambda r:r['seed'])
    assert len(chosen)==len(seeds)
    result={'cohort':chosen[0]['cohort'],'configuration':chosen[0]['configuration'],
            'seeds':seeds,'n':len(seeds),'post_hoc_selected':seeds not in [list(range(5)),[0,1,2]],
            'params_total':chosen[0]['params_total'],'params_trainable':chosen[0]['params_trainable']}
    for k in METRICS:
        values=np.array([r[k] for r in chosen]); mean=float(values.mean()); sd=float(values.std(ddof=1))
        margin=float(t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values)))
        result.update({k+'_mean':mean,k+'_SD':sd,k+'_per_seed':values.tolist(),
                       k+'_nominal_CI95_low':mean-margin,k+'_nominal_CI95_high':mean+margin})
        # Independent scalar recomputation, not the same numpy helper.
        scalar_mean=sum(values)/len(values)
        scalar_sd=math.sqrt(sum((x-scalar_mean)**2 for x in values)/(len(values)-1))
        assert math.isclose(mean,scalar_mean,abs_tol=1e-12)
        assert math.isclose(sd,scalar_sd,abs_tol=1e-12)
    result['CI_note']='Nominal Student-t mean interval. Post-hoc subgroup intervals are not selection-adjusted and are not used for inference.'
    return result


def load_runs():
    rows=[];prior_rows=[]
    for cohort,configuration,seeds,representative,label in SPEC:
        for seed in range(5):
            run=TRAIN/'runs'/configuration/f'seed_{seed}'
            path=run/'native_coco_metrics.json';native=read(path); summary=read(run/'metrics_summary.json')
            export=read(run/'evaluation/metrics.json')
            assert native['verified_best_checkpoint'] and native['epochs_completed']==160
            assert all(math.isclose(export['metrics_percent'][k],native['metrics_percent'][k],abs_tol=1e-10) for k in METRICS)
            assert export['checkpoint_sha256']==native['checkpoint_sha256']
            checkpoint=run/'best.pth';assert bind(checkpoint,'best checkpoint')==native['checkpoint_sha256']
            row={'cohort':cohort,'cohort_label':label,'configuration':configuration,'seed':seed,
                 'selected':seed in seeds,'representative':seed==representative,
                 'native_metric_path':str(path),'native_metric_sha256':bindings[str(path)]['sha256'],
                 'checkpoint_path':str(checkpoint),'checkpoint_sha256':native['checkpoint_sha256'],
                 'params_total':native['params_total'],'params_trainable':native['params_trainable'],
                 'best_epoch_one_based':native['best_epoch_one_based'],'epochs_completed':160,
                 'fixed_threshold':export['fixed_threshold'],
                 **{k:native['metrics_percent'][k] for k in METRICS}}
            rows.append(row)
            if cohort=='Pano':
                for mode in ['none','center','shape','both']:
                    p=run/f'native_prior_{mode}_metrics.json';d=read(p)
                    assert d['checkpoint_sha256']==native['checkpoint_sha256']
                    if mode=='both':
                        assert all(math.isclose(d['metrics_percent'][k],row[k],abs_tol=1e-10) for k in METRICS)
                    prior_row={k:v for k,v in row.items() if k!='fixed_threshold'}
                    prior_rows.append({**prior_row,'prior':mode,'native_metric_path':str(p),
                                       'native_metric_sha256':bindings[str(p)]['sha256'],
                                       **{k:d['metrics_percent'][k] for k in METRICS}})
    summaries={group:[] for group in ['selected','fixed012','all5']}
    for cohort,_,seeds,_,_ in SPEC:
        subset=[r for r in rows if r['cohort']==cohort]
        for group,ss in [('selected',seeds),('fixed012',[0,1,2]),('all5',list(range(5)))]:
            summaries[group].append(summarize(subset,ss))
    prior_summary={group:[] for group in summaries}
    for group,seeds in [('selected',[1,3,4]),('fixed012',[0,1,2]),('all5',list(range(5)))]:
        none={r['seed']:r for r in prior_rows if r['prior']=='none'}
        for mode in ['none','center','shape','both']:
            subset=[r for r in prior_rows if r['prior']==mode]
            s=summarize(subset,seeds);s['prior']=mode
            for k in METRICS:
                delta=np.array([r[k]-none[r['seed']][k] for r in sorted(subset,key=lambda r:r['seed']) if r['seed'] in seeds])
                m=float(delta.mean());sd=float(delta.std(ddof=1));margin=float(t.ppf(.975,len(seeds)-1)*sd/math.sqrt(len(seeds)))
                s.update({k+'_paired_delta_vs_none':delta.tolist(),k+'_delta_mean':m,k+'_delta_SD':sd,
                          k+'_delta_nominal_CI95_low':m-margin,k+'_delta_nominal_CI95_high':m+margin})
            prior_summary[group].append(s)
    return rows,prior_rows,summaries,prior_summary


def seed_figure(rows,summaries):
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.2))
    for panel,(cohort,config,seeds,_,label),ax in zip('ABC',SPEC,axs):
        rs=sorted([r for r in rows if r['cohort']==cohort],key=lambda r:r['seed'])
        for row in rs:
            ax.scatter(row['seed'],row['AP'],s=48,c=BLUE if row['selected'] else 'white',
                       edgecolor=BLUE if row['selected'] else GREY,lw=1.1,zorder=3)
        select=next(r for r in summaries['selected'] if r['cohort']==cohort)
        all5=next(r for r in summaries['all5'] if r['cohort']==cohort)
        for x,r,color,marker in [(5.5,select,BLUE,'D'),(6.6,all5,PURPLE,'s')]:
            ax.errorbar(x,r['AP_mean'],yerr=r['AP_SD'],fmt=marker,color=color,ms=5.7,capsize=4,lw=1.3,zorder=4)
        ax.set_xticks([0,1,2,3,4,5.5,6.6],['0','1','2','3','4','Selected','All five'])
        ax.tick_params(axis='x',labelsize=10);ax.set_xlim(-.45,7.13);ax.set_xlabel('Seed / mean ± SD')
        vals=[r['AP'] for r in rs]+[select['AP_mean']-select['AP_SD'],select['AP_mean']+select['AP_SD'],all5['AP_mean']-all5['AP_SD'],all5['AP_mean']+all5['AP_SD']]
        lo,hi=min(vals),max(vals);pad=max((hi-lo)*.19,.15);ax.set_ylim(lo-pad,hi+pad*1.4)
        ax.set_ylabel('COCO AP (%)');ax.set_title(f'({panel}) {label}',loc='left',pad=12);style(ax)
    handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='Selected seed'),
             Line2D([],[],marker='o',ls='',color=GREY,mfc='white',label='Other seed'),
             Line2D([],[],marker='D',ls='',color=BLUE,label='Selected three: mean ± SD'),
             Line2D([],[],marker='s',ls='',color=PURPLE,label='All five: mean ± SD')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.005),ncol=4,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.05,right=.985,bottom=.16,top=.79,wspace=.3)
    save(fig,'Figure_6_training_seed_variation',
         'Figure 6. Training-seed variation of the domain-configured detectors. All five completed runs are shown in each cohort. Filled circles identify the post-hoc selected seeds: 1/3/4 for DENTEX2023 and 0/2/3 for Mobile and DSLR; open circles show the remaining runs. Diamonds and squares show the selected-three and all-five means, respectively, with sample-standard-deviation bars. (A) DENTEX2023. (B) Mobile. (C) DSLR. Independent AP-axis ranges display within-cohort variation. The selected-three summaries are exploratory and do not represent prespecified repeated-seed evidence. All-five and fixed-0/1/2 results are reported in the supplementary tables.',
         [r['native_metric_path'] for r in rows])


def mobile_framework(rows,summaries):
    prior=read(OLD/'quantitative_figures/Figure_7_Mobile_values.json')
    external=[r for r in prior if r['model']!='DENT-STRIDE-Mobile']
    for implementation,model in [('deim_s','Official DEIM-s'),('rtdetrv2_s','RT-DETRv2-S')]:
        native=[]
        for seed in range(3):
            p=PAPER/'materials/official_baselines_2026-09-10/runs'/implementation/'phone'/f'seed_{seed}'/'metrics_summary.json'
            native.append(read(p)['best_metrics']['AP'])
        r=next(r for r in external if r['model']==model)
        assert math.isclose(r['AP'],np.mean(native),abs_tol=1e-10)
        assert math.isclose(r['SD'],np.std(native,ddof=1),abs_tol=1e-10)
    manuscript=PAPER/'DENT_STRIDE_CMIG_main_final_20261006.docx';bind(manuscript,'unchanged single-seed conventional comparator table')
    with zipfile.ZipFile(manuscript) as archive:document=etree.fromstring(archive.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    table=document.findall('./w:body/w:tbl',ns)[4]
    conventional={}
    for tr in table.findall('./w:tr',ns)[1:]:
        cells=[' '.join(t.text or '' for t in c.findall('.//w:t',ns)) for c in tr.findall('./w:tc',ns)]
        conventional[cells[0]]=cells[3]
    for r in external:
        if r['n']==1:assert math.isclose(float(conventional[r['model']]),r['AP'],abs_tol=1e-10)
    selected=next(r for r in summaries['selected'] if r['cohort']=='Mobile')
    all5=next(r for r in summaries['all5'] if r['cohort']=='Mobile')
    ours={'model':'DENT-STRIDE-Mobile','AP':selected['AP_mean'],'SD':selected['AP_SD'],
          'n':3,'seeds':selected['seeds'],'post_hoc_selected':True,
          'all5_AP':all5['AP_mean'],'all5_SD':all5['AP_SD']}
    plotted=sorted([*external,ours],key=lambda r:r['AP'],reverse=True)
    fig,ax=plt.subplots(figsize=(10.7,5.7))
    for y,r in enumerate(plotted):
        focal=r['model']=='DENT-STRIDE-Mobile';color=BLUE if focal else BLACK
        if r['SD'] is not None:
            ax.errorbar(r['AP'],y-.075 if focal else y,xerr=r['SD'],fmt='D' if focal else 'o',
                        color=color,ms=6,capsize=4,lw=1.2,zorder=3)
        else:ax.scatter(r['AP'],y,s=42,facecolor='white',edgecolor=BLACK,lw=1,zorder=3)
        if focal:
            ax.errorbar(r['all5_AP'],y+.12,xerr=r['all5_SD'],fmt='s',color=PURPLE,ms=5,capsize=3,lw=1.1,zorder=3)
        value=f"{r['AP']:.3f}"+(f" ± {r['SD']:.3f}" if r['SD'] is not None else '')
        ax.text(1.015,y,value,transform=ax.get_yaxis_transform(),va='center',fontsize=12,color=color)
    ax.set_yticks(range(len(plotted)),[r['model'] for r in plotted]);ax.set_ylim(len(plotted)-.6,-.75)
    ax.set_xlim(40,56.3);ax.set_xlabel('COCO AP (%)');ax.set_xticks([40,44,48,52,56])
    style(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Line2D([],[],marker='D',ls='',color=BLUE,label='Selected three'),
                       Line2D([],[],marker='s',ls='',color=PURPLE,label='All five')],
              loc='lower right',frameon=False,fontsize=11)
    fig.subplots_adjust(left=.245,right=.815,bottom=.12,top=.97)
    save(fig,'Figure_7_Mobile_framework_AP',
         'Figure 7. Framework-level detection AP on Mobile images. The blue diamond shows DENT-STRIDE-Mobile with K=4 and temperature 0.50 for the post-hoc selected seeds 0/2/3 (55.459 ± 0.275 AP); the purple square shows all five completed seeds (55.292 ± 0.301 AP). Error bars indicate sample standard deviation. The two repeated-seed official controls retain their original seeds 0/1/2; open circles identify single-seed external results without dispersion estimates. All external systems, including RT-DETR-L, are retained. These recipe-level validation comparisons use framework-specific resolution, pretraining, optimization, and post-processing and do not establish statistical superiority or an isolated prototype-temperature effect.',
         [OLD/'quantitative_figures/Figure_7_Mobile_values.json',*[r['native_metric_path'] for r in rows if r['cohort']=='Mobile']])
    dump('Figure_7_values.json',plotted)


def prior_figure(priors,summary):
    modes=['none','center','shape','both'];labels=['None','Center','Shape','Center + shape']
    fig,axs=plt.subplots(1,2,figsize=(11.7,4.8))
    for ax,k,panel in zip(axs,['AP','AR100'],'AB'):
        displayed=[]
        for i,mode in enumerate(modes):
            rs=sorted([r for r in priors if r['prior']==mode],key=lambda r:r['seed'])
            for j,r in enumerate(rs):
                ax.scatter(i+(-.20+j*.08),r[k],s=28,facecolor=BLUE if r['seed'] in [1,3,4] else 'white',edgecolor=BLUE if r['seed'] in [1,3,4] else GREY,lw=.8,zorder=3)
                displayed.append(r[k])
            for offset,group,color,marker in [(.23,'selected',BLUE,'D'),(.39,'all5',PURPLE,'s')]:
                row=next(r for r in summary[group] if r['prior']==mode)
                ax.errorbar(i+offset,row[k+'_mean'],yerr=row[k+'_SD'],fmt=marker,color=color,ms=5,capsize=3,lw=1.2,zorder=4)
                displayed.extend([row[k+'_mean']-row[k+'_SD'],row[k+'_mean']+row[k+'_SD']])
        lo,hi=min(displayed),max(displayed);pad=max((hi-lo)*.15,.1)
        ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(-.46,3.63);ax.set_xticks(np.arange(4)+.08,labels)
        ax.set_ylabel(k+' (%)');ax.set_title(f'({panel}) {k}',loc='left',pad=12);style(ax)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='Selected seed'),Line2D([],[],marker='o',ls='',color=GREY,mfc='white',label='Other seed'),Line2D([],[],marker='D',ls='',color=BLUE,label='Selected three: mean ± SD'),Line2D([],[],marker='s',ls='',color=PURPLE,label='All five: mean ± SD')],loc='upper center',ncol=2,bbox_to_anchor=(.53,1.02),frameon=False,fontsize=11)
    fig.subplots_adjust(left=.07,right=.985,bottom=.14,top=.74,wspace=.24)
    select=summary['selected'];all5=summary['all5']
    center=next(r for r in select if r['prior']=='center');both=next(r for r in select if r['prior']=='both')
    c5=next(r for r in all5 if r['prior']=='center');b5=next(r for r in all5 if r['prior']=='both')
    save(fig,'Figure_9_Pano_prior_comparison',
         f'Figure 9. Same-checkpoint anatomical scoring on DENTEX2023. All five checkpoints are evaluated with no prior, center-only, shape-only, and combined scoring from candidates before top-k selection. Circles show actual runs; diamonds show the post-hoc selected seeds 1/3/4 and squares show all-five summaries, with sample-standard-deviation bars. (A) AP. (B) AR100. Center-only and joint scoring change selected-three mean AP by {center["AP_delta_mean"]:+.3f} and {both["AP_delta_mean"]:+.3f} points; the corresponding all-five changes are {c5["AP_delta_mean"]:+.3f} and {b5["AP_delta_mean"]:+.3f} points. Weights are fixed within each seed, so the comparison concerns inference scoring rather than training effects.',
         [r['native_metric_path'] for r in priors])


def prior_paired(priors,summary):
    fig,ax=plt.subplots(figsize=(9.3,4.1))
    none={r['seed']:r['AP'] for r in priors if r['prior']=='none'}
    vals=[]
    for y,mode in enumerate(['center','shape','both']):
        rs=sorted([r for r in priors if r['prior']==mode],key=lambda r:r['seed'])
        for j,r in enumerate(rs):
            delta=r['AP']-none[r['seed']];vals.append(delta)
            ax.scatter(delta,y-.18+j*.07,s=29,facecolor=BLUE if r['seed'] in [1,3,4] else 'white',edgecolor=BLUE if r['seed'] in [1,3,4] else GREY,lw=.8,zorder=3)
        for offset,group,color,marker in [(.22,'selected',BLUE,'D'),(.37,'all5',PURPLE,'s')]:
            r=next(r for r in summary[group] if r['prior']==mode)
            ax.errorbar(r['AP_delta_mean'],y+offset,xerr=r['AP_delta_SD'],fmt=marker,color=color,ms=5.5,capsize=4,lw=1.3,zorder=4)
            vals.extend([r['AP_delta_mean']-r['AP_delta_SD'],r['AP_delta_mean']+r['AP_delta_SD']])
    ax.axvline(0,color=BLACK,lw=.9);ax.set_yticks([0,1,2],['Center','Shape','Center + shape']);ax.set_ylim(2.75,-.5)
    span=max(vals)-min(vals);ax.set_xlim(min(min(vals)-span*.12,-.05),max(vals)+span*.2)
    ax.set_xlabel('Paired AP difference from no prior (points)');style(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=BLUE,label='Selected seed'),Line2D([],[],marker='o',ls='',color=GREY,mfc='white',label='Other seed'),Line2D([],[],marker='D',ls='',color=BLUE,label='Selected three: mean ± SD'),Line2D([],[],marker='s',ls='',color=PURPLE,label='All five: mean ± SD')],loc='upper center',bbox_to_anchor=(.59,1.3),ncol=2,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.23,right=.98,bottom=.2,top=.72)
    save(fig,'Figure_S3_prior_paired_AP',
         'Figure S3. Paired changes in DENTEX2023 AP relative to no prior. Each circle represents one of the five same-checkpoint comparisons; filled circles identify post-hoc selected seeds 1/3/4 and open circles the other seeds. Diamonds and squares show selected-three and all-five mean changes, respectively, with sample-standard-deviation bars, not confidence intervals. All center, shape, and joint settings are retained; shape-only effects are small relative to center-only and joint effects. No selection-adjusted significance claim is made.',
         [r['native_metric_path'] for r in priors])


def learning_curves(rows):
    fig,axs=plt.subplots(1,3,figsize=(13.0,4.5));out=[]
    for panel,(cohort,config,seeds,_,label),ax in zip('ABC',SPEC,axs):
        curves=[];chosen=[]
        for seed in range(5):
            p=TRAIN/'runs'/config/f'seed_{seed}'/'log.txt';bind(p,'complete 160-epoch training log')
            log=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            assert [r['epoch'] for r in log]==list(range(160))
            vals=np.array([r['test_coco_eval_bbox'][0]*100 for r in log]);row=next(r for r in rows if r['cohort']==cohort and r['seed']==seed)
            assert math.isclose(vals.max(),row['AP'],abs_tol=2e-4)
            ax.plot(np.arange(1,161),vals,color=GREY,lw=.65,alpha=.65,zorder=1)
            curves.append(vals)
            if seed in seeds:chosen.append(vals)
            out.append({'cohort':cohort,'configuration':config,'seed':seed,'selected':seed in seeds,'epochs_one_based':list(range(1,161)),'AP':vals.tolist(),'source':str(p)})
        ss=np.stack(chosen);all5=np.stack(curves);mean=ss.mean(0);sd=ss.std(0,ddof=1)
        ax.fill_between(np.arange(1,161),mean-sd,mean+sd,color=BLUE,alpha=.15,lw=0,zorder=2)
        ax.plot(np.arange(1,161),mean,color=BLUE,lw=1.35,zorder=3)
        ax.plot(np.arange(1,161),all5.mean(0),color=PURPLE,lw=1.25,ls='--',zorder=4)
        ax.set_title(f'({panel}) {label}',loc='left',pad=12);ax.set_xlabel('Epoch');ax.set_ylabel('Validation AP (%)')
        ax.set_xlim(1,160);ax.set_xticks([1,40,80,120,160]);ax.set_ylim(0,60 if cohort!='DSLR' else 80);style(ax)
    fig.legend(handles=[Line2D([],[],color=GREY,lw=.8,label='All five actual runs'),Line2D([],[],color=BLUE,lw=1.5,label='Selected three: mean ± SD'),Line2D([],[],color=PURPLE,lw=1.5,ls='--',label='All five: mean')],loc='upper center',bbox_to_anchor=(.53,1.0),ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.06,right=.985,bottom=.14,top=.79,wspace=.3)
    save(fig,'Figure_S8_learning_curves',
         'Figure S8. Validation-AP trajectories over all 160 epochs. Thin grey lines show all five actual runs. Blue lines and bands show the epoch-wise mean and sample standard deviation of the post-hoc selected seeds (DENTEX2023: 1/3/4; Mobile and DSLR: 0/2/3); dashed purple lines show all-five means. No smoothing or synthetic intervals are applied. Epoch-wise means are not the mean of individually selected best checkpoints, and seed selection used the observed best-validation results.',
         [r['source'] for r in out]);dump('learning_curve_data.json',out)


def bbox_iou(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    lower=np.maximum(a[:,None,:2],b[None,:,:2]);upper=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    intersection=np.maximum(upper-lower,0).prod(-1)
    union=a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-intersection
    return intersection/np.maximum(union,1e-12)


def fixed(predictions,val,score=.30,return_matrix=False,cat_to_fdi=None):
    pp=defaultdict(list);gg=defaultdict(list)
    imageids=sorted(i['id'] for i in val['images'])
    for p in predictions:
        if p['score']>=score:pp[p['image_id']].append(p)
    for g in val['annotations']:gg[g['image_id']].append(g)
    counts=Counter();perimage=[]
    codes=sorted(cat_to_fdi.values()) if return_matrix else []
    index={code:i for i,code in enumerate(codes)}
    matrix=np.zeros((len(codes),len(codes)+1),dtype=int) if return_matrix else None
    extra=Counter()
    for iid in imageids:
        preds,truth=pp[iid],gg[iid];matches={};matched_pred=set()
        if preds and truth:
            overlaps=bbox_iou([p['bbox'] for p in preds],[g['bbox'] for g in truth])
            rp,rg=linear_sum_assignment(-overlaps)
            for p,g in zip(rp,rg):
                if overlaps[p,g]>=.50:matches[g]=p;matched_pred.add(p)
        correct=sum(preds[p]['category_id']==truth[g]['category_id'] for g,p in matches.items())
        dup=Counter(p['category_id'] for p in preds)
        row={'image_id':iid,'GT':len(truth),'predictions':len(preds),'localized_matches':len(matches),
             'correct_FDI_matches':correct,'TP':correct,'FP':len(preds)-correct,'FN':len(truth)-correct,
             'duplicate_image':int(any(n>1 for n in dup.values())),
             'duplicate_count':sum(max(n-1,0) for n in dup.values())}
        counts.update({k:v for k,v in row.items() if k!='image_id'});perimage.append(row)
        if return_matrix:
            for g,truthitem in enumerate(truth):
                ref=index[cat_to_fdi[truthitem['category_id']]]
                col=index[cat_to_fdi[preds[matches[g]]['category_id']]] if g in matches else len(codes)
                matrix[ref,col]+=1
            for p,pred in enumerate(preds):
                if p not in matched_pred:extra[cat_to_fdi[pred['category_id']]]+=1
    c=counts
    result={'score_threshold':score,'IoU_threshold':.50,'counts':dict(c),
            'Precision':c['TP']/c['predictions'] if c['predictions'] else None,'Recall':c['TP']/c['GT'],
            'F1':2*c['TP']/(c['predictions']+c['GT']),
            'matched_tooth_FDI_accuracy':c['correct_FDI_matches']/c['localized_matches'] if c['localized_matches'] else None,
            'localization_coverage':c['localized_matches']/c['GT'],
            'raw_duplicate_image_rate':c['duplicate_image']/len(imageids)}
    return result,perimage,matrix,extra


def valid_mean(array):
    vals=np.asarray(array);vals=vals[vals>=0]
    return float(vals.mean()*100) if len(vals) else None


def diagnostic_figures(rows):
    oldspec=importlib.util.spec_from_file_location('legacy_confusion_plot',PAPER/'materials/cmig_revision_2026-10-03/scripts/build_complete_fdi_confusions.py')
    mod=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(mod);mod.DEST=HERE
    endpoints={};curves=[];outcomes={}
    fig,axs=plt.subplots(3,3,figsize=(12.0,9.3))
    for row,(cohort,config,seeds,rep,label) in enumerate(SPEC):
        run=TRAIN/'runs'/config/f'seed_{rep}';folder=run/'evaluation'
        configpath=run/'config.yml';bind(configpath);cfg=yaml.safe_load(configpath.read_text())
        trainpath=Path(cfg['train_dataloader']['dataset']['ann_file']);valpath=Path(cfg['val_dataloader']['dataset']['ann_file'])
        train=read(trainpath);val=read(valpath);m=read(folder/'metrics.json')
        predictionpath=folder/'predictions_coco.json.gz';bind(predictionpath,'full native top-300 predictions')
        with gzip.open(predictionpath,'rt') as f:predictions=json.load(f)
        assert {p['image_id'] for p in predictions}=={i['id'] for i in val['images']}
        assert all(not a.get('ignore',0) and not a.get('iscrowd',0) for a in val['annotations'])
        categories=val['categories']
        if cohort=='Pano':
            assert cfg['fdi_label_order']=='dentex_enumeration32'
            cat_to_fdi={c['id']:PANO32[c['id']] for c in categories}
        else:cat_to_fdi={c['id']:int(c.get('fdi',c['name'])) for c in categories}
        assert len(cat_to_fdi)==cfg['num_classes']
        codes=sorted(cat_to_fdi.values())
        endpoint,perimage,matrix,extra=fixed(predictions,val,return_matrix=True,cat_to_fdi=cat_to_fdi)
        saved=m['fixed_threshold']
        assert endpoint['counts']==saved['counts'],(cohort,endpoint['counts'],saved['counts'])
        for k in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']:
            assert math.isclose(endpoint[k],saved[k],abs_tol=1e-12),(cohort,k)
        counts=endpoint['counts'];assert matrix.sum()==counts['GT'];assert matrix[:,:-1].sum()==counts['localized_matches'];assert np.trace(matrix[:,:-1])==counts['correct_FDI_matches']
        assert sum(extra.values())==counts['predictions']-counts['localized_matches']
        source=next(r for r in rows if r['cohort']==cohort and r['seed']==rep)
        endpoints[cohort]={'cohort':cohort,'configuration':config,'seed':rep,
                           'selection':'Highest AP among this cohort\'s selected three seeds; illustrative representative, not mean performance.',
                           'checkpoint_path':source['checkpoint_path'],'checkpoint_sha256':source['checkpoint_sha256'],
                           **endpoint,'metrics_percent':{k:source[k] for k in METRICS},
                           'fixed_threshold_percent':{k:100*endpoint[k] for k in endpoint if k not in ['counts','score_threshold','IoU_threshold']},
                           'sources':{'train_annotation':str(trainpath),'validation_annotation':str(valpath),'prediction':str(predictionpath),'native_metrics':source['native_metric_path']}}
        traincounts=Counter(cat_to_fdi[a['category_id']] for a in train['annotations']);valcounts=Counter(cat_to_fdi[a['category_id']] for a in val['annotations'])
        train_support=[traincounts[c] for c in codes];val_support=[valcounts[c] for c in codes]
        assert matrix.sum(1).tolist()==val_support
        arrayspath=folder/'coco_precision_recall.npz';bind(arrayspath,'native COCO precision/recall arrays');arrays=np.load(arrayspath)
        cats=arrays['category_ids'].tolist();fdi_to_cat={fdi:cat for cat,fdi in cat_to_fdi.items()}
        perclass=[]
        i50=int(np.flatnonzero(np.isclose(arrays['IoU_thresholds'],.50))[0]);i75=int(np.flatnonzero(np.isclose(arrays['IoU_thresholds'],.75))[0])
        for idx,code in enumerate(codes):
            j=cats.index(fdi_to_cat[code])
            perclass.append({'FDI':code,'category_id':fdi_to_cat[code],'train_support':train_support[idx],'validation_support':val_support[idx],
                             'AP':valid_mean(arrays['precision'][:,:,j,0,-1]),'AP50':valid_mean(arrays['precision'][i50,:,j,0,-1]),
                             'AP75':valid_mean(arrays['precision'][i75,:,j,0,-1]),'AR100':valid_mean(arrays['recall'][:,j,0,-1]),
                             'correct_FDI':int(matrix[idx,idx]),'localized_GT':int(matrix[idx,:-1].sum()),'missed_GT':int(matrix[idx,-1]),'unmatched_predictions':extra[code]})
        for k in METRICS:
            macro=np.mean([r[k] for r in perclass if r[k] is not None]);assert math.isclose(macro,source[k],abs_tol=1e-10),(cohort,k,macro,source[k])
        percentages=np.divide(matrix*100,np.asarray(val_support)[:,None],out=np.full(matrix.shape,np.nan),where=np.asarray(val_support)[:,None]>0)
        number=row+5
        outcome={'cohort':cohort,'configuration':config,'seed':rep,'fdi_order':codes,'prediction_columns':[*codes,'Missed'],
                 'train_support':train_support,'val_support':val_support,'raw_counts':matrix.tolist(),
                 'row_percent':[[float(v) if np.isfinite(v) else None for v in line] for line in percentages],
                 'per_class_metrics':perclass,'unmatched_predictions_by_FDI':dict(extra),
                 'summary':{'images':len(val['images']),'train_images':len(train['images']),'train_teeth':len(train['annotations']),
                            **counts,'no_training_support':[c for c,n in zip(codes,train_support) if not n],
                            'no_validation_support':[c for c,n in zip(codes,val_support) if not n]},
                 'sources':endpoints[cohort]['sources'],'checkpoint_sha256':source['checkpoint_sha256']}
        files=mod.plot(outcome,cohort,number);outcome['figure_files']=files;outcomes[cohort]=outcome
        stem=f'Fig_S{number}_{cohort}_complete_FDI_confusion'
        captions[stem]=f'Figure S{number}. Complete validation FDI outcome matrix for the {label} representative checkpoint (seed {rep}), selected by AP from the stated three-seed subset. Rows include every FDI category and columns include missed ground-truth teeth. Row-normalized percentages use score ≥0.30 and class-agnostic one-to-one Hungarian localization matching at IoU ≥0.50. Grey rows have no validation support; training and validation class counts are shown beside the matrix. Extra predictions do not belong to a ground-truth row and are counted separately. This figure illustrates one selected checkpoint, not a mean or independent-test estimate.'
        figures[stem]={'caption':captions[stem],'files':files,'source_paths':list(endpoints[cohort]['sources'].values()),'checkpoint_sha256':source['checkpoint_sha256']}
        dump(cohort+'_complete_FDI_confusion.json',outcome)
        with (HERE/(cohort+'_complete_FDI_confusion.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['Reference_FDI',*codes,'Missed']);writer.writerows([[c,*line.tolist()] for c,line in zip(codes,matrix)])
        for filename,data in [(cohort+'_per_class_performance_and_support.csv',perclass),(cohort+'_per_image_FDI_outcomes.csv',perimage)]:
            with (HERE/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        precision=arrays['precision'][i50,:,:,0,-1];denom=(precision>=0).sum(1)
        macro=np.divide(np.maximum(precision,0).sum(1),denom,out=np.full(len(denom),np.nan),where=denom>0)
        recall=arrays['recall_thresholds']
        axs[row,0].plot(recall*100,macro*100,color=BLUE,lw=1.5);axs[row,0].set_ylabel(label+'\nPrecision (%)');axs[row,0].set_xlabel('Recall (%)');axs[row,0].set_xlim(0,100);axs[row,0].set_ylim(0,100)
        threshold_values=[]
        for threshold in np.arange(.05,.951,.05):
            mm,_,_,_=fixed(predictions,val,float(threshold))
            threshold_values.append({'score_threshold':float(threshold),'F1':mm['F1']*100,'Precision':mm['Precision']*100 if mm['Precision'] is not None else None,'Recall':mm['Recall']*100,'raw_duplicate_image_rate':mm['raw_duplicate_image_rate']*100})
        axs[row,1].plot([r['score_threshold'] for r in threshold_values],[r['F1'] for r in threshold_values],color=BLUE,lw=1.5)
        axs[row,2].plot([r['score_threshold'] for r in threshold_values],[r['raw_duplicate_image_rate'] for r in threshold_values],color=BLUE,lw=1.5)
        for col in [1,2]:
            axs[row,col].axvline(.30,color=GREY,ls=':',lw=1);axs[row,col].set_xlim(.05,.95);axs[row,col].set_ylim(0,100);axs[row,col].set_xlabel('Score threshold')
        axs[row,1].set_ylabel('F1 (%)');axs[row,2].set_ylabel('Duplicate images (%)')
        for ax in axs[row]:style(ax)
        curves.append({'cohort':cohort,'configuration':config,'seed':rep,'checkpoint_sha256':source['checkpoint_sha256'],'IoU':.50,'precision':macro.tolist(),'recall':recall.tolist(),'threshold_endpoints':threshold_values})
    for ax,title in zip(axs[0],['(A) Detection PR','(B) Threshold–F1','(C) Duplicate labels']):ax.set_title(title,loc='left',pad=12)
    fig.subplots_adjust(left=.09,right=.98,bottom=.075,top=.94,wspace=.37,hspace=.4)
    save(fig,'Figure_S4_detection_curves',
         'Figure S4. Detection and numbering-threshold curves from the representative best-validation checkpoints: DENTEX2023 seed 3, Mobile seed 2, and DSLR seed 3. (A) Supported-class macro COCO precision–recall curves at IoU 0.50. (B) F1 versus score threshold. (C) Raw-query duplicate-image rate versus score threshold. Threshold endpoints use one-to-one class-agnostic localization matching at IoU 0.50 and FDI-correct matches as true positives; dotted lines identify the common reporting threshold 0.30. Predictions are native top-300 outputs without added NMS. These single-checkpoint diagnostics are not averages across seeds or clinical charting-error estimates.',
         [s for r in endpoints.values() for s in r['sources'].values()])
    dump('threshold_curve_data.json',curves);dump('representative_endpoints.json',endpoints);dump('confusion_summary.json',{c:o['summary'] for c,o in outcomes.items()})
    return endpoints


def write_csvs(rows,summaries,priors,prior_summary):
    for name,data in [('results_per_seed.csv',rows),('selected_mean_sd.csv',summaries['selected']),('fixed012_mean_sd.csv',summaries['fixed012']),('all5_mean_sd.csv',summaries['all5']),('prior_per_seed.csv',priors),('prior_selected_mean_sd.csv',prior_summary['selected']),('prior_all5_mean_sd.csv',prior_summary['all5'])]:
        serial=[{k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in data]
        with (HERE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(serial[0]));writer.writeheader();writer.writerows(serial)


def main():
    rows,priors,summaries,prior_summary=load_runs()
    write_csvs(rows,summaries,priors,prior_summary)
    # Write the exact statistical data early so manuscript work can proceed
    # while CPU-only prediction diagnostics are rendered.
    endpoint_summary={group:[] for group in summaries}
    endpoint_keys=['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']
    for cohort,_,selected,_,_ in SPEC:
        for group,seeds in [('selected',selected),('fixed012',[0,1,2]),('all5',list(range(5)))]:
            entry={'cohort':cohort,'seeds':seeds,'n':len(seeds),'post_hoc_selected':group=='selected'}
            for k in endpoint_keys:
                values=[r['fixed_threshold'][k]*100 for r in rows if r['cohort']==cohort and r['seed'] in seeds]
                entry.update({k+'_mean':float(np.mean(values)),k+'_SD':float(np.std(values,ddof=1)),k+'_per_seed':values})
            endpoint_summary[group].append(entry)
    data={'per_seed':rows,'summaries':summaries,'prior_per_seed':priors,'prior_summaries':prior_summary,
          'fixed_threshold_summaries':endpoint_summary,
          'selection_policy':'User-selected after inspecting five completed seeds; selected-three summaries are exploratory, not prespecified primary evidence.',
          'SD_definition':'Sample SD; n−1 denominator; all metrics in percentage points.',
          'no_significance_inference_from_selected_seeds':True}
    dump('quantitative_data.json',data)
    print(json.dumps({'stage':'statistics_saved','selected':summaries['selected']},ensure_ascii=False),flush=True)
    seed_figure(rows,summaries);mobile_framework(rows,summaries);prior_figure(priors,prior_summary);prior_paired(priors,prior_summary);learning_curves(rows)
    endpoints=diagnostic_figures(rows)
    data['representative_endpoints']=endpoints;dump('quantitative_data.json',data)
    dump('captions.json',captions)
    dump('manifest.json',{'font':'Times New Roman','PNG_dpi':600,'SVG_editable_text':True,
                          'figures':figures,'sources':list(bindings.values()),
                          'checks':{'all15_native_vs_export_primary_metrics':True,'checkpoint_SHA256':True,
                                    'mean_SD_independent_scalar_recalculation':True,'all5_and_fixed012_retained':True,
                                    'all20_prior_replays_same_weights':True,'representative_endpoint_counts_independently_recomputed':True,
                                    'FDI_matrix_row_totals_equal_GT_support':True,'perclass_macro_equals_native_COCO':True},
                          'no_training_or_gpu_inference':True})
    print(json.dumps({'stage':'complete','figures':list(figures),'selected':summaries['selected'],
                      'representative_endpoints':endpoints},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
