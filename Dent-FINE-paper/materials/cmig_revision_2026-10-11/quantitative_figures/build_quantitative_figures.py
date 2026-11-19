#!/usr/bin/env python3
"""Publication figures bound to completed native COCO metric records."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile
from lxml import etree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_best3_seeds_2026-10-08/manuscript_data')
OLD = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_final_2026-10-06/figures/performance')
FONT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman')
for name in ['Times.TTF', 'Timesbd.TTF', 'Timesi.TTF', 'Timesbi.TTF']:
    font_manager.fontManager.addfont(str(FONT/name))
assert font_manager.FontProperties(fname=str(FONT/'Times.TTF')).get_name() == 'Times New Roman'
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.titleweight': 'bold', 'axes.linewidth': .8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'text.color': '#222222', 'axes.labelcolor': '#222222',
    'xtick.color': '#222222', 'ytick.color': '#222222',
})
BLUE = '#365E7D'
GREY = '#AAB2B9'
BLACK = '#252A2E'
LIGHT = '#E5E7E9'
bindings = []
captions = {}


def read_json(name):
    p = DATA/name
    value = json.loads(p.read_text())
    bindings.append({'path':str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return value


def bind_native(rows):
    for row in rows:
        p=Path(row['native_metric_path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['native_metric_sha256'], p
        native=json.loads(p.read_text())
        for key in ['AP','AP50','AP75','AR100']:
            assert math.isclose(row[key], native['metrics_percent'][key], abs_tol=1e-10), (p,key)
        bindings.append({'path':str(p),'sha256':row['native_metric_sha256']})


def style(ax, gridaxis='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    if gridaxis:
        ax.grid(axis=gridaxis, color=LIGHT, linewidth=.6)


def save(fig,name):
    fig.savefig(HERE/(name+'.svg'), bbox_inches='tight', metadata={'Date':None})
    fig.savefig(HERE/(name+'.pdf'), bbox_inches='tight', metadata={'CreationDate':None, 'ModDate':None})
    fig.savefig(HERE/(name+'.png'), dpi=500, bbox_inches='tight')
    fig.savefig(HERE/(name+'_preview.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def dump(name,rows):
    (HERE/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False))


def primary_figure(primary):
    order=[('Pano','DENTEX2023'),('Mobile','Mobile'),('DSLR','DSLR')]
    fig,axs=plt.subplots(1,3,figsize=(12.0,3.9))
    out=[]
    for panel,(cohort,label),ax in zip('ABC',order,axs):
        rows=sorted([r for r in primary if r['cohort']==cohort],key=lambda r:r['seed'])
        vals=np.array([r['AP'] for r in rows])
        m,s=vals.mean(),vals.std(ddof=1)
        ax.scatter([0,1,2], vals, color=GREY, edgecolor=BLACK, linewidth=.7,s=48,zorder=3)
        ax.errorbar(3.35,m,yerr=s,fmt='D',color=BLUE,ms=6,capsize=5,lw=1.4,zorder=4)
        ax.set_xticks([0,1,2,3.35],['Seed 0','Seed 1','Seed 2','Mean'])
        lo=min(vals.min(),m-s);hi=max(vals.max(),m+s)
        pad=max((hi-lo)*.25,.2)
        ax.set_ylim(lo-pad,hi+pad*2.3)
        ax.set_xlim(-.45,3.9)
        ax.set_ylabel('COCO AP (%)')
        ax.set_title(f'({panel}) {label}',loc='left',pad=12)
        ax.text(.5,.95,f'{m:.3f} ± {s:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=12,color=BLUE)
        style(ax)
        out.append({'cohort':cohort,'AP_seeds':vals.tolist(),'AP_mean':m,'AP_SD':s,'configuration':rows[0]['configuration']})
    fig.subplots_adjust(left=.06,right=.985,bottom=.14,top=.86,wspace=.36)
    save(fig,'Figure_6_three_seed_AP')
    dump('Figure_6_values',out)
    captions['Figure_6_three_seed_AP']='Figure 6. Training-seed variation of the three domain-configured DENT-STRIDE models. Circles show the actual AP obtained with seeds 0, 1, and 2; diamonds and error bars show the mean and sample standard deviation. (A) DENTEX2023 with anatomical scoring. (B) Mobile. (C) DSLR. Each panel has an independent AP axis to display within-cohort variation. All estimates use validation-selected checkpoints on the respective fixed image partitions; the error bars are not patient-level confidence intervals.'


def framework_figure(primary):
    source=OLD/'dslr_accuracy_size_snapshot.json'
    snapshot=json.loads(source.read_text())
    bindings.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    for src in snapshot['upstream_metric_sources']:
        assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
        bindings.append(src)
    rows=[dict(r) for r in snapshot['rows'] if r['model'] not in ['Controlled base','CMFA + AdamW']]
    ours=next(r for r in rows if r['model']=='DENT-STRIDE-DSLR')
    selected=[r for r in primary if r['cohort']=='DSLR']
    vals=np.array([r['AP'] for r in selected])
    ours.update(AP=float(vals.mean()),SD=float(vals.std(ddof=1)),Params_M=selected[0]['params_total']/1e6,
                source='Current DSLR CMFA + DualOpt; seeds 0/1/2; native COCO evaluator')
    fig,axs=plt.subplots(1,2,figsize=(12.0,5.2),gridspec_kw={'width_ratios':[1.05,1]})
    offset_full={
        'RT-DETR-L':(5,8),'DENT-STRIDE-DSLR':(20,19),'Official DEIM-s':(20,52),
        'YOLOv8-s':(18,-22),'Dome-DETR-S':(35,-35),'RT-DETRv2-S':(6,10),
        'YOLO11-s':(10,-40),'CAF-YOLO-s':(12,3),'Faster R-CNN R50-FPN':(-8,12),
    }
    offset_detail={
        'DENT-STRIDE-DSLR':(25,3),'Official DEIM-s':(-17,28),
        'YOLOv8-s':(25,-3),'Dome-DETR-S':(-15,-20),'YOLO11-s':(10,-22),
    }
    for ax,offsets,title in zip(axs,[offset_full,offset_detail],['(A) Framework comparison','(B) Smaller-model detail']):
        for r in rows:
            if r['model'] not in offsets:continue
            x,y=r['Params_M'],r['AP'];focal=r['model']=='DENT-STRIDE-DSLR'
            color=BLUE if focal else BLACK
            marker='D' if focal else 'o'
            if r['SD'] is not None:
                ax.errorbar(x,y,yerr=r['SD'],fmt=marker,ms=5.5,color=color,capsize=3,lw=1.1,zorder=3)
            else:ax.scatter([x],[y],s=37,color='white',edgecolor=BLACK,linewidth=1,zorder=3)
            off=offsets[r['model']]
            ax.annotate(r['model'],(x,y),xytext=off,textcoords='offset points',
                        ha='right' if off[0]<0 else 'left',va='bottom',fontsize=11,
                        color=color,arrowprops={'arrowstyle':'-','color':GREY,'lw':.7},zorder=4)
        ax.set_title(title,loc='left',pad=12)
        ax.set_xlabel('Parameters (M)');ax.set_ylabel('COCO AP (%)')
        style(ax)
    axs[0].set_xlim(7.5,44.6);axs[0].set_ylim(61.7,76)
    axs[1].set_xlim(8.85,14.8);axs[1].set_ylim(70.3,73.9)
    axs[1].set_xticks([9,10,11,12,13,14])
    fig.subplots_adjust(left=.06,right=.985,bottom=.12,top=.89,wspace=.25)
    save(fig,'Figure_7_DSLR_accuracy_size')
    dump('Figure_7_values',rows)
    captions['Figure_7_DSLR_accuracy_size']='Optional DSLR size comparison. (A) Framework-level comparison, including RT-DETR-L. (B) Enlarged view of the smaller models. DENT-STRIDE-DSLR, Official DEIM-s, and RT-DETRv2-S are plotted as three-seed means with sample-standard-deviation error bars; open circles denote single-seed conventional-detector results without a dispersion estimate. DENT-STRIDE-DSLR uses 11.154 M parameters and achieves 72.293 ± 0.651 AP. The detector systems retain their framework-specific training and pretraining settings. No Pareto-frontier connection is imposed, and the figure does not establish a size–accuracy advantage over Official DEIM-s.'


def mobile_framework_figure(primary):
    manuscript=Path('/home/wuxi/Dent-FINE-paper/DENT_STRIDE_CMIG_main_final_20261006.docx')
    with zipfile.ZipFile(manuscript) as archive:
        root=etree.fromstring(archive.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    table=root.xpath('.//w:body/w:tbl',namespaces=ns)[4]
    original={}
    for row in table.xpath('./w:tr',namespaces=ns)[1:]:
        cells=[' '.join(c.xpath('.//w:t/text()',namespaces=ns)) for c in row.xpath('./w:tc',namespaces=ns)]
        original[cells[0]]=cells[3]
    bindings.append({'path':str(manuscript),'sha256':hashlib.sha256(manuscript.read_bytes()).hexdigest(),'table':5})
    raw=[('Faster R-CNN R50-FPN',41.182),('CAF-YOLO-s',43.771),
         ('YOLOv8-s',46.937),('YOLO11-s',47.064),('RT-DETR-L',54.751),('Dome-DETR-S',51.828)]
    for name,ap in raw:assert math.isclose(float(original[name]),ap,abs_tol=1e-10)
    rows=[{'model':name,'AP':ap,'SD':None,'n':1,'source':'Main manuscript Table 5, unchanged framework comparator'} for name,ap in raw]
    for impl,name in [('deim_s','Official DEIM-s'),('rtdetrv2_s','RT-DETRv2-S')]:
        vals=[]
        for seed in range(3):
            p=Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10/runs')/impl/'phone'/f'seed_{seed}'/'metrics_summary.json'
            d=json.loads(p.read_text());vals.append(d['best_metrics']['AP'])
            bindings.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        a=np.array(vals)
        rows.append({'model':name,'AP':float(a.mean()),'SD':float(a.std(ddof=1)),'n':3,'per_seed_AP':vals,'source':'Official implementation best-validation checkpoints, seeds 0/1/2'})
    vals=np.array([r['AP'] for r in sorted(primary,key=lambda r:r['seed']) if r['cohort']=='Mobile'])
    rows.append({'model':'DENT-STRIDE-Mobile','AP':float(vals.mean()),'SD':float(vals.std(ddof=1)),'n':3,'per_seed_AP':vals.tolist(),'source':'Current native Mobile checkpoints, seeds 0/1/2'})
    rows=sorted(rows,key=lambda r:r['AP'],reverse=True)
    fig,ax=plt.subplots(figsize=(10.0,4.9))
    for y,r in enumerate(rows):
        c=BLUE if r['model']=='DENT-STRIDE-Mobile' else BLACK
        if r['SD'] is not None:
            ax.errorbar(r['AP'],y,xerr=r['SD'],fmt='D' if r['model']=='DENT-STRIDE-Mobile' else 'o',color=c,ms=6,capsize=4,lw=1.2,zorder=3)
        else:
            ax.scatter(r['AP'],y,s=42,facecolor='white',edgecolor=BLACK,lw=1,zorder=3)
        label=f"{r['AP']:.3f}"+(f" ± {r['SD']:.3f}" if r['SD'] is not None else '')
        ax.text(1.015,y,label,transform=ax.get_yaxis_transform(),va='center',fontsize=12,color=c)
    ax.set_yticks(range(len(rows)),[r['model'] for r in rows]);ax.set_ylim(len(rows)-.6,-.6)
    ax.set_xlim(40,56.3);ax.set_xlabel('COCO AP (%)');ax.set_xticks([40,44,48,52,56])
    style(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.25,right=.82,bottom=.14,top=.97)
    save(fig,'Figure_7_Mobile_framework_AP');dump('Figure_7_Mobile_values',rows)
    captions['Figure_7_Mobile_framework_AP']='Figure 7. Framework-level detection AP on Mobile images. DENT-STRIDE-Mobile, Official DEIM-s, and RT-DETRv2-S are shown as three-seed means with sample-standard-deviation error bars; open circles indicate single-seed detector results. All external comparators are retained, including RT-DETR-L. DENT-STRIDE-Mobile achieves the highest numerical mean AP among the displayed detector systems (55.219 ± 0.327). The models retain framework-specific resolution, pretraining, optimization, and post-processing, so the plot describes complete detector systems rather than an isolated module effect or statistical superiority.'


def prior_figure(priors,summary):
    order=['none','center','shape','both'];labels=['None','Center','Shape','Center + shape']
    fig,axs=plt.subplots(1,2,figsize=(10.9,4.1))
    for ax,key,panel in zip(axs,['AP','AR100'],'AB'):
        for i,mode in enumerate(order):
            vals=np.array([r[key] for r in sorted(priors,key=lambda r:r['seed']) if r['prior']==mode])
            m,s=vals.mean(),vals.std(ddof=1)
            expected=next(r for r in summary if r['prior']==mode)
            assert math.isclose(m,expected[key+'_mean'],abs_tol=1e-10)
            assert math.isclose(s,expected[key+'_SD'],abs_tol=1e-10)
            ax.scatter(i+np.array([-.12,0,.12]),vals,color=GREY,edgecolor='white',linewidth=.4,s=30,zorder=3)
            ax.errorbar(i+.2,m,yerr=s,fmt='D',color=BLUE,ms=5,capsize=4,lw=1.2,zorder=4)
        ax.set_xticks(np.arange(4)+.05,labels)
        ax.set_xlim(-.45,3.6)
        ax.set_title(f'({panel}) {key}',loc='left',pad=12)
        ax.set_ylabel(f'{key} (%)')
        style(ax)
    axs[0].set_ylim(52.65,54.65);axs[1].set_ylim(65.7,70.1)
    handles=[Line2D([],[],marker='o',ls='',color=GREY,label='Individual seed'),
             Line2D([],[],marker='D',ls='',color=BLUE,label='Mean ± SD')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.07,right=.985,bottom=.16,top=.83,wspace=.24)
    save(fig,'Figure_9_Pano_prior_comparison')
    dump('Figure_9_values',summary)
    captions['Figure_9_Pano_prior_comparison']='Figure 9. Anatomical scoring on the same three DENTEX2023 checkpoints. None, center-only, shape-only, and combined center–shape settings are evaluated from candidates before top-k selection. Circles show individual-seed results and diamonds show mean ± sample standard deviation. (A) COCO AP. (B) AR100. Center scoring improves mean AP by 0.494 points relative to no prior; combined scoring improves AP by 0.484 points and gives the highest mean AR100. The same checkpoint is retained across the four scoring settings within each seed; these comparisons measure inference corrections rather than training effects.'


def delta_panel(ax,rows,labels,title,xlim=None):
    y=np.arange(len(rows))
    vals=np.array([r['delta_AP_vs_reference'] for r in rows])
    ax.axvline(0,color=BLACK,lw=.9)
    for i,v in enumerate(vals):
        ax.barh(i,v,height=.5,color=BLUE if v<0 else GREY,edgecolor=BLACK,linewidth=.5,zorder=3)
        if abs(v)<1e-12:ax.plot(0,i,'o',ms=4,color=BLACK,zorder=4)
    ax.set_yticks(y,labels);ax.set_ylim(len(rows)-.5,-.5)
    low=min(vals.min(),0);high=max(vals.max(),0)
    span=max(high-low,.4)
    ax.set_xlim(*(xlim or (low-.20*span,high+.32*span)))
    for i,v in enumerate(vals):
        ax.text(1.015,i,f'{v:+.3f}',transform=ax.get_yaxis_transform(),
                ha='left',va='center',fontsize=11)
    ax.set_title(title,loc='left',pad=12)
    ax.set_xlabel('AP difference (points)');style(ax,'x')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)


def learning_effect_figure(internal):
    specs=[('HOOA branches',['Full','− High-pass','− Second-order','− Both'],'(A) HOOA branches'),
           ('Factor supervision',['All three','− Quadrant','− Position','− Dentition'],'(B) Factor supervision')]
    fig,axs=plt.subplots(1,2,figsize=(11.2,3.6))
    selected=[]
    for ax,(component,labels,title) in zip(axs,specs):
        rows=[r for r in internal if r['component']==component]
        delta_panel(ax,rows,labels,title,(-1.55,.25));selected.extend(rows)
    fig.subplots_adjust(left=.14,right=.975,bottom=.19,top=.86,wspace=.50)
    save(fig,'Figure_11_HOOA_Factor_learning_effects')
    dump('Figure_11_values',selected)
    captions['Figure_11_HOOA_Factor_learning_effects']='Figure 11. Learning-effect ablations on Mobile images with seed 0. Bars show AP differences from the intact setting within each group, using separately trained models. (A) Removing the HOOA high-pass branch, second-order interaction, or both reduces AP by 0.399, 0.967, or 1.349 points relative to the full-module reference (54.910 AP). (B) Removing quadrant, position, or dentition supervision reduces AP by 0.654, 1.274, or 0.244 points relative to the all-factor reference (55.537 AP). Training and inference factor-score fusion are disabled in all supervision comparisons. No across-seed uncertainty is assigned to these single-seed ablations.'


def sensitivity_figure(internal):
    def rows(group):return [r for r in internal if r['component']==group]
    prototypes=rows('Prototype design')
    specs=[
        (rows('CMFA gates'),['Both','Channel only','Spatial only','Neither'],'(A) CMFA gates'),
        (prototypes[:4],['K=4','No prototype','K=1','K=2'],'(B) Prototype number: τ=0.20'),
        ([prototypes[i] for i in [4,0,5]],['τ=0.10','τ=0.20','τ=0.50'],'(C) Prototype temperature: K=4'),
        (rows('DualOpt operations'),['AdamW','Rollback','Extra decay','Both'],'(D) DualOpt operations'),
        (rows('Mobile background losses'),['Both','Neither','NWD only','Arch only'],'(E) Background losses'),
        (rows('Factor fusion'),['On / On','On / Off','Off / Off','Off / On'],'(F) Factor fusion: train / infer'),
    ]
    fig,axs=plt.subplots(3,2,figsize=(12.6,10.4))
    for ax,(rs,labels,title) in zip(axs.flat,specs):delta_panel(ax,rs,labels,title)
    fig.subplots_adjust(left=.13,right=.975,bottom=.065,top=.96,wspace=.55,hspace=.65)
    save(fig,'Figure_S6_component_sensitivity')
    dump('Figure_S6_values',[r for r in internal if r['component'] not in ['HOOA branches','Factor supervision']])
    captions['Figure_S6_component_sensitivity']='Figure S6. Component and training-setting sensitivity on seed 0. Signed AP differences are relative to the stated group reference, not to a universal baseline. (A–C, E) Reference: the Mobile configuration with both CMFA gates, four prototypes at temperature 0.20, and both background losses (54.910 AP). (D) Reference: DSLR CMFA with ordinary AdamW (72.892 AP), with its standard weight decay unchanged across the additional operations. (F) Reference: Mobile with factor fusion disabled during training and inference (55.537 AP). All learning changes use separately trained models; within each training-fusion state, inference on/off reuses the same weights. Every tested setting is retained, including settings that exceed the intact configuration. These single-seed comparisons have no repeated-seed error bars.'


def prior_interval_figure(priors,summary):
    modes=['center','shape','both'];labels=['Center','Shape','Center + shape']
    fig,ax=plt.subplots(figsize=(8.8,3.3));out=[]
    tcrit=math.sqrt(2*.95**2/(1-.95**2))
    none={r['seed']:r['AP'] for r in priors if r['prior']=='none'}
    for y,mode in enumerate(modes):
        delta=np.array([r['AP']-none[r['seed']] for r in sorted(priors,key=lambda r:r['seed']) if r['prior']==mode])
        m,s=delta.mean(),delta.std(ddof=1);width=tcrit*s/math.sqrt(3)
        r=next(r for r in summary if r['prior']==mode)
        assert math.isclose(m,r['AP_delta_vs_none_mean'],abs_tol=1e-10)
        assert math.isclose(m-width,r['AP_delta_vs_none_CI95_low'],abs_tol=1e-10)
        assert math.isclose(m+width,r['AP_delta_vs_none_CI95_high'],abs_tol=1e-10)
        ax.scatter(delta,y+np.array([-.12,0,.12]),s=25,color=GREY,zorder=3)
        ax.errorbar(m,y,xerr=width,fmt='D',color=BLUE,capsize=5,ms=6,lw=1.4,zorder=4)
        ax.text(.93,y,f'{m:+.3f}',va='center',ha='left',fontsize=12)
        out.append({'prior':mode,'paired_deltas':delta.tolist(),'mean':m,'SD':s,'CI95_low':m-width,'CI95_high':m+width})
    ax.axvline(0,color=BLACK,lw=.9)
    ax.set_yticks(range(3),labels);ax.set_ylim(2.55,-.55);ax.set_xlim(-.15,1.1)
    ax.set_xlabel('Paired AP difference from no prior (points)')
    style(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.23,right=.97,bottom=.23,top=.95)
    save(fig,'Figure_S12_prior_paired_AP_intervals')
    dump('Figure_S12_values',out)
    captions['Figure_S12_prior_paired_AP_intervals']='Figure S12. Paired AP effects of DENTEX2023 anatomical scoring relative to no prior, using the same checkpoint within each seed. Grey circles show the three actual paired differences; diamonds show their mean and horizontal bars show Student-t 95% confidence intervals for the three-seed mean (df=2). Center-only, shape-only, and joint scoring are all retained. The checkpoint selection used the configured prior, and the intervals describe seed variation on the same validation split; they are not patient-level sampling intervals or independent-test evidence.'


def learning_curves(primary):
    fig,axs=plt.subplots(1,3,figsize=(12.0,3.9));out=[]
    order=[('Pano','DENTEX2023'),('Mobile','Mobile'),('DSLR','DSLR')]
    for ax,(cohort,label),panel in zip(axs,order,'ABC'):
        rows=sorted([r for r in primary if r['cohort']==cohort],key=lambda r:r['seed'])
        curves=[]
        for row,ls in zip(rows,['solid','dashed','dotted']):
            p=Path(row['native_metric_path']).parent/'log.txt'
            log=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            assert [r['epoch'] for r in log]==list(range(160)),p
            vals=np.array([r['test_coco_eval_bbox'][0]*100 for r in log])
            assert math.isclose(vals.max(),row['AP'],abs_tol=2e-4),(p,vals.max(),row['AP'])
            curves.append(vals)
            ax.plot(np.arange(1,161),vals,color=GREY,ls=ls,lw=.75,alpha=.8,zorder=2)
            bindings.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            out.append({'cohort':cohort,'configuration':row['configuration'],'seed':row['seed'],
                        'epochs_one_based':list(range(1,161)),'AP':vals.tolist()})
        array=np.stack(curves);mean=array.mean(axis=0);sd=array.std(axis=0,ddof=1)
        ax.fill_between(np.arange(1,161),mean-sd,mean+sd,color=BLUE,alpha=.16,lw=0,zorder=1)
        ax.plot(np.arange(1,161),mean,color=BLUE,lw=1.5,zorder=3)
        ax.set_title(f'({panel}) {label}',loc='left',pad=12)
        ax.set_xlabel('Epoch');ax.set_ylabel('Validation AP (%)')
        ax.set_xlim(1,160);ax.set_xticks([1,40,80,120,160])
        ax.set_ylim(0,60 if cohort!='DSLR' else 80)
        style(ax)
    handles=[Line2D([],[],color=GREY,ls=ls,label=f'Seed {i}') for i,ls in enumerate(['solid','dashed','dotted'])]
    handles.append(Line2D([],[],color=BLUE,lw=1.5,label='Mean ± SD'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,1.015),ncol=4,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.06,right=.985,bottom=.15,top=.82,wspace=.35)
    save(fig,'Figure_S5_learning_curves');dump('Figure_S5_values',out)
    captions['Figure_S5_learning_curves']='Figure S5. Validation-AP learning curves of the final domain-configured models over 160 epochs. (A) DENTEX2023. (B) Mobile. (C) DSLR. Thin grey lines show the actual epoch-wise results of seeds 0, 1, and 2, distinguished by solid, dashed, and dotted line styles. The blue line and shaded band show the epoch-wise mean and sample standard deviation across the three seeds, without smoothing. The panels include all 160 recorded epochs from the current completed runs. These epoch-wise averages differ from the mean of each seed’s individually selected best checkpoint reported in Table 3.'


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    primary=read_json('primary_seed_results.json');internal=read_json('Table7_internal_seed0.json')
    priors=read_json('spatial_shape_per_seed.json');summary=read_json('Table6_SpatialShape_three_seed.json')
    bind_native(primary);bind_native(internal);bind_native(priors)
    assert len(primary)==9 and len(priors)==12
    primary_figure(primary);framework_figure(primary);mobile_framework_figure(primary);prior_figure(priors,summary)
    learning_effect_figure(internal);sensitivity_figure(internal);prior_interval_figure(priors,summary)
    learning_curves(primary)
    (HERE/'captions.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False))
    unique={r['path']:r for r in bindings}
    (HERE/'source_bindings.json').write_text(json.dumps({
        'font':'Times New Roman','PNG_dpi':500,'SVG_editable_text':True,
        'sources':list(unique.values()),'native_COCO_values_verified':True,
        'primary_seed_count':9,'prior_replays':12,
        'no_new_training_or_checkpoint_selection':True,
        'Figure7_excluded_points':['Controlled base','CMFA + AdamW'],
        'main_Figure7':'Mobile framework AP; all eight external comparators retained',
        'optional_DSLR_plot':'Updated proposed point only; obsolete controlled-family points omitted',
    },indent=2))
    print(json.dumps({'directory':str(HERE),'figures':list(captions),'sources':len(unique)}))


if __name__=='__main__':main()
