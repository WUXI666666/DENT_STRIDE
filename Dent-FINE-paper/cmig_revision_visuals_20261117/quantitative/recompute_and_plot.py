"""Recompute module contrasts from existing per-run exports, without model training.

The source paths are explicit and hashes are verified before plotting. All six
available CMFA pairs are included. Prototype is one separately trained seed-0
pair, with no seed-variability estimate. Run with the original workspace files.
"""
from pathlib import Path
import hashlib
import csv
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
MATERIALS = Path('/home/wuxi/Dent-FINE-paper/materials')
EVIDENCE = MATERIALS/'effectiveness_evidence_2026-10-30/controlled_evidence.json'
GALLERY = MATERIALS/'module_cam_gallery_2026-10-30'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),''):
            if not block: break
            h.update(block)
    return h.hexdigest()

def diff(a,b,path=''):
    changes=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            if key not in ('output_dir','summary_dir'):
                changes.extend(diff(a.get(key),b.get(key),path+'.'+key))
    elif a!=b:
        changes.append({'field':path,'before':a,'after':b})
    return changes

verified=[]
def verify_run(run):
    metric=read(run['metric_file'])
    cp=Path(run['checkpoint'])
    assert sha(cp)==metric['checkpoint_sha256']==run['checkpoint_sha256']
    config=Path(run['config'])
    cfg=yaml.safe_load(config.read_text())
    # Original saved hashes and current files are independently retained.
    assert sha(config)==metric['config_sha256']
    assert np.isclose(metric['metrics_percent']['AP'],run['metrics']['AP'])
    verified.append({'configuration':run['configuration'],'seed':run['seed'],
                     'metric_file':run['metric_file'],'metric_sha256':sha(run['metric_file']),
                     'checkpoint':str(cp),'checkpoint_sha256':sha(cp),
                     'config':str(config),'config_sha256':sha(config),
                     'precision':metric['precision'],'independent_test':metric['independent_test']})
    return metric,cfg

evidence=read(EVIDENCE)
rows=[]
all_metrics=[]
for pair in evidence['groups']['CMFA']['runs']:
    mb,cb=verify_run(pair['before'])
    ma,ca=verify_run(pair['after'])
    changes=diff(cb,ca)
    assert changes==[{'field':'.HybridEncoder.use_cmfa_fusion','before':False,'after':True}]
    before,after=mb['metrics_percent']['AP'],ma['metrics_percent']['AP']
    rows.append({'group':'CMFA','cohort':'DSLR','optimizer':pair['optimizer'],
                 'seed':pair['seed'],'before_AP':before,'after_AP':after,'delta_AP':after-before})
    for metric in ('AP','AP50','AP75','AR100'):
        bv,av=mb['metrics_percent'][metric],ma['metrics_percent'][metric]
        all_metrics.append({'group':'CMFA','optimizer':pair['optimizer'],'seed':pair['seed'],
                            'metric':metric,'before':bv,'after':av,'delta':av-bv})

summaries=[]
for opt in ('adamw','dualopt'):
    rr=[r for r in rows if r['optimizer']==opt]
    assert {r['seed'] for r in rr}=={0,1,2}
    delta=np.array([r['delta_AP'] for r in rr])
    summaries.append({'group':'CMFA','optimizer':opt,'n_pairs':len(rr),
                      'before_mean_AP':float(np.mean([r['before_AP'] for r in rr])),
                      'after_mean_AP':float(np.mean([r['after_AP'] for r in rr])),
                      'paired_AP_deltas':delta.tolist(),
                      'mean_paired_delta_AP':float(delta.mean()),
                      'sample_SD_paired_delta_AP':float(delta.std(ddof=1))})

prototype=[]
for side in ('before','after'):
    source=read(GALLERY/f'Prototype_{side}_capture.json')['source']
    root=Path(source['job']['output_dir'])
    mf=root/'native_coco_metrics.json'
    if not mf.exists(): mf=root/'metrics_summary.json'
    run={'metric_file':str(mf),'checkpoint':source['checkpoint'],
         'checkpoint_sha256':source['checkpoint_sha256'],'config':source['job']['config'],
         'metrics':source['metrics'],'configuration':source['job']['configuration'],'seed':0}
    m,c=verify_run(run)
    prototype.append((m,c))
pm0,pc0=prototype[0];pm1,pc1=prototype[1]
pchanges=diff(pc0,pc1)
assert {d['field'] for d in pchanges}=={
    '.DEIMCriterion.losses','.DFINETransformer.prototype_tau','.DFINETransformer.use_tooth_prototype'}
assert pc0['DFINETransformer']['use_tooth_prototype'] is False
assert pc1['DFINETransformer']['prototype_tau']==0.5
assert pc1['DFINETransformer'].get('prototype_fusion_weight')==0
prow={'group':'Prototype','cohort':'Mobile','optimizer':'adamw','seed':0,
      'before_AP':pm0['metrics_percent']['AP'],'after_AP':pm1['metrics_percent']['AP'],
      'delta_AP':pm1['metrics_percent']['AP']-pm0['metrics_percent']['AP']}
rows.append(prow)
for metric in ('AP','AP50','AP75','AR100'):
    bv,av=pm0['metrics_percent'][metric],pm1['metrics_percent'][metric]
    all_metrics.append({'group':'Prototype','optimizer':'adamw','seed':0,
                        'metric':metric,'before':bv,'after':av,'delta':av-bv})

for filename,data in [('paired_ap.csv',rows),('all_primary_secondary_endpoints.csv',all_metrics)]:
    with (OUT/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
summary={'CMFA':summaries,'Prototype':prow,'prototype_config_differences':pchanges,
         'DualOpt_definition':'AdamW + extra decay 0.01 + initialization rollback 0.005; not the final extra-decay-only DSLR configuration',
         'scope':'Within-configuration internal-validation contrasts; no significance tests or external-generalization inference',
         'all_available_CMFA_pairs_included':True,
         'sources':verified,'dataset_audit':str(MATERIALS/'cmig_controlled_2026-09-29/audit/dataset_audit.json')}
(OUT/'recomputed_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

font_dir=MATERIALS/'cmig_revision_2026-10-03/references/times_new_roman'
for f in font_dir.glob('*.TTF'): font_manager.fontManager.addfont(str(f))
family='Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
plt.rcParams.update({'font.family':family,'font.size':8.5,'axes.titlesize':9.5,
                     'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
                     'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
fig,axes=plt.subplots(1,3,figsize=(174/25.4,69/25.4),gridspec_kw={'wspace':0.35})
blue='#33658a';teal='#257c70';gray='#83949e'
markers=['o','^','s']
for ax,opt,color,label in zip(axes[:2],('adamw','dualopt'),(blue,teal),('A','B')):
    rr=[r for r in rows if r['group']=='CMFA' and r['optimizer']==opt]
    for r,offset in zip(rr,(-.065,0,.065)):
        xx=np.array([0,1])+offset
        ax.plot(xx,[r['before_AP'],r['after_AP']],color=color,lw=1.0,alpha=.7,zorder=2)
        ax.scatter(xx[0],r['before_AP'],s=25,marker=markers[r['seed']],facecolor='white',edgecolor=color,linewidth=.9,zorder=3)
        ax.scatter(xx[1],r['after_AP'],s=25,marker=markers[r['seed']],facecolor=color,edgecolor=color,linewidth=.9,zorder=3)
    ax.set_title(f'({label}) CMFA: '+('AdamW' if opt=='adamw' else 'DualOpt'),loc='left',fontweight='bold')
    ax.set_xlim(-.32,1.32);ax.set_ylim(71.75,73.65)
    ax.set_xticks([0,1],['Without\nCMFA','With\nCMFA']);ax.set_yticks([72,72.5,73,73.5])
    ax.set_ylabel('Validation AP (%)' if opt=='adamw' else '')
    su=next(s for s in summaries if s['optimizer']==opt)
    ax.text(.5,.95,'ΔAP = '+f"{su['mean_paired_delta_AP']:+.3f} ± {su['sample_SD_paired_delta_AP']:.3f}",transform=ax.transAxes,ha='center',va='top',fontsize=8.4,color=color)
    ax.text(.5,-.27,'Three paired seeds; mean ± sample SD',transform=ax.transAxes,ha='center',fontsize=7.5)

ax=axes[2]
ax.set_title('(C) Prototype: Mobile',loc='left',fontweight='bold')
ax.plot([0,1],[prow['before_AP'],prow['after_AP']],color=blue,lw=1.1,zorder=2)
ax.scatter(0,prow['before_AP'],s=29,facecolor='white',edgecolor=blue,linewidth=1,zorder=3)
ax.scatter(1,prow['after_AP'],s=29,facecolor=blue,edgecolor=blue,linewidth=1,zorder=3)
for pos,k in [(0,'before_AP'),(1,'after_AP')]:
    ax.annotate(f"{prow[k]:.3f}",(pos,prow[k]),xytext=(0,-16 if pos==0 else 7),textcoords='offset points',ha='center',fontsize=8.3,color=blue)
ax.set_xlim(-.32,1.32);ax.set_ylim(54.35,56.2)
ax.set_xticks([0,1],['Without\nprototype','With\nprototype'])
ax.set_yticks([54.5,55,55.5,56])
ax.text(.5,.95,'ΔAP = '+f"{prow['delta_AP']:+.3f}",transform=ax.transAxes,ha='center',va='top',fontsize=8.4,color=blue)
ax.text(.5,-.27,'Seed 0; K = 4, τ = 0.50',transform=ax.transAxes,ha='center',fontsize=7.5)
for ax in axes:
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e4e9ec',lw=.45)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#87939c');ax.spines[side].set_linewidth(.65)
    ax.tick_params(length=2.3,width=.65,color='#87939c')
handles=[Line2D([0],[0],marker=m,color='none',markeredgecolor='#354b5d',markerfacecolor='white',markersize=4,label=f'Seed {s}') for s,m in enumerate(markers)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.355,-.005),ncol=3,frameon=False,fontsize=7.5,columnspacing=1.1,handletextpad=.35)
fig.subplots_adjust(left=.08,right=.985,top=.86,bottom=.29)
fig.savefig(OUT/'Figure_Module_controlled_AP.pdf',dpi=300)
fig.savefig(OUT/'Figure_Module_controlled_AP.svg',dpi=300)
fig.savefig(OUT/'Figure_Module_controlled_AP_preview.png',dpi=200)
plt.close(fig)
print(json.dumps({'CMFA':summaries,'Prototype':prow,'sources_verified':len(verified)},indent=2))
