"""Re-use established diagnostic renderers against the new, immutable predictions."""
from pathlib import Path
import importlib.util
import json
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PAPER = Path('/home/wuxi/Dent-FINE-paper')
ROOT = PAPER / 'materials/cmig_revision_2026-10-11'
TRAIN = PAPER / 'materials/cmig_best3_seeds_2026-10-08'
OLD = PAPER / 'materials/cmig_revision_2026-10-03'
DEST = ROOT / 'prediction_diagnostics'
DEST.mkdir(parents=True, exist_ok=True)
for f in (OLD / 'references/times_new_roman').glob('*.TTF'):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman', 'font.size':10,
                     'svg.fonttype':'none','pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

spec = importlib.util.spec_from_file_location('confusion_renderer', OLD/'scripts/build_complete_fdi_confusions.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TRAIN, module.DEST = TRAIN, DEST
original_read = module.read

def read_export_metrics(path):
    data = original_read(path)
    # Per-category arrays use the exported pixel-space detections. Preserve this
    # source's own metrics rather than falsely comparing to a different native
    # bbox conversion. Published aggregate AP stays the native result.
    if str(path).endswith('/metrics.json') and 'export_metrics_fraction' in data:
        data = dict(data)
        data['metrics_percent'] = {k:100*v for k,v in data['export_metrics_fraction'].items()}
    return data

module.read = read_export_metrics
JOBS = [('Pano','pano_historical_spatialshape','evaluation',7),
        ('Mobile','mobile_1024_full','evaluation',8),
        ('DSLR','dslr_cmfa_dualopt','evaluation',9)]
outcomes = [module.analyze(*job) for job in JOBS]
(DEST/'confusion_summary.json').write_text(json.dumps({r['cohort']:r['summary'] for r in outcomes},indent=2))

# Detection PR and threshold curves use all seed-0 validation outputs; no data
# smoothing, aggregation over seeds, or synthetic confidence bounds are added.
import sys
sys.path.insert(0,str(TRAIN))
from evaluate_controlled import fixed_endpoints
from pycocotools.coco import COCO
fig, axes = plt.subplots(3,3,figsize=(8.4,6.7),layout='constrained')
curve_data=[]
for row,(cohort,configuration,replay,_) in enumerate(JOBS):
    folder=TRAIN/'runs'/configuration/'seed_0'/replay
    job=json.loads((TRAIN/'jobs'/(configuration+'_seed0.json')).read_text())
    import yaml
    cfg=yaml.safe_load(Path(job['config']).read_text())
    gt=COCO(cfg['val_dataloader']['dataset']['ann_file'])
    with gzip.open(folder/'predictions_coco.json.gz','rt') as handle:
        predictions=json.load(handle)
    arrays=np.load(folder/'coco_precision_recall.npz')
    precision=arrays['precision'][0,:,:,0,-1]
    counts=(precision>=0).sum(1)
    macro=np.divide(np.maximum(precision,0).sum(1),counts,
                    out=np.full(len(counts),np.nan),where=counts>0)
    recall=arrays['recall_thresholds']
    axes[row,0].plot(recall*100,macro*100,color='#315c79',lw=1.5)
    axes[row,0].set_xlim(0,100);axes[row,0].set_ylim(0,100)
    axes[row,0].set_ylabel(cohort+'\nPrecision (%)')
    thresholds=np.arange(.05,.951,.05)
    endpoint=[]
    for threshold in thresholds:
        m,_=fixed_endpoints(predictions,gt,float(threshold),.50)
        endpoint.append({'score':float(threshold),'F1':m['F1']*100,
                         'duplicate_image_rate':m['raw_duplicate_image_rate']*100})
    axes[row,1].plot(thresholds,[e['F1'] for e in endpoint],color='#315c79',lw=1.5)
    axes[row,2].plot(thresholds,[e['duplicate_image_rate'] for e in endpoint],color='#315c79',lw=1.5)
    for col in [1,2]:
        axes[row,col].axvline(.30,color='#7a7a7a',ls=':',lw=.8)
        axes[row,col].set_xlim(.05,.95);axes[row,col].set_ylim(0,100)
        axes[row,col].set_xlabel('Score threshold')
    axes[row,1].set_ylabel('F1 (%)');axes[row,2].set_ylabel('Duplicate images (%)')
    axes[row,0].set_xlabel('Recall (%)')
    for ax in axes[row]:ax.grid(color='#e8e8e8',lw=.5)
    curve_data.append({'cohort':cohort,'configuration':configuration,
                       'precision':macro.tolist(),'recall':recall.tolist(),'threshold_endpoints':endpoint})
for ax,title in zip(axes[0],['(A) Detection PR','(B) Threshold–F1','(C) Duplicate labels']):
    ax.set_title(title,loc='left',fontsize=11)
for ext in ['png','svg','pdf']:
    fig.savefig(DEST/f'Figure_S6_detection_curves.{ext}',dpi=600,bbox_inches='tight',pad_inches=.08)
fig.savefig(DEST/'Figure_S6_detection_curves_preview.png',dpi=140,bbox_inches='tight')
plt.close(fig)
(DEST/'threshold_curve_data.json').write_text(json.dumps(curve_data,indent=2))
print(json.dumps({r['cohort']:r['summary'] for r in outcomes},indent=2))
