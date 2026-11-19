"""Publication figures connecting retrained controls to observed predictions."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
MATERIALS=Path('/home/wuxi/Dent-FINE-paper/materials')
for f in (MATERIALS/'cmig_revision_2026-10-03/references/times_new_roman').glob('*.TTF'):
 font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':10,
 'axes.titlesize':10.5,'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42,
 'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
EVIDENCE=json.loads((HERE/'controlled_evidence.json').read_text())
BLUE='#285E7A';GRAY='#8B949B';TEAL='#087B80';ORANGE='#AD5528'
CAPTIONS={};FIGURES={}


def export(fig,name,caption,source):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.findobj(match=lambda o:isinstance(o,matplotlib.text.Text)):
  if not t.get_text() or not t.get_visible():continue
  bb=t.get_window_extent(renderer)
  assert bb.x0>=-.5 and bb.y0>=-.5 and bb.x1<=fig.bbox.width+.5 and bb.y1<=fig.bbox.height+.5,(name,t.get_text(),bb)
 for ext in ['png','pdf','svg']:fig.savefig(HERE/f'{name}.{ext}',dpi=600)
 fig.savefig(HERE/f'{name}_preview.png',dpi=150)
 plt.close(fig)
 CAPTIONS[name]=caption;FIGURES[name]=source


def lookup(cohort):
 ann=json.loads(Path(EVIDENCE['data'][cohort][0]).read_text())
 return {r['id']:int(r.get('fdi',r['name'])) for r in ann['categories']}


def crop_for(box,W,H):
 x,y,w,h=box
 cw=max(3.1*w,2.25*h*4/3);ch=cw*3/4
 scale=min(1,W/cw,H/ch);cw*=scale;ch*=scale
 x0=min(max(x+w/2-cw/2,0),W-cw);y0=min(max(y+h/2-ch/2,0),H-ch)
 return [float(x0),float(y0),float(x0+cw),float(y0+ch)]


def photo_grid(fig,cases,bottoms,height=.175,titles=None,row_labels=None):
 titles=titles or ['Original','Ground truth','Control','Complete']
 for i,(case,bottom) in enumerate(zip(cases,bottoms)):
  im=Image.open(case['image_file']).convert('RGB');crop=crop_for(case['target_annotation']['bbox'],im.width,im.height)
  case['display_crop_xyxy']=crop
  fdi=lookup(case['cohort'])
  for j in range(4):
   ax=fig.add_axes([.040+j*.240,bottom,.215,height])
   ax.imshow(im,interpolation='antialiased')
   if j:
    gt=case['target_annotation'];gx,gy,gw,gh=gt['bbox']
    if j>=2:ax.add_patch(Rectangle((gx,gy),gw,gh,fill=False,edgecolor=TEAL,linewidth=.9,linestyle=(0,(3,2))))
    p=gt if j==1 else case['before' if j==2 else 'after']['prediction']
    if p:
     x,y,w,h=p['bbox'];color=TEAL if j==1 else ORANGE
     ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=1.25))
     ax.text(x+.03*w,y-.035*(crop[3]-crop[1]),str(fdi[p['category_id']]),
             color=color,fontsize=10.8,fontweight='bold',ha='left',va='bottom',
             bbox={'facecolor':'white','edgecolor':'none','alpha':.93,'pad':.65})
   ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   if i==0:ax.set_title(titles[j],fontsize=10,pad=7)
  label=row_labels[i] if row_labels else f'({chr(ord("b")+i)})'
  fig.text(.008,bottom+height+.003,label,fontsize=10.5,ha='left',va='bottom')


def cmfa():
 group=EVIDENCE['groups']['CMFA'];fig=plt.figure(figsize=(8.35,5.85))
 ax=fig.add_axes([.115,.705,.845,.235])
 for i,s in enumerate(group['summaries']):
  for dx,value in zip([-.12,0,.12],s['paired_AP_deltas']):
   ax.scatter(i+dx,value,s=31,facecolor='white',edgecolor=GRAY,lw=1.15,zorder=4)
  ax.errorbar(i,s['mean_paired_delta_AP'],yerr=s['sample_SD_paired_delta_AP'],fmt='D',
              color=BLUE,markersize=5.5,capsize=4,lw=1.5,zorder=5)
  ax.text(i,.785,f"+{s['mean_paired_delta_AP']:.3f} ± {s['sample_SD_paired_delta_AP']:.3f}",
           ha='center',va='center',color=BLUE,fontsize=10.5,fontweight='bold')
 ax.set_xlim(-.55,1.55);ax.set_ylim(0,.85);ax.set_xticks([0,1],['AdamW','DualOpt'])
 ax.set_ylabel('Paired AP gain (points)');ax.grid(axis='y',color='#E5E8EA',lw=.55)
 ax.tick_params(axis='x',length=0,pad=5);ax.set_yticks([0,.2,.4,.6,.8])
 fig.text(.008,.955,'(a)',fontsize=11,va='top')
 photo_grid(fig,group['cases'],[.375,.065],height=.235,
            titles=['Original','Ground truth','Without CMFA','With CMFA'])
 cap=('Controlled evidence for CMFA on DSLR images. (a) Paired COCO AP differences between separately trained models '
      'with and without CMFA at seeds 0, 1, and 2. Open circles show individual-seed differences; diamonds and bars show '
      'the mean and sample SD. Mean gains are 0.498±0.145 points with AdamW and 0.513±0.091 with DualOpt. '
      'Only the CMFA configuration flag differs within each optimizer/seed pair; DualOpt here includes extra decay and rollback. '
      '(b) A seed-0 AdamW example changes an incorrect FDI 44 prediction to the reference 84. '
      '(c) For reference FDI 65, localization IoU increases from 0.553 to 0.894. '
      'Teal boxes denote ground truth, also shown dashed behind orange prediction boxes. '
      'Predictions use score≥0.30; one-to-one localization matching requires IoU≥0.50. '
      'The examples are illustrative and pertain to the stated controlled DSLR configurations. '
      'The seed bars describe validation training variability, not patient-level or independent-test uncertainty.')
 export(fig,'Figure_CMFA_controlled_evidence',cap,group)


def hooa():
 group=EVIDENCE['groups']['HOOA'];rows=group['runs']
 order=[3,1,2,0];labels=['Neither','No HP','No 2nd','Full']
 fig=plt.figure(figsize=(8.35,6.15))
 metrics=[('AP','COCO AP (%)',lambda r:r['metrics']['AP']),
          ('F1','F1 (%)',lambda r:r['fixed_threshold']['F1']*100),
          ('Dup','Duplicate-image rate (%)',lambda r:r['fixed_threshold']['raw_duplicate_image_rate']*100)]
 for j,(_,title,value) in enumerate(metrics):
  ax=fig.add_axes([.090+j*.312,.715,.260,.215]);vv=[value(rows[k]) for k in order]
  span=max(vv)-min(vv);lo=min(vv)-span*.25;hi=max(vv)+span*.40
  for x,y in enumerate(vv):
   ax.scatter(x,y,color=BLUE if x==3 else GRAY,s=33,zorder=3)
   ax.text(x,y+span*.07,f'{y:.2f}',ha='center',va='bottom',fontsize=8.4,
           color=BLUE if x==3 else '#333333',weight='bold' if x==3 else 'normal')
  ax.set_xlim(-.45,3.45);ax.set_ylim(lo,hi);ax.set_xticks(range(4),labels,fontsize=8.4)
  ax.set_ylabel(title,fontsize=9.5);ax.grid(axis='y',color='#E5E8EA',lw=.55);ax.tick_params(axis='x',length=0,pad=5)
 fig.text(.008,.95,'(a)',fontsize=11,va='top')
 cases=[]
 for source in sorted(group['cases'],key=lambda r:0 if r['target_FDI']==81 else 1):
  cases.append({'cohort':'Mobile','image_id':source['image_id'],'image_file':source['image_file'],
    'target_annotation':source['target_annotation'],'target_FDI':source['target_FDI'],
    'before':source['statuses']['Neither branch'],'after':source['statuses']['Full HOOA']})
 photo_grid(fig,cases,[.380,.075],height=.225,
            titles=['Original','Ground truth','Neither branch','Full HOOA'])
 cap=('Retrained HOOA branch controls on the Mobile validation cohort. (a) COCO AP, fixed-threshold F1, and raw-query '
      'duplicate-image rate for models trained without either branch, without high-pass, without second-order interaction, '
      'and with complete HOOA. Relative to neither branch, complete HOOA increases AP from 53.561 to 54.910 and F1 from '
      '78.861% to 82.431%, while raw duplicate-image rate decreases from 43.182% to 27.273%. '
      'All four are single-seed retraining controls in the same reference configuration (prototype temperature 0.20); '
      'no seed uncertainty is assigned. Removing the second-order interaction retains the context branch in its place; '
      'removing high-pass supplies a zero branch. (b,c) Identical target crops show FDI 81 and 41, respectively, '
      'misclassified as 41 and 42 without either branch and correctly identified by the full module. '
      'Ground truth is teal and predictions orange; dashed teal boxes indicate the same ground truth in prediction panels. '
      'Score≥0.30 and one-to-one IoU≥0.50 matching are used. Duplicate-image rate is a raw-output diagnostic, '
      'not a clinical charting-error rate. Selected examples do not estimate correction frequencies.')
 export(fig,'Figure_HOOA_controlled_evidence',cap,{'group':group,'cases':cases})


def factor_learning():
 group=EVIDENCE['groups']['Factor'];rows=group['runs'];fig=plt.figure(figsize=(8.35,7.6))
 ax=fig.add_axes([.110,.785,.850,.170])
 values=[r['metrics']['AP'] for r in rows]
 for x,y in enumerate(values):
  ax.scatter(x,y,s=43,color=BLUE if x==0 else GRAY,zorder=4)
  ax.text(x,y+.13,f'{y:.3f}',ha='center',va='bottom',fontsize=10,color=BLUE if x==0 else '#333333',weight='bold' if x==0 else 'normal')
 ax.set_xticks(range(4),['All factors','No quadrant loss','No position loss','No dentition loss'],fontsize=9)
 ax.set_xlim(-.45,3.45);ax.set_ylim(53.85,55.95);ax.set_ylabel('COCO AP (%)')
 ax.grid(axis='y',color='#E5E8EA',lw=.55);ax.tick_params(axis='x',length=0,pad=5)
 fig.text(.008,.97,'(a)',fontsize=11,va='top')
 photo_grid(fig,group['cases'],[.520,.300,.080],height=.170,
  titles=['Original','Ground truth','Head loss omitted','All factor losses'],
  row_labels=['(b) Quadrant','(c) Position','(d) Dentition'])
 cap=('Learning effects of the individual factor-supervision heads. (a) COCO AP with all three factor losses and '
      'after removing quadrant, within-quadrant position, or dentition supervision. Each control is independently trained '
      'at seed 0 with the same reference configuration and prototype temperature 0.20. Only the stated loss term is '
      'omitted; factor heads remain instantiated, and factor-score fusion is disabled during both training and inference. '
      'AP decreases by 0.654, 1.274, and 0.244 points, respectively. '
      '(b) Without quadrant supervision, the reference FDI 31 is predicted as 41. '
      '(c) Without position supervision, FDI 64 is predicted as 65. '
      '(d) Without dentition supervision, permanent FDI 33 is predicted as primary FDI 73. '
      'All-factor supervision yields the correct highest retained label in these examples. Teal denotes ground truth '
      'and orange predictions; dashed ground-truth boxes are repeated in prediction panels. '
      'Score≥0.30 and one-to-one IoU≥0.50 matching are used. Cases are illustrative; the AP result does not imply '
      'that every fixed-threshold metric improves or that statistical superiority has been established.')
 export(fig,'Figure_Factor_supervision_evidence',cap,group)


def oral_crop(record):
 ann=json.loads(Path(EVIDENCE['data']['Mobile'][0]).read_text())
 boxes=np.asarray([g['bbox'] for g in ann['annotations'] if g['image_id']==record['image_id']])
 lo=boxes[:,:2].min(0);hi=(boxes[:,:2]+boxes[:,2:]).max(0);cx,cy=(lo+hi)/2
 W,H=record['image_record']['width'],record['image_record']['height']
 cw=max(1.12*(hi[0]-lo[0]),1.12*(hi[1]-lo[1])*4/3);ch=cw*3/4
 assert cw<=W and ch<=H,'Use a padded canvas rather than silently clipping teeth'
 x0=min(max(cx-cw/2,0),W-cw);y0=min(max(cy-ch/2,0),H-ch)
 return [float(x0),float(y0),float(x0+cw),float(y0+ch)]


def attributions(method):
 prov=json.loads((HERE/'factor_head_attribution_provenance.json').read_text())
 fig=plt.figure(figsize=(8.35,5.9));records=prov['records'];display=[]
 for row,record in enumerate(records):
  im=Image.open(Path(EVIDENCE['data']['Mobile'][1])/record['image_record']['file_name']).convert('RGB')
  data=np.load(record['maps']);crop=oral_crop(record);y0=[.710,.420,.130][row]
  for col,name in enumerate(['original','quadrant','position','dentition']):
   ax=fig.add_axes([.040+col*.240,y0,.215,.225]);ax.imshow(im,interpolation='antialiased')
   if col:
    key=name+'_margin_'+method;raw=data[key];relative=raw/max(float(raw.max()),1e-12)
    rgba=plt.colormaps['inferno'](relative);rgba[...,3]=.80*relative
    ax.imshow(rgba,extent=(0,im.width,im.height,0),interpolation='bilinear')
    h=record['heads'][name];k=h['predicted_index']
    label='Q'+str(k+1) if name=='quadrant' else str(k+1) if name=='position' else 'Primary' if k==0 else 'Permanent'
    title={'quadrant':'Quadrant','position':'Position','dentition':'Dentition'}[name]+': '+label
    display.append({'image_id':record['image_id'],'head':name,'map':key,'normalization':'raw positive attribution divided by full-image maximum',
                    'raw_max':float(raw.max()),'predicted_component':label,'probability':h['target_probability']})
   else:title='FDI '+str(record['FDI'])
   x,y,w,h=record['target_annotation']['bbox']
   ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=TEAL if not col else 'white',linewidth=1.05,linestyle=(0,(3,2)) if col else 'solid'))
   ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   ax.set_title(title,fontsize=9.8,pad=7)
  fig.text(.008,y0+.220,'('+chr(97+row)+')',fontsize=10.5,va='top')
 cbax=fig.add_axes([.520,.055,.410,.018])
 cb=fig.colorbar(plt.cm.ScalarMappable(cmap='inferno',norm=Normalize(0,1)),cax=cbax,orientation='horizontal',ticks=[0,.5,1])
 cb.ax.set_title('Relative attribution',fontsize=9.5,pad=4);cb.ax.tick_params(length=2,labelsize=8);cb.outline.set_visible(False)
 stem='Figure_Factor_head_'+('HiResCAM' if method=='hirescam' else 'GradCAM')
 cap=('Query-specific spatial attributions of the three factor heads from the representative DENT-STRIDE-Mobile '
      'seed-2 checkpoint (prototype temperature 0.50). Rows show reference FDI 64, 11, and 46; column headings give '
      'the actual predicted quadrant, position, and dentition. For each head, the scalar target is the correct factor '
      'logit minus its largest competing logit, evaluated for the highest-scoring correct-FDI query with localization '
      'IoU≥0.50 and score≥0.30. The attributed feature is the output of encoder.hooa_attention.0. '
      + ('The positive map is ReLU(sum_c A_c * d margin/d A_c), retaining spatial gradients (HiResCAM form). '
         if method=='hirescam' else 'Grad-CAM first averages each channel gradient spatially before weighting activations. ')
      + 'The actual gradient maps are computed over the full image without ground-truth masking; the shown oral crops '
      'are identical across heads. Each map is scaled by its full-image maximum to display relative spatial structure; '
      'color intensity is not comparable as an absolute effect across heads. Teal/white boxes locate the queried reference '
      'tooth and do not constrain the attribution. The native detached query-content initialization path is not traversed '
      'by these gradients; conditional input occlusion complements this partial-path explanation. '
      'These are post hoc gradient explanations, not native attention gates, '
      'a faithfulness guarantee for this Transformer architecture, or standalone evidence of a performance gain. '
      'The retrained supervision controls provide the performance evidence.')
 export(fig,stem,cap,{'provenance':prov,'display':display,'method':method})


def head_confusions():
 source=MATERIALS/'cmig_selected_seed_revision_2026-10-12/prediction'
 counts=np.load(source/'factor_confusion_raw.npz');prov=json.loads((source/'mechanism_provenance.json').read_text())
 fig=plt.figure(figsize=(8.35,3.05))
 for j,(head,title) in enumerate([('quadrant','Quadrant'),('position','Position'),('dentition','Dentition')]):
  ax=fig.add_axes([.075+j*.315,.225,.240,.625])
  matrix=counts[head];den=matrix.sum(1,keepdims=True)
  val=np.divide(matrix,den,out=np.zeros_like(matrix,dtype=float),where=den>0)*100
  val[den[:,0]==0]=np.nan
  cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#E4E6E8')
  ax.imshow(val,cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
  acc=np.trace(matrix)/matrix.sum()*100
  ax.set_title(f'{title}\n{acc:.2f}%',fontsize=11,pad=8)
  labels=[str(i) for i in range(1,9)] if head!='dentition' else ['Primary','Permanent']
  ax.set_xticks(range(len(labels)),labels,fontsize=8.3);ax.set_yticks(range(len(labels)),labels,fontsize=8.3)
  if head=='dentition':ax.tick_params(axis='x',labelsize=8.8);ax.tick_params(axis='y',labelsize=8.8)
  for row in range(len(matrix)):
   for col in range(len(matrix)):
    if not np.isfinite(val[row,col]):
     if row==col:ax.text(col,row,'—',ha='center',va='center',fontsize=9,color='#555555')
    elif row==col or val[row,col]>=3:
     ax.text(col,row,f'{val[row,col]:.0f}',ha='center',va='center',fontsize=8.6 if len(matrix)>2 else 12,
             color='white' if val[row,col]>=60 else '#222222')
  ax.set_xlabel('Predicted',fontsize=9);ax.set_ylabel('Reference',fontsize=9)
 cap=('Factor-head classification on 2,339 uniquely localized ordinary queries from all 220 Mobile validation images '
      'using the representative seed-2 checkpoint (prototype temperature 0.50). Matrices are row-normalized percentages; '
      'quadrant, position, and dentition accuracies are 95.08%, 96.07%, and 95.55%, respectively. Gray rows have no '
      'reference observations. A query is eligible when its maximum sigmoid score is≥0.30; class-agnostic one-to-one '
      'localization matching requires IoU≥0.50. These conditional head accuracies are not detector recall or complete '
      'FDI detection accuracy; no calibration or clinical utility claim follows from them.')
 export(fig,'Figure_Factor_head_confusions',cap,{'raw_count_file':str(source/'factor_confusion_raw.npz'),
            'checkpoint_sha256':prov['checkpoint_sha256'],'matched_queries':int(counts['quadrant'].sum())})


def attribution_support():
 """Make sparse attribution visible via explicitly defined positive-mass sets."""
 from matplotlib.patches import Patch
 prov=json.loads((HERE/'factor_head_attribution_provenance.json').read_text())
 fig=plt.figure(figsize=(8.35,5.9));display=[]
 for row,record in enumerate(prov['records']):
  im=Image.open(Path(EVIDENCE['data']['Mobile'][1])/record['image_record']['file_name']).convert('RGB')
  maps=np.load(record['maps']);crop=oral_crop(record);bottom=[.710,.420,.130][row]
  for col,name in enumerate(['original','quadrant','position','dentition']):
   ax=fig.add_axes([.040+col*.240,bottom,.215,.225]);ax.imshow(im,interpolation='antialiased')
   if col:
    raw=maps[name+'_margin_hirescam'];rank=np.sort(raw.ravel())[::-1];cumulative=np.cumsum(rank,dtype=np.float64)
    cuts={f:float(rank[min(int(np.searchsorted(cumulative,f*cumulative[-1])),len(rank)-1)]) for f in [.50,.90]}
    mask90=raw>=cuts[.90];mask50=raw>=cuts[.50]
    assert cuts[.90]>0 and np.all(~mask50|mask90)
    rgba=np.zeros((*raw.shape,4),dtype=float)
    rgba[mask90]=[1.,.65,.02,.52];rgba[mask50]=[.90,.15,.06,.78]
    ax.imshow(rgba,extent=[0,im.width,im.height,0],interpolation='nearest')
    head=record['heads'][name];k=head['predicted_index']
    label='Q'+str(k+1) if name=='quadrant' else str(k+1) if name=='position' else 'Primary' if k==0 else 'Permanent'
    title=name.capitalize()+': '+label
    display.append({'image_id':record['image_id'],'head':name,'FDI':record['FDI'],
       'mass50_threshold':cuts[.50],'mass90_threshold':cuts[.90],
       'mass50_retained':float(raw[mask50].sum()/raw.sum()),'mass90_retained':float(raw[mask90].sum()/raw.sum()),
       'mass50_grid_cells':int(mask50.sum()),'mass90_grid_cells':int(mask90.sum()),
       'GT_used_as_spatial_mask_or_to_shape_support':False,
       'GT_used_to_select_query_and_target_factor':True})
   else:title='FDI '+str(record['FDI'])
   x,y,w,h=record['target_annotation']['bbox']
   ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=TEAL if col==0 else 'white',
                          linewidth=1.05,linestyle='solid' if col==0 else (0,(3,2))))
   ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   ax.set_title(title,fontsize=9.8,pad=7)
  fig.text(.008,bottom+.220,'('+chr(97+row)+')',fontsize=10.5,va='top')
 fig.legend(handles=[Patch(facecolor='#E62610',alpha=.78,label='50% positive attribution mass'),
                     Patch(facecolor='#FFA605',alpha=.65,label='90% positive attribution mass')],
            loc='lower center',bbox_to_anchor=(.64,.020),ncol=2,frameon=False,fontsize=9.3,
            handlelength=1.2,columnspacing=1.4)
 cap=('Spatial support of the individual factor-head decisions. Three Mobile examples use the representative seed-2 '
      'checkpoint with prototype temperature 0.50. The target FDI and each actual head prediction are shown above the '
      'images. Positive attributions are computed for the correct factor logit minus its strongest alternative at the '
      'HOOA level-0 output, using ReLU(sum_c activation_c times gradient_c). Red and amber mark the smallest sets of '
      'feature-grid cells accounting for at least 50% and 90% of the full-image positive attribution, respectively; '
      'the sets are nested and ties at a threshold are retained. Maps are computed over the full image, not masked '
      'by the ground-truth tooth. Dashed boxes only locate the queried reference tooth. This mass-support display '
      'makes sparse gradients visible without smoothing, painting anatomical regions, or interpreting them as native '
      'attention probabilities. Context outside the tooth can also contribute. These selected examples explain '
      'factor decisions; retrained supervision ablations, not the colors, provide evidence of detection benefit. '
      'The native detached query-content initialization path is not traversed by these gradients; conditional input '
      'occlusion complements this partial-path explanation.')
 export(fig,'Figure_Factor_head_spatial_support',cap,{'provenance':prov,'display':display})


def occlusion(overlay=False):
 prov=json.loads((HERE/'factor_occlusion_provenance.json').read_text())
 records=prov['records'];limits={};maps={}
 for entry in records:maps[entry['reference']['image_id']]=np.load(entry['maps'])
 for name in ['quadrant','position','dentition']:
  limits[name]=max(float(np.max(np.abs(maps[r['reference']['image_id']][name+'_dense_margin_drop']))) for r in records)
 fig=plt.figure(figsize=(8.35,6.0));display=[]
 for row,entry in enumerate(records):
  record=entry['reference'];iid=record['image_id'];im=Image.open(Path(EVIDENCE['data']['Mobile'][1])/record['image_record']['file_name']).convert('RGB')
  crop=oral_crop(record);bottom=[.720,.430,.140][row]
  for col,name in enumerate(['original','quadrant','position','dentition']):
   ax=fig.add_axes([.040+col*.240,bottom,.215,.220])
   if col==0 or overlay:ax.imshow(im,interpolation='antialiased')
   if col:
    v=maps[iid][name+'_dense_margin_drop']
    ax.imshow(v,cmap='RdBu_r',vmin=-limits[name],vmax=limits[name],extent=(0,im.width,im.height,0),
              interpolation='nearest',alpha=.65 if overlay else 1.)
    h=record['heads'][name];k=h['predicted_index']
    label='Q'+str(k+1) if name=='quadrant' else str(k+1) if name=='position' else 'Primary' if k==0 else 'Permanent'
    title=name.capitalize()+': '+label
    display.append({'image_id':iid,'head':name,'raw_signed_domain':[-limits[name],limits[name]],'crop_xyxy':crop})
   else:title='FDI '+str(record['FDI'])
   x,y,w,h=record['target_annotation']['bbox']
   ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=TEAL if col==0 else '#444444',
                          linewidth=1.1,linestyle='solid' if col==0 else (0,(3,2))))
   ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   ax.set_title(title,fontsize=9.8,pad=7)
  fig.text(.008,bottom+.215,'('+chr(97+row)+')',fontsize=10.5,va='top')
 for j,name in enumerate(['quadrant','position','dentition']):
  cax=fig.add_axes([.285+j*.240,.066,.205,.018]);v=limits[name]
  cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-v,v),cmap='RdBu_r'),cax=cax,
                  orientation='horizontal',ticks=[-v,0,v])
  cb.ax.set_xticklabels([f'{-v:.1f}','0',f'{v:.1f}']);cb.ax.tick_params(length=2,labelsize=8)
  cb.ax.set_title('Margin drop',fontsize=9,pad=4);cb.outline.set_visible(False)
 stem='Figure_Factor_head_occlusion_'+('overlay' if overlay else 'heatmaps')
 cap=('Conditional input-occlusion sensitivity of the factor heads. Rows show Mobile reference FDI 64, 11, and 46 '
      'using the same seed-2 checkpoint (prototype temperature 0.50). Each 1024×1024 input is perturbed at 841 sliding '
      'locations using a 128×128 patch with stride 32, filled with that image\'s RGB channel means. The scalar is the '
      'correct factor logit minus its strongest unoccluded alternative. Red indicates that occluding the region reduces '
      'the decision margin; blue indicates that it increases the margin. The dense signed map averages margin drops '
      'over all patches covering each pixel; no Gaussian smoothing or ground-truth spatial masking is applied. '
      'Each head has one common symmetric logit-margin scale across the three images. Encoder anchor selection '
      'and query order are fixed to the unoccluded pass, while query content and downstream predictions are recomputed '
      'for every perturbed input. This prevents query swaps and includes the query-content path detached in the '
      'gradient analysis. Unoccluded predictions are exactly unchanged by this conditioning. '
      'Dashed boxes locate the queried tooth; surrounding context can also influence the factor decisions. '
      'This is conditional-query sensitivity, not native attention, native detector accuracy on occluded images, '
      'or an independent proof of model efficacy. Mean-color masks may introduce artificial edges. '
      + ('Overlays use a constant opacity of 0.65.' if overlay else 'Pure sensitivity maps are shown to avoid dilution by photographic colors.'))
 export(fig,stem,cap,{'provenance':prov,'display':display,'overlay':overlay})


def main():
 cmfa();hooa();factor_learning();attributions('hirescam');attributions('gradcam');head_confusions();attribution_support()
 occlusion(False);occlusion(True)
 (HERE/'figure_captions.json').write_text(json.dumps(CAPTIONS,ensure_ascii=False,indent=2))
 (HERE/'figure_manifest.json').write_text(json.dumps(FIGURES,ensure_ascii=False,indent=2))
 print('GENERATED',list(FIGURES),flush=True)


if __name__=='__main__':main()
