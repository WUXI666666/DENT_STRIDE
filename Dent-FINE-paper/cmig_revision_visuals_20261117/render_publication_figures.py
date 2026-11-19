"""Recompose selected real CAM and occlusion evidence for a 174 mm journal page.

No model is retrained and no attribution values are modified. The imported
renderer loads saved full-input maps; normalisation and cropping match its
documented protocol. Outputs are separate from the original evidence files.
"""
from pathlib import Path
import importlib.util, json, hashlib
from io import BytesIO
import numpy as np
from PIL import Image
import fitz
from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT=Path('/home/wuxi/Dent-FINE-paper')
WORK=ROOT/'cmig_revision_visuals_20261117'
OUT=WORK/'figures';OUT.mkdir(parents=True,exist_ok=True)
GALLERY=ROOT/'materials/module_cam_gallery_2026-10-30'
EVIDENCE=ROOT/'materials/effectiveness_evidence_2026-10-30'
spec=importlib.util.spec_from_file_location('original_cam_renderer',GALLERY/'render_gallery.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,
 'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6,'savefig.facecolor':'white'})
WIDTH=174/25.4
TEAL=r.TEAL;ORANGE=r.ORANGE;GRAY=r.GRAY
PW=1.215;PH=PW*.75;GAP=.14;LEFT=(WIDTH-5*PW-4*GAP)/2
TOP=.34;ROWSTEP=PH+.35;BOTTOM=.44
CAPTIONS={};MANIFEST={}

def box(ax,bb,color=TEAL,label=None,style='solid',case=None):
 x,y,w,h=bb
 ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,lw=.85,linestyle=style))
 if label is not None:
  crop=r.case_crop(case)
  ax.text(x+.02*w,y-.025*(crop[3]-crop[1]),str(label),ha='left',va='bottom',
    fontsize=8.5,color=color,fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':.93,'pad':.5})
r.box=box

def shared_footer(fig,with_cam=True):
 h=fig.get_figheight()
 fig.text(.018,.085/h,'Ground truth',fontsize=7.5,color=TEAL,va='bottom')
 fig.text(.185,.085/h,'Prediction',fontsize=7.5,color=ORANGE,va='bottom')
 if with_cam:
  ax=fig.add_axes([.50,.19/h,.26,.065/h])
  cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='Reds'),cax=ax,
   orientation='horizontal',ticks=[0,1]);cb.ax.tick_params(length=1.5,pad=1,labelsize=7)
  cb.outline.set_visible(False)
  fig.text(.63,.32/h,'Relative positive attribution',ha='center',fontsize=7.5)

def export(fig,stem,caption,metadata,lossless=False):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for text in fig.findobj(match=lambda o:isinstance(o,plt.matplotlib.text.Text)):
  if not text.get_visible() or not text.get_text():continue
  bb=text.get_window_extent(renderer)
  assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(stem,text.get_text(),bb)
 fig.savefig(OUT/f'{stem}.svg',dpi=300)
 temp=OUT/f'{stem}_raw.pdf';fig.savefig(temp,dpi=300)
 fig.savefig(OUT/f'{stem}_preview.png',dpi=170)
 plt.close(fig)
 d=fitz.open(temp);page=d[0];text_before=page.get_text();encodings=[]
 for item in page.get_images(full=True):
  xref=item[0];im=Image.open(BytesIO(d.extract_image(xref)['image'])).convert('RGB')
  buf=BytesIO()
  # Pure signed heatmaps stay lossless. For CAM overlays, photometric compression
  # affects only the displayed raster layer, never the saved attribution arrays.
  if lossless:im.save(buf,'PNG',optimize=True);kind='lossless PNG'
  else:im.save(buf,'JPEG',quality=94,subsampling=0,optimize=True);kind='JPEG94 4:4:4'
  page.replace_image(xref,stream=buf.getvalue());encodings.append(kind)
 # Reconstruct the page to discard replaced, now-unused image streams.
 clean=fitz.open();clean.new_page(width=page.rect.width,height=page.rect.height)
 clean[0].show_pdf_page(clean[0].rect,d,0)
 dst=OUT/f'{stem}.pdf';clean.save(dst,garbage=4,deflate=True)
 assert fitz.open(dst)[0].get_text()==text_before
 temp.unlink()
 CAPTIONS[stem]=caption
 MANIFEST[stem]={**metadata,'size_inches':[WIDTH,float(page.rect.height/72)],
   'raster_dpi':300,'output_pdf_bytes':dst.stat().st_size,'raster_encodings':encodings,
   'clinical_image_or_attribution_values_modified':False}
 print(stem,round(dst.stat().st_size/1024),'KiB',flush=True)

def panel_axes(fig,row,col,nrows):
 h=fig.get_figheight();bottom=h-TOP-PH-row*ROWSTEP
 return fig.add_axes([(LEFT+col*(PW+GAP))/WIDTH,bottom/h,PW/WIDTH,PH/h])

def scores(ax,case):
 a=r.maps(case,'after')['prototype_class_logits'];cid=case['target_annotation']['category_id']
 train=json.loads(Path('/home/wuxi/coco/train/annotations_fdi_types.json').read_text())
 supported={v['category_id'] for v in train['annotations']}
 top=[int(i) for i in np.argsort(-a) if int(i) in supported][:5]
 labels=r.label_map(r.DATA['Prototype']);ys=np.arange(5)
 pos=ax.get_position();ax.set_position([pos.x0+.016,pos.y0,pos.width-.016,pos.height])
 ax.barh(ys,a[top],height=.62,color=[TEAL if i==cid else GRAY for i in top])
 ax.set_yticks(ys,[str(labels[i]) for i in top]);ax.set_ylim(4.6,-.6)
 ax.set_xlim(0,max(float(a[top].max())*1.15,1));ax.set_xticks([0,1.5,3])
 ax.tick_params(length=2,pad=1.5);ax.set_xlabel('Auxiliary score',labelpad=2)
 return [{'FDI':labels[i],'score':float(a[i])} for i in top]

def comparison(group,case_ids,stem,prototype=False):
 cases=[next(c for c in r.DATA[group]['cases'] if c['case_id']==cid) for cid in case_ids]
 n=len(cases);fig=plt.figure(figsize=(WIDTH,TOP+PH+(n-1)*ROWSTEP+BOTTOM))
 if prototype:titles=['Ground truth','No prototype loss','Prototype loss','Prototype CAM','Auxiliary scores']
 else:titles=['Ground truth','Reduced HOOA' if group=='HOOA' else 'Without CMFA',
  'Full HOOA' if group=='HOOA' else 'With CMFA','Control CAM',f'{group} CAM']
 source=[]
 for row,case in enumerate(cases):
  extras={}
  for col in range(5):
   ax=panel_axes(fig,row,col,n)
   if col==0:r.original(ax,case)
   elif col in [1,2]:r.prediction(ax,case,'before' if col==1 else 'after')
   elif prototype and col==3:r.cam(ax,case,'after','Prototype_logit_hirescam')
   elif prototype and col==4:extras['auxiliary_scores']=scores(ax,case)
   else:r.cam(ax,case,'before' if col==3 else 'after','FDI_logit_hirescam')
   if row==0:ax.set_title(titles[col],pad=6)
  if n>1:fig.text(.004,1-(TOP+row*ROWSTEP-.08)/fig.get_figheight(),f'({chr(97+row)})',weight='bold',fontsize=8.5)
  records={side:r.CAPTURE[group,side][case['case_id']] for side in ['before','after']}
  source.append({'case_id':case['case_id'],'image_id':case['image_id'],'target_FDI':case['target_FDI'],
    'before_FDI':case['before']['prediction']['FDI'],'after_FDI':case['after']['prediction']['FDI'],
    'crop_xyxy':r.case_crop(case),'image_file':case['image_file'],
    'raw_maps':{side:records[side]['raw_maps'] for side in records},**extras})
 shared_footer(fig)
 common=('Teal boxes show ground truth and orange boxes show predictions. Positive elementwise gradient-times-activation '
  '(HiResCAM-form) maps are computed on the full input and normalized by their own full-image maxima before identical '
  'cropping. Dashed ground-truth boxes are display references, not attribution masks. Colors represent relative spatial '
  'attribution and cannot compare contribution magnitudes between models. Predictions satisfy score >= 0.30 and '
  'class-agnostic one-to-one matching at IoU >= 0.50. These selected examples accompany the controlled validation results. ')
 if prototype:
  caption=('Prototype auxiliary identity supervision on a Mobile example (image 44, reference FDI 51). Independently '
   'trained seed-0 models without and with prototype loss predict FDI 11 and 51, respectively. The enabled bank has four '
   'prototypes per class and temperature 0.50; prototype inference-score fusion is disabled. Prototype CAM explains the '
   'auxiliary class score, not an additional inference classifier. The final panel shows the five highest supported-class '
   'LogSumExp scores; these unnormalized scores are not probabilities. '+common)
 elif group=='HOOA':
  caption=('HOOA branch comparison on a Mobile example (image 77, reference FDI 13). The reduced model removes the '
   'high-pass contribution and replaces the multiplicative second-order response with the context response; the remaining '
   'HOOA operations stay active. Both models are independently trained with seed 0 and prototype temperature 0.20. Full '
   'HOOA changes the matched prediction from FDI 53 to 13. CAMs explain the reference-FDI detection logit at the finest '
   'encoder output, with an independently matched native query in each model. '+common)
 else:
  caption=('CMFA comparison on two DSLR examples under the controlled AdamW setting, seed 0. The reference teeth are '
   '(a) FDI 16 in image 125 and (b) FDI 41 in image 124. Enabling CMFA changes the matched predictions from FDI 55 to 16 '
   'and from FDI 81 to 41. CAMs explain the reference-FDI logit at the finest encoder output using independently matched '
   'native queries in the separately trained models. '+common)
 export(fig,stem,caption,{'group':group,'cases':source,'full_input_normalization':'positive map / its own full-image maximum'})

def final_mobile():
 data=json.loads((GALLERY/'final_mobile_capture.json').read_text())
 ids=['FinalMobile_image44_FDI51','FinalMobile_image115_FDI64']
 records=[next(rec for rec in data['records'] if rec['case_id']==cid) for cid in ids]
 r.CAPTURE['FinalMobile','after']={rec['case_id']:rec for rec in records}
 fig=plt.figure(figsize=(WIDTH,TOP+PH+ROWSTEP+BOTTOM))
 titles=['Prediction','Detection CAM','CMFA CAM','HOOA edge CAM','Prototype CAM']
 keys=['Detection','CMFA','HOOA_edge','Prototype']
 for row,rec in enumerate(records):
  case=rec['case']
  for col in range(5):
   ax=panel_axes(fig,row,col,2)
   if col==0:r.prediction(ax,case,'after')
   else:r.cam(ax,case,'after',keys[col-1]+'_hirescam')
   if row==0:ax.set_title(titles[col],pad=6)
  fig.text(.004,1-(TOP+row*ROWSTEP-.08)/fig.get_figheight(),f'({chr(97+row)})',weight='bold',fontsize=8.5)
 shared_footer(fig)
 caption=('Module-specific attribution in the final Mobile model (EMA seed 2, four prototypes per class, temperature '
  '0.50). Rows show (a) the partly visible reference FDI 51 in image 44 and (b) reference FDI 64 in the blurred image 115. '
  'All columns within a row use the same image, checkpoint and native query. Detection, CMFA and HOOA edge CAMs explain '
  'the same reference-FDI detection logit at the finest encoder output, finer top-down fusion output and high-pass branch '
  'tensor, respectively. Prototype CAM explains the auxiliary prototype class score; prototype scores are not fused at '
  'inference. Each positive elementwise gradient-times-activation map is divided by its own full-image maximum before '
  'cropping. Orange boxes are predictions and dashed teal boxes are ground truth. The maps show spatial attribution, '
  'not comparable contribution magnitudes across modules, and complement rather than replace retrained ablations.')
 export(fig,'New_FinalMobile',caption,{'checkpoint':data['checkpoint'],'cases':[{'case_id':rec['case_id'],
  'query':rec['query'],'target_FDI':rec['case']['target_FDI'],'crop_xyxy':r.case_crop(rec['case']),
  'image_file':rec['case']['image_file'],'raw_maps':rec['raw_maps']} for rec in records]})

def factor_sensitivity():
 prov=json.loads((EVIDENCE/'factor_occlusion_provenance.json').read_text())
 evidence=json.loads((EVIDENCE/'controlled_evidence.json').read_text())
 ann=json.loads(Path(evidence['data']['Mobile'][0]).read_text())
 records=prov['records'];maps={r['reference']['image_id']:np.load(r['maps']) for r in records}
 names=['quadrant','position','dentition']
 limits={name:max(float(np.max(np.abs(maps[r['reference']['image_id']][name+'_dense_margin_drop']))) for r in records) for name in names}
 # Wider 4-column panels are matched within this figure; a common 4:3 ratio,
 # typography and box colors are retained across the full five-figure set.
 pw=1.48;ph=pw*.75;gap=.16;left=(WIDTH-4*pw-3*gap)/2;top=.36;step=ph+.36;height=top+ph+2*step+.53
 fig=plt.figure(figsize=(WIDTH,height));metadata=[]
 for row,entry in enumerate(records):
  rec=entry['reference'];iid=rec['image_id'];im=Image.open(Path(evidence['data']['Mobile'][1])/rec['image_record']['file_name']).convert('RGB')
  boxes=np.asarray([g['bbox'] for g in ann['annotations'] if g['image_id']==iid]);lo=boxes[:,:2].min(0);hi=(boxes[:,:2]+boxes[:,2:]).max(0);cx,cy=(lo+hi)/2
  cw=max(1.12*(hi[0]-lo[0]),1.12*(hi[1]-lo[1])*4/3);ch=.75*cw
  assert cw<=im.width and ch<=im.height
  x0=min(max(cx-cw/2,0),im.width-cw);y0=min(max(cy-ch/2,0),im.height-ch)
  crop=[x0,y0,x0+cw,y0+ch];bottom=height-top-ph-row*step
  for col,name in enumerate(['original']+names):
   ax=fig.add_axes([(left+col*(pw+gap))/WIDTH,bottom/height,pw/WIDTH,ph/height])
   if col==0:ax.imshow(im,interpolation='antialiased');title=f'Reference FDI {rec["FDI"]}'
   else:
    a=maps[iid][name+'_dense_margin_drop']
    ax.imshow(a,cmap='RdBu_r',vmin=-limits[name],vmax=limits[name],extent=[0,im.width,im.height,0],interpolation='nearest')
    k=rec['heads'][name]['predicted_index'];label='Q'+str(k+1) if name=='quadrant' else str(k+1) if name=='position' else 'Primary' if k==0 else 'Permanent'
    title=f'{name.capitalize()}: {label}'
   x,y,w,h=rec['target_annotation']['bbox']
   ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=TEAL if col==0 else '#444444',lw=.85,linestyle='solid' if col==0 else (0,(3,2))))
   ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)
   ax.set_title(title,pad=5,fontsize=8.5)
  fig.text(.004,(bottom+ph+.05)/height,f'({chr(97+row)})',fontsize=8.5,weight='bold')
  metadata.append({'image_id':iid,'FDI':rec['FDI'],'query':entry['query_index'],'crop_xyxy':crop,'raw_maps':entry['maps']})
 for j,name in enumerate(names):
  cax=fig.add_axes([(left+(j+1)*(pw+gap)+.05)/WIDTH,.19/height,(pw-.1)/WIDTH,.065/height])
  v=limits[name];cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-v,v),cmap='RdBu_r'),cax=cax,orientation='horizontal',ticks=[-v,0,v])
  cb.ax.set_xticklabels([f'{-v:.1f}','0',f'{v:.1f}']);cb.ax.tick_params(length=1.5,pad=1.5,labelsize=7.5);cb.outline.set_visible(False)
  cb.ax.set_title('Logit-margin drop',fontsize=8,pad=4)
 caption=('Conditional judgment sensitivity of the quadrant, position and dentition heads to input occlusion. Rows show Mobile reference '
  'FDI 64, 11 and 46 using the final seed-2 checkpoint (prototype temperature 0.50). Each 1024 by 1024 input is perturbed '
  'at 841 locations using a 128 by 128 mean-RGB patch with stride 32. Sensitivity is the unoccluded minus occluded '
  'difference in the correct-factor logit margin over its strongest unoccluded alternative; the competitor is fixed. '
  'Red indicates support for the correct decision and blue indicates a region whose occlusion increases that margin. '
  'Dense maps average the signed margin changes across overlapping patches. Each head shares one symmetric raw '
  'logit-margin color scale across all three cases. Encoder top-k anchor indices and query order are fixed to the '
  'unoccluded pass; content and downstream predictions are recomputed for every perturbation. Dashed boxes mark the '
  'target tooth without masking the sensitivity calculation. These are conditional judgment-sensitivity maps, not native '
  'attention maps. Mean-color occlusion may introduce artificial edges.')
 export(fig,'New_FactorSensitivity',caption,{'provenance':str(EVIDENCE/'factor_occlusion_provenance.json'),
  'cases':metadata,'common_symmetric_limits_by_head':limits,'patches_per_image':prov['patches_per_image'],
  'occluder_size':prov['occluder_size'],'stride':prov['stride'],'full_provenance_checkpoint':prov['checkpoint']},lossless=True)

comparison('Prototype',['Prototype_correction_01_image44_FDI51'],'New_Prototype',True)
comparison('HOOA',['HOOA_correction_03_image77_FDI13'],'New_HOOA')
comparison('CMFA',['CMFA_correction_01_image125_FDI16','CMFA_correction_05_image124_FDI41'],'New_CMFA')
final_mobile();factor_sensitivity()
for stem,meta in MANIFEST.items():
 for case in meta.get('cases',[]):
  paths=case.get('raw_maps',{})
  if isinstance(paths,str):paths={'map':paths}
  case['raw_map_sha256']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()}
(WORK/'publication_figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
(WORK/'publication_figure_captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')
(WORK/'publication_figure_captions_en.txt').write_text('\n\n'.join(k+'\n'+v for k,v in CAPTIONS.items())+'\n')
print('Total PDFs',sum((OUT/f'{s}.pdf').stat().st_size for s in MANIFEST),'bytes')
