"""Selected same-case diagrams from saved, unretouched model predictions."""
from pathlib import Path
from collections import defaultdict, Counter
import gzip, json, hashlib, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
CONTROL=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')
OUT=ROOT/'figures/qualitative'
OUT.mkdir(parents=True,exist_ok=True)
SCORE=.30
IOU=.50
DATA={
 'Mobile': {'annotation':Path('/home/wuxi/coco/valid/annotations_fdi_types.json'),'images':Path('/home/wuxi/coco/valid'), 'base':'mobile_1024_base','full':'mobile_1024_full'},
 'Pano': {'annotation':Path('/home/wuxi/DENTEX2023/annotations/instances_val2017.json'),'images':Path('/home/wuxi/DENTEX2023/val2017'),'base':'pano_base','full':'pano_factor_prototype'},
 'DSLR': {'annotation':Path('/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json'),'images':Path('/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val'),'base':'dslr_base_adamw','full':'dslr_cmfa_dualopt'},
}
PANO_FDI=[18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]
EXTERNAL={
 'Pano':('RT-DETR-L',Path('/home/wuxi/Dent-FINE/outputs/cross_dataset_baseline_comparison/dentex/runs/rtdetr_l/coco_eval/coco_predictions.json')),
 'Mobile':('RT-DETR-L',Path('/home/wuxi/Dent-FINE/outputs/cross_dataset_baseline_comparison/phone/runs/rtdetr_l/coco_eval/coco_predictions.json')),
 'DSLR':('RT-DETR-L',Path('/home/wuxi/Dent-FINE/outputs/cross_dataset_baseline_comparison/dslr/runs/rtdetr_l/coco_eval/coco_predictions.json')),
}
GT_COLOR='#006D77';PRED_COLOR='#B75A2A';ROI_COLOR='#222222'

def load_json(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())

def dataset(cohort):
    d=DATA[cohort]
    raw=load_json(d['annotation'])
    imgs={x['id']:x for x in raw['images']}
    gt=defaultdict(list)
    for a in raw['annotations']:gt[a['image_id']].append(a)
    fdi={x['id']:(PANO_FDI[x['id']] if cohort=='Pano' else int(x.get('fdi',x['name']))) for x in raw['categories']}
    return imgs,gt,fdi

def predictions(configuration,folder='evaluation'):
    directory=CONTROL/'runs'/configuration/'seed_0'/folder
    path=directory/'predictions_coco.json.gz'
    rows=load_json(path)
    by=defaultdict(list)
    for p in rows:
        if p['score']>=SCORE:by[p['image_id']].append(p)
    metric=load_json(directory/'metrics.json')
    return by,{'prediction_file':str(path),'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_file':str(directory.parent/'best.pth'),'checkpoint_sha256':metric['checkpoint_sha256'],'configuration':configuration,'seed':0,'prior':metric.get('prior','off')}

def box_iou(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    inter=np.clip(hi-lo,0,None).prod(-1);union=a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-inter
    return inter/np.maximum(union,1e-12)

def match(pp,gg):
    status={g['id']:{'status':'missed','prediction':None,'IoU':None} for g in gg}
    ov=box_iou([p['bbox'] for p in pp],[g['bbox'] for g in gg])
    if len(pp) and len(gg):
        rr,cc=linear_sum_assignment(-ov)
        for r,c in zip(rr,cc):
            if ov[r,c]>=IOU:status[gg[c]['id']]={'status':'correct' if pp[r]['category_id']==gg[c]['category_id'] else 'wrong','prediction':pp[r],'IoU':float(ov[r,c])}
    tp=sum(s['status']=='correct' for s in status.values())
    dup=sum(max(v-1,0) for v in Counter(p['category_id'] for p in pp).values())
    counts={'GT':len(gg),'predictions':len(pp),'TP':tp,'FP':len(pp)-tp,'FN':len(gg)-tp,'duplicate_count':dup}
    return status,counts

def selection(cohort,basepred,fullpred,require_same_box=False):
    imgs,gt,fdi=dataset(cohort)
    candidates={'improvement':[],'failure':[]}
    for iid in imgs:
        gg=gt[iid];bs,bc=match(basepred[iid],gg);fs,fc=match(fullpred[iid],gg)
        if cohort=='Mobile':
            boxes=np.array([g['bbox'] for g in gg]);lo=boxes[:,:2].min(0);hi=(boxes[:,:2]+boxes[:,2:]).max(0)
            oral_fraction=float(np.prod(hi-lo)/(imgs[iid]['width']*imgs[iid]['height']))
            if oral_fraction<.20:continue
        for g in gg:
            gb,gf=bs[g['id']],fs[g['id']]
            if require_same_box:
                if gb['prediction'] is None or gf['prediction'] is None:continue
                if box_iou([gb['prediction']['bbox']],[gf['prediction']['bbox']])[0,0]<.99999:continue
            typ='improvement' if gf['status']=='correct' and gb['status']!='correct' else ('failure' if gb['status']=='correct' and gf['status']!='correct' else None)
            if not typ:continue
            net=fc['TP']-bc['TP']
            rank=(net if typ=='improvement' else -net, (bc['duplicate_count']-fc['duplicate_count']) if typ=='improvement' else (fc['duplicate_count']-bc['duplicate_count']), int(gb['status']=='wrong' if typ=='improvement' else gf['status']=='wrong'), g['bbox'][2]*g['bbox'][3])
            candidates[typ].append((rank,{'cohort':cohort,'image_id':iid,'image_file':str(DATA[cohort]['images']/imgs[iid]['file_name']),'image_record':imgs[iid],'type':typ,'target_annotation':g,'target_FDI':fdi[g['category_id']],'base_status':gb,'full_status':gf,'base_counts':bc,'full_counts':fc}))
    picked=[]
    for typ in ['improvement','failure']:
        if not candidates[typ]:continue
        picked.append(sorted(candidates[typ],key=lambda v:v[0],reverse=True)[0][1])
    return picked

def prior_query_selection(off,both):
    """Track the highest-scoring class of identical query boxes before/after prior."""
    imgs,gt,fdi=dataset('Pano')
    ranked={'improvement':[],'failure':[]}
    def top_by_box(rows):
        result={}
        for p in rows:
            key=tuple(round(v,3) for v in p['bbox'])
            if key not in result or p['score']>result[key]['score']:result[key]=p
        return result
    for iid in imgs:
        top0=top_by_box(off[iid]);top1=top_by_box(both[iid]);gg=gt[iid]
        for key in set(top0)&set(top1):
            a,b=top0[key],top1[key]
            if a['category_id']==b['category_id']:continue
            ov=box_iou([a['bbox']],[g['bbox'] for g in gg])[0]
            j=int(np.argmax(ov))
            if ov[j]<IOU:continue
            g=gg[j];ca=a['category_id']==g['category_id'];cb=b['category_id']==g['category_id']
            typ='improvement' if cb and not ca else ('failure' if ca and not cb else None)
            if not typ:continue
            _,ac=match(off[iid],gg);_,bc=match(both[iid],gg)
            item={'cohort':'Pano','image_id':iid,'image_file':str(DATA['Pano']['images']/imgs[iid]['file_name']),'image_record':imgs[iid],'type':typ,'target_annotation':g,'target_FDI':fdi[g['category_id']],'base_status':{'status':'correct' if ca else 'wrong','prediction':a,'IoU':float(ov[j])},'full_status':{'status':'correct' if cb else 'wrong','prediction':b,'IoU':float(ov[j])},'base_counts':ac,'full_counts':bc,'selection_scope':'highest-scoring retained class for the same unmodified query box, separately from Hungarian endpoint matching'}
            rank=(float(ov[j]),a['score']+b['score'],g['bbox'][2]*g['bbox'][3])
            ranked[typ].append((rank,item))
    return [sorted(ranked[t],key=lambda v:v[0],reverse=True)[0][1] for t in ['improvement','failure'] if ranked[t]]

def prior_suppression_selection(off,both):
    imgs,gt,fdi=dataset('Pano');ranked=[]
    for iid in imgs:
        ast,ac=match(off[iid],gt[iid]);bst,bc=match(both[iid],gt[iid])
        if bc['TP']!=ac['TP'] or bc['FP']>=ac['FP']:continue
        kept={(p['category_id'],tuple(round(v,3) for v in p['bbox'])) for p in both[iid]}
        for p in off[iid]:
            key=(p['category_id'],tuple(round(v,3) for v in p['bbox']))
            if key in kept:continue
            ov=box_iou([p['bbox']],[g['bbox'] for g in gt[iid]])[0];j=int(np.argmax(ov));g=gt[iid][j]
            if ov[j]<IOU or p['category_id']==g['category_id'] or bst[g['id']]['status']!='correct':continue
            item={'cohort':'Pano','image_id':iid,'image_file':str(DATA['Pano']['images']/imgs[iid]['file_name']),'image_record':imgs[iid],'type':'suppression','target_annotation':g,'target_FDI':fdi[g['category_id']],'base_status':ast[g['id']],'full_status':bst[g['id']],'base_counts':ac,'full_counts':bc,'suppressed_prediction':p,'suppressed_FDI':fdi[p['category_id']],'selection_scope':'incorrect query-class candidate falls below the fixed display threshold; correct localized target retained; not a same-query class correction'}
            rank=(ac['FP']-bc['FP'],ac['duplicate_count']-bc['duplicate_count'],float(ov[j]),p['score'])
            ranked.append((rank,item))
    return sorted(ranked,key=lambda v:v[0],reverse=True)[0][1] if ranked else None

def select_all():
    result={'protocol':{'score_threshold':SCORE,'matching_IoU_threshold':IOU,'assignment':'class-agnostic Hungarian maximizing total IoU; identity checked after localization','selection':'illustrative examples selected by paired image-level TP improvement/decline, then duplicate-count change, then wrong-FDI change, then target area; Mobile requires the union of tooth annotation boxes to span at least 20% of image area for an adequate oral field; no frequency or causal claim','seed':0},'cohort_comparison':{},'mobile_inter_module':{},'pano_prior':{}}
    for cohort,d in DATA.items():
        bp,bmeta=predictions(d['base']);fp,fmeta=predictions(d['full'],'replay_prior_both' if cohort=='Pano' else 'evaluation')
        cases=selection(cohort,bp,fp)
        result['cohort_comparison'][cohort]={'cases':cases,'sources':{'base':bmeta,'full':fmeta},'annotation':str(d['annotation']),'annotation_sha256':hashlib.sha256(d['annotation'].read_bytes()).hexdigest()}
    result['mobile_inter_module']={'cases':result['cohort_comparison']['Mobile']['cases'],'sources':{}}
    for label,config in [('Base','mobile_1024_base'),('Features','mobile_1024_features'),('Identity','mobile_1024_identity'),('Full','mobile_1024_full')]:
        _,meta=predictions(config);result['mobile_inter_module']['sources'][label]=meta
    off,ometa=predictions('pano_factor_prototype');both,bmeta=predictions('pano_factor_prototype','replay_prior_both')
    cases=prior_query_selection(off,both)
    if not any(c['type']=='improvement' for c in cases):
        suppression=prior_suppression_selection(off,both)
        if suppression:cases.insert(0,suppression)
    result['pano_prior']={'cases':cases,'sources':{}}
    for label,folder in [('Off','evaluation'),('Center','replay_prior_center'),('Shape','replay_prior_shape'),('Combined','replay_prior_both')]:
        _,meta=predictions('pano_factor_prototype',folder);result['pano_prior']['sources'][label]=meta
    (OUT/'source_case_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for typ,v in result['cohort_comparison'].items():
        print(typ,[(c['image_id'],c['type'],c['target_FDI'],c['base_status']['status'],c['full_status']['status'],c['base_counts']['TP'],c['full_counts']['TP']) for c in v['cases']])
    print('Pano prior',[(c['image_id'],c['type'],c['target_FDI']) for c in cases])
    return result

def roi(case):
    x,y,w,h=case['target_annotation']['bbox'];W=case['image_record']['width'];H=case['image_record']['height']
    rw=max(3.8*w,2.8*h*4/3);rh=rw*3/4
    rw=min(rw,W);rh=min(rh,H)
    x0=min(max(x+w/2-rw/2,0),W-rw);y0=min(max(y+h/2-rh/2,0),H-rh)
    return [float(x0),float(y0),float(x0+rw),float(y0+rh)]

def nearby(bbox,r):
    x,y,w,h=bbox
    return r[0]<=x+w/2<=r[2] and r[1]<=y+h/2<=r[3]

def setup_fonts():
    directory=ROOT/'references/times_new_roman'
    for p in directory.glob('*.TTF'):fm.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.4,'savefig.facecolor':'white'})

def panel(ax,image,rows,fdi,case,crop=False,ground_truth=False):
    W,H=image.size;r=case['crop_xyxy']
    ax.imshow(image,interpolation='antialiased',zorder=0)
    col=GT_COLOR if ground_truth else PRED_COLOR
    zoom=[]
    matched_target=None
    if not ground_truth and rows:
        _,gg,_=dataset(case['cohort'])
        matched_target=match(rows,gg[case['image_id']])[0][case['target_annotation']['id']]['prediction']
    for row in rows:
        x,y,w,h=row['bbox'];in_roi=nearby(row['bbox'],r)
        if crop and not in_roi:continue
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=col,linewidth=.65 if crop else .40,alpha=.95 if in_roi else .6,zorder=2))
        if crop and in_roi:
            is_suppressed=case.get('suppressed_prediction')==row
            if (ground_truth and row['id']==case['target_annotation']['id']) or (not ground_truth and (row==matched_target or is_suppressed)):zoom.append((row,x,y,w,h))
    # Assign collision-free label positions inside boxes; a repeated query box may carry multiple labels.
    label_groups=defaultdict(list)
    for row,x,y,w,h in zoom:label_groups[(round(x,1),round(y,1),round(w,1),round(h,1))].append((row,x,y,w,h))
    for group in label_groups.values():
        group.sort(key=lambda z:z[0].get('score',1),reverse=True)
        for offset,(row,x,y,w,h) in enumerate(group):
            px=min(max(x+0.04*w,r[0]+.025*(r[2]-r[0])),r[2]-.06*(r[2]-r[0]))
            py=min(max(y+.03*h+offset*.085*(r[3]-r[1]),r[1]+.035*(r[3]-r[1])),r[3]-.075*(r[3]-r[1]))
            ax.text(px,py,str(fdi[row['category_id']]),fontsize=8.0,color='white',ha='left',va='top',clip_on=True,bbox={'facecolor':col,'edgecolor':'none','boxstyle':'square,pad=0.08','alpha':.92},zorder=3)
    if crop:
        ax.set_xlim(r[0],r[2]);ax.set_ylim(r[3],r[1])
    else:
        ax.add_patch(Rectangle((r[0],r[1]),r[2]-r[0],r[3]-r[1],fill=False,edgecolor=ROI_COLOR,linewidth=.75,linestyle=(0,(4,2)),zorder=4))
        ax.set_xlim(0,W);ax.set_ylim(H,0)
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_aspect('equal')

def caption_case(case,fdi):
    target=case['target_FDI'];bs=case['base_status'];fs=case['full_status']
    def label(st):return 'not localized' if st['prediction'] is None else str(fdi[st['prediction']['category_id']])
    return f"Image {case['image_id']}: the highlighted reference tooth is FDI {target}; the controlled baseline {bs['status']} assignment ({label(bs)}) and full-model {fs['status']} assignment ({label(fs)}) accompany image-level correct-match counts of {case['base_counts']['TP']} and {case['full_counts']['TP']}."

def export(fig,stem):
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600,bbox_inches=None,pad_inches=.02,metadata={'Title':stem} if ext=='pdf' else None)
    # Low-resolution inspection copy is a companion, not a submission figure.
    fig.savefig(OUT/f'{stem}_preview.png',dpi=160)
    plt.close(fig)

def comparison_grid(cases,cohort,methods,stem):
    imgs,gt,fdi=dataset(cohort);n=len(methods)+2
    coltitles=['Original','Ground truth']+[label for label,_ in methods]
    width=7.25;cellw=(width-.34)/n
    whole_ratio=np.mean([c['image_record']['height']/c['image_record']['width'] for c in cases])
    wh=cellw*whole_ratio;zh=cellw*.75
    group_h=wh+zh+.24
    height=len(cases)*group_h+.32
    fig=plt.figure(figsize=(width,height),facecolor='white')
    gs=fig.add_gridspec(len(cases)*2,n,left=.036,right=.995,bottom=.012,top=.925,wspace=.07,hspace=.09,height_ratios=[whole_ratio,.75]*len(cases))
    for i,case in enumerate(cases):
        case['crop_xyxy']=roi(case);case['image_sha256']=hashlib.sha256(Path(case['image_file']).read_bytes()).hexdigest();im=Image.open(case['image_file']).convert('RGB')
        for j in range(n):
            top=fig.add_subplot(gs[2*i,j]);bottom=fig.add_subplot(gs[2*i+1,j])
            rows=[] if j==0 else (gt[case['image_id']] if j==1 else methods[j-2][1][case['image_id']])
            panel(top,im,rows,fdi,case,crop=False,ground_truth=j==1)
            panel(bottom,im,rows,fdi,case,crop=True,ground_truth=j==1)
            if i==0:top.set_title(coltitles[j],fontsize=8.7,pad=5,fontweight='bold')
            if j==0:top.text(-.09,1.0,chr(ord('A')+i),transform=top.transAxes,fontsize=11,fontweight='bold',va='top',ha='right',clip_on=False)
    export(fig,stem)

def audit_prior_case_queries(result):
    path=CONTROL/'runs/pano_factor_prototype/seed_0/evaluation/pre_topk_candidates.npz'
    cached=np.load(path)
    for c in result['pano_prior']['cases']:
        at=int(np.flatnonzero(cached['image_ids']==c['image_id'])[0]);W,H=cached['original_sizes'][at]
        raw=cached['boxes_cxcywh'][at].astype(float);boxes=raw*np.array([W,H,W,H]);boxes[:,:2]-=boxes[:,2:]/2
        records={'off':c['base_status']['prediction'],'combined':c['full_status']['prediction']}
        if c.get('suppressed_prediction'):records['suppressed']=c['suppressed_prediction']
        c['raw_query_verification']={}
        for label,p in records.items():
            err=np.abs(boxes-np.asarray(p['bbox'])).max(1);q=int(np.argmin(err));assert err[q]<.02,(c['image_id'],label,float(err[q]))
            c['raw_query_verification'][label]={'query_index':q,'maximum_box_coordinate_difference_pixels':float(err[q]),'pre_topk_source':str(path)}
    assert len({v['checkpoint_sha256'] for v in result['pano_prior']['sources'].values()})==1

def render_all(result,cohort_only=None):
    setup_fonts();captions=load_json(OUT/'figure_captions.json') if cohort_only else {}
    external_audit=load_json(ROOT/'evidence/external_prediction_sources.json')['entries']
    for cohort in DATA:
        if cohort_only and cohort!=cohort_only:continue
        bp,_=predictions(DATA[cohort]['base']);fp,_=predictions(DATA[cohort]['full'],'replay_prior_both' if cohort=='Pano' else 'evaluation')
        extlabel,extpath=EXTERNAL[cohort];ep=defaultdict(list)
        for p in load_json(extpath):
            if p['score']>=SCORE:ep[p['image_id']].append(p)
        spec=result['cohort_comparison'][cohort]
        authority=next(r for r in external_audit if r['cohort']==cohort and r['model']==extlabel)
        assert authority['COCO_bbox_content_identical'] and authority['class_map_matches_training_export']
        spec['sources']['external']={'method':extlabel,'prediction_file':str(extpath),'prediction_sha256':hashlib.sha256(extpath.read_bytes()).hexdigest(),'checkpoint_file':authority['weight_path'],'checkpoint_sha256':authority['weight_sha256'],'evaluation_annotation_match':'original image IDs, FDI categories, and validation box content confirmed by controlled-results audit','native_postprocessing':authority['native_postprocessing'],'pretraining_path':authority['pretraining_path']}
        stem=f'{cohort}_same_image_comparison'
        comparison_grid(spec['cases'],cohort,[('Controlled base',bp),(extlabel,ep),('DENT-STRIDE',fp)],stem)
        _,_,fdi=dataset(cohort)
        captions[stem]=f"Same-image {cohort.lower()} validation comparison. Columns show the original image, reference annotations, controlled baseline, {extlabel}, and DENT-STRIDE; each enlarged row uses an identical target region across columns. Teal boxes denote reference annotations and orange boxes denote model predictions. Saved seed-0 model outputs use score ≥0.30 and clinical FDI decoding; FDI text labels identify the highlighted target’s localization match, while other retained predictions remain visible as boxes. "+' '.join(f"({chr(65+i)}) "+caption_case(c,fdi) for i,c in enumerate(spec['cases']))+' Examples include an improvement and a deterioration and do not estimate their population frequencies. The external detector retains its framework-specific training and post-processing.'
    if cohort_only:
        (OUT/'source_case_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        (OUT/'figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
        return
    sources=[]
    for label,config in [('Base','mobile_1024_base'),('Features','mobile_1024_features'),('Identity','mobile_1024_identity'),('Full','mobile_1024_full')]:sources.append((label,predictions(config)[0]))
    comparison_grid(result['mobile_inter_module']['cases'],'Mobile',sources,'Mobile_inter_module_comparison')
    captions['Mobile_inter_module_comparison']='Mobile inter-module comparison using independently trained controlled seed-0 models at the same resolution and operating threshold. Columns show the original image, reference annotations, baseline, features only, identity only, and complete configuration; each case pairs the whole image with an identical local enlargement. Features comprise CMFA and HOOA; identity comprises factorized supervision/fusion and prototype auxiliary supervision. FDI text labels identify the highlighted target’s localization match, while other retained predictions remain visible as boxes. Panel A uses image 111 and panel B image 56. Complete validation results, rather than these selected examples, establish each configuration’s measured performance.'
    prior_methods=[('No prior',predictions('pano_factor_prototype')[0])]+[(label,predictions('pano_factor_prototype',folder)[0]) for label,folder in [('Center','replay_prior_center'),('Shape','replay_prior_shape'),('Center + shape','replay_prior_both')]]
    comparison_grid(result['pano_prior']['cases'],'Pano',prior_methods,'Pano_same_checkpoint_prior_comparison')
    captions['Pano_same_checkpoint_prior_comparison']='Same-checkpoint DENTEX anatomical-prior replay. Each column starts from the same 300 pre-top-k query boxes and logits; only the training-derived center/shape score terms change before native top-k selection. Rows compare reference annotations with no prior, center only, shape only, and combined scoring using score ≥0.30; all local enlargements share the same region. (A) Image 66: combined scoring suppresses incorrect/repeated candidates, reducing false-positive predictions from 9 to 6 while retaining 27 correct matches; the highlighted FDI 12 tooth is retained and an additional FDI 13 candidate falls below the display threshold. This is score-based candidate suppression, not a corrected highest-class identity. (B) Image 214: the same highlighted query box changes its highest retained label from the correct FDI 27 to FDI 28; image-level correct matches decrease from 15 to 14. Bounding boxes are unchanged. These selected examples illustrate both favourable and adverse scoring effects.'
    montage(result,'Acquisition_cohort_montage')
    captions['Acquisition_cohort_montage']='Representative images and reference FDI annotations from the three validation cohorts: (A) panoramic radiographs, (B) smartphone intraoral photographs, and (C) five-view DSLR photographs. Two images per cohort illustrate differences in field of view, framing, illumination, and appearance. Teal boxes mark reference tooth annotations. Visible gaps or occlusions are not interpreted as diagnoses of missing teeth.'
    audit_prior_case_queries(result)
    (OUT/'source_case_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/'figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))

def montage(result,stem):
    width=7.25;height=5.50;fig=plt.figure(figsize=(width,height),facecolor='white')
    gs=fig.add_gridspec(3,4,left=.041,right=.995,bottom=.017,top=.949,wspace=.06,hspace=.20)
    for i,cohort in enumerate(['Pano','Mobile','DSLR']):
        _,gt,fdi=dataset(cohort)
        for k,case in enumerate(result['cohort_comparison'][cohort]['cases']):
            im=Image.open(case['image_file']).convert('RGB')
            for j in range(2):
                ax=fig.add_subplot(gs[i,2*k+j]);ax.imshow(im)
                if j:
                    for g in gt[case['image_id']]:
                        x,y,w,h=g['bbox'];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=GT_COLOR,linewidth=.6))
                ax.axis('off')
                if i==0:ax.set_title('Original' if not j else 'Ground truth',fontsize=9.5,fontweight='bold',pad=5)
        fig.text(.046, .945-i*.318,chr(65+i)+'  '+{'Pano':'Panoramic','Mobile':'Mobile','DSLR':'DSLR'}[cohort],fontsize=10,fontweight='bold',ha='left',va='bottom')
    export(fig,stem)

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--cohort':render_all(load_json(OUT/'source_case_selection.json'),sys.argv[2])
    else:render_all(select_all())
