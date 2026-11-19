"""Submission panels from observed seed-0 detections; clinical pixels are unchanged."""
from pathlib import Path
from collections import defaultdict, Counter
from functools import lru_cache
import gzip
import hashlib
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
RUNS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_best3_seeds_2026-10-08/runs')
OFFICIAL = Path('/home/wuxi/Dent-FINE-paper/materials/official_baselines_2026-09-10/runs/deim_s')
SCORE, IOU = .30, .50
FDI_PANO = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]
DATA = {
    'Mobile': {'ann': '/home/wuxi/coco/valid/annotations_fdi_types.json', 'images': '/home/wuxi/coco/valid', 'official': 'phone', 'final': 'mobile_1024_full'},
    'Pano': {'ann': '/home/wuxi/DENTEX2023/annotations/instances_val2017.json', 'images': '/home/wuxi/DENTEX2023/val2017', 'official': 'dentex', 'final': 'pano_historical_spatialshape'},
    'DSLR': {'ann': '/home/wuxi/Dent-FINE/datasets/viewnet3_dslr_fdi52/annotations/tooth_instances_val_fdi52.json', 'images': '/home/wuxi/View-Net3_fixed/datasets/dental_dslr_view_template_dataset_copy/images/val', 'official': 'dslr', 'final': 'dslr_cmfa_dualopt'},
}
GT_COLOR, PRED_COLOR = '#087B80', '#AD5528'

def read(path):
    path = Path(path)
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as f: return json.load(f)
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@lru_cache(None)
def dataset(cohort):
    cfg=DATA[cohort]; raw=read(cfg['ann'])
    imgs={int(r['id']):r for r in raw['images']}; gt=defaultdict(list)
    for row in raw['annotations']: gt[int(row['image_id'])].append(row)
    fdi={int(r['id']):FDI_PANO[int(r['id'])] if cohort=='Pano' else int(r.get('fdi',r['name'])) for r in raw['categories']}
    return imgs,gt,fdi

def predictions(path,cohort,meta=None):
    path=Path(path); rows=read(path); by=defaultdict(list)
    imgs,_,fdi=dataset(cohort)
    for row in rows:
        assert int(row['image_id']) in imgs, (path,row['image_id'])
        assert int(row['category_id']) in fdi, (path,row['category_id'])
        assert np.isfinite(row['score']) and np.isfinite(row['bbox']).all()
        if row['score']>=SCORE:by[int(row['image_id'])].append(row)
    provenance={'prediction_file':str(path),'prediction_sha256':sha(path),'score_threshold':SCORE}
    if meta is not None:provenance.update(meta)
    return by,provenance

def run_predictions(config,cohort,folder='evaluation',seed=0):
    directory=RUNS/config/f'seed_{seed}'/folder
    metric=read(directory/'metrics.json')
    return predictions(directory/'predictions_coco.json.gz',cohort,{
        'checkpoint_file':str(directory.parent/'best.pth'),
        'checkpoint_sha256':metric['checkpoint_sha256'],'configuration':config,
        'seed':seed,'folder':folder,'prior':metric.get('prior','off')})

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    return intersection/np.maximum(a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-intersection,1e-12)

def match(pp,gg):
    status={g['id']:{'status':'missed','prediction':None,'IoU':None} for g in gg}
    ov=iou([p['bbox'] for p in pp],[g['bbox'] for g in gg])
    if len(pp) and len(gg):
        rr,cc=linear_sum_assignment(-ov)
        for r,c in zip(rr,cc):
            if ov[r,c]>=IOU:
                status[gg[c]['id']]={'status':'correct' if pp[r]['category_id']==gg[c]['category_id'] else 'wrong','prediction':pp[r],'IoU':float(ov[r,c])}
    tp=sum(s['status']=='correct' for s in status.values())
    return status,{'GT':len(gg),'predictions':len(pp),'TP':tp,'FP':len(pp)-tp,'FN':len(gg)-tp,'duplicate_count':sum(max(n-1,0) for n in Counter(p['category_id'] for p in pp).values())}

def case_record(cohort,iid,g,typ,methods):
    imgs,gt,fdi=dataset(cohort)
    imfile=Path(DATA[cohort]['images'])/imgs[iid]['file_name']
    assert imfile.exists(), imfile
    im=Image.open(imfile)
    assert im.size==(imgs[iid]['width'],imgs[iid]['height']), (imfile,im.size)
    statuses={};counts={}
    for name,pp in methods:
        st,ct=match(pp[iid],gt[iid]);statuses[name]=st[g['id']];counts[name]=ct
    return {'cohort':cohort,'image_id':iid,'image_file':str(imfile),'image_sha256':sha(imfile),'image_record':imgs[iid],'type':typ,'target_annotation':g,'target_FDI':fdi[g['category_id']],'statuses':statuses,'counts':counts}

def oral_fraction(gg,img):
    box=np.asarray([g['bbox'] for g in gg]);lo=box[:,:2].min(0);hi=(box[:,:2]+box[:,2:]).max(0)
    return np.prod(hi-lo)/(img['width']*img['height'])

def select_pair(cohort,base,full,methods):
    imgs,gt,_=dataset(cohort); ranked={'improvement':[],'adverse':[]}
    for iid,img in imgs.items():
        if not gt[iid]:continue
        if cohort=='Mobile' and oral_fraction(gt[iid],img)<.20:continue
        bs,bc=match(base[iid],gt[iid]);fs,fc=match(full[iid],gt[iid])
        additional=[]
        for name,pp in methods:
            if name not in ['Official DEIM-s','DENT-STRIDE']:
                st,ct=match(pp[iid],gt[iid]);additional.append((st,ct))
        for g in gt[iid]:
            b,f=bs[g['id']],fs[g['id']]
            typ='improvement' if f['status']=='correct' and b['status']!='correct' else ('adverse' if b['status']=='correct' and f['status']!='correct' else None)
            if typ is None:continue
            sign=1 if typ=='improvement' else -1
            # Favor readable illustrative corrections shared by both external references,
            # while retaining an adverse counterpart and the complete quantitative table.
            shared=sum(st[g['id']]['status']!='correct' for st,_ in additional) if typ=='improvement' else sum(st[g['id']]['status']=='correct' for st,_ in additional)
            deltas=[sign*(fc['TP']-bc['TP'])]+[sign*(fc['TP']-ct['TP']) for _,ct in additional]
            rank=(shared,min(deltas),sign*(fc['TP']-bc['TP']),sign*(bc['duplicate_count']-fc['duplicate_count']),int(b['status']=='wrong' if typ=='improvement' else f['status']=='wrong'),g['bbox'][2]*g['bbox'][3])
            ranked[typ].append((rank,iid,g))
    picked=[]
    for typ in ['improvement','adverse']:
        assert ranked[typ],(cohort,typ)
        _,iid,g=max(ranked[typ],key=lambda x:x[0]);picked.append(case_record(cohort,iid,g,typ,methods))
    return picked

def select_hooa(methods):
    imgs,gt,_=dataset('Mobile');ranked=[]
    for iid,img in imgs.items():
        if not gt[iid] or oral_fraction(gt[iid],img)<.20:continue
        allst={};allct={}
        for name,pp in methods:allst[name],allct[name]=match(pp[iid],gt[iid])
        full=allst['Full HOOA'];fullct=allct['Full HOOA']
        controls=[n for n,_ in methods if n!='Full HOOA']
        for g in gt[iid]:
            if full[g['id']]['status']!='correct':continue
            bad=sum(allst[n][g['id']]['status']!='correct' for n in controls)
            if bad<2:continue
            deltas=[fullct['TP']-allct[n]['TP'] for n in controls]
            rank=(bad,min(deltas),sum(deltas),g['bbox'][2]*g['bbox'][3])
            ranked.append((rank,iid,g))
    picked=[];used=set()
    for _,iid,g in sorted(ranked,key=lambda x:x[0],reverse=True):
        if iid in used:continue
        picked.append(case_record('Mobile',iid,g,'illustrative_branch_benefit',methods));used.add(iid)
        if len(picked)==2:break
    assert len(picked)==2
    return picked

def select_prior(methods):
    pp=dict(methods);off,both=pp['No prior'],pp['Center + shape'];imgs,gt,fdi=dataset('Pano')
    candidates={'improvement':[],'adverse':[]}
    def top(rows):
        out={}
        for p in rows:
            key=tuple(round(v,3) for v in p['bbox'])
            if key not in out or p['score']>out[key]['score']:out[key]=p
        return out
    for iid in imgs:
        a,b=top(off[iid]),top(both[iid])
        _,ac=match(off[iid],gt[iid]);_,bc=match(both[iid],gt[iid])
        for key in a.keys()&b.keys():
            p,q=a[key],b[key]
            if p['category_id']==q['category_id']:continue
            ov=iou([p['bbox']],[g['bbox'] for g in gt[iid]])[0];j=int(ov.argmax())
            if ov[j]<IOU:continue
            g=gt[iid][j];ok0=p['category_id']==g['category_id'];ok1=q['category_id']==g['category_id']
            typ='improvement' if ok1 and not ok0 else ('adverse' if ok0 and not ok1 else None)
            if typ is None:continue
            sign=1 if typ=='improvement' else -1
            rank=(sign*(bc['TP']-ac['TP']),float(ov[j]),g['bbox'][2]*g['bbox'][3])
            candidates[typ].append((rank,iid,g,{'mechanism':'same-query highest retained FDI label change','tracked_no_prior':p,'tracked_combined':q}))
    if not candidates['improvement']:
        for iid in imgs:
            ast,ac=match(off[iid],gt[iid]);bst,bc=match(both[iid],gt[iid])
            if bc['TP']<ac['TP'] or bc['FP']>=ac['FP']:continue
            kept={(p['category_id'],tuple(round(v,3) for v in p['bbox'])) for p in both[iid]}
            for p in off[iid]:
                if (p['category_id'],tuple(round(v,3) for v in p['bbox'])) in kept:continue
                ov=iou([p['bbox']],[g['bbox'] for g in gt[iid]])[0];j=int(ov.argmax());g=gt[iid][j]
                if ov[j]<IOU or p['category_id']==g['category_id'] or bst[g['id']]['status']!='correct':continue
                rank=(ac['FP']-bc['FP'],ac['duplicate_count']-bc['duplicate_count'],float(ov[j]))
                candidates['improvement'].append((rank,iid,g,{'mechanism':'incorrect query-class candidate suppression','suppressed_prediction':p,'suppressed_FDI':fdi[p['category_id']]}))
    picked=[]
    for typ in ['improvement','adverse']:
        assert candidates[typ],('prior',typ)
        _,iid,g,extra=max(candidates[typ],key=lambda x:x[0]);rec=case_record('Pano',iid,g,typ,methods);rec.update(extra);picked.append(rec)
    return picked

def roi(case):
    x,y,w,h=case['target_annotation']['bbox'];W,H=case['image_record']['width'],case['image_record']['height']
    rw=min(max(3.8*w,2.8*h*4/3),W);rh=min(rw*.75,H)
    x0=min(max(x+w/2-rw/2,0),W-rw);y0=min(max(y+h/2-rh/2,0),H-rh)
    return [float(x0),float(y0),float(x0+rw),float(y0+rh)]

def nearby(box,r):
    x,y,w,h=box;return r[0]<=x+w/2<=r[2] and r[1]<=y+h/2<=r[3]

def panel(ax,image,rows,fdi,case,method,crop=False,ground_truth=False):
    W,H=image.size;r=case['crop_xyxy'];col=GT_COLOR if ground_truth else PRED_COLOR
    ax.imshow(image,interpolation='antialiased',zorder=0)
    labelrows=[]
    target_pred=case['statuses'].get(method,{}).get('prediction')
    for row in rows:
        x,y,w,h=row['bbox'];inside=nearby(row['bbox'],r)
        if crop and not inside:continue
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=col,linewidth=.75 if crop else .45,alpha=.98 if inside else .65,zorder=2))
        if crop and inside:
            target=ground_truth and row['id']==case['target_annotation']['id']
            if not ground_truth:target=row==target_pred or row==case.get('suppressed_prediction')
            if target:labelrows.append(row)
    groups=defaultdict(list)
    for row in labelrows:groups[tuple(round(z,1) for z in row['bbox'])].append(row)
    for group in groups.values():
        for offset,row in enumerate(sorted(group,key=lambda p:p.get('score',1),reverse=True)):
            x,y,w,h=row['bbox'];px=np.clip(x+.04*w,r[0]+.025*(r[2]-r[0]),r[2]-.06*(r[2]-r[0]));py=np.clip(y+.03*h+offset*.095*(r[3]-r[1]),r[1]+.04*(r[3]-r[1]),r[3]-.08*(r[3]-r[1]))
            ax.text(px,py,str(fdi[row['category_id']]),fontsize=9.0,color='white',ha='left',va='top',clip_on=True,bbox={'facecolor':col,'edgecolor':'none','boxstyle':'square,pad=.10','alpha':.94},zorder=3)
    if crop:ax.set_xlim(r[0],r[2]);ax.set_ylim(r[3],r[1])
    else:
        ax.add_patch(Rectangle((r[0],r[1]),r[2]-r[0],r[3]-r[1],fill=False,edgecolor='#333333',linewidth=.85,linestyle=(0,(4,2)),zorder=4))
        ax.set_xlim(0,W);ax.set_ylim(H,0)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal')
    for sp in ax.spines.values():sp.set_visible(False)

def export(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipping=[]
    for text in fig.findobj(match=lambda o:isinstance(o,matplotlib.text.Text)):
        if not text.get_visible() or not text.get_text():continue
        bb=text.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:clipping.append(text.get_text())
    assert not clipping,(stem,clipping)
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=600,bbox_inches=None,pad_inches=0)
    fig.savefig(OUT/f'{stem}_preview.png',dpi=160)
    plt.close(fig)

def grid(cases,cohort,methods,stem):
    imgs,gt,fdi=dataset(cohort);n=len(methods)+2
    titles=['Original','Ground truth']+[name for name,_ in methods]
    display={'No high-pass':'No high-pass','No second-order':'No second-order','Neither branch':'Neither branch','Full HOOA':'Full HOOA','Center + shape':'Center + shape','Official DEIM-s':'Official DEIM-s','DENT-STRIDE':'DENT-STRIDE'}
    width=7.40;cellw=(width-.36)/n;ratio=float(np.mean([c['image_record']['height']/c['image_record']['width'] for c in cases]))
    image_height=cellw*(ratio+.75);height=len(cases)*(image_height+.20)+.44
    fig=plt.figure(figsize=(width,height),facecolor='white')
    # Reserve 0.32 inch explicitly for titles; no tight crop can remove half a glyph.
    top=1-.34/height
    gs=fig.add_gridspec(len(cases)*2,n,left=.037,right=.993,bottom=.018,top=top,wspace=.065,hspace=.13,height_ratios=[ratio,.75]*len(cases))
    for i,case in enumerate(cases):
        case['crop_xyxy']=roi(case);image=Image.open(case['image_file']).convert('RGB')
        for j in range(n):
            ax=fig.add_subplot(gs[2*i,j]);zoom=fig.add_subplot(gs[2*i+1,j])
            name=titles[j];rows=[] if j==0 else (gt[case['image_id']] if j==1 else methods[j-2][1][case['image_id']])
            panel(ax,image,rows,fdi,case,name,crop=False,ground_truth=j==1)
            panel(zoom,image,rows,fdi,case,name,crop=True,ground_truth=j==1)
            if i==0:
                # Figure-coordinate title anchors guarantee a common, non-clipped header.
                pos=ax.get_position();fig.text(pos.x0+pos.width/2,1-.17/height,display.get(name,name),fontsize=9.3,fontweight='bold',ha='center',va='center')
            if j==0:
                pos=ax.get_position();fig.text(.008,pos.y1,chr(ord('A')+i),fontsize=11,fontweight='bold',ha='left',va='top')
    export(fig,stem)

def describe(case,method):
    _,_,fdi=dataset(case['cohort']);st=case['statuses'][method];pred=st['prediction']
    label='not localized' if pred is None else 'FDI '+str(fdi[pred['category_id']])
    ct=case['counts'][method]
    return f"{method}: {label}, {ct['TP']} correct matches and {ct['FP']} false positives"

def verify_prior_boxes(cases,seed=0):
    path=RUNS/'pano_historical_spatialshape'/f'seed_{seed}'/'evaluation/pre_topk_candidates.npz';cached=np.load(path)
    for c in cases:
        at=int(np.flatnonzero(cached['image_ids']==c['image_id'])[0]);W,H=cached['original_sizes'][at]
        boxes=cached['boxes_cxcywh'][at].astype(float)*np.array([W,H,W,H]);boxes[:,:2]-=boxes[:,2:]/2
        checks={}
        for name,p in [('suppressed',c.get('suppressed_prediction')),('no_prior',c.get('tracked_no_prior')),('combined',c.get('tracked_combined'))]:
            if p is None:continue
            err=np.max(np.abs(boxes-np.asarray(p['bbox'])),axis=1);idx=int(err.argmin());assert err[idx]<.025,(c['image_id'],name,float(err[idx]))
            checks[name]={'query_index':idx,'maximum_box_coordinate_difference_pixels':float(err[idx]),'pre_topk_file':str(path)}
        c['same_box_checks']=checks

def main():
    for path in Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman').glob('*.TTF'):fm.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    result={'protocol':{'score_threshold':SCORE,'IoU_threshold':IOU,'matching':'class-agnostic Hungarian maximizing total IoU, identity checked afterwards','seed':0,'selection':'Illustrative outcome-based examples, not a random sample or estimate of benefit frequency. Aggregate effects must be read from complete validation tables.'},'figures':{}}
    captions={}
    hooa=[];sources={}
    for label,config in [('No high-pass','internal_mobile_hooa_no_highpass'),('No second-order','internal_mobile_hooa_no_second_order'),('Neither branch','internal_mobile_hooa_neither'),('Full HOOA','mobile_1024_full')]:
        by,meta=run_predictions(config,'Mobile');hooa.append((label,by));sources[label]=meta
    cases=select_hooa(hooa);stem='Figure_8_HOOA_same_image';grid(cases,'Mobile',hooa,stem)
    result['figures'][stem]={'cohort':'Mobile','cases':cases,'sources':sources,'annotation_file':DATA['Mobile']['ann'],'annotation_sha256':sha(DATA['Mobile']['ann'])}
    captions[stem]='HOOA branch comparison on the same smartphone images. Independently trained seed-0 models retain the same training protocol and differ only in the high-pass and second-order branches. Columns show the original image, ground truth, three branch-removal configurations, and full HOOA; lower rows show identical target enlargements. Teal boxes are ground truth and orange boxes are predictions retained at score ≥0.30. '+ ' '.join(f"({chr(65+i)}) The highlighted reference tooth is FDI {c['target_FDI']}; "+'; '.join(describe(c,n) for n,_ in hooa)+'.' for i,c in enumerate(cases))+' These selected examples illustrate branch effects; the complete validation ablation determines their aggregate AP differences.'
    prior=[];sources={}
    for label,folder in [('No prior','replay_prior_none'),('Center','replay_prior_center'),('Shape','replay_prior_shape'),('Center + shape','replay_prior_both')]:
        by,meta=run_predictions('pano_historical_spatialshape','Pano',folder,seed=2);prior.append((label,by));sources[label]=meta
    assert len({m['checkpoint_sha256'] for m in sources.values()})==1
    cases=select_prior(prior);verify_prior_boxes(cases,seed=2);stem='Figure_10_Pano_same_weight_prior';grid(cases,'Pano',prior,stem)
    result['figures'][stem]={'cohort':'Pano','cases':cases,'sources':sources,'annotation_file':DATA['Pano']['ann'],'annotation_sha256':sha(DATA['Pano']['ann'])}
    phrases=[]
    for i,c in enumerate(cases):
        if c['mechanism']=='incorrect query-class candidate suppression':
            phrase=f"({chr(65+i)}) An incorrect FDI {c['suppressed_FDI']} candidate near the highlighted FDI {c['target_FDI']} tooth is suppressed; correct matches change from {c['counts']['No prior']['TP']} to {c['counts']['Center + shape']['TP']} and false positives from {c['counts']['No prior']['FP']} to {c['counts']['Center + shape']['FP']}."
        else:
            fdi=dataset('Pano')[2];a=fdi[c['tracked_no_prior']['category_id']];b=fdi[c['tracked_combined']['category_id']]
            phrase=f"({chr(65+i)}) At the same query box, the highest retained label changes from FDI {a} to FDI {b} for reference FDI {c['target_FDI']}; image-level correct matches change from {c['counts']['No prior']['TP']} to {c['counts']['Center + shape']['TP']}."
        phrases.append(phrase)
    captions[stem]='Same-weight DENTEX anatomical-prior comparison. The same seed-2 checkpoint is evaluated without a prior, with center scoring, with shape scoring, and with both. Scores are recomputed from all ordinary query–class candidates before native top-k selection; box coordinates do not change. Original images and reference annotations are accompanied by identical target enlargements. Teal boxes denote ground truth and orange boxes predictions retained at score ≥0.30. '+' '.join(phrases)+' The favorable and adverse examples illustrate score-ranking effects rather than a change in learned localization.'
    external_entries=read('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/evidence/external_prediction_sources.json')['entries']
    for cohort,letter in [('Mobile','A'),('Pano','B'),('DSLR','C')]:
        cfg=DATA[cohort];base_dir=OFFICIAL/cfg['official']/'seed_0';base_summary=read(base_dir/'metrics_summary.json')
        base,bmeta=predictions(base_dir/'best_predictions.coco.json.gz',cohort,{'method':'Official DEIM-s','checkpoint_sha256':base_summary['checkpoint_sha256'],'seed':0,'role':'framework-level reference, native training protocol'})
        ext=next(m for m in external_entries if m['cohort']==cohort and m['model']=='RT-DETR-L')
        assert ext['COCO_bbox_content_identical'] and ext['class_map_matches_training_export']
        assert sha(ext['prediction_path'])==ext['prediction_sha256']
        external,emeta=predictions(ext['prediction_path'],cohort,{'method':'RT-DETR-L','checkpoint_file':ext['weight_path'],'checkpoint_sha256':ext['weight_sha256'],'role':'framework-level reference, native training and post-processing'})
        full,fmeta=run_predictions(cfg['final'],cohort,'replay_prior_both' if cohort=='Pano' else 'evaluation')
        methods=[('Official DEIM-s',base),('RT-DETR-L',external),('DENT-STRIDE',full)]
        cases=select_pair(cohort,base,full,methods);stem=f'Figure_12{letter}_{cohort}_same_image';grid(cases,cohort,methods,stem)
        result['figures'][stem]={'cohort':cohort,'cases':cases,'sources':{'Official DEIM-s':bmeta,'RT-DETR-L':emeta,'DENT-STRIDE':fmeta},'annotation_file':cfg['ann'],'annotation_sha256':sha(cfg['ann'])}
        captions[stem]=f"Same-image {cohort.lower()} detector comparison. Columns show the original image, ground truth, official DEIM-s, RT-DETR-L, and DENT-STRIDE; each lower row uses the same target enlargement. Teal boxes denote reference annotations and orange boxes model predictions retained at score ≥0.30. FDI labels identify the highlighted localization match. "+' '.join(f"({chr(65+i)}) Reference FDI {c['target_FDI']}; "+'; '.join(describe(c,n) for n,_ in methods)+'.' for i,c in enumerate(cases))+' Frameworks retain their native training and output-processing settings; this is a system-level comparison, not an isolated module ablation. The selected improvement and adverse examples do not estimate population frequencies.'
    (OUT/'prediction_figure_sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/'figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
    print(json.dumps({stem:[{'image_id':c['image_id'],'FDI':c['target_FDI'],'type':c['type'],'counts':c['counts']} for c in row['cases']] for stem,row in result['figures'].items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
