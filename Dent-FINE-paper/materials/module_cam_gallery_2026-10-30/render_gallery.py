"""Publication CAM gallery: real computations, shared labels and explicit scope."""
from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
for f in (HERE.parent/'cmig_revision_2026-10-03/references/times_new_roman').glob('*.TTF'):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':10,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none',
    'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
PLAN=json.loads((HERE/'gallery_plan.json').read_text())
TEAL='#087B80';ORANGE='#AD5528';GRAY='#869099'
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
CAPTIONS={};MANIFEST={}
DATA={g['name']:g for g in PLAN['groups']}
CAPTURE={}
for g in PLAN['groups']:
    for side in ['before','after']:
        report=json.loads((HERE/f'{g["name"]}_{side}_capture.json').read_text())
        CAPTURE[g['name'],side]={r['case_id']:r for r in report['records']}


def save(fig,name,caption,source):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(match=lambda o:isinstance(o,matplotlib.text.Text)):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.width+.5 and b.y1<=fig.bbox.height+.5,(name,t.get_text(),b)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
    fig.savefig(OUT/f'{name}_preview.png',dpi=150)
    plt.close(fig);CAPTIONS[name]=caption;MANIFEST[name]=source
    print('EXPORTED',name,flush=True)


def maps(case,side):
    return np.load(CAPTURE[case['group'],side][case['case_id']]['raw_maps'])


def bbox_mask(a,case):
    H,W=a.shape;im=case['image_record'];x,y,w,h=case['target_annotation']['bbox']
    xx=(np.arange(W)+.5)*im['width']/W;yy=(np.arange(H)+.5)*im['height']/H
    return (xx[None,:]>=x)&(xx[None,:]<=x+w)&(yy[:,None]>=y)&(yy[:,None]<=y+h)


def measure(a,case):
    mask=bbox_mask(a,case);total=float(a.sum());area=float(mask.mean())
    mass=float(a[mask].sum()/total) if total>0 else 0
    peak=np.unravel_index(int(a.argmax()),a.shape)
    return {'positive_mass_inside_GT':mass,'GT_fraction':area,'enrichment':mass/area if area else None,
        'peak_inside_GT':bool(mask[peak]),'raw_maximum':float(a.max()),'positive_sum':total}


def case_crop(case):
    im=case['image_record'];W,H=im['width'],im['height'];x,y,w,h=case['target_annotation']['bbox']
    cw=max(2.6*w,2.0*h*4/3);ch=cw*.75;scale=min(1,W/cw,H/ch);cw*=scale;ch*=scale
    left=min(max(x+w/2-cw/2,0),W-cw);top=min(max(y+h/2-ch/2,0),H-ch)
    return [left,top,left+cw,top+ch]


def base(ax,case,gray=False):
    im=Image.open(case['image_file']).convert('RGB')
    crop=case_crop(case);bounds=[int(np.floor(crop[0])),int(np.floor(crop[1])),int(np.ceil(crop[2])),int(np.ceil(crop[3]))]
    shown=im.crop(bounds)
    ax.imshow(shown.convert('L') if gray else shown,cmap='gray',vmin=0,vmax=255,
        extent=[bounds[0],bounds[2],bounds[3],bounds[1]])
    return im


def finish(ax,case):
    crop=case_crop(case);ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1])
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)


def box(ax,bb,color=TEAL,label=None,style='solid',case=None):
    x,y,w,h=bb
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,lw=1.1,linestyle=style))
    if label is not None:
        crop=case_crop(case)
        ax.text(x+.02*w,y-.027*(crop[3]-crop[1]),str(label),ha='left',va='bottom',
            fontsize=10.5,color=color,fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':.93,'pad':.6})


def original(ax,case):
    base(ax,case);box(ax,case['target_annotation']['bbox'],label=case['target_FDI'],case=case);finish(ax,case)


def prediction(ax,case,side):
    base(ax,case)
    box(ax,case['target_annotation']['bbox'],style=(0,(3,2)),case=case)
    recorded=case[side]['prediction'];live=CAPTURE[case['group'],side][case['case_id']].get('live_prediction',recorded)
    box(ax,live['bbox'],ORANGE,recorded['FDI'],case=case);finish(ax,case)


def cam(ax,case,side,key='FDI_logit_gradcam'):
    base(ax,case,gray=True);a=maps(case,side)[key];norm=a/max(float(a.max()),1e-20)
    # No target masking, hand-drawn hotspot, percentile clipping, or smoothing.
    # Normalize by the maximum over the entire original feature plane.
    im=case['image_record'];rgba=plt.get_cmap('Reds')(norm)
    rgba[:,:,3]=.73*np.minimum(norm/.25,1.0)
    H,W=norm.shape;crop=case_crop(case)
    x0=max(0,int(np.floor(crop[0]/im['width']*W))-1);x1=min(W,int(np.ceil(crop[2]/im['width']*W))+1)
    y0=max(0,int(np.floor(crop[1]/im['height']*H))-1);y1=min(H,int(np.ceil(crop[3]/im['height']*H))+1)
    ax.imshow(rgba[y0:y1,x0:x1],extent=[x0/W*im['width'],x1/W*im['width'],y1/H*im['height'],y0/H*im['height']],interpolation='bilinear')
    box(ax,case['target_annotation']['bbox'],style=(0,(3,2)),case=case)
    finish(ax,case)


def label_map(g):
    annpath='/home/wuxi/coco/valid/annotations_fdi_types.json'
    ann=json.loads(Path(annpath).read_text())
    return {c['id']:int(c['name']) for c in ann['categories']}


def cosine(ax,case):
    cid=case['target_annotation']['category_id'];info=CAPTURE['Prototype','after'][case['case_id']]
    competitor=int(info['scores']['prototype_competitor_category']);a=maps(case,'after')['prototype_cosine_allclasses']
    labels=label_map(DATA['Prototype']);ys=np.arange(4)
    ax.barh(ys-.17,a[cid],height=.30,color=TEAL,label=f"FDI {labels[cid]}")
    ax.barh(ys+.17,a[competitor],height=.30,color=GRAY,label=f"FDI {labels[competitor]}")
    ax.axvline(0,color='#A0A6AA',lw=.6);ax.set_xlim(-1,1);ax.set_ylim(3.6,-1.25)
    ax.set_yticks(ys,['P1','P2','P3','P4']);ax.set_xticks([-1,0,1]);ax.tick_params(length=2,pad=2)
    ax.set_xlabel('Cosine similarity',fontsize=8,labelpad=2)
    ax.legend(frameon=False,fontsize=7.5,loc='upper center',bbox_to_anchor=(.50,1.0),ncol=2,
        handlelength=.8,columnspacing=.7,handletextpad=.3,borderpad=0)


def class_scores(ax,case):
    a=maps(case,'after')['prototype_class_logits'];cid=case['target_annotation']['category_id']
    train=json.loads(Path('/home/wuxi/coco/train/annotations_fdi_types.json').read_text())
    supported={v['category_id'] for v in train['annotations']}
    top=[int(i) for i in np.argsort(-a) if int(i) in supported][:5]
    labels=label_map(DATA['Prototype']);ys=np.arange(len(top))
    colors=[TEAL if i==cid else GRAY for i in top]
    pos=ax.get_position();ax.set_position([pos.x0+.015,pos.y0,pos.width-.015,pos.height])
    ax.barh(ys,a[top],height=.63,color=colors)
    ax.set_yticks(ys,[str(labels[i]) for i in top]);ax.set_ylim(len(top)-.4,-.6)
    ax.set_xlim(0,max(float(a[top].max())*1.12,1));ax.set_xticks([0,1.5,3])
    ax.tick_params(length=2,pad=2);ax.set_xlabel('Auxiliary score',fontsize=8,labelpad=2)


def grid(cases,name,prototype=False,method='gradcam',prototype_scores=False):
    n=len(cases);height=.46+n*1.48;fig=plt.figure(figsize=(8.35,height))
    panelh=1.10/height
    if prototype:titles=['Reference','Without prototype','With prototype','Prototype CAM',
        'Prototype scores' if prototype_scores else 'Prototype similarity']
    else:
        group=cases[0]['group'];before='Reduced HOOA' if group=='HOOA' else f'Without {group}'
        after='Full HOOA' if group=='HOOA' else f'With {group}'
        titles=['Reference','Control CAM',f'{group} CAM',before,after]
    for r,case in enumerate(cases):
        bottom=1-(.35+r*1.48+1.12)/height
        for c in range(5):
            ax=fig.add_axes([.036+c*.192,bottom,.173,panelh])
            if prototype:
                if c==0:original(ax,case)
                elif c in [1,2]:prediction(ax,case,'before' if c==1 else 'after')
                elif c==3:cam(ax,case,'after','Prototype_logit_'+method)
                else:(class_scores if prototype_scores else cosine)(ax,case)
            elif c==0:original(ax,case)
            elif c==1:cam(ax,case,'before','FDI_logit_'+method)
            elif c==2:cam(ax,case,'after','FDI_logit_'+method)
            else:prediction(ax,case,'before' if c==3 else 'after')
            if r==0:ax.set_title(titles[c],fontsize=9,pad=6)
        fig.text(.004,bottom+panelh+.005,f'({chr(97+r)})',ha='left',va='bottom',fontsize=10)
    group=cases[0]['group'];g=DATA[group]
    caption=(f'Query-specific {group} visualizations from separately trained controls. '
        f'Validation COCO AP is {g["before"]["metrics"]["AP"]:.3f} without the stated mechanism and '
        f'{g["after"]["metrics"]["AP"]:.3f} with it. '
        'Reference boxes are teal; predicted boxes and FDI labels are orange. CAMs target the reference FDI class at '
        'the independently matched ordinary query in each model, using the same finest encoder output level. '
        'Each positive map is divided by its own full-image maximum; colors therefore show relative spatial attribution, '
        'not comparable attribution magnitude across models. Maps are computed over the full input before cropping; '
        'GT boxes do not mask or define the attribution. Bilinear interpolation is used only for display. Native top1 predictions require score≥0.30 and '
        'class-agnostic one-to-one Hungarian localization matching at IoU≥0.50. '
        f'The method is {"Grad-CAM" if method=="gradcam" else "positive elementwise gradient-times-activation (HiResCAM form)"}. '
        'The native detached query-content initialization is not traversed by gradients, so CAMs are partial-path explanations. '
        'Examples are selected illustrations, not correction-rate estimates or independent-test evidence. ')
    if group=='HOOA':caption+=('The reduced model removes the high-pass contribution and replaces the second-order interaction '
        'with the context branch; it does not remove the whole HOOA module. Both runs use seed 0 and prototype temperature 0.20. ')
    elif group=='CMFA':caption+='Both runs use the controlled DSLR AdamW setting and seed 0. '
    else:
        caption+=('Prototype supervision uses four prototypes per class and temperature 0.50, with prototype inference fusion weight zero. '
            'The no-prototype control is independently retrained with the same other settings. Both use seed 0. ')
    if prototype:
        caption+=('The prototype CAM targets the auxiliary class score computed by LogSumExp of four normalized query–prototype '
            'cosine similarities divided by temperature, not the final detection logit. ')
        if prototype_scores:
            caption+=('The last column shows the five highest auxiliary prototype class scores among classes with training '
                'support; the reference class is teal. These unnormalized scores are not probabilities. ')
        else:
            caption+=('The last column shows the four cosine similarities for the reference class and the highest-scoring '
                'competing prototype class. P1–P4 are unordered slots within each class, not pairwise-corresponding anatomical '
                'types. Similar values across slots do not demonstrate prototype diversity. ')
        caption+=('The auxiliary branch is not used '
            'for inference score fusion and should not be described as spatial prototype templates. ')
    caption+='Cases: '+ '; '.join(f"({chr(97+i)}) image {c['image_id']}, reference {c['target_FDI']}, {c['before']['prediction']['FDI']}→{c['after']['prediction']['FDI']}, {c['kind']}" for i,c in enumerate(cases))+'.'
    save(fig,name,caption,{'group':group,'cases':[c['case_id'] for c in cases],'method':method,'prototype_layout':prototype})


def response(ax,case,key,cmap='Blues',fixed=False):
    a=maps(case,'after')[key].squeeze();im=case['image_record']
    norm=Normalize(0,1) if fixed else Normalize(0,max(float(a.max()),1e-20))
    H,W=a.shape;crop=case_crop(case)
    x0=max(0,int(np.floor(crop[0]/im['width']*W))-1);x1=min(W,int(np.ceil(crop[2]/im['width']*W))+1)
    y0=max(0,int(np.floor(crop[1]/im['height']*H))-1);y1=min(H,int(np.ceil(crop[3]/im['height']*H))+1)
    img=ax.imshow(a[y0:y1,x0:x1],cmap=cmap,norm=norm,
        extent=[x0/W*im['width'],x1/W*im['width'],y1/H*im['height'],y0/H*im['height']],interpolation='bilinear')
    box(ax,case['target_annotation']['bbox'],style=(0,(3,2)),case=case);finish(ax,case)
    return img


def branches(cases,name):
    fig=plt.figure(figsize=(8.35,.5+len(cases)*1.5));H=fig.get_figheight();ph=1.08/H
    keys=['encoder_hooa_attention_0_local_branch_RMS','encoder_hooa_attention_0_context_branch_RMS',
        'encoder_hooa_attention_0_edge_branch_RMS','HOOA_second_order_RMS']
    titles=['Reference','Local','Context','High-pass','Second-order','FDI CAM']
    for r,c in enumerate(cases):
        bottom=1-(.36+r*1.5+1.10)/H
        for j in range(6):
            ax=fig.add_axes([.035+j*.160,bottom,.143,ph])
            if j==0:original(ax,c)
            elif j==5:cam(ax,c,'after')
            else:response(ax,c,keys[j-1])
            if r==0:ax.set_title(titles[j],fontsize=9,pad=6)
        fig.text(.004,bottom+ph+.01,f'({chr(97+r)})',va='bottom')
    cap=('Native HOOA feature responses and query-specific attribution. Local, context, high-pass, and second-order '
        'panels show channel RMS magnitude of the actual internal branch tensors, normalized by their respective full-image '
        'maxima. They are feature responses, not class-specific attention or probabilities. The final column is Grad-CAM '
        'for the reference FDI detection logit at the finest encoder output. Darker branch responses do not establish '
        'greater discriminative value; efficacy is assessed by separately retrained controls. All examples use the complete '
        'reference Mobile model, seed 0, temperature 0.20. GT boxes are shown solely as spatial references. Cases: '+
        '; '.join(f"image {c['image_id']}, FDI {c['target_FDI']}" for c in cases)+'.')
    save(fig,name,cap,{'cases':[c['case_id'] for c in cases],'type':'native_RMS_and_GradCAM'})


def native_gates(cases,name):
    fig=plt.figure(figsize=(8.35,.6+len(cases)*1.45));H=fig.get_figheight();ph=1.04/H
    prefix=['encoder_top_down_fusions_1','encoder_bottom_up_fusions_0']
    for r,c in enumerate(cases):
        bottom=1-(.32+r*1.45+1.10)/H
        for j in range(4):
            ax=fig.add_axes([.040+j*.240,bottom,.212,ph])
            if j==0:original(ax,c)
            elif j in [1,2]:response(ax,c,prefix[j-1]+'_spatial_gate',fixed=True)
            else:cam(ax,c,'after','CMFA_output_margin_gradcam')
            if r==0:ax.set_title(['Reference','Top-down spatial gate','Bottom-up spatial gate','CMFA output CAM'][j],fontsize=9,pad=6)
        fig.text(.005,bottom+ph+.01,f'({chr(97+r)})',va='bottom')
    cax=fig.add_axes([.35,.035,.28,.018]);fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='Blues'),
        cax=cax,orientation='horizontal',ticks=[0,.5,1]);cax.tick_params(labelsize=8,length=2)
    cap=('Native CMFA spatial gates on an absolute sigmoid scale of 0–1, without per-image contrast stretching. '
        'Top-down refers to encoder.top_down_fusions.1 and bottom-up to encoder.bottom_up_fusions.0. '
        'The final column is Grad-CAM for the correct-FDI versus fixed-alternative logit margin at the top-down CMFA output. '
        'These are DSLR seed-0 AdamW examples. A spatially uniform gate must not be interpreted as localized attention. '
        'Channel gates are channel weights and are not reshaped into invented two-dimensional spatial maps. '
        'Gate and CAM colors represent different quantities; the colorbar applies only to the two native gate columns.')
    save(fig,name,cap,{'cases':[c['case_id'] for c in cases],'type':'native_gates_absolute_0_1'})


def main():
    comparisons=[];recommended={}
    for group in PLAN['groups']:
        name=group['name'];scored=[]
        for c in group['cases']:
            a,b=[measure(maps(c,s)['FDI_logit_gradcam'],c) for s in ['before','after']]
            ah,bh=[measure(maps(c,s)['FDI_logit_hirescam'],c) for s in ['before','after']]
            info={'case_id':c['case_id'],'group':name,'image_id':c['image_id'],'FDI':c['target_FDI'],
                'kind':c['kind'],'before_FDI':c['before']['prediction']['FDI'],'after_FDI':c['after']['prediction']['FDI'],
                'before_CAM':a,'after_CAM':b,'enrichment_change':b['enrichment']-a['enrichment'],
                'before_HiResCAM':ah,'after_HiResCAM':bh,'HiResCAM_enrichment_change':bh['enrichment']-ah['enrichment']}
            if name=='Prototype':
                p=measure(maps(c,'after')['Prototype_logit_gradcam'],c)
                info['prototype_CAM']=p
                info['prototype_HiResCAM']=measure(maps(c,'after')['Prototype_logit_hirescam'],c)
                pred=CAPTURE[name,'after'][c['case_id']]['scores']['prototype_class_prediction']
                info['prototype_classifier_correct']=int(pred)==c['target_annotation']['category_id']
            comparisons.append(info)
            grid([c],c['case_id']+'_GradCAM')
            grid([c],c['case_id']+'_HiResCAM',method='hirescam')
            if name=='Prototype':grid([c],c['case_id']+'_Prototype_detail',prototype=True)
            if c['kind']=='correction':
                if name=='Prototype':
                    rank=(info['prototype_classifier_correct'],info['prototype_HiResCAM']['peak_inside_GT'],info['prototype_HiResCAM']['enrichment'])
                else:rank=(bh['peak_inside_GT'],info['HiResCAM_enrichment_change']>0,info['HiResCAM_enrichment_change'],bh['enrichment'])
                scored.append((rank,c,info))
        scored.sort(key=lambda z:z[0],reverse=True)
        count=1 if name=='HOOA' else 3
        top=[v[1] for v in scored[:count]]
        recommended[name]={'case_ids':[c['case_id'] for c in top],'diagnostics':[v[2] for v in scored[:count]],
            'selection_rule':'First require a real FDI correction; then prefer target-localized elementwise-gradient CAMs. Prototype cases additionally prefer correct auxiliary prototype classification. HOOA has only one case with a clear target-localized peak and improved concentration among these candidates, so only that case is recommended. Illustrative selection is disclosed, not a prevalence estimate.'}
        grid(top,'Recommended_'+name+'_GradCAM')
        grid(top,'Recommended_'+name+'_HiResCAM',method='hirescam')
        if name=='HOOA':branches([v[1] for v in scored[:3]],'Supplement_HOOA_internal_responses')
        if name=='CMFA':native_gates(top,'Supplement_CMFA_native_gates')
        if name=='Prototype':grid(top,'Recommended_Prototype_mechanism',prototype=True,method='hirescam',prototype_scores=True)
        grid(group['cases'],'Gallery_'+name+'_all_cases_GradCAM')
    alltop=[next(c for c in DATA[n]['cases'] if c['case_id']==recommended[n]['case_ids'][0]) for n in ['CMFA','HOOA','Prototype']]
    # Mixed-group overview deliberately uses neutral control/full titles.
    # Per-group plates above have unambiguous method-specific titles and are preferred.
    (HERE/'CAM_spatial_diagnostics.json').write_text(json.dumps(comparisons,indent=2))
    (HERE/'recommendations.json').write_text(json.dumps(recommended,indent=2))
    (HERE/'figure_captions.json').write_text(json.dumps(CAPTIONS,indent=2))
    (HERE/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
    with zipfile.ZipFile(HERE/'Recommended_CAM_figures.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.glob('Recommended_*')):
            z.write(p,p.name)
        for name in ['figure_captions.json','recommendations.json','CAM_spatial_diagnostics.json']:
            z.write(HERE/name,name)
    print('FIGURES',len(MANIFEST),'RECOMMENDED',json.dumps(recommended,indent=2),flush=True)


if __name__=='__main__':main()
