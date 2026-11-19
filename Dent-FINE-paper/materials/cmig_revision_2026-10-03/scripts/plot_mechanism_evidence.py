"""Publication figures from captured model tensors, without generative image editing."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
DEST=ROOT/'figures/mechanism'
for f in (ROOT/'references/times_new_roman').glob('*.TTF'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(DEST/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.08)
    fig.savefig(DEST/f'{name}_preview.png',dpi=150,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def tensor_figures():
    ann=json.loads(Path('/home/wuxi/coco/valid/annotations_fdi_types.json').read_text())
    images={i['id']:i for i in ann['images']}
    prov=json.loads((DEST/'mechanism_provenance.json').read_text())
    for record in prov['selected_cases']:
        iid=record['image_id'];data=np.load(record['raw_tensor_file'])
        im=np.asarray(Image.open(Path('/home/wuxi/coco/valid')/images[iid]['file_name']).convert('RGB'))
        fig,axes=plt.subplots(3,4,figsize=(7.2,5.6),layout='constrained')
        axes[0,0].imshow(im);axes[0,0].set_title('Input')
        spec=[(axes[0,1],'CMFA_gA','CMFA channel gate A','gate'),(axes[0,2],'CMFA_gB','CMFA channel gate B','gate'),(axes[0,3],'CMFA_Gs','CMFA spatial gate','gate'),(axes[1,0],'HOOA_local','Local response','response'),(axes[1,1],'HOOA_context','Context response','response'),(axes[1,2],'HOOA_edge','High-pass response','response'),(axes[1,3],'HOOA_second_order','Second-order response','response'),(axes[2,0],'HOOA_channel_gate','HOOA channel gate','gate'),(axes[2,1],'HOOA_spatial_gate','HOOA spatial gate','gate'),(axes[2,2],'HOOA_input','HOOA input','response'),(axes[2,3],'HOOA_output','HOOA output','response')]
        for ax,key,title,kind in spec:
            x=data[key]
            if x.ndim==1:x=x.reshape(16,16)
            upper=1 if kind=='gate' else max(float(np.percentile(x,99)),1e-10)
            shown=x if kind=='gate' else np.clip(x/upper,0,1)
            ax.imshow(shown,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(title)
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        # One small common key covers both true [0,1] gates and normalized responses.
        scalar=plt.cm.ScalarMappable(cmap='Blues',norm=plt.Normalize(0,1))
        cb=fig.colorbar(scalar,ax=list(axes.flat),location='right',fraction=.018,pad=.015,shrink=.85,ticks=[0,.5,1])
        cb.ax.tick_params(labelsize=8);cb.outline.set_visible(False)
        save(fig,f'actual_CMFA_HOOA_image_{iid}')


def prototypes():
    data=np.load(DEST/'prototype_bank.npz');p=data['normalized_prototypes'];codes=data['fdi_codes'];support=data['train_support']
    mean=p.mean(1);mean/=np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-12)
    similarity=mean@mean.T
    unsupported=support==0;similarity[unsupported,:]=np.nan;similarity[:,unsupported]=np.nan
    triangle=np.triu_indices(4,1)
    cohesion=np.array([(row@row.T)[triangle].mean() for row in p]);cohesion[unsupported]=np.nan
    np.savez_compressed(DEST/'prototype_summary.npz',fdi_codes=codes,centroid_similarity=similarity,cohesion=cohesion,train_support=support)
    with (DEST/'prototype_cohesion_by_FDI.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['FDI','train_instances','four_prototype_mean_pairwise_cosine'])
        for c,n,v in zip(codes,support,cohesion):writer.writerow([int(c),int(n),'' if not np.isfinite(v) else float(v)])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1.1,1]},layout='constrained')
    cmap=plt.colormaps['RdBu_r'].copy();cmap.set_bad('#e8e8e8')
    mm=ax.imshow(similarity,cmap=cmap,vmin=-1,vmax=1,interpolation='none')
    starts=[0,8,16,24,32,37,42,47];ticks=[3.5,11.5,19.5,27.5,34,39,44,49]
    ax.set_xticks(ticks,[f'Q{i}' for i in range(1,9)]);ax.set_yticks(ticks,[f'Q{i}' for i in range(1,9)])
    for v in starts[1:]:ax.axhline(v-.5,color='white',lw=.6);ax.axvline(v-.5,color='white',lw=.6)
    ax.set_title('(A) Class-centroid similarity',loc='left')
    cb=fig.colorbar(mm,ax=ax,fraction=.046,pad=.025);cb.ax.tick_params(labelsize=8);cb.set_label('Cosine similarity')
    cohorts=[cohesion[(codes//10)==q] for q in range(1,9)]
    rng=np.random.default_rng(3)
    for q,vals in enumerate(cohorts,1):
        vals=vals[np.isfinite(vals)]
        bx.scatter(q+rng.uniform(-.10,.10,len(vals)),vals,s=18,facecolors='white',edgecolors='#355b78',linewidths=.7)
        bx.plot([q-.18,q+.18],[np.mean(vals)]*2,color='#222',lw=1.6)
    bx.set_xticks(range(1,9),[f'Q{i}' for i in range(1,9)]);bx.set_xlabel('FDI quadrant');bx.set_ylabel('Within-class cosine cohesion')
    bx.set_title('(B) Four-prototype cohesion',loc='left');bx.set_ylim(0,1);bx.grid(axis='y',color='.92',lw=.5)
    bx.set_ylim(0,1.03)
    save(fig,'current_Mobile_prototype_diagnostics')
    fig,(ax,bx)=plt.subplots(2,1,figsize=(7.2,4.4),layout='constrained')
    for axis,keep in [(ax,codes<50),(bx,codes>=50)]:
        xx=np.arange(sum(keep));yy=cohesion[keep];valid=np.isfinite(yy)
        axis.plot(xx[valid],yy[valid],'o',ms=3.6,color='#355b78')
        for x,v,n in zip(xx,yy,support[keep]):
            if not n:axis.text(x,.05,'—',ha='center',va='bottom',fontsize=9,color='.5')
        axis.set_xticks(xx,[str(int(c)) for c in codes[keep]],rotation=0,fontsize=7.5);axis.set_ylim(0,1.03);axis.set_ylabel('Prototype cohesion');axis.grid(axis='y',color='.92',lw=.5)
    ax.set_title('(A) Permanent FDI classes',loc='left');bx.set_title('(B) Primary FDI classes',loc='left');bx.set_xlabel('FDI code')
    save(fig,'current_Mobile_cohesion_all_FDI')


def factor_confusion():
    x=np.load(DEST/'factor_confusion_raw.npz');prov=json.loads((DEST/'mechanism_provenance.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.55),layout='constrained',gridspec_kw={'width_ratios':[1,1,.80]})
    for ax,name,title,labels in zip(axes,['quadrant','position','dentition'],['(A) Quadrant','(B) Tooth position','(C) Dentition'],[[str(i) for i in range(1,9)],[str(i) for i in range(1,9)],['Primary','Permanent']]):
        counts=x[name];den=counts.sum(1,keepdims=True);val=np.divide(counts,den,out=np.zeros_like(counts,dtype=float),where=den>0)*100
        display=val.copy();display[den[:,0]==0,:]=np.nan
        cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#e8e8e8')
        ax.imshow(display,cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
        for i in range(len(val)):
            if den[i,0]==0:
                ax.text(i,i,'—',ha='center',va='center',fontsize=9,color='#555')
                continue
            for j in range(len(val)):
                if i==j or val[i,j]>=3:
                    ax.text(j,i,f'{val[i,j]:.0f}',ha='center',va='center',fontsize=7.5 if len(val)>2 else 10,color='white' if val[i,j]>60 else '#222')
        ax.set_title(title,loc='left');ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(labels)),labels)
        if len(labels)==2:ax.tick_params(axis='x',rotation=25,labelsize=8)
        ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
    save(fig,'current_Mobile_factor_confusions')
    counts={name:{'matched_queries':int(x[name].sum()),'accuracy':float(np.trace(x[name])/x[name].sum())} for name in x.files}
    assert all(v['matched_queries']==prov['captured_queries_total'] for v in counts.values())
    (DEST/'factor_confusion_summary.json').write_text(json.dumps(counts,indent=2))


if __name__=='__main__':
    tensor_figures();prototypes();factor_confusion()
