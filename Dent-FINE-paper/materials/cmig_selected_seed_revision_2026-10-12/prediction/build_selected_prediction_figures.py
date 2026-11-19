"""Same-image figures from selected-run native exports; no model reranking edits."""
from pathlib import Path
import importlib.util
import json
import numpy as np

OUT=Path(__file__).resolve().parent
TRAIN=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
BASE=Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-11/prediction_figures/build_prediction_figures.py')
spec=importlib.util.spec_from_file_location('source_prediction_renderer',BASE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=OUT;m.RUNS=TRAIN/'runs'
CONFIGS={'Mobile':('mobile_proto_k4_tau0p50',2),'Pano':('pano_spatialshape_original',3),'DSLR':('dslr_cmfa_extra_decay_only',3)}

def verify_prior_boxes(cases):
    path=m.RUNS/'pano_spatialshape_original/seed_3/evaluation/pre_topk_candidates.npz'
    cached=np.load(path)
    for c in cases:
        at=int(np.flatnonzero(cached['image_ids']==c['image_id'])[0]);W,H=cached['original_sizes'][at]
        boxes=cached['boxes_cxcywh'][at].astype(float)*np.array([W,H,W,H]);boxes[:,:2]-=boxes[:,2:]/2
        checks={}
        for name,p in [('suppressed',c.get('suppressed_prediction')),('no_prior',c.get('tracked_no_prior')),('combined',c.get('tracked_combined'))]:
            if p is None:continue
            err=np.max(np.abs(boxes-np.asarray(p['bbox'])),axis=1);idx=int(err.argmin())
            assert err[idx]<.025,(c['image_id'],name,float(err[idx]))
            checks[name]={'query_index':idx,'maximum_box_coordinate_difference_pixels':float(err[idx]),'pre_topk_file':str(path)}
        c['same_box_checks']=checks

def main():
    for path in Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman').glob('*.TTF'):m.fm.fontManager.addfont(str(path))
    m.plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    result={'protocol':{'score_threshold':m.SCORE,'IoU_threshold':m.IOU,'matching':'class-agnostic maximum-total-IoU Hungarian localization, then identity comparison','representative_seed_by_cohort':{c:s for c,(_,s) in CONFIGS.items()},'selection':'Highest validation AP among the author-selected seeds supplies each representative checkpoint. Examples include both improvement and adverse outcomes; they are illustrative rather than a random sample.'},'figures':{}}
    captions={}
    prior=[];sources={}
    for label,folder in [('No prior','replay_prior_none'),('Center','replay_prior_center'),('Shape','replay_prior_shape'),('Center + shape','replay_prior_both')]:
        by,meta=m.run_predictions('pano_spatialshape_original','Pano',folder,seed=3);prior.append((label,by));sources[label]=meta
    assert len({r['checkpoint_sha256'] for r in sources.values()})==1
    cases=m.select_prior(prior);verify_prior_boxes(cases)
    stem='Figure_10_Pano_same_weight_prior';m.grid(cases,'Pano',prior,stem)
    result['figures'][stem]={'cohort':'Pano','cases':cases,'sources':sources,'annotation_file':m.DATA['Pano']['ann'],'annotation_sha256':m.sha(m.DATA['Pano']['ann'])}
    phrases=[]
    for i,c in enumerate(cases):
        if c['mechanism']=='incorrect query-class candidate suppression':
            phrase=f"({chr(65+i)}) An incorrect FDI {c['suppressed_FDI']} candidate near reference FDI {c['target_FDI']} is suppressed; correct matches change from {c['counts']['No prior']['TP']} to {c['counts']['Center + shape']['TP']} and false positives from {c['counts']['No prior']['FP']} to {c['counts']['Center + shape']['FP']}."
        else:
            fdi=m.dataset('Pano')[2];a=fdi[c['tracked_no_prior']['category_id']];b=fdi[c['tracked_combined']['category_id']]
            phrase=f"({chr(65+i)}) At the same query box, the highest retained label changes from FDI {a} to FDI {b} for reference FDI {c['target_FDI']}; image-level correct matches change from {c['counts']['No prior']['TP']} to {c['counts']['Center + shape']['TP']}."
        phrases.append(phrase)
    captions[stem]='Same-weight DENTEX anatomical-prior comparison using the representative seed-3 checkpoint. The four scoring conditions use no prior, center only, shape only, and center plus shape. Scores are recomputed before native top-300 selection and box coordinates are unchanged. Teal boxes denote ground truth and orange boxes predictions retained at score ≥0.30; lower rows show identical target enlargements. '+' '.join(phrases)+' The favorable and adverse examples illustrate score-ranking effects, not changes in learned localization.'
    entries=m.read('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/evidence/external_prediction_sources.json')['entries']
    for cohort,letter in [('Mobile','A'),('Pano','B'),('DSLR','C')]:
        cfg=m.DATA[cohort];base_dir=m.OFFICIAL/cfg['official']/'seed_0';summary=m.read(base_dir/'metrics_summary.json')
        base,bmeta=m.predictions(base_dir/'best_predictions.coco.json.gz',cohort,{'method':'Official DEIM-s','checkpoint_sha256':summary['checkpoint_sha256'],'seed':0,'role':'framework-level reference, native training protocol'})
        ext=next(r for r in entries if r['cohort']==cohort and r['model']=='RT-DETR-L')
        assert ext['COCO_bbox_content_identical'] and ext['class_map_matches_training_export'] and m.sha(ext['prediction_path'])==ext['prediction_sha256']
        external,emeta=m.predictions(ext['prediction_path'],cohort,{'method':'RT-DETR-L','checkpoint_file':ext['weight_path'],'checkpoint_sha256':ext['weight_sha256'],'role':'framework-level reference, native training and post-processing'})
        config,seed=CONFIGS[cohort]
        full,fmeta=m.run_predictions(config,cohort,'replay_prior_both' if cohort=='Pano' else 'evaluation',seed=seed)
        methods=[('Official DEIM-s',base),('RT-DETR-L',external),('DENT-STRIDE',full)]
        cases=m.select_pair(cohort,base,full,methods);stem=f'Figure_12{letter}_{cohort}_same_image';m.grid(cases,cohort,methods,stem)
        result['figures'][stem]={'cohort':cohort,'cases':cases,'sources':{'Official DEIM-s':bmeta,'RT-DETR-L':emeta,'DENT-STRIDE':fmeta},'annotation_file':cfg['ann'],'annotation_sha256':m.sha(cfg['ann'])}
        captions[stem]=f"Same-image {cohort.lower()} detector comparison using the representative DENT-STRIDE seed-{seed} checkpoint. Columns show original image, ground truth, official DEIM-s, RT-DETR-L, and DENT-STRIDE; lower rows share identical target enlargements. Teal denotes ground truth and orange predictions retained at score ≥0.30. "+' '.join(f"({chr(65+i)}) Reference FDI {c['target_FDI']}; "+'; '.join(m.describe(c,n) for n,_ in methods)+'.' for i,c in enumerate(cases))+' Frameworks retain their native training and output-processing settings. Improvement and adverse examples are illustrative and do not estimate population frequencies.'
    (OUT/'prediction_figure_sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/'figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
    print(json.dumps({stem:[{'image_id':c['image_id'],'FDI':c['target_FDI'],'type':c['type'],'counts':c['counts']} for c in row['cases']] for stem,row in result['figures'].items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
