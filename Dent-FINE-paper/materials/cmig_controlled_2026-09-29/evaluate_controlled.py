"""Native FP32 checkpoint replay, full pre-top-k candidates, COCO and fixed endpoints."""
import argparse
from collections import Counter
from pathlib import Path
from hashlib import sha256
import gzip
import json
import sys
import numpy as np


def fixed_endpoints(predictions, gt, score=.30, iou=.50):
    from scipy.optimize import linear_sum_assignment
    by_pred={i:[] for i in gt.imgs}
    for p in predictions:
        if p['score']>=score: by_pred[p['image_id']].append(p)
    counts=Counter();rows=[]
    for iid in sorted(gt.imgs):
        pp=by_pred[iid];gg=gt.imgToAnns.get(iid,[])
        pc=Counter(p['category_id'] for p in pp)
        matched=correct=0
        if pp and gg:
            a=np.array([p['bbox'] for p in pp]);b=np.array([g['bbox'] for g in gg])
            lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
            inter=np.clip(hi-lo,0,None).prod(-1);union=a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-inter
            overlaps=inter/np.maximum(union,1e-12)
            rr,cc=linear_sum_assignment(-overlaps)
            for r,c in zip(rr,cc):
                if overlaps[r,c]>=iou:
                    matched+=1;correct+=int(pp[r]['category_id']==gg[c]['category_id'])
        row={'image_id':iid,'GT':len(gg),'predictions':len(pp),'localized_matches':matched,'correct_FDI_matches':correct,'TP':correct,'FP':len(pp)-correct,'FN':len(gg)-correct,'duplicate_image':int(any(v>1 for v in pc.values())),'duplicate_count':sum(max(v-1,0) for v in pc.values())}
        rows.append(row);counts.update({k:v for k,v in row.items() if k!='image_id'})
    def ratio(a,b):return a/b if b else None
    metrics={'score_threshold':score,'IoU_threshold':iou,'counts':dict(counts),'Precision':ratio(counts['TP'],counts['TP']+counts['FP']),'Recall':ratio(counts['TP'],counts['TP']+counts['FN']),'F1':ratio(2*counts['TP'],2*counts['TP']+counts['FP']+counts['FN']),'matched_tooth_FDI_accuracy':ratio(counts['correct_FDI_matches'],counts['localized_matches']),'localization_coverage':ratio(counts['localized_matches'],counts['GT']),'raw_duplicate_image_rate':ratio(counts['duplicate_image'],len(gt.imgs)),'denominators':'P:predictions, R:GT, FDI accuracy:localized matches, coverage:GT, duplicate-image rate:all images'}
    return metrics,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--prior',choices=['center','shape','both']);p.add_argument('--factor-inference',choices=['on','off']);a=p.parse_args()
    job=json.loads(Path(a.job).read_text());run=Path(job['output_dir'])
    destination=run/('replay_prior_'+a.prior if a.prior else 'replay_factor_'+a.factor_inference if a.factor_inference else 'evaluation')
    destination.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,job['implementation'])
    import torch
    from engine.core import YAMLConfig
    from controlled_runtime import install_controls
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    torch.set_num_threads(4)
    control=job['controls'].copy()
    if a.factor_inference:control['factor_inference_fusion']=a.factor_inference=='on'
    install_controls(control)
    cfg=YAMLConfig(job['config'],device='cuda',use_amp=True)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    post=cfg.postprocessor.cuda()
    if a.prior:
        post.use_dental_spatial_prior=a.prior in ['center','both'];post.use_dental_shape_prior=a.prior in ['shape','both']
        post.use_dental_view_prior=False;post.use_dental_appearance_prior=False
    all_predictions=[];logits=[];boxes=[];ids=[];sizes=[]
    params=None
    params_trainable=None
    def consume(outputs, image_ids, original_sizes):
        with torch.no_grad(): results=post(outputs,original_sizes)
        for iid,result in zip(image_ids,results):
            for label,score,bb in zip(result['labels'].tolist(),result['scores'].tolist(),result['boxes'].tolist()):
                x1,y1,x2,y2=bb;all_predictions.append({'image_id':int(iid),'category_id':int(label),'bbox':[x1,y1,x2-x1,y2-y1],'score':float(score)})
    if a.prior:
        cached=np.load(run/'evaluation/pre_topk_candidates.npz',allow_pickle=False)
        for start in range(0,len(cached['image_ids']),16):
            s=slice(start,start+16)
            outputs={'pred_logits':torch.from_numpy(cached['logits'][s]).cuda(),'pred_boxes':torch.from_numpy(cached['boxes_cxcywh'][s]).cuda()}
            consume(outputs,cached['image_ids'][s],torch.from_numpy(cached['original_sizes'][s]).cuda())
    else:
        model=cfg.model.cuda().eval()
        checkpoint=torch.load(run/'best.pth',map_location='cpu',weights_only=False)
        state=checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        model.load_state_dict(state,strict=True);del checkpoint,state
        params=sum(p.numel() for p in model.parameters())
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        loader=cfg.val_dataloader
        for samples,targets in loader:
            with torch.no_grad():outputs=model(samples.cuda())
            assert torch.isfinite(outputs['pred_logits']).all() and torch.isfinite(outputs['pred_boxes']).all()
            batchids=[int(t['image_id'].item()) for t in targets]
            batchsizes=torch.stack([t['orig_size'] for t in targets]).cuda()
            consume(outputs,batchids,batchsizes)
            logits.append(outputs['pred_logits'].detach().cpu().float().numpy());boxes.append(outputs['pred_boxes'].detach().cpu().float().numpy())
            ids.extend(batchids);sizes.extend(batchsizes.cpu().tolist())
        np.savez_compressed(destination/'pre_topk_candidates.npz',image_ids=np.asarray(ids),original_sizes=np.asarray(sizes),logits=np.concatenate(logits),boxes_cxcywh=np.concatenate(boxes))
    with gzip.open(destination/'predictions_coco.json.gz','wt',encoding='utf-8') as f:json.dump(all_predictions,f,separators=(',',':'))
    gt=COCO(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file']);dt=gt.loadRes(all_predictions)
    evaluator=COCOeval(gt,dt,'bbox');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    names=['AP','AP50','AP75','APsmall','APmedium','APlarge','AR1','AR10','AR100','ARsmall','ARmedium','ARlarge']
    stats={n:float(v) for n,v in zip(names,evaluator.stats)}
    metrics,per_image=fixed_endpoints(all_predictions,gt)
    result={'job_id':job['id'],'configuration':job['configuration'],'cohort':job['cohort'],'seed':job['seed'],'config_sha256':job['config_sha256'],'checkpoint_sha256':sha256((run/'best.pth').read_bytes()).hexdigest(),'precision':'FP32 native validation replay','params_total':params,'metrics_fraction':stats,'metrics_percent':{k:v*100 for k,v in stats.items()},'fixed_threshold':metrics,'controls':control,'prior':a.prior or 'off','independent_test':False,'known_content_overlap_retained':True}
    result['params_trainable']=params_trainable
    (destination/'metrics.json').write_text(json.dumps(result,indent=2))
    (destination/'per_image_counts.json').write_text(json.dumps(per_image,indent=2))
    np.savez_compressed(destination/'coco_precision_recall.npz',precision=evaluator.eval['precision'],recall=evaluator.eval['recall'],scores=evaluator.eval['scores'],category_ids=evaluator.params.catIds,IoU_thresholds=evaluator.params.iouThrs,recall_thresholds=evaluator.params.recThrs)
    if not a.prior and not a.factor_inference:
        rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        assert len(rows)==160 and [r['epoch'] for r in rows]==list(range(160)), 'Incomplete/duplicated training budget'
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        selected=json.loads((run/'best_selection.json').read_text())
        assert selected['epoch_zero_based']==best['epoch']
        assert abs(selected['AP_fraction']-best['test_coco_eval_bbox'][0])<1e-12
        result['best_epoch_zero_based']=best['epoch'];result['best_epoch_one_based']=best['epoch']+1
        result['final_epoch_zero_based']=159
        result['logged_best_metrics_fraction']=dict(zip(names,best['test_coco_eval_bbox']))
        result['replay_minus_logged_AP_points']=100*(stats['AP']-best['test_coco_eval_bbox'][0])
        assert abs(result['replay_minus_logged_AP_points'])<.02, 'Best checkpoint replay differs by >=0.02 AP points; review required'
        (run/'metrics_summary.json').write_text(json.dumps(result,indent=2))
    print('CONTROLLED_EVALUATION_COMPLETE',job['id'],stats['AP'],flush=True)

if __name__=='__main__':main()
