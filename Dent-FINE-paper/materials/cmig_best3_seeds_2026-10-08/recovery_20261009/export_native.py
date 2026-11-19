"""Export real predictions by observing the exact frozen native EMA evaluation.

This sidecar never changes model, training, checkpoint, or native evaluator code.
It replaces the independent reconstructed-model export path, not its checks.
"""
from pathlib import Path
import argparse
import gzip
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    switches = parser.add_mutually_exclusive_group()
    switches.add_argument('--prior', choices=['none', 'center', 'shape', 'both'])
    switches.add_argument('--factor-inference', choices=['on', 'off'])
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text())
    run = Path(job['output_dir'])
    mode = ('prior_'+args.prior if args.prior else 'factor_'+args.factor_inference if args.factor_inference else 'default')
    destination = run/('replay_prior_'+args.prior if args.prior else 'replay_factor_'+args.factor_inference if args.factor_inference else 'evaluation')
    capture_dir = run/('native_candidate_export_'+mode)
    assert not capture_dir.exists(), 'Native export already exists; review before retry'
    native_path = run/(f'native_prior_{args.prior}_metrics.json' if args.prior else f'native_factor_{args.factor_inference}_metrics.json' if args.factor_inference else 'native_coco_metrics.json')
    native = json.loads(native_path.read_text())
    sys.path.insert(0, str(ROOT))
    replay = load_module('frozen_native_export_replay', ROOT/'evaluate_native.py')
    endpoint_module = load_module('frozen_secondary_endpoints', ROOT/'evaluate_controlled.py')
    before = {name: replay.sha(run/name) for name in ['best.pth','log.txt']}
    config_before = replay.sha(job['config'])
    assert before['best.pth']==native['checkpoint_sha256']
    assert config_before==job['config_sha256']==native['config_sha256']
    assert native['verified_best_checkpoint'] and native['authoritative_native_evaluation'] and native['evaluated_ema']
    rows, best = replay.validate_log(run)

    sys.path.insert(0, job['implementation'])
    import torch
    import engine
    from engine.solver import det_solver
    from engine.data.dataset.coco_eval import convert_to_xywh
    assert Path(engine.__file__).resolve().parent==Path(job['implementation'])/'engine'
    original_evaluate = det_solver.evaluate
    collected = {'logits': [], 'boxes': [], 'ids': [], 'sizes': [], 'predictions': []}
    observed = {}

    class ObservedLoader:
        def __init__(self, loader):
            self.loader=loader
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for samples, targets in self.loader:
                observed['ids']=[int(t['image_id'].item()) for t in targets]
                observed['sizes']=torch.stack([t['orig_size'] for t in targets]).tolist()
                yield samples, targets

    def observe_native_evaluation(model, criterion, postprocessor, loader, evaluator, device):
        def capture_candidates(module, inputs, outputs):
            assert torch.isfinite(outputs['pred_logits']).all() and torch.isfinite(outputs['pred_boxes']).all()
            collected['logits'].append(outputs['pred_logits'].detach().cpu().float().numpy())
            collected['boxes'].append(outputs['pred_boxes'].detach().cpu().float().numpy())
            collected['ids'].extend(observed['ids'])
            collected['sizes'].extend(observed['sizes'])
        def capture_predictions(module, inputs, results):
            for iid, result in zip(observed['ids'],results):
                boxes = convert_to_xywh(result['boxes']).tolist()
                for label,score,box in zip(result['labels'].tolist(),result['scores'].tolist(),boxes):
                    collected['predictions'].append({'image_id':iid,'category_id':int(label),'bbox':box,'score':float(score)})
        handles=[model.register_forward_hook(capture_candidates),postprocessor.register_forward_hook(capture_predictions)]
        try:
            result=original_evaluate(model,criterion,postprocessor,ObservedLoader(loader),evaluator,device)
        finally:
            for handle in handles:
                handle.remove()
        observed['coco_gt']=evaluator.coco_gt
        payload=evaluator.coco_eval['bbox']
        observed['precision']={name:payload.eval[name] for name in ['precision','recall','scores']}
        observed['params']=payload.params
        observed['stats']=[float(v) for v in result[0]['coco_eval_bbox']]
        return result

    det_solver.evaluate=observe_native_evaluation
    replay.evaluation_directory=lambda *_a, **_k: capture_dir
    replay.native_child(job, args.prior, args.factor_inference)
    assert before=={name: replay.sha(run/name) for name in before}, 'Weight or training log changed'
    assert replay.sha(job['config'])==config_before, 'Training configuration changed'
    values=dict(zip(replay.METRIC_NAMES,observed['stats']))
    differences={name:values[name]-native['metrics_fraction'][name] for name in replay.VERIFIED_METRICS}
    assert all(abs(v)<=replay.REPLAY_ABS_TOLERANCE for v in differences.values()), differences
    ids=np.asarray(collected['ids'])
    gt=observed['coco_gt']
    assert len(ids)==len(gt.imgs) and len(set(ids.tolist()))==len(gt.imgs)
    assert set(ids.tolist())==set(gt.imgs)
    fixed, per_image=endpoint_module.fixed_endpoints(collected['predictions'],gt)
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'pre_topk_candidates.npz',image_ids=ids,
                        original_sizes=np.asarray(collected['sizes']),
                        logits=np.concatenate(collected['logits']),boxes_cxcywh=np.concatenate(collected['boxes']))
    with gzip.open(destination/'predictions_coco.json.gz','wt',encoding='utf-8') as stream:
        json.dump(collected['predictions'],stream,separators=(',',':'))
    (destination/'per_image_counts.json').write_text(json.dumps(per_image,indent=2))
    parameters=observed['params']
    np.savez_compressed(destination/'coco_precision_recall.npz',**observed['precision'],
                        category_ids=parameters.catIds,IoU_thresholds=parameters.iouThrs,
                        recall_thresholds=parameters.recThrs)
    controls=replay.effective_job_controls(job,args.factor_inference)
    result={'job_id':job['id'],'configuration':job['configuration'],'cohort':job['cohort'],'seed':job['seed'],
            'config_sha256':config_before,'checkpoint_sha256':before['best.pth'],
            'precision':'native FP32 EMA evaluation with passive forward hooks; no reconstructed standalone model',
            'params_total':native['params_total'],'params_trainable':native['params_trainable'],
            'metrics_fraction':native['metrics_fraction'],'metrics_percent':native['metrics_percent'],
            'export_metrics_fraction':values,'export_minus_native_AP_points':differences['AP']*100,
            'native_export_minus_reference_fraction':differences,
            'fixed_threshold':fixed,'controls':controls,'prior':args.prior or job.get('inference_priors','off'),
            'independent_test':False,'known_content_overlap_retained':True,
            'authoritative_native_evaluation':True,
            'primary_metric_source':native_path.name+': exact frozen native evaluator and EMA model',
            'fixed_threshold_metric_source':'predictions observed during the same native full-candidate forward; score>=0.30, IoU>=0.50; localization matching',
            'prediction_export_provenance':str(capture_dir/'captured_metrics.json'),
            'prediction_export_script_sha256':replay.sha(Path(__file__)),
            'prediction_export_recovery_version':'2026-10-09 native-forward observation',
            'log_sha256':before['log.txt']}
    if not args.prior and not args.factor_inference:
        result.update(best_epoch_zero_based=best['epoch'],best_epoch_one_based=best['epoch']+1,
                      final_epoch_zero_based=159,logged_best_metrics_fraction=dict(zip(replay.METRIC_NAMES,best['test_coco_eval_bbox'])),
                      replay_minus_logged_AP_points=100*(values['AP']-best['test_coco_eval_bbox'][0]))
        assert abs(result['replay_minus_logged_AP_points'])<=100*replay.REPLAY_ABS_TOLERANCE
        replay.atomic_json(run/'metrics_summary.json',result)
    replay.atomic_json(destination/'metrics.json',result)
    provenance={'job':job['id'],'native_reference':str(native_path),'native_reference_sha256':replay.sha(native_path),
                'checkpoint_sha256':before['best.pth'],'log_sha256':before['log.txt'],'config_sha256':config_before,
                'script_sha256':replay.sha(Path(__file__)),'native_export_minus_reference_fraction':differences,
                'retained_original_training_and_native_validation':True,
                'files':{name:replay.sha(destination/name) for name in ['pre_topk_candidates.npz','predictions_coco.json.gz','coco_precision_recall.npz','per_image_counts.json','metrics.json']}}
    replay.atomic_json(capture_dir/'export_provenance.json',provenance)
    print('NATIVE_PREDICTION_EXPORT_VERIFIED',job['id'],mode,values['AP']*100,flush=True)


if __name__=='__main__':
    main()
