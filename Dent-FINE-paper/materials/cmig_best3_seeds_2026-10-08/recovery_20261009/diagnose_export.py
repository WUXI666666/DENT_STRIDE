"""Read-only candidate diagnosis; never change a checkpoint or training input."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--output-name', default='export_diagnosis.json')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    run = Path(job['output_dir'])
    original = {name: digest(run/name) for name in ['best.pth', 'log.txt', 'native_coco_metrics.json']}
    sys.path.insert(0, job['implementation'])
    import torch
    from engine.core import YAMLConfig
    from engine.data.dataset.coco_eval import convert_to_xywh
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    torch.set_num_threads(4)
    cfg = YAMLConfig(job['config'], device='cuda', use_amp=True)
    post = cfg.postprocessor.cuda()
    native_evaluator = cfg.evaluator
    native_evaluator.cleanup()
    candidates = np.load(run/'evaluation/pre_topk_candidates.npz', allow_pickle=False)
    native, python = [], []
    for start in range(0, len(candidates['image_ids']), 16):
        sl = slice(start, start+16)
        output = {'pred_logits': torch.from_numpy(candidates['logits'][sl]).cuda(),
                  'pred_boxes': torch.from_numpy(candidates['boxes_cxcywh'][sl]).cuda()}
        with torch.no_grad():
            results = post(output, torch.from_numpy(candidates['original_sizes'][sl]).cuda())
        native_evaluator.update({int(iid): result for iid,result in zip(candidates['image_ids'][sl],results)})
        for iid, result in zip(candidates['image_ids'][sl], results):
            for label, score, xyxy, xywh in zip(result['labels'].tolist(), result['scores'].tolist(),
                                               result['boxes'].tolist(), convert_to_xywh(result['boxes']).tolist()):
                base = {'image_id': int(iid), 'category_id': int(label), 'score': float(score)}
                native.append({**base, 'bbox': xywh})
                x1,y1,x2,y2=xyxy
                python.append({**base, 'bbox': [x1,y1,x2-x1,y2-y1]})
    with gzip.open(run/'evaluation/predictions_coco.json.gz', 'rt') as f:
        previous = json.load(f)
    report = {'job': job['id'], 'cached_python_predictions_exactly_match_failed_export': python==previous,
              'bbox_rows_changed_by_float32_tensor_conversion': sum(a['bbox']!=b['bbox'] for a,b in zip(native,python)),
              'before_hashes': original, 'native_record_AP': json.loads((run/'native_coco_metrics.json').read_text())['metrics_percent']['AP']}
    gt = COCO(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
    for name, predictions in [('python_float_subtraction',python), ('native_tensor_subtraction',native)]:
        evaluation = COCOeval(gt, gt.loadRes(predictions), 'bbox')
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
        report[name] = {n: float(evaluation.stats[i])*100 for n,i in [('AP',0),('AP50',1),('AP75',2),('AR100',8)]}
    native_evaluator.synchronize_between_processes()
    native_evaluator.accumulate()
    native_evaluator.summarize()
    report['native_batch_evaluator'] = {n: float(native_evaluator.coco_eval['bbox'].stats[i])*100 for n,i in [('AP',0),('AP50',1),('AP75',2),('AR100',8)]}
    assert original=={name: digest(run/name) for name in original}
    report['checkpoint_and_training_log_unchanged']=True
    destination = Path(__file__).parent/args.output_name
    assert not destination.exists(), 'Do not overwrite an earlier diagnosis'
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    main()
