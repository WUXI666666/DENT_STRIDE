"""Actual CMFA/HOOA feature changes and same-checkpoint factor-fusion switches.

No training, checkpoint writes, source edits, or Grad-CAM. Mobile tau=.50,
seed 2, original validation transform and native top-300 postprocessor.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRAIN = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_configs_multiseed_2026-10-11')
JOB_PATH = TRAIN / 'jobs/mobile_proto_k4_tau0p50_seed2.json'
JOB = json.loads(JOB_PATH.read_text())
sys.path.insert(0, JOB['implementation'])
sys.path.insert(0, str(TRAIN))
from controlled_runtime import install_controls
from engine.core import YAMLConfig

CASES = [182, 155, 139]


def rms(x):
    return x.detach().float().square().mean(1).sqrt()[0].cpu().numpy()


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    install_controls(JOB['controls'])
    cfg = YAMLConfig(JOB['config'], device=str(device), use_amp=False)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 0
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 1
    model = cfg.model.to(device).eval()
    checkpoint = Path(JOB['output_dir']) / 'best.pth'
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = raw['ema']['module'] if 'ema' in raw else raw['model']
    model.load_state_dict(state, strict=True)
    del raw, state
    post = cfg.postprocessor.to(device).eval()
    annotation_path = Path(cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
    annotations = json.loads(annotation_path.read_text())
    records = {i['id']: i for i in annotations['images']}
    dataset = cfg.val_dataloader.dataset
    cmfa = model.encoder.top_down_fusions[1]
    hooa = model.encoder.hooa_attention[0]
    decoder = model.decoder.decoder
    assert isinstance(cmfa.out_proj, torch.nn.Identity), 'Need equal-channel cat reference'
    assert not decoder.use_uni_gate_infer
    names = ['factor_alpha_quad', 'factor_alpha_tooth', 'factor_alpha_dent', 'fdi_factor_logit_scale']
    alphas = {n: float(getattr(decoder, n)) for n in names}
    assert [alphas[n] for n in names[:3]] == [.10, .05, .03]
    assert model.decoder.prototype_fusion_weight == 0
    cache = {}
    enabled = False
    handles = []

    def cmfa_before(module, inputs):
        if enabled:
            cache['CMFA_A'] = inputs[0].detach()
            cache['CMFA_B'] = inputs[1].detach()

    def cmfa_after(module, inputs, output):
        if enabled:
            cache['CMFA_after'] = output.detach()

    def hooa_before(module, inputs):
        if enabled:
            cache['HOOA_before'] = inputs[0].detach()

    def hooa_after(module, inputs, output):
        if enabled:
            cache['HOOA_after'] = output.detach()

    handles += [cmfa.register_forward_pre_hook(cmfa_before), cmfa.register_forward_hook(cmfa_after),
                hooa.register_forward_pre_hook(hooa_before), hooa.register_forward_hook(hooa_after)]
    results = {'off': [], 'on': []}
    candidate_logits = {'off': [], 'on': []}
    boxes = []
    ids = []
    checks = []
    feature_stats = []
    sample_for_check = None
    hooked_logits_for_check = None
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample, target = dataset[index]
            iid = int(target['image_id'])
            sample = sample.as_subclass(torch.Tensor).unsqueeze(0).to(device)
            assert list(sample.shape) == [1, 3, 1024, 1024]
            enabled = iid in CASES
            cache.clear()
            on = model(sample)
            factor_cache = dict(cache) if enabled else None
            enabled = False
            for name in names:
                setattr(decoder, name, 0.)
            off = model(sample)
            for name, value in alphas.items():
                setattr(decoder, name, value)
            max_box_error = float((on['pred_boxes'] - off['pred_boxes']).abs().max())
            assert max_box_error == 0, (iid, max_box_error)
            # Disabling a nonpositive logit correction cannot lower any class score.
            max_positive = float((on['pred_logits'] - off['pred_logits']).max())
            assert max_positive < 3e-6, (iid, max_positive)
            info = records[iid]
            size = torch.tensor([[info['width'], info['height']]], device=device)
            for mode, outputs in [('on', on), ('off', off)]:
                rows = post(outputs, size)[0]
                # Match the native flatten-class top-k to retain the query identity.
                scores, flat = torch.topk(outputs['pred_logits'].sigmoid().flatten(1), post.num_top_queries, dim=-1)
                assert torch.equal(scores[0], rows['scores'])
                assert torch.equal(flat[0] % 52, rows['labels'])
                for lab, box, score, query in zip(rows['labels'].cpu().tolist(), rows['boxes'].cpu().tolist(),
                                                 rows['scores'].cpu().tolist(), (flat[0] // 52).cpu().tolist()):
                    x1, y1, x2, y2 = box
                    results[mode].append({'image_id': iid, 'category_id': lab, 'bbox': [x1, y1, x2-x1, y2-y1],
                                          'score': score, 'query_id': query})
                candidate_logits[mode].append(outputs['pred_logits'][0].cpu().numpy())
            ids.append(iid)
            boxes.append(on['pred_boxes'][0].cpu().numpy())
            checks.append({'image_id': iid, 'max_box_difference': max_box_error,
                           'maximum_on_minus_off_logit': max_positive})
            if factor_cache:
                feature = {}
                case_info = {'image_id': iid, 'image_record': info, 'modules': {}}
                before_cmfa = torch.cat([factor_cache['CMFA_A'], factor_cache['CMFA_B']], 1)
                for module, before, after in [('CMFA', before_cmfa, factor_cache['CMFA_after']),
                                               ('HOOA', factor_cache['HOOA_before'], factor_cache['HOOA_after'])]:
                    feature[module+'_before'] = rms(before)
                    feature[module+'_after'] = rms(after)
                    feature[module+'_delta_RMS'] = feature[module+'_after']-feature[module+'_before']
                    feature[module+'_true_update_RMS'] = rms(after-before)
                    case_info['modules'][module] = {'input_shape': list(before.shape),
                        'output_shape': list(after.shape), 'relative_update_L2': float((after-before).norm()/before.norm()),
                        'before_RMS_mean': float(feature[module+'_before'].mean()),
                        'after_RMS_mean': float(feature[module+'_after'].mean())}
                feature['CMFA_input_A'] = rms(factor_cache['CMFA_A'])
                feature['CMFA_input_B'] = rms(factor_cache['CMFA_B'])
                np.savez_compressed(HERE/f'features_image_{iid}.npz', **feature)
                feature_stats.append(case_info)
                print('CAPTURE', json.dumps(case_info), flush=True)
                if iid == 182:
                    sample_for_check = sample.clone()
                    hooked_logits_for_check = on['pred_logits'].clone()
            if (index+1) % 25 == 0:
                print('PROGRESS', index+1, '/', len(dataset), 'seconds', round(time.time()-started, 1), flush=True)
        for handle in handles:
            handle.remove()
        plain = model(sample_for_check)
        invariance = float((plain['pred_logits']-hooked_logits_for_check).abs().max())
        assert invariance == 0, invariance
    for mode in ['on', 'off']:
        with gzip.open(HERE/f'factor_fusion_{mode}_predictions.json.gz', 'wt') as stream:
            json.dump(results[mode], stream)
    np.savez_compressed(HERE/'same_weight_factor_candidates.npz', image_ids=np.array(ids),
                        logits_on=np.stack(candidate_logits['on']), logits_off=np.stack(candidate_logits['off']),
                        boxes_cxcywh=np.stack(boxes))
    provenance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'configuration': JOB['configuration'], 'seed': 2, 'prototype_temperature': .50,
                   'config': JOB['config'], 'annotation': str(annotation_path),
                   'annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                   'image_root': cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'],
                   'precision': 'FP32; no AMP; TF32 disabled', 'device': str(device),
                   'CMFA_module': 'encoder.top_down_fusions.1', 'HOOA_module': 'encoder.hooa_attention.0',
                   'CMFA_before_definition': 'Concatenation of the two aligned 256-channel inputs, in the same order as native output',
                   'heatmap_definition': 'sqrt(mean_c(feature^2))',
                   'change_map_definition': 'RMS(after)-RMS(before); signed response-magnitude change, not feature-vector difference',
                   'true_update_map_definition': 'sqrt(mean_c((after-before)^2))',
                   'factor_comparison': 'Same trained weights, inference-score fusion off versus on; not training without factor heads',
                   'factor_fusion_on': alphas, 'factor_fusion_off': {k: 0. for k in names},
                   'postprocessing': 'Native global query-class top-300, no added NMS; display score>=0.30',
                   'validation_images': len(ids), 'images_evaluated': ids,
                   'feature_cases': feature_stats, 'per_image_switch_checks': checks,
                   'passive_hook_prediction_max_difference': invariance, 'elapsed_seconds': time.time()-started}
    (HERE/'capture_provenance.json').write_text(json.dumps(provenance, indent=2))
    print('COMPLETE', round(time.time()-started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
