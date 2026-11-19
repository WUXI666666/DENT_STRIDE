#!/usr/bin/env python3
"""Real full-batch GPU stress probes plus a frozen preflight acceptance record."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from prepare_migration import HERE, ROOT, SELECTED_TASKS, sha256, write_json, protected_files
from run_experiment import verify_frozen, install_runtime, now


def verify_batch(images, targets):
    assert images.shape[1:] == (3, 640, 640), images.shape
    assert torch.isfinite(images).all()
    for target in targets:
        boxes, labels = target['boxes'], target['labels']
        assert len(boxes) == len(labels) == len(target['area']) == len(target['iscrowd'])
        assert torch.isfinite(boxes).all() and torch.isfinite(target['area']).all()
        assert (boxes >= 0).all() and (boxes <= 1).all() and (boxes[:, 2:] > 0).all()
        assert torch.allclose(target['area'], boxes[:, 2]*boxes[:, 3]*640*640, rtol=1e-4, atol=0.1)


def probe(task, gpu, memory_limit_mib):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        memory_limit_mib*1024**2/torch.cuda.get_device_properties(0).total_memory, device=0)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    frozen = verify_frozen(protocol, task, require_preflight=False)
    YAMLConfig, dist_utils, SOLVERS, install_modules, audit_model = install_runtime()
    os.chdir(protocol['upstream'])
    dist_utils.setup_distributed(seed=91)
    output = HERE / 'probes' / task
    output.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(frozen['config'], seed=91, device='cuda', use_amp=True,
                     output_dir=str(output), summary_dir=str(output / 'summary'),
                     train_dataloader={'num_workers': 0}, val_dataloader={'num_workers': 0})
    model = install_modules(cfg, frozen['modules'])
    schema = json.loads((ROOT / 'profiles' / f"deim_s_{frozen['modules']['cohort']}.json").read_text())
    audit = audit_model(model, schema['parameter_schema'], frozen['modules'])
    write_json(output / 'architecture_audit.json', audit)
    model = model.cuda()
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    ema = cfg.ema.to('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    train = cfg.train_dataloader
    assert train.batch_size == 8 and train.collate_fn.scales is None
    counts = Counter(row['image_id'] for row in train.dataset.coco.dataset['annotations'])
    ids_to_indices = {image_id: index for index, image_id in enumerate(train.dataset.ids)}
    indices = [ids_to_indices[image_id] for image_id, _ in counts.most_common(8)]
    assert len(indices) == 8
    reports = []

    def train_step(images, cpu_targets, label):
        verify_batch(images, cpu_targets)
        images = images.cuda()
        targets = [{key: value.cuda() for key, value in target.items()} for target in cpu_targets]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            loss = sum(losses.values())
        assert torch.isfinite(loss), (label, losses)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        extra_names = set(audit['additional_parameter_schema'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for name, p in model.named_parameters() if name in extra_names)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        assert torch.isfinite(norm)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        torch.cuda.synchronize()
        reports.append(dict(phase=label, batch_size=8, input_size=[640,640],
                            gt_counts=[len(target['labels']) for target in targets],
                            actual_dn_queries=outputs['dn_meta']['dn_num_split'][0],
                            loss=float(loss.detach()), grad_norm=float(norm.detach()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            passed=True))
        print(json.dumps(reports[-1]), flush=True)
        del outputs, losses, loss, norm, images, targets
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    for epoch in [4, 70, 130]:
        random.seed(900+epoch)
        np.random.seed(900+epoch)
        torch.manual_seed(900+epoch)
        train.set_epoch(epoch)
        train.collate_fn.mixup_prob = 1.0 if epoch == 4 else 0.5
        images, targets = train.collate_fn([train.dataset[index] for index in indices])
        train_step(images, targets, f'epoch{epoch}_real_max_count_sources')
        if epoch == 4 and frozen['modules']['cohort'] == 'dentex':
            dense_count = 8*max(counts.values())
            assert dense_count <= cfg.model.decoder.num_queries, dense_count
            dense_targets = []
            for target in targets:
                repeated = dict(target)
                n = len(target['labels'])
                assert n > 0
                selection = torch.arange(dense_count) % n
                for name in ['boxes', 'labels', 'area', 'iscrowd', 'mixup']:
                    if name in target:
                        repeated[name] = target[name][selection].clone()
                dense_targets.append(repeated)
            train_step(images, dense_targets, f'synthetic_dense_upper_bound_{dense_count}_GT')
            del dense_targets
        del images, targets
    val = cfg.val_dataloader
    assert val.batch_size == 16
    val.set_epoch(0)
    images, targets = val.collate_fn([val.dataset[index] for index in range(16)])
    assert tuple(images.shape) == (16,3,640,640)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = ema.module(images.cuda())  # Native validation is FP32, not AMP.
        assert torch.isfinite(outputs['pred_boxes']).all() and torch.isfinite(outputs['pred_logits']).all()
        results = cfg.postprocessor.cuda()(outputs, torch.stack([t['orig_size'] for t in targets]).cuda())
        assert len(results) == 16 and all(len(result['scores']) == 300 for result in results)
    torch.cuda.synchronize()
    reports.append(dict(phase='validation_FP32', batch_size=16, input_size=[640,640],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), passed=True))
    report = dict(task=task, passed=True, phases=reports, updated_at=now(),
                  protocol_sha256=sha256(HERE / 'protocol.json'),
                  memory_limit_mib=memory_limit_mib,
                  maximum_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in reports),
                  maximum_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in reports))
    write_json(output / 'probe_result.json', report)
    print(json.dumps(report, indent=2), flush=True)
    dist_utils.cleanup()


def finalize():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    protocol_hash = sha256(HERE / 'protocol.json')
    for task in SELECTED_TASKS:
        verify_frozen(protocol, task, require_preflight=False)
    aug = json.loads((HERE / 'augmentation_validation.json').read_text())
    assert aug['passed']
    queue = json.loads((HERE / 'queue_validation.json').read_text())
    assert queue['passed']
    configs = json.loads((HERE / 'config_audit.json').read_text())
    assert configs['passed']
    probes, smokes = {}, {}
    for task in SELECTED_TASKS:
        probe_result = json.loads((HERE / 'probes' / task / 'probe_result.json').read_text())
        assert probe_result['passed'] and probe_result['protocol_sha256'] == protocol_hash
        probes[task] = probe_result
        output = HERE / 'smoke_runs' / task
        assert (output / 'COMPLETED').is_file()
        summary = json.loads((output / 'metrics_summary.json').read_text())
        verified = json.loads((output / 'verified_coco_metrics.json').read_text())
        assert summary['smoke_only'] and summary['epochs_completed'] == 3
        assert summary['stage1_RAM_restores'] >= 1
        assert verified['matches_selected_checkpoint']
        assert sha256(output / 'best.pth') == summary['checkpoint_sha256'] == verified['checkpoint_sha256']
        assert list(output.rglob('*.pth')) == [output / 'best.pth']
        smokes[task] = dict(passed=True, epochs=3, stage1_RAM_restores=summary['stage1_RAM_restores'],
                           config_sha256=summary['config_sha256'],
                           peak_reserved_bytes=summary['peak_cuda_reserved_bytes'],
                           one_best_checkpoint=True, verified_best_checkpoint=True)
    before = json.loads((HERE / 'protected_files_before.json').read_text())
    assert before == protected_files(), 'Protected original source/config/manuscript changed'
    peak = max([r['maximum_peak_reserved_bytes'] for r in probes.values()]
               + [r['peak_reserved_bytes'] for r in smokes.values()])
    report = dict(passed=True, verified_at=now(), protocol_sha256=protocol_hash,
                  source_sha256=protocol['source_sha256'], config_audit=True,
                  augmentation_validation=aug, queue_validation=queue,
                  gpu_full_batch_probes=probes, three_epoch_solver_smokes=smokes,
                  recommended_min_free_gpu_mib=max(9216, math.ceil(peak/1024**2)+2048),
                  original_code_configs_manuscripts_unchanged=True,
                  limitations=['Shared GPU performance is not an efficiency benchmark',
                               'Only global best checkpoint is persistent; exact interrupted-run resume is unsupported',
                               'Original upstream EMA decay closure and validation-guided rollback retained identically'])
    write_json(HERE / 'preflight_validation.json', report)
    print(json.dumps({'passed':True, 'recommended_min_free_gpu_mib':report['recommended_min_free_gpu_mib']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=SELECTED_TASKS)
    parser.add_argument('--gpu', type=int, default=0, choices=[0])
    parser.add_argument('--memory-limit-mib', type=int, default=9216)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        if not args.task:
            parser.error('--task required for a GPU probe')
        probe(args.task, args.gpu, args.memory_limit_mib)
