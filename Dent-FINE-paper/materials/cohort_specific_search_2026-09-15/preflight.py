"""Fresh, discarded smoke-test model for one immutable study configuration."""
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    job = json.loads(Path(sys.argv[1]).read_text())
    implementation = Path(job['implementation_dir'])
    sys.path.insert(0, str(implementation))
    import numpy as np
    import torch
    import engine
    from engine.core import YAMLConfig
    assert Path(engine.__file__).resolve().parent == implementation / 'engine'
    torch.set_num_threads(4)
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg_hash = hashlib.sha256(Path(job['config']).read_bytes()).hexdigest()
    assert cfg_hash == job['config_sha256']
    cfg = YAMLConfig(job['config'])
    size = cfg.yaml_cfg['eval_spatial_size'][0]
    model = cfg.model.cuda()
    criterion = cfg.criterion.cuda()
    postprocessor = cfg.postprocessor.cuda()
    ema = cfg.ema.to(torch.device('cuda'))
    optimizer = cfg.optimizer
    scaler = cfg.scaler
    train_loader, val_loader = cfg.train_dataloader, cfg.val_dataloader
    assert train_loader.batch_size == 8 and val_loader.batch_size == 16
    assert tuple(model.encoder.eval_spatial_size) == (size, size)
    assert tuple(model.decoder.eval_spatial_size) == (size, size)
    expected_arch = 0.1 if job['task'] == 'phone_a07_proto_aux' else 0.25
    assert criterion.weight_dict['loss_arch'] == expected_arch
    expected_params = {'phone_dualopt': 10629283, 'dentex_spatial_shape': 10650249,
                       'dslr_cmfa_dualopt': 11153963, 'phone_a07_proto_aux': 12513162}
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == expected_params[job['task']], (job['task'], n_params)
    dataset = train_loader.dataset
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset.coco.imgToAnns.get(dataset.ids[i], [])), reverse=True)[:8]
    recipe = job['augmentation']
    collate = train_loader.collate_fn
    assert collate.mixup_prob == recipe['mixup']
    if recipe['profile'] == 'legacy640_multiscale':
        assert collate.scales is not None and min(collate.scales) == 480 and max(collate.scales) == 800
    else:
        assert collate.scales is None
    report = {'job_id': job['job_id'], 'seed': seed, 'config_sha256': cfg_hash,
              'implementation': str(implementation), 'gpu': torch.cuda.get_device_name(0),
              'params_trainable': n_params, 'loss_arch_weight': expected_arch,
              'augmentation': recipe, 'tests': [], 'tests_are_sampled_not_full_data_audit': True,
              'preflight_updates_are_discarded': True}
    for epoch in [0, 4, 80, 130]:
        model.train()
        criterion.train()
        train_loader.set_epoch(epoch)
        random.seed(epoch + seed)
        torch.manual_seed(epoch + seed)
        items = [dataset[i] for i in indices]
        # Force a draw below 0.5 for the native batch-level MixUp decision;
        # keep the real configured probability unchanged.
        random.seed(1)
        samples, targets = collate(items)
        assert samples.shape[:2] == (8, 3)
        assert samples.shape[-1] == samples.shape[-2]
        if collate.scales is None or epoch >= collate.stop_epoch:
            assert samples.shape[-2:] == (size, size)
        else:
            assert samples.shape[-1] in collate.scales
        mixup_expected = recipe['mixup'] > 0 and 4 <= epoch < 80
        assert all(('mixup' in t) == mixup_expected for t in targets)
        for target in targets:
            assert len(target['boxes']) == len(target['labels'])
            assert torch.isfinite(target['boxes']).all()
            assert (target['boxes'][:, 2:] > 0).all()
            if len(target['labels']):
                assert target['labels'].min() >= 0
                assert target['labels'].max() < cfg.yaml_cfg['num_classes']
        samples = samples.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(12):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', cache_enabled=True):
                outputs = model(samples, targets=targets)
            assert torch.isfinite(outputs['pred_boxes']).all()
            assert torch.isfinite(outputs['pred_logits']).all()
            with torch.autocast(device_type='cuda', enabled=False):
                losses = criterion(outputs, targets, epoch=epoch, step=attempt,
                    global_step=epoch * len(train_loader) + attempt, epoch_step=len(train_loader))
            loss = sum(losses.values())
            assert torch.isfinite(loss), losses
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
            if torch.isfinite(norm):
                break
            print(json.dumps({'epoch': epoch, 'native_amp_retry': attempt + 1,
                              'scale': scaler.get_scale()}), flush=True)
            del outputs, losses, loss
        else:
            raise RuntimeError('No finite-gradient update after 12 native GradScaler adaptation attempts')
        torch.cuda.synchronize()
        row = {'epoch_zero_based': epoch, 'input_shape': list(samples.shape),
               'target_counts': [len(t['labels']) for t in targets], 'mixup_exercised': mixup_expected,
               'loss': float(loss.detach()), 'loss_arch': float(losses.get('loss_arch', 0)),
               'gradient_norm': float(norm), 'amp_retries': attempt,
               'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
               'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30}
        report['tests'].append(row)
        print(json.dumps(row), flush=True)
        del samples, targets, items, outputs, losses, loss
        gc.collect()
        torch.cuda.empty_cache()
    assert any(t['loss_arch'] > 0 for t in report['tests'])
    ema.module.eval()
    val_loader.set_epoch(0)
    items = [val_loader.dataset[i] for i in range(16)]
    samples, targets = val_loader.collate_fn(items)
    assert list(samples.shape) == [16, 3, size, size]
    samples = samples.cuda()
    sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = ema.module(samples)
        predictions = postprocessor(outputs, sizes)
    assert len(predictions) == 16
    assert all(torch.isfinite(p['scores']).all() and torch.isfinite(p['boxes']).all() for p in predictions)
    report['validation'] = {'input_shape': list(samples.shape),
        'native_fp32_eval': True, 'postprocessor_class': type(postprocessor).__name__,
        'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30}
    report['passed'] = True
    report['completed_at'] = datetime.now().astimezone().isoformat()
    (Path(job['output_dir']) / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'job': job['job_id']}), flush=True)


if __name__ == '__main__':
    main()
