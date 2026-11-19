#!/usr/bin/env python3
"""Freeze input-1024 experiments with the unchanged Mosaic-640 recipe.

All six configurations are prepared. Only the three previously selected recipes
enter the authorized queue by default; --with-baselines also schedules their
new, augmentation-matched Plain DEIM controls. Historical metrics are references,
never substitutes for those matched controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = Path('/home/wuxi/Dent-FINE')
OLD = ROOT / 'module_migration_mosaic640_2026-09-12'
LEGACY = ROOT / 'module_migration_2026-09-11'
MODEL_INPUT_SIZE = 1024
SELECTED_TASKS = ('phone_dualopt', 'dentex_spatial_shape', 'dslr_cmfa_dualopt')
TASKS = {
    'phone_plain': {'cohort': 'phone', 'cmfa': False, 'structured': False, 'spatial_shape': False, 'dualopt': False},
    'phone_dualopt': {'cohort': 'phone', 'cmfa': False, 'structured': False, 'spatial_shape': False, 'dualopt': True},
    'dentex_plain': {'cohort': 'dentex', 'cmfa': False, 'structured': False, 'spatial_shape': False, 'dualopt': False},
    'dentex_spatial_shape': {'cohort': 'dentex', 'cmfa': False, 'structured': True, 'spatial_shape': True, 'dualopt': False},
    'dslr_plain': {'cohort': 'dslr', 'cmfa': False, 'structured': False, 'spatial_shape': False, 'dualopt': False},
    'dslr_cmfa_dualopt': {'cohort': 'dslr', 'cmfa': True, 'structured': False, 'spatial_shape': False, 'dualopt': True},
}
DUALOPT_SETTINGS = {
    'type': 'DualOptAdamW', 'rollback_weight': 0.005, 'realtime_decay': 0.01,
    'layer_decay_power': 1.5, 'layer_decay_min': 0.5, 'layer_decay_max': 1.5,
}
FACTOR_LOSSES = ('quad', 'tooth', 'dent', 'proto')
FACTOR_WEIGHTS = {'loss_quad': 0.10, 'loss_tooth': 0.10, 'loss_dent': 0.05, 'loss_proto': 0.05}
MOSAIC = {
    'type': 'Mosaic', 'output_size': 640, 'rotation_range': 10,
    'translation_range': [0.1, 0.1], 'scaling_range': [0.5, 1.5],
    'probability': 1.0, 'fill_value': 0, 'use_cache': False,
    'max_cached_images': 50, 'random_pop': True,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def protected_files():
    """Hash protected sources, configurations and manuscripts, not image blobs."""
    files = set()
    for folder in ('engine', 'configs', 'tools'):
        files.update(p for p in (LOCAL / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yml', '.yaml', '.sh'))
    files.update(p for p in LOCAL.glob('*.py') if p.is_file())
    files.update(p for p in LOCAL.parent.joinpath('Dent-FINE-paper').glob('*.docx') if p.is_file())
    files.update(p for p in (ROOT / 'configs').rglob('*') if p.is_file())
    for previous in (OLD, LEGACY):
        files.update(p for p in (previous / 'configs').rglob('*') if p.is_file())
        files.update(p for p in previous.glob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    files.update(p for p in (ROOT / 'scripts').glob('*.py') if p.is_file())
    files.update(p for p in (ROOT / 'protocol.json', ROOT / 'dataset_evidence.json',
                             ROOT / 'verified_backbone_sources.json') if p.is_file())
    upstream = ROOT / 'upstream/DEIM'
    tracked = subprocess.check_output(['git', '-C', str(upstream), 'ls-files', '-z'])
    files.update(upstream / name.decode() for name in tracked.split(b'\0')
                 if name and (upstream / name.decode()).is_file())
    return {str(path): sha256(path) for path in sorted(files)}


def only_op(ops, name):
    matches = [op for op in ops if op.get('type') == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def apply_shared_recipe(config):
    """Identical data/runtime settings for both models within every cohort."""
    config.update(seed=0, epoches=160, eval_spatial_size=[MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                  no_aug_epoch=30, use_amp=True, use_ema=True)
    assert not config.get('tuning') and not config.get('resume'), 'Fresh backbone initialization required'
    assert config['HGNetv2']['pretrained'] is True
    assert config['DFINETransformer']['num_denoising'] == 100
    train = config['train_dataloader']
    val = config['val_dataloader']
    train['total_batch_size'], val['total_batch_size'] = 8, 16
    transforms = train['dataset']['transforms']
    mosaic = only_op(transforms['ops'], 'Mosaic')
    mosaic.clear()
    mosaic.update(copy.deepcopy(MOSAIC))
    only_op(transforms['ops'], 'RandomIoUCrop')['p'] = 0.5
    only_op(transforms['ops'], 'Resize')['size'] = [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
    for op in transforms['ops']:
        if op['type'] == 'SanitizeBoundingBoxes':
            op['type'] = 'AlignedSanitizeBoundingBoxes'
    assert sum(op['type'] == 'AlignedSanitizeBoundingBoxes' for op in transforms['ops']) == 2
    transforms['mosaic_prob'] = 1.0
    transforms['policy']['name'] = 'stop_epoch'
    transforms['policy']['epoch'] = [4, 70, 130]
    assert set(transforms['policy']['ops']) == {
        'Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop'}
    train['collate_fn'].update(type='AlignedBatchImageCollateFunction',
                               base_size=MODEL_INPUT_SIZE, base_size_repeat=None,
                               stop_epoch=130, mixup_prob=0.5, mixup_epochs=[4, 80])
    # Validation uses the unchanged native, non-multiscale default collator.
    assert val['collate_fn'] == {'type': 'BatchImageCollateFunction'}
    only_op(val['dataset']['transforms']['ops'], 'Resize')['size'] = [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
    assert config['lrsheduler'] == 'flatcosine'
    assert config['flat_epoch'] == 70
    assert config['scaler']['enabled'] is True
    assert config['PostProcessor']['num_top_queries'] == 300
    assert not any(op['type'] == 'RandomHorizontalFlip' for op in transforms['ops'])
    return config


def verify_resolution_only_change(task, previous, current):
    """Allow only the four input sizes and isolated run/prior path relocation."""
    normalized = copy.deepcopy(current)
    allowed = ['eval_spatial_size', 'train_dataloader.dataset.transforms.Resize.size',
               'val_dataloader.dataset.transforms.Resize.size',
               'train_dataloader.collate_fn.base_size', 'output_dir']
    assert previous['eval_spatial_size'] == [640, 640], task
    assert current['eval_spatial_size'] == [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], task
    normalized['eval_spatial_size'] = previous['eval_spatial_size']
    for split in ('train', 'val'):
        old_loader, new_loader = previous[f'{split}_dataloader'], normalized[f'{split}_dataloader']
        old_resize = only_op(old_loader['dataset']['transforms']['ops'], 'Resize')
        new_resize = only_op(new_loader['dataset']['transforms']['ops'], 'Resize')
        assert old_resize['size'] == [640, 640], (task, split)
        assert new_resize['size'] == [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], (task, split)
        new_resize['size'] = copy.deepcopy(old_resize['size'])
    assert previous['train_dataloader']['collate_fn']['base_size'] == 640, task
    assert normalized['train_dataloader']['collate_fn']['base_size'] == MODEL_INPUT_SIZE, task
    normalized['train_dataloader']['collate_fn']['base_size'] = 640
    assert previous['output_dir'] == str(ROOT / 'runs/deim_s_paper_modules_mosaic640' / task / 'seed_0'), task
    assert current['output_dir'] == str(ROOT / 'runs/deim_s_paper_modules_input1024' / task / 'seed_0'), task
    normalized['output_dir'] = previous['output_dir']
    if current['paper_modules']['spatial_shape']:
        assert previous['paper_modules']['prior_file'] == str(OLD / 'dentex_train_templates.json'), task
        assert current['paper_modules']['prior_file'] == str(HERE / 'dentex_train_templates.json'), task
        assert current['paper_modules']['prior_sha256'] == previous['paper_modules']['prior_sha256'], task
        normalized['paper_modules']['prior_file'] = previous['paper_modules']['prior_file']
        allowed.append('paper_modules.prior_file (same bytes)')
    assert normalized == previous, (task, 'Unexpected non-resolution difference from frozen 640 configuration')
    return {'passed': True, 'only_input_resolution_and_isolation_paths_changed': True,
            'allowed_differences': allowed, 'mosaic_output_size_unchanged': 640,
            'train_batch_size_unchanged': 8, 'validation_batch_size_unchanged': 16}


def remove_allowed_module_differences(config):
    """Remove only enumerated recipe differences; everything else must match."""
    reduced = copy.deepcopy(config)
    spec = reduced.pop('paper_modules')
    reduced.pop('output_dir')
    if spec['dualopt']:
        assert all(reduced['optimizer'].get(k) == v for k, v in DUALOPT_SETTINGS.items())
        for key in DUALOPT_SETTINGS:
            reduced['optimizer'].pop(key)
        reduced['optimizer']['type'] = 'AdamW'
    if spec['structured']:
        criterion = reduced['DEIMCriterion']
        for loss in FACTOR_LOSSES:
            assert criterion['losses'].count(loss) == 1
            criterion['losses'].remove(loss)
        for name, value in FACTOR_WEIGHTS.items():
            assert criterion['weight_dict'].pop(name) == value
    return reduced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-baselines', action='store_true',
                        help='Authorize all six jobs; otherwise queue only the three selected models')
    args = parser.parse_args()
    generated = [HERE / 'protocol.json', HERE / 'config_audit.json',
                 HERE / 'protected_files_before.json', HERE / 'dentex_train_templates.json']
    generated += [HERE / 'configs' / f'{task}.yml' for task in TASKS]
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise RuntimeError(f'Refusing to overwrite an existing experiment freeze: {existing}')

    base_protocol_path = ROOT / 'protocol.json'
    old_protocol_path = OLD / 'protocol.json'
    base_protocol = json.loads(base_protocol_path.read_text())
    old_protocol = json.loads(old_protocol_path.read_text())
    evidence = json.loads((ROOT / 'dataset_evidence.json').read_text())
    upstream = ROOT / 'upstream/DEIM'
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == base_protocol['models']['deim_s']['commit'] == old_protocol['upstream_commit']
    assert sha256(base_protocol_path) == old_protocol['base_protocol_sha256']
    protected = protected_files()

    frozen_annotations = {}
    for cohort in ('phone', 'dentex', 'dslr'):
        frozen_annotations[cohort] = {}
        for split in ('train', 'val'):
            entry = evidence[cohort][split]
            assert sha256(entry['annotation']) == entry['sha256'], (cohort, split, 'Annotation changed')
            data = json.loads(Path(entry['annotation']).read_text())
            category_map = sorted(data['categories'], key=lambda row: row['id'])
            assert [row['id'] for row in category_map] == list(range(base_protocol['datasets'][cohort]['classes']))
            assert category_map == sorted(entry['category_map'], key=lambda row: row['id'])
            frozen_annotations[cohort][split] = copy.deepcopy(entry)
        assert frozen_annotations[cohort]['train']['category_map'] == frozen_annotations[cohort]['val']['category_map']

    pretrained = Path('/home/wuxi/RT-DETR-main/D-FINE/weight/hgnetv2/PPHGNetV2_B0_stage1.pth')
    verified_weights = json.loads((ROOT / 'verified_backbone_sources.json').read_text())
    verified = next(entry for entry in verified_weights if Path(entry['file']) == pretrained)
    assert verified['exact_official_bytes'] and sha256(pretrained) == verified['sha256']
    old_modules = OLD / 'paper_modules.py'
    old_validation = json.loads((OLD / 'preflight_validation.json').read_text())
    assert old_validation['passed'] and old_validation['protocol_sha256'] == sha256(old_protocol_path)
    assert sha256(old_modules) == old_validation['source_sha256'][str(old_modules)]
    for filename in ('paper_modules.py', 'augmentation_runtime.py'):
        source = OLD / filename
        assert sha256(source) == old_protocol['source_sha256'][str(source)]
        assert sha256(HERE / filename) == sha256(source), (filename, 'Algorithm code must remain byte-identical')

    old_prior_path = Path(old_protocol['runs']['dentex_spatial_shape']['modules']['prior_file'])
    prior_bytes = old_prior_path.read_bytes()
    assert hashlib.sha256(prior_bytes).hexdigest() == old_protocol['runs']['dentex_spatial_shape']['modules']['prior_sha256']
    prior = json.loads(prior_bytes)
    assert prior['annotation_sha256'] == frozen_annotations['dentex']['train']['sha256']
    assert Path(prior['annotation_file']) == Path(frozen_annotations['dentex']['train']['annotation'])
    assert prior['fdi_label_order'] == old_protocol['runs']['dentex_spatial_shape']['modules']['fdi_codes']
    prior_path = HERE / 'dentex_train_templates.json'

    configurations, runs, resolution_comparisons = {}, {}, {}
    for task, flags in TASKS.items():
        cohort = flags['cohort']
        base_config = ROOT / 'configs' / f'deim_s_{cohort}.yml'
        assert sha256(base_config) == base_protocol['configs'][f'deim_s_{cohort}']['sha256']
        config = apply_shared_recipe(yaml.safe_load(base_config.read_text()))
        for split in ('train', 'val'):
            dataset = config[f'{split}_dataloader']['dataset']
            assert Path(dataset['ann_file']) == Path(frozen_annotations[cohort][split]['annotation'])
            assert Path(dataset['img_folder']) == Path(frozen_annotations[cohort][split]['image_root'])
        assert Path(config['HGNetv2']['local_model_dir']) / pretrained.name == pretrained
        selected_task = next(name for name in SELECTED_TASKS if TASKS[name]['cohort'] == cohort)
        selected_record = old_protocol['runs'][selected_task]
        assert sha256(selected_record['config']) == selected_record['config_sha256']
        is_plain = task.endswith('_plain')
        spec = copy.deepcopy(flags if is_plain else selected_record['modules'])
        assert all(spec[key] == value for key, value in flags.items())
        if spec['dualopt']:
            old_config = yaml.safe_load(Path(selected_record['config']).read_text())
            assert all(old_config['optimizer'][key] == value for key, value in DUALOPT_SETTINGS.items())
            config['optimizer'].update(DUALOPT_SETTINGS)
        if spec['structured']:
            assert not any(name in config['DEIMCriterion']['losses'] for name in FACTOR_LOSSES)
            config['DEIMCriterion']['losses'] += list(FACTOR_LOSSES)
            config['DEIMCriterion']['weight_dict'].update(FACTOR_WEIGHTS)
        if spec['spatial_shape']:
            spec['prior_file'] = str(prior_path)
            spec['prior_sha256'] = hashlib.sha256(prior_bytes).hexdigest()
        output = ROOT / 'runs/deim_s_paper_modules_input1024' / task / 'seed_0'
        assert not output.exists(), f'Fresh run directory required: {output}'
        config['paper_modules'], config['output_dir'] = spec, str(output)
        config_path = HERE / 'configs' / f'{task}.yml'
        matched_plain_output = ROOT / 'runs/deim_s_paper_modules_input1024' / f'{cohort}_plain' / 'seed_0'
        historical = copy.deepcopy(selected_record['historical_reference'])
        for role in ('plain', 'selected'):
            assert sha256(historical[f'{role}_metrics']) == historical[f'{role}_metrics_sha256']
        previous_record = old_protocol['runs'][task]
        previous_path = Path(previous_record['config'])
        assert previous_path == OLD / 'configs' / f'{task}.yml'
        assert sha256(previous_path) == previous_record['config_sha256'], task
        resolution_comparisons[task] = verify_resolution_only_change(
            task, yaml.safe_load(previous_path.read_text()), config)
        resolution_comparisons[task].update(source_config=str(previous_path),
                                            source_config_sha256=sha256(previous_path))
        configurations[task] = config
        runs[task] = {
            'config': str(config_path), 'output': str(output), 'modules': spec,
            'is_plain_control': is_plain,
            'baseline_config': str(HERE / 'configs' / f'{cohort}_plain.yml'),
            'baseline_metrics': str(matched_plain_output / 'metrics_summary.json'),
            'baseline_task': f'{cohort}_plain',
            'source_config': str(base_config), 'source_config_sha256': sha256(base_config),
            'source_selected_config': selected_record['config'],
            'source_selected_config_sha256': selected_record['config_sha256'],
            'source_640_config': str(previous_path),
            'source_640_config_sha256': sha256(previous_path),
            'historical_reference': historical,
        }

    pairs = {}
    for selected in SELECTED_TASKS:
        cohort = TASKS[selected]['cohort']
        plain = f'{cohort}_plain'
        assert remove_allowed_module_differences(configurations[plain]) == remove_allowed_module_differences(configurations[selected]), cohort
        pairs[cohort] = {'plain_task': plain, 'selected_task': selected,
                         'all_non_module_settings_exactly_equal': True,
                         'data_transforms_exactly_equal': configurations[plain]['train_dataloader'] == configurations[selected]['train_dataloader'],
                         'allowed_differences': ['paper_modules', 'output_dir']
                         + (['optimizer type and five DualOpt coefficients'] if TASKS[selected]['dualopt'] else [])
                         + (['four auxiliary losses and their weights'] if TASKS[selected]['structured'] else [])}

    # No generated artifact is written until source, annotation and pair checks pass.
    HERE.mkdir(parents=True, exist_ok=True)
    prior_path.write_bytes(prior_bytes)
    for task, config in configurations.items():
        config_path = Path(runs[task]['config'])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        runs[task]['config_sha256'] = sha256(config_path)
    for task in runs:
        runs[task]['baseline_config_sha256'] = runs[runs[task]['baseline_task']]['config_sha256']
    write_json(HERE / 'protected_files_before.json', protected)
    config_audit = {
        'passed': True, 'pairs': pairs, 'all_six_configs_prepared': True,
        'source_mosaic640_protocol': str(old_protocol_path),
        'source_mosaic640_protocol_sha256': sha256(old_protocol_path),
        'resolution_comparisons_against_640': resolution_comparisons,
        'all_six_configs_differ_from_640_only_in_input_resolution_and_isolation_paths': True,
        'fixed_model_input': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], 'train_multiscale_disabled': True,
        'no_aug_scheduler_aligned_at_epoch_zero_based': 130,
        'all_annotation_hashes_match_prior_evidence': True,
        'prior_reused_byte_for_byte_from_train_only_source': True,
        'mosaic_op_probability': 1.0, 'compose_mosaic_prob': 1.0,
        'crop_probability_when_scheduled': 0.5,
        'sanitize_registration': 'AlignedSanitizeBoundingBoxes; both outer sanitizer stages; same for every model',
        'runtime_forward_augmentation_validation': 'Required separately before training; this is a static configuration audit',
    }
    write_json(HERE / 'config_audit.json', config_audit)
    # The root runner waits until all new source files are complete before freeze.
    # Freeze Python runtime code here; JSON evidence has separate named hashes.
    runtime_sources = set(HERE.glob('*.py'))
    runtime_sources.update((old_modules, ROOT / 'scripts/run_official.py',
                            ROOT / 'scripts/prepare_comparison.py'))
    source_sha256 = {str(path): sha256(path) for path in sorted(runtime_sources)}
    annotation_sha256 = {
        str(Path(entry['annotation'])): entry['sha256']
        for cohort in frozen_annotations.values() for entry in cohort.values()
    }
    queue_order = list(TASKS) if args.with_baselines else list(SELECTED_TASKS)
    write_json(HERE / 'protocol.json', {
        'frozen_at': datetime.now().astimezone().isoformat(),
        'upstream': str(upstream), 'upstream_commit': revision,
        'base_protocol_sha256': sha256(base_protocol_path),
        'source_migration_protocol': str(old_protocol_path),
        'source_migration_protocol_sha256': sha256(old_protocol_path),
        'scope': 'Three selected main configurations plus six prepared configurations; no module sweep',
        'with_baselines': args.with_baselines, 'queue_order': queue_order,
        'runs': runs, 'seeds': [0], 'epochs': 160,
        'dataset_annotations': annotation_sha256, 'dataset_evidence': frozen_annotations,
        'pretrained_backbone': {'path': str(pretrained), 'sha256': sha256(pretrained)},
        'pretrained_evidence': verified, 'source_sha256': source_sha256,
        'config_audit': str(HERE / 'config_audit.json'),
        'config_audit_sha256': sha256(HERE / 'config_audit.json'),
        'initialization': 'Fresh verified official ImageNet HGNetv2-B0 backbone; no detector checkpoint fine-tuning',
        'selection': 'Maximum validation COCO AP; earliest epoch on exact tie',
        'scaffold': 'Official DEIM-D-FINE-S, unchanged native decoder, nearest upsampling and sinusoidal encoder positions',
        'matching': 'Within each cohort, every new Plain/selected setting is identical except the explicit paper modules/losses/DualOpt/SpatialShape',
        'denoising_queries': 100,
        'denoising_note': 'num_denoising=100 is the native grouping target, not an upper bound on the realized number of DN queries; Mosaic/MixUp can increase DN slots',
        'input': 'Fixed 1024x1024 training and validation; unchanged Mosaic output_size=640 resizes each source image before its larger composite is resized to model input',
        'resolution_change': {
            'previous_model_input': [640, 640], 'model_input': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            'mosaic_output_size_unchanged': 640,
            'all_six_configs_verified_against_frozen_640': True,
            'continuation': 'Fresh initialization; do not resume the superseded 640 training or reuse its preflight as 1024 evidence',
            'comparison_caveat': '640 versus 1024 changes input resolution; differences are not pure module effects',
        },
        'batches': {'train': 8, 'validation': 16}, 'amp': True, 'ema': True,
        'augmentation': {
            'Mosaic': MOSAIC, 'RandomIoUCrop': {'type': 'RandomIoUCrop', 'p': 0.5},
            'compose_mosaic_prob': 1.0, 'policy_epochs_zero_based': [4, 70, 130],
            'effective_schedule': [
                '0-3: policy-listed Mosaic/photometric/ZoomOut/IoUCrop disabled',
                '4-69: Mosaic always selected; native Compose skips ZoomOut and IoUCrop on Mosaic samples',
                '70-129: Mosaic disabled; IoUCrop p=0.5 and native ZoomOut/photometric active',
                '130-159: policy-listed augmentation disabled; no_aug_epoch=30 aligns the flat-cosine schedule',
            ],
            'MixUp': {'probability': 0.5, 'epochs_zero_based_start_inclusive_end_exclusive': [4, 80]},
            'sanitizer': 'AlignedSanitizeBoundingBoxes keeps per-instance fields aligned in every configuration',
        },
        'postprocessing': 'Native top-300, no score cutoff or NMS; SpatialShape only in the selected DENTEX run',
        'storage': 'One global best.pth per run; native stage-1 rollback state in RAM only',
        'excluded': ['QueryFusion', 'DySample', 'learned row/column positions', 'query/graph/ROI refiners',
                     'HOOA', 'CEDv2', 'SMWG', 'MPF', 'UCM', 'all other unselected modules'],
        'interpretation': 'New seed-0 development experiments on unchanged internal-validation splits; no guaranteed leading result, independent test, repeated-seed conclusion, or pure module attribution against unmatched historical baselines',
    })
    assert protected_files() == protected, 'Protected prior source or manuscript changed during preparation'
    print(json.dumps({'prepared': str(HERE), 'configured_tasks': list(TASKS),
                      'authorized_queue_order': queue_order, 'config_pairs_verified': True,
                      'prior_reused_without_reestimation': True,
                      'resolution_only_difference_from_640_verified_for_all_six': True}, indent=2))


if __name__ == '__main__':
    main()
