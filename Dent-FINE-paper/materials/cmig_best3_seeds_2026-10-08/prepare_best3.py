"""Freeze the narrowed best-three experiment: 9 main + 20 internal + 1 reference.

This preparation never launches training, resumes interrupted runs, or changes
the previous experiment. Run only after all REQUIRED_SCRIPTS are ready/reviewed.
The frozen historical recipes/source/data are copied into a fresh isolated root.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil

import yaml


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'cmig_historical_recipe_2026-10-08'
HISTORY = ROOT.parent / 'cohort_specific_search_2026-09-15'
MAIN_CONFIGURATIONS = ('mobile_1024_full', 'pano_historical_spatialshape', 'dslr_cmfa_dualopt')
REFERENCE_SOURCE_ID = 'dslr_cmfa_adamw_seed0'
REFERENCE_NAME = 'internal_dslr_cmfa_adamw'
REQUIRED_SCRIPTS = (
    'prepare_best3.py', 'train_historical.py', 'evaluate_native.py',
    'evaluate_controlled.py', 'controlled_runtime.py', 'smoke_controlled.py',
    'queue_worker.py', 'launch_two_gpus.py', 'collect_tables.py',
)
OPTIONAL_SCRIPTS = ('finalize_launch_manifest.py', 'test_historical_execution.py')
ADMIN_FIELDS = {'output_dir', 'summary_dir'}


def digest(path):
    hasher = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require_hash(path, expected):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), f'Missing or symlink input: {path}'
    actual = digest(path)
    assert actual == expected, f'Changed frozen input: {path} ({actual} != {expected})'
    return actual


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


def semantic(config):
    value = deepcopy(config)
    for key in ADMIN_FIELDS:
        value.pop(key, None)
    return value


def select_jobs(manifest):
    """Explicit allowlist; no learning-primary grid is inherited by accident."""
    by_id = {job['id']: job for job in manifest['jobs']}
    assert len(by_id) == len(manifest['jobs']), 'Duplicate previous job IDs'
    main_ids = [f'{name}_seed{seed}' for seed in [0, 1, 2] for name in MAIN_CONFIGURATIONS]
    selected = [(by_id[identifier], 'main') for identifier in main_ids]
    internal = [job for job in manifest['jobs'] if job['phase'] == 'internal']
    assert len(internal) == 20 and all(job['seed'] == 0 for job in internal)
    selected.extend((job, 'internal') for job in internal)
    selected.append((by_id[REFERENCE_SOURCE_ID], 'internal_reference'))
    assert len(selected) == 30 and len({job['id'] for job, _ in selected}) == 30
    return selected


def main():
    generated = (
        'execution_manifest.json', 'configs', 'jobs', 'source_snapshots', 'audit',
        'source_snapshot_manifest.json', 'frozen_image_manifest.json',
        'configuration_differences.json', 'historical_mapping.json',
        'scope_declaration.json', 'cost_estimate.json',
    )
    assert not any((ROOT / name).exists() for name in generated), 'Preparation exists; refusing replacement'
    for name in REQUIRED_SCRIPTS:
        assert (ROOT / name).is_file() and not (ROOT / name).is_symlink(), f'Execution script not ready: {name}'
    previous = read_json(PREVIOUS / 'execution_manifest.json')
    history = read_json(HISTORY / 'manifest.json')
    previous_mapping = read_json(PREVIOUS / 'historical_mapping.json')
    previous_differences = {row['job_id']: row for row in read_json(PREVIOUS / 'configuration_differences.json')}
    assert len(history['source_sha256']) == 132, 'Unexpected historical source inventory'
    assert previous['native_cwd'] == '/home/wuxi/Dent-FINE'
    selected = select_jobs(previous)

    # Copy from the known frozen source, independently checking it against all
    # original historical Python hashes rather than trusting a new source tree.
    source_records, historical_seen = [], set()
    for record in previous['source_files']:
        source = Path(record['snapshot'])
        require_hash(source, record['sha256'])
        relative = source.relative_to(PREVIOUS / 'source_snapshots')
        assert relative.parts[0] in {'local', 'mobile'}
        original = record['source']
        if source.suffix == '.py':
            assert original in history['source_sha256'], f'Unregistered historical source: {original}'
            assert record['sha256'] == history['source_sha256'][original], f'Source is not historical: {original}'
            historical_seen.add(original)
        else:
            assert source.name in {'LICENSE', 'requirements.txt'}, f'Unexpected snapshot artifact: {source}'
        source_records.append({
            'source': str(source), 'original_historical_source': original,
            'snapshot': str(ROOT / 'source_snapshots' / relative),
            'sha256': record['sha256'], 'historical_source_hash_verified': source.suffix == '.py',
        })
    assert historical_seen == set(history['source_sha256']), 'Not all 132 historical source files are covered'

    jobs, contents, differences = [], {}, []
    for source_job, role in selected:
        require_hash(source_job['config'], source_job['config_sha256'])
        old_config = yaml.safe_load(Path(source_job['config']).read_text())
        config = deepcopy(old_config)
        name = REFERENCE_NAME if role == 'internal_reference' else source_job['configuration']
        seed = source_job['seed']
        out = ROOT / 'runs' / name / f'seed_{seed}'
        config['output_dir'] = str(out)
        config['summary_dir'] = str(out / 'tensorboard')
        assert semantic(config) == semantic(old_config), f'Non-administrative config drift: {name}'
        assert config['epoches'] == 160
        assert config['train_dataloader']['total_batch_size'] == 8
        assert config['val_dataloader']['total_batch_size'] == 16
        if role == 'main':
            assert source_job['historical_recipe_exact'] and not source_job['controls']
            anchor = Path(source_job['historical_anchor_config'])
            require_hash(anchor, source_job['historical_anchor_config_sha256'])
            assert semantic(config) == semantic(yaml.safe_load(anchor.read_text())), f'Main is not historical recipe: {name}'
        if source_job['cohort'] == 'Pano':
            assert name == 'pano_historical_spatialshape' and role == 'main'
            assert all(config['PostProcessor'][key] for key in
                       ['use_dental_spatial_prior', 'use_dental_shape_prior', 'use_dental_view_prior'])
        path = ROOT / 'configs' / name / f'seed_{seed}.yml'
        content = yaml.safe_dump(config, sort_keys=False)
        assert path not in contents
        contents[path] = content
        job = deepcopy(source_job)
        implementation_relative = Path(source_job['implementation']).relative_to(PREVIOUS / 'source_snapshots')
        implementation = ROOT / 'source_snapshots' / implementation_relative
        job.update({
            'id': f'{name}_seed{seed}', 'configuration': name,
            'phase': 'primary' if role == 'main' else 'internal', 'scope_role': role,
            'config': str(path), 'config_sha256': sha256(content.encode()).hexdigest(),
            'output_dir': str(out), 'implementation': str(implementation),
            'train_script': str(implementation / 'train.py'),
            'source_frozen_job_id': source_job['id'], 'source_frozen_job_phase': source_job['phase'],
            'source_frozen_config': source_job['config'],
            'source_frozen_config_sha256': source_job['config_sha256'],
            'fresh_initialization': True, 'resume_interrupted_run': False,
            'reused_training_weights': False,
        })
        if role == 'internal_reference':
            job['reference_purpose'] = 'CMFA fixed, ordinary AdamW: DualOpt neither/none reference; seed0 only'
        job['job_file'] = str(ROOT / 'jobs' / f"{job['id']}.json")
        jobs.append(job)
        audit = deepcopy(previous_differences[source_job['id']])
        audit.update({
            'job_id': job['id'], 'source_frozen_job_id': source_job['id'],
            'source_frozen_config': source_job['config'],
            'source_frozen_config_sha256': source_job['config_sha256'],
            'source_frozen_semantic_config_identical': True, 'scope_role': role,
            'administrative_changes': {'output_dir': str(out), 'summary_dir': str(out / 'tensorboard')},
        })
        assert audit['only_declared_ablation_variables_changed']
        differences.append(audit)
    assert len(jobs) == 30
    assert Counter(job['scope_role'] for job in jobs) == {'main': 9, 'internal': 20, 'internal_reference': 1}
    assert Counter(job['phase'] for job in jobs) == {'primary': 9, 'internal': 21}
    assert {job['configuration'] for job in jobs if job['scope_role'] == 'main'} == set(MAIN_CONFIGURATIONS)
    for name in MAIN_CONFIGURATIONS:
        group = [job for job in jobs if job['configuration'] == name]
        assert {job['seed'] for job in group} == {0, 1, 2}
        variants = [semantic(yaml.safe_load(contents[Path(job['config'])])) for job in group]
        assert variants[0] == variants[1] == variants[2], f'Across-seed drift: {name}'

    priority = {name: index for index, name in enumerate(MAIN_CONFIGURATIONS)}
    roles = {'main': 0, 'internal_reference': 1, 'internal': 2}
    jobs.sort(key=lambda job: (roles[job['scope_role']], job['seed'],
                              priority.get(job['configuration'], 10), job['configuration']))
    gpu_hours, orders = {0: 0., 1: 0.}, {0: 0, 1: 0}
    for index, job in enumerate(jobs):
        gpu = 0 if index == 0 else 1 if index == 1 else min(gpu_hours, key=gpu_hours.get)
        hours = float(job['estimated_gpu_hours'])
        assert hours > 0
        job['gpu'], job['queue_order'] = gpu, orders[gpu]
        orders[gpu] += 1
        gpu_hours[gpu] += hours
    assert jobs[0]['id'] == 'mobile_1024_full_seed0' and jobs[0]['gpu'] == 0
    assert jobs[1]['id'] == 'pano_historical_spatialshape_seed0' and jobs[1]['gpu'] == 1
    assert jobs[2]['id'] == 'dslr_cmfa_dualopt_seed0'

    # Keep relevant audited data/anchors, but do not lock unused learning-grid configs.
    input_kinds = {'annotation', 'spatial_shape_prior', 'pretrained_backbone',
                   'selected_config', 'historical_config', 'manuscript'}
    inputs = [deepcopy(row) for row in previous['input_files'] if row['kind'] in input_kinds]
    for row in inputs:
        if row['kind'] != 'manuscript':
            require_hash(row['path'], row['sha256'])
    tracked = {row['path'] for row in inputs}
    def track(path, kind, expected=None):
        path = Path(path)
        value = require_hash(path, expected) if expected else digest(path)
        if str(path) not in tracked:
            inputs.append({'kind': kind, 'path': str(path), 'sha256': value})
            tracked.add(str(path))
    for job, _ in selected:
        track(job['config'], 'source_frozen_config', job['config_sha256'])
    for path in [PREVIOUS / 'execution_manifest.json', PREVIOUS / 'source_snapshot_manifest.json',
                 PREVIOUS / 'historical_mapping.json', PREVIOUS / 'configuration_differences.json',
                 PREVIOUS / 'audit/image_hash_manifest.csv', HISTORY / 'manifest.json']:
        track(path, 'provenance_manifest')
    require_hash(previous['pretrained']['path'], previous['pretrained']['sha256'])
    for row in previous['prior_files']:
        require_hash(row['path'], row['sha256'])
    require_hash(previous['image_manifest'], previous['image_manifest_sha256'])
    images = read_json(previous['image_manifest'])
    assert len(images) == 2672, 'Unexpected original image inventory'
    for row in images:
        assert not row['error'], f"Image failed original audit: {row['path']}"
        require_hash(row['path'], row['bytes_sha256'])
        stat = Path(row['path']).stat()
        row['size'], row['mtime_ns'] = stat.st_size, stat.st_mtime_ns

    now = datetime.now(timezone.utc).isoformat()
    scope = {
        'created_at': now, 'total_fresh_training_jobs': 30,
        'main_training_jobs': 9, 'main_configurations': list(MAIN_CONFIGURATIONS), 'main_seeds': [0, 1, 2],
        'internal_ablation_jobs': 20, 'internal_seeds': [0],
        'internal_reference_jobs': 1, 'internal_reference': REFERENCE_NAME,
        'excluded': ['14-configuration primary learning grid', 'four Pano learning-primary configurations',
                     'historical/interrupted checkpoint reuse', 'original source/data changes'],
        'pano_prior_comparison': 'None / center / shape / center+shape are inference-only replays of the same three historical-Pano main checkpoints; no extra Pano training.',
        'factor_comparison': 'Existing declared factor controls and same-checkpoint native/FP32 inference replays only.',
        'checkpoint_retention': 'Each successful job retains best.pth only; stage1 temporary native rollback weights are removed after verified replay.',
        'historical_training_weights_reused': 0, 'interrupted_training_runs_resumed': 0,
        'old_interrupted_experiment': {'root': str(PREVIOUS), 'preserved': True,
                                       'policy': 'Interrupted broader queue is retained unchanged; no resume, overwrite, or deletion.'},
        'authorization': 'User narrowed scope to the three selected historical recipes with three seeds each, internal ablations, and one necessary DualOpt AdamW reference; restart on two GPUs.',
        'preparation_starts_training': False,
    }
    cost = {'created_at': now, 'runs': len(jobs),
            'estimated_total_gpu_hours': sum(gpu_hours.values()),
            'estimated_gpu_hours_by_queue': gpu_hours,
            'estimated_two_gpu_wall_hours': max(gpu_hours.values()),
            'queue_jobs': dict(Counter(job['gpu'] for job in jobs)),
            'basis': 'Inherited matching-run estimates from frozen historical-recipe preparation; includes train/replay where measured.',
            'uncertainty': 'Planning estimate only; allow runtime, replay, and startup variation.',
            'disk_min_free_bytes': previous['disk_min_free_bytes']}
    mapping = deepcopy(previous_mapping)
    mapping.update({'created_at': now, 'scope': scope,
                    'previous_frozen_manifest': str(PREVIOUS / 'execution_manifest.json'),
                    'historical_training_runs_reused': 0,
                    'primary_full_recipe_mapping': {'mobile_1024_full': 'phone_a07_proto_aux_C05',
                                                    'pano_historical_spatialshape': 'R01_fixed640_mosaic0_mixup0p5',
                                                    'dslr_cmfa_dualopt': 'dslr_cmfa_dualopt_C02'},
                    'pano_learning_ablation_note': 'No Pano learning-primary grid is scheduled; four inference-prior settings replay the same three main weights.'})
    script_names = list(REQUIRED_SCRIPTS) + [name for name in OPTIONAL_SCRIPTS if (ROOT / name).is_file()]
    scripts = {name: digest(ROOT / name) for name in script_names}

    # All read-only validation has passed; create only fresh task-owned artifacts.
    for row in source_records:
        destination = Path(row['snapshot'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copy2(row['source'], destination)
        require_hash(destination, row['sha256'])
    for path, content in contents.items():
        write_new(path, content)
    for job in jobs:
        write_new(job['job_file'], job)
    write_new(ROOT / 'source_snapshot_manifest.json', source_records)
    write_new(ROOT / 'frozen_image_manifest.json', images)
    write_new(ROOT / 'audit/source_manifest.json', inputs)
    shutil.copy2(PREVIOUS / 'audit/image_hash_manifest.csv', ROOT / 'audit/image_hash_manifest.csv')
    write_new(ROOT / 'configuration_differences.json', differences)
    write_new(ROOT / 'historical_mapping.json', mapping)
    write_new(ROOT / 'scope_declaration.json', scope)
    write_new(ROOT / 'cost_estimate.json', cost)
    manifest = {
        'frozen_at': now, 'authorization': scope['authorization'], 'scope': scope,
        'python': previous['python'], 'native_cwd': previous['native_cwd'],
        'primary_jobs': 9, 'main_jobs': 9, 'internal_jobs': 21,
        'internal_ablation_jobs': 20, 'internal_reference_jobs': 1,
        'historical_reproduction_jobs': 0, 'total_jobs': len(jobs), 'historical_runs_reused': 0,
        'jobs': jobs, 'source_files': source_records, 'input_files': inputs,
        'image_manifest': str(ROOT / 'frozen_image_manifest.json'),
        'image_manifest_sha256': digest(ROOT / 'frozen_image_manifest.json'),
        'pretrained': deepcopy(previous['pretrained']), 'prior_files': deepcopy(previous['prior_files']),
        'scripts': scripts, 'gpus': [0, 1], 'disk_min_free_bytes': cost['disk_min_free_bytes'],
        'configuration_differences': str(ROOT / 'configuration_differences.json'),
        'configuration_differences_sha256': digest(ROOT / 'configuration_differences.json'),
        'historical_mapping': str(ROOT / 'historical_mapping.json'),
        'historical_mapping_sha256': digest(ROOT / 'historical_mapping.json'),
        'scope_declaration': str(ROOT / 'scope_declaration.json'),
        'scope_declaration_sha256': digest(ROOT / 'scope_declaration.json'),
        'cost_estimate': cost, 'stage_policy': previous['stage_policy'],
        'launch_policy': previous['launch_policy'], 'data_policy': previous['data_policy'],
        'previous_interrupted_root': str(PREVIOUS),
        'manifest_immutability': 'Created last; preparation refuses existing generated artifacts. No training is started here.',
    }
    write_new(ROOT / 'execution_manifest.json', manifest)
    print(json.dumps({'prepared': True, 'training_started': False, 'jobs': len(jobs),
                      'scope_counts': dict(Counter(job['scope_role'] for job in jobs)),
                      'first_jobs': {gpu: next(job['id'] for job in jobs if job['gpu'] == gpu) for gpu in [0, 1]},
                      'estimated_two_gpu_wall_hours': cost['estimated_two_gpu_wall_hours']}, indent=2))


if __name__ == '__main__':
    main()
