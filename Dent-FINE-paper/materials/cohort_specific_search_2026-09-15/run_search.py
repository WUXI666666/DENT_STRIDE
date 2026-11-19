"""Independent, bounded augmentation selection; never edits model code or old runs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import traceback
import zipfile

import yaml

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'augmentation_search_2026-09-13'
BUNDLE = ROOT.parent / 'best_model_bundle_2026-09-15'
CONFIGS = Path('/home/wuxi/Dent-FINE/configs/paper_cohort_search_20260915')
OUTPUTS = Path('/home/wuxi/Dent-FINE/outputs/paper_cohort_search_20260915')
TASKS = ('phone_dualopt', 'dslr_cmfa_dualopt', 'phone_a07_proto_aux')
STARTING_IDS = {
    'dentex_spatial_shape': 'R01_fixed640_mosaic0_mixup0p5',
    'phone_dualopt': 'FINAL_phone_dualopt',
    'dslr_cmfa_dualopt': 'FINAL_dslr_cmfa_dualopt',
    'phone_a07_proto_aux': 'FINAL_phone_a07_proto_aux',
}
spec = importlib.util.spec_from_file_location('verified_augmentation_helpers', PREVIOUS / 'experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
# Redirect orchestration outputs only. The imported helper and model source stay unchanged.
e.ROOT, e.CONFIGS, e.OUTPUTS = ROOT, CONFIGS, OUTPUTS


def checked_copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    digest = e.sha(source)
    if expected is not None and digest != expected:
        raise RuntimeError(f'Source checksum mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or e.sha(destination) != digest:
            raise RuntimeError(f'Refusing to overwrite a different artifact: {destination}')
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=2**20)
    if e.sha(destination) != digest:
        raise RuntimeError(f'Archive copy verification failed: {destination}')
    return digest


def verify_result(result):
    out = Path(result['output_dir'])
    marker = e.read_json(out / 'COMPLETED.json')
    assert e.sha(out / 'metrics_summary.json') == marker['metrics_sha256']
    assert e.sha(result['checkpoint']) == result['checkpoint_sha256'] == marker['checkpoint_sha256']
    assert e.sha(result['config']) == result['config_sha256']
    assert e.sha(out / 'config.yml') == result['config_sha256']
    assert e.sha(out / 'log.txt') == result['log_sha256']
    rows = e.read_rows(out)
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
    assert best['epoch'] + 1 == result['best_epoch_one_based']
    for metric in ('AP', 'AP50', 'AP75', 'AR100'):
        assert abs(best['test_coco_eval_bbox'][e.METRIC_NAMES.index(metric)]
                   - result['metrics_fraction'][metric]) < 2e-6
    assert result['verified_best_checkpoint'] and result['seed'] == 0


def archive_model(result, metadata, collection):
    verify_result(result)
    task = result['task']
    relative = Path(collection) / f"{task}_AP{result['metrics_percent']['AP']:.3f}"
    folder = BUNDLE / relative
    out = Path(result['output_dir'])
    files = {}
    for name in ('best.pth', 'config.yml', 'metrics_summary.json', 'coco_metrics.json',
                 'log.txt', 'COMPLETED.json', 'run_protocol.json', 'job.json',
                 'preflight.json', 'checkpoint_retention.json', 'best_evaluation.log'):
        files[name] = checked_copy(out / name, folder / name)
    assets = {}
    for name, item in metadata['initialization_and_prior_files'].items():
        asset = Path('shared_assets') / f"{item['sha256'][:12]}_{Path(item['path']).name}"
        checked_copy(item['path'], BUNDLE / asset, item['sha256'])
        assets[name] = {**item, 'bundle_relative_path': str(asset)}
    # Retain the exact category definitions; do not infer a clinical FDI code from an index.
    category_info = {}
    for split, item in metadata['datasets'].items():
        assert e.sha(item['ann_file']) == item['ann_sha256']
        annotations = e.read_json(item['ann_file'])
        category_info[split] = {'annotation_sha256': item['ann_sha256'],
                                'categories': annotations['categories']}
    e.write_json(folder / 'category_definitions.json', category_info)
    cfg = yaml.safe_load((folder / 'config.yml').read_text())
    entry = {'task': task, 'model': result['model'], 'AP': result['metrics_percent']['AP'],
             'AP50': result['metrics_percent']['AP50'], 'AP75': result['metrics_percent']['AP75'],
             'AR100': result['metrics_percent']['AR100'], 'seed': result['seed'],
             'best_epoch_one_based': result['best_epoch_one_based'],
             'params_trainable': result['params_trainable'], 'augmentation': result['augmentation'],
             'source_output': str(out), 'bundle_relative_path': str(relative),
             'files_sha256': files, 'assets': assets, 'datasets': metadata['datasets'],
             'fdi_label_order': cfg.get('fdi_label_order'),
             'implementation_dir': result['implementation_dir'],
             'configuration_note': 'config.yml is byte-identical to the actual run, including original absolute paths. '
                 'Shared assets are archived separately. Dataset images and full model source are not in this bundle.'}
    e.write_json(folder / 'provenance.json', entry)
    return entry


def create_zip(name, model_entries):
    """Create an immutable archive containing only explicit selected model folders and assets."""
    target = BUNDLE / name
    if target.exists():
        raise RuntimeError(f'Archive already exists; refusing replacement: {target}')
    members = {BUNDLE / 'README.md'}
    for entry in model_entries:
        members.update(p for p in (BUNDLE / entry['bundle_relative_path']).rglob('*') if p.is_file())
        members.update(BUNDLE / a['bundle_relative_path'] for a in entry['assets'].values())
    index = {'models': model_entries, 'created_at': e.now(), 'search_root': str(ROOT)}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                         allowZip64=True) as handle:
        handle.writestr('bundle_index.json', json.dumps(index, ensure_ascii=False, indent=2))
        for path in sorted(members):
            assert BUNDLE in path.parents and not path.is_symlink()
            handle.write(path, str(path.relative_to(BUNDLE)))
    with zipfile.ZipFile(target) as handle:
        if handle.testzip() is not None:
            raise RuntimeError('ZIP CRC validation failed')
        for entry in model_entries:
            for filename, digest in entry['files_sha256'].items():
                import hashlib
                h = hashlib.sha256()
                with handle.open(str(Path(entry['bundle_relative_path']) / filename)) as stream:
                    for chunk in iter(lambda: stream.read(2**20), b''):
                        h.update(chunk)
                assert h.hexdigest() == digest
    e.write_json(target.with_suffix('.zip.sha256.json'),
                 {'path': str(target), 'sha256': e.sha(target), 'bytes': target.stat().st_size,
                  'crc_verified': True, 'model_artifact_hashes_verified': True})
    return str(target)


def core_recipes(task):
    native_mixup = 0.0 if task == 'phone_a07_proto_aux' else 0.5
    # The matching fixed640/MixUp=.5 reference is reused, not trained again.
    return [e.augmentation('legacy640_multiscale', mixup=native_mixup),
            e.augmentation('legacy640_multiscale', mixup=0.5 - native_mixup),
            e.augmentation('fixed640', mixup=0.0),
            e.augmentation('fixed1024', mixup=0.5),
            e.augmentation('fixed1024', mixup=0.0)]


def make_job(manifest, job_id, task, recipe, phase):
    meta = manifest['tasks'][task]
    base = yaml.safe_load(Path(meta['base_config']).read_text())
    cfg = e.apply_augmentation(base, recipe)
    # Compose already samples the outer gate. A second .25 gate would yield .0625, not .25.
    if recipe['mosaic'] > 0:
        e.transform(cfg, 'Mosaic')['probability'] = 1.0
    out = OUTPUTS / job_id / 'seed_0'
    cfg['output_dir'] = str(out)
    path = CONFIGS / f'{job_id}_seed0.yml'
    content = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'Immutable config changed: {path}')
    if not path.exists():
        path.write_text(content)
    return {'job_id': job_id, 'task': task, 'cohort': meta['cohort'], 'model': meta['model'],
            'phase': phase, 'seed': 0, 'augmentation': recipe, 'config': str(path),
            'config_sha256': e.sha(path), 'output_dir': str(out),
            'implementation_dir': meta['implementation_dir'], 'train_script': meta['train_script'],
            'driver_script': str(Path(__file__)), 'driver_script_sha256': e.sha(__file__),
            'changes_from_aligned_base': e.diff(base, cfg),
            'mosaic_gate': {'outer_probability': recipe['mosaic'],
                            'inner_probability': e.transform(cfg, 'Mosaic')['probability'],
                            'active_epochs_zero_based': [4, 69],
                            'native_crop_zoom_exclusion_preserved': True}}


def refinement_jobs(manifest, task, core_winner):
    recipe = core_winner['augmentation']
    assert recipe['mosaic'] == 0 and recipe['crop'] == .8 and recipe['zoom'] == .5
    changes = [('R01_crop0p5', {'crop': .5}), ('R02_zoom0', {'zoom': 0.0}),
               ('R03_mosaic0p25', {'mosaic': .25})]
    return [make_job(manifest, f'{task}_{tag}', task, {**recipe, **change}, 'single_factor_refinement')
            for tag, change in changes]


def prepare():
    if (ROOT / 'manifest.json').exists() or BUNDLE.exists() or CONFIGS.exists() or OUTPUTS.exists():
        raise RuntimeError('Study/archive/output already exists; refusing to re-prepare')
    previous = e.read_json(PREVIOUS / 'manifest.json')
    e.verify_sources(previous)
    CONFIGS.mkdir(parents=True)
    OUTPUTS.mkdir(parents=True)
    (ROOT / 'reference_configs').mkdir()
    BUNDLE.mkdir()
    checked_copy(PREVIOUS / 'preflight.py', ROOT / 'preflight.py')
    policy = {'registered_at': e.now(), 'primary_endpoint': 'Validation COCO bbox AP, IoU .50:.95',
              'selection': 'Highest independently re-evaluated best-checkpoint AP, separately per model/cohort',
              'seed': 0, 'epochs': 160, 'fresh_initialization': True,
              'secondary_metrics': ['AP50', 'AP75', 'AR100'],
              'core_grid': 'legacy640 multiscale / fixed640 / fixed1024 x MixUp 0 / .5; '
                           'Mosaic 0, IoUCrop .8, ZoomOut .5',
              'reused_candidates_per_task': 1, 'new_core_candidates_per_task': 5,
              'refinement': 'Three separate single-factor trials on each core winner: IoUCrop .5, '
                            'ZoomOut 0, or Mosaic .25. No untested combinations are claimed optimal.',
              'total_new_runs': 24, 'dentex_retraining': False,
              'final_allowed_profiles': ['legacy640_multiscale', 'fixed640', 'fixed1024'],
              'tie_break': 'AP descending, then smaller evaluation size, lower Mosaic, lower MixUp, '
                           'higher crop/zoom, lexical job ID',
              'native_unchanged': ['datasets/splits/label maps', 'model/modules/losses', 'optimizer/LR',
                                  'EMA/stage behavior', 'train batch 8, val batch 16', 'initial backbone'],
              'retention': 'Keep every candidate config/log/metrics and best.pth; delete only native '
                           'temporary last/stage weights after successful final re-evaluation.',
              'scope': 'Exploratory bounded validation selection, not globally optimal or a locked-test/SOTA claim.'}
    e.write_json(ROOT / 'selection_policy.json', policy)
    manifest = {'created_at': e.now(), 'project': str(e.REPO), 'python': e.PYTHON, 'gpu': 0,
                'epochs': 160, 'search_seed': 0, 'configs': str(CONFIGS), 'output_root': str(OUTPUTS),
                'bundle_root': str(BUNDLE), 'tasks': {}, 'source_sha256': copy.deepcopy(previous['source_sha256']),
                'fixed_file_sha256': copy.deepcopy(previous['fixed_file_sha256']),
                'reference_results': {}, 'initial_jobs': [], 'total_new_runs': 24,
                'helper_script': str(PREVIOUS / 'experiment.py'), 'driver_script': str(Path(__file__)),
                'source_note': 'The previous aligned configs recovered from actual historical TensorBoard configs '
                    'are the source of model settings. Main Arch=.25; A07 Arch=.1 and historical best-only behavior.'}
    all_results = e.read_json(PREVIOUS / 'results_summary.json')
    by_id = {r['job_id']: r for r in all_results}
    entries = []
    for task, job_id in STARTING_IDS.items():
        meta = copy.deepcopy(previous['tasks'][task])
        old_base = Path(meta['base_config'])
        base = ROOT / 'reference_configs' / old_base.name
        checked_copy(old_base, base, meta['base_config_sha256'])
        meta.update(base_config=str(base), base_config_sha256=e.sha(base))
        manifest['tasks'][task] = meta
        result = by_id[job_id]
        verify_result(result)
        # Verify the reused result equals the aligned model with only its declared input/augmentation changes.
        expected = e.apply_augmentation(yaml.safe_load(base.read_text()), result['augmentation'])
        actual = yaml.safe_load(Path(result['config']).read_text())
        assert e.config_semantics(expected) == e.config_semantics(actual), task
        manifest['reference_results'][task] = result
        entries.append(archive_model(result, meta, 'starting_models'))
        manifest['fixed_file_sha256'][str(base)] = e.sha(base)
        for path in (result['config'], str(Path(result['output_dir']) / 'metrics_summary.json')):
            manifest['fixed_file_sha256'][path] = e.sha(path)
    # Round-robin priority starts each model's historical profile before deeper grid trials.
    for number in range(5):
        for task in TASKS:
            manifest['initial_jobs'].append(make_job(manifest, f'{task}_C{number+1:02}', task,
                                                     core_recipes(task)[number], 'core_grid'))
    for path in (Path(__file__), ROOT / 'preflight.py', ROOT / 'test_search.py',
                 ROOT / 'selection_policy.json', PREVIOUS / 'experiment.py'):
        manifest['fixed_file_sha256'][str(path)] = e.sha(path)
    e.write_json(ROOT / 'manifest.json', manifest)
    e.write_json(ROOT / 'core_single_factor_contrasts.json',
                 {t: e.build_contrasts([manifest['reference_results'][t]] +
                         [j for j in manifest['initial_jobs'] if j['task'] == t]) for t in TASKS})
    e.write_json(BUNDLE / 'starting_models_index.json', {'models': entries, 'created_at': e.now()})
    (BUNDLE / 'README.md').write_text(
        '# DENT-STRIDE 配置与最佳权重归档\n\n'
        'starting_models 保存本轮搜索前已完成并复测的四个指定结果，DENTEX AP 54.385 不重训、不覆盖。\n'
        '每个模型目录的 config.yml 与 best.pth 与原运行文件逐字节一致，provenance.json 记录 SHA256、指标和来源。\n'
        'config.yml 保留原绝对路径；共享初始化权重及 DENTEX SpatialShape 模板见 shared_assets。'
        '这里不是含数据集和完整源代码的独立训练安装包，移动后需要调整路径。\n'
        'selected_models 会保存各模型完成本轮有限候选搜索后的最佳结果；低分试验不会从结果表中删除。\n'
        'AP 等为百分数；best_epoch 从 1 开始。seed 0、160 轮、验证 COCO AP 选 checkpoint。\n'
        'Phone/DSLR 使用原 Dent-FINE 实现，A07 使用该项目内此前整合的隔离实现。\n'
        '本轮为开发/内部验证选优，不表示锁定测试结果、统计显著提升或全局最优。\n', encoding='utf-8')
    archive = create_zip('starting_models_verified.zip', entries)
    e.summarize([manifest['reference_results'][t] for t in TASKS])
    e.verify_sources(manifest)
    e.write_json(ROOT / 'preparation_complete.json', {'completed_at': e.now(), 'archive': archive,
                  'manifest_sha256': e.sha(ROOT / 'manifest.json'), 'total_new_runs': 24})
    print(json.dumps({'prepared': True, 'archive': archive, 'new_runs': 24}, ensure_ascii=False), flush=True)


def minimum_memory(job):
    if job['augmentation']['profile'] != 'fixed1024':
        return 8000
    if job['task'] == 'phone_a07_proto_aux':
        return 18000 if job['augmentation']['mosaic'] and job['augmentation']['mixup'] else 13500
    return 11000


def choose_ready(pending, free):
    return next((job for job in pending if free >= minimum_memory(job)), None)


def run_study():
    manifest = e.read_json(ROOT / 'manifest.json')
    with (ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        e.STATE = {'queue_pid': os.getpid(), 'started_at': e.now(), 'gpu': 0,
                   'state': 'initializing', 'child_pid': None, 'completed': [], 'total_new_runs': 24}
        e.save_state()
        signal.signal(signal.SIGTERM, e.interrupted)
        signal.signal(signal.SIGINT, e.interrupted)
        results = [manifest['reference_results'][t] for t in TASKS]
        pending = list(manifest['initial_jobs'])
        refined, final_entries = set(), {}
        try:
            e.verify_sources(manifest)
            for r in results:
                verify_result(r)
            while pending:
                free = e.gpu_free()
                job = choose_ready(pending, free)
                if job is None:
                    # Nothing runnable: wait for the least-demanding pending job, never alter batch/GPU.
                    job = min(pending, key=minimum_memory)
                    e.save_state(current_job=job['job_id'], current_task=job['task'], current_seed=0,
                                 pending_jobs=[j['job_id'] for j in pending])
                    e.wait_resources(job)
                result = e.run_job(manifest, job)
                results.append(result)
                pending.remove(job)
                e.summarize(results)
                for task in TASKS:
                    task_pending = [j for j in pending if j['task'] == task]
                    if task not in refined and not any(j['phase'] == 'core_grid' for j in task_pending):
                        candidates = [r for r in results if r['task'] == task]
                        winner = min(candidates, key=e.rank_key)
                        new_jobs = refinement_jobs(manifest, task, winner)
                        e.write_json(ROOT / f'{task}_core_selection.json',
                                     {'selected': winner, 'ranking': [r['job_id'] for r in sorted(candidates, key=e.rank_key)],
                                      'refinement_jobs': new_jobs})
                        e.write_json(ROOT / f'{task}_refinement_contrasts.json', e.build_contrasts([winner] + new_jobs))
                        pending.extend(new_jobs)
                        refined.add(task)
                    elif task in refined and not task_pending and task not in final_entries:
                        candidates = [r for r in results if r['task'] == task]
                        assert len(candidates) == 9
                        winner = min(candidates, key=e.rank_key)
                        entry = archive_model(winner, manifest['tasks'][task], 'selected_models')
                        final_entries[task] = entry
                        e.write_json(ROOT / f'{task}_selected_result.json',
                                     {'selected': winner, 'archive': entry, 'candidate_count': len(candidates),
                                      'ranking': [r['job_id'] for r in sorted(candidates, key=e.rank_key)],
                                      'delta_AP_from_start': winner['metrics_percent']['AP']
                                          - manifest['reference_results'][task]['metrics_percent']['AP'],
                                      'scope': 'Best within nine validation candidates, not a global optimum.'})
                        e.write_json(BUNDLE / 'selected_models_index.json', final_entries)
                e.save_state(pending_jobs=[j['job_id'] for j in pending],
                             finalized_models=list(final_entries), completed_new_runs=len(results) - 3)
            assert len(results) == 27 and len(final_entries) == 3
            dentex = archive_model(manifest['reference_results']['dentex_spatial_shape'],
                                   manifest['tasks']['dentex_spatial_shape'], 'selected_models')
            archive = create_zip('selected_models_after_search.zip', [dentex] + list(final_entries.values()))
            e.verify_sources(manifest)
            e.save_state(state='completed', current_job=None, child_pid=None, completed_at=e.now(), final_archive=archive)
        except BaseException as exc:
            e.stop_child()
            e.save_state(state='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                         child_pid=None, error=str(exc))
            traceback.print_exc()
            raise


def launch():
    complete = e.read_json(ROOT / 'preparation_complete.json')
    assert complete['manifest_sha256'] == e.sha(ROOT / 'manifest.json')
    e.verify_sources(e.read_json(ROOT / 'manifest.json'))
    subprocess.run([e.PYTHON, '-m', 'unittest', '-v', 'test_search'], cwd=ROOT, check=True)
    log = ROOT / 'queue.log'
    if log.exists():
        raise RuntimeError('Launch log exists; inspect the previous queue instead of starting a duplicate')
    with (ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('x', buffering=1) as handle:
        child = subprocess.Popen([e.PYTHON, '-u', str(Path(__file__)), 'run'], cwd=ROOT,
                                 env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    e.write_json(ROOT / 'launch.json', {'pid': child.pid, 'gpu': 0, 'started_at': e.now(),
                                      'driver_script_sha256': e.sha(__file__), 'log': str(log)})
    print(json.dumps({'launched': True, 'pid': child.pid, 'gpu': 0, 'log': str(log)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'launch', 'run', 'status'])
    args = parser.parse_args()
    {'prepare': prepare, 'launch': launch, 'run': run_study, 'status': e.status}[args.command]()
