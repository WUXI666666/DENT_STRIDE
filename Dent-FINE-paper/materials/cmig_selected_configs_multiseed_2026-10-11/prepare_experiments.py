"""Freeze selected model settings and a seed schedule before launching any run."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'cmig_best3_seeds_2026-10-08'
SEEDS = [0, 1, 2, 3, 4]
TEMPLATES = [
    ('mobile_proto_k4_tau0p50', 'internal_mobile_proto_k4_tau0p5_seed0'),
    ('dslr_cmfa_extra_decay_only', 'internal_dslr_extra_decay_only_seed0'),
    ('pano_spatialshape_original', 'pano_historical_spatialshape_seed0'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def scientific_config(cfg):
    cfg = deepcopy(cfg)
    cfg.pop('output_dir', None)
    cfg.pop('summary_dir', None)
    return cfg


def main():
    assert not (ROOT / 'execution_manifest.json').exists(), 'Preparation must not overwrite a frozen experiment'
    original = json.loads((SOURCE / 'execution_manifest.json').read_text())
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3, 'Insufficient free space for launch'
    for row in original['source_files']:
        assert sha(row['snapshot']) == row['sha256'], row['snapshot']
    copy_names = ['train_historical.py', 'evaluate_native.py', 'evaluate_controlled.py',
                  'controlled_runtime.py', 'smoke_controlled.py']
    copied = []
    for name in copy_names:
        assert sha(SOURCE / name) == original['scripts'][name]
        shutil.copy2(SOURCE / name, ROOT / name)
        copied.append({'source': str(SOURCE / name), 'destination': str(ROOT / name), 'sha256': sha(ROOT / name)})
    (ROOT / 'native_export').mkdir()
    export_source = SOURCE / 'power_restart_20261009/export_native.py'
    recovery_freeze = json.loads((SOURCE / 'power_restart_20261009/RECOVERY_FREEZE.json').read_text())
    assert sha(export_source) == recovery_freeze['scripts']['export_native.py']
    shutil.copy2(export_source, ROOT / 'native_export/export_native.py')
    copied.append({'source': str(export_source), 'destination': str(ROOT / 'native_export/export_native.py'), 'sha256': sha(export_source)})
    shutil.copytree(SOURCE / 'source_snapshots', ROOT / 'source_snapshots',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sources = []
    for row in original['source_files']:
        new = ROOT / Path(row['snapshot']).relative_to(SOURCE)
        assert sha(new) == row['sha256']
        sources.append({'path': str(new), 'sha256': row['sha256'], 'copied_from': row['snapshot']})
    image_manifest = ROOT / 'frozen_image_manifest.json'
    assert sha(original['image_manifest']) == original['image_manifest_sha256']
    shutil.copy2(original['image_manifest'], image_manifest)
    inputs = [r for r in original['input_files'] if r['kind'] in
              ['annotation', 'spatial_shape_prior', 'pretrained_backbone']]
    for row in inputs:
        assert sha(row['path']) == row['sha256'], row['path']
    templates = {}
    baselines = []
    for name, job_id in TEMPLATES:
        old = json.loads((SOURCE / 'jobs' / (job_id + '.json')).read_text())
        assert sha(old['config']) == old['config_sha256']
        cfg = yaml.safe_load(Path(old['config']).read_text())
        assert cfg['epoches'] == 160
        assert cfg['train_dataloader']['total_batch_size'] == 8
        assert cfg['val_dataloader']['total_batch_size'] == 16
        assert old['controls'] == {}
        if old['cohort'] == 'Mobile':
            assert cfg['DFINETransformer']['prototype_count'] == 4
            assert cfg['DFINETransformer']['prototype_tau'] == 0.50
            assert cfg['save_best_only'] is True
        if old['cohort'] == 'DSLR':
            assert cfg['optimizer']['rollback_weight'] == 0.0
            assert cfg['optimizer']['realtime_decay'] == 0.01
            assert cfg['HybridEncoder']['use_cmfa_fusion'] is True
        if old['cohort'] == 'Pano':
            assert cfg['PostProcessor']['use_dental_spatial_prior'] is True
            assert cfg['PostProcessor']['use_dental_shape_prior'] is True
        templates[name] = (old, cfg)
        metrics = json.loads((Path(old['output_dir']) / 'native_coco_metrics.json').read_text())
        baselines.append({'configuration': name, 'source_job': old['job_file'],
                          'config': old['config'], 'config_sha256': old['config_sha256'],
                          'seed0_reference_metrics_percent': metrics['metrics_percent'],
                          'use': 'Historical configuration reference only; no checkpoint reuse'})
        inputs.append({'kind': 'template_configuration', 'path': old['config'], 'sha256': old['config_sha256']})
    jobs = []
    for seed in SEEDS:
        for name, _ in TEMPLATES:
            old, template = templates[name]
            out = ROOT / 'runs' / name / ('seed_' + str(seed))
            config_path = ROOT / 'configs' / name / ('seed_' + str(seed) + '.yml')
            cfg = deepcopy(template)
            cfg['output_dir'] = str(out)
            cfg['summary_dir'] = str(out / 'tensorboard')
            assert scientific_config(cfg) == scientific_config(template)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            implementation = ROOT / Path(old['implementation']).relative_to(SOURCE)
            jid = name + '_seed' + str(seed)
            job = {
                'id': jid, 'configuration': name, 'cohort': old['cohort'], 'seed': seed,
                'phase': 'primary' if seed in [0, 1, 2] else 'stability',
                'config': str(config_path), 'config_sha256': sha(config_path),
                'output_dir': str(out), 'implementation': str(implementation),
                'train_script': str(implementation / 'train.py'),
                'native_cwd': original['native_cwd'], 'controls': {}, 'epochs': 160,
                'expected_input': old['expected_input'], 'inference_priors': old['inference_priors'],
                'python': original['python'], 'queue_order': len(jobs),
                'job_file': str(ROOT / 'jobs' / (jid + '.json')),
                'template_job': old['job_file'], 'template_config': old['config'],
                'template_config_sha256': old['config_sha256'],
                'only_config_changes': ['output_dir', 'summary_dir'],
                'seed_set_via_native_cli': True, 'fresh_initialization': True,
                'reused_training_weights': False, 'resume_interrupted_run': False,
                'prior_selection_policy': old['prior_selection_policy'],
            }
            write(Path(job['job_file']), job)
            jobs.append(job)
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                        '--format=csv,noheader'], text=True).splitlines()
    gpu_map = {}
    for line in gpu_lines:
        index, uuid, name = [x.strip() for x in line.split(',', 2)]
        if int(index) in [0, 1]:
            gpu_map[index] = {'uuid': uuid, 'name': name}
    assert set(gpu_map) == {'0', '1'}
    for name in ['software_environment.json', 'requirements-lock.txt']:
        reference = ROOT / ('reference_' + name)
        shutil.copy2(SOURCE / name, reference)
    environment_code = (
        'import json,platform,sys,torch,numpy,scipy; '
        'print(json.dumps(dict(python=sys.version,executable=sys.executable,'
        'os=platform.platform(),torch=torch.__version__,numpy=numpy.__version__,'
        'scipy=scipy.__version__,cuda_runtime=torch.version.cuda,'
        'cudnn=torch.backends.cudnn.version())))'
    )
    environment = json.loads(subprocess.check_output([original['python'], '-c', environment_code], text=True))
    environment.update(collected_at=datetime.now(timezone.utc).isoformat(),
                       devices=gpu_map, reference_environment='reference_software_environment.json',
                       nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                                          '--format=csv,noheader'], text=True).splitlines())
    write(ROOT / 'software_environment.json', environment)
    (ROOT / 'requirements-lock.txt').write_text(subprocess.check_output(
        [original['python'], '-m', 'pip', 'freeze'], text=True))
    write(ROOT / 'configuration_references.json', baselines)
    scripts = {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / 'prepare_experiments.py', ROOT / 'queue_worker.py', ROOT / 'launch.py',
        *[ROOT / name for name in copy_names], ROOT / 'native_export/export_native.py']}
    manifest = {
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'python': original['python'],
        'native_cwd': original['native_cwd'], 'total_jobs': len(jobs), 'jobs': jobs,
        'source_files': sources, 'input_files': inputs, 'scripts': scripts, 'copied_scripts': copied,
        'image_manifest': str(image_manifest), 'image_manifest_sha256': sha(image_manifest),
        'pretrained': original['pretrained'], 'gpu_map': gpu_map, 'gpus': [0, 1],
        'disk_min_free_bytes': 8 * 1024**3,
        'reporting_protocol': {
            'primary_seeds': [0, 1, 2], 'stability_seeds': SEEDS,
            'primary_endpoint': 'Native COCO AP, selected maximum validation AP over 160 epochs',
            'selection': 'All seeds fixed before launch; no ranking-dependent seed inclusion',
            'summaries': 'Per-seed values; mean, sample SD and Student-t 95% CI for fixed 0/1/2 and all 0..4',
            'failed_seeds': 'Recorded explicitly; no replacement seed or result-conditioned retry',
            'dataset_policy': 'Existing images, partitions and FDI maps preserved; internal validation only',
            'configuration_selection': 'Mobile tau=.50 and DSLR decay-only chosen using prior single-seed ablations; not independent confirmatory test evidence',
        },
        'checkpoint_policy': 'Preserve native stage1 rollback temporarily; after exact EMA replay retain best.pth only',
        'scheduling': 'Two detached workers share atomic task claims; wait for GPU with no other compute processes',
        'no_paper_changes': True, 'no_source_or_dataset_changes': True,
        'environment_files': {name: sha(ROOT / name) for name in ['software_environment.json', 'requirements-lock.txt']},
        'cost_estimate': {'gpu_hours_total': 37.0, 'two_gpu_ideal_hours': 19.0,
                          'best_weights_gib': 2.6, 'note': 'Estimate excludes time waiting for unrelated GPU1 job'},
    }
    write(ROOT / 'execution_manifest.json', manifest)
    write(ROOT / 'LAUNCH_FREEZE.json', {'execution_manifest_sha256': sha(ROOT / 'execution_manifest.json'),
                                      'jobs': len(jobs), 'scripts': scripts})
    print('PREPARATION_VERIFIED', len(jobs), 'fresh runs; seeds', SEEDS, flush=True)


if __name__ == '__main__':
    main()
