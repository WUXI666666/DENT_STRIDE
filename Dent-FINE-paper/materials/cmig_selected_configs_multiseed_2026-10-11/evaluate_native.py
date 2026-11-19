"""Strict best-checkpoint replay through the historical native train.py entry.

The private child process only observes native setup/evaluation. It does not
replace the model, loader, EMA, postprocessor, precision, or configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
ORIGINAL_CWD = Path('/home/wuxi/Dent-FINE')
METRIC_NAMES = [
    'AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
    'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge',
]
VERIFIED_METRICS = ('AP', 'AP50', 'AP75', 'AR100')
REPLAY_ABS_TOLERANCE = 2e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def paths(job):
    implementation_value = job.get('implementation') or job.get('implementation_dir')
    if not implementation_value:
        raise ValueError('Job needs implementation or implementation_dir')
    implementation = Path(implementation_value).resolve()
    train_script = Path(job.get('train_script', implementation / 'train.py')).resolve()
    if train_script.parent != implementation:
        raise ValueError('Native train.py must belong to the selected implementation')
    output = Path(job['output_dir']).resolve()
    return implementation, train_script, output


def evaluation_directory(job, prior=None, factor_inference=None):
    if prior is not None and factor_inference is not None:
        raise ValueError('Prior and factor inference overrides are mutually exclusive')
    _, _, output = paths(job)
    return output / (
        f'native_prior_{prior}' if prior is not None else
        f'native_factor_{factor_inference}' if factor_inference is not None else
        'native_best_evaluation'
    )


def native_arguments(job, prior=None, factor_inference=None):
    _, train_script, output = paths(job)
    destination = evaluation_directory(job, prior, factor_inference)
    return [
        str(train_script), '-c', str(Path(job['config']).resolve()),
        '--seed', str(job['seed']), '--use-amp', '-d', 'cuda',
        '--resume', str(output / 'best.pth'), '--test-only',
        '--output-dir', str(destination),
        '--summary-dir', str(destination / 'tensorboard'),
    ]


def effective_job_controls(job, factor_inference=None):
    controls = dict(job.get('controls', {}))
    if factor_inference is not None:
        controls['factor_inference_fusion'] = factor_inference == 'on'
    return controls


def install_job_controls(job, factor_inference=None):
    controls = effective_job_controls(job, factor_inference)
    if not controls:
        return None
    controls_path = Path(job.get(
        'controls_script',
        ROOT / 'controlled_runtime.py' if (ROOT / 'controlled_runtime.py').is_file()
        else ROOT.parent / 'cmig_controlled_2026-09-29' / 'controlled_runtime.py',
    )).resolve()
    spec = importlib.util.spec_from_file_location('historical_replay_controls', controls_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import runtime controls: {controls_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_controls(controls)
    return {'path': str(controls_path), 'sha256': sha(controls_path), 'settings': controls}


def native_child(job, prior=None, factor_inference=None):
    implementation, train_script, output = paths(job)
    sys.path.insert(0, str(implementation))
    import torch
    import engine
    from engine.solver import det_solver
    from engine.solver import _solver

    if Path(engine.__file__).resolve().parent != implementation / 'engine':
        raise RuntimeError(f'Wrong engine imported: {engine.__file__}')
    torch.set_num_threads(4)
    # Resolve the effective flag before the single runtime installation. An
    # explicit ON replay must not inherit an already-installed OFF wrapper.
    controls = install_job_controls(job, factor_inference)
    captured = {'calls': 0}
    original_setup = _solver.BaseSolver._setup

    def observe_setup(solver, *args, **kwargs):
        result = original_setup(solver, *args, **kwargs)
        # EMA parameters are intentionally all requires_grad=False. Count the
        # original model instead, which retains the native trainability flags.
        captured['solver'] = solver
        captured['params_total'] = sum(p.numel() for p in solver.model.parameters())
        captured['params_trainable'] = sum(
            p.numel() for p in solver.model.parameters() if p.requires_grad
        )
        return result

    _solver.BaseSolver._setup = observe_setup
    original_evaluate = det_solver.evaluate

    def observe_evaluate(*args, **kwargs):
        postprocessor = args[2] if len(args) > 2 else kwargs['postprocessor']
        if prior is not None:
            # Change inference flags only, after native setup/checkpoint load.
            # Prior weights and the frozen YAML/model/checkpoint stay unchanged.
            postprocessor.use_dental_spatial_prior = prior in ('center', 'both')
            postprocessor.use_dental_shape_prior = prior in ('shape', 'both')
            postprocessor.use_dental_view_prior = False
            postprocessor.use_dental_appearance_prior = False
        captured['postprocessor_priors'] = {
            name: getattr(postprocessor, name) for name in (
                'use_dental_spatial_prior', 'use_dental_shape_prior',
                'use_dental_view_prior', 'use_dental_appearance_prior',
                'spatial_prior_weight', 'shape_prior_weight', 'view_prior_weight',
                'appearance_prior_weight',
            )
        }
        result = original_evaluate(*args, **kwargs)
        captured['calls'] += 1
        stats = result[0]['coco_eval_bbox']
        if len(stats) != len(METRIC_NAMES) or not all(math.isfinite(float(v)) for v in stats):
            raise RuntimeError('Native evaluator returned invalid COCO statistics')
        solver = captured['solver']
        captured['statistics'] = [float(v) for v in stats]
        captured['checkpoint_internal_last_epoch'] = int(solver.last_epoch)
        captured['evaluated_ema'] = bool(solver.ema is not None)
        captured['evaluation_model_params_total'] = sum(p.numel() for p in args[0].parameters())
        return result

    det_solver.evaluate = observe_evaluate
    argv = native_arguments(job, prior, factor_inference)
    sys.argv = argv
    runpy.run_path(str(train_script), run_name='__main__')
    if captured['calls'] != 1:
        raise RuntimeError(f"Expected one native evaluation, got {captured['calls']}")
    if captured['evaluation_model_params_total'] != captured['params_total']:
        raise RuntimeError('EMA/original model parameter counts differ')
    captured.pop('solver')
    captured.update({
        'native_arguments': argv,
        'prior_override': prior,
        'factor_inference_override': factor_inference,
        'effective_controls': effective_job_controls(job, factor_inference),
        'implementation': str(implementation),
        'engine_import': str(Path(engine.__file__).resolve()),
        'native_train_script_sha256': sha(train_script),
        'native_evaluator_source_sha256': sha(Path(det_solver.__file__).with_name('det_engine.py')),
        'controls': controls,
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'validation_precision': 'native FP32; no validation autocast added',
    })
    atomic_json(evaluation_directory(job, prior, factor_inference) / 'captured_metrics.json', captured)


def validate_log(output):
    rows = [
        json.loads(line) for line in (output / 'log.txt').read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 160 or [r.get('epoch') for r in rows] != list(range(160)):
        raise RuntimeError('Expected exactly 160 unique, consecutive logged epochs 0..159')
    for row in rows:
        values = row.get('test_coco_eval_bbox', [])
        if len(values) != len(METRIC_NAMES) or not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(f"Invalid native validation metrics at epoch {row['epoch']}")
    # Python max preserves the first row on exact AP ties.
    best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
    selection_path = output / 'best_selection.json'
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        epoch = selection.get('epoch_zero_based', selection.get('best_epoch_zero_based'))
        ap = selection.get('AP_fraction')
        if epoch != best['epoch'] or ap is None or abs(ap - best['test_coco_eval_bbox'][0]) > 1e-12:
            raise RuntimeError('Existing best_selection.json is not the first global logged AP maximum')
    return rows, best


def replay(job_path, prior=None, factor_inference=None):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    implementation, train_script, output = paths(job)
    if not output.is_dir() or not (output / 'best.pth').is_file():
        raise RuntimeError('Formal training output and best.pth must already exist')
    if (output / 'best.pth').is_symlink():
        raise RuntimeError('Refusing a symlink best checkpoint')
    config = Path(job['config']).resolve()
    config_hash = sha(config)
    if job.get('config_sha256') and config_hash != job['config_sha256']:
        raise RuntimeError('Job configuration hash changed')
    if (output / 'config.yml').is_file() and sha(output / 'config.yml') != config_hash:
        raise RuntimeError('Run config.yml differs from the job configuration')
    checkpoint_hash = sha(output / 'best.pth')
    log_hash = sha(output / 'log.txt')
    rows, best = validate_log(output)
    destination = evaluation_directory(job, prior, factor_inference)
    inference_override = prior is not None or factor_inference is not None
    destination.mkdir(parents=True, exist_ok=True)
    capture_path = destination / 'captured_metrics.json'
    # A failed replay must not accidentally accept a stale successful capture.
    if capture_path.exists():
        raise RuntimeError(f'Native replay already has a capture; review before re-running: {capture_path}')
    python = str(job.get('python', sys.executable))
    command = [python, '-u', str(Path(__file__).resolve()), str(job_path), '--native-child']
    if prior is not None:
        command.extend(['--prior', prior])
    if factor_inference is not None:
        command.extend(['--factor-inference', factor_inference])
    environment = os.environ.copy()
    environment.update({
        'PYTHONPATH': str(implementation), 'PYTHONHASHSEED': str(job['seed']),
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
    })
    started = time.time()
    with (destination / 'console.log').open('x') as console:
        subprocess.run(command, cwd=ORIGINAL_CWD, env=environment,
                       stdout=console, stderr=subprocess.STDOUT, check=True)
    if sha(config) != config_hash or sha(output / 'best.pth') != checkpoint_hash or sha(output / 'log.txt') != log_hash:
        raise RuntimeError('Configuration, checkpoint, or training log changed during native replay')
    capture = json.loads(capture_path.read_text())
    if not (destination / 'eval.pth').is_file():
        raise RuntimeError('Native val() did not save its COCO evaluator payload')
    native_values = capture['statistics']
    logged_values = best['test_coco_eval_bbox']
    differences = {}
    for name in VERIFIED_METRICS:
        index = METRIC_NAMES.index(name)
        delta = float(native_values[index]) - float(logged_values[index])
        differences[name] = delta
        if not inference_override and abs(delta) > REPLAY_ABS_TOLERANCE:
            raise RuntimeError(
                f'Native best replay differs from global logged maximum for {name}: '
                f'{native_values[index]} vs {logged_values[index]} (delta={delta})'
            )
    if capture['params_trainable'] != best['n_parameters']:
        raise RuntimeError('Native model trainable parameter count differs from the training log')
    selection_path = output / 'best_selection.json'
    if not inference_override and not selection_path.exists():
        atomic_json(selection_path, {
            'epoch_zero_based': best['epoch'],
            'epoch_one_based': best['epoch'] + 1,
            'AP_fraction': logged_values[0],
            'native_state_last_epoch': capture['checkpoint_internal_last_epoch'],
            'selection': 'global maximum native validation AP; first logged epoch on exact ties',
            'evidence': 'complete native training log and strict independent native best-checkpoint replay',
            'derived_after_native_replay': True,
        })
    metrics = dict(zip(METRIC_NAMES, native_values))
    provenance_path = output / (
        f'native_prior_{prior}_provenance.json' if prior is not None
        else f'native_factor_{factor_inference}_provenance.json' if factor_inference is not None
        else 'native_evaluation_provenance.json'
    )
    report_path = output / (
        f'native_prior_{prior}_metrics.json' if prior is not None else
        f'native_factor_{factor_inference}_metrics.json' if factor_inference is not None else
        'native_coco_metrics.json'
    )
    provenance = {
        'job_file': str(job_path), 'job_file_sha256': sha(job_path),
        'config': str(config), 'config_sha256': config_hash,
        'checkpoint': str(output / 'best.pth'), 'checkpoint_sha256': checkpoint_hash,
        'training_log_sha256': log_hash, 'evaluator_script_sha256': sha(__file__),
        'native_train_script': str(train_script),
        'native_train_script_sha256': sha(train_script),
        'native_eval_payload': str(destination / 'eval.pth'),
        'native_eval_payload_sha256': sha(destination / 'eval.pth'),
        'child_command': command, 'native_arguments': native_arguments(job, prior, factor_inference),
        'prior_override': prior,
        'factor_inference_override': factor_inference,
        'effective_controls': effective_job_controls(job, factor_inference),
        'cwd': str(ORIGINAL_CWD),
        'environment': {name: environment.get(name) for name in [
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONHASHSEED',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'DETERMINISTIC', 'STRICT_DETERMINISTIC',
        ]},
        'started_unix': started, 'finished_unix': time.time(),
        'native_capture': capture,
    }
    report = {
        'job_id': job.get('id', job.get('job_id')),
        'configuration': job.get('configuration', job.get('task')),
        'cohort': job.get('cohort'), 'seed': job['seed'],
        'config_sha256': config_hash, 'checkpoint_sha256': checkpoint_hash,
        'metric_names': METRIC_NAMES, 'metrics_fraction': metrics,
        'metrics_percent': {name: value * 100 if value >= 0 else None for name, value in metrics.items()},
        'params_total': capture['params_total'],
        'params_trainable': capture['params_trainable'],
        'best_epoch_zero_based': best['epoch'], 'best_epoch_one_based': best['epoch'] + 1,
        'epochs_completed': len(rows), 'final_epoch_zero_based': rows[-1]['epoch'],
        'checkpoint_internal_last_epoch': capture['checkpoint_internal_last_epoch'],
        'checkpoint_epoch_note': 'Native stage rollback can reset last_epoch; selected epoch is from the verified full training log.',
        'logged_best_metrics_fraction': dict(zip(METRIC_NAMES, logged_values)),
        'native_replay_minus_logged_fraction': differences,
        'strict_replay_tolerance_fraction': REPLAY_ABS_TOLERANCE if not inference_override else None,
        'selection_metrics_agreement_checked': not inference_override,
        'different_prior_metrics_expected': prior is not None,
        'different_factor_metrics_expected': factor_inference is not None,
        'verified_best_checkpoint': True, 'authoritative_native_evaluation': True,
        'precision': capture['validation_precision'], 'evaluated_ema': capture['evaluated_ema'],
        'controls': effective_job_controls(job, factor_inference),
        'prior_override': prior,
        'factor_inference_override': factor_inference,
        'postprocessor_priors': capture['postprocessor_priors'],
        'postprocessor_policy': (
            'same native model/checkpoint/config; only center/shape inference flags switched; '
            'view and appearance flags disabled; configured prior weights unchanged'
            if prior is not None else 'exact job configuration; native historical priors are not disabled'
        ),
        'factor_fusion_policy': (
            'explicit inference-only fusion override installed once before native setup; '
            'training configuration and all other controls preserved'
            if factor_inference is not None else 'configured native/runtime-control inference fusion'
        ),
        'independent_test': False,
        'provenance': str(provenance_path),
    }
    atomic_json(provenance_path, provenance)
    atomic_json(report_path, report)
    print('NATIVE_PRIOR_REPLAY_VERIFIED' if prior is not None else
          'NATIVE_FACTOR_REPLAY_VERIFIED' if factor_inference is not None else
          'NATIVE_BEST_REPLAY_VERIFIED',
          report['job_id'], f'prior={prior if prior is not None else "configured"}',
          f'factor={factor_inference if factor_inference is not None else "configured"}',
          f"AP={metrics['AP'] * 100:.9f}", f"epoch={best['epoch'] + 1}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', help='Frozen job JSON')
    overrides = parser.add_mutually_exclusive_group()
    overrides.add_argument('--prior', choices=['none', 'center', 'shape', 'both'],
                           help='Replay the same checkpoint with only inference-prior flags changed')
    overrides.add_argument('--factor-inference', choices=['on', 'off'],
                           help='Replay the same checkpoint with inference-only factor fusion changed')
    parser.add_argument('--native-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_child:
        native_child(json.loads(Path(args.job).read_text()), args.prior, args.factor_inference)
    else:
        replay(args.job, args.prior, args.factor_inference)


if __name__ == '__main__':
    main()
