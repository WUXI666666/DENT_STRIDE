"""GPU-safe detached worker. Each run is fresh and all scheduled seeds are reported."""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
CHILD = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify(manifest, job=None):
    frozen = json.loads((ROOT / 'LAUNCH_FREEZE.json').read_text())
    assert sha(ROOT / 'execution_manifest.json') == frozen['execution_manifest_sha256']
    for name, digest in manifest['scripts'].items():
        assert sha(ROOT / name) == digest, 'Execution script changed: ' + name
    for row in manifest['source_files']:
        assert sha(row['path']) == row['sha256'], 'Model source changed: ' + row['path']
    for row in manifest['input_files']:
        assert sha(row['path']) == row['sha256'], 'Input changed: ' + row['path']
    assert sha(manifest['image_manifest']) == manifest['image_manifest_sha256']
    for row in json.loads(Path(manifest['image_manifest']).read_text()):
        st = Path(row['path']).stat()
        assert st.st_size == row['size'] and st.st_mtime_ns == row['mtime_ns'], row['path']
    if job is not None:
        assert json.loads(Path(job['job_file']).read_text()) == job
        assert sha(job['config']) == job['config_sha256']
    assert shutil.disk_usage(ROOT).free > manifest['disk_min_free_bytes'], 'Free disk below 8 GiB safety reserve'


def summary_group(rows, seeds):
    from scipy.stats import t
    present = {r['seed'] for r in rows}
    chosen = [r for r in rows if r['seed'] in seeds]
    complete = present.issuperset(seeds)
    result = {'scheduled_seeds': seeds, 'completed_seeds': sorted(r['seed'] for r in chosen),
              'complete': complete, 'metrics': {}}
    if complete:
        n = len(seeds)
        for name in ['AP', 'AP50', 'AP75', 'AR100']:
            vals = [r[name] for r in chosen]
            mean, sd = statistics.mean(vals), statistics.stdev(vals)
            half = float(t.ppf(.975, n - 1)) * sd / n**.5
            result['metrics'][name] = {'mean': mean, 'SD': sd, 'n': n,
                                      'CI95': [mean - half, mean + half]}
    return result


def aggregate(manifest):
    with (ROOT / 'aggregate.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for job in manifest['jobs']:
            out = Path(job['output_dir'])
            if not (out / 'COMPLETED.json').exists():
                continue
            marker = json.loads((out / 'COMPLETED.json').read_text())
            assert sha(out / 'best.pth') == marker['checkpoint_sha256']
            assert sha(out / 'metrics_summary.json') == marker['metrics_sha256']
            result = json.loads((out / 'metrics_summary.json').read_text())
            metrics = result['metrics_percent']
            rows.append({'configuration': job['configuration'], 'cohort': job['cohort'], 'seed': job['seed'],
                         **{k: metrics[k] for k in ['AP', 'AP50', 'AP75', 'AR100']},
                         'best_epoch_one_based': result['best_epoch_one_based'],
                         'params_total': result['params_total'], 'checkpoint': str(out / 'best.pth'),
                         'metrics': str(out / 'metrics_summary.json')})
        if rows:
            temp = ROOT / 'results_per_seed.csv.tmp'
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp, ROOT / 'results_per_seed.csv')
        groups = []
        for name in dict.fromkeys(j['configuration'] for j in manifest['jobs']):
            group = [r for r in rows if r['configuration'] == name]
            groups.append({'configuration': name,
                           'primary_three_seed': summary_group(group, [0, 1, 2]),
                           'all_five_seed_stability': summary_group(group, [0, 1, 2, 3, 4])})
        atomic(ROOT / 'fixed_seed_mean_sd_ci.json', groups)
        atomic(ROOT / 'overall_progress.json', {'completed': len(rows), 'scheduled': len(manifest['jobs']),
                                               'all_completed': len(rows) == len(manifest['jobs'])})


def pending_jobs(manifest):
    return [j for j in manifest['jobs'] if not (ROOT / 'claims' / (j['id'] + '.json')).exists()]


def claim(manifest, gpu):
    with (ROOT / 'claims.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = pending_jobs(manifest)
        if not pending:
            return None
        job = pending[0]
        atomic(ROOT / 'claims' / (job['id'] + '.json'),
               {'job': job['id'], 'gpu': gpu, 'worker_pid': os.getpid(),
                'claimed_unix': time.time(), 'status': 'claimed'})
        return job


def gpu_compute_pids(uuid):
    raw = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True)
    return [int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()]


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    gpu = args.gpu
    manifest = json.loads((ROOT / 'execution_manifest.json').read_text())
    uuid = manifest['gpu_map'][str(gpu)]['uuid']
    lock_root = ROOT.parent / '.gpu_queue_locks'
    lock_root.mkdir(exist_ok=True)
    gpu_lock = (lock_root / (uuid + '.lock')).open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statepath = ROOT / ('gpu' + str(gpu) + '_state.json')
    state = {'gpu': gpu, 'gpu_uuid': uuid, 'worker_pid': os.getpid(),
             'started_unix': time.time(), 'state': 'preparing', 'completed': []}
    atomic(statepath, state)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', DETERMINISTIC='0', STRICT_DETERMINISTIC='0', PYTHONDONTWRITEBYTECODE='1')
    job = None

    def terminate(signum, frame):
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
        state['state'] = 'interrupted'
        atomic(statepath, state)
        if job is not None:
            atomic(Path(job['output_dir']) / 'INTERRUPTED.json',
                   {'job': job['id'], 'signal': signum, 'unix': time.time()})
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def run(command, logpath, phase):
        global CHILD
        state.update(state=phase, command=command)
        atomic(statepath, state)
        with logpath.open('a') as console:
            CHILD = subprocess.Popen(command, cwd=job['native_cwd'], env=env,
                                     stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            state['child_pid'] = CHILD.pid
            atomic(statepath, state)
            code = CHILD.wait()
        CHILD = None
        state.pop('child_pid', None)
        if code:
            raise RuntimeError(f'{phase} failed, exit {code}; see {logpath}')

    try:
        verify(manifest)
        while True:
            if not pending_jobs(manifest):
                state['state'] = 'queue_drained'
                state['finished_unix'] = time.time()
                atomic(statepath, state)
                aggregate(manifest)
                return
            others = gpu_compute_pids(uuid)
            if others:
                state.update(state='waiting_for_gpu', external_compute_pids=others)
                atomic(statepath, state)
                time.sleep(30)
                continue
            state.pop('external_compute_pids', None)
            verify(manifest)
            job = claim(manifest, gpu)
            if job is None:
                continue
            verify(manifest, job)
            out = Path(job['output_dir'])
            assert not out.exists(), 'A formal run must not reuse partial outputs: ' + str(out)
            out.mkdir(parents=True)
            shutil.copy2(job['config'], out / 'config.yml')
            atomic(out / 'job.json', job)
            state.update(current_job=job['id'], output_dir=str(out))
            atomic(statepath, state)
            env.update(PYTHONHASHSEED=str(job['seed']), PYTHONPATH=job['implementation'])
            python = manifest['python']
            run([python, '-u', str(ROOT / 'smoke_controlled.py'), job['job_file']],
                out / 'preflight.log', 'preflight')
            check = json.loads((out / 'preflight.json').read_text())
            assert check['passed'] and check['config_sha256'] == job['config_sha256']
            verify(manifest, job)
            run([python, '-u', str(ROOT / 'train_historical.py'), job['job_file']],
                out / 'console.log', 'training')
            run([python, '-u', str(ROOT / 'evaluate_native.py'), job['job_file']],
                out / 'native_evaluation_driver.log', 'native_best_replay')
            run([python, '-u', str(ROOT / 'native_export/export_native.py'), job['job_file']],
                out / 'prediction_export.log', 'native_prediction_export')
            if job['cohort'] == 'Pano':
                for mode in ['none', 'center', 'shape', 'both']:
                    run([python, '-u', str(ROOT / 'evaluate_native.py'), job['job_file'], '--prior', mode],
                        out / ('native_prior_' + mode + '_driver.log'), 'same_weight_prior_replay')
                    run([python, '-u', str(ROOT / 'native_export/export_native.py'), job['job_file'], '--prior', mode],
                        out / ('prior_' + mode + '_export.log'), 'same_weight_prior_export')
            result = json.loads((out / 'metrics_summary.json').read_text())
            assert result['checkpoint_sha256'] == sha(out / 'best.pth')
            assert result['config_sha256'] == job['config_sha256']
            assert all(abs(v) <= 2e-6 for v in result['native_export_minus_reference_fraction'].values())
            removed = []
            for name in ['best_stg1.pth', 'best_stg2.pth', 'last.pth']:
                p = out / name
                if p.exists():
                    assert p.parent.resolve() == out.resolve() and not p.is_symlink()
                    p.unlink()
                    removed.append(name)
            assert [p.name for p in out.glob('*.pth')] == ['best.pth']
            atomic(out / 'checkpoint_retention.json', {'retained': 'best.pth',
                    'removed_temporary_training_stage_weights': removed,
                    'native_stage_behavior_preserved': True, 'native_best_replay_verified': True})
            atomic(out / 'COMPLETED.json', {'job': job['id'], 'checkpoint_sha256': sha(out / 'best.pth'),
                   'metrics_sha256': sha(out / 'metrics_summary.json'),
                   'native_metrics_sha256': sha(out / 'native_coco_metrics.json'), 'finished_unix': time.time()})
            atomic(ROOT / 'claims' / (job['id'] + '.json'), {'job': job['id'], 'gpu': gpu,
                   'worker_pid': os.getpid(), 'status': 'completed', 'finished_unix': time.time()})
            state['completed'].append(job['id'])
            state['state'] = 'job_completed'
            atomic(statepath, state)
            aggregate(manifest)
            print('JOB_COMPLETED', job['id'], result['metrics_percent']['AP'], flush=True)
            job = None
    except Exception as exc:
        state.update(state='failed_stopped', error=str(exc), traceback=traceback.format_exc())
        atomic(statepath, state)
        if job is not None:
            failure = {'job': job['id'], 'gpu': gpu, 'error': str(exc), 'unix': time.time()}
            Path(job['output_dir']).mkdir(parents=True, exist_ok=True)
            atomic(Path(job['output_dir']) / 'FAILED.json', failure)
            atomic(ROOT / 'claims' / (job['id'] + '.json'), dict(failure, status='failed'))
        print(state['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
