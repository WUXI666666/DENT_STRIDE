#!/usr/bin/env python3
"""Finite, fail-fast serial GPU0 queue. Never signals pre-existing GPU users."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = '/home/wuxi/miniconda3/envs/deimv2/bin/python'
MIN_DISK_BYTES = 5 * 1024**3
EXPECTED_EPOCHS = 160
DEFAULT_MEMORY_LIMIT_MIB = 10752
DEFAULT_MIN_FREE_MIB = 11520
MEMORY_HEADROOM_MIB = 768


def now():
    return datetime.now().astimezone().isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_resource_policy(here, min_free_mib, memory_limit_mib):
    """Require this protocol's completed preflight before using its GPU budget."""
    here = Path(here)
    preflight_path = here / 'preflight_validation.json'
    if not preflight_path.is_file():
        raise RuntimeError('Complete preflight_validation.json before starting the queue')
    preflight = read_json(preflight_path)
    if preflight.get('passed') is not True:
        raise RuntimeError('Preflight validation has not passed; queue not started')
    protocol_hash = sha256(here / 'protocol.json')
    if preflight.get('protocol_sha256') != protocol_hash:
        raise RuntimeError('Preflight protocol SHA256 differs from current protocol.json; rerun preflight')
    recommended = preflight.get('recommended_min_free_gpu_mib')
    validated_limit = preflight.get('memory_limit_mib')
    for name, value in [('min_free_mib', min_free_mib),
                        ('memory_limit_mib', memory_limit_mib),
                        ('preflight recommended_min_free_gpu_mib', recommended),
                        ('preflight memory_limit_mib', validated_limit)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if memory_limit_mib < validated_limit:
        raise ValueError('Requested memory_limit_mib is below the validated preflight memory limit')
    return {'protocol_sha256': protocol_hash,
            'preflight_validation_sha256': sha256(preflight_path),
            'cli_min_free_mib': min_free_mib,
            'preflight_recommended_min_free_gpu_mib': recommended,
            'preflight_memory_limit_mib': validated_limit,
            'memory_limit_mib': memory_limit_mib,
            'min_free_mib': max(min_free_mib, recommended,
                                memory_limit_mib + MEMORY_HEADROOM_MIB),
            'memory_headroom_mib': MEMORY_HEADROOM_MIB}


class QueueInterrupted(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'Queue received {signal.Signals(signum).name}')


@contextmanager
def exclusive_queue_lock(here):
    """Keep this FD inherited by the child, also protecting against parent SIGKILL."""
    with (Path(here) / 'queue_gpu0.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another queue or its surviving training child owns the GPU0 queue lock') from exc
        # Do not explicitly LOCK_UN: a surviving inheriting child must retain it.
        yield lock


def gpu0_snapshot():
    query = subprocess.run(
        ['nvidia-smi', '--id=0',
         '--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=15,
    )
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one physical GPU0 row, observed {rows!r}')
    fields = [field.strip() for field in rows[0].split(',')]
    if len(fields) != 6 or int(fields[0]) != 0:
        raise RuntimeError(f'Unexpected physical GPU0 query: {rows[0]}')
    return {'index': 0, 'uuid': fields[1], 'free_mib': int(fields[2]),
            'used_mib': int(fields[3]), 'total_mib': int(fields[4]),
            'utilization_percent': int(fields[5]), 'observed_at': now()}


def verify_completed(output):
    """Validate the actual completed run, not just its marker or a zero exit code."""
    output = Path(output)
    required = ['COMPLETED', 'metrics_summary.json', 'verified_coco_metrics.json', 'log.txt', 'best.pth']
    absent = [name for name in required if not (output / name).is_file()]
    if absent:
        raise RuntimeError(f'Incomplete run {output}: missing {absent}')
    weights = sorted(path.resolve() for path in output.rglob('*.pth'))
    if weights != [(output / 'best.pth').resolve()] or (output / 'best.pth.tmp').exists():
        raise RuntimeError(f'Best-only checkpoint policy failed for {output}: {weights}')
    summary = read_json(output / 'metrics_summary.json')
    verified = read_json(output / 'verified_coco_metrics.json')
    rows = [json.loads(line) for line in (output / 'log.txt').read_text().splitlines() if line.strip()]
    if [row['epoch'] for row in rows] != list(range(EXPECTED_EPOCHS)):
        raise RuntimeError(f'Expected {EXPECTED_EPOCHS} contiguous epochs in {output}')
    if summary.get('epochs_completed') != EXPECTED_EPOCHS or summary.get('final_epoch_zero_based') != EXPECTED_EPOCHS - 1:
        raise RuntimeError(f'Summary epoch counts disagree with complete log: {output}')
    if verified.get('matches_selected_checkpoint') is not True:
        raise RuntimeError(f'Best-checkpoint re-evaluation not verified: {output}')
    aps = [float(row['test_coco_eval_bbox'][0]) for row in rows]
    if any(not math.isfinite(ap) for ap in aps):
        raise RuntimeError(f'Nonfinite validation AP in {output}')
    best_index = max(range(len(rows)), key=lambda index: aps[index])
    if summary.get('best_epoch_zero_based') != best_index:
        raise RuntimeError(f'Selected epoch is not earliest global AP maximum: {output}')
    expected_ap = aps[best_index]
    if not math.isclose(float(verified['metrics_fraction']['AP']), expected_ap, abs_tol=2e-6, rel_tol=0):
        raise RuntimeError(f'Verified AP differs from global AP maximum: {output}')
    if not math.isclose(float(summary['best_metrics']['AP']), expected_ap * 100, abs_tol=2e-4, rel_tol=0):
        raise RuntimeError(f'Summary AP differs from global AP maximum: {output}')
    checkpoint_hash = sha256(output / 'best.pth')
    if verified.get('checkpoint_sha256') != checkpoint_hash or summary.get('checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError(f'Checkpoint SHA256 verification failed: {output}')
    if 'prediction_sha256' in verified:
        prediction = output / 'best_predictions.coco.json.gz'
        if not prediction.is_file() or sha256(prediction) != verified['prediction_sha256']:
            raise RuntimeError(f'Prediction export SHA256 verification failed: {output}')
    return {'output': str(output), 'epochs_completed': EXPECTED_EPOCHS,
            'best_epoch_zero_based': best_index, 'best_AP': summary['best_metrics']['AP'],
            'checkpoint_sha256': checkpoint_hash, 'verified_at': now()}


class SerialQueue:
    def __init__(self, here, protocol, tasks, python=DEFAULT_PYTHON,
                 min_free_mib=DEFAULT_MIN_FREE_MIB,
                 poll_seconds=30, gpu_query=gpu0_snapshot, disk_free=None,
                 popen=subprocess.Popen, sleep=time.sleep,
                 memory_limit_mib=DEFAULT_MEMORY_LIMIT_MIB):
        self.here = Path(here)
        self.protocol = protocol
        self.tasks = list(tasks)
        self.python = str(python)
        self.min_free_mib = min_free_mib
        self.memory_limit_mib = memory_limit_mib
        self.poll_seconds = poll_seconds
        self.gpu_query = gpu_query
        self.disk_free = disk_free or (lambda: shutil.disk_usage(self.here).free)
        self.popen = popen
        self.sleep = sleep
        self.stop_signal = None
        self.child = None
        self.child_pgid = None
        self.old_handlers = {}
        self.state = {'state': 'starting', 'gpu': 0, 'queue_pid': os.getpid(),
                      'queue_pgid': os.getpgrp(), 'jobs': self.tasks, 'completed': [],
                      'skipped': [], 'verification': {}, 'active_task': None,
                      'child_pid': None, 'child_pgid': None, 'command': None,
                      'wait_reason': None, 'started_at': now(),
                      'min_free_mib': self.min_free_mib,
                      'memory_limit_mib': self.memory_limit_mib}
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError('Provide a nonempty task list without duplicates')
        unknown = set(self.tasks) - set(protocol['runs'])
        if unknown:
            raise ValueError(f'Unknown tasks: {sorted(unknown)}')
        outputs = [Path(protocol['runs'][task]['output']).resolve() for task in self.tasks]
        if len(set(outputs)) != len(outputs):
            raise ValueError('Every task must have its own unique output directory')
        if min_free_mib < 1 or not 1 <= poll_seconds <= 60:
            raise ValueError('min_free_mib must be positive and poll_seconds must be within 1..60')
        if memory_limit_mib < 1 or memory_limit_mib >= min_free_mib:
            raise ValueError('memory_limit_mib must be positive and below min_free_mib')

    def record(self, **changes):
        self.state.update(changes, updated_at=now())
        write_json(self.here / 'queue_state.json', self.state)

    def signal_handler(self, signum, frame):
        # Only set a flag, avoiding an exception between Popen() and recording ownership.
        self.stop_signal = signum

    def check_interrupted(self):
        if self.stop_signal is not None:
            raise QueueInterrupted(self.stop_signal)

    def install_handlers(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.old_handlers[signum] = signal.signal(signum, self.signal_handler)

    def restore_handlers(self):
        for signum, handler in self.old_handlers.items():
            signal.signal(signum, handler)
        self.old_handlers.clear()

    def wait_for_resources(self, task):
        while True:
            self.check_interrupted()
            disk_bytes = int(self.disk_free())
            if disk_bytes < MIN_DISK_BYTES:
                raise RuntimeError(f'Less than 5 GiB free disk; not starting {task}')
            snapshot = self.gpu_query()
            self.check_interrupted()
            if snapshot['index'] != 0:
                raise RuntimeError('Refusing a non-GPU0 resource snapshot')
            self.record(gpu_snapshot=snapshot, disk_free_bytes=disk_bytes)
            if snapshot['free_mib'] >= self.min_free_mib:
                return
            reason = f"GPU0 free {snapshot['free_mib']} MiB < required {self.min_free_mib} MiB; other processes untouched"
            self.record(state='waiting_resources', active_task=task, wait_reason=reason)
            # Sleep in short slices so a requested stop is handled promptly.
            for _ in range(self.poll_seconds):
                self.sleep(1)
                self.check_interrupted()

    def terminate_owned_child(self):
        child, pgid = self.child, self.child_pgid
        if child is None or pgid is None:
            return
        if pgid != child.pid or pgid == os.getpgrp() or pgid <= 1:
            raise RuntimeError('Refusing to signal a process group not created by this queue')
        self.record(state='terminating_owned_child', child_pid=child.pid, child_pgid=pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=30)
        # A terminated leader may leave only its own data workers in the same group.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child = None
        self.child_pgid = None
        self.record(child_pid=None, child_pgid=None)

    def run(self):
        with exclusive_queue_lock(self.here) as lock:
            self.install_handlers()
            try:
                self.record()
                env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                for task in self.tasks:
                    self.check_interrupted()
                    output = Path(self.protocol['runs'][task]['output'])
                    self.record(active_task=task, state='checking_task', wait_reason=None,
                                child_pid=None, child_pgid=None, command=None)
                    if (output / 'COMPLETED').exists():
                        self.state['verification'][task] = verify_completed(output)
                        self.state['completed'].append(task)
                        self.state['skipped'].append(task)
                        self.record()
                        continue
                    if (output / 'log.txt').exists() or (output / 'best.pth').exists():
                        raise RuntimeError(f'Refusing an incomplete existing run: {output}; inspect it before a new attempt')
                    self.wait_for_resources(task)
                    self.check_interrupted()
                    output.mkdir(parents=True, exist_ok=True)
                    command = [self.python, '-B', str(self.here / 'run_experiment.py'),
                               '--task', task, '--gpu', '0',
                               f'--memory-limit-mib={self.memory_limit_mib}']
                    self.record(state='launching', command=command, wait_reason=None)
                    with (output / 'training_stdout.log').open('a', buffering=1) as log:
                        self.child = self.popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=env, start_new_session=True, pass_fds=(lock.fileno(),))
                        self.child_pgid = self.child.pid
                        self.record(state='running', child_pid=self.child.pid, child_pgid=self.child_pgid)
                        while True:
                            self.check_interrupted()
                            returncode = self.child.poll()
                            if returncode is not None:
                                break
                            self.sleep(1)
                        self.child.wait()  # Explicitly reap our own child before advancing.
                        if returncode:
                            self.record(child_returncode=returncode)
                            raise RuntimeError(f'Task {task} failed with exit status {returncode}; remaining tasks not started')
                        self.child = None
                        self.child_pgid = None
                    self.record(state='verifying', child_pid=None, child_pgid=None, child_returncode=returncode)
                    self.state['verification'][task] = verify_completed(output)
                    self.state['completed'].append(task)
                    self.record()
                self.record(state='completed', active_task=None, wait_reason=None, completed_at=now())
                return 0
            except BaseException as exc:
                try:
                    self.terminate_owned_child()
                finally:
                    self.record(state='interrupted' if isinstance(exc, QueueInterrupted) else 'failed',
                                error=repr(exc), stop_signal=self.stop_signal, wait_reason=None)
                if isinstance(exc, QueueInterrupted):
                    return 128 + exc.signum
                raise
            finally:
                self.restore_handlers()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', help='Explicit serial order; default protocol.queue_order')
    parser.add_argument('--gpu', type=int, choices=[0], default=0)
    parser.add_argument('--gpu0', action='store_true', help='Alias confirming physical GPU0')
    parser.add_argument('--min-free-mib', type=int, default=DEFAULT_MIN_FREE_MIB,
                        help='Lower bound; preflight recommendation and memory-limit + 768 MiB also apply')
    parser.add_argument('--memory-limit-mib', type=int, default=DEFAULT_MEMORY_LIMIT_MIB,
                        help='Per-process training allocator cap; must not be below the preflight cap')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    args = parser.parse_args(argv)
    protocol = read_json(HERE / 'protocol.json')
    resource_policy = preflight_resource_policy(HERE, args.min_free_mib, args.memory_limit_mib)
    tasks = args.tasks if args.tasks is not None else protocol['queue_order']
    queue = SerialQueue(HERE, protocol, tasks, python=args.python,
                        min_free_mib=resource_policy['min_free_mib'], poll_seconds=args.poll_seconds,
                        memory_limit_mib=args.memory_limit_mib)
    queue.state['resource_policy'] = resource_policy
    return queue.run()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'[serial-gpu0-queue] {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
