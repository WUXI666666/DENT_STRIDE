#!/usr/bin/env python3
"""Detached finite queue: GPU0 Phone; GPU1 DSLR then DENTEX. No polling service."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from prepare_migration import HERE, ROOT, write_json


def now():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    # Share the physical GPU lock with the preceding official-baseline queue.
    lock = (ROOT / f'queue_gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = ['phone_dualopt'] if args.gpu == 0 else ['dslr_cmfa_dualopt', 'dentex_spatial_shape']
    protocol = json.loads((HERE / 'protocol.json').read_text())
    status = HERE / f'queue_gpu{args.gpu}.json'
    done = []
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for task in jobs:
        output = ROOT / 'runs/deim_s_paper_modules' / task / 'seed_0'
        assert str(output) == protocol['runs'][task]['output']
        if (output / 'COMPLETED').exists():
            done.append(task)
            continue
        if shutil.disk_usage(ROOT).free < 3 * 1024**3:
            write_json(status, {'state': 'failed', 'error': 'less than 3 GiB free', 'completed': done, 'updated_at': now()})
            raise RuntimeError('Insufficient disk space; no next experiment started')
        output.mkdir(parents=True, exist_ok=True)
        state = {'state': 'running', 'gpu': args.gpu, 'queue_pid': os.getpid(), 'active_task': task,
                 'completed': done, 'jobs': jobs, 'updated_at': now()}
        write_json(status, state)
        command = [sys.executable, '-B', str(HERE / 'run_migrated.py'), '--task', task, '--gpu', str(args.gpu)]
        with (output / 'training_stdout.log').open('a', buffering=1) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            state.update(state='failed', returncode=result.returncode, updated_at=now())
            write_json(status, state)
            raise SystemExit(result.returncode)
        done.append(task)
    write_json(status, {'state': 'completed', 'gpu': args.gpu, 'completed': done, 'updated_at': now()})


if __name__ == '__main__':
    main()
