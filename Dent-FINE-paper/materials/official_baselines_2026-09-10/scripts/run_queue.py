#!/usr/bin/env python3
"""One native upstream model per GPU; nine prespecified cohort/seed runs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from prepare_comparison import ROOT, MODELS, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()
    lock = (ROOT / f'queue_gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [(cohort, seed) for seed in (0, 1, 2) for cohort in ('phone', 'dslr', 'dentex')]
    status_file = ROOT / f'queue_gpu{args.gpu}.json'
    done = []
    for cohort, seed in jobs:
        output = ROOT / 'runs' / args.model / cohort / f'seed_{seed}'
        if (output / 'COMPLETED').exists():
            done.append({'cohort': cohort, 'seed': seed})
            continue
        if shutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise RuntimeError('Low disk space; stopping before next job')
        output.mkdir(parents=True, exist_ok=True)
        entry = 'run_official_safe_images.py' if args.model == 'rtdetrv2_s' else 'run_official.py'
        command = [sys.executable, str(ROOT / 'scripts' / entry), '--model', args.model,
                   '--cohort', cohort, '--seed', str(seed), '--gpu', str(args.gpu)]
        state = {'state': 'running', 'queue_pid': os.getpid(), 'model': args.model, 'gpu': args.gpu,
                 'active_cohort': cohort, 'active_seed': seed, 'completed': done,
                 'total': len(jobs), 'updated_at': datetime.now().astimezone().isoformat()}
        write_json(status_file, state)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        print(f'Starting {args.model} {cohort} seed={seed} on GPU {args.gpu}', flush=True)
        with (output / 'training_stdout.log').open('a', buffering=1) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            state.update(state='failed', returncode=result.returncode, log=str(output / 'training_stdout.log'))
            write_json(status_file, state)
            raise SystemExit(result.returncode)
        done.append({'cohort': cohort, 'seed': seed})
    write_json(status_file, {'state': 'completed', 'model': args.model, 'gpu': args.gpu,
                            'completed': done, 'total': len(jobs), 'updated_at': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
