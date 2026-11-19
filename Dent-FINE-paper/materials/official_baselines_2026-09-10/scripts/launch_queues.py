#!/usr/bin/env python3
"""Detach validated comparison queues so closing this conversation does not stop them."""
import subprocess
import sys
from prepare_comparison import ROOT, write_json


def main():
    if (ROOT / 'queue_launch.json').exists():
        raise RuntimeError('Queues already launched; inspect their status instead of duplicating them.')
    launches = []
    for model, gpu in [('deim_s', 0), ('rtdetrv2_s', 1)]:
        marker = ROOT / 'smoke_runs' / model / 'phone/seed_0/COMPLETED'
        if not marker.is_file():
            raise RuntimeError(f'Missing successful native smoke test: {marker}')
    for model, gpu in [('deim_s', 0), ('rtdetrv2_s', 1)]:
        log_path = ROOT / f'queue_gpu{gpu}.log'
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_queue.py'),
                   '--model', model, '--gpu', str(gpu)]
        with log_path.open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        launches.append({'model': model, 'gpu': gpu, 'pid': process.pid, 'command': command, 'log': str(log_path)})
    write_json(ROOT / 'queue_launch.json', launches)
    for launch in launches:
        print(launch)


if __name__ == '__main__':
    main()
