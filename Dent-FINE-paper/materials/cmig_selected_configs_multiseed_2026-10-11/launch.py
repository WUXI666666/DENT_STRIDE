"""Launch two detached workers once, without taking over any occupied GPU."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT))
    import queue_worker
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT / 'LAUNCHED.json').exists(), 'Already launched; do not start duplicate workers'
        manifest = json.loads((ROOT / 'execution_manifest.json').read_text())
        queue_worker.verify(manifest)
        (ROOT / 'claims').mkdir()
        record = {'launched_unix': time.time(), 'workers': {},
                  'protocol': manifest['reporting_protocol'], 'not_monitoring_training': True}
        for gpu in [0, 1]:
            with (ROOT / ('gpu' + str(gpu) + '_queue.log')).open('x') as console:
                command = [manifest['python'], '-u', str(ROOT / 'queue_worker.py'), '--gpu', str(gpu)]
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=console, stderr=subprocess.STDOUT, start_new_session=True,
                                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                record['workers'][str(gpu)] = {'pid': process.pid, 'command': command}
        queue_worker.atomic(ROOT / 'LAUNCHED.json', record)
        print(json.dumps(record['workers'], indent=2), flush=True)


if __name__ == '__main__':
    main()
