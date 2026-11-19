#!/usr/bin/env python3
"""Archive the identified infrastructure failure and restart its exact seed cleanly."""
import json
import os
import subprocess
import sys
from datetime import datetime
from prepare_comparison import ROOT, sha256, write_json


if __name__ == '__main__':
    model, cohort, seed, gpu = 'rtdetrv2_s', 'phone', 0, 1
    run = ROOT / 'runs' / model / cohort / f'seed_{seed}'
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'failed' and 'DecompressionBombError' in status['error']
    try:
        os.kill(status['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Failed training process has not exited')
    queue_state = json.loads((ROOT / 'queue_gpu1.json').read_text())
    assert queue_state['state'] == 'failed'
    archive = ROOT / 'failed_attempts' / 'rtdetrv2_s_phone_seed0_pillow_guard'
    assert not archive.exists()
    archive.parent.mkdir(parents=True, exist_ok=True)
    run.rename(archive)
    record = {
        'reason': 'Pillow crop pixel guard; not a model/metric-based retry',
        'archive': str(archive), 'failed_checkpoint_sha256': sha256(archive / 'best.pth'),
        'restart': 'fresh seed 0, same frozen configuration, all 160 epochs',
        'time': datetime.now().astimezone().isoformat(),
    }
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_queue.py'), '--model', model, '--gpu', str(gpu)]
    with (ROOT / 'queue_gpu1.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
    record['new_queue_pid'] = process.pid
    write_json(ROOT / 'pillow_compatibility_retry.json', record)
    print(json.dumps(record, indent=2))
