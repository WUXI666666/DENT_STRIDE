#!/usr/bin/env python3
"""Remove only this task's verified smoke and failed-attempt weights, never full runs."""
import json
from datetime import datetime
from prepare_comparison import ROOT, sha256, write_json


if __name__ == '__main__':
    retried = ROOT / 'runs/rtdetrv2_s/phone/seed_0'
    rows = [json.loads(x) for x in (retried / 'log.txt').read_text().splitlines()]
    assert len(rows) >= 3 and [r['epoch'] for r in rows[:3]] == [0, 1, 2]
    assert json.loads((retried / 'status.json').read_text())['state'] != 'failed'
    targets = [
        ROOT / 'smoke_runs/deim_s/phone/seed_0/best.pth',
        ROOT / 'smoke_runs/rtdetrv2_s/phone/seed_0/best.pth',
        ROOT / 'failed_attempts/rtdetrv2_s_phone_seed0_pillow_guard/best.pth',
    ]
    removed = []
    for path in targets:
        assert path.is_file() and not path.is_symlink()
        digest = sha256(path)
        if 'smoke_runs' in path.parts:
            assert (path.parent / 'COMPLETED').is_file()
            expected = json.loads((path.parent / 'metrics_summary.json').read_text())['checkpoint_sha256']
        else:
            expected = json.loads((ROOT / 'pillow_compatibility_retry.json').read_text())['failed_checkpoint_sha256']
        assert digest == expected
        record = {'file': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                  'reason': 'non-research pipeline test or superseded infrastructure-failed attempt',
                  'removed_at': datetime.now().astimezone().isoformat(),
                  'recovery': 'not retained; requires rerunning the corresponding non-research test'}
        write_json(path.parent / 'test_weight_pruned.json', record)
        path.unlink()
        removed.append(record)
        write_json(ROOT / 'pipeline_test_weights_pruned.json', removed)
    print(json.dumps({'files_removed': len(removed), 'GiB_freed': sum(r['bytes'] for r in removed)/1024**3,
                      'full_run_and_historical_checkpoints_untouched': True}))
