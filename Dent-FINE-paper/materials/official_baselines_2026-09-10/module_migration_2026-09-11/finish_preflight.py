#!/usr/bin/env python3
"""Freeze passed preflight evidence and prune only this task's test weights."""
import json
from datetime import datetime
from pathlib import Path

from prepare_migration import HERE, ROOT, TASKS, protected_files, sha256, write_json


def main():
    if (HERE / 'preflight_validation.json').exists():
        raise RuntimeError('Preflight already frozen')
    tests = json.loads((HERE / 'unit_validation.json').read_text())
    assert tests['passed']
    checks, removed = {}, []
    for task in TASKS:
        output = HERE / 'smoke_runs' / task
        summary = json.loads((output / 'metrics_summary.json').read_text())
        assert summary['smoke_only'] and summary['epochs_completed'] == 2
        assert (output / 'COMPLETED').exists()
        assert summary['stage1_RAM_restores'] >= 1
        verified = json.loads((output / 'verified_coco_metrics.json').read_text())
        assert verified['matches_selected_checkpoint']
        assert sha256(output / 'best.pth') == verified['checkpoint_sha256']
        assert list(output.rglob('*.pth')) == [output / 'best.pth']
        profile = json.loads((HERE / 'profiles' / f'{task}.json').read_text())
        assert profile['params_training_total'] == summary['params_training_total']
        checks[task] = {'two_epoch_native_train_passed': True, 'AMP_EMA_validation_passed': True,
                        'stage1_RAM_restores': summary['stage1_RAM_restores'],
                        'best_checkpoint_reevaluation_passed': True,
                        'best_only_policy_passed': True, 'profile': profile}
    before = json.loads((HERE / 'protected_files_before.json').read_text())
    after = protected_files()
    assert before == after, 'Protected original source/manuscript changed'
    code_paths = list(HERE.glob('*.py')) + list((HERE / 'configs').glob('*.yml'))
    code_paths += [HERE / 'protocol.json', HERE / 'dentex_train_templates.json',
                   ROOT / 'scripts/run_official.py', ROOT / 'scripts/prepare_comparison.py']
    sources = {str(path): sha256(path) for path in sorted(code_paths)}
    # These three exact files were created solely by the migration smoke tests.
    # Preserve their full logs, metrics and SHA records; no research weights touched.
    for task in TASKS:
        path = HERE / 'smoke_runs' / task / 'best.pth'
        assert path.resolve().parent == (HERE / 'smoke_runs' / task).resolve()
        removed.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        path.unlink()
    write_json(HERE / 'preflight_validation.json', {
        'passed': True, 'verified_at': datetime.now().astimezone().isoformat(),
        'original_source_and_manuscripts_unchanged': True, 'protected_file_count': len(before),
        'unit_checks': tests['checks'], 'native_smoke_checks': checks,
        'source_sha256': sources, 'smoke_checkpoint_pruning': removed,
    })
    print(json.dumps({'passed': True, 'tasks': list(checks), 'protected_files_unchanged': len(before),
                      'only_smoke_weights_pruned_MiB': sum(x['bytes'] for x in removed)/1024**2}, indent=2))


if __name__ == '__main__':
    main()
