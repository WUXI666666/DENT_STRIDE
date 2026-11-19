"""Execute the native historical CLI; only suppress redundant checkpoint writes."""
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    implementation = Path(job['implementation']).resolve()
    sys.path.insert(0, str(implementation))
    import engine
    from engine.misc import dist_utils
    from controlled_runtime import install_controls
    assert Path(engine.__file__).resolve().parent == implementation / 'engine'
    assert os.environ['PYTHONHASHSEED'] == str(job['seed'])
    assert Path.cwd() == Path(job['native_cwd'])
    install_controls(job['controls'])
    out = Path(job['output_dir'])
    native_save = dist_utils.save_on_master

    def save(state, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.parent.resolve() == out.resolve():
            # Stage-1 snapshots affect native rollback and MUST remain untouched.
            # Native code never reloads last/checkpoint/stage-2 snapshots here.
            if destination.name in ['last.pth', 'best_stg2.pth'] or destination.name.startswith('checkpoint'):
                return
            if destination.name == 'best.pth':
                pending = out / 'best.pending.pth'
                native_save(state, pending, *args, **kwargs)
                os.replace(pending, destination)
                return
        return native_save(state, destination, *args, **kwargs)

    dist_utils.save_on_master = save
    record = {'job': job['id'], 'pid': os.getpid(), 'started_unix': time.time(),
              'seed': job['seed'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'PYTHONHASHSEED': os.environ['PYTHONHASHSEED'], 'controls': job['controls'],
              'native_cwd': str(Path.cwd()), 'native_entrypoint': str(implementation / 'train.py'),
              'initialization': 'Fresh pretrained backbone; no resume or tuning',
              'checkpoint_policy': 'Native selection and stage-1 rollback preserved; final best only'}
    (out / 'TRAINING_STARTED.json').write_text(json.dumps(record, indent=2))
    print('HISTORICAL_RECIPE_TRAINING_STARTED', job['id'], flush=True)
    sys.argv = [str(implementation / 'train.py'), '-c', job['config'], '--seed', str(job['seed']),
                '--use-amp', '-d', 'cuda', '--output-dir', str(out)]
    runpy.run_path(str(implementation / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
