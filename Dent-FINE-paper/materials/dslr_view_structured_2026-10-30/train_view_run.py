"""Install additions, then execute the unchanged native historical training CLI."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    job=json.loads(Path(sys.argv[1]).read_text());sys.path.insert(0,job['implementation'])
    from view_identity_runtime import install_view_identity_runtime
    runtime=install_view_identity_runtime(job['runtime_settings'])
    from engine.misc import dist_utils
    assert os.environ['PYTHONHASHSEED']==str(job['seed'])
    assert Path.cwd()==Path(job['native_cwd'])
    output=Path(job['output_dir']);native_save=dist_utils.save_on_master
    def save(state,destination,*args,**kwargs):
        path=Path(destination)
        if path.parent.resolve()==output.resolve():
            if path.name in ['last.pth','best_stg2.pth'] or path.name.startswith('checkpoint'):return
            if path.name=='best.pth':
                pending=output/'best.pending.pth';native_save(state,pending,*args,**kwargs)
                os.replace(pending,path);return
        return native_save(state,destination,*args,**kwargs)
    dist_utils.save_on_master=save
    record={'job_id':job['id'],'pid':os.getpid(),'started_unix':time.time(),
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'seed':job['seed'],
        'initialization':job['initialization'],'runtime':runtime,
        'storage':'best.pth only at completion; native best_stg1.pth retained temporarily for required stage rollback',
        'original_source_edited':False}
    (output/'TRAINING_STARTED.json').write_text(json.dumps(record,indent=2))
    print('DSLR_VIEW_TRAINING_STARTED',job['id'],flush=True)
    sys.argv=[job['train_script'],'-c',job['config'],'--seed',str(job['seed']),'--use-amp','-d','cuda',
        '--output-dir',job['output_dir']]
    runpy.run_path(job['train_script'],run_name='__main__')


if __name__=='__main__':main()
