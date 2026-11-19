"""Audit first, then discard ONLY two validated interrupted attempts' weights.

Completed results remain untouched. Logs are archived without any .pth files.
Nothing in the original frozen execution/configuration/data protocol is edited.
"""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

SIDE=Path(__file__).resolve().parent
ROOT=SIDE.parent
EXPECTED={'dslr_cmfa_dualopt_seed2':0,'internal_mobile_cmfa_no_gates_seed0':1}
STATE_FILES=['gpu0_state.json','gpu1_state.json']


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    os.replace(temporary,path)


def main():
    assert not (SIDE/'RESTART_READY.json').exists(),'Restart was already prepared'
    assert not (SIDE/'RECOVERY_FREEZE.json').exists(),'Restart plan already exists; inspect instead of overwriting'
    collector=module('restart_frozen_collector','collect_tables.py')
    replay=module('restart_frozen_replay','evaluate_native.py')
    sha=collector.sha
    report,manifest,verified,auxiliary=collector.readiness()
    assert report['completion_markers']==report['verified_completed_jobs']==11,report
    assert report['invalid_completed_jobs']==0,report
    assert not [i for i in report['issues'] if i['kind']!='not_completed'],report['issues']
    frozen=json.loads((ROOT/'LAUNCH_FREEZE.json').read_text())
    manifest_sha=sha(ROOT/'execution_manifest.json')
    assert manifest_sha==frozen['execution_manifest_sha256']
    for name,digest in manifest['scripts'].items():assert sha(ROOT/name)==digest,name
    for source in manifest['source_files']:assert sha(source['snapshot'])==source['sha256'],source['snapshot']
    for source in manifest['input_files']:
        if source['kind']!='manuscript':assert sha(source['path'])==source['sha256'],source['path']
    assert sha(manifest['pretrained']['path'])==manifest['pretrained']['sha256']
    assert sha(manifest['image_manifest'])==manifest['image_manifest_sha256']
    for image in json.loads(Path(manifest['image_manifest']).read_text()):
        st=Path(image['path']).stat()
        assert st.st_size==image['size'] and st.st_mtime_ns==image['mtime_ns'],image['path']
    assert shutil.disk_usage(ROOT).free>manifest['disk_min_free_bytes']
    locks=[]
    for gpu in [0,1]:
        lock=(ROOT/f'gpu{gpu}.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    states={name:json.loads((ROOT/name).read_text()) for name in STATE_FILES}
    for name,state in states.items():
        gpu=int(name[3])
        assert EXPECTED.get(state['current_job'])==gpu
        for key in ['worker_pid','child_pid']:
            pid=state.get(key)
            if pid:
                proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists():
                    command=proc.read_bytes().replace(b'\x00',b' ').decode(errors='replace')
                    assert str(ROOT) not in command,'Experiment PID is still alive: '+command
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in gpu_rows.strip().splitlines():
        idx,memory,utilization=map(int,line.split(','))
        if idx in [0,1]:assert memory<1000 and utilization<10,f'GPU {idx} busy'
    unfinished=[job for job in manifest['jobs'] if (job['configuration'],job['seed']) not in verified]
    assert len(unfinished)==19
    interrupted=[job for job in unfinished if (Path(job['output_dir'])/'TRAINING_STARTED.json').exists()]
    assert {job['id'] for job in interrupted}==set(EXPECTED)
    by_id={job['id']:job for job in manifest['jobs']}
    records=[]
    for job in interrupted:
        out=Path(job['output_dir'])
        assert out.resolve()==ROOT/'runs'/job['configuration']/f"seed_{job['seed']}"
        assert not out.is_symlink() and not (out/'COMPLETED.json').exists()
        assert json.loads(Path(job['job_file']).read_text())==job
        assert sha(job['config'])==sha(out/'config.yml')==job['config_sha256']
        log_rows=[json.loads(line) for line in (out/'log.txt').read_text().splitlines() if line.strip()]
        assert log_rows and len(log_rows)<160
        assert [row['epoch'] for row in log_rows]==list(range(len(log_rows)))
        weights=[]
        for path in sorted(out.rglob('*.pth')):
            assert path.is_file() and not path.is_symlink()
            assert path.resolve().is_relative_to(out.resolve())
            weights.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path),
                            'kind':'training_checkpoint' if path.parent==out else 'temporary_evaluation_payload'})
        assert (out/'best.pth').is_file()
        archive=SIDE/'interrupted_logs'/job['id']
        assert not archive.exists()
        records.append({'job':job['id'],'source_output':str(out),'archived_logs':str(archive),
                        'completed_epochs_before_power_loss':len(log_rows),'training_log_sha256':sha(out/'log.txt'),
                        'deleted_files':weights,'fresh_restart_seed':job['seed'],'load_interrupted_checkpoint':False})
    assignments={0:['dslr_cmfa_dualopt_seed2'],1:['internal_mobile_cmfa_no_gates_seed0']}
    costs={gpu:float(by_id[jids[0]]['estimated_gpu_hours']) for gpu,jids in assignments.items()}
    pending=sorted([job for job in unfinished if job['id'] not in EXPECTED],key=lambda job:(-job['estimated_gpu_hours'],job['id']))
    for job in pending:
        gpu=min(costs,key=costs.get)
        assignments[gpu].append(job['id']);costs[gpu]+=job['estimated_gpu_hours']
    scheduled=[jid for jids in assignments.values() for jid in jids]
    assert len(scheduled)==len(set(scheduled))==19
    assert set(scheduled)=={job['id'] for job in unfinished}
    plan={'created_unix':time.time(),'original_manifest_sha256':manifest_sha,
          'scripts':{name:sha(SIDE/name) for name in ['prepare_restart.py','launch_restart.py','queue_worker_restart.py','export_native.py']},
          'completed_jobs_preserved':report['verified_evidence'],'total_original_jobs':30,'completed_count':11,
          'fresh_training_jobs':19,'restarted_interrupted_jobs':records,'gpu_assignments':assignments,
          'estimated_gpu_hours':costs,'estimated_two_gpu_wall_hours':max(costs.values()),
          'changes':'Physical GPU scheduling only. Two power-interrupted attempts discarded; no resume; all remaining training starts from frozen pretrained initialization, original seed/config and 160 epochs. Completed checkpoints untouched.',
          'weight_deletion_authorized_by_user':True,'interrupted_weight_backups_created':False,
          'prediction_export':'Observe exact native EMA evaluation using the independently versioned sidecar; retain original strict metric checks.',
          'assistant_monitoring':'Startup check only; no recurring assistant monitoring'}
    # Record every exact target and its hash BEFORE any deletion. Never recurse-delete a directory.
    atomic(SIDE/'RECOVERY_FREEZE.json',plan)
    for name in STATE_FILES:shutil.copy2(ROOT/name,SIDE/('before_'+name))
    for record in records:
        out=Path(record['source_output'])
        for target in record['deleted_files']:
            path=Path(target['path'])
            assert path.resolve().is_relative_to(out.resolve())
            assert path.stat().st_size==target['size_bytes'] and sha(path)==target['sha256']
            path.unlink()
        assert not list(out.rglob('*.pth'))
        destination=Path(record['archived_logs']);destination.parent.mkdir(parents=True,exist_ok=True)
        out.rename(destination)
        assert not out.exists()
    for evidence in verified.values():collector.verify_job(evidence['job'])
    atomic(SIDE/'RESTART_READY.json',{'ready':True,'plan_sha256':sha(SIDE/'RECOVERY_FREEZE.json'),
                                    'completed_checkpoints_preserved':11,'restarted_jobs':list(EXPECTED),
                                    'discarded_model_checkpoints':sum(row['kind']=='training_checkpoint' for r in records for row in r['deleted_files']),
                                    'deleted_file_count':sum(len(r['deleted_files']) for r in records),
                                    'remaining_training_jobs':19})
    print(json.dumps({'ready':True,'completed_preserved':11,'restart_jobs':list(EXPECTED),
                      'fresh_jobs_total':19,'gpu_jobs':{g:len(ids) for g,ids in assignments.items()},
                      'estimated_two_gpu_wall_hours':max(costs.values()),
                      'deleted_model_weights':sum(row['kind']=='training_checkpoint' for r in records for row in r['deleted_files'])},indent=2),flush=True)


if __name__=='__main__':main()
