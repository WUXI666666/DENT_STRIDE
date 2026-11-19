from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
from common import ROOT, MODELS, NAMES, run_dir, write, now, coco_metrics, fixed_metrics, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('family');a=p.parse_args();out=run_dir(a.family)
    summary=json.loads((out/'metrics_summary.json').read_text())
    candidates=[out/'best_predictions.coco.json.gz',out/'evaluation/predictions.coco.json.gz',
                out/'coco_eval/coco_instances_results.json']
    source=next(x for x in candidates if x.is_file())
    with (gzip.open(source,'rt') if source.suffix=='.gz' else source.open()) as f:pred=json.load(f)
    ev=out/'evaluation';values=coco_metrics(pred,ev)
    delta=(values['AP']-summary['metrics_fraction']['AP'])*100
    if abs(delta)>.02:raise RuntimeError(f'Selected-output COCO replay differs by {delta} AP points.')
    summary['export_replay_metrics_fraction']=values
    summary['export_replay_minus_native_AP_points']=delta
    summary['fixed_threshold']=fixed_metrics(pred,ev)
    compute=out/'compute_profile.json';benchmark=out/'inference_benchmark.json'
    if compute.is_file():summary['compute']=json.loads(compute.read_text())
    if benchmark.is_file():summary['inference_benchmark']=json.loads(benchmark.read_text())
    summary['independent_test']=False
    summary['checkpoint_sha256']=sha(summary['checkpoint'])
    summary['finalized_at']=now()
    write(out/'metrics_summary.json',summary)
    write(out/'status.json',{'state':'completed','metrics_summary':str(out/'metrics_summary.json'),'at':now()})
    (out/'COMPLETED').write_text(now())
    rows=[]
    for family in MODELS:
        directory=run_dir(family);p=directory/'metrics_summary.json'
        row={'Method':NAMES[family],'seed':0,'input':'1024x1024','state':'pending'}
        if p.is_file():
            s=json.loads(p.read_text());ft=s.get('fixed_threshold',{});comp=s.get('compute',{})
            row.update(state='completed' if (directory/'COMPLETED').exists() else 'training_completed',
                       AP=s['metrics_percent']['AP'],AP50=s['metrics_percent']['AP50'],AP75=s['metrics_percent']['AP75'],
                       AR100=s['metrics_percent']['AR100'],Params_M=comp.get('Params_M'),GFLOPs=comp.get('GFLOPs'),
                       GMACs=comp.get('GMACs'),best_epoch=s.get('best_epoch_one_based'))
            for key in ['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']:
                row[key]=None if ft.get(key) is None else ft[key]*100
        rows.append(row)
    write(ROOT/'results_summary.json',rows)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'results_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    print('FINALIZED',a.family,summary['metrics_percent']['AP'])


if __name__=='__main__':main()
