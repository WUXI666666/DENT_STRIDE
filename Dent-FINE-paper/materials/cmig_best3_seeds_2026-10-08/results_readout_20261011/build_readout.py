"""Read-only independent calculations and a new Excel result readout.

Never edits training outputs, the frozen collector, documents or source data.
"""
from pathlib import Path
from collections import Counter
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile

import numpy as np
import xlsxwriter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
DATA=ROOT/'manuscript_data'
METRICS=['AP','AP50','AP75','AR100']
FIXED=['Precision','Recall','F1','matched_tooth_FDI_accuracy','localization_coverage','raw_duplicate_image_rate']
HISTORICAL={'mobile_1024_full':55.701,'pano_historical_spatialshape':54.385,'dslr_cmfa_dualopt':73.115}
COHORT={'Mobile':'Smartphone','Pano':'DENTEX2023','DSLR':'DSLR'}
NAMES={'mobile_1024_full':'Proto Aux','pano_historical_spatialshape':'Factor + Prototype + SpatialShape','dslr_cmfa_dualopt':'CMFA + DualOpt'}
T95_DF2=4.302652729911275


def read(path):
    return json.loads(Path(path).read_text())


def check_close(actual, expected, where, tolerance=1e-9):
    assert math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=tolerance),(where,actual,expected)


def main():
    destination=OUT/'DENT_STRIDE_all_results_20261011.xlsx'
    assert not destination.exists(),'Existing readout is never overwritten'
    manifest=read(ROOT/'execution_manifest.json')
    jobs=manifest['jobs'];by_id={j['id']:j for j in jobs}
    all_rows=read(DATA/'all_seed_results.json')
    table3=read(DATA/'Table3_primary_three_seed.json')
    table6=read(DATA/'Table6_SpatialShape_three_seed.json')
    table7=read(DATA/'Table7_internal_seed0.json')
    factors=read(DATA/'factor_fusion_2x2_seed0.json')
    priors=read(DATA/'spatial_shape_per_seed.json')
    assert len(jobs)==len(all_rows)==30 and len({r['job_id'] for r in all_rows})==30
    assert {r['job_id'] for r in all_rows}==set(by_id)
    by_result={r['job_id']:r for r in all_rows}
    raw={}
    for job in jobs:
        run=Path(job['output_dir'])
        assert (run/'COMPLETED.json').is_file()
        rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        assert len(rows)==160 and [r['epoch'] for r in rows]==list(range(160))
        best=max(rows,key=lambda row:row['test_coco_eval_bbox'][0])
        native=read(run/'native_coco_metrics.json')
        summary=read(run/'metrics_summary.json')
        assert native['best_epoch_zero_based']==best['epoch']
        result=by_result[job['id']]
        for name,index in [('AP',0),('AP50',1),('AP75',2),('AR100',8)]:
            check_close(native['metrics_fraction'][name],best['test_coco_eval_bbox'][index],job['id']+name,2e-6)
            check_close(result[name],100*native['metrics_fraction'][name],job['id']+name)
        counts=Counter()
        per_image=read(run/'evaluation/per_image_counts.json')
        assert len(per_image)==len({r['image_id'] for r in per_image})
        for row in per_image:
            counts.update({k:v for k,v in row.items() if k!='image_id'})
        values={'Precision':counts['TP']/(counts['TP']+counts['FP']),
                'Recall':counts['TP']/(counts['TP']+counts['FN']),
                'F1':2*counts['TP']/(2*counts['TP']+counts['FP']+counts['FN']),
                'matched_tooth_FDI_accuracy':counts['correct_FDI_matches']/counts['localized_matches'],
                'localization_coverage':counts['localized_matches']/counts['GT'],
                'raw_duplicate_image_rate':counts['duplicate_image']/len(per_image)}
        for name,value in values.items():
            check_close(summary['fixed_threshold'][name],value,job['id']+name)
            check_close(result[name],value*100,job['id']+name)
        raw[job['id']]={'native':native,'fixed':values,'best':best}
    for summary in table3:
        selected=sorted([j for j in jobs if j['configuration']==summary['configuration']],key=lambda j:j['seed'])
        assert [j['seed'] for j in selected]==[0,1,2]
        for metric in METRICS+FIXED:
            values=np.array([raw[j['id']]['native']['metrics_fraction'][metric]*100 if metric in METRICS
                             else raw[j['id']]['fixed'][metric]*100 for j in selected])
            check_close(summary[metric+'_mean'],float(np.mean(values)),summary['configuration']+metric)
            check_close(summary[metric+'_SD'],float(np.std(values,ddof=1)),summary['configuration']+metric)
    per_prior={}
    for row in priors:
        base=next(j for j in jobs if j['configuration']=='pano_historical_spatialshape' and j['seed']==row['seed'])
        native=read(Path(base['output_dir'])/f"native_prior_{row['prior']}_metrics.json")
        assert native['checkpoint_sha256']==raw[base['id']]['native']['checkpoint_sha256']
        for metric in METRICS:check_close(row[metric],100*native['metrics_fraction'][metric],str((row['seed'],row['prior'],metric)))
        per_prior[(row['seed'],row['prior'])]=row
    assert len(per_prior)==12
    for row in table6:
        for metric in METRICS:
            vals=np.array([per_prior[(seed,row['prior'])][metric] for seed in range(3)])
            check_close(row[metric+'_mean'],np.mean(vals),row['prior']+metric)
            check_close(row[metric+'_SD'],np.std(vals,ddof=1),row['prior']+metric)
        deltas=np.array([per_prior[(seed,row['prior'])]['AP']-per_prior[(seed,'none')]['AP'] for seed in range(3)])
        mean=float(np.mean(deltas));sd=float(np.std(deltas,ddof=1));half=T95_DF2*sd/math.sqrt(3)
        for suffix,value in [('mean',mean),('SD',sd),('CI95_low',mean-half),('CI95_high',mean+half)]:
            check_close(row['AP_delta_vs_none_'+suffix],value,row['prior']+suffix)
    for row in table7+factors:
        native=read(row['native_metric_path'])
        for metric in METRICS:check_close(row[metric],native['metrics_fraction'][metric]*100,row['setting']+metric)
        check_close(row['delta_AP_vs_reference'],row['AP']-row['reference_AP'],row['setting']+'delta')
    assert len(factors)==4
    verification={'reviewed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  'scope':'Only cmig_best3_seeds_2026-10-08 frozen runs; interrupted logs excluded',
                  'completed_training_jobs':30,'primary_training_jobs':9,'internal_training_jobs':21,
                  'full_160_epoch_logs_checked':30,'native_best_log_metrics_checked':30,
                  'fixed_threshold_count_recalculations_checked':30,'primary_mean_sample_SD_checked':3,
                  'same_weight_prior_rows_checked':12,'factor_fusion_rows_checked':4,
                  'sample_SD_ddof':1,'CI_method':'Student-t df=2 for same-weight prior seed deltas only; not patient/test uncertainty',
                  'errors':[],'original_outputs_edited':False}

    book=xlsxwriter.Workbook(destination,{'strings_to_urls':False,'strings_to_formulas':False})
    book.set_properties({'title':'DENT-STRIDE experiment results, 2026-10-11','comments':'Source-backed result readout; no model or manuscript changes.'})
    head=book.add_format({'bold':True,'bg_color':'#DCE6EF','font_color':'#19334A','border':1,'text_wrap':True})
    number=book.add_format({'num_format':'0.000'})
    wrap=book.add_format({'text_wrap':True,'valign':'top'})
    def sheet(name,rows):
        ws=book.add_worksheet(name)
        if not rows:return
        fields=list(dict.fromkeys(key for row in rows for key in row))
        for c,key in enumerate(fields):
            ws.write(0,c,key,head)
            width=48 if any(token in key for token in ['path','configuration','setting','说明','配置','位置']) else 22
            ws.set_column(c,c,width)
        for r,row in enumerate(rows,1):
            for c,key in enumerate(fields):
                value=row.get(key)
                if value is None:ws.write_blank(r,c,None)
                elif isinstance(value,(dict,list)):ws.write(r,c,json.dumps(value,ensure_ascii=False),wrap)
                elif isinstance(value,float):ws.write_number(r,c,value,number)
                else:ws.write(r,c,value)
        ws.freeze_panes(1,2);ws.autofilter(0,0,len(rows),len(fields)-1);ws.set_row(0,32)
    notes=[{'条目':'完成状态','说明':'30/30 训练完成并核验：9 项主实验 + 21 项 seed-0 内部训练/参考。另有 12 路先验重放和 2 路额外因子重放。'},
           {'条目':'数据范围','说明':'原划分内部验证，不是独立测试；未重新训练三种子基线，不能与历史基线拼接新配对统计结论。'},
           {'条目':'训练与轮次','说明':'每项完整训练 160 epochs；最佳轮次表列从 1 开始，同时保留原 0-based 索引。'},
           {'条目':'AP 单位','说明':'AP/AP50/AP75/AR100 为百分数或 AP 点；固定阈值指标为百分数。'},
           {'条目':'不确定性','说明':'主实验 seeds 0/1/2，均值 ± 样本 SD（ddof=1）；内部训练只有 seed 0，不填写虚构 SD。'},
           {'条目':'固定阈值','说明':'score≥0.30、IoU≥0.50、一对一 Hungarian 定位匹配；TP 为定位匹配且 FDI 正确的预测；原生 top-300，无额外 NMS。'},
           {'条目':'分母','说明':'Precision:预测数；Recall:GT数；FDI accuracy:定位匹配牙齿数；定位覆盖率:GT数；重复图像率:全部图像数。'},
           {'条目':'先验对照','说明':'每个 Pano seed 同一 checkpoint，来自历史配置 prior-ON 选模。四路重放只切换 center/shape，禁用 view/appearance；不是四组重新训练。'},
           {'条目':'先验区间','说明':'ΔAP 的 Student-t 95% CI 来自三个种子的同权重差值（df=2）；探索性结果，不是患者层级独立测试区间。'},
           {'条目':'Factor 参考','说明':'移除象限/位置/乳恒牙辅助监督的比较参考为三种监督保留、训练和推理融合均关闭的 seed-0 模型（AP55.537）。'},
           {'条目':'DualOpt 参考','说明':'四组均保留 CMFA；比较 AdamW、仅回拉、仅额外衰减、两种操作都有。不是无 CMFA 的基础模型。'},
           {'条目':'参数量','说明':'同时保留全部参数和可训练参数；两者固定标量差为5，不是架构变化。'},
           {'条目':'断电处理','说明':'两项中断权重已删除并从头重训；不使用 archived interrupted_logs 的数值。'},
           {'条目':'文件变更','说明':'本次仅生成独立结果汇总，不修改论文、训练结果、数据、模型源码或原表格文件。'}]
    sheet('说明',notes)
    pretty=[]
    for row in sorted(table3,key=lambda r:['Pano','Mobile','DSLR'].index(r['cohort'])):
        result={'数据域':COHORT[row['cohort']],'模型配置':NAMES[row['configuration']]}
        for metric in METRICS:result[metric+' mean±SD']=f"{row[metric+'_mean']:.3f}±{row[metric+'_SD']:.3f}"
        result['可训练参数(M)']=row['params_trainable']/1e6
        pretty.append(result)
    sheet('主实验_汇总',pretty)
    sheet('主实验_原始统计',table3)
    sheet('主实验_逐种子',read(DATA/'primary_seed_results.json'))
    sheet('内部消融_seed0',table7)
    sheet('先验_三种子汇总',table6)
    sheet('先验_逐种子',priors)
    sheet('因子融合_2x2',factors)
    sheet('固定阈值_三种子汇总',[{'cohort':r['cohort'],'configuration':r['configuration'],
                                     **{name:r[name] for metric in FIXED for name in [metric+'_mean',metric+'_SD']}} for r in table3])
    sheet('30项训练_全部指标',all_rows)
    indexes=[]
    for job in jobs:
        run=Path(job['output_dir']);result=by_result[job['id']]
        indexes.append({'job':job['id'],'cohort':job['cohort'],'phase':job['phase'],'seed':job['seed'],
                        'config_path':job['config'],'checkpoint_path':str(run/'best.pth'),
                        'metrics_path':str(run/'native_coco_metrics.json'),'log_path':str(run/'log.txt'),
                        'best_epoch_one_based':result['best_epoch_one_based'],
                        'checkpoint_sha256':result['checkpoint_sha256'],'config_sha256':result['config_sha256'],
                        'params_trainable':result['params_trainable'],'params_total':result['params_total']})
    sheet('配置与权重索引',indexes)
    history=[]
    for row in table3:
        old=HISTORICAL[row['configuration']]
        seed0=next(r for r in all_rows if r['configuration']==row['configuration'] and r['seed']==0)
        history.append({'cohort':row['cohort'],'historical_selected_single_seed_AP':old,
                        'current_seed0_AP':seed0['AP'],'current_3seed_AP_mean':row['AP_mean'],
                        'current_3seed_AP_SD':row['AP_SD'],'mean_minus_historical_AP_points':row['AP_mean']-old,
                        'interpretation':'Descriptive comparison only; historical selected maximum is not an unbiased paired baseline.'})
    sheet('与历史最高AP对照',history)
    sheet('统计复核',[{'check':key,'value':value} for key,value in verification.items()])
    book.close()
    # Inspect the completed XLSX container, sheets and row coverage without changing it.
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len([name for name in archive.namelist() if name.startswith('xl/worksheets/sheet') and name.endswith('.xml')])==13
    (OUT/'independent_calculation_checks.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    print(json.dumps({'excel':str(destination),'bytes':destination.stat().st_size,'sheets':13,
                      'verified_training_jobs':30,'calculation_errors':0,
                      'primary':pretty,'history':history},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
