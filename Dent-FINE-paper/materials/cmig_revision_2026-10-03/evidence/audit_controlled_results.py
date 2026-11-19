"""Read-only audit of frozen CMIG runs; writes only a new evidence directory."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')
OUT = Path(__file__).resolve().parent
T95_DF2 = 4.302652729911275
COCO_NAMES = ['AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
              'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge']
MAIN_METRICS = ['AP', 'AP50', 'AP75', 'AR100']
FIXED = ['Precision', 'Recall', 'F1', 'matched_tooth_FDI_accuracy',
         'localization_coverage', 'raw_duplicate_image_rate']
ALL_METRICS = MAIN_METRICS + FIXED


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(2 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def csv_write(name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def numeric(result):
    values = {k: result['metrics_percent'][k] for k in MAIN_METRICS}
    values.update({k: 100 * result['fixed_threshold'][k] for k in FIXED})
    return values


def stats(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = T95_DF2 * sd / math.sqrt(n) if n == 3 else None
    return {'n': n, 'values': values, 'mean': mean, 'SD': sd,
            'CI95_low': mean - half if half is not None else None,
            'CI95_high': mean + half if half is not None else None}


def check_saved_arrays(directory, result):
    with np.load(directory / 'coco_precision_recall.npz') as saved:
        p = saved['precision']
        r = saved['recall']
        ious = saved['IoU_thresholds']
        def valid_mean(x):
            return float(x[x > -1].mean()) if np.any(x > -1) else -1.
        actual = {'AP': valid_mean(p[:, :, :, 0, -1]),
                  'AP50': valid_mean(p[np.isclose(ious, .5), :, :, 0, -1]),
                  'AP75': valid_mean(p[np.isclose(ious, .75), :, :, 0, -1]),
                  'AR100': valid_mean(r[:, :, 0, -1])}
    for name, value in actual.items():
        assert abs(value - result['metrics_fraction'][name]) < 1e-12, (directory, name)


def check_fixed(directory, result):
    rows = read(directory / 'per_image_counts.json')
    assert len({r['image_id'] for r in rows}) == len(rows)
    total = Counter()
    for row in rows:
        for k, v in row.items():
            if k != 'image_id':
                total[k] += v
        assert row['TP'] == row['correct_FDI_matches']
        assert row['FP'] == row['predictions'] - row['TP']
        assert row['FN'] == row['GT'] - row['TP']
        assert row['TP'] <= row['localized_matches'] <= row['GT']
    original = result['fixed_threshold']
    assert dict(total) == original['counts'], directory
    recalculated = {'Precision': total['TP'] / total['predictions'],
                    'Recall': total['TP'] / total['GT'],
                    'F1': 2 * total['TP'] / (total['predictions'] + total['GT']),
                    'matched_tooth_FDI_accuracy': total['TP'] / total['localized_matches'],
                    'localization_coverage': total['localized_matches'] / total['GT'],
                    'raw_duplicate_image_rate': total['duplicate_image'] / len(rows)}
    for k, value in recalculated.items():
        assert abs(value - original[k]) < 1e-12, (directory, k)
    assert original['score_threshold'] == .30 and original['IoU_threshold'] == .50
    return {'images': len(rows), **dict(total)}


def active_settings(cfg, job, log_rows):
    enc, dec, criterion = cfg['HybridEncoder'], cfg['DFINETransformer'], cfg['DEIMCriterion']
    losses = criterion['losses']
    components = {}
    for name in ['mal', 'bbox', 'giou', 'fgl', 'ddf', 'nwd', 'arch', 'quad', 'tooth', 'dent', 'proto']:
        keys = sorted({k for r in log_rows for k in r if k.startswith('train_loss_' + name)
                       and (k == 'train_loss_' + name or k.startswith('train_loss_' + name + '_'))})
        components[name] = {'logged_keys': keys,
                            'ever_nonzero': any(abs(r.get(k, 0.)) > 1e-12 for r in log_rows for k in keys)}
    return {'CMFA': enc['use_cmfa_fusion'], 'HOOA': enc['use_hooa_attention'],
            'Factor_heads': dec['use_factor_heads'], 'Prototype_head': dec['use_tooth_prototype'],
            'prototype_count': dec['prototype_count'], 'prototype_tau': dec['prototype_tau'],
            'prototype_fusion_weight': dec['prototype_fusion_weight'],
            'factor_fusion_train': dec['factor_fusion_train'],
            'factor_inference_fusion': job['controls'].get('factor_inference_fusion', True),
            'factor_alpha': [dec['factor_alpha_quad'], dec['factor_alpha_tooth'], dec['factor_alpha_dent']],
            'optimizer': cfg['optimizer']['type'],
            'rollback_weight': cfg['optimizer'].get('rollback_weight', 0),
            'realtime_decay': cfg['optimizer'].get('realtime_decay', 0),
            'AdamW_weight_decay': cfg['optimizer']['weight_decay'],
            'eval_size': cfg['eval_spatial_size'],
            'multiscale_training': cfg['train_dataloader']['collate_fn'].get('base_size_repeat'),
            'MixUp_probability': cfg['train_dataloader']['collate_fn']['mixup_prob'],
            'criterion_losses': losses, 'explicit_loss_weights': criterion['weight_dict'],
            'logged_loss_activity': components,
            'disabled_feature_modules': [k for k, v in enc.items() if k.startswith('use_') and v is False],
            'inert_weight_entries': ['loss_nwd'] if 'nwd' not in losses else [],
            'prototype_weight_attribute_unused_in_forward': dec.get('prototype_weight'),
            'runtime_controls': job['controls']}


def main():
    snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    manifest = read(ROOT / 'execution_manifest.json')
    jobs = manifest['jobs']
    completed_ids = {j['id'] for j in jobs if (Path(j['output_dir']) / 'COMPLETED.json').is_file()}
    runs, pending, replays, provenance, settings, learning = [], [], [], [], {}, {}
    by_configuration = defaultdict(dict)
    for job in jobs:
        directory = Path(job['output_dir'])
        if job['id'] not in completed_ids:
            logs = [json.loads(s) for s in (directory / 'log.txt').read_text().splitlines() if s.strip()] if (directory / 'log.txt').is_file() else []
            pending.append({'job_id': job['id'], 'configuration': job['configuration'],
                            'cohort': job['cohort'], 'phase': job['phase'], 'seed': job['seed'],
                            'completed_epochs': len(logs), 'status': 'pending',
                            'maximum_partial_log_AP_not_reportable': 100 * max((r['test_coco_eval_bbox'][0] for r in logs), default=0.)})
            continue
        marker = read(directory / 'COMPLETED.json')
        result = read(directory / 'metrics_summary.json')
        cfg = yaml.safe_load(Path(job['config']).read_text())
        assert marker['job'] == job['id'] == result['job_id']
        hashes = {'checkpoint': sha(directory / 'best.pth'), 'metrics': sha(directory / 'metrics_summary.json'),
                  'config': sha(job['config']), 'log': sha(directory / 'log.txt')}
        assert hashes['checkpoint'] == marker['checkpoint_sha256'] == result['checkpoint_sha256']
        assert hashes['metrics'] == marker['metrics_sha256']
        assert hashes['config'] == job['config_sha256'] == result['config_sha256'] == sha(directory / 'config.yml')
        log_rows = [json.loads(s) for s in (directory / 'log.txt').read_text().splitlines() if s.strip()]
        assert len(log_rows) == 160 and [r['epoch'] for r in log_rows] == list(range(160))
        assert cfg['epoches'] == 160 and cfg['train_dataloader']['total_batch_size'] == 8 and cfg['val_dataloader']['total_batch_size'] == 16
        best = max(log_rows, key=lambda r: r['test_coco_eval_bbox'][0])
        selection = read(directory / 'best_selection.json')
        assert best['epoch'] == selection['epoch_zero_based'] == result['best_epoch_zero_based']
        assert best['epoch'] + 1 == result['best_epoch_one_based']
        for index, name in enumerate(COCO_NAMES):
            assert best['test_coco_eval_bbox'][index] == result['logged_best_metrics_fraction'][name]
        assert abs(result['replay_minus_logged_AP_points']) < .02
        assert abs(100 * (result['metrics_fraction']['AP'] - best['test_coco_eval_bbox'][0]) - result['replay_minus_logged_AP_points']) < 1e-10
        evaluation = read(directory / 'evaluation/metrics.json')
        assert evaluation['metrics_percent'] == result['metrics_percent'] and evaluation['fixed_threshold'] == result['fixed_threshold']
        check_saved_arrays(directory / 'evaluation', result)
        totals = check_fixed(directory / 'evaluation', result)
        assert cfg['PostProcessor']['num_top_queries'] == 300
        assert result['prior'] == 'off' and not result['independent_test']
        row = {'job_id': job['id'], 'configuration': job['configuration'], 'cohort': job['cohort'],
               'phase': job['phase'], 'seed': job['seed'], 'status': 'complete', **numeric(result),
               'params_M': result['params_total'] / 1e6, 'params_total': result['params_total'],
               'best_epoch': result['best_epoch_one_based'], 'replay_minus_logged_AP_points': result['replay_minus_logged_AP_points'],
               'metric_path': str(directory / 'metrics_summary.json'), 'checkpoint_path': str(directory / 'best.pth'), **totals}
        runs.append(row)
        by_configuration[job['configuration']][job['seed']] = row
        settings[job['id']] = active_settings(cfg, job, log_rows)
        learning[job['id']] = [{'epoch': r['epoch'] + 1, 'AP': 100 * r['test_coco_eval_bbox'][0],
                                'train_loss': r['train_loss']} for r in log_rows]
        provenance.append({'job_id': job['id'], 'hashes': hashes, 'completion': marker,
                           'arrays_independently_recomputed': True, 'fixed_counts_independently_recomputed': True,
                           'native_training_epoch_budget': 160, 'best_epoch': row['best_epoch'], 'checks_passed': True})
        for child in sorted(directory.glob('replay_*/metrics.json')):
            replay = read(child)
            assert replay['checkpoint_sha256'] == hashes['checkpoint']
            assert replay['config_sha256'] == hashes['config']
            check_saved_arrays(child.parent, replay)
            replay_counts = check_fixed(child.parent, replay)
            rrow = {'job_id': job['id'], 'configuration': job['configuration'], 'cohort': job['cohort'],
                    'seed': job['seed'], 'replay': child.parent.name,
                    'factor_fusion_train': cfg['DFINETransformer']['factor_fusion_train'],
                    'factor_inference_fusion': replay['controls'].get('factor_inference_fusion', True),
                    'prior': replay['prior'], **numeric(replay), **replay_counts,
                    'metric_path': str(child)}
            replays.append(rrow)

    summaries = []
    primary_names = sorted({j['configuration'] for j in jobs if j['phase'] == 'primary'})
    for name in primary_names:
        group = by_configuration[name]
        assert sorted(group) == [0, 1, 2], f'Primary group incomplete: {name}'
        row = {'configuration': name, 'cohort': group[0]['cohort'], 'n': 3,
               'seeds': [0, 1, 2], 'params_M': group[0]['params_M'],
               'metrics': {k: stats([group[s][k] for s in [0, 1, 2]]) for k in ALL_METRICS}}
        summaries.append(row)
    previous = {r['configuration']: r for r in read(ROOT / 'three_seed_mean_sd.json')}
    for row in summaries:
        for k in MAIN_METRICS:
            assert abs(row['metrics'][k]['mean'] - previous[row['configuration']]['metrics'][k]['mean']) < 1e-12
            assert abs(row['metrics'][k]['SD'] - previous[row['configuration']]['metrics'][k]['SD']) < 1e-12

    comparisons = [
        ('Mobile: full minus base, 1024', 'mobile_1024_full', 'mobile_1024_base'),
        ('Mobile: features minus base, 1024', 'mobile_1024_features', 'mobile_1024_base'),
        ('Mobile: identity minus base, 1024', 'mobile_1024_identity', 'mobile_1024_base'),
        ('Mobile: full minus features, 1024', 'mobile_1024_full', 'mobile_1024_features'),
        ('Mobile: full minus identity, 1024', 'mobile_1024_full', 'mobile_1024_identity'),
        ('Mobile: full minus base, 640', 'mobile_640_full', 'mobile_640_base'),
        ('Mobile: base 1024 minus base 640', 'mobile_1024_base', 'mobile_640_base'),
        ('Mobile: full 1024 minus full 640', 'mobile_1024_full', 'mobile_640_full'),
        ('Pano: factor minus base', 'pano_factor', 'pano_base'),
        ('Pano: prototype minus base', 'pano_prototype', 'pano_base'),
        ('Pano: factor+prototype minus base', 'pano_factor_prototype', 'pano_base'),
        ('Pano: factor+prototype minus factor', 'pano_factor_prototype', 'pano_factor'),
        ('DSLR: CMFA+AdamW minus base+AdamW', 'dslr_cmfa_adamw', 'dslr_base_adamw'),
        ('DSLR: base+DualOpt minus base+AdamW', 'dslr_base_dualopt', 'dslr_base_adamw'),
        ('DSLR: CMFA+DualOpt minus base+AdamW', 'dslr_cmfa_dualopt', 'dslr_base_adamw'),
        ('DSLR: CMFA+DualOpt minus CMFA+AdamW', 'dslr_cmfa_dualopt', 'dslr_cmfa_adamw'),
        ('DSLR: CMFA+DualOpt minus base+DualOpt', 'dslr_cmfa_dualopt', 'dslr_base_dualopt')]
    contrasts = []
    for label, proposed, baseline in comparisons:
        row = {'comparison': label, 'proposed': proposed, 'reference': baseline,
               'metrics': {k: stats([by_configuration[proposed][s][k] - by_configuration[baseline][s][k]
                                     for s in [0, 1, 2]]) for k in ALL_METRICS}}
        contrasts.append(row)
    # Interaction contrasts quantify whether two modules add beyond individual effects.
    for label, full, a, b, base in [
        ('Mobile feature-identity AP interaction', 'mobile_1024_full', 'mobile_1024_features', 'mobile_1024_identity', 'mobile_1024_base'),
        ('Pano factor-prototype AP interaction', 'pano_factor_prototype', 'pano_factor', 'pano_prototype', 'pano_base'),
        ('DSLR CMFA-DualOpt AP interaction', 'dslr_cmfa_dualopt', 'dslr_cmfa_adamw', 'dslr_base_dualopt', 'dslr_base_adamw')]:
        contrasts.append({'comparison': label, 'formula': 'full - A - B + base',
                          'metrics': {k: stats([by_configuration[full][s][k] - by_configuration[a][s][k]
                                                - by_configuration[b][s][k] + by_configuration[base][s][k]
                                                for s in [0, 1, 2]]) for k in ALL_METRICS}})

    prior_rows = []
    for seed in [0, 1, 2]:
        baseline = by_configuration['pano_factor_prototype'][seed]
        prior_rows.append({'seed': seed, 'prior': 'none', **{k: baseline[k] for k in ALL_METRICS}, 'delta_AP_vs_none': 0., 'metric_path': baseline['metric_path']})
        for mode in ['center', 'shape', 'both']:
            r = next(r for r in replays if r['configuration'] == 'pano_factor_prototype' and r['seed'] == seed and r['prior'] == mode)
            prior_rows.append({'seed': seed, 'prior': mode, **{k: r[k] for k in ALL_METRICS},
                               'delta_AP_vs_none': r['AP'] - baseline['AP'], 'metric_path': r['metric_path']})
    prior_summary = []
    for mode in ['none', 'center', 'shape', 'both']:
        rows = sorted([r for r in prior_rows if r['prior'] == mode], key=lambda r: r['seed'])
        prior_summary.append({'prior': mode, 'metrics': {k: stats([r[k] for r in rows]) for k in ALL_METRICS},
                              'paired_AP_difference_vs_none': stats([r['delta_AP_vs_none'] for r in rows])})

    factor_rows = []
    for name, training in [('mobile_1024_full', True), ('internal_mobile_factor_train_off', False)]:
        baseline = by_configuration[name][0]
        factor_rows.append({'training_fusion': training, 'inference_fusion': training,
                            **{k: baseline[k] for k in ALL_METRICS}, 'checkpoint_path': baseline['checkpoint_path'],
                            'metric_path': baseline['metric_path']})
        r = next(r for r in replays if r['configuration'] == name)
        factor_rows.append({'training_fusion': training, 'inference_fusion': not training,
                            **{k: r[k] for k in ALL_METRICS}, 'checkpoint_path': baseline['checkpoint_path'],
                            'metric_path': r['metric_path']})

    # Internal runs compare against their explicit, same-seed reference.
    internal_rows = []
    for job in jobs:
        if job['phase'] != 'internal':
            continue
        name = job['configuration']
        reference_name = 'dslr_cmfa_dualopt' if job['cohort'] == 'DSLR' else 'internal_mobile_factor_train_off' if '_factor_no_' in name else 'mobile_1024_full'
        ref = by_configuration[reference_name][0]
        r = by_configuration.get(name, {}).get(0)
        internal_rows.append({'configuration': name, 'cohort': job['cohort'], 'seed': 0,
                              'status': 'complete' if r else 'pending', 'reference': reference_name,
                              **{k: r[k] if r else None for k in ALL_METRICS},
                              'delta_AP_vs_reference': r['AP'] - ref['AP'] if r else None,
                              'params_M': r['params_M'] if r else None,
                              'best_epoch': r['best_epoch'] if r else None,
                              'metric_path': r['metric_path'] if r else None,
                              'controls': job['controls']})

    report = {'snapshot_time_Asia_Shanghai': snapshot, 'source_manifest_sha256': sha(ROOT / 'execution_manifest.json'),
              'jobs_total': len(jobs), 'completed_runs': len(runs), 'primary_completed': sum(r['phase'] == 'primary' for r in runs),
              'internal_completed': sum(r['phase'] == 'internal' for r in runs),
              'pending_jobs': pending, 'all_completed_runs_verified': True,
              'statistical_unit_CI': 'Paired training seeds; df=2 Student-t mean CI. No patient-level sampling CI and no independent test.',
              'fixed_threshold_protocol': 'Native top-300, score>=0.30, class-agnostic maximum-sum-IoU Hungarian assignment then accept IoU>=0.50; FDI agreement defines correct detections.',
              'CI_multiplicity': 'Exploratory multiple contrasts; CIs are unadjusted. Do not present as confirmatory significance.',
              'primary_summary': summaries, 'paired_contrasts': contrasts, 'prior_summary': prior_summary,
              'factor_fusion_2x2': factor_rows, 'internal_ablations': internal_rows, 'run_provenance': provenance}
    write('controlled_evidence.json', report)
    write('active_configuration_roles.json', settings)
    write('learning_curves.json', learning)
    csv_write('all_completed_runs.csv', runs)
    csv_write('primary_seed_results.csv', [r for r in runs if r['phase'] == 'primary'])
    csv_write('pending_runs.csv', pending)
    csv_write('internal_ablations_seed0.csv', internal_rows)
    csv_write('spatial_shape_per_seed.csv', prior_rows)
    csv_write('factor_fusion_2x2_seed0.csv', factor_rows)
    csv_write('primary_mean_sd.csv', [{'configuration': r['configuration'], 'cohort': r['cohort'], 'params_M': r['params_M'],
                                      **{f'{k}_{v}': r['metrics'][k][v] for k in ALL_METRICS for v in ['mean', 'SD', 'CI95_low', 'CI95_high']}}
                                     for r in summaries])
    csv_write('paired_contrasts.csv', [{'comparison': r['comparison'], **{f'{k}_{v}': r['metrics'][k][v]
                                       for k in ALL_METRICS for v in ['mean', 'SD', 'CI95_low', 'CI95_high']},
                                       'AP_seed0': r['metrics']['AP']['values'][0], 'AP_seed1': r['metrics']['AP']['values'][1],
                                       'AP_seed2': r['metrics']['AP']['values'][2]} for r in contrasts])
    print(json.dumps({k: report[k] for k in ['snapshot_time_Asia_Shanghai', 'jobs_total', 'completed_runs', 'primary_completed', 'internal_completed', 'all_completed_runs_verified']}, indent=2))
    for row in summaries:
        print(row['configuration'], 'AP', f"{row['metrics']['AP']['mean']:.3f} ± {row['metrics']['AP']['SD']:.3f}")
    for row in prior_summary:
        print('SpatialShape', row['prior'], row['metrics']['AP'], 'delta', row['paired_AP_difference_vs_none'])
    for row in contrasts:
        print(row['comparison'], row['metrics']['AP'])


if __name__ == '__main__':
    main()
