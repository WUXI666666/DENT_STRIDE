"""Export complete FDI outcome matrices from frozen seed-0 native predictions.

CPU-only. Does not rerun a detector, modify datasets, or change training outputs.
Every GT contributes one outcome: a localized predicted identity or Missed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03')
TRAIN = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')
DEST = ROOT / 'figures/errors'
PANO32 = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
          38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
JOBS = [('Pano', 'pano_factor_prototype', 'replay_prior_both', 9),
        ('Mobile', 'mobile_1024_full', 'evaluation', 10),
        ('DSLR', 'dslr_cmfa_dualopt', 'evaluation', 11)]
SCORE, IOU = .30, .50


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    digest = sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    union = a[:, 2:].prod(-1)[:, None] + b[:, 2:].prod(-1)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def mean_valid(a):
    valid = np.asarray(a)[np.asarray(a) > -1]
    return float(valid.mean() * 100) if len(valid) else None


def per_class_coco(source, category_id):
    cats = source['category_ids'].tolist()
    index = cats.index(category_id)
    precision, recall = source['precision'], source['recall']
    i50 = int(np.flatnonzero(np.isclose(source['IoU_thresholds'], .50))[0])
    i75 = int(np.flatnonzero(np.isclose(source['IoU_thresholds'], .75))[0])
    return {'AP': mean_valid(precision[:, :, index, 0, -1]),
            'AP50': mean_valid(precision[i50, :, index, 0, -1]),
            'AP75': mean_valid(precision[i75, :, index, 0, -1]),
            'AR100': mean_valid(recall[:, index, 0, -1])}


def plot(outcome, cohort, number):
    codes = outcome['fdi_order']; matrix = np.asarray(outcome['row_percent'], dtype=float)
    count = len(codes)
    # Fixed [0,100] scale across all domains; NaN means no validation GT support.
    cmap = LinearSegmentedColormap.from_list('paper_blue', ['#ffffff', '#adc5d8', '#174968'])
    cmap.set_bad('#dedede')
    fig = plt.figure(figsize=(7.7, 7.7), facecolor='white')
    ax = fig.add_axes([.095, .20, .75, .71])
    # Equal cell dimensions make the matrix a spatially faithful complete grid.
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=100,
                      aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(count + 1)); ax.set_yticks(np.arange(count))
    ax.set_xticklabels([*map(str, codes), 'Missed'], fontsize=8,
                       rotation=90 if count > 32 else 0)
    ax.set_yticklabels(codes, fontsize=8)
    ax.get_xticklabels()[-1].set_rotation(90)
    ax.tick_params(axis='both', length=0, pad=3)
    ax.set_xlabel('Predicted FDI', fontsize=10, labelpad=9)
    ax.set_ylabel('Reference FDI', fontsize=10, labelpad=6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.6)
    # Quadrant boundaries preserve clinical ordering and reveal common confusions.
    for boundary in [i for i in range(1, count) if codes[i] // 10 != codes[i - 1] // 10]:
        ax.axhline(boundary - .5, color='#b7b7b7', linewidth=.4)
        ax.axvline(boundary - .5, color='#b7b7b7', linewidth=.4)
    ax.axvline(count - .5, color='#777777', linewidth=.7)
    supports = fig.add_axes([.86, .20, .13, .71], sharey=ax)
    supports.set_xlim(-.05, 1.10)
    supports.set_ylim(count - .5, -.5)
    supports.axis('off')
    supports.text(.21, 1.025, 'Train', transform=supports.transAxes, ha='center', fontsize=9)
    supports.text(.74, 1.025, 'Val.', transform=supports.transAxes, ha='center', fontsize=9)
    for row, (train, val) in enumerate(zip(outcome['train_support'], outcome['val_support'])):
        if not val:
            supports.axhspan(row - .5, row + .5, color='#dedede', linewidth=0)
        supports.text(.20, row, str(train), ha='center', va='center', fontsize=8)
        supports.text(.80, row, str(val), ha='center', va='center', fontsize=8)
    cax = fig.add_axes([.31, .07, .41, .017])
    bar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=[0, 25, 50, 75, 100])
    bar.ax.tick_params(length=2, labelsize=8)
    bar.set_label('Reference-row percentage (%)', fontsize=9, labelpad=3)
    # The caption, not a checklist-like internal title, describes cohort and scope.
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label,
                 *bar.ax.get_xticklabels(), bar.ax.xaxis.label]:
        bb = text.get_window_extent(fig.canvas.get_renderer())
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= width and bb.y1 <= height, text.get_text()
    stem = DEST / f'Fig_S{number}_{cohort}_complete_FDI_confusion'
    fig.savefig(str(stem) + '.png', dpi=600, facecolor='white')
    fig.savefig(str(stem) + '.svg', facecolor='white')
    fig.savefig(str(stem) + '.pdf', facecolor='white')
    plt.close(fig)
    return {suffix: str(stem) + '.' + suffix for suffix in ['png', 'svg', 'pdf']}


def analyze(cohort, configuration, replay, number):
    config_path = TRAIN / 'configs' / configuration / 'seed_0.yml'
    config = yaml.safe_load(config_path.read_text())
    train_path = Path(config['train_dataloader']['dataset']['ann_file'])
    val_path = Path(config['val_dataloader']['dataset']['ann_file'])
    run = TRAIN / 'runs' / configuration / 'seed_0'
    source = run / replay
    predictions_path = source / 'predictions_coco.json.gz'
    metrics_path = source / 'metrics.json'
    coco_path = source / 'coco_precision_recall.npz'
    train, val, metrics = read(train_path), read(val_path), read(metrics_path)
    categories = val['categories']
    if cohort == 'Pano':
        assert config['fdi_label_order'] == 'dentex_enumeration32'
        cat_to_fdi = {c['id']: PANO32[c['id']] for c in categories}
    else:
        cat_to_fdi = {c['id']: int(c.get('fdi', c['name'])) for c in categories}
    assert len(set(cat_to_fdi.values())) == config['num_classes']
    assert set(c['id'] for c in train['categories']) == set(cat_to_fdi)
    with gzip.open(predictions_path, 'rt') as file:
        predictions = json.load(file)
    image_ids = {image['id'] for image in val['images']}
    assert {p['image_id'] for p in predictions} == image_ids
    assert {p['category_id'] for p in predictions} <= set(cat_to_fdi)
    assert not any(a.get('iscrowd', 0) or a.get('ignore', 0) for a in val['annotations'])
    codes = sorted(cat_to_fdi.values()); fdi_to_row = {fdi: i for i, fdi in enumerate(codes)}
    num_classes = len(codes)
    matrix = np.zeros((num_classes, num_classes + 1), dtype=int)
    by_prediction, by_gt = defaultdict(list), defaultdict(list)
    for p in predictions:
        if p['score'] >= SCORE:
            by_prediction[p['image_id']].append(p)
    for g in val['annotations']:
        by_gt[g['image_id']].append(g)
    extra = Counter(); per_image = []
    for image_id in sorted(image_ids):
        pp, gg = by_prediction[image_id], by_gt[image_id]
        accepted_gt, accepted_pred = {}, set()
        if pp and gg:
            overlap = bbox_iou([p['bbox'] for p in pp], [g['bbox'] for g in gg])
            rows, cols = linear_sum_assignment(-overlap)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if overlap[r, c] >= IOU:
                    accepted_gt[c] = r; accepted_pred.add(r)
        correct = 0
        for index, g in enumerate(gg):
            ref = fdi_to_row[cat_to_fdi[g['category_id']]]
            if index in accepted_gt:
                prediction = pp[accepted_gt[index]]
                predicted = fdi_to_row[cat_to_fdi[prediction['category_id']]]
                matrix[ref, predicted] += 1
                correct += int(prediction['category_id'] == g['category_id'])
            else:
                matrix[ref, -1] += 1
        for index, p in enumerate(pp):
            if index not in accepted_pred:
                extra[cat_to_fdi[p['category_id']]] += 1
        per_image.append({'image_id': image_id, 'GT': len(gg), 'retained_predictions': len(pp),
                          'localized_matches': len(accepted_gt), 'correct_FDI_matches': correct,
                          'missed_GT': len(gg) - len(accepted_gt),
                          'unmatched_predictions': len(pp) - len(accepted_pred)})
    train_count = Counter(cat_to_fdi[a['category_id']] for a in train['annotations'])
    val_count = Counter(cat_to_fdi[a['category_id']] for a in val['annotations'])
    train_support = [train_count[fdi] for fdi in codes]
    val_support = [val_count[fdi] for fdi in codes]
    assert matrix.sum(axis=1).tolist() == val_support
    total, localized, correct = int(matrix.sum()), int(matrix[:, :-1].sum()), int(np.trace(matrix[:, :-1]))
    retained = sum(len(p) for p in by_prediction.values())
    counts = metrics['fixed_threshold']['counts']
    assert (total, localized, correct, retained) == (counts['GT'], counts['localized_matches'],
                                                   counts['correct_FDI_matches'], counts['predictions'])
    assert sum(extra.values()) == retained - localized
    assert retained - correct == counts['FP'] and total - correct == counts['FN']
    percentages = np.divide(matrix * 100., np.asarray(val_support)[:, None],
                            out=np.full(matrix.shape, np.nan), where=np.asarray(val_support)[:, None] > 0)
    coco = np.load(coco_path)
    perclass = []
    native_by_fdi = {fdi: category for category, fdi in cat_to_fdi.items()}
    for index, code in enumerate(codes):
        perclass.append({'FDI': code, 'native_category_id': native_by_fdi[code],
                         'train_support': train_support[index], 'validation_support': val_support[index],
                         **per_class_coco(coco, native_by_fdi[code]),
                         'correct_FDI': int(matrix[index, index]), 'localized_GT': int(matrix[index, :-1].sum()),
                         'missed_GT': int(matrix[index, -1]), 'unmatched_predictions': extra[code]})
    # Macro COCO means (supported classes only) independently reproduce saved metrics.
    for key in ['AP', 'AP50', 'AP75', 'AR100']:
        aggregate = np.mean([row[key] for row in perclass if row[key] is not None])
        assert np.isclose(aggregate, metrics['metrics_percent'][key], rtol=0, atol=1e-10), (cohort, key)
    outcome = {'cohort': cohort, 'configuration': configuration, 'seed': 0, 'prior': metrics['prior'],
               'figure_number': number, 'score_threshold': SCORE, 'IoU_threshold': IOU,
               'assignment': 'class-agnostic Hungarian maximum total IoU, followed by IoU acceptance',
               'fdi_order': codes, 'prediction_columns': [*codes, 'Missed'],
               'native_category_to_FDI': cat_to_fdi, 'train_support': train_support,
               'val_support': val_support, 'raw_counts': matrix.tolist(),
               'row_percent': [[None if not np.isfinite(v) else float(v) for v in row] for row in percentages],
               'per_class_metrics': perclass, 'unmatched_predictions_by_FDI': {fdi: extra[fdi] for fdi in codes},
               'summary': {'images': len(image_ids), 'train_images': len(train['images']),
                           'train_teeth': len(train['annotations']), 'validation_GT': total,
                           'retained_predictions': retained, 'localized_matches': localized,
                           'correct_FDI_matches': correct, 'wrong_FDI_localized_matches': localized - correct,
                           'missed_GT': total - localized, 'unmatched_predictions': retained - localized,
                           'detection_FP_including_wrong_FDI': retained - correct,
                           'detection_FN_including_wrong_FDI': total - correct,
                           'no_validation_support': [fdi for fdi, n in zip(codes, val_support) if n == 0],
                           'no_training_support': [fdi for fdi, n in zip(codes, train_support) if n == 0]},
               'sources': {'train_annotation': str(train_path), 'train_annotation_sha256': file_sha(train_path),
                           'validation_annotation': str(val_path), 'validation_annotation_sha256': file_sha(val_path),
                           'prediction': str(predictions_path), 'prediction_sha256': file_sha(predictions_path),
                           'config': str(config_path), 'config_sha256': file_sha(config_path),
                           'checkpoint': str(run / 'best.pth'), 'checkpoint_sha256': metrics['checkpoint_sha256'],
                           'metrics': str(metrics_path), 'metrics_sha256': file_sha(metrics_path),
                           'coco_arrays': str(coco_path), 'coco_arrays_sha256': file_sha(coco_path)},
               'verification': {'row_totals_equal_validation_class_support': True,
                                'totals_equal_frozen_fixed_threshold_metrics': True,
                                'macro_per_class_COCO_equal_saved_metrics': True},
               'matrix_scope': 'One outcome per GT. Unmatched/extra predictions are excluded from the GT-row matrix and counted separately. Wrong-FDI localized matches are off diagonal; missed means no accepted localized match.'}
    # None converts to NaN in plotting; JSON uses null, never non-standard NaN.
    stem = DEST / f'{cohort}_complete_FDI_confusion'
    with (stem.with_suffix('.csv')).open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['Reference_FDI', *codes, 'Missed'])
        writer.writerows([[fdi, *row.tolist()] for fdi, row in zip(codes, matrix)])
    with (DEST / f'{cohort}_per_class_performance_and_support.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(perclass[0])); writer.writeheader(); writer.writerows(perclass)
    with (DEST / f'{cohort}_per_image_FDI_outcomes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image[0])); writer.writeheader(); writer.writerows(per_image)
    outcome['figure_files'] = plot(outcome, cohort, number)
    stem.with_suffix('.json').write_text(json.dumps(outcome, indent=2, allow_nan=False))
    return outcome


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'references/times_new_roman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Times New Roman', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.linewidth': .6, 'savefig.facecolor': 'white'})
    assert font_manager.findfont('Times New Roman', fallback_to_default=False).endswith('Times.TTF')
    outcomes = [analyze(*job) for job in JOBS]
    manifest = {'created_Asia_Shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'script': str(Path(__file__).resolve()), 'script_sha256': file_sha(__file__),
                'cohorts': {outcome['cohort']: {'summary': outcome['summary'], 'sources': outcome['sources'],
                                             'figure_files': outcome['figure_files'],
                                             'verification': outcome['verification']} for outcome in outcomes}}
    (DEST / 'complete_FDI_confusion_provenance.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({o['cohort']: o['summary'] for o in outcomes}, indent=2))


if __name__ == '__main__':
    main()
