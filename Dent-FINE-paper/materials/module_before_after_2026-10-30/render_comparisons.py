"""Publication-ready static panels from the captured features and predictions."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
FONTS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman')
for path in FONTS.glob('*.TTF'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                     'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'savefig.facecolor': 'white'})
PROV = json.loads((HERE/'capture_provenance.json').read_text())
ANN = json.loads(Path(PROV['annotation']).read_text())
IMAGES = {i['id']: i for i in ANN['images']}
GT = defaultdict(list)
for a in ANN['annotations']:
    GT[a['image_id']].append(a)
FDI = {c['id']: int(c['name']) for c in ANN['categories']}
ROOT = Path(PROV['image_root'])
CASES = [182, 155, 139]
CAPTIONS = {}
MANIFEST = {'checkpoint': PROV['checkpoint'], 'checkpoint_sha256': PROV['checkpoint_sha256'],
            'seed': PROV['seed'], 'prototype_temperature': .50, 'figures': {}}


def image(iid):
    return Image.open(ROOT/IMAGES[iid]['file_name']).convert('RGB')


def box_crop(box, iw, ih, scale=3.35):
    x, y, w, h = box
    cw = max(scale*w, 2.5*h*4/3)
    ch = cw*3/4
    ratio = min(1., iw/cw, ih/ch)
    cw *= ratio
    ch *= ratio
    x0 = min(max(x+w/2-cw/2, 0), iw-cw)
    y0 = min(max(y+h/2-ch/2, 0), ih-ch)
    return [float(x0), float(y0), float(x0+cw), float(y0+ch)]


def oral_crop(iid):
    # Portrait images cannot show the whole dental field in a landscape cell
    # without cropping teeth. Use explicitly identified incisor regions instead.
    if iid in [155, 139]:
        code = 41 if iid == 155 else 81
        target = next(g for g in GT[iid] if FDI[g['category_id']] == code)
        return box_crop(target['bbox'], IMAGES[iid]['width'], IMAGES[iid]['height'])
    b = np.array([g['bbox'] for g in GT[iid]])
    lo = b[:, :2].min(0)
    hi = (b[:, :2]+b[:, 2:]).max(0)
    iw, ih = IMAGES[iid]['width'], IMAGES[iid]['height']
    cx, cy = (lo+hi)/2
    cw = max(1.15*(hi[0]-lo[0]), 1.15*(hi[1]-lo[1])*4/3)
    ch = cw*3/4
    ratio = min(1., iw/cw, ih/ch)
    cw *= ratio
    ch *= ratio
    x0 = min(max(cx-cw/2, 0), iw-cw)
    y0 = min(max(cy-ch/2, 0), ih-ch)
    return [float(x0), float(y0), float(x0+cw), float(y0+ch)]


def setup_image_axes(ax, crop):
    ax.set_xlim(crop[0], crop[2])
    ax.set_ylim(crop[3], crop[1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def export(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        if text.get_visible() and text.get_text():
            bb = text.get_window_extent(renderer)
            assert bb.x0 >= -.5 and bb.y0 >= -.5 and bb.x1 <= fig.bbox.width+.5 and bb.y1 <= fig.bbox.height+.5, (stem, text.get_text(), bb)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(HERE/f'{stem}.{ext}', dpi=600)
    fig.savefig(HERE/f'{stem}_preview.png', dpi=160)
    plt.close(fig)


def subset(arr, iid):
    r = oral_crop(iid)
    H, W = arr.shape
    iw, ih = IMAGES[iid]['width'], IMAGES[iid]['height']
    xx = (np.arange(W)+.5)/W*iw
    yy = (np.arange(H)+.5)/H*ih
    mask = ((xx[None, :] >= r[0]) & (xx[None, :] <= r[2]) &
            (yy[:, None] >= r[1]) & (yy[:, None] <= r[3]))
    return arr[mask]


def heatmap_panels(module, overlay):
    data = {iid: np.load(HERE/f'features_image_{iid}.npz') for iid in CASES}
    displayed = np.concatenate([subset(data[iid][module+'_'+mode], iid)
                                for iid in CASES for mode in ['before', 'after']])
    upper = float(np.percentile(displayed, 99.5))
    delta_upper = max(float(np.max(np.abs(subset(data[iid][module+'_delta_RMS'], iid)))) for iid in CASES)
    fig = plt.figure(figsize=(8.35, 6.25))
    left, gap, width, height = .040, .025, .215, .215
    bottoms = [.710, .425, .140]
    titles = ['Original', 'Before '+module, 'After '+module, 'Δ RMS']
    clipped = []
    for row, iid in enumerate(CASES):
        im = image(iid)
        extent = [0, im.width, im.height, 0]
        crop = oral_crop(iid)
        for col in range(4):
            ax = fig.add_axes([left+col*(width+gap), bottoms[row], width, height])
            if col == 0:
                ax.imshow(im, interpolation='antialiased')
            elif col in [1, 2]:
                key = module+('_before' if col == 1 else '_after')
                if overlay:
                    ax.imshow(im, interpolation='antialiased')
                ax.imshow(data[iid][key], extent=extent, cmap='Blues', vmin=0, vmax=upper,
                          alpha=.60 if overlay else 1., interpolation='bilinear')
                clipped.append({'image_id': iid, 'map': key,
                                'fraction_above_shared_limit_in_crop': float((subset(data[iid][key], iid)>upper).mean())})
            else:
                ax.imshow(data[iid][module+'_delta_RMS'], extent=extent,
                          cmap='RdBu_r', vmin=-delta_upper, vmax=delta_upper, interpolation='bilinear')
            setup_image_axes(ax, crop)
            if row == 0:
                ax.set_title(titles[col], pad=8, fontsize=10.1)
        fig.text(.007, bottoms[row]+height-.005, '('+chr(97+row)+')', va='top', fontsize=10.5)
    cax = fig.add_axes([.282, .075, .420, .018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, upper), cmap='Blues'), cax=cax,
                      orientation='horizontal', ticks=[0, upper/2, upper])
    cb.ax.set_title('Feature RMS', fontsize=9.2, pad=5)
    cb.ax.set_xticklabels([f'{v:.2f}' for v in [0, upper/2, upper]])
    cb.ax.tick_params(length=2, labelsize=8)
    cb.outline.set_visible(False)
    cax = fig.add_axes([.780, .075, .200, .018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-delta_upper, delta_upper), cmap='RdBu_r'),
                      cax=cax, orientation='horizontal', ticks=[-delta_upper, 0, delta_upper])
    cb.ax.set_title('Δ RMS', fontsize=9.2, pad=5)
    cb.ax.set_xticklabels([f'{v:+.2f}' if v != 0 else '0' for v in [-delta_upper, 0, delta_upper]])
    cb.ax.tick_params(length=2, labelsize=8)
    cb.outline.set_visible(False)
    stem = f'Figure_{module}_before_after_'+('overlay' if overlay else 'heatmaps')
    export(fig, stem)
    definition = 'concatenation of the two spatially aligned inputs' if module == 'CMFA' else 'the input to the residual attention module'
    caption = (f'Actual feature responses before and after {module} in three Mobile validation images. '
               f'The same DENT-STRIDE-Mobile checkpoint (seed 2; prototype temperature 0.50) supplies every panel. '
               f'Before denotes {definition}. All before and after responses are channel RMS with one shared color scale across the three images; '
               'the upper display limit is the pooled 99.5th percentile within the displayed crops. '
               'The difference map is RMS(after) minus RMS(before), with red indicating increased magnitude and blue decreased magnitude; '
               'it is not the RMS of the vector difference. The same crop and image registration are used within each row. '
               'Row (a) shows an oral overview; rows (b,c) show local incisor regions centered on reference FDI 41 and 81, respectively. '
               + ('Blue feature maps are overlaid with constant alpha 0.60. ' if overlay else '')
               + 'These are actual feature-response maps, not attention probabilities or Grad-CAM; stronger responses do not by themselves establish improved detection.')
    CAPTIONS[stem] = caption
    MANIFEST['figures'][stem] = {'image_ids': CASES, 'overlay': overlay,
                'module': PROV[module+'_module'], 'shared_RMS_vmin': 0., 'shared_RMS_vmax': upper,
                'delta_RMS_domain': [-delta_upper, delta_upper],
                'display_clipping': clipped, 'crop_by_image': {str(iid): oral_crop(iid) for iid in CASES},
                'raw_data': [str(HERE/f'features_image_{iid}.npz') for iid in CASES]}


def iou(a, b):
    a = np.array(a, float).reshape(-1, 4)
    b = np.array(b, float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2]+a[:, None, 2:], b[None, :, :2]+b[None, :, 2:])
    inter = np.maximum(hi-lo, 0).prod(-1)
    return inter/np.maximum(a[:, 2:].prod(-1)[:, None]+b[:, 2:].prod(-1)[None, :]-inter, 1e-12)


def match(predictions, truths):
    st = {g['id']: None for g in truths}
    matrix = iou([p['bbox'] for p in predictions], [g['bbox'] for g in truths])
    for p, g in zip(*linear_sum_assignment(-matrix)):
        if matrix[p, g] >= .50:
            st[truths[g]['id']] = {'prediction': predictions[p], 'IoU': float(matrix[p, g])}
    return st


def factor_effects():
    predictions = {}
    for mode in ['off', 'on']:
        by = defaultdict(list)
        with gzip.open(HERE/f'factor_fusion_{mode}_predictions.json.gz', 'rt') as stream:
            for row in json.load(stream):
                if row['score'] >= .30:
                    by[row['image_id']].append(row)
        predictions[mode] = by
    raw = np.load(HERE/'same_weight_factor_candidates.npz')
    indexes = {int(v): j for j, v in enumerate(raw['image_ids'])}
    inventory = {'improvement': [], 'adverse': []}
    totals = {mode: Counter() for mode in ['off', 'on']}
    for iid, gt in GT.items():
        states = {m: match(predictions[m][iid], gt) for m in ['off', 'on']}
        tp = {}
        for mode in ['off', 'on']:
            tp[mode] = sum(states[mode][g['id']] is not None and
                           states[mode][g['id']]['prediction']['category_id'] == g['category_id'] for g in gt)
            totals[mode].update(TP=tp[mode], GT=len(gt), predictions=len(predictions[mode][iid]),
                duplicate_images=int(any(n > 1 for n in Counter(p['category_id'] for p in predictions[mode][iid]).values())))
        for g in gt:
            before = states['off'][g['id']]
            after = states['on'][g['id']]
            ok0 = before is not None and before['prediction']['category_id'] == g['category_id']
            ok1 = after is not None and after['prediction']['category_id'] == g['category_id']
            kind = 'improvement' if ok1 and not ok0 else 'adverse' if ok0 and not ok1 else None
            if not kind or before is None or after is None:
                continue
            b, a = before['prediction'], after['prediction']
            if b['query_id'] != a['query_id']:
                continue
            q, at = a['query_id'], indexes[iid]
            if int(raw['logits_off'][at, q].argmax()) != b['category_id'] or int(raw['logits_on'][at, q].argmax()) != a['category_id']:
                continue
            assert np.max(np.abs(np.array(b['bbox'])-a['bbox'])) == 0
            scores = {}
            for mode in ['off', 'on']:
                logits = raw['logits_'+mode][at, q]
                prob = 1/(1+np.exp(-np.clip(logits, -80, 80)))
                scores[mode] = {str(FDI[cid]): float(prob[cid]) for cid in set([g['category_id'], a['category_id'], b['category_id']])}
            inventory[kind].append({'image_id': iid, 'target_annotation': g, 'target_FDI': FDI[g['category_id']],
                    'off': before, 'on': after, 'off_FDI': FDI[b['category_id']], 'on_FDI': FDI[a['category_id']],
                    'query_id': q, 'type': kind, 'image_correct_matches': tp, 'same_query_class_scores': scores})
    # Include two distinct kinds of favorable change and the observed adverse change.
    favorable = inventory['improvement']
    mixed = [r for r in favorable if (r['off_FDI'] >= 50) != (r['on_FDI'] >= 50)]
    position = [r for r in favorable if r['off_FDI']//10 == r['on_FDI']//10]
    chosen = [max(mixed, key=lambda r: r['target_annotation']['area']),
              max(position, key=lambda r: r['target_annotation']['area']),
              max(inventory['adverse'], key=lambda r: r['target_annotation']['area'])]
    fig = plt.figure(figsize=(8.35, 6.20))
    captions = ['Original', 'Ground truth', 'Factor fusion off', 'Factor fusion on']
    bottoms = [.705, .415, .125]
    for row, case in enumerate(chosen):
        iid = case['image_id']
        im = image(iid)
        crop = box_crop(case['target_annotation']['bbox'], im.width, im.height)
        case['crop_xyxy'] = crop
        for col, title in enumerate(captions):
            ax = fig.add_axes([.040+col*.240, bottoms[row], .215, .215])
            ax.imshow(im, interpolation='antialiased')
            if col > 0:
                p = case['target_annotation'] if col == 1 else case['off' if col == 2 else 'on']['prediction']
                color = '#087B80' if col == 1 else '#AD5528'
                x, y, w, h = p['bbox']
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.2))
                ax.text(x+.035*w, y-.035*(crop[3]-crop[1]), str(FDI[p['category_id']]),
                        color=color, ha='left', va='bottom', fontsize=11, fontweight='bold',
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .92, 'pad': .65})
            setup_image_axes(ax, crop)
            if row == 0:
                ax.set_title(title, pad=8, fontsize=10.1)
        fig.text(.007, bottoms[row]+.210, '('+chr(97+row)+')', fontsize=10.5, va='top')
    stem = 'Figure_Mobile_factor_fusion_before_after'
    export(fig, stem)
    CAPTIONS[stem] = ('Same-checkpoint effect of factor-score fusion in Mobile images. The trained DENT-STRIDE-Mobile '
        'seed-2 checkpoint (prototype temperature 0.50) is unchanged; inference fusion weights are zero in the off condition '
        'and 0.10/0.05/0.03 for quadrant/position/dentition in the on condition. '
        '(a) The highest-scoring FDI of the same query changes from 16 to the reference 55. '
        '(b) It changes from 65 to the reference 64. (c) An adverse example changes the correct 81 to 41. '
        'Identical local crops are used across columns; ground truth is teal and predictions orange. '
        'Only the target ground-truth box and the highest-scoring class of the tracked query are annotated; '
        'other class candidates can coexist in native top-300 outputs. Both tracked predictions have score >=0.30 '
        'and localization IoU >=0.50. Model weights, query boxes, and native postprocessing are unchanged. '
        'These examples illustrate score reordering, not the effect of training without factor-head supervision '
        'or the population frequency of corrections.')
    MANIFEST['figures'][stem] = {'cases': chosen, 'complete_same_query_top1_change_inventory': inventory,
        'fixed_threshold_counts_GPU_FP32': {k: dict(v) for k, v in totals.items()},
        'note': 'These batch-1 GPU FP32 diagnostics do not replace the manuscript archived evaluation endpoints.',
        'display': 'Tracked query top-1 FDI; does not assert that other native candidates are absent',
        'same_weights': True, 'same_query_boxes': True, 'factor_training_supervision_retained': True}


def main():
    for module in ['CMFA', 'HOOA']:
        for overlay in [False, True]:
            heatmap_panels(module, overlay)
    factor_effects()
    (HERE/'figure_captions.json').write_text(json.dumps(CAPTIONS, indent=2, ensure_ascii=False))
    (HERE/'figure_sources_and_display_scales.json').write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False))
    print(json.dumps({'figures': list(MANIFEST['figures']),
                      'factor_cases': [{k: r[k] for k in ['image_id', 'target_FDI', 'off_FDI', 'on_FDI', 'type']}
                         for r in MANIFEST['figures']['Figure_Mobile_factor_fusion_before_after']['cases']]}, indent=2))


if __name__ == '__main__':
    main()
