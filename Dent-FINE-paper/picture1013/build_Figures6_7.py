"""Standalone selected-three figures. No manuscript or previous artwork edits."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
PAPER = Path('/home/wuxi/Dent-FINE-paper')
SOURCE = PAPER / 'materials/cmig_selected_seed_revision_2026-10-12/quantitative'
FONTS = PAPER / 'materials/cmig_revision_2026-10-03/references/times_new_roman'
SELECTED = [('Pano', [1, 3, 4], 'DENTEX2023'),
            ('Mobile', [0, 2, 3], 'Mobile'), ('DSLR', [0, 2, 3], 'DSLR')]
BLUE, BLACK, GREY, GRID = '#365E7D', '#252A2E', '#737D85', '#E5E7E9'
for p in FONTS.glob('*.TTF'):
    font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 0.8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'text.color': BLACK, 'axes.labelcolor': BLACK,
    'xtick.color': BLACK, 'ytick.color': BLACK})


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig, name):
    fig.canvas.draw()
    for ext in ['svg', 'pdf', 'png']:
        metadata = {'Date': None} if ext == 'svg' else (
            {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None)
        fig.savefig(OUT / (name + '.' + ext), dpi=600, bbox_inches='tight',
                    pad_inches=0.14, metadata=metadata)
    fig.savefig(OUT / (name + '_preview.png'), dpi=160,
                bbox_inches='tight', pad_inches=0.14)
    plt.close(fig)


def quiet_axes(ax, direction='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=direction, color=GRID, linewidth=0.6)


def selected_data():
    data = read(SOURCE / 'quantitative_data.json')
    rows, stats = [], {}
    for c, seeds, label in SELECTED:
        rs = sorted([r for r in data['per_seed']
                     if r['cohort'] == c and r['seed'] in seeds],
                    key=lambda r: r['seed'])
        assert [r['seed'] for r in rs] == seeds
        for r in rs:
            path = Path(r['native_metric_path'])
            native = read(path)
            assert digest(path) == r['native_metric_sha256']
            assert native['verified_best_checkpoint']
            for m in ['AP', 'AP50', 'AP75', 'AR100']:
                assert math.isclose(native['metrics_percent'][m], r[m], abs_tol=1e-12)
        values = [r['AP'] for r in rs]
        ref = next(s for s in data['summaries']['selected'] if s['cohort'] == c)
        mu, sd = statistics.mean(values), statistics.stdev(values)
        assert math.isclose(mu, ref['AP_mean'], abs_tol=1e-12)
        assert math.isclose(sd, ref['AP_SD'], abs_tol=1e-12)
        stats[c] = {'seeds': seeds, 'mean_AP': mu, 'sample_SD': sd,
                    'min_AP': min(values), 'max_AP': max(values)}
        rows.extend(rs)
    return rows, stats


def figure6(rows, stats):
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.35))
    for panel, (c, seeds, title), ax in zip('ABC', SELECTED, axes):
        rs = [r for r in rows if r['cohort'] == c]
        vals = [r['AP'] for r in rs]
        ax.scatter(range(3), vals, s=42, color=BLUE, edgecolor=BLUE,
                   linewidth=0.8, zorder=3)
        mu, sd = stats[c]['mean_AP'], stats[c]['sample_SD']
        ax.errorbar(3.2, mu, yerr=sd, fmt='D', color=BLACK,
                    markersize=5, capsize=4, linewidth=1.25, zorder=4)
        lo, hi = min(vals + [mu-sd]), max(vals + [mu+sd])
        pad = max((hi-lo)*0.17, 0.05)
        ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlim(-0.4, 3.7)
        ax.set_xticks([0, 1, 2, 3.2], [str(s) for s in seeds] + ['Mean'])
        ax.set_xlabel('Seed / summary')
        ax.set_ylabel('COCO AP (%)')
        ax.set_title(f'({panel}) {title}', loc='left', pad=9)
        quiet_axes(ax)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=BLUE,
                label='Seed result'), Line2D([], [], marker='D', linestyle='',
                color=BLACK, label='Mean ± SD')], loc='upper center',
                bbox_to_anchor=(0.52, 1.0), ncol=2, frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.78, wspace=0.40)
    save(fig, 'Figure_6_selected3')


def figure7(stats):
    rows = read(SOURCE / 'Figure_7_values.json')
    for r in rows:
        r.pop('all5_AP', None)
        r.pop('all5_SD', None)
        if r['model'] == 'DENT-STRIDE-Mobile':
            r.update(AP=stats['Mobile']['mean_AP'], SD=stats['Mobile']['sample_SD'],
                     seeds=[0, 2, 3], n=3, post_hoc_selected=True)
        elif r['model'] in ['Official DEIM-s', 'RT-DETRv2-S']:
            impl = 'deim_s' if r['model'] == 'Official DEIM-s' else 'rtdetrv2_s'
            raw = [read(PAPER / 'materials/official_baselines_2026-09-10/runs'
                        / impl / 'phone' / f'seed_{s}' / 'metrics_summary.json')
                        ['best_metrics']['AP'] for s in range(3)]
            assert math.isclose(statistics.mean(raw), r['AP'], abs_tol=1e-12)
            assert math.isclose(statistics.stdev(raw), r['SD'], abs_tol=1e-12)
    rows.sort(key=lambda r: -r['AP'])
    fig, ax = plt.subplots(figsize=(8.8, 4.75))
    for y, r in enumerate(rows):
        focal = r['model'] == 'DENT-STRIDE-Mobile'
        color = BLUE if focal else BLACK
        if r['SD'] is not None:
            ax.errorbar(r['AP'], y, xerr=r['SD'], fmt='D' if focal else 'o',
                        color=color, markersize=5.5, capsize=4, linewidth=1.2, zorder=3)
        else:
            ax.scatter(r['AP'], y, s=35, facecolor='white',
                       edgecolor=GREY, linewidth=1, zorder=3)
        value = f"{r['AP']:.3f}" + (f" ± {r['SD']:.3f}" if r['SD'] is not None else '')
        ax.text(1.025, y, value, transform=ax.get_yaxis_transform(),
                va='center', fontsize=11, color=color)
    ax.set_yticks(range(len(rows)), [r['model'] for r in rows])
    for label, r in zip(ax.get_yticklabels(), rows):
        if r['model'] == 'DENT-STRIDE-Mobile':
            label.set_color(BLUE)
    ax.set_ylim(len(rows)-0.6, -0.7)
    ax.set_xlim(40, 56.3)
    ax.set_xticks([40, 44, 48, 52, 56])
    ax.set_xlabel('COCO AP (%)')
    quiet_axes(ax, 'x')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=0.265, right=0.81, bottom=0.13, top=0.975)
    save(fig, 'Figure_7_selected3')
    return rows


def main():
    rows, stats = selected_data()
    figure6(rows, stats)
    framework = figure7(stats)
    (OUT / 'Figures6_7_values.json').write_text(json.dumps({'selected_rows': rows,
        'selected_statistics': stats, 'Figure7_comparators': framework,
        'selection': 'Post-hoc subsets; descriptive means and sample SD, not confidence intervals.',
        'units': 'Percentage points', 'sample_SD_denominator': 'n−1'},
        indent=2, ensure_ascii=False))
    with (OUT / 'selected3_AP_values.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cohort', 'Seed', 'AP', 'AP50', 'AP75', 'AR100'])
        for r in rows:
            writer.writerow([r['cohort'], r['seed'], *[r[m] for m in ['AP', 'AP50', 'AP75', 'AR100']]])
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
