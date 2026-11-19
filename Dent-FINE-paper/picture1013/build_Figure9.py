#!/usr/bin/env python3
"""Selected-three Pano prior figure, independently verified from native metrics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_selected_seed_revision_2026-10-12/quantitative/quantitative_data.json')
FONTS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman')
SEEDS = [1, 3, 4]
METRICS = ['AP', 'AP50', 'AP75', 'AR100']
MODES = ['none', 'center', 'shape', 'both']
LABELS = ['None', 'Center', 'Shape', 'Center + shape']
BLUE = '#365E7D'
BLACK = '#252A2E'
LIGHT = '#E5E7E9'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_sd(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values)
    # Independent scalar arithmetic check, distinct from statistics helpers.
    check_mean = sum(values) / len(values)
    check_sd = math.sqrt(sum((v-check_mean)**2 for v in values)/(len(values)-1))
    assert math.isclose(m, check_mean, abs_tol=1e-12)
    assert math.isclose(sd, check_sd, abs_tol=1e-12)
    return {'mean': m, 'SD': sd, 'per_seed': values}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SOURCE.read_text())
    rows = sorted((r for r in data['prior_per_seed'] if r['seed'] in SEEDS),
                  key=lambda r:(MODES.index(r['prior']), r['seed']))
    assert len(rows) == 12
    sources = []
    seed_hashes = {}
    for row in rows:
        path = Path(row['native_metric_path'])
        native = json.loads(path.read_text())
        assert digest(path) == row['native_metric_sha256']
        for metric in METRICS:
            assert math.isclose(native['metrics_percent'][metric], row[metric], abs_tol=1e-10)
        assert native['checkpoint_sha256'] == row['checkpoint_sha256']
        seed_hashes.setdefault(row['seed'], native['checkpoint_sha256'])
        assert seed_hashes[row['seed']] == native['checkpoint_sha256']
        sources.append({'seed':row['seed'], 'prior':row['prior'], 'path':str(path),
                        'sha256':digest(path), 'checkpoint_sha256':native['checkpoint_sha256']})

    none = {r['seed']:r for r in rows if r['prior']=='none'}
    statistics_by_prior = []
    for mode in MODES:
        rs = [r for r in rows if r['prior']==mode]
        summary = {'prior':mode, 'seeds':SEEDS}
        existing = next(s for s in data['prior_summaries']['selected'] if s['prior']==mode)
        for metric in METRICS:
            summary[metric] = mean_sd([r[metric] for r in rs])
            summary[metric+'_paired_delta_vs_none'] = mean_sd([r[metric]-none[r['seed']][metric] for r in rs])
            assert math.isclose(summary[metric]['mean'],existing[metric+'_mean'],abs_tol=1e-10)
            assert math.isclose(summary[metric]['SD'],existing[metric+'_SD'],abs_tol=1e-10)
            assert math.isclose(summary[metric+'_paired_delta_vs_none']['mean'],existing[metric+'_delta_mean'],abs_tol=1e-10)
        statistics_by_prior.append(summary)

    cohort_ranges = []
    for cohort, seeds in [('Pano',[1,3,4]),('Mobile',[0,2,3]),('DSLR',[0,2,3])]:
        rs = sorted([r for r in data['per_seed'] if r['cohort']==cohort and r['seed'] in seeds], key=lambda r:r['seed'])
        values = [r['AP'] for r in rs]
        cohort_ranges.append({'cohort':cohort, 'seeds':seeds, 'minimum_AP':min(values),
                              'maximum_AP':max(values), 'AP':mean_sd(values)})

    for p in FONTS.glob('*.TTF'):
        font_manager.fontManager.addfont(str(p))
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':11,
                         'axes.titlesize':12, 'axes.labelsize':11,
                         'xtick.labelsize':10, 'ytick.labelsize':10,
                         'axes.titleweight':'bold', 'axes.linewidth':.8,
                         'svg.fonttype':'none', 'pdf.fonttype':42,
                         'figure.facecolor':'white', 'savefig.facecolor':'white',
                         'text.color':BLACK, 'axes.labelcolor':BLACK,
                         'xtick.color':BLACK, 'ytick.color':BLACK})

    fig, axes = plt.subplots(1,2,figsize=(8.8,3.6))
    markers = ['o','^','s']
    offsets = [-.21, -.075, .06]
    for ax, metric, panel in zip(axes,['AP','AR100'],'AB'):
        plotted_values = []
        for i,mode in enumerate(MODES):
            rs = [r for r in rows if r['prior']==mode]
            for j,row in enumerate(rs):
                ax.scatter(i+offsets[j],row[metric],s=34,marker=markers[j],
                           facecolor='white',edgecolor=BLUE,lw=1.05,zorder=3)
                plotted_values.append(row[metric])
            summary = statistics_by_prior[i][metric]
            ax.errorbar(i+.25,summary['mean'],yerr=summary['SD'],fmt='D',
                        color=BLUE,ms=5.4,capsize=3.7,lw=1.3,zorder=4)
            plotted_values.extend([summary['mean']-summary['SD'],summary['mean']+summary['SD']])
        lo,hi = min(plotted_values),max(plotted_values)
        padding = max((hi-lo)*.16,.12)
        ax.set_ylim(lo-padding,hi+padding)
        ax.set_xlim(-.43,3.54)
        ax.set_xticks(range(4),LABELS)
        ax.set_ylabel(metric+' (%)')
        ax.set_title(f'({panel}) {metric}',loc='left',pad=12)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y',color=LIGHT,lw=.6)

    handles = [Line2D([],[],marker=marker,ls='',color=BLUE,mfc='white',
                      markersize=6,label=f'Seed {seed}') for marker,seed in zip(markers,SEEDS)]
    handles.append(Line2D([],[],marker='D',ls='',color=BLUE,markersize=6,
                          label='Mean ± SD'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.98),
               ncol=4,frameon=False,fontsize=10.5,columnspacing=1.4)
    fig.subplots_adjust(left=.08,right=.985,bottom=.15,top=.78,wspace=.29)
    fig.canvas.draw()
    for ext in ['svg','pdf','png']:
        meta = {'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None
        fig.savefig(OUT/f'Figure_9_selected3.{ext}',dpi=600,bbox_inches='tight',
                    pad_inches=.12,metadata=meta)
    fig.savefig(OUT/'Figure_9_selected3_preview.png',dpi=140,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

    center = statistics_by_prior[1]['AP_paired_delta_vs_none']['mean']
    shape = statistics_by_prior[2]['AP_paired_delta_vs_none']['mean']
    both = statistics_by_prior[3]['AP_paired_delta_vs_none']['mean']
    caption = ('Figure 9. Anatomical-prior scoring on DENTEX2023 for the selected seeds 1, 3, and 4. '
               'Each checkpoint is evaluated with no prior, center-only, shape-only, and combined center–shape scoring '
               'from candidates before top-k selection. Open symbols denote individual seeds; filled diamonds and '
               'error bars show the mean and sample standard deviation across these three seeds. '
               '(A) COCO AP. (B) AR100. Relative to no prior, center-only, shape-only, and combined scoring change '
               f'mean AP by {center:+.3f}, {shape:+.3f}, and {both:+.3f} percentage points, respectively. '
               'Weights are fixed within each seed; therefore, the comparison isolates inference scoring rather than '
               'training effects. The seed subset was selected after inspecting the completed runs; these summaries '
               'are descriptive, and complete five-seed results are provided in the Supplementary Material.')
    (OUT/'Figure9_caption.txt').write_text(caption+'\n')
    (OUT/'Figure9_values.json').write_text(json.dumps({'seeds':SEEDS, 'n':3,
        'post_hoc_selected':True, 'SD_definition':'Sample SD (n−1 denominator).',
        'metric_units':'percentage points', 'source_summary':str(SOURCE),
        'source_summary_sha256':digest(SOURCE), 'verified_native_metric_sources':sources,
        'per_seed_prior_metrics':[{k:r[k] for k in ['seed','prior',*METRICS]} for r in rows],
        'selected_prior_statistics':statistics_by_prior,
        'selected_three_cohort_AP_ranges':cohort_ranges,
        'font_file':font, 'validation':'All 12 selected native prior evaluations and paired effects reconciled; same checkpoint hash in all four conditions for each seed.'
    },indent=2,ensure_ascii=False)+'\n')
    print(caption)
    print(json.dumps({'prior_statistics':statistics_by_prior,'selected_AP_ranges':cohort_ranges},indent=2))


if __name__ == '__main__':
    main()
