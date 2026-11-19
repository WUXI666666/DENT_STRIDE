#!/usr/bin/env python3
"""Publication figures from a frozen, inspectable controlled-experiment snapshot.

No CUDA, model inference, dataset write, or training-state mutation is used.
Run with the existing deimv2 Python. First execution captures currently completed
metrics and full 160-epoch logs; subsequent executions reproduce that snapshot.
"""
from __future__ import annotations
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_controlled_2026-09-29')
FONT_DIRECTORY = HERE.parents[1] / 'references/times_new_roman'
if FONT_DIRECTORY.is_dir():
    for font_file in FONT_DIRECTORY.rglob('*'):
        if font_file.suffix.lower() in {'.ttf', '.otf'}:
            fm.fontManager.addfont(str(font_file))
BLUE, ORANGE, GREY, BLACK = '#315B7B', '#A46635', '#8D969D', '#263238'
SEED_MARKERS = ['o', 's', '^']
SEED_OFFSETS = [-0.10, 0.0, 0.10]
FAMILIES = {x.name for x in fm.fontManager.ttflist}
FONT = 'Times New Roman' if 'Times New Roman' in FAMILIES else 'Liberation Serif'
plt.rcParams.update({
    'font.family': FONT, 'font.size': 14, 'axes.labelsize': 14,
    'axes.titlesize': 15, 'axes.titleweight': 'normal', 'xtick.labelsize': 13,
    'ytick.labelsize': 13, 'legend.fontsize': 13, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': BLACK, 'axes.labelcolor': BLACK,
    'text.color': BLACK, 'xtick.color': BLACK, 'ytick.color': BLACK,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})

PRIMARY_GROUPS = [
    ('A', 'Mobile · 1024', ['mobile_1024_base', 'mobile_1024_features',
                           'mobile_1024_identity', 'mobile_1024_full'],
     ['Base', 'Features', 'Identity', 'Full'], (52.7, 56.3)),
    ('B', 'Mobile · 640', ['mobile_640_base', 'mobile_640_full'],
     ['Base', 'Full'], (52.7, 56.3)),
    ('C', 'Pano · 640', ['pano_base', 'pano_factor', 'pano_prototype',
                        'pano_factor_prototype'],
     ['Base', 'Factor', 'Prototype', 'Both'], (52.1, 54.2)),
    ('D', 'DSLR', ['dslr_base_adamw', 'dslr_cmfa_adamw',
                   'dslr_base_dualopt', 'dslr_cmfa_dualopt'],
     ['Base\nAdamW', 'CMFA\nAdamW', 'Base\nDualOpt', 'CMFA\nDualOpt'], (71.65, 73.6)),
]
LEARNING_GROUPS = [
    ('A', 'Mobile', 'mobile_1024_base', 'mobile_1024_full'),
    ('B', 'Pano', 'pano_base', 'pano_factor_prototype'),
    ('C', 'DSLR', 'dslr_base_adamw', 'dslr_cmfa_dualopt'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture():
    manifest = json.loads((ROOT / 'execution_manifest.json').read_text())
    rows = list(csv.DictReader((ROOT / 'results_summary.csv').open()))
    completed_ids = {(r['configuration'], int(r['seed'])) for r in rows}
    data = {'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'source_root': str(ROOT), 'font': FONT, 'completed_count': len(rows),
            'total_jobs': len(manifest['jobs']), 'metrics': {}, 'priors': {},
            'factor_replays': {}, 'curves': {}, 'sources': []}

    def read_json(path):
        data['sources'].append({'path': str(path), 'sha256': digest(path)})
        return json.loads(path.read_text())

    for job in manifest['jobs']:
        cfg, seed = job['configuration'], job['seed']
        path = ROOT / 'runs' / cfg / f'seed_{seed}' / 'metrics_summary.json'
        if (cfg, seed) in completed_ids:
            value = read_json(path)
            for metric in ['AP', 'AP50', 'AP75', 'AR100']:
                summary_value = float(next(r[metric] for r in rows
                    if r['configuration'] == cfg and int(r['seed']) == seed))
                assert abs(summary_value - value['metrics_percent'][metric]) < 1e-9
            data['metrics'][f'{cfg}/{seed}'] = value

    for seed in range(3):
        for prior, directory in [('off', 'evaluation'), ('center', 'replay_prior_center'),
                                 ('shape', 'replay_prior_shape'), ('both', 'replay_prior_both')]:
            path = ROOT / 'runs/pano_factor_prototype' / f'seed_{seed}' / directory / 'metrics.json'
            if path.is_file():
                data['priors'][f'{prior}/{seed}'] = read_json(path)
    for cfg, directory in [('mobile_1024_full', 'replay_factor_off'),
                           ('internal_mobile_factor_train_off', 'replay_factor_on')]:
        path = ROOT / 'runs' / cfg / 'seed_0' / directory / 'metrics.json'
        if path.is_file():
            data['factor_replays'][f'{cfg}/{directory}'] = read_json(path)

    for _, _, base, full in LEARNING_GROUPS:
        for cfg in [base, full]:
            for seed in range(3):
                path = ROOT / 'runs' / cfg / f'seed_{seed}' / 'log.txt'
                lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
                assert [x['epoch'] for x in lines] == list(range(160)), str(path)
                data['sources'].append({'path': str(path), 'sha256': digest(path)})
                data['curves'][f'{cfg}/{seed}'] = [100.0 * x['test_coco_eval_bbox'][0] for x in lines]
    (HERE / 'source_snapshot.json').write_text(json.dumps(data, indent=2))
    return data


def style(ax, grid=True):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=0.7, length=3)
    if grid:
        ax.grid(axis='y', color='#E7EAEC', lw=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(HERE / f'{name}.{extension}', dpi=600,
                    bbox_inches='tight', pad_inches=0.13)
    # A bounded preview lets visual checks inspect text at normal figure size.
    fig.savefig(HERE / f'{name}_preview.png', dpi=150,
                bbox_inches='tight', pad_inches=0.13)
    plt.close(fig)


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def get_ap(data, cfg, seed=0):
    entry = data['metrics'].get(f'{cfg}/{seed}')
    return None if entry is None else entry['metrics_percent']['AP']


def primary(data):
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.1))
    rows = []
    for ax, (letter, title, configs, labels, limits) in zip(axs.flat, PRIMARY_GROUPS):
        values = np.array([[get_ap(data, cfg, s) for s in range(3)] for cfg in configs])
        assert values.shape == (len(configs), 3) and np.isfinite(values).all()
        for seed in range(3):
            # Only baseline-to-full endpoints are joined, preserving seed pairing.
            ax.plot([SEED_OFFSETS[seed], len(configs)-1+SEED_OFFSETS[seed]],
                    [values[0, seed], values[-1, seed]], color=GREY, lw=.7,
                    alpha=.32, zorder=1)
        for j, cfg in enumerate(configs):
            mean, sd = values[j].mean(), values[j].std(ddof=1)
            color = ORANGE if j == len(configs)-1 else BLUE
            ax.errorbar(j, mean, yerr=sd, fmt='D', color=color, mfc=color, mec='white',
                        mew=.8, ms=6.1, capsize=5, elinewidth=1.3, zorder=5)
            for seed in range(3):
                ax.scatter(j+SEED_OFFSETS[seed], values[j, seed], s=29,
                           marker=SEED_MARKERS[seed], facecolors='white',
                           edgecolors=BLACK, linewidths=.75, zorder=4)
                rows.append({'panel': letter, 'cohort': title, 'configuration': cfg,
                             'label': labels[j].replace('\n', ' '), 'seed': seed,
                             'AP': values[j, seed], 'mean_AP': mean, 'SD_AP': sd})
        ax.set_title(f'({letter}) {title}', loc='left', pad=10)
        ax.set_xticks(range(len(configs)), labels)
        ax.set_xlim(-.4, len(configs)-.6)
        ax.set_ylim(limits)
        ax.set_ylabel('COCO AP (%)'); style(ax)
    handles = [Line2D([], [], marker=m, linestyle='None', markersize=5.3,
                      markerfacecolor='white', markeredgecolor=BLACK, label=f'Seed {s}')
               for s, m in enumerate(SEED_MARKERS)]
    handles.append(Line2D([], [], marker='D', color=BLUE, linestyle='None',
                          markersize=6, label='Mean ± SD'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.004),
               ncol=4, frameon=False, columnspacing=2)
    fig.subplots_adjust(left=.075, right=.985, bottom=.085, top=.905,
                        wspace=.26, hspace=.45)
    write_csv('primary_three_seed_values.csv', rows)
    save(fig, 'controlled_primary_three_seed_AP')


def internal(data):
    full = 'mobile_1024_full'
    groups = [
        ('A', 'CMFA gates', full,
         [('Both', full), ('Channel only', 'internal_mobile_cmfa_channel_only'),
          ('Spatial only', 'internal_mobile_cmfa_spatial_only'),
          ('Neither', 'internal_mobile_cmfa_no_gates')]),
        ('B', 'HOOA branches', full,
         [('Full', full), ('No high-pass', 'internal_mobile_hooa_no_highpass'),
          ('No second-order', 'internal_mobile_hooa_no_second_order'),
          ('Neither', 'internal_mobile_hooa_neither')]),
        ('C', 'Factor supervision', 'internal_mobile_factor_train_off',
         [('All factors', 'internal_mobile_factor_train_off'),
          ('No quadrant', 'internal_mobile_factor_no_quadrant'),
          ('No position', 'internal_mobile_factor_no_position'),
          ('No dentition', 'internal_mobile_factor_no_dentition')]),
        ('D', 'Prototype number', full,
         [('No prototype', 'internal_mobile_proto_off'),
          ('K = 1', 'internal_mobile_proto_k1_tau0p2'),
          ('K = 2', 'internal_mobile_proto_k2_tau0p2'), ('K = 4', full)]),
        ('E', 'Prototype temperature', full,
         [('τ = 0.10', 'internal_mobile_proto_k4_tau0p1'), ('τ = 0.20', full),
          ('τ = 0.50', 'internal_mobile_proto_k4_tau0p5')]),
        ('F', 'DualOpt operations', 'dslr_cmfa_adamw',
         [('Neither', 'dslr_cmfa_adamw'), ('Rollback only', 'internal_dslr_rollback_only'),
          ('Decay only', 'internal_dslr_extra_decay_only'), ('Both', 'dslr_cmfa_dualopt')]),
        ('G', 'Mobile background losses', full,
         [('NWD + arch', full), ('NWD only', 'internal_mobile_background_nwd_only'),
          ('Arch only', 'internal_mobile_background_arch_only'),
          ('Neither', 'internal_mobile_background_neither')]),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(14.3, 6.6))
    rows = []
    limits = (-.95, 2.3)
    for ax, (letter, title, ref, variants) in zip(axs.flat, groups):
        baseline = get_ap(data, ref)
        for i, (label, cfg) in enumerate(variants):
            value = get_ap(data, cfg)
            delta = None if value is None else value-baseline
            if delta is not None:
                color = BLUE if delta >= 0 else ORANGE
                ax.hlines(i, 0, delta, color=color, lw=1.3)
                ax.scatter(delta, i, s=32, c=color, marker='o', zorder=3)
                ax.text(2.2, i, f'{delta:+.3f}', ha='right', va='center', fontsize=10)
            else:
                ax.text(2.2, i, '—', ha='right', va='center', fontsize=11)
            rows.append({'panel': letter, 'group': title, 'variant': label,
                         'configuration': cfg, 'reference_configuration': ref,
                         'reference_AP': baseline, 'AP': value, 'delta_AP': delta,
                         'seed': 0, 'status': 'Available' if value is not None else '—'})
        ax.axvline(0, color=BLACK, lw=.8, zorder=1)
        ax.set_yticks(range(len(variants)), [label for label, _ in variants])
        ax.set_ylim(len(variants)-.55, -.45); ax.set_xlim(limits)
        ax.set_xticks([-.5, 0, 1])
        ax.set_xlabel('ΔAP (percentage points)', fontsize=11)
        ax.set_title(f'({letter}) {title}', loc='left', pad=10, fontsize=12)
        style(ax, False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(labelsize=10)
    # Full 2 × 2 fusion definition, calculated with the matched training state.
    ax = axs.flat[-1]
    reference = get_ap(data, 'internal_mobile_factor_train_off')
    fusion = [
        ('Off / off', reference),
        ('Off / on', data['factor_replays']['internal_mobile_factor_train_off/replay_factor_on']['metrics_percent']['AP']),
        ('On / off', data['factor_replays']['mobile_1024_full/replay_factor_off']['metrics_percent']['AP']),
        ('On / on', get_ap(data, full)),
    ]
    for i, (label, value) in enumerate(fusion):
        delta=value-reference; color=BLUE if delta>=0 else ORANGE
        ax.hlines(i,0,delta,color=color,lw=1.3);ax.scatter(delta,i,s=32,c=color,zorder=3)
        ax.text(2.2,i,f'{delta:+.3f}',ha='right',va='center',fontsize=10)
        rows.append({'panel':'H','group':'Factor fusion (train / inference)',
                     'variant':label,'configuration':'Matched training state + native inference replay',
                     'reference_configuration':'internal_mobile_factor_train_off',
                     'reference_AP':reference,'AP':value,'delta_AP':delta,'seed':0,'status':'Available'})
    ax.set_title('(H) Factor fusion',loc='left',pad=10,fontsize=12)
    ax.set_yticks(range(4),[x[0] for x in fusion]);ax.set_ylim(3.45,-.45)
    ax.set_xlim(limits);ax.set_xticks([-.5,0,1]);ax.set_xlabel('ΔAP (percentage points)',fontsize=11)
    ax.axvline(0,color=BLACK,lw=.8);style(ax,False);ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=.1,right=.992,bottom=.10,top=.92,wspace=.88,hspace=.68)
    write_csv('internal_ablation_values.csv',rows)
    save(fig,'controlled_internal_ablation_AP_deltas')
    # Two full-width four-panel versions preserve label size in portrait Word.
    for part, letters in [(1, ['A', 'C', 'D', 'F']), (2, ['B', 'E', 'G', 'H'])]:
        fig_part, ax_part = plt.subplots(2, 2, figsize=(10.6, 6.8))
        for local_index, (ax_new, letter) in enumerate(zip(ax_part.flat, letters)):
            panel_rows = [row for row in rows if row['panel'] == letter]
            for i, row in enumerate(panel_rows):
                delta = row['delta_AP']
                if delta is not None:
                    color = BLUE if delta >= 0 else ORANGE
                    ax_new.hlines(i, 0, delta, color=color, lw=1.3)
                    ax_new.scatter(delta, i, s=32, c=color, zorder=3)
                    ax_new.text(2.2, i, f'{delta:+.3f}', ha='right', va='center', fontsize=13)
                else:
                    ax_new.text(2.2, i, '—', ha='right', va='center', fontsize=14)
            ax_new.set_yticks(range(len(panel_rows)), [row['variant'] for row in panel_rows])
            ax_new.set_ylim(len(panel_rows)-.55, -.45);ax_new.set_xlim(limits)
            ax_new.set_xticks([-.5, 0, .5, 1]);ax_new.axvline(0, color=BLACK, lw=.8)
            ax_new.set_xlabel('ΔAP (percentage points)')
            title = panel_rows[0]['group'].replace(' (train / inference)', '')
            ax_new.set_title(f'({chr(65+local_index)}) {title}', loc='left', pad=10)
            style(ax_new, False);ax_new.spines['left'].set_visible(False)
            ax_new.tick_params(axis='y',length=0)
        fig_part.subplots_adjust(left=.20,right=.985,bottom=.11,top=.925,wspace=.75,hspace=.6)
        save(fig_part,f'controlled_internal_ablation_AP_deltas_part{part}')


def spatial(data):
    priors=['off','center','shape','both'];labels=['None','Center','Shape','Center + shape']
    fig, axs=plt.subplots(1,2,figsize=(10.4,3.8))
    rows=[]
    for ax,metric,letter in zip(axs,['AP','AR100'],['A','B']):
        all_values=np.array([[data['priors'][f'{p}/{s}']['metrics_percent'][metric]
                              for s in range(3)] for p in priors])
        for s in range(3):
            ax.plot(np.arange(4)+SEED_OFFSETS[s],all_values[:,s],
                    color=GREY,lw=.7,alpha=.5,zorder=1)
        for j,p in enumerate(priors):
            values=all_values[j];mean=values.mean();sd=values.std(ddof=1)
            color=ORANGE if p=='both' else BLUE
            ax.errorbar(j,mean,yerr=sd,fmt='D',color=color,ms=6,mec='white',mew=.8,
                        capsize=5,elinewidth=1.3,zorder=5)
            for s in range(3):
                ax.scatter(j+SEED_OFFSETS[s],values[s],s=29,marker=SEED_MARKERS[s],
                           facecolors='white',edgecolors=BLACK,linewidths=.75,zorder=4)
                rows.append({'prior':p,'seed':s,'metric':metric,'value':values[s],
                             'mean':mean,'SD':sd,
                             'paired_delta_vs_off':values[s]-all_values[0,s]})
        ax.set_xticks(range(4),labels);ax.set_xlim(-.4,3.4)
        ax.set_ylabel('COCO AP (%)' if metric=='AP' else 'AR100 (%)')
        ax.set_title(f'({letter}) '+('Detection accuracy' if metric=='AP' else 'Detection recall'),loc='left',pad=10)
        style(ax)
    axs[0].set_ylim(52.35,54.65);axs[1].set_ylim(65.4,71.1)
    handles=[Line2D([],[],marker=m,linestyle='None',markersize=5.3,
                    markerfacecolor='white',markeredgecolor=BLACK,label=f'Seed {s}')
             for s,m in enumerate(SEED_MARKERS)]
    handles.append(Line2D([],[],marker='D',color=BLUE,linestyle='None',markersize=6,label='Mean ± SD'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.05),ncol=4,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.81,wspace=.23)
    write_csv('spatialshape_same_weight_values.csv',rows)
    save(fig,'controlled_spatialshape_same_weight')


def learning(data):
    fig,axs=plt.subplots(1,3,figsize=(11.6,3.6),sharey=True)
    rows=[]
    for ax,(letter,title,base,full) in zip(axs,LEARNING_GROUPS):
        for cfg,label,color,ls in [(base,'Base',BLUE,'--'),(full,'Full',ORANGE,'-')]:
            values=np.array([data['curves'][f'{cfg}/{s}'] for s in range(3)])
            means=values.mean(axis=0);sds=values.std(axis=0,ddof=1);epochs=np.arange(1,161)
            ax.fill_between(epochs,means-sds,means+sds,color=color,alpha=.14,lw=0)
            ax.plot(epochs,means,color=color,lw=1.5,ls=ls,label=label)
            for s in range(3):
                rows.extend({'cohort':title,'configuration':cfg,'seed':s,'epoch':int(e),
                             'validation_AP':float(v),'mean_AP':float(m),'SD_AP':float(sd)}
                            for e,v,m,sd in zip(epochs,values[s],means,sds))
        ax.set_title(f'({letter}) {title}',loc='left',pad=9)
        ax.set_xlabel('Epoch');ax.set_xlim(1,160);ax.set_xticks([1,40,80,120,160])
        ax.set_ylim(0,80);style(ax)
    axs[0].set_ylabel('Validation COCO AP (%)')
    axs[0].legend(frameon=False,loc='lower right')
    fig.subplots_adjust(left=.065,right=.99,bottom=.15,top=.87,wspace=.12)
    write_csv('validation_AP_learning_curves.csv',rows)
    save(fig,'controlled_three_seed_learning_curves')


def captions(data):
    caption_text = '''controlled_primary_three_seed_AP
Controlled module and resolution comparisons across three acquisition domains. (A) Mobile at 1024 × 1024: controlled base, feature processing (CMFA + HOOA), structured identity (Factor + Prototype), and their combination. (B) Mobile base and combined model at 640 × 640. (C) Pano auxiliary-supervision comparisons at 640 × 640, with factor inference fusion and SpatialShape disabled. (D) DSLR CMFA × optimizer comparison. Open symbols show the three actual seeds; filled diamonds and error bars show the mean and sample standard deviation (n = 3). Thin lines connect corresponding base and combined-model seeds. Vertical ranges differ by domain; panels A and B share the same range. All points are best-validation-checkpoint FP32 COCO replay estimates.

controlled_internal_ablation_AP_deltas
Internal module ablations on seed 0. Differences are expressed in AP percentage points relative to the indicated intact setting for each panel: (A, B, D, E, G) the Mobile combined model; (C) all factor supervision with training and inference fusion disabled; (F) DSLR CMFA with ordinary AdamW; and (H) factor fusion disabled during both training and inference. Positive and negative values are shown without suppressing adverse outcomes. In H, labels denote training/inference fusion. The NWD and arch-position losses remain fixed except in G. An em dash denotes an unavailable completed result and has no plotted numeric point. These single-seed experiments describe configuration sensitivity and do not establish statistical superiority.

controlled_internal_ablation_AP_deltas_part1
Internal module sensitivity on seed 0. (A) CMFA gates, relative to both gates enabled in the Mobile combined model. (B) Factor supervision, relative to all three factor losses with factor fusion disabled during both training and inference. (C) Prototype number at temperature 0.20, relative to four prototypes per class. (D) DualOpt operations, relative to DSLR CMFA with ordinary AdamW and its unchanged base weight decay. Positive values mean that the displayed alternative exceeds its specified reference; they are not gains attributable to the intact module. An em dash denotes an unavailable completed result and has no numeric mark. These single-seed results describe sensitivity without statistical superiority claims.

controlled_internal_ablation_AP_deltas_part2
Internal branch, temperature, and background-loss sensitivity on seed 0. (A) HOOA branches, relative to the intact Mobile combined model. (B) Prototype temperature with four prototypes per class, relative to temperature 0.20. (C) NWD and arch-position losses, relative to their joint use. (D) Factor training/inference fusion settings, relative to fusion disabled in both phases. In D, the labels indicate training/inference fusion; each inference comparison uses the same frozen weight within its training state. Positive values mean that the displayed alternative exceeds its specified reference. An em dash denotes an unavailable completed result.

controlled_spatialshape_same_weight
Same-weight Pano anatomical-prior comparison across seeds 0, 1, and 2. Each frozen Factor + Prototype checkpoint was replayed with no prior, center only, shape only, or the combined prior; class scores were recomputed from all native candidates before top-k selection. (A) COCO AP. (B) AR100. Open symbols show seed-specific values; filled diamonds and error bars show mean ± sample standard deviation. Lines connect the same checkpoint across the four prior settings. Factor inference fusion is disabled throughout.

controlled_three_seed_learning_curves
Validation learning curves for controlled base and combined-model configurations on Mobile, Pano, and DSLR. Lines show the mean validation COCO AP at each actual epoch and bands show the sample standard deviation over seeds 0, 1, and 2. All 160 recorded epochs are displayed without smoothing. Pano curves exclude anatomical priors and factor inference fusion. These are online training-validation records; endpoint tables use FP32 frozen-checkpoint replay.
'''
    caption_path=HERE/'CAPTIONS.txt'
    if caption_path.is_file():
        existing=caption_path.read_text()
        extra_marker='\ncontrolled_prediction_curves\n'
        if extra_marker in existing:
            caption_text+=existing[existing.index(extra_marker):]
    caption_path.write_text(caption_text)
    meta={'snapshot_time':data['captured_at'],'completed_count':data['completed_count'],
          'total_jobs':data['total_jobs'],'font_requested':'Times New Roman',
          'font_used':FONT,'font_note':('Times New Roman installed' if FONT=='Times New Roman'
              else 'Times New Roman is unavailable in the environment; Liberation Serif is the compatible serif fallback.'),
          'exports':['Editable-text SVG','Vector PDF with embedded fonts','600-dpi PNG'],
          'interval':'Sample standard deviation, ddof=1; not a confidence interval',
          'provenance':'source_snapshot.json records all source hashes and exact plotted inputs',
          'missing_marker':'Em dash; missing observations are not zero'}
    (HERE/'ASSET_METADATA.json').write_text(json.dumps(meta,indent=2))


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    snapshot=HERE/'source_snapshot.json'
    data=json.loads(snapshot.read_text()) if snapshot.is_file() else capture()
    primary(data);internal(data);spatial(data);learning(data);captions(data)
    print(json.dumps({'directory':str(HERE),'font':FONT,
                      'completed_snapshot':data['completed_count'],'figures':4}))


if __name__ == '__main__':
    main()
