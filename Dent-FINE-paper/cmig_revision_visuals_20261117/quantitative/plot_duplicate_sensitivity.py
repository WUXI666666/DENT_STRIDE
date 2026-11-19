"""Plot all five saved seeds and their post-hoc validation sensitivity results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
DATA = json.loads((OUT / 'duplicate_sensitivity.json').read_text())
FONTS = Path('/home/wuxi/Dent-FINE-paper/materials/cmig_revision_2026-10-03/references/times_new_roman')
for path in FONTS.glob('*.TTF'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5,
                     'axes.titlesize': 9.5, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
colors = {'Pano': '#33658a', 'Mobile': '#257c70'}
markers = ['o', '^', 's', 'D', 'p']
fig, axes = plt.subplots(1, 2, figsize=(174/25.4, 88/25.4), sharey=True)
fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.35, wspace=.22)
for ax, cohort, panel in zip(axes, ['Pano', 'Mobile'], ['A', 'B']):
    records = [r for r in DATA['per_seed'] if r['cohort'] == cohort]
    color = colors[cohort]
    for r, offset in zip(records, np.linspace(-.08, .08, 5)):
        xx = np.array([0, 1]) + offset
        yy = [r['original_metrics_percent']['AP'], r['restricted_metrics_percent']['AP']]
        ax.plot(xx, yy, lw=.9, color=color, alpha=.6, zorder=2)
        ax.scatter(xx, yy, s=28, marker=markers[r['seed']],
                   facecolors=color if r['selected_for_reported_mean'] else 'white',
                   edgecolors=color, linewidths=.85, zorder=3)
    counts = DATA['cohorts'][cohort]
    summary = DATA['summaries'][cohort]['all_five']
    selected = DATA['summaries'][cohort]['reported_three']
    ax.set_title(f'({panel}) '+ ('Pano (DENTEX2023)' if cohort == 'Pano' else 'Mobile'),
                 loc='left', fontweight='bold', pad=23)
    ax.text(.5, 1.035, 'Mean ΔAP = '+
            f"{summary['delta_AP_pp']['mean']:+.3f} ± {summary['delta_AP_pp']['sample_sd']:.3f} pp",
            ha='center', va='bottom', transform=ax.transAxes, color=color, fontsize=8.6)
    ax.set_xlim(-.30, 1.30)
    ax.set_ylim(53.2, 55.9)
    ax.set_yticks([53.5, 54, 54.5, 55, 55.5])
    ax.set_xticks([0, 1], [f"Original split\n{counts['full_images']} images",
                          f"Exact duplicates excluded\n{counts['retained_images']} images"])
    ax.grid(axis='y', color='#e4e9ec', linewidth=.45)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#87939c')
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(length=2.3, width=.65, color='#87939c')
    fmt = lambda x: f"{x['mean']:.3f} ± {x['sample_sd']:.3f}"
    ax.text(.5, -.30, f"All five: {fmt(summary['original_AP'])} → {fmt(summary['restricted_AP'])}",
            transform=ax.transAxes, ha='center', fontsize=7.9)
    ax.text(.5, -.42, f"Reported three: {fmt(selected['original_AP'])} → {fmt(selected['restricted_AP'])}",
            transform=ax.transAxes, ha='center', fontsize=7.9)
axes[0].set_ylabel('Validation AP (%)')
handles = [Line2D([0], [0], marker=m, linestyle='none', color='#354b5d',
                  markerfacecolor='white', markeredgewidth=.8, markersize=4.4,
                  label=f'Seed {s}') for s, m in enumerate(markers)]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.045), ncol=5,
           frameon=False, fontsize=8, columnspacing=1.7, handletextpad=.4)
fig.text(.5, .023, 'Filled: three seeds used in the reported mean; open: remaining completed seeds.',
         ha='center', fontsize=8)
fig.savefig(OUT/'Figure_Validation_sensitivity.pdf', dpi=300)
fig.savefig(OUT/'Figure_Validation_sensitivity.svg', dpi=300)
fig.savefig(OUT/'Figure_Validation_sensitivity_preview.png', dpi=220)
plt.close(fig)
print('Saved Figure_Validation_sensitivity.{pdf,svg} and preview.')
