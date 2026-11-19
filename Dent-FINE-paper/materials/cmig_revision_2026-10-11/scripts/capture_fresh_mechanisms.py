"""Run existing passive capture hooks with the newly trained Mobile EMA weights."""
from pathlib import Path
import importlib.util
import sys

PAPER=Path('/home/wuxi/Dent-FINE-paper')
OLD=PAPER/'materials/cmig_revision_2026-10-03'
ROOT=PAPER/'materials/cmig_revision_2026-10-11'
TRAIN=PAPER/'materials/cmig_best3_seeds_2026-10-08'
spec=importlib.util.spec_from_file_location('fresh_capture',OLD/'scripts/export_mechanism_evidence.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=ROOT
module.TRAIN=TRAIN
module.main()
spec=importlib.util.spec_from_file_location('fresh_mechanism_plot',OLD/'scripts/plot_mechanism_evidence.py')
plotter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)
plotter.ROOT=ROOT
plotter.DEST=ROOT/'figures/mechanism'
plotter.tensor_figures()
plotter.prototypes()
plotter.factor_confusion()
