# Selected-configuration repetitions

The model configurations are copied from the completed October 8 experiments,
not reconstructed from a rounded manuscript table. No dataset or project source
files are changed. All runs start with the original pretrained backbone and no
previous training checkpoint is reused.

| Cohort | Configuration | Epochs | Fixed seeds |
|---|---|---:|---|
| Mobile | Full Mobile model; four prototypes/class, temperature 0.50 | 160 | 0, 1, 2, 3, 4 |
| DSLR | CMFA + DualOpt additional decay; rollback weight 0 | 160 | 0, 1, 2, 3, 4 |
| DENTEX / Pano | Original fixed-640 Factor + Prototype + SpatialShape configuration | 160 | 0, 1, 2, 3, 4 |

The primary three-seed set is fixed as 0/1/2 before launch. The stability analysis
includes all five scheduled seeds. Individual results and both fixed-set summaries
are saved; no best-three selection is made after viewing outcomes. Sample SD and
Student-t confidence intervals summarize training-seed variation, not an
independent clinical test or patient-level confidence interval. The configurations
were selected from previous validation ablations, so these repetitions are not
independent confirmation of state-of-the-art performance.

Two detached workers share task claims. GPU0 starts if available; GPU1 waits until
no unrelated compute process occupies it. The workers never stop other processes.
Training is not monitored interactively after successful startup. Failures and
changed inputs stop the affected worker rather than replace an unfavorable seed.

Each job directory under `runs/<configuration>/seed_<n>` saves full configuration,
native training logs, COCO metrics, real pre-top-k candidates and prediction exports.
Native stage-1 checkpoints may be needed temporarily for Pano/DSLR training and
are deleted only after training and exact EMA replay pass. A successful run retains
only `best.pth` as its training weight. Evaluator `.pth` payloads in evaluation
subdirectories contain COCO state, not additional model checkpoints.

`results_per_seed.csv` and `fixed_seed_mean_sd_ci.json` are updated automatically
after each successful job. Incomplete summaries remain explicitly marked incomplete.
The original papers and prior experiments are not modified.
