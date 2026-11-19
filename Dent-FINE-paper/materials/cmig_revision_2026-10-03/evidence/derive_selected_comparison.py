"""Derive final-recipe rows from the fixed, verified evidence snapshot."""
import csv
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
e = json.loads((OUT / 'controlled_evidence.json').read_text())
runs = list(csv.DictReader((OUT / 'all_completed_runs.csv').open(encoding='utf-8-sig')))
prior = list(csv.DictReader((OUT / 'spatial_shape_per_seed.csv').open(encoding='utf-8-sig')))
metrics = ['AP', 'AP50', 'AP75', 'AR100', 'Precision', 'Recall', 'F1',
           'matched_tooth_FDI_accuracy', 'localization_coverage', 'raw_duplicate_image_rate']
rows = []
comparisons = [
    ('Pano', 'DENT-STRIDE-Pano', 'pano_factor_prototype', 'pano_base', 'center+shape'),
    ('Mobile', 'DENT-STRIDE-Mobile', 'mobile_1024_full', 'mobile_1024_base', 'off'),
    ('DSLR', 'DENT-STRIDE-DSLR', 'dslr_cmfa_dualopt', 'dslr_base_adamw', 'off')]
for cohort, label, final, baseline, mode in comparisons:
    selected = sorted([r for r in prior if r['prior'] == 'both'], key=lambda r: int(r['seed'])) if cohort == 'Pano' else sorted([r for r in runs if r['configuration'] == final], key=lambda r: int(r['seed']))
    reference = sorted([r for r in runs if r['configuration'] == baseline], key=lambda r: int(r['seed']))
    assert len(selected) == len(reference) == 3
    row = {'cohort': cohort, 'model': label, 'configuration': final, 'prior': mode, 'reference': baseline}
    for key in metrics:
        values = [float(r[key]) for r in selected]
        sd = statistics.stdev(values)
        diffs = [float(a[key]) - float(b[key]) for a, b in zip(selected, reference)]
        dsd = statistics.stdev(diffs); dm = statistics.mean(diffs); dh = 4.302652729911275 * dsd / math.sqrt(3)
        row.update({key + '_mean': statistics.mean(values), key + '_SD': sd,
                    key + '_delta_mean': dm, key + '_delta_SD': dsd,
                    key + '_delta_CI95_low': dm - dh, key + '_delta_CI95_high': dm + dh})
    rows.append(row)
payload = {'snapshot_time_Asia_Shanghai': e['snapshot_time_Asia_Shanghai'], 'rows': rows,
           'note': 'Final-recipe rows preserve declared architecture choices. Do not substitute the numerically best post-hoc internal setting; candidate internal runs are single-seed.'}
(OUT / 'selected_controlled_comparison.json').write_text(json.dumps(payload, indent=2))
with (OUT / 'selected_controlled_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r['cohort'], r['AP_mean'], r['AP_SD'], r['AP_delta_mean'], r['AP_delta_CI95_low'], r['AP_delta_CI95_high'])
