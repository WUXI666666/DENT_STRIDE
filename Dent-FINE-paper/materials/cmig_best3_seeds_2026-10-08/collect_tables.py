#!/usr/bin/env python3
"""Collect verified results only after all 30 fresh jobs have finished.

Run ``python collect_tables.py --check-only`` for a read-only readiness report.
Run without that flag after completion to create a NEW manuscript_data directory.
Existing output is never replaced. This script does not edit any DOCX or figure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent
METRICS = ("AP", "AP50", "AP75", "AR100")
ENDPOINTS = ("Precision", "Recall", "F1", "matched_tooth_FDI_accuracy",
             "localization_coverage", "raw_duplicate_image_rate")
PRIOR_MODES = ("none", "center", "shape", "both")
MAX_NATIVE_REPLAY_TOLERANCE = 2e-6  # Fraction units; matches evaluate_native.py.
PRIMARY_CONFIGURATIONS = (
    "mobile_1024_full", "pano_historical_spatialshape", "dslr_cmfa_dualopt",
)
INTERNAL_CONFIGURATIONS = (
    "internal_mobile_cmfa_channel_only", "internal_mobile_cmfa_spatial_only",
    "internal_mobile_cmfa_no_gates", "internal_mobile_hooa_no_highpass",
    "internal_mobile_hooa_no_second_order", "internal_mobile_hooa_neither",
    "internal_mobile_factor_no_quadrant", "internal_mobile_factor_no_position",
    "internal_mobile_factor_no_dentition", "internal_mobile_factor_train_off",
    "internal_mobile_proto_off", "internal_mobile_proto_k1_tau0p2",
    "internal_mobile_proto_k2_tau0p2", "internal_mobile_proto_k4_tau0p1",
    "internal_mobile_proto_k4_tau0p5", "internal_dslr_rollback_only",
    "internal_dslr_extra_decay_only", "internal_mobile_background_neither",
    "internal_mobile_background_nwd_only", "internal_mobile_background_arch_only",
    "internal_dslr_cmfa_adamw",
)


class EvidenceError(ValueError):
    """Missing or inconsistent evidence must never be converted into zero."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"Missing file: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha(path):
    path = Path(path)
    require(path.is_file(), f"Missing file for hash check: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_matches(path, expected):
    require(isinstance(expected, str) and len(expected) == 64,
            f"Missing/invalid SHA256 for {path}")
    actual = sha(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return actual


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"Missing/non-numeric value: {label}")
    result = float(value)
    require(math.isfinite(result), f"Non-finite value: {label}")
    return result


def native_record(job, path, checkpoint_sha, *, mode_kind=None, mode=None):
    """Verify authoritative native evidence without loading a GPU or checkpoint."""
    record = read_json(path)
    for key, expected in (("job_id", job["id"]), ("configuration", job["configuration"]),
                          ("cohort", job["cohort"]), ("seed", job["seed"]),
                          ("checkpoint_sha256", checkpoint_sha),
                          ("config_sha256", job["config_sha256"])):
        require(record.get(key) == expected, f"{path}: inconsistent {key}")
    require(record.get("authoritative_native_evaluation") is True,
            f"{path}: not authoritative native evaluation")
    require(record.get("verified_best_checkpoint") is True,
            f"{path}: best checkpoint not verified")
    require(record.get("evaluated_ema") is True, f"{path}: EMA evaluation missing")
    require("FP32" in str(record.get("precision", "")), f"{path}: not FP32")
    require(record.get("epochs_completed") == job["epochs"] == 160,
            f"{path}: incomplete training budget")
    epoch = record.get("best_epoch_zero_based")
    require(isinstance(epoch, int) and 0 <= epoch < 160,
            f"{path}: invalid selected epoch")
    require(record.get("best_epoch_one_based") == epoch + 1,
            f"{path}: inconsistent epoch indexing")
    for key in ("params_total", "params_trainable"):
        require(isinstance(record.get(key), int) and record[key] > 0,
                f"{path}: missing parameter count {key}")
    require(record["params_total"] >= record["params_trainable"],
            f"{path}: impossible parameter counts")
    values = {}
    for metric in METRICS:
        fraction = number(record.get("metrics_fraction", {}).get(metric), f"{path}:{metric}")
        percent = number(record.get("metrics_percent", {}).get(metric), f"{path}:{metric}%")
        require(0 <= fraction <= 1 and math.isclose(percent, fraction * 100, abs_tol=1e-10),
                f"{path}: invalid/inconsistent {metric} units")
        values[metric] = percent
    if mode_kind is None:
        require(record.get("selection_metrics_agreement_checked") is True,
                f"{path}: selection/replay agreement was not checked")
        tolerance = number(record.get("strict_replay_tolerance_fraction"), f"{path}:tolerance")
        require(0 <= tolerance <= MAX_NATIVE_REPLAY_TOLERANCE,
                f"{path}: native replay tolerance was relaxed")
        for metric in METRICS:
            delta = number(record.get("native_replay_minus_logged_fraction", {}).get(metric),
                           f"{path}:{metric} replay delta")
            logged = number(record.get("logged_best_metrics_fraction", {}).get(metric),
                            f"{path}:{metric} logged value")
            fraction = record["metrics_fraction"][metric]
            require(abs(delta) <= tolerance and math.isclose(fraction - logged, delta, abs_tol=1e-12),
                    f"{path}: strict native replay mismatch for {metric}")
    elif mode_kind == "prior":
        require(record.get("prior_override") == mode, f"{path}: wrong prior override")
        flags = record.get("postprocessor_priors", {})
        expected = {"use_dental_spatial_prior": mode in ("center", "both"),
                    "use_dental_shape_prior": mode in ("shape", "both"),
                    "use_dental_view_prior": False, "use_dental_appearance_prior": False}
        for flag, value in expected.items():
            require(flags.get(flag) is value, f"{path}: unexpected {flag}")
    elif mode_kind == "factor":
        observed = record.get("factor_inference_override", record.get("factor_inference"))
        require(observed == mode, f"{path}: missing/wrong native factor override")
    else:
        raise EvidenceError(f"Unknown native replay kind: {mode_kind}")

    provenance_path = Path(record.get("provenance", ""))
    require(provenance_path.is_file(), f"{path}: native provenance missing")
    require(provenance_path.resolve().parent == Path(job["output_dir"]).resolve(),
            f"{path}: provenance is outside this run")
    provenance = read_json(provenance_path)
    for key, expected in (("checkpoint_sha256", checkpoint_sha),
                          ("config_sha256", job["config_sha256"])):
        require(provenance.get(key) == expected, f"{provenance_path}: inconsistent {key}")
    hash_matches(job["job_file"], provenance.get("job_file_sha256"))
    hash_matches(Path(job["output_dir"]) / "log.txt", provenance.get("training_log_sha256"))
    hash_matches(job["train_script"], provenance.get("native_train_script_sha256"))
    hash_matches(provenance.get("native_eval_payload", ""), provenance.get("native_eval_payload_sha256"))
    captured = provenance.get("native_capture", {})
    require(captured.get("params_total") == record["params_total"] and
            captured.get("params_trainable") == record["params_trainable"],
            f"{path}: captured parameter counts differ")
    names = record.get("metric_names", [])
    captured_values = captured.get("statistics", [])
    require(len(names) == len(captured_values), f"{path}: incomplete native statistics capture")
    captured_metrics = dict(zip(names, captured_values))
    for metric in METRICS:
        captured_value = number(captured_metrics.get(metric), f"{path}:captured {metric}")
        require(math.isclose(captured_value, record["metrics_fraction"][metric], abs_tol=1e-12),
                f"{path}: metric differs from native capture")
    return {"native": record, "values": values, "metric_path": str(path),
            "metric_sha256": sha(path), "provenance_path": str(provenance_path),
            "provenance_sha256": sha(provenance_path)}


def endpoint_values(summary, label):
    fixed = summary.get("fixed_threshold")
    require(isinstance(fixed, dict), f"{label}: fixed-threshold endpoints missing")
    result = {}
    for key in ENDPOINTS:
        require(key in fixed, f"{label}: endpoint {key} missing")
        # A genuinely undefined ratio stays null; it is never filled with zero.
        value = fixed[key]
        if value is None:
            result[key] = None
        else:
            fraction = number(value, f"{label}:{key}")
            require(0 <= fraction <= 1, f"{label}: invalid endpoint {key}")
            result[key] = fraction * 100
    return result


def verify_job(job):
    run = Path(job["output_dir"])
    require(not (run / "best.pth").is_symlink(), f"{job['id']}: symlink checkpoint refused")
    marker = read_json(run / "COMPLETED.json")
    require(marker.get("job") == job["id"], f"{job['id']}: wrong completion marker")
    require(job.get("fresh_initialization") is True, f"{job['id']}: not a fresh run")
    require(read_json(job["job_file"]) == job, f"{job['id']}: job JSON changed")
    hash_matches(job["config"], job["config_sha256"])
    hash_matches(run / "config.yml", job["config_sha256"])
    checkpoint_sha = hash_matches(run / "best.pth", marker.get("checkpoint_sha256"))
    summary_sha = hash_matches(run / "metrics_summary.json", marker.get("metrics_sha256"))
    native_sha = hash_matches(run / "native_coco_metrics.json", marker.get("native_metrics_sha256"))
    evidence = native_record(job, run / "native_coco_metrics.json", checkpoint_sha)
    summary = read_json(run / "metrics_summary.json")
    for key, expected in (("job_id", job["id"]), ("configuration", job["configuration"]),
                          ("seed", job["seed"]), ("checkpoint_sha256", checkpoint_sha),
                          ("config_sha256", job["config_sha256"])):
        require(summary.get(key) == expected, f"{job['id']}: summary {key} differs")
    require("native_coco_metrics.json" in str(summary.get("primary_metric_source", "")),
            f"{job['id']}: summary does not identify authoritative native metrics")
    for metric in METRICS:
        require(math.isclose(number(summary.get("metrics_percent", {}).get(metric), metric),
                             evidence["values"][metric], abs_tol=1e-10),
                f"{job['id']}: exported summary replaced native {metric}")
    export_delta = number(summary.get("export_minus_native_AP_points"), f"{job['id']}:export delta")
    evidence.update(job=job, summary=summary, endpoints=endpoint_values(summary, job["id"]),
                    checkpoint_sha256=checkpoint_sha, summary_sha256=summary_sha,
                    native_sha256=native_sha, export_minus_native_AP_points=export_delta)
    return evidence


def validate_manifest(manifest):
    jobs = manifest.get("jobs", [])
    require(len(jobs) == manifest.get("total_jobs") == 30, "Expected exactly 30 frozen jobs")
    require(Counter(job.get("phase") for job in jobs) ==
            {"primary": 9, "internal": 21}, "Wrong phase counts")
    require(manifest.get("historical_runs_reused") == 0, "Historical training reuse is not allowed")
    require(len({job["id"] for job in jobs}) == 30, "Duplicate job identifiers")
    expected = {(name, seed, "primary") for name in PRIMARY_CONFIGURATIONS for seed in (0, 1, 2)}
    expected |= {(name, 0, "internal") for name in INTERNAL_CONFIGURATIONS}
    actual = {(job["configuration"], job["seed"], job["phase"]) for job in jobs}
    require(actual == expected, "Frozen configuration/seed scope differs from the declared 30 jobs")
    for job in jobs:
        require(Path(job["output_dir"]).resolve() == ROOT / "runs" / job["configuration"] / f"seed_{job['seed']}",
                f"{job['id']}: output is not owned by this experiment root")
    return jobs


def readiness():
    manifest_path = ROOT / "execution_manifest.json"
    manifest = read_json(manifest_path)
    jobs = validate_manifest(manifest)
    verified, issues = {}, []
    marked = 0
    for job in jobs:
        if not (Path(job["output_dir"]) / "COMPLETED.json").is_file():
            issues.append({"job": job["id"], "kind": "not_completed"})
            continue
        marked += 1
        try:
            verified[(job["configuration"], job["seed"])] = verify_job(job)
        except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"job": job["id"], "kind": "invalid_completed_evidence", "error": str(error)})
    auxiliary = {"priors": [], "factors": []}
    # Readiness includes all twelve native same-weight prior evaluations and
    # both native factor replays, not just the presence of completion markers.
    for seed in (0, 1, 2):
        base = verified.get(("pano_historical_spatialshape", seed))
        if base is not None:
            for mode in PRIOR_MODES:
                path = Path(base["job"]["output_dir"]) / f"native_prior_{mode}_metrics.json"
                try:
                    record = native_record(base["job"], path, base["checkpoint_sha256"],
                                           mode_kind="prior", mode=mode)
                    record.update(job=base["job"], prior=mode)
                    auxiliary["priors"].append(record)
                except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
                    issues.append({"job": base["job"]["id"], "kind": "invalid_prior_replay",
                                   "mode": mode, "error": str(error)})
    for name, mode in (("mobile_1024_full", "off"), ("internal_mobile_factor_train_off", "on")):
        base = verified.get((name, 0))
        if base is not None:
            path = Path(base["job"]["output_dir"]) / f"native_factor_{mode}_metrics.json"
            try:
                record = native_record(base["job"], path, base["checkpoint_sha256"],
                                       mode_kind="factor", mode=mode)
                secondary_path = Path(base["job"]["output_dir"]) / f"replay_factor_{mode}" / "metrics.json"
                secondary = read_json(secondary_path)
                require(secondary.get("checkpoint_sha256") == base["checkpoint_sha256"],
                        f"{secondary_path}: wrong factor replay checkpoint")
                for metric in METRICS:
                    require(math.isclose(number(secondary.get("metrics_percent", {}).get(metric), metric),
                                         record["values"][metric], abs_tol=1e-10),
                            f"{secondary_path}: factor replay did not merge native {metric}")
                record.update(job=base["job"], factor_inference=mode,
                              endpoints=endpoint_values(secondary, str(secondary_path)),
                              secondary_metric_path=str(secondary_path), secondary_metric_sha256=sha(secondary_path))
                auxiliary["factors"].append(record)
            except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
                issues.append({"job": base["job"]["id"], "kind": "invalid_factor_replay",
                               "mode": mode, "error": str(error)})
    ready = (len(verified) == 30 and len(auxiliary["priors"]) == 12 and
             len(auxiliary["factors"]) == 2 and not issues)
    report = {"status": "ready" if ready else "not_ready", "jobs_total": 30,
              "completion_markers": marked, "verified_completed_jobs": len(verified),
              "not_completed_jobs": 30 - marked, "invalid_completed_jobs": marked - len(verified),
              "verified_historical_prior_replays": len(auxiliary["priors"]), "required_prior_replays": 12,
              "verified_native_factor_replays": len(auxiliary["factors"]), "required_factor_replays": 2,
              "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
              "outputs_written": False, "issues": issues}
    report["verified_evidence"] = [
        {"job_id": record["job"]["id"], "checkpoint_sha256": record["checkpoint_sha256"],
         "summary_sha256": record["summary_sha256"], "native_metrics_sha256": record["native_sha256"],
         "native_provenance_sha256": record["provenance_sha256"]}
        for _, record in sorted(verified.items())]
    report["verified_auxiliary_evidence"] = [
        {"job_id": record["job"]["id"], "metric_path": record["metric_path"],
         "metric_sha256": record["metric_sha256"], "provenance_sha256": record["provenance_sha256"],
         "secondary_metric_sha256": record.get("secondary_metric_sha256")}
        for records in auxiliary.values() for record in records]
    return report, manifest, verified, auxiliary


def statistics_row(values, prefix):
    require(bool(values), f"No values for {prefix}")
    clean = [number(value, prefix) for value in values]
    return {prefix + "_mean": statistics.mean(clean),
            prefix + "_SD": statistics.stdev(clean) if len(clean) > 1 else None}


def paired_statistics(values, references, prefix):
    require(len(values) == len(references) == 3, f"Expected three paired seeds for {prefix}")
    from scipy.stats import t
    differences = [a - b for a, b in zip(values, references)]
    result = statistics_row(differences, prefix)
    half = float(t.ppf(.975, df=2)) * result[prefix + "_SD"] / math.sqrt(3)
    result.update({prefix + "_CI95_low": result[prefix + "_mean"] - half,
                   prefix + "_CI95_high": result[prefix + "_mean"] + half})
    return result


def per_seed_row(record):
    job = record["job"]
    row = {"job_id": job["id"], "configuration": job["configuration"], "cohort": job["cohort"],
           "phase": job["phase"], "seed": job["seed"], **record["values"],
           "params_total": record["native"]["params_total"],
           "params_trainable": record["native"]["params_trainable"],
           "params_M": record["native"]["params_total"] / 1e6,
           "best_epoch_zero_based": record["native"]["best_epoch_zero_based"],
           "best_epoch_one_based": record["native"]["best_epoch_one_based"],
           "native_metric_path": record["metric_path"], "native_metric_sha256": record["metric_sha256"],
           "checkpoint_sha256": record["native"]["checkpoint_sha256"],
           "config_sha256": record["native"]["config_sha256"],
           "primary_metric_eligible": True, "independent_test": False}
    if "endpoints" in record:
        row.update(record["endpoints"])
    if "export_minus_native_AP_points" in record:
        row["export_minus_native_AP_points"] = record["export_minus_native_AP_points"]
    return row


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["configuration"]].append(row)
    output = []
    for name, group in sorted(groups.items()):
        group.sort(key=lambda row: row["seed"])
        for key in ("params_total", "params_trainable"):
            require(len({row[key] for row in group}) == 1, f"{name}: parameter count changed between seeds")
        row = {"configuration": name, "cohort": group[0]["cohort"], "phase": group[0]["phase"],
               "n": len(group), "seeds": [r["seed"] for r in group],
               "params_total": group[0]["params_total"], "params_trainable": group[0]["params_trainable"],
               "params_M": group[0]["params_M"], "dispersion": "sample SD; null for n=1"}
        for metric in METRICS:
            row.update(statistics_row([r[metric] for r in group], metric))
        for key in ENDPOINTS:
            values = [r.get(key) for r in group]
            row[key + "_n_defined"] = sum(value is not None for value in values)
            if all(value is not None for value in values):
                row.update(statistics_row(values, key))
            else:
                row.update({key + "_mean": None, key + "_SD": None})
        output.append(row)
    return output


def build_tables(verified, auxiliary):
    all_rows = [per_seed_row(record) for _, record in sorted(verified.items())]
    summaries = summarize(all_rows)
    primary = [row for row in summaries if row["phase"] == "primary"]
    require(len(primary) == 3 and all(row["n"] == 3 for row in primary), "Incomplete Table 3")
    table4 = []
    for name in PRIMARY_CONFIGURATIONS:
        group = [verified[(name, seed)] for seed in (0, 1, 2)]
        row = {"configuration": name, "cohort": group[0]["job"]["cohort"],
               "n": 3, "seeds": [0, 1, 2], "new_paired_control_available": False,
               "interpretation": "descriptive training-seed stability; no between-model paired effect or CI"}
        for metric in METRICS:
            values = [record["values"][metric] for record in group]
            for seed, value in enumerate(values):
                row[f"{metric}_seed{seed}"] = value
            row.update(statistics_row(values, metric))
            row.update({metric + "_min": min(values), metric + "_max": max(values),
                        metric + "_range": max(values) - min(values)})
        table4.append(row)
    prior_rows = []
    prior_records = {(record["job"]["seed"], record["prior"]): record for record in auxiliary["priors"]}
    for seed in (0, 1, 2):
        base_ap = prior_records[(seed, "none")]["values"]["AP"]
        for mode in PRIOR_MODES:
            row = per_seed_row(prior_records[(seed, mode)])
            row.update(prior=mode, delta_AP_vs_none=row["AP"] - base_ap,
                       selection_policy="historical native configured prior-ON selected weights",
                       replay_policy="center/shape override; view and appearance disabled")
            prior_rows.append(row)
    table6 = []
    for mode in PRIOR_MODES:
        group = sorted([row for row in prior_rows if row["prior"] == mode], key=lambda row: row["seed"])
        references = sorted([row for row in prior_rows if row["prior"] == "none"], key=lambda row: row["seed"])
        row = {"prior": mode, "configuration": "pano_historical_spatialshape", "n": 3, "seeds": [0, 1, 2],
               "selection_policy": group[0]["selection_policy"], "replay_policy": group[0]["replay_policy"]}
        for metric in METRICS:
            row.update(statistics_row([r[metric] for r in group], metric))
        row.update(paired_statistics([r["AP"] for r in group], [r["AP"] for r in references], "AP_delta_vs_none"))
        table6.append(row)
    table7 = []
    groups = (
        ("CMFA gates", "mobile_1024_full", (("Both gates", "mobile_1024_full"), ("Channel only", "internal_mobile_cmfa_channel_only"), ("Spatial only", "internal_mobile_cmfa_spatial_only"), ("Neither", "internal_mobile_cmfa_no_gates"))),
        ("HOOA branches", "mobile_1024_full", (("Full module", "mobile_1024_full"), ("Without high-pass", "internal_mobile_hooa_no_highpass"), ("Without second-order", "internal_mobile_hooa_no_second_order"), ("Without both", "internal_mobile_hooa_neither"))),
        ("Factor supervision", "internal_mobile_factor_train_off", (("All three factors; train/inference fusion off", "internal_mobile_factor_train_off"), ("Without quadrant", "internal_mobile_factor_no_quadrant"), ("Without position", "internal_mobile_factor_no_position"), ("Without dentition", "internal_mobile_factor_no_dentition"))),
        ("Prototype design", "mobile_1024_full", (("Four prototypes; tau=0.20", "mobile_1024_full"), ("No prototype", "internal_mobile_proto_off"), ("One prototype; tau=0.20", "internal_mobile_proto_k1_tau0p2"), ("Two prototypes; tau=0.20", "internal_mobile_proto_k2_tau0p2"), ("Four prototypes; tau=0.10", "internal_mobile_proto_k4_tau0p1"), ("Four prototypes; tau=0.50", "internal_mobile_proto_k4_tau0p5"))),
        ("DualOpt operations", "internal_dslr_cmfa_adamw", (("Neither; ordinary AdamW", "internal_dslr_cmfa_adamw"), ("Rollback only", "internal_dslr_rollback_only"), ("Extra decay only", "internal_dslr_extra_decay_only"), ("Both operations", "dslr_cmfa_dualopt"))),
        ("Mobile background losses", "mobile_1024_full", (("NWD + arch-position", "mobile_1024_full"), ("Neither", "internal_mobile_background_neither"), ("NWD only", "internal_mobile_background_nwd_only"), ("Arch-position only", "internal_mobile_background_arch_only"))),
    )
    for component, reference, settings in groups:
        reference_ap = verified[(reference, 0)]["values"]["AP"]
        for setting, name in settings:
            row = per_seed_row(verified[(name, 0)])
            row.update(component=component, setting=setting, reference=reference,
                       reference_AP=reference_ap, delta_AP_vs_reference=row["AP"] - reference_ap,
                       evidence_type="seed-0 fresh training")
            table7.append(row)
    factor_records = {(record["job"]["configuration"], record["factor_inference"]): record
                      for record in auxiliary["factors"]}
    factor_reference = "internal_mobile_factor_train_off"
    factor_base_ap = verified[(factor_reference, 0)]["values"]["AP"]
    settings = (("On / On", "mobile_1024_full", None), ("On / Off", "mobile_1024_full", "off"),
                ("Off / Off", factor_reference, None), ("Off / On", factor_reference, "on"))
    for setting, name, mode in settings:
        record = verified[(name, 0)] if mode is None else factor_records[(name, mode)]
        row = per_seed_row(record)
        row.update(component="Factor fusion", setting=setting, reference=factor_reference,
                   reference_AP=factor_base_ap, delta_AP_vs_reference=row["AP"] - factor_base_ap,
                   training_fusion=setting.split(" / ")[0], inference_fusion=setting.split(" / ")[1],
                   evidence_type="seed-0 fresh training" if mode is None else "same-weight native factor inference replay")
        table7.append(row)
    require(len(table7) == 30, "Incomplete Table 7 groups")
    require(set(INTERNAL_CONFIGURATIONS) <= {row["configuration"] for row in table7},
            "Table 7 omitted an internal training")
    return {"all_seed_results": all_rows, "all_configuration_summary": summaries,
            "primary_seed_results": [row for row in all_rows if row["phase"] == "primary"],
            "primary_mean_sd": primary,
            "best_configuration_seed_results": [row for row in all_rows if row["phase"] == "primary"],
            "three_seed_mean_sd": [row for row in summaries if row["n"] == 3],
            "spatial_shape_per_seed": prior_rows,
            "factor_fusion_2x2_seed0": [row for row in table7 if row["component"] == "Factor fusion"],
            "Table3_primary_three_seed": primary, "Table4_seed_stability": table4,
            "Table6_SpatialShape_three_seed": table6, "Table7_internal_seed0": table7}


def dependencies(report):
    return {"status": "verified_data_ready_for_manuscript_update", "created_UTC": datetime.now(timezone.utc).isoformat(),
            "source_manifest": report["manifest"], "source_manifest_sha256": report["manifest_sha256"],
            "all_30_jobs_verified": True, "main_configurations": 3, "main_training_runs": 9,
            "internal_seed0_training_runs": 21, "DOCX_edited": False, "figures_regenerated": False,
            "main_source": str(PAPER / "DENT_STRIDE_CMIG_main_final_20261006.docx"),
            "supplement_source": str(PAPER / "DENT_STRIDE_CMIG_supplement_final_20261006.docx"),
            "tables": {
                "Table3": {"data": "Table3_primary_three_seed.csv", "direct_figures": ["Figure 6 (redesign)", "Figure S5"], "note": "Only the three historical best complete configurations, seeds0/1/2; no new14-configuration module/resolution study."},
                "Table4": {"data": "Table4_seed_stability.csv", "direct_figures": ["Figure S12 (replace old between-model contrasts)"], "note": "Per-seed AP/AP50/AP75/AR100, sample SD, min/max/range. No fresh paired baseline exists; no between-model paired CI or causal claim."},
                "Table6": {"data": "Table6_SpatialShape_three_seed.csv", "per_seed": "spatial_shape_per_seed.csv", "direct_figures": ["Figure 9", "Figure 10"], "note": "Same three final Pano weights, native none/center/shape/both inference replay. Paired prior deltas and exploratory df2 intervals are valid within these same weights."},
                "Table7": {"data": "Table7_internal_seed0.csv", "direct_figures": ["Figure 11", "Figure S6"], "note": "Old20 internal designs plus new internal_dslr_cmfa_adamw seed0 anchor; new full-model seed0 anchors and two native factor replays. No dispersion bars."}},
            "historical_results_separation": {
                "old_14_configuration_results": str(PAPER / "materials/cmig_final_2026-10-06/evidence/primary_mean_sd.csv"),
                "policy": "May retain as explicitly historical, separate-batch evidence. Do not pool old/new seeds, relabel old runs as new controls, or compute paired treatment effects using old baselines."},
            "additional_dependencies": [
                "Table5: update new complete-model rows; label retained older comparator batches.",
                "Table8/resolution and old Figure6 module panels: historical only; not rerun in this scope.",
                "Table9: new complete-model endpoints; older baselines must be explicitly separate.",
                "Table10: new source complete-model checkpoints need fresh transfer evaluation if updated; no automatic new baseline-gain claim.",
                "Table11: unified parameter counts and checkpoint/config provenance.",
                "Figure7, Figure12, Figures13-15, FiguresS7-S11: refresh affected full-model sources; distinguish older baseline/external outputs.",
                "Figure8 feature-only/identity-only comparison: historical only unless separately authorized."],
            "prose_to_update": ["Abstract", "Results", "Discussion", "Conclusions", "table/figure captions", "supplementary tables and narrative", "hard-coded numeric prose in existing DOCX builders"],
            "limitations": ["Internal validation on unchanged splits; not an independent test set.",
                            "Known content overlap retained and disclosed.",
                            "Sample SD for three best configurations. Paired Student-t df2 intervals only for same-weight Pano prior changes, unadjusted for multiplicity.",
                            "No baseline3-seed training, no14-group rerun, no640-resolution rerun in this scope.",
                            "No seed or post-hoc internal-setting cherry-picking.",
                            "Primary AP is authoritative native FP32; export rounding diagnostics cannot replace it."],
            "handoff": "Use actual verified results to create a new DOCX/figure package after consistency review. Preserve October6 originals; no automatic DOCX edits are performed."}


def write_csv(path, rows):
    require(bool(rows), f"Refusing an empty output table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def publish(report, tables):
    output = ROOT / "manuscript_data"
    require(not output.exists(), f"Output already exists; refusing to overwrite: {output}")
    staging = Path(tempfile.mkdtemp(prefix=".manuscript_data_staging_", dir=ROOT))
    try:
        for name, rows in tables.items():
            write_csv(staging / (name + ".csv"), rows)
            write_json(staging / (name + ".json"), rows)
        write_json(staging / "all_per_seed_and_mean.json", tables)
        write_json(staging / "update_dependencies.json", dependencies(report))
        write_json(staging / "evidence_validation.json", report)
        # Directory publication is atomic, and a previously published nonempty
        # directory is never replaced. No existing individual files are edited.
        require(not output.exists(), f"Output appeared while collecting: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Read-only completeness/integrity check; never write outputs")
    args = parser.parse_args()
    try:
        report, _, verified, auxiliary = readiness()
        if report["status"] != "ready":
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            return 2
        if args.check_only:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        tables = build_tables(verified, auxiliary)
        # Revalidate the complete snapshot before publishing. Worker completion
        # does not authorize mutation of already accepted source evidence.
        final_report, _, _, _ = readiness()
        require(final_report == report, "Evidence changed during collection; no outputs published")
        output = publish(report, tables)
        print(json.dumps({**report, "outputs_written": True, "output_directory": str(output),
                          "DOCX_edited": False}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"status": "not_ready", "outputs_written": False, "error": str(error)},
                         ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
