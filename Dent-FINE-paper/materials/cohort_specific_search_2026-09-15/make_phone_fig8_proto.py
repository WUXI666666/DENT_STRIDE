#!/usr/bin/env python3
"""Render Figure 8A with the selected Phone A07 Proto Aux predictions."""
from pathlib import Path
import importlib.util
import json

SCRIPT = Path("/home/wuxi/Dent-FINE-paper/materials/scripts/make_final_multimodel_figure8.py")
PREDICTIONS = Path("/home/wuxi/Dent-FINE-paper/materials/protoaux_revision_assets_2026-09-18/predictions/coco_predictions.json")
OUTPUT = Path("/home/wuxi/Dent-FINE-paper/materials/protoaux_revision_assets_2026-09-18/figure8_phone")

spec = importlib.util.spec_from_file_location("figure8_builder", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.COHORTS["Phone"]["models"]["DENT-STRIDE"] = PREDICTIONS
OUTPUT.mkdir(parents=True, exist_ok=True)
metadata = module.render("Phone", module.COHORTS["Phone"], OUTPUT)
(OUTPUT / "provenance.json").write_text(json.dumps({"protocol": {"score_threshold": module.SCORE, "matching_iou": module.IOU}, **metadata}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2))
