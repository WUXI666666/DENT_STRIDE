#!/usr/bin/env python3
"""Re-render Figure 8 panels after removing the unmatched RT-DETR-l comparator."""
from pathlib import Path
import importlib.util
import json

SCRIPT = Path("/home/wuxi/Dent-FINE-paper/materials/scripts/make_final_multimodel_figure8.py")
PREDICTIONS = Path("/home/wuxi/Dent-FINE-paper/materials/protoaux_revision_assets_2026-09-18/predictions/coco_predictions.json")
OUTPUT = Path("/home/wuxi/Dent-FINE-paper/materials/protoaux_revision_assets_2026-09-18/figure8_no_rtdetr_l")

spec = importlib.util.spec_from_file_location("figure8_builder_no_rtdetr", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

for cohort in module.COHORTS.values():
    cohort["models"].pop("RT-DETR-l", None)
module.COHORTS["Phone"]["models"]["DENT-STRIDE"] = PREDICTIONS

OUTPUT.mkdir(parents=True, exist_ok=True)
payload = {"protocol": {"score_threshold": module.SCORE, "matching_iou": module.IOU, "excluded_model": "RT-DETR-l"}}
for name, config in module.COHORTS.items():
    payload[name] = module.render(name, config, OUTPUT)
(OUTPUT / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({name: payload[name]["figure"] for name in module.COHORTS}, indent=2))
