# Isolated A07 AP-patch experiment

This directory contains a checked snapshot of the Python runtime from
`/home/wuxi/Dent-FINE_ap_patch`, for the explicitly requested A07 rerun only.
It does not overwrite or monkey-patch the parent Dent-FINE implementation.

Use this directory's `train.py` with
`/home/wuxi/Dent-FINE/configs/paper_input1024_20260913/phone_a07_proto_aux.yml`.
The working directory must remain `/home/wuxi/Dent-FINE`.
The other three paper runs must use the parent project's `train.py`.

Source/configuration checksums, the archived runtime configuration, the GPU0
queue, and the experiment protocol are recorded in
`/home/wuxi/Dent-FINE-paper/materials/original_project_input1024_2026-09-13/`.
