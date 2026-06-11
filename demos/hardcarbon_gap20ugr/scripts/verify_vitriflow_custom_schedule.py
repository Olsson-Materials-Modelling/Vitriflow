#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml
import vitriflow
from vitriflow.config import RunConfig

pkg = Path(vitriflow.__file__).resolve().parent
root = Path(__file__).resolve().parents[1]
print(f"VitriFlow package: {pkg}")

custom = importlib.import_module("vitriflow.workflows.custom_schedule")
assert hasattr(custom, "run_custom_schedule"), "vitriflow.workflows.custom_schedule.run_custom_schedule missing"
print(f"Custom schedule workflow: {Path(inspect.getsourcefile(custom)).resolve()}")

# The normal workflows must not contain custom-schedule execution branches.
for rel in ["workflows/autotune.py", "workflows/run.py"]:
    text = (pkg / rel).read_text()
    forbidden = ["run_custom_schedule(", "custom_stage_schedule", "hardcarbon_schedule"]
    bad = [s for s in forbidden if s in text]
    if bad:
        raise SystemExit(f"ERROR: standard workflow {rel} contains custom-schedule tokens: {bad}")
print("OK: standard run/autotune workflows contain no custom-schedule branches")

cli_text = (pkg / "cli.py").read_text()
if "run-schedule" not in cli_text or "run-custom" not in cli_text:
    raise SystemExit("ERROR: CLI lacks run-schedule/run-custom command")
print("OK: CLI exposes run-schedule / run-custom")

configs = sorted((root / "configs").glob("hc_C_GAP20Ugr_hc_custom*.yaml"))
if configs:
    for cfg_path in configs:
        cfg = RunConfig.from_yaml(cfg_path)
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        schedule = custom._schedule_from_raw(raw)
        roles = custom._validate_schedule(schedule)
        steps = custom._schedule_steps(schedule, md_use=cfg.md, time_unit_ps=1.0)
        if "pilot" not in cfg_path.name:
            expected = {
                "randomisation": 10000,
                "prequench": 6000,
                "graphitisation": 400000,
                "final_quench": 20000,
                "relaxation": 20000,
            }
            if steps != expected:
                raise SystemExit(f"ERROR: {cfg_path.name} schedule steps {steps} != {expected}")
        if roles != {"melt": "graphitisation", "quench": "final_quench", "relax": "relaxation"}:
            raise SystemExit(f"ERROR: {cfg_path.name} analysis roles {roles}")
        if cfg.md.ensemble.lower() != "nvt":
            raise SystemExit(f"ERROR: {cfg_path.name} must use NVT")
        if cfg.md.stage_continuity.lower() != "continuous":
            raise SystemExit(f"ERROR: {cfg_path.name} must use continuous stages")
    print(f"OK: validated {len(configs)} custom-schedule config(s)")
