from __future__ import annotations

import json
from pathlib import Path

import pytest

from vitriflow.config import RunConfig
from vitriflow.workflows.preflight import CoreRepulsionResult, PreflightError, run_preflight


class _DummyRunner:
    pass


def _base_config(preflight_dt_candidates=None) -> RunConfig:
    preflight = {
        "enabled": True,
        "ensembles": ["npt"],
        "tdamp_factors": [100.0],
        "pdamp_factors": [1000.0],
        "equil_steps": 10,
        "run_steps": 10,
        "confirm_equil_steps": 10,
        "confirm_run_steps": 10,
        "confirm_topk": 1,
        "confirm_temps": [4200.0],
    }
    if preflight_dt_candidates is not None:
        preflight["dt_candidates"] = preflight_dt_candidates
    cfg = {
        "engine": "lammps",
        "random_seed": 23,
        "potential": {
            "kind": "lammps",
            "user_units": "metal",
            "interactions": ["Sm", "O"],
            "commands": [
                "pair_style buck/coul/long 10.0",
                "pair_coeff 1 1 0.0 1.0 0.0",
                "pair_coeff 1 2 1252.94 0.3590 0.0",
                "pair_coeff 2 2 22764.30 0.1490 43.0",
                "kspace_style pppm 1.0e-5",
            ],
            "core_repulsion": {
                "enabled": True,
                "style": "zbl",
                "dt_candidates": [0.001],
                "test_equil_steps": 10,
                "test_run_steps": 10,
            },
        },
        "structure": {
            "charges": {"Sm": 3.0, "O": -2.0},
            "generate": {
                "method": "random",
                "formula": "Sm2O3",
                "n_formula_units": 1,
            },
        },
        "md": {
            "timestep": 0.001,
            "atom_style": "charge",
            "ensemble": "npt",
            "temperature": 300.0,
            "pressure": 0.0,
            "thermo_every": 100,
            "dump_every": 1000,
        },
        "autotune": {
            "preflight": preflight,
            "tm_scan": {"t_min": 1500.0, "t_max": 5500.0, "dT": 1000.0},
            "metrics": {"enabled": False},
        },
    }
    return RunConfig.model_validate(cfg)


def test_preflight_does_not_invent_undeclared_timestep(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.preflight as pf_mod

    cfg = _base_config(preflight_dt_candidates=None)
    dummy_data = tmp_path / "input.data"
    dummy_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr(
        pf_mod,
        "_maybe_apply_core_repulsion",
        lambda *args, **kwargs: (
            ["pair_style buck/coul/long 10.0"],
            CoreRepulsionResult(
                enabled=True,
                applied=True,
                style="zbl",
                base_pair_style="buck/coul/long",
                r_inner=1.0,
                r_outer=1.2,
                attempts=1,
                success=True,
                note="ok",
            ),
            0.001,
        ),
    )

    calls: list[float] = []

    def _fail_select(*args, **kwargs):
        calls.append(float(kwargs["timestep"]))
        raise PreflightError("forced failure", candidates=[])

    monkeypatch.setattr(pf_mod, "_select_md_settings", _fail_select)

    with pytest.raises(PreflightError):
        run_preflight(_DummyRunner(), cfg, dummy_data, tmp_path)

    assert calls == [0.001]
    report = json.loads((tmp_path / "preflight" / "preflight_results.json").read_text())
    assert report["timestep_candidate_source"] == "potential.core_repulsion.dt_candidates"
    assert report["timestep_candidates_tried"] == [0.001]



def test_preflight_uses_only_explicit_preflight_timestep_candidates(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.preflight as pf_mod

    cfg = _base_config(preflight_dt_candidates=[0.001, 0.0005])
    dummy_data = tmp_path / "input.data"
    dummy_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr(
        pf_mod,
        "_maybe_apply_core_repulsion",
        lambda *args, **kwargs: (
            ["pair_style buck/coul/long 10.0"],
            CoreRepulsionResult(
                enabled=True,
                applied=True,
                style="zbl",
                base_pair_style="buck/coul/long",
                r_inner=1.0,
                r_outer=1.2,
                attempts=1,
                success=True,
                note="ok",
            ),
            0.001,
        ),
    )

    calls: list[float] = []

    def _select(*args, **kwargs):
        dt = float(kwargs["timestep"])
        calls.append(dt)
        if abs(dt - 0.001) < 1e-15:
            raise PreflightError("0.001 unstable", candidates=[])
        md_sel = cfg.md.model_copy(
            deep=True,
            update={
                "timestep": dt,
            },
        )
        return md_sel, []

    monkeypatch.setattr(pf_mod, "_select_md_settings", _select)

    res = run_preflight(_DummyRunner(), cfg, dummy_data, tmp_path)

    assert calls == [0.001, 0.0005]
    assert abs(res.selected_timestep - 0.0005) < 1e-15
    report = json.loads((tmp_path / "preflight" / "preflight_results.json").read_text())
    assert report["timestep_candidate_source"] == "autotune.preflight.dt_candidates"
    assert report["timestep_candidates_tried"] == [0.001, 0.0005]
    assert abs(report["selected_timestep"] - 0.0005) < 1e-15
    assert any("timestep fallback activated" in msg for msg in report.get("warnings", []))
    condensed = (tmp_path / "condensed.log").read_text()
    assert "thermo preflight: timestep fallback activated" in condensed
    assert "selected dt=0.0005" in condensed


def test_preflight_does_not_log_fallback_warning_when_first_candidate_passes(monkeypatch, tmp_path: Path):
    import vitriflow.workflows.preflight as pf_mod

    cfg = _base_config(preflight_dt_candidates=[0.001, 0.0005])
    dummy_data = tmp_path / "input.data"
    dummy_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr(
        pf_mod,
        "_maybe_apply_core_repulsion",
        lambda *args, **kwargs: (
            ["pair_style buck/coul/long 10.0"],
            CoreRepulsionResult(
                enabled=True,
                applied=True,
                style="zbl",
                base_pair_style="buck/coul/long",
                r_inner=1.0,
                r_outer=1.2,
                attempts=1,
                success=True,
                note="ok",
            ),
            0.001,
        ),
    )

    def _select(*args, **kwargs):
        dt = float(kwargs["timestep"])
        md_sel = cfg.md.model_copy(deep=True, update={"timestep": dt})
        return md_sel, []

    monkeypatch.setattr(pf_mod, "_select_md_settings", _select)

    res = run_preflight(_DummyRunner(), cfg, dummy_data, tmp_path)

    assert abs(res.selected_timestep - 0.001) < 1e-15
    report = json.loads((tmp_path / "preflight" / "preflight_results.json").read_text())
    assert report.get("warnings", []) == []
    condensed = tmp_path / "condensed.log"
    if condensed.exists():
        assert "timestep fallback activated" not in condensed.read_text()
