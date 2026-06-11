from __future__ import annotations

import pytest

pytest.importorskip("ase")

from pathlib import Path
from types import SimpleNamespace


def test_run_meltquench_reports_replicate_convergence(monkeypatch, tmp_path: Path):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import run as run_mod

    cfg = RunConfig.model_validate(
        {
            "potential": {"kind": "mg2_sin", "user_units": "metal", "interactions": ["Si", "N"]},
            "structure": {"generate": {"method": "random", "formula": "Si3N4", "n_formula_units": 1}},
            "autotune": {
                "metrics": {
                    "enabled": True,
                    "voids": {"enabled": True},
                    "pairs": [{"pair": ["Si", "N"]}],
                },
                "production": {
                    "enabled": True,
                    "check_convergence": True,
                    "min_boxes": 2,
                    "consecutive_converged_checks": 1,
                    "store_distributions": True,
                },
            },
        }
    )

    dummy_data = tmp_path / "initial.data"
    dummy_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr(run_mod, "prepare_initial_structure", lambda config, outdir: str(dummy_data))
    monkeypatch.setattr(run_mod, "ensure_model_installed", lambda model: None)
    monkeypatch.setattr(
        run_mod,
        "run_preflight",
        lambda runner, config, initial_data, outdir: SimpleNamespace(
            selected_timestep=1.0,
            selected_ensemble="npt",
            selected_tdamp=100.0,
            selected_pdamp=1000.0,
            potential_lines=None,
        ),
    )

    def _fake_exec(**kwargs):
        kwargs["progress"].convergence(
            "production",
            {
                "passed": True,
                "groups": {"short": True, "medium": True, "long": True},
                "metrics": {
                    "density": {"passed": True},
                    "ring_frac_3": {"passed": True},
                    "Si-N": {"passed": True},
                },
            },
        )
        return {
            "status": "ok",
            "converged": True,
            "n_boxes_total": 2,
            "boxes": [
                {"box": 1, "density": 2.21, "distributions": {}, "metrics": {}},
                {"box": 2, "density": 2.20, "distributions": {}, "metrics": {}},
            ],
            "rejected_boxes": [],
            "convergence": {"passed": True},
            "cutoffs": [{"pair": [1, 2], "cutoff": 2.5}],
        }

    monkeypatch.setattr(run_mod, "_run_production_executor", _fake_exec)

    outdir = tmp_path / "run_out"
    summary = run_mod.run_meltquench(cfg, outdir, n_replicates=2)

    assert summary["production"]["n_boxes_total"] == 2
    assert summary["production"]["converged"] is True
    assert summary["production"]["convergence"]["passed"] is True
    assert len(summary["replicates"]) == 2
    assert summary["production_plan"]["execution_mode"] == "fixed"
    assert summary["production_plan"]["seed_base"] == cfg.random_seed + 13579
    assert (outdir / "run_results.json").exists()

    condensed = (outdir / "condensed.log").read_text()
    assert "total convergence=pass" in condensed
    assert "per-metric convergence" in condensed
