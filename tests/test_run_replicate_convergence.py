from __future__ import annotations

import pytest

pytest.importorskip("ase")
pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")

from pathlib import Path
from types import SimpleNamespace
import json


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
        boxes = []
        for box, density in ((1, 2.21), (2, 2.20)):
            bdir = Path(kwargs["outdir"]) / "production" / f"box_{box:03d}"
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / "relax.data").write_text(f"box-{box}\n")
            (bdir / "structure_snapshot.json").write_text(
                json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1})
            )
            manifest = {"structure_hash": f"structure-{box}"}
            (bdir / "structure_manifest.json").write_text(
                json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest]})
            )
            boxes.append({
                "box": box,
                "density": density,
                "distributions": {},
                "metrics": {},
                "paths": {
                    "relax_data": str((bdir / "relax.data").relative_to(kwargs["outdir"])),
                    "structure_snapshot": str((bdir / "structure_snapshot.json").relative_to(kwargs["outdir"])),
                    "structure_manifest": str((bdir / "structure_manifest.json").relative_to(kwargs["outdir"])),
                },
                "structure_manifest": manifest,
            })
        return {
                "status": "ok",
                "converged": True,
                "converged_md": True,
                "check_convergence": True,
                "resumable": True,
                "convergence_streak": 1,
                "required_convergence_streak": 1,
                "last_convergence_evaluated_n_boxes_total": 2,
                "last_convergence_evaluated_n_boxes_accepted": 2,
            "min_boxes": 2,
            "n_boxes": 2,
            "n_boxes_accepted": 2,
            "n_boxes_rejected": 0,
            "n_boxes_total": 2,
            "boxes": boxes,
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
