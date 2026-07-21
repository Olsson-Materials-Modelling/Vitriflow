from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("ase")
pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _write_box_provenance(outdir: Path, box: int) -> tuple[dict[str, str], dict[str, object]]:
    box_dir = outdir / "production" / f"box_{box:03d}"
    relax_data = box_dir / "relax.data"
    relax_data.write_text("recovered structure\n", encoding="utf-8")
    snapshot_path = box_dir / "structure_snapshot.json"
    snapshot_path.write_text(
        json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1}),
        encoding="utf-8",
    )
    manifest_row: dict[str, object] = {"structure_hash": f"structure-{box}"}
    manifest_path = box_dir / "structure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "vitriflow.structure_manifest.v2",
                "structures": [manifest_row],
            }
        ),
        encoding="utf-8",
    )
    return (
        {
            "relax_data": str(relax_data.relative_to(outdir)),
            "structure_snapshot": str(snapshot_path.relative_to(outdir)),
            "structure_manifest": str(manifest_path.relative_to(outdir)),
        },
        manifest_row,
    )


def _production_inputs(tmp_path: Path) -> dict[str, object]:
    from vitriflow.workflows.progress import CondensedProgressLog

    config = SimpleNamespace(
        random_seed=7,
        engine="lammps",
        md=SimpleNamespace(pressure=0.0),
        autotune=SimpleNamespace(
            production=SimpleNamespace(
                enabled=True,
                min_boxes=1,
                max_boxes=1,
                batch_boxes=1,
                check_convergence=True,
                dump_trajectory=False,
                dump_every_steps=500,
                dft_opt=None,
                exclude_coordination_defects=False,
                rejects_subdir="rejects",
                store_distributions=True,
                consecutive_converged_checks=1,
                bondlen_cdf_points=200,
                angle_cdf_points=180,
                warmup_start_temperature=300.0,
                warmup_duration_ps=5.0,
            ),
            convergence=SimpleNamespace(),
        ),
    )
    metrics_cfg = SimpleNamespace(
        enabled=True,
        time_average_frames=1,
        time_average_stride=1,
        elastic=SimpleNamespace(),
    )
    base_data = tmp_path / "base.data"
    base_data.write_text("# deterministic production input\n", encoding="utf-8")
    return {
        "config": config,
        "outdir": tmp_path,
        "runner": object(),
        "pot_cfg": SimpleNamespace(user_units="metal"),
        "md_use": SimpleNamespace(
            timestep=0.001,
            stage_continuity="discontinuous",
            force_isotropic=False,
            pressure=0.0,
            neighbor_skin=0.25,
        ),
        "potential_lines": None,
        "type_to_species": ["Ga", "O"],
        "metrics_cfg": metrics_cfg,
        "tm_cfg": SimpleNamespace(msd_every=100),
        "q_cfg": SimpleNamespace(t_final=300.0, relax_steps=100),
        "size_base_data": base_data,
        "chosen_replicate": [1, 1, 1],
        "chosen_rate": 10.0,
        "dt_ref": 0.001,
        "dt_mq": 0.001,
        "cooling_rate_ps": 10.0,
        "cutoffs_rate": {},
        "cutoffs_size": {},
        "T_high": 1200.0,
        "high_total_steps": 5000,
        "progress": CondensedProgressLog(tmp_path / "condensed.log"),
        "seed_base": 1234,
    }


def _patch_production_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    module = "vitriflow.workflows.autotune"
    monkeypatch.setattr(
        f"{module}.plan_production_stage_diagnostics",
        lambda **kwargs: {
            "dump_traj": False,
            "dump_every": 500,
            "collect_stage_metric_series": False,
            "collect_elastic_series": {
                "melt": False,
                "quench": False,
                "relax": False,
            },
            "need_stage_dump": {
                "melt": False,
                "quench": False,
                "relax": False,
            },
            "quench_dump_every": 250,
            "quench_window_steps_range": (0.0, 10.0),
        },
    )
    monkeypatch.setattr(
        f"{module}.required_pairs_from_metrics", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        f"{module}.fixed_cutoffs_from_metrics", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        f"{module}.should_run_elastic_screen",
        lambda *args, **kwargs: (False, False, None),
    )
    monkeypatch.setattr(
        f"{module}.should_collect_elastic_stage_timeseries",
        lambda *args, **kwargs: (False, False, None),
    )
    monkeypatch.setattr(
        f"{module}.build_production_convergence_spec",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        f"{module}.validate_production_entry_against_spec",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        f"{module}.check_production_convergence",
        lambda *args, **kwargs: (True, {"ok": True}),
    )
    monkeypatch.setattr(
        f"{module}.summarize_production_crystal_motifs",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        f"{module}.graph_analysis_requested", lambda *args, **kwargs: False
    )


def test_resume_reuses_completed_main_stages_after_postprocess_plot_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A plotting failure must not make four completed MD stages run twice."""

    from vitriflow.workflows.autotune import _run_production_ensemble

    _patch_production_dependencies(monkeypatch)
    kwargs = _production_inputs(tmp_path)

    executed: list[tuple[str, int]] = []

    def fake_stage_run(runner, pot_cfg, md_cfg, stage, stage_dir, **unused):
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "stage.data").write_text(
            f"completed {stage.name}\n", encoding="utf-8"
        )
        executed.append((str(stage.name), int(stage.seed)))
        return SimpleNamespace(
            output_data="stage.data",
            density_mean=5.5,
            density_stderr=0.01,
        )

    monkeypatch.setattr(
        "vitriflow.workflows.autotune._stage_run", fake_stage_run
    )

    analysis_calls: list[dict[str, int]] = []

    def fail_plot_once(*, box_id: int, seeds, **unused):
        analysis_calls.append({str(key): int(value) for key, value in seeds.items()})
        if len(analysis_calls) == 1:
            raise RuntimeError(
                "Requested stage metrics plot failed for stage_role=melt: "
                "metrics-timeseries column 'sq_Ga_Ga_peak_fwhm' contains no "
                "finite coordinate/value pairs"
            )
        paths, manifest_row = _write_box_provenance(tmp_path, int(box_id))
        return (
            {
                "box": int(box_id),
                "density": 5.5,
                "metrics": {},
                "distributions": {},
                "paths": paths,
                "structure_manifest": manifest_row,
            },
            {},
        )

    monkeypatch.setattr(
        "vitriflow.workflows.autotune.analyse_production_box", fail_plot_once
    )

    checkpoints: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="Requested stage metrics plot failed"):
        _run_production_ensemble(
            **kwargs,
            checkpoint_cb=lambda state: checkpoints.append(state),
        )

    assert [name for name, _seed in executed] == [
        "warmup",
        "melt",
        "quench",
        "relax",
    ]
    assert checkpoints
    resume_state = checkpoints[-1]
    assert resume_state["n_boxes_total"] == 0
    first_seeds = dict(analysis_calls[0])

    executed.clear()
    recovered: list[tuple[str, int]] = []

    def fake_recover(stage_dir, *, md_cfg, stage, expected_engine, **unused):
        assert expected_engine == "lammps"
        assert stage_dir == tmp_path / "production" / "box_001" / stage.name
        recovered.append((str(stage.name), int(stage.seed)))
        return SimpleNamespace(
            output_data="stage.data",
            density_mean=5.5,
            density_stderr=0.01,
        )

    monkeypatch.setattr(
        "vitriflow.workflows.autotune.recover_completed_stage_outcome",
        fake_recover,
    )

    resumed = _run_production_ensemble(
        **kwargs,
        resume_state=resume_state,
    )

    assert executed == [], "resume reran an already-completed main engine stage"
    assert [name for name, _seed in recovered] == [
        "warmup",
        "melt",
        "quench",
        "relax",
    ]
    assert {name: seed for name, seed in recovered} == first_seeds
    assert analysis_calls == [first_seeds, first_seeds]
    assert resumed["n_boxes"] == 1
    assert resumed["n_boxes_total"] == 1
    assert resumed["boxes"][0]["box"] == 1
    assert resumed["boxes"][0]["resume_recovery"] == {
        "mode": "postprocessing_only",
        "engine_stages_reused": ["warmup", "melt", "quench", "relax"],
        "reason": "engine execution completed before the prior analysis/plotting failure",
    }
    assert (tmp_path / "production" / "box_001").is_dir()
    quarantine = tmp_path / "interrupted_attempts" / "production"
    assert not quarantine.exists() or not any(quarantine.iterdir())


def test_completed_stage_recovery_authenticates_canonical_csvs(
    tmp_path: Path,
) -> None:
    """Recovery accepts a complete stage and rejects later CSV tampering."""

    from vitriflow.io.stage_manifest import write_stage_artifact_manifest
    from vitriflow.lammps_input import StageSpec
    from vitriflow.workflows.stage_runner import recover_completed_stage_outcome

    stage_dir = tmp_path / "melt"
    stage_dir.mkdir()
    (stage_dir / "input.data").write_text(
        "LAMMPS input\n\n1 atoms\n", encoding="utf-8"
    )
    (stage_dir / "output.data").write_text(
        "LAMMPS output\n\n1 atoms\n", encoding="utf-8"
    )
    thermo = stage_dir / "thermo.csv"
    thermo.write_text(
        "Step,Temp,Press,PotEng,Volume,Density\n"
        "0,1200,0.0,-1.0,100.0,5.5\n"
        "2,1201,0.1,-0.9,101.0,5.4\n"
        "4,1199,-0.1,-0.8,100.5,5.45\n",
        encoding="utf-8",
    )
    msd = stage_dir / "msd.csv"
    msd.write_text(
        "Step,MSD\n" + "".join(f"{step},{0.006 * step:.16g}\n" for step in range(5)),
        encoding="utf-8",
    )
    write_stage_artifact_manifest(
        stage_dir,
        engine="lammps",
        lammps_units_style="metal",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    stage = StageSpec(
        name="melt",
        input_data=stage_dir / "input.data",
        output_data=Path("output.data"),
        temperature_start=1200.0,
        temperature_stop=1200.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=4,
        seed=12345,
        write_dump=False,
    )
    md_cfg = SimpleNamespace(timestep=0.001, neighbor_skin=0.25)

    outcome = recover_completed_stage_outcome(
        stage_dir,
        md_cfg=md_cfg,
        stage=stage,
        expected_engine="lammps",
        lammps_units_style="metal",
    )

    assert outcome.name == "melt"
    assert outcome.seed == 12345
    assert outcome.n_atoms == 1
    assert outcome.output_data == "output.data"
    assert outcome.density_mean == pytest.approx(5.45, rel=0.02)

    thermo.write_text(
        thermo.read_text(encoding="utf-8")
        + "5,1200,0.0,-0.7,100.0,5.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        recover_completed_stage_outcome(
            stage_dir,
            md_cfg=md_cfg,
            stage=stage,
            expected_engine="lammps",
            lammps_units_style="metal",
        )
