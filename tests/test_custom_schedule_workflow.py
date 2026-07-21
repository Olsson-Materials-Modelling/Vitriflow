from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def _generic_hc_raw() -> dict:
    return {
        "custom_schedule": {
            "workflow_label": "hardcarbon_gap20ugr_literature_inspired_schedule",
            "require_continuity": True,
            "stages": [
                {"name": "randomisation", "temperature_K": 9000.0, "time_ps": 10.0, "velocity_mode": "create"},
                {"name": "prequench", "temperature_start_K": 9000.0, "temperature_stop_K": 3500.0, "time_ps": 6.0},
                {"name": "graphitisation", "temperature_K": 3500.0, "time_ps": 400.0, "role": "melt"},
                {"name": "final_quench", "temperature_start_K": 3500.0, "temperature_stop_K": 300.0, "time_ps": 20.0, "role": "quench"},
                {"name": "relaxation", "temperature_K": 300.0, "time_ps": 20.0, "role": "relax"},
            ],
            "analysis_stages": {"melt": "graphitisation", "quench": "final_quench", "relax": "relaxation"},
            "sampling_hint": {"Tm": 3500.0, "freeze_temperature": 300.0},
        }
    }


def test_custom_schedule_gap20ugr_demonstrator_step_counts():
    from vitriflow.config import MDConfig
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _schedule_report, _schedule_steps, _validate_schedule

    schedule = _schedule_from_raw(_generic_hc_raw())
    roles = _validate_schedule(schedule)
    md = MDConfig(timestep=0.001, atom_style="atomic", ensemble="nvt")
    steps = _schedule_steps(schedule, md_use=md, time_unit_ps=1.0)
    assert steps == {
        "randomisation": 10000,
        "prequench": 6000,
        "graphitisation": 400000,
        "final_quench": 20000,
        "relaxation": 20000,
    }
    rep = _schedule_report(schedule, steps, md_use=md, time_unit_ps=1.0, analysis_roles=roles)
    by_name = {s["name"]: s for s in rep["stages"]}
    assert by_name["prequench"]["rate_K_per_ps"] == pytest.approx((9000.0 - 3500.0) / 6.0)
    assert by_name["final_quench"]["rate_K_per_ps"] == pytest.approx((3500.0 - 300.0) / 20.0)
    assert rep["analysis_roles"] == {"melt": "graphitisation", "quench": "final_quench", "relax": "relaxation"}


def test_custom_schedule_rejects_temperature_discontinuity():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _validate_schedule

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][1]["temperature_start_K"] = 8999.0
    with pytest.raises(ValueError, match="discontinuous"):
        _validate_schedule(_schedule_from_raw(raw))


def test_custom_schedule_rejects_later_velocity_create_in_continuous_mode():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _validate_schedule

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][1]["velocity_mode"] = "create"
    with pytest.raises(ValueError, match="velocity creation on the first stage"):
        _validate_schedule(_schedule_from_raw(raw))


def test_custom_schedule_rejects_fractional_step_override():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][0]["steps"] = 10.5
    with pytest.raises(ValueError, match="integer"):
        _schedule_from_raw(raw)


@pytest.mark.parametrize(
    "unsafe_name",
    [".", "..", "../escape", "sub/stage", " melt", "melt "],
)
def test_custom_schedule_rejects_stage_names_that_escape_box_directory(
    unsafe_name: str,
):
    from vitriflow.workflows.custom_schedule import _schedule_from_raw

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][0]["name"] = unsafe_name
    with pytest.raises(ValueError, match="not path-safe"):
        _schedule_from_raw(raw)


def test_custom_schedule_rejects_unknown_fields_and_duplicate_aliases():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][0]["dump_evry_steps"] = 10
    with pytest.raises(ValueError, match=r"unknown field.*dump_evry_steps"):
        _schedule_from_raw(raw)

    raw = _generic_hc_raw()
    raw["custom_schedule"]["stages"][0]["duration_ps"] = 10
    with pytest.raises(ValueError, match=r"multiple aliases.*time_ps.*duration_ps"):
        _schedule_from_raw(raw)


def test_custom_schedule_cli_subcommands_are_registered(capsys):
    import vitriflow.cli as cli

    for cmd in ["run-schedule", "run-custom", "run-hardcarbon", "run-hc"]:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([cmd, "--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "--config" in out and "--outdir" in out


def test_custom_schedule_demo_config_parse_and_locked_schedule(tmp_path: Path):
    from vitriflow.config import RunConfig
    from vitriflow.workflows import custom_schedule as cs

    packaged_cfg = Path(__file__).resolve().parents[1] / "vitriflow" / "examples" / "hc_C_GAP20Ugr_hc_custom_demo.yaml"
    assert packaged_cfg.exists()
    with pytest.raises(ValueError, match=r"potential\.files entry is not a file"):
        RunConfig.from_yaml(packaged_cfg)

    # The demonstrator deliberately does not bundle the third-party GAP
    # potential.  Materialise placeholder declared files only for parser and
    # schedule-lock testing; this is not an engine execution test.
    raw = yaml.safe_load(packaged_cfg.read_text()) or {}
    potential_files: list[str] = []
    for index, original in enumerate(raw["potential"]["files"]):
        placeholder = tmp_path / f"potential_{index}_{Path(original).name}"
        placeholder.write_text(f"test-placeholder-{index}\n")
        potential_files.append(str(placeholder))
    raw["potential"]["files"] = potential_files
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    cfg = RunConfig.from_yaml(cfg_path)
    assert cfg.engine == "lammps"
    assert cfg.md.ensemble == "nvt"
    assert cfg.md.stage_continuity == "continuous"

    schedule = cs._schedule_from_raw(raw)
    roles = cs._validate_schedule(schedule)
    steps = cs._schedule_steps(schedule, md_use=cfg.md, time_unit_ps=1.0)
    assert steps["randomisation"] == 10000
    assert steps["prequench"] == 6000
    assert steps["graphitisation"] == 400000
    assert steps["final_quench"] == 20000
    assert steps["relaxation"] == 20000
    assert roles == {"melt": "graphitisation", "quench": "final_quench", "relax": "relaxation"}


def test_standard_workflows_do_not_contain_custom_schedule_branches():
    import vitriflow

    pkg = Path(vitriflow.__file__).resolve().parent
    forbidden = ["custom_schedule", "run_custom_schedule(", "hardcarbon_schedule"]
    for rel in ["workflows/run.py", "workflows/autotune.py"]:
        text = (pkg / rel).read_text()
        assert not any(token in text for token in forbidden), rel


def test_unstructured_restart_is_not_sent_to_generic_ase_reader(tmp_path: Path):
    from vitriflow.analysis.trajectory import read_last_frames_auto

    restart = tmp_path / "calc-1.restart"
    restart.write_text("final restart\n")
    with pytest.raises(ValueError, match="strict final restart could not be parsed"):
        read_last_frames_auto(restart, 1, type_to_species=["Al"])


def test_custom_schedule_string_booleans_are_parsed_not_python_truthy():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _validate_schedule

    raw = _generic_hc_raw()
    raw["custom_schedule"]["require_continuity"] = "false"
    raw["custom_schedule"]["stages"][0]["write_dump"] = "false"
    raw["custom_schedule"]["stages"][0]["force_isotropic"] = "false"
    # Deliberately introduce a discontinuity: it should be accepted because require_continuity=false.
    raw["custom_schedule"]["stages"][1]["temperature_start_K"] = 8999.0
    schedule = _schedule_from_raw(raw)
    assert schedule.enforce_temperature_continuity is False
    assert schedule.stages[0].write_dump is False
    assert schedule.stages[0].force_isotropic is False
    _validate_schedule(schedule)


def test_custom_schedule_guard_rejects_dft_refinement_path():
    from types import SimpleNamespace

    from vitriflow.workflows.custom_schedule import _guard_custom_schedule_supported_equivalence_paths

    cfg = SimpleNamespace(
        autotune=SimpleNamespace(
            production=SimpleNamespace(dft_opt=SimpleNamespace(enabled=True))
        )
    )
    metrics = SimpleNamespace(elastic=SimpleNamespace(enabled=False))
    with pytest.raises(ValueError, match=r"dft_opt\.enabled=true"):
        _guard_custom_schedule_supported_equivalence_paths(
            config=cfg,
            metrics_cfg=metrics,
            runner=object(),
            force_isotropic=False,
        )


def test_custom_schedule_guard_rejects_silently_ignored_autocore():
    from types import SimpleNamespace

    from vitriflow.workflows.custom_schedule import (
        _guard_custom_schedule_supported_equivalence_paths,
    )

    cfg = SimpleNamespace(
        kim=SimpleNamespace(
            core_repulsion=SimpleNamespace(enabled=True),
        ),
        autotune=SimpleNamespace(
            production=SimpleNamespace(dft_opt=SimpleNamespace(enabled=False))
        ),
    )
    metrics = SimpleNamespace(elastic=SimpleNamespace(enabled=False))
    with pytest.raises(ValueError, match="does not silently realize autocore"):
        _guard_custom_schedule_supported_equivalence_paths(
            config=cfg,
            metrics_cfg=metrics,
            runner=object(),
            force_isotropic=False,
        )


def test_custom_schedule_guard_rejects_elastic_auto_paths():
    from types import SimpleNamespace

    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.custom_schedule import _guard_custom_schedule_supported_equivalence_paths

    cfg = SimpleNamespace(
        autotune=SimpleNamespace(
            production=SimpleNamespace(dft_opt=SimpleNamespace(enabled=False))
        )
    )
    metrics = SimpleNamespace(
        elastic=SimpleNamespace(
            enabled="auto",
            run_on_relax=True,
            run_on_highT_when_force_isotropic=True,
            collect_during_production_stages=True,
            strict_when_force_isotropic=True,
        )
    )
    with pytest.raises(ValueError, match="elastic production screens/timeseries"):
        _guard_custom_schedule_supported_equivalence_paths(
            config=cfg,
            metrics_cfg=metrics,
            runner=LammpsRunner(LammpsConfig()),
            force_isotropic=False,
        )


def test_custom_schedule_guard_accepts_elastic_disabled():
    from types import SimpleNamespace

    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.custom_schedule import _guard_custom_schedule_supported_equivalence_paths

    cfg = SimpleNamespace(
        autotune=SimpleNamespace(
            production=SimpleNamespace(dft_opt=SimpleNamespace(enabled=False))
        )
    )
    metrics = SimpleNamespace(elastic=SimpleNamespace(enabled=False))
    _guard_custom_schedule_supported_equivalence_paths(
        config=cfg,
        metrics_cfg=metrics,
        runner=LammpsRunner(LammpsConfig()),
        force_isotropic=False,
    )


def test_custom_schedule_runner_scope_is_lammps_continuous_only():
    from types import SimpleNamespace

    from vitriflow.workflows.custom_schedule import _guard_custom_schedule_runner_scope

    cfg = SimpleNamespace(
        engine="cp2k",
        kim=object(),
        md=SimpleNamespace(stage_continuity="continuous"),
    )
    with pytest.raises(ValueError, match="engine='lammps'"):
        _guard_custom_schedule_runner_scope(cfg)

    cfg = SimpleNamespace(
        engine="lammps",
        kim=object(),
        md=SimpleNamespace(stage_continuity="discontinuous"),
    )
    with pytest.raises(ValueError, match="stage_continuity: continuous"):
        _guard_custom_schedule_runner_scope(cfg)


def test_custom_schedule_resume_admission_is_fail_closed(tmp_path: Path):
    from vitriflow.workflows.custom_schedule import (
        _resolve_custom_schedule_resume_mode,
    )

    outdir = tmp_path / "custom"
    outdir.mkdir()
    result = outdir / "run_results.json"
    assert not _resolve_custom_schedule_resume_mode(
        outdir=outdir,
        results_path=result,
        resume=None,
    )

    orphan = outdir / "production" / "box_000" / "melt"
    orphan.mkdir(parents=True)
    (orphan / "partial.out").write_text("partial")
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        _resolve_custom_schedule_resume_mode(
            outdir=outdir,
            results_path=result,
            resume=None,
        )
    assert (orphan / "partial.out").read_text() == "partial"

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(RuntimeError, match="--resume was requested"):
        _resolve_custom_schedule_resume_mode(
            outdir=missing,
            results_path=missing / "run_results.json",
            resume=True,
        )

def test_custom_schedule_final_status_reports_max_box_nonconvergence():
    from vitriflow.workflows.custom_schedule import _final_status

    status, error = _final_status(
        n_accepted=20,
        min_boxes=20,
        check_convergence=True,
        converged=False,
        max_boxes=80,
        n_total=80,
    )
    assert status == "not_converged"
    assert "max_boxes=80" in str(error)

    status, error = _final_status(
        n_accepted=0,
        min_boxes=1,
        check_convergence=False,
        converged=True,
        max_boxes=1,
        n_total=1,
    )
    assert status == "incomplete"
    assert "below min_boxes" in str(error)


def _write_fingerprint_demo_config(tmp_path: Path, *, xml_text: str = '<GAP></GAP>') -> Path:
    pot_dir = tmp_path / "potentials"
    pot_dir.mkdir()
    xml_path = pot_dir / "gap_test.xml"
    xml_path.write_text(xml_text)
    sidecar = pot_dir / "gap_test.xml.sparseX.TEST"
    sidecar.write_text("sparse data\n")
    cfg = {
        "engine": "lammps",
        "random_seed": 2468,
        "lammps": {"lammps_cmd": "lmp", "nprocs": 1},
        "potential": {
            "kind": "lammps",
            "user_units": "metal",
            "interactions": ["C"],
            "files": [str(xml_path), str(sidecar)],
            "commands": [
                "pair_style quip",
                'pair_coeff * * gap_test.xml "Potential xml_label=GAP_TEST" 6',
            ],
        },
        "md": {
            "timestep": 0.001,
            "atom_style": "atomic",
            "ensemble": "nvt",
            "temperature": 300.0,
            "pressure": 0.0,
            "stage_continuity": "continuous",
            "thermostat": {"style": "nose-hoover", "tdamp": 0.1},
            "barostat": {"style": "nose-hoover", "pdamp": 1.0},
        },
        "structure": {
            "generate": {
                "method": "random",
                "formula": "C",
                "n_formula_units": 2,
                "random_fallback_density_g_cm3": 2.0,
                "random_min_distance": 1.0,
                "seed": 123,
            }
        },
        "custom_schedule": {
            "workflow_label": "fingerprint_test",
            "stages": [
                {"name": "melt", "temperature_K": 1000.0, "steps": 2, "role": "melt", "velocity_mode": "create"},
                {"name": "quench", "temperature_start_K": 1000.0, "temperature_stop_K": 300.0, "steps": 2, "role": "quench"},
                {"name": "relax", "temperature_K": 300.0, "steps": 2, "role": "relax"},
            ],
            "analysis_roles": {"melt": "melt", "quench": "quench", "relax": "relax"},
        },
        "autotune": {
            "preflight": {"enabled": False},
            "metrics": {
                "enabled": True,
                "type_to_species": ["C"],
                "elastic": {"enabled": False},
                "pairs": [{"pair": ["C", "C"], "cutoff": 1.85}],
                "voids": {"enabled": True, "default_radius": 1.7, "radii": {"C": 1.7}},
                "amorphous": {"enabled": False},
            },
            "production": {
                "enabled": True,
                "min_boxes": 1,
                "max_boxes": 1,
                "batch_boxes": 1,
                "check_convergence": False,
                "store_distributions": True,
            },
            "convergence": {"mode": "both", "familywise": "none"},
        },
    }
    cfg_path = tmp_path / "fingerprint_demo.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _fingerprint_from_config_path(cfg_path: Path) -> dict:
    from vitriflow.config import RunConfig
    from vitriflow.workflows.custom_schedule import (
        _build_resume_fingerprint,
        _schedule_from_raw,
        _schedule_report,
        _schedule_steps,
        _validate_schedule,
    )

    cfg = RunConfig.from_yaml(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    schedule = _schedule_from_raw(raw)
    roles = _validate_schedule(schedule)
    steps = _schedule_steps(schedule, md_use=cfg.md, time_unit_ps=1.0)
    report = _schedule_report(schedule, steps, md_use=cfg.md, time_unit_ps=1.0, analysis_roles=roles)
    return _build_resume_fingerprint(
        config=cfg,
        schedule=schedule,
        analysis_roles=roles,
        steps=steps,
        sched_report=report,
        time_unit_ps=1.0,
        md_pressure=0.0,
        lammps_units="metal",
        config_path=cfg_path,
    )


def test_custom_schedule_resume_fingerprint_is_deterministic_and_hashes_gap_xml(tmp_path: Path):
    import hashlib

    xml_text = '<GAP label="GAP_TEST"></GAP>\n'
    cfg_path = _write_fingerprint_demo_config(tmp_path, xml_text=xml_text)
    fp1 = _fingerprint_from_config_path(cfg_path)
    fp2 = _fingerprint_from_config_path(cfg_path)
    assert fp1["sha256"] == fp2["sha256"]

    xml_id = fp1["payload"]["potential"]["gap_xml_identity"]["configured_xml_files"][0]
    assert xml_id["filename"] == "gap_test.xml"
    assert xml_id["sha256"] == hashlib.sha256(xml_text.encode()).hexdigest()
    cmd_ref = fp1["payload"]["potential"]["gap_xml_identity"]["command_xml_references"][0]
    assert cmd_ref["xml_labels"] == ["GAP_TEST"]


def test_custom_schedule_resume_fingerprint_changes_when_schedule_changes(tmp_path: Path):
    from vitriflow.workflows.custom_schedule import _validate_resume_fingerprint_or_raise

    cfg_path = _write_fingerprint_demo_config(tmp_path)
    fp1 = _fingerprint_from_config_path(cfg_path)
    data = yaml.safe_load(cfg_path.read_text())
    data["custom_schedule"]["stages"][1]["steps"] = 3
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
    fp2 = _fingerprint_from_config_path(cfg_path)
    assert fp1["sha256"] != fp2["sha256"]
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _validate_resume_fingerprint_or_raise({"resume_fingerprint": fp1}, fp2, outdir=tmp_path)


def test_custom_schedule_resume_fingerprint_changes_when_xml_content_changes(tmp_path: Path):
    cfg_path = _write_fingerprint_demo_config(tmp_path, xml_text="<GAP>A</GAP>\n")
    fp1 = _fingerprint_from_config_path(cfg_path)
    (tmp_path / "potentials" / "gap_test.xml").write_text("<GAP>B</GAP>\n")
    fp2 = _fingerprint_from_config_path(cfg_path)
    assert fp1["sha256"] != fp2["sha256"]


def test_custom_schedule_resume_requires_existing_fingerprint(tmp_path: Path):
    from vitriflow.workflows.custom_schedule import _validate_resume_fingerprint_or_raise

    cfg_path = _write_fingerprint_demo_config(tmp_path)
    fp = _fingerprint_from_config_path(cfg_path)
    with pytest.raises(RuntimeError, match="no custom-schedule provenance fingerprint"):
        _validate_resume_fingerprint_or_raise({"status": "running"}, fp, outdir=tmp_path)


def _run_resumed_graph_finalization(
    monkeypatch,
    tmp_path: Path,
    *,
    graph_rules,
    writer,
    check_convergence: bool = False,
    required_streak: int = 1,
    store_distributions: bool = True,
    converged_now: bool = False,
    convergence_streak: int = 0,
    last_evaluated: bool = False,
    convergence_report: dict | None = None,
):
    import json

    from vitriflow.config import RunConfig
    from vitriflow.workflows.custom_schedule import run_custom_schedule
    from vitriflow.workflows import metrics_policy, production_common, progress
    import vitriflow.analysis.motif_summary as motif_summary
    import vitriflow.structuregen as structuregen

    cfg_path = _write_fingerprint_demo_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["autotune"]["metrics"]["graph_rules"] = list(graph_rules)
    raw["autotune"]["production"]["check_convergence"] = bool(check_convergence)
    raw["autotune"]["production"]["consecutive_converged_checks"] = int(
        required_streak
    )
    raw["autotune"]["production"]["store_distributions"] = bool(
        store_distributions
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = RunConfig.from_yaml(cfg_path)
    fingerprint = _fingerprint_from_config_path(cfg_path)

    outdir = tmp_path / "run"
    outdir.mkdir()
    source = outdir / "production" / "box_000" / "relax.data"
    source.parent.mkdir(parents=True)
    source.write_text("box-0\n")
    snapshot = source.parent / "structure_snapshot.json"
    manifest_path = source.parent / "structure_manifest.json"
    snapshot.write_text(json.dumps({"schema": "vitriflow.structure_snapshot.v1", "n_atoms": 1}))
    manifest = {"structure_hash": "structure-0"}
    manifest_path.write_text(
        json.dumps({"schema": "vitriflow.structure_manifest.v2", "structures": [manifest]})
    )
    previous = {
        "status": "running",
        "execution_status": "running",
        "resume_fingerprint": fingerprint,
        "production": {
            "enabled": True,
            "status": "running",
            "execution_status": "running",
            "converged": False,
            "converged_md": bool(converged_now),
            "check_convergence": bool(check_convergence),
            "resumable": True,
            "convergence_streak": int(convergence_streak),
            "required_convergence_streak": int(required_streak),
            "last_convergence_evaluated_n_boxes_total": (1 if last_evaluated else None),
            "last_convergence_evaluated_n_boxes_accepted": (1 if last_evaluated else None),
            "min_boxes": 1,
            "max_boxes": 1,
            "batch_boxes": 1,
            "n_boxes": 1,
            "n_boxes_accepted": 1,
            "n_boxes_rejected": 0,
            "n_boxes_total": 1,
            "boxes": [{
                "box": 0,
                "density": 2.35,
                "metrics": {},
                "distributions": {},
                "paths": {
                    "relax_data": "production/box_000/relax.data",
                    "structure_snapshot": "production/box_000/structure_snapshot.json",
                    "structure_manifest": "production/box_000/structure_manifest.json",
                },
                "structure_manifest": manifest,
            }],
            "rejected_boxes": [],
            "cutoffs": [{"pair": [1, 1], "cutoff": 1.85}],
            "convergence": dict(convergence_report or {}),
        },
    }
    from vitriflow.workflows.autotune import _attach_production_state_integrity

    previous["production"] = _attach_production_state_integrity(
        previous["production"], outdir=outdir
    )
    (outdir / "run_results.json").write_text(json.dumps(previous))

    def _prepare(_config, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "initial.data"
        path.write_text("# fixture\n")
        return path

    monkeypatch.setattr(structuregen, "prepare_initial_structure", _prepare)
    monkeypatch.setattr(
        metrics_policy,
        "resolve_effective_metrics_config",
        lambda metrics, **kwargs: (metrics, {}, {}),
    )
    monkeypatch.setattr(
        production_common,
        "plan_production_stage_diagnostics",
        lambda **kwargs: {
            "dump_traj": False,
            "dump_every": 1,
            "need_stage_dump": {"melt": False, "quench": False, "relax": False},
            "quench_dump_every": 1,
            "quench_window_steps_range": None,
        },
    )
    monkeypatch.setattr(
        production_common,
        "resolve_production_relax_dump_settings",
        lambda **kwargs: {
            "write_dump": False,
            "dump_every": None,
            "tail_dump_frames": None,
            "tail_dump_stride": None,
            "mode": "none",
        },
    )
    monkeypatch.setattr(production_common, "write_graph_analysis_outputs", writer)
    monkeypatch.setattr(motif_summary, "summarize_production_crystal_motifs", lambda *args, **kwargs: {})

    written_states: list[dict] = []
    real_write = progress.atomic_write_json

    def _record_write(path, payload):
        if Path(path).name == "run_results.json":
            written_states.append(json.loads(json.dumps(payload)))
        return real_write(path, payload)

    monkeypatch.setattr(progress, "atomic_write_json", _record_write)
    result = run_custom_schedule(config, outdir, config_path=cfg_path, resume=True)
    return result, written_states


def test_custom_schedule_default_never_calls_graph_writer(monkeypatch, tmp_path: Path):
    def _unexpected(*args, **kwargs):
        raise AssertionError("graph writer called without explicit graph_rules")

    result, states = _run_resumed_graph_finalization(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        writer=_unexpected,
    )

    assert states and all(state["graph_outputs"] == {} for state in states)
    assert all(state["production"]["graph_outputs"] == {} for state in states)
    assert result["graph_outputs"] == {}
    assert result["status"] == "ok"
    assert result["production"]["converged"] is None
    assert result["production"]["converged_md"] is None
    assert result["production"]["convergence_status"] == "fixed_count_unassessed"
    assert result["production"]["convergence_inference_status"] == (
        "fixed_n_terminal_posthoc_not_sequentially_valid"
    )
    assert result["production"]["achieved_convergence_degree"]["n_boxes"] == 1
    assert result["production"]["posthoc_convergence_criterion_met"] is False
    report = result["production"]["convergence"]
    assert report["status"] == "fixed_n_terminal_posthoc_assessed"
    assert report["sampling_design"] == "fixed_n"
    assert report["used_for_stopping"] is False
    assert report["stopping_status"] == "fixed_count_unassessed"
    assert report["posthoc_failed_items"] == [
        {
            "section": "ci",
            "name": "scalar:density",
            "reason": "tolerance_not_met",
        },
        {
            "section": "stability",
            "name": "stability",
            "reason": "active_section_unassessed",
        },
    ]


def test_custom_schedule_explicit_graph_rules_finalize_once_after_checkpoints(monkeypatch, tmp_path: Path):
    calls: list[dict] = []

    def _writer(*args, **kwargs):
        calls.append(dict(kwargs))
        (Path(args[0]) / "graph_rules.json").write_text("{}\n")
        return {"graph_rules": "graph_rules.json"}

    result, states = _run_resumed_graph_finalization(
        monkeypatch,
        tmp_path,
        graph_rules=[{"name": "requested", "kind": "hard_cutoff", "parameters": {"cutoff": 1.85}}],
        writer=_writer,
    )

    assert len(calls) == 1
    assert len(states) >= 2
    assert all(state["graph_outputs"] == {} for state in states[:-1])
    assert all(state["production"]["graph_outputs"] == {} for state in states[:-1])
    assert states[-1]["graph_outputs"] == {"graph_rules": "graph_rules.json"}
    assert result["graph_outputs"] == {"graph_rules": "graph_rules.json"}


def test_custom_schedule_rejects_potential_mutated_during_resume_finalization(
    monkeypatch,
    tmp_path: Path,
):
    potential = tmp_path / "potentials" / "gap_test.xml"

    def _mutating_writer(*_args, **_kwargs):
        potential.write_text('<GAP label="MUTATED_DURING_EXECUTION"></GAP>\n')
        return {}

    with pytest.raises(
        RuntimeError,
        match="scientific input bytes changed during execution",
    ):
        _run_resumed_graph_finalization(
            monkeypatch,
            tmp_path,
            graph_rules=[
                {
                    "name": "mutation-trigger",
                    "kind": "hard_cutoff",
                    "parameters": {"cutoff": 1.85},
                }
            ],
            writer=_mutating_writer,
        )


def test_custom_resume_preserves_exact_streak_and_full_running_state(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import production_common

    monkeypatch.setattr(
        production_common,
        "check_production_convergence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged custom ensemble must not be re-evaluated")
        ),
    )
    result, states = _run_resumed_graph_finalization(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        writer=lambda *args, **kwargs: {},
        check_convergence=True,
        required_streak=2,
        store_distributions=False,
        converged_now=True,
        convergence_streak=1,
        last_evaluated=True,
    )

    assert result["status"] == "not_converged"
    assert result["production"]["convergence_streak"] == 1
    assert result["production"]["resumable"] is False
    assert "distributions" not in result["production"]["boxes"][0]
    assert len(states) >= 2
    assert all(
        "distributions" in state["production"]["boxes"][0]
        for state in states[:-1]
    )


def test_custom_terminal_state_surfaces_inference_qualified_degree(
    monkeypatch, tmp_path: Path
):
    convergence_report = {
        "inference_contract": {"sequentially_valid": False},
        "achieved_convergence_degree": {
            "n_boxes": 1,
            "overall_active": {"worst_tolerance_utilization_ratio": 0.25},
        },
        "convergence_degree": {
            "overall": {"n_checked": 2, "n_passed": 2, "pass_fraction": 1.0}
        },
    }
    result, _states = _run_resumed_graph_finalization(
        monkeypatch,
        tmp_path,
        graph_rules=[],
        writer=lambda *args, **kwargs: {},
        check_convergence=True,
        required_streak=1,
        converged_now=True,
        convergence_streak=1,
        last_evaluated=True,
        convergence_report=convergence_report,
    )

    production = result["production"]
    assert production["convergence_status"] == "converged"
    assert production["convergence_inference_status"] == (
        "criterion_met_repeated_looks_not_sequentially_valid"
    )
    assert production["achieved_convergence_degree"] == convergence_report[
        "achieved_convergence_degree"
    ]
    assert production["convergence_criterion_coverage"]["overall"]["n_checked"] == 2
