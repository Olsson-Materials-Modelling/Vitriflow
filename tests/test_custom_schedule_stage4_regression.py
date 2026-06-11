from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _write_minimal_lammps_data(path: Path) -> Path:
    path.write_text(
        """
LAMMPS data file via vitriflow stage-4 regression test

1 atoms
1 atom types

0.0 1.0 xlo xhi
0.0 1.0 ylo yhi
0.0 1.0 zlo zhi

Masses

1 12.011

Atoms # atomic

1 1 0.0 0.0 0.0
""".lstrip()
    )
    return path


def _five_stage_specs(tmp_path: Path):
    from vitriflow.lammps_input import StageSpec

    input_data = _write_minimal_lammps_data(tmp_path / "input.data")
    return [
        StageSpec(
            name="randomisation",
            input_data=input_data,
            output_data=tmp_path / "randomisation.data",
            temperature_start=9000.0,
            temperature_stop=9000.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=10,
            seed=101,
            velocity_mode="create",
            write_dump=True,
        ),
        StageSpec(
            name="prequench",
            input_data=tmp_path / "randomisation.data",
            output_data=tmp_path / "prequench.data",
            temperature_start=9000.0,
            temperature_stop=3500.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=6,
            seed=102,
            velocity_mode="preserve",
            write_dump=True,
        ),
        StageSpec(
            name="graphitisation",
            input_data=tmp_path / "prequench.data",
            output_data=tmp_path / "graphitisation.data",
            temperature_start=3500.0,
            temperature_stop=3500.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=40,
            seed=103,
            velocity_mode="preserve",
            write_dump=True,
        ),
        StageSpec(
            name="final_quench",
            input_data=tmp_path / "graphitisation.data",
            output_data=tmp_path / "final_quench.data",
            temperature_start=3500.0,
            temperature_stop=300.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=20,
            seed=104,
            velocity_mode="preserve",
            write_dump=True,
        ),
        StageSpec(
            name="relaxation",
            input_data=tmp_path / "final_quench.data",
            output_data=tmp_path / "relaxation.data",
            temperature_start=300.0,
            temperature_stop=300.0,
            pressure=0.0,
            equil_steps=0,
            run_steps=20,
            seed=105,
            velocity_mode="preserve",
            write_dump=True,
        ),
    ]


def test_stage4_full_five_stage_render_has_single_read_and_velocity_create(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_continuous_stages

    stages = _five_stage_specs(tmp_path)
    stage_dirs = {st.name: f"../{st.name}" for st in stages}
    script = render_continuous_stages(
        KimConfig(model="TEST_C_MODEL", interactions=["C"]),
        MDConfig(ensemble="nvt", timestep=0.001, atom_style="atomic"),
        stages,
        stage_dir_prefixes=stage_dirs,
    )

    assert script.count("\nread_data ") == 1
    assert script.count("velocity all create") == 1
    assert "write_data" not in script


def test_stage4_each_stage_has_marker_and_final_snapshot_in_order(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_continuous_stages

    stages = _five_stage_specs(tmp_path)
    stage_dirs = {st.name: f"../{st.name}" for st in stages}
    script = render_continuous_stages(
        KimConfig(model="TEST_C_MODEL", interactions=["C"]),
        MDConfig(ensemble="nvt", timestep=0.001, atom_style="atomic"),
        stages,
        stage_dir_prefixes=stage_dirs,
    )

    previous_index = -1
    for st in stages:
        marker = f"# VITRIFLOW_STAGE: {st.name}"
        final_snapshot = (
            f"write_dump all custom ../{st.name}/{st.name}.final.lammpstrj "
            "id type xu yu zu modify sort id"
        )
        assert script.count(marker) == 1
        assert script.count(final_snapshot) == 1
        idx = script.index(marker)
        assert idx > previous_index
        previous_index = idx


def test_stage4_later_velocity_create_is_rejected_by_yaml_validator():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _validate_schedule

    raw = {
        "custom_schedule": {
            "stages": [
                {"name": "melt", "temperature_K": 1000.0, "steps": 2, "role": "melt", "velocity_mode": "create"},
                {
                    "name": "quench",
                    "temperature_start_K": 1000.0,
                    "temperature_stop_K": 300.0,
                    "steps": 2,
                    "role": "quench",
                    "velocity_mode": "create",
                },
                {"name": "relax", "temperature_K": 300.0, "steps": 2, "role": "relax"},
            ],
            "analysis_roles": {"melt": "melt", "quench": "quench", "relax": "relax"},
        }
    }
    with pytest.raises(ValueError, match="velocity creation on the first stage"):
        _validate_schedule(_schedule_from_raw(raw))


def test_stage4_dft_enabled_custom_schedule_guard_raises():
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner
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
            runner=LammpsRunner(LammpsConfig()),
            force_isotropic=False,
        )


def test_stage4_elastic_enabled_custom_schedule_guard_raises():
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
            enabled=True,
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


def test_stage4_non_lammps_or_discontinuous_custom_schedule_guard_raises():
    from vitriflow.workflows.custom_schedule import _guard_custom_schedule_runner_scope

    with pytest.raises(ValueError, match="engine='lammps'"):
        _guard_custom_schedule_runner_scope(
            SimpleNamespace(engine="cp2k", kim=object(), md=SimpleNamespace(stage_continuity="continuous"))
        )
    with pytest.raises(ValueError, match="LAMMPS potential block"):
        _guard_custom_schedule_runner_scope(
            SimpleNamespace(engine="lammps", kim=None, md=SimpleNamespace(stage_continuity="continuous"))
        )
    with pytest.raises(ValueError, match="stage_continuity: continuous"):
        _guard_custom_schedule_runner_scope(
            SimpleNamespace(engine="lammps", kim=object(), md=SimpleNamespace(stage_continuity="discontinuous"))
        )


def test_stage4_resume_with_changed_schedule_fails_before_mixing_boxes(tmp_path: Path):
    from vitriflow.workflows.custom_schedule import _sha256_canonical_json, _validate_resume_fingerprint_or_raise

    stored_payload = {
        "runner": {"name": "run_custom_schedule", "execution": "continuous_lammps"},
        "custom_schedule": {"stages": [{"name": "quench", "steps": 2}]},
    }
    current_payload = {
        "runner": {"name": "run_custom_schedule", "execution": "continuous_lammps"},
        "custom_schedule": {"stages": [{"name": "quench", "steps": 3}]},
    }
    stored = {"sha256": _sha256_canonical_json(stored_payload), "payload": stored_payload}
    current = {"sha256": _sha256_canonical_json(current_payload), "payload": current_payload}

    with pytest.raises(RuntimeError, match="fingerprint mismatch") as excinfo:
        _validate_resume_fingerprint_or_raise({"resume_fingerprint": stored}, current, outdir=tmp_path)
    assert "custom_schedule.stages[0].steps" in str(excinfo.value)


def test_stage4_docs_and_demo_configs_validate():
    from vitriflow.config import RunConfig
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _schedule_steps, _validate_schedule

    root = Path(__file__).resolve().parents[1]
    cfg_paths = [root / "vitriflow" / "examples" / "hc_C_GAP20Ugr_hc_custom_demo.yaml"]
    cfg_paths.extend(sorted((root / "demos" / "hardcarbon_gap20ugr" / "configs").glob("hc_C_GAP20Ugr_hc_custom_*.yaml")))
    assert len(cfg_paths) >= 3

    for cfg_path in cfg_paths:
        cfg = RunConfig.from_yaml(cfg_path)
        assert cfg.engine == "lammps"
        assert cfg.md.stage_continuity == "continuous"
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        schedule = _schedule_from_raw(raw)
        roles = _validate_schedule(schedule)
        steps = _schedule_steps(schedule, md_use=cfg.md, time_unit_ps=1.0)
        assert roles == {"melt": "graphitisation", "quench": "final_quench", "relax": "relaxation"}
        if "pilot" in cfg_path.name:
            assert steps == {
                "randomisation": 10,
                "prequench": 10,
                "graphitisation": 20,
                "final_quench": 10,
                "relaxation": 10,
            }
        else:
            assert steps["randomisation"] == 10000
            assert steps["prequench"] == 6000
            assert steps["graphitisation"] == 400000
            assert steps["final_quench"] == 20000
            assert steps["relaxation"] == 20000
