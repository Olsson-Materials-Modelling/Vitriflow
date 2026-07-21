from __future__ import annotations

import re
from pathlib import Path

import pytest


def _buck_long_commands() -> list[str]:
    return [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0000",
        "kspace_style pppm 1.0e-6",
    ]


def _hybrid_commands() -> list[str]:
    return [
        "pair_style hybrid/overlay coul/long 15.0 buck 15.0 morse 15.0",
        "kspace_style pppm 1.0e-6",
        "pair_coeff 1 1 coul/long",
        "pair_coeff 1 1 buck 139349.01 0.21 171.08",
        "pair_coeff 1 2 coul/long",
        "pair_coeff 1 2 buck 412.55 0.3 0.0",
        "pair_coeff 1 2 morse 0.44 2.57 1.91",
        "pair_coeff 2 2 coul/long",
        "pair_coeff 2 2 buck 1388.77 0.36 175.0",
    ]


def _config(*, commands: list[str] | None = None, hybrid: bool = False):
    from vitriflow.config import RunConfig

    species = ["Ga", "O"] if hybrid else ["Si", "O"]
    charges = {"Ga": 1.8, "O": -1.2} if hybrid else {"Si": 2.4, "O": -1.2}
    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": species,
                "commands": commands or _buck_long_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "table_points": 3000,
                    "table_points_max": 3000,
                    "table_verify_points": 2001,
                    "table_filename": "safe_core.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.001, 0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": charges,
                "generate": {
                    "method": "random",
                    "formula": "Ga2O3" if hybrid else "SiO2",
                    "n_formula_units": 1,
                },
            },
            "md": {
                "atom_style": "charge",
                "timestep": 0.001,
                "neighbor_skin_autotune": False,
            },
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )


def _write_charge_data(path: Path, *, hybrid: bool = False) -> None:
    q1, q2 = ((1.8, -1.2) if hybrid else (2.4, -1.2))
    path.write_text(
        f"""LAMMPS data file via autocore ordering test

2 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 28.0855
2 15.999

Atoms # charge

1 1 {q1:.16g} 1.0 1.0 1.0
2 2 {q2:.16g} 3.0 3.0 3.0
"""
    )


def test_original_potential_gewald_probe_is_strictly_run_zero(tmp_path: Path) -> None:
    from vitriflow.workflows.preflight import _probe_original_potential_gewald

    cfg = _config()
    input_data = tmp_path / "input.data"
    _write_charge_data(input_data)
    scripts: list[str] = []

    class FakeRunner:
        def run(self, script, workdir, log_name, timeout_sec=None):
            scripts.append(str(script))
            (Path(workdir) / log_name).write_text(
                "LAMMPS test\nG vector (1/distance) = 0.224358\n"
            )

    value = _probe_original_potential_gewald(
        FakeRunner(),
        cfg,
        input_data,
        outdir=tmp_path,
        potential_lines=_buck_long_commands(),
    )

    assert value == pytest.approx(0.224358)
    assert len(scripts) == 1
    script = scripts[0]
    assert len(re.findall(r"(?m)^\s*run\s+0\s*$", script)) == 1
    for forbidden in ("minimize", "velocity", "timestep", "fix nve", "fix nvt", "write_data"):
        assert forbidden not in script.lower()
    report = (tmp_path / "preflight" / "core_kspace_probe" / "gewald_probe.json").read_text()
    assert '"dynamics_performed": false' in report


def test_source_equivalence_pairwrite_never_samples_below_any_outer_join(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from vitriflow.potential import (
        _parse_tabulated_core_spec,
        build_tabulated_buckingham_core_lines,
    )
    from vitriflow.workflows.preflight import (
        _audit_original_potential_above_core_joins,
        _render_pair_write_script,
        _source_equivalence_component_references,
    )

    cfg = _config()
    lines = build_tabulated_buckingham_core_lines(
        _buck_long_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="safe_core.table",
        table_r_min=0.1,
        charges={"Si": 2.4, "O": -1.2},
        gewald=0.224358,
        has_bonded_topology=False,
        table_style="spline",
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None
    smallest_join = min(float(pair["r_out"]) for pair in spec["pairs"])

    def fake_pair_write(*args, **kwargs):
        audit_spec = kwargs["spec"]
        assert float(audit_spec["r_min"]) == pytest.approx(smallest_join)
        assert all(
            float(pair["source_audit_r_min"]) == pytest.approx(float(pair["r_out"]))
            for pair in audit_spec["pairs"]
        )
        script = _render_pair_write_script(
            cfg,
            potential_lines=kwargs["potential_lines"],
            spec=audit_spec,
            npoints=kwargs["npoints"],
            output_name=kwargs["output_name"],
        )
        pair_write_lines = [line for line in script.splitlines() if line.startswith("pair_write ")]
        assert len(pair_write_lines) == 3
        # Every Coulombic source probe explicitly supplies Qi and Qj.
        assert all(len(line.split()) == 11 for line in pair_write_lines)
        sections, _noncoul, _coul, _scale = _source_equivalence_component_references(
            audit_spec,
            npoints=kwargs["npoints"],
        )
        return {
            "path": Path(kwargs["stage_dir"]) / kwargs["output_name"],
            "sections": sections,
            "warnings": [],
        }

    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves",
        fake_pair_write,
    )
    report = _audit_original_potential_above_core_joins(
        object(),
        cfg,
        outdir=tmp_path,
        source_potential_lines=_buck_long_commands(),
        spec=spec,
    )
    assert report["passed"] is True
    assert report["r_min"] == pytest.approx(smallest_join)
    assert report["r_min_by_section"] == pytest.approx(
        {str(pair["section"]): float(pair["r_out"]) for pair in spec["pairs"]}
    )
    assert report["dynamics_performed"] is False


def test_autocore_order_is_probe_build_audit_materialize_verify_then_table_dynamics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vitriflow.workflows.preflight as preflight
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    cfg = _config()
    input_data = tmp_path / "input.data"
    _write_charge_data(input_data)
    events: list[str] = []
    original_builder = preflight.build_tabulated_buckingham_core_lines

    monkeypatch.setattr(preflight, "_read_nn_median_from_datafile", lambda *a, **k: 2.0)
    monkeypatch.setattr(
        preflight,
        "_probe_original_potential_gewald",
        lambda *a, **k: events.append("probe") or 0.224358,
    )

    def recording_builder(*args, **kwargs):
        events.append("build")
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(preflight, "build_tabulated_buckingham_core_lines", recording_builder)
    monkeypatch.setattr(
        preflight,
        "_audit_original_potential_above_core_joins",
        lambda *a, **k: events.append("source_audit")
        or {"passed": True, "comparison": {"overall": {}}},
    )

    def fake_materialize(*, outdir, spec):
        events.append("materialize")
        path = Path(outdir) / "preflight" / "potential_override" / spec["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bounded table fixture\n")
        return path

    monkeypatch.setattr(preflight, "_materialize_generated_tabulated_core_source", fake_materialize)
    monkeypatch.setattr(
        preflight,
        "_verify_tabulated_core_against_source",
        lambda *a, **k: events.append("verify")
        or {
            "passed": True,
            "warnings": [],
            "comparison": {"overall": {"max_energy_ratio": 0.0, "max_force_ratio": 0.0}},
            "self_consistency": {"overall": {"max_force_ratio": 0.0}},
        },
    )

    stability_blocks: list[str] = []

    def fake_stability(*args, potential_lines, timestep, **kwargs):
        text = "\n".join(str(line) for line in potential_lines)
        stability_blocks.append(text)
        events.append(f"stability:{float(timestep):g}")
        return float(timestep) == pytest.approx(0.0005)

    monkeypatch.setattr(preflight, "_run_stability_test", fake_stability)

    lines, result, selected_dt = preflight._maybe_apply_core_repulsion(
        LammpsRunner(LammpsConfig()),
        cfg,
        input_data,
        tmp_path,
        T_test=4000.0,
    )

    assert result.success is True
    assert selected_dt == pytest.approx(0.0005)
    assert lines is not None
    assert events == [
        "probe",
        "build",
        "source_audit",
        "build",
        "materialize",
        "verify",
        "stability:0.001",
        "stability:0.0005",
    ]
    assert stability_blocks
    assert all("pair_style table" in block for block in stability_blocks)
    assert all("pair_style hybrid/overlay" not in block for block in stability_blocks)
    assert all(" zbl " not in block for block in stability_blocks)


def test_missing_gewald_probe_result_fails_before_build_or_dynamics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vitriflow.workflows.preflight as preflight
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    cfg = _config()
    input_data = tmp_path / "input.data"
    _write_charge_data(input_data)
    calls = {"build": 0, "stability": 0}
    monkeypatch.setattr(preflight, "_read_nn_median_from_datafile", lambda *a, **k: 2.0)
    monkeypatch.setattr(
        preflight,
        "_probe_original_potential_gewald",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no G vector in log")),
    )
    monkeypatch.setattr(
        preflight,
        "build_tabulated_buckingham_core_lines",
        lambda *a, **k: calls.__setitem__("build", calls["build"] + 1),
    )
    monkeypatch.setattr(
        preflight,
        "_run_stability_test",
        lambda *a, **k: calls.__setitem__("stability", calls["stability"] + 1),
    )

    with pytest.raises(preflight.PreflightError, match="run-0 probe"):
        preflight._maybe_apply_core_repulsion(
            LammpsRunner(LammpsConfig()),
            cfg,
            input_data,
            tmp_path,
            T_test=4000.0,
        )
    assert calls == {"build": 0, "stability": 0}


def test_hybrid_overlay_buckingham_is_detected_and_parser_failure_is_fail_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vitriflow.workflows.preflight as preflight
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    commands = _hybrid_commands()
    assert preflight._potential_contains_buckingham(commands) is True
    assert preflight._potential_contains_coulomb(commands) is True
    assert preflight._potential_requires_gewald(commands) is True

    cfg = _config(commands=commands, hybrid=True)
    input_data = tmp_path / "input.data"
    _write_charge_data(input_data, hybrid=True)
    stability_calls = 0
    monkeypatch.setattr(preflight, "_read_nn_median_from_datafile", lambda *a, **k: 2.0)
    monkeypatch.setattr(preflight, "_probe_original_potential_gewald", lambda *a, **k: 0.224358)
    monkeypatch.setattr(
        preflight,
        "build_tabulated_buckingham_core_lines",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("unsupported additive fixture")),
    )

    def no_stability(*args, **kwargs):
        nonlocal stability_calls
        stability_calls += 1
        return True

    monkeypatch.setattr(preflight, "_run_stability_test", no_stability)
    with pytest.raises(preflight.PreflightError, match="table initialization failed"):
        preflight._maybe_apply_core_repulsion(
            LammpsRunner(LammpsConfig()),
            cfg,
            input_data,
            tmp_path,
            T_test=4000.0,
        )
    assert stability_calls == 0


def test_multiple_pair_style_block_containing_buckingham_fails_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import vitriflow.workflows.preflight as preflight
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner

    commands = ["pair_style zero 10.0", *_buck_long_commands()]
    cfg = _config(commands=commands)
    calls = {"nearest_neighbour": 0, "probe": 0, "stability": 0}
    monkeypatch.setattr(
        preflight,
        "_read_nn_median_from_datafile",
        lambda *a, **k: calls.__setitem__(
            "nearest_neighbour", calls["nearest_neighbour"] + 1
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_probe_original_potential_gewald",
        lambda *a, **k: calls.__setitem__("probe", calls["probe"] + 1),
    )
    monkeypatch.setattr(
        preflight,
        "_run_stability_test",
        lambda *a, **k: calls.__setitem__("stability", calls["stability"] + 1),
    )

    with pytest.raises(
        preflight.PreflightError,
        match="compatibility audit failed before any potential execution",
    ):
        preflight._maybe_apply_core_repulsion(
            LammpsRunner(LammpsConfig()),
            cfg,
            tmp_path / "not-read.data",
            tmp_path,
            T_test=4000.0,
        )
    assert calls == {"nearest_neighbour": 0, "probe": 0, "stability": 0}
