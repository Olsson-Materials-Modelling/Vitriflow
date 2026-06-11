from __future__ import annotations

from pathlib import Path


def _base_buck_commands() -> list[str]:
    return [
        "pair_style buck/coul/long 10.0",
        "pair_coeff 1 1 0.0 1.0 0.0",
        "pair_coeff 1 2 18003.7572 0.205205 133.5381",
        "pair_coeff 2 2 1388.7730 0.362319 175.0000",
        "pair_modify shift yes",
        "kspace_style pppm 1.0e-5",
    ]


def _charges() -> dict[str, float]:
    return {"Si": 2.4, "O": -1.2}


def test_build_tabulated_buckingham_core_lines_and_materialize_table(tmp_path: Path) -> None:
    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files

    pot_cfg = LammpsPotentialConfig(
        interactions=["Si", "O"],
        commands=_base_buck_commands(),
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=4000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )

    joined = "\n".join(lines)
    assert lines[0].startswith("# vitriflow_core_table_begin ")
    assert "pair_style table linear 4000 pppm" in joined
    assert "pair_coeff 1 2 buck_core.table P1_2 10" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_modify pair coul/long compute no" not in joined
    assert "kspace_modify gewald 0.224358" in joined
    assert "kspace_style pppm 1.0e-5" in joined
    assert "pair_modify shift yes" not in joined

    prepare_potential_files(pot_cfg, tmp_path, lines)
    table_path = tmp_path / "buck_core.table"
    assert table_path.exists()
    text = table_path.read_text()
    assert "P1_1" in text
    assert "P1_2" in text
    assert "P2_2" in text
    assert "N 4000 RSQ 0.1 10 FPRIME " in text
    sec = text.split("P1_1", 1)[1].split("P1_2", 1)[0]
    rows = []
    for ln in sec.splitlines():
        toks = ln.split()
        if len(toks) == 4 and toks[0].isdigit():
            rows.append((float(toks[1]), float(toks[2]), float(toks[3])))
    assert rows
    assert any((r > 1.5 and abs(u) > 1.0e-6) for r, u, _f in rows)


def test_core_repulsion_tabulate_requires_zbl_style() -> None:
    from pydantic import ValidationError

    from vitriflow.config import RunConfig

    try:
        RunConfig.model_validate(
            {
                "potential": {
                    "kind": "lammps",
                    "user_units": "metal",
                    "interactions": ["Si", "O"],
                    "commands": _base_buck_commands(),
                    "core_repulsion": {
                        "enabled": True,
                        "style": "lj_repulsive",
                        "tabulate": True,
                    },
                },
                "structure": {"generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1}},
            }
        )
    except ValidationError as exc:
        assert "tabulate currently supports only style='zbl'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected validation failure for lj_repulsive tabulation")


def test_preflight_core_repulsion_tabulate_returns_table_override(monkeypatch, tmp_path: Path) -> None:
    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.potential import _parse_tabulated_core_spec
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_points": 3000,
                    "table_points_max": 3000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )

    input_data = tmp_path / "input.data"
    input_data.write_text("LAMMPS data file\n\n0 atoms\n")

    calls: list[str] = []

    monkeypatch.setattr("vitriflow.workflows.preflight._read_nn_median_from_datafile", lambda *a, **k: 2.0)

    def _fake_stability_test(runner, config, input_data, *, outdir, potential_lines, temperature, timestep, label=""):
        text = "\n".join(str(x) for x in potential_lines)
        calls.append(text)
        return True

    def _fake_pair_write(*args, **kwargs):
        import numpy as np

        spec = kwargs["spec"]
        sections = {}
        for pair in spec["pairs"]:
            sections[pair["section"]] = {
                "r": np.asarray([0.1, float(pair["pair_cutoff"])], dtype=float),
                "energy": np.asarray([1.0, 0.0], dtype=float),
                "force": np.asarray([1.0, 0.0], dtype=float),
            }
        return {"path": Path(kwargs["stage_dir"]) / kwargs["output_name"], "sections": sections, "warnings": []}

    monkeypatch.setattr("vitriflow.workflows.preflight._run_stability_test", _fake_stability_test)
    monkeypatch.setattr("vitriflow.workflows.preflight._pair_write_potential_curves", _fake_pair_write)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *a, **k: {"passed": True, "warnings": [], "comparison": {"overall": {"max_energy_ratio": 0.0, "max_force_ratio": 0.0}}},
    )

    potential_lines, core_res, dt_sel = _maybe_apply_core_repulsion(
        LammpsRunner(LammpsConfig()),
        cfg,
        input_data,
        tmp_path,
        T_test=4000.0,
    )

    assert dt_sel == 0.0005
    assert potential_lines is not None
    joined = "\n".join(potential_lines)
    spec = _parse_tabulated_core_spec(potential_lines)
    assert spec is not None
    assert calls[0].count("pair_style") == 1 and " zbl " in calls[0]
    assert calls[1].count("pair_style") == 1 and "pair_style table linear 3000 pppm" in calls[1]
    assert joined.startswith("# vitriflow_core_table_begin ")
    assert "pair_coeff 1 2 buck_zbl.table P1_2 10" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_modify pair coul/long compute no" not in joined
    assert "kspace_modify gewald 0.224358" in joined
    assert spec.get("generated_by") == "vitriflow_generated"
    assert spec.get("force_mode") == "analytic"
    assert bool(spec.get("include_fprime", False)) is True
    assert isinstance(spec.get("sha256"), str) and len(spec["sha256"]) == 64
    assert core_res.success is True
    assert "accepted after refinement" in core_res.note




def test_preflight_core_repulsion_tabulate_falls_back_to_analytic_with_summary(monkeypatch, tmp_path: Path) -> None:
    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.preflight import _maybe_apply_core_repulsion

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_points": 3000,
                    "table_points_max": 6000,
                    "table_filename": "buck_zbl.table",
                    "table_r_min": 0.1,
                    "table_gewald": 0.224358,
                    "dt_candidates": [0.0005],
                    "r_out_factor": 0.5,
                    "r_out_min": 0.6,
                    "r_out_max": 1.6,
                    "r_in_factor": 0.8,
                    "test_run_steps": 10,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge", "timestep": 0.0005},
            "autotune": {
                "preflight": {"enabled": True},
                "tm_scan": {"equil_steps": 10, "sample_steps": 10},
            },
        }
    )

    input_data = tmp_path / "input.data"
    input_data.write_text("LAMMPS data file\n\n0 atoms\n")

    monkeypatch.setattr("vitriflow.workflows.preflight._read_nn_median_from_datafile", lambda *a, **k: 2.0)

    def _fake_stability_test(runner, config, input_data, *, outdir, potential_lines, temperature, timestep, label=""):
        return "pair_style hybrid/overlay" in "\n".join(str(x) for x in potential_lines)

    def _fake_pair_write(*args, **kwargs):
        import numpy as np

        spec = kwargs["spec"]
        sections = {}
        for pair in spec["pairs"]:
            sections[pair["section"]] = {
                "r": np.asarray([0.1, float(pair["pair_cutoff"])], dtype=float),
                "energy": np.asarray([1.0, 0.0], dtype=float),
                "force": np.asarray([1.0, 0.0], dtype=float),
            }
        return {"path": Path(kwargs["stage_dir"]) / kwargs["output_name"], "sections": sections, "warnings": []}

    monkeypatch.setattr("vitriflow.workflows.preflight._run_stability_test", _fake_stability_test)
    monkeypatch.setattr("vitriflow.workflows.preflight._pair_write_potential_curves", _fake_pair_write)
    monkeypatch.setattr(
        "vitriflow.workflows.preflight._verify_tabulated_core_against_source",
        lambda *a, **k: {
            "passed": False,
            "warnings": ["WARNING: force values in table are inconsistent with -dE/dr."],
            "comparison": {"overall": {"max_energy_ratio": 9.0, "max_force_ratio": 11.0}},
            "self_consistency": {"overall": {"max_force_ratio": 25.0}},
        },
    )

    potential_lines, core_res, dt_sel = _maybe_apply_core_repulsion(
        LammpsRunner(LammpsConfig()),
        cfg,
        input_data,
        tmp_path,
        T_test=4000.0,
    )

    assert dt_sel == 0.0005
    assert potential_lines is not None
    joined = "\n".join(potential_lines)
    assert "pair_style hybrid/overlay" in joined
    assert "pair_style table linear" not in joined
    assert core_res.success is True
    assert "fell back to analytic hybrid/overlay core" in core_res.note
    summary_txt = tmp_path / "preflight" / "table_verify" / "refinement_summary.txt"
    summary_json = tmp_path / "preflight" / "table_verify" / "refinement_report.json"
    assert summary_txt.exists()
    assert summary_json.exists()
    log_text = (tmp_path / "condensed.log").read_text()
    assert "falling back to analytic hybrid/overlay core" in log_text

def test_build_tabulated_buckingham_core_lines_assigns_coul_cut_substyle() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    cmds = [
        "pair_style buck/coul/cut 8.0",
        "pair_coeff 1 1 100.0 0.3 0.0",
        "pair_coeff 1 2 200.0 0.25 10.0",
        "pair_coeff 2 2 300.0 0.2 20.0",
    ]

    lines = build_tabulated_buckingham_core_lines(
        cmds,
        species=["Na", "Cl"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges={"Na": 1.0, "Cl": -1.0},
    )

    joined = "\n".join(lines)
    assert "pair_style table linear 3000" in joined
    assert "pair_style hybrid/overlay" not in joined
    assert "pair_coeff * * coul/cut" not in joined


def test_build_tabulated_buckingham_core_lines_requires_fixed_charges_for_coulombic_styles() -> None:
    from vitriflow.potential import build_tabulated_buckingham_core_lines

    try:
        build_tabulated_buckingham_core_lines(
            _base_buck_commands(),
            species=["Si", "O"],
            units_style="metal",
            r_in=0.8,
            r_out=1.0,
            table_points=3000,
            table_filename="buck_core.table",
            table_r_min=0.1,
            gewald=0.224358,
        )
    except ValueError as exc:
        assert "requires fixed species charges" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing-charge failure for tabulated buck/coul/long")


def test_parse_gewald_from_modern_lammps_log_format(tmp_path: Path) -> None:
    from vitriflow.workflows.preflight import _parse_gewald_from_log

    log = tmp_path / "log.lammps"
    log.write_text(
        """
PPPM initialization ...
  using 12-bit tables for long-range coulomb
  G vector (1/distance) = 0.278015
  grid = 96 90 128
"""
    )

    val = _parse_gewald_from_log(log)
    assert val is not None
    assert abs(val - 0.278015) < 1.0e-12


def test_prepare_potential_files_prefers_validated_pairwrite_source(monkeypatch, tmp_path: Path) -> None:
    import hashlib

    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files, update_tabulated_core_metadata_lines

    pot_cfg = LammpsPotentialConfig(interactions=["Si", "O"], commands=_base_buck_commands())
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )
    preflight_dir = tmp_path / "preflight" / "potential_override"
    preflight_dir.mkdir(parents=True)
    src = preflight_dir / "buck_core.table"
    src.write_text("validated table content\n")
    lines = update_tabulated_core_metadata_lines(
        lines,
        generated_by="lammps_pair_write",
        sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("fallback writer should not be used when validated source exists")

    monkeypatch.setattr("vitriflow.potential.write_tabulated_buckingham_core_table", _boom)
    stage_dir = tmp_path / "stages" / "prod_box_001" / "stage1"
    prepare_potential_files(pot_cfg, stage_dir, lines)
    copied = stage_dir / "buck_core.table"
    assert copied.exists()
    assert copied.read_text() == "validated table content\n"


def test_prepare_potential_files_requires_validated_pairwrite_source_when_declared(tmp_path: Path) -> None:
    from vitriflow.config import LammpsPotentialConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, prepare_potential_files, update_tabulated_core_metadata_lines

    pot_cfg = LammpsPotentialConfig(interactions=["Si", "O"], commands=_base_buck_commands())
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )
    lines = update_tabulated_core_metadata_lines(lines, generated_by="lammps_pair_write", sha256="0" * 64)

    try:
        prepare_potential_files(pot_cfg, tmp_path / "stage", lines)
    except FileNotFoundError as exc:
        assert "validated tabulated-core source file not found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing validated-source failure")


def test_render_pair_write_script_injects_explicit_gewald_for_long_range_tabulation() -> None:
    from vitriflow.config import RunConfig
    from vitriflow.potential import build_tabulated_buckingham_core_lines, _parse_tabulated_core_spec
    from vitriflow.workflows.preflight import _render_pair_write_script

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_gewald": 0.224358,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge"},
        }
    )

    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None

    script = _render_pair_write_script(
        cfg,
        potential_lines=[
            "pair_style hybrid/overlay buck/coul/long 10.0 zbl 0.8 1.0",
            "pair_coeff 1 1 buck/coul/long 0.0 1.0 0.0",
            "pair_coeff 1 2 buck/coul/long 18003.7572 0.205205 133.5381",
            "pair_coeff 2 2 buck/coul/long 1388.7730 0.362319 175.0000",
            "pair_coeff 1 1 zbl 14 14",
            "pair_coeff 1 2 zbl 14 8",
            "pair_coeff 2 2 zbl 8 8",
            "kspace_style pppm 1.0e-5",
        ],
        spec=spec,
        npoints=1001,
        output_name="reference.table",
    )

    assert "kspace_modify gewald 0.224358" in script
    assert "pair_write 1 1 1001 rsq 0.1 10 reference.table P1_1 2.4 2.4" in script


def test_verify_tabulated_core_report_includes_force_energy_self_consistency(monkeypatch, tmp_path: Path) -> None:
    import json
    import numpy as np

    from vitriflow.config import LammpsConfig, RunConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.potential import build_tabulated_buckingham_core_lines, _parse_tabulated_core_spec
    from vitriflow.workflows.preflight import _verify_tabulated_core_against_source

    cfg = RunConfig.model_validate(
        {
            "potential": {
                "kind": "lammps",
                "user_units": "metal",
                "interactions": ["Si", "O"],
                "commands": _base_buck_commands(),
                "core_repulsion": {
                    "enabled": True,
                    "style": "zbl",
                    "tabulate": True,
                    "table_gewald": 0.224358,
                    "table_verify_points": 2001,
                },
            },
            "structure": {
                "charges": _charges(),
                "generate": {"method": "random", "formula": "SiO2", "n_formula_units": 1},
            },
            "md": {"atom_style": "charge"},
            "autotune": {"preflight": {"enabled": True}, "tm_scan": {"equil_steps": 10, "sample_steps": 10}},
        }
    )
    lines = build_tabulated_buckingham_core_lines(
        _base_buck_commands(),
        species=["Si", "O"],
        units_style="metal",
        r_in=0.8,
        r_out=1.0,
        table_points=3000,
        table_filename="buck_core.table",
        table_r_min=0.1,
        charges=_charges(),
        gewald=0.224358,
    )
    spec = _parse_tabulated_core_spec(lines)
    assert spec is not None

    sections = {}
    for pair in spec["pairs"]:
        r = np.asarray([0.1, 2.0, float(pair["pair_cutoff"])], dtype=float)
        u = np.asarray([4.0, 1.0, 0.0], dtype=float)
        f = -np.gradient(u, r, edge_order=2)
        sections[pair["section"]] = {"r": r, "energy": u, "force": f}

    monkeypatch.setattr(
        "vitriflow.workflows.preflight._pair_write_potential_curves",
        lambda *a, **k: {"path": Path(k["stage_dir"]) / k["output_name"], "sections": sections, "warnings": []},
    )

    report = _verify_tabulated_core_against_source(
        LammpsRunner(LammpsConfig()),
        cfg,
        outdir=tmp_path,
        table_potential_lines=lines,
        reference_sections=sections,
        spec=spec,
    )

    assert report["passed"] is True
    assert report["self_consistency"]["overall"]["max_force_ratio"] <= 1.0
    saved = json.loads((tmp_path / "preflight" / "table_verify" / "verification_report.json").read_text())
    assert "self_consistency" in saved
