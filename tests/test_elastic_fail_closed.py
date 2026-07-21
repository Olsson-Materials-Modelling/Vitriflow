from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FailingRunner:
    def run(self, *_args, **_kwargs):
        raise RuntimeError("synthetic elastic execution failure")


class _SuccessfulRunner:
    def run(self, *_args, **_kwargs):
        workdir = Path(_args[1])
        (workdir / "born_raw.txt").write_text("synthetic current-run raw\n")
        (workdir / "local_stress.dump").write_text("synthetic current-run dump\n")
        return None


class _ZeroExitWithoutOutputsRunner:
    def run(self, *_args, **_kwargs):
        return None


def _patch_elastic_setup(monkeypatch, elastic_screen) -> None:
    monkeypatch.setattr(
        elastic_screen, "strip_lammps_data_pair_coeff_sections", lambda _path: None
    )
    monkeypatch.setattr(
        elastic_screen, "prepare_potential_files", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        elastic_screen, "render_elastic_screen", lambda *_args, **_kwargs: "run 0\n"
    )


def _cfg(*, enabled, make_plot: bool):
    return SimpleNamespace(
        elastic=SimpleNamespace(
            enabled=enabled,
            make_plot=bool(make_plot),
            strict_when_force_isotropic=True,
        )
    )


def test_explicit_elastic_execution_failure_is_not_returned_as_success(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import elastic_screen

    _patch_elastic_setup(monkeypatch, elastic_screen)
    structure = tmp_path / "input.data"
    structure.write_text("synthetic\n")
    stage_dir = tmp_path / "relax"

    with pytest.raises(RuntimeError, match="Requested elastic screen failed"):
        elastic_screen.run_elastic_screen_lammps(
            _FailingRunner(),
            SimpleNamespace(user_units="metal"),
            SimpleNamespace(atom_style="atomic"),
            structure_data=structure,
            stage_dir=stage_dir,
            metrics_cfg=_cfg(enabled=True, make_plot=False),
        )

    summary = json.loads(
        (stage_dir / "elastic" / "elastic_screen.json").read_text()
    )
    assert summary["status"] == "failed"
    assert "synthetic elastic execution failure" in summary["error"]
    assert summary["plot_status"] == "not_requested"


def test_auto_elastic_execution_failure_is_explicitly_qualified(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import elastic_screen

    _patch_elastic_setup(monkeypatch, elastic_screen)
    structure = tmp_path / "input.data"
    structure.write_text("synthetic\n")

    result = elastic_screen.run_elastic_screen_lammps(
        _FailingRunner(),
        SimpleNamespace(user_units="metal"),
        SimpleNamespace(atom_style="atomic"),
        structure_data=structure,
        stage_dir=tmp_path / "relax",
        metrics_cfg=_cfg(enabled="auto", make_plot=False),
    )

    assert result["status"] == "failed"
    assert result["plot"] is None


def test_zero_exit_elastic_attempt_cannot_reuse_stale_native_outputs(
    monkeypatch, tmp_path: Path
):
    from vitriflow.workflows import elastic_screen

    _patch_elastic_setup(monkeypatch, elastic_screen)
    structure = tmp_path / "input.data"
    structure.write_text("synthetic\n")
    elastic_dir = tmp_path / "relax" / "elastic"
    elastic_dir.mkdir(parents=True)
    raw = elastic_dir / "born_raw.txt"
    dump = elastic_dir / "local_stress.dump"
    raw.write_text("stale raw\n")
    dump.write_text("stale dump\n")

    with pytest.raises(RuntimeError, match="did not create required elastic-screen artifact"):
        elastic_screen.run_elastic_screen_lammps(
            _ZeroExitWithoutOutputsRunner(),
            SimpleNamespace(user_units="metal"),
            SimpleNamespace(atom_style="atomic"),
            structure_data=structure,
            stage_dir=tmp_path / "relax",
            metrics_cfg=_cfg(enabled=True, make_plot=False),
        )

    assert not raw.exists()
    assert not dump.exists()
    summary = json.loads((elastic_dir / "elastic_screen.json").read_text())
    assert summary["status"] == "failed"


def test_explicit_elastic_plot_failure_is_fail_closed(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import elastic_screen

    _patch_elastic_setup(monkeypatch, elastic_screen)
    monkeypatch.setattr(
        elastic_screen,
        "parse_born_stress_raw",
        lambda _path: {
            "born21": np.zeros(21),
            "global_stress_voigt": np.zeros(6),
            "volume": 10.0,
        },
    )
    dump_columns = {
        "id": np.array([1.0]),
        "type": np.array([1.0]),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        **{f"c_pst[{i}]": np.array([0.0]) for i in range(1, 7)},
    }
    monkeypatch.setattr(
        elastic_screen,
        "read_single_custom_dump",
        lambda _path: {"data": dump_columns, "volume": 10.0},
    )
    monkeypatch.setattr(elastic_screen, "count_atoms_in_datafile", lambda _path: 1)
    monkeypatch.setattr(
        elastic_screen,
        "build_elastic_screen_summary",
        lambda **_kwargs: {
            "status": "ok",
            "flags": [],
            "units": {"pressure_native": "bar"},
            "born_matrix_native": np.zeros((6, 6)).tolist(),
        },
    )
    monkeypatch.setattr(elastic_screen, "write_born_matrix_csv", lambda *_a, **_k: None)
    monkeypatch.setattr(elastic_screen, "write_local_stress_csv", lambda *_a, **_k: None)

    plotting_stub = types.ModuleType("vitriflow.plotting")

    def _fail_plot(*_args, **_kwargs):
        raise RuntimeError("synthetic plot failure")

    plotting_stub.plot_elastic_screen = _fail_plot
    monkeypatch.setitem(sys.modules, "vitriflow.plotting", plotting_stub)

    structure = tmp_path / "input.data"
    structure.write_text("synthetic\n")
    stage_dir = tmp_path / "relax"
    with pytest.raises(RuntimeError, match="Requested elastic-screen plot failed"):
        elastic_screen.run_elastic_screen_lammps(
            _SuccessfulRunner(),
            SimpleNamespace(user_units="metal"),
            SimpleNamespace(atom_style="atomic"),
            structure_data=structure,
            stage_dir=stage_dir,
            metrics_cfg=_cfg(enabled=True, make_plot=True),
        )

    summary = json.loads(
        (stage_dir / "elastic" / "elastic_screen.json").read_text()
    )
    assert summary["status"] == "degraded"
    assert summary["plot_status"] == "failed"
    assert "synthetic plot failure" in summary["plot_error"]


def test_explicit_elastic_selection_is_strict_without_force_isotropic():
    from vitriflow.config import LammpsConfig
    from vitriflow.runner import LammpsRunner
    from vitriflow.workflows.elastic_screen import should_run_elastic_screen

    run, strict, _ = should_run_elastic_screen(
        SimpleNamespace(
            elastic=SimpleNamespace(enabled=True, run_on_relax=True)
        ),
        runner=LammpsRunner(LammpsConfig(lammps_cmd="lmp")),
        stage_role="relax",
        force_isotropic=False,
    )
    assert run is True
    assert strict is True
