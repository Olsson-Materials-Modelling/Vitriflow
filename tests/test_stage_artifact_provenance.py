from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_valid_stage_csvs(stage: Path) -> tuple[Path, Path]:
    thermo = stage / "thermo.csv"
    thermo.write_text(
        "Step,Temp,Press,PotEng,Volume,Density\n"
        "0,300,0.0,-1.0,100.0,2.5\n"
        "2,301,0.1,-0.9,101.0,2.4\n"
    )
    msd = stage / "msd.csv"
    msd.write_text("Step,MSD\n0,0.0\n1,0.1\n2,0.3\n")
    return thermo, msd


def test_stage_manifest_is_atomic_versioned_and_identity_bound(tmp_path: Path) -> None:
    from vitriflow.io.stage_manifest import (
        STAGE_ARTIFACT_MANIFEST_NAME,
        load_stage_artifact_manifest,
        verify_manifest_artifact,
        write_stage_artifact_manifest,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    path = write_stage_artifact_manifest(
        stage,
        engine="lammps",
        lammps_units_style="electron",
        timestep_ps=0.0005,
        thermo_csv=thermo,
        msd_csv=msd,
    )

    assert path == stage / STAGE_ARTIFACT_MANIFEST_NAME
    manifest = load_stage_artifact_manifest(path)
    assert manifest["schema"] == "vitriflow.stage_artifacts.v1"
    assert manifest["schema_version"] == 1
    assert manifest["reporting_contract"] == "vitriflow.canonical_physical_units.v1"
    assert manifest["engine"] == "lammps"
    assert manifest["native_source_units"]["lammps_units_style"] == "electron"
    assert manifest["native_source_units"]["length"] == "bohr"
    assert manifest["canonical_reporting_units"]["pressure"] == "GPa"
    assert manifest["timestep_ps"] == pytest.approx(0.0005)
    assert verify_manifest_artifact(
        stage_dir=stage, manifest=manifest, artifact_key="thermo_csv"
    )
    assert verify_manifest_artifact(
        stage_dir=stage, manifest=manifest, artifact_key="msd_csv"
    )
    assert not list(stage.glob(f".{STAGE_ARTIFACT_MANIFEST_NAME}.*.tmp"))

    thermo.write_text(thermo.read_text() + "4,302,0.2,-0.8,102,2.3\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_manifest_artifact(
            stage_dir=stage, manifest=manifest, artifact_key="thermo_csv"
        )


def test_stage_manifest_rejects_tampered_native_unit_semantics(tmp_path: Path) -> None:
    from vitriflow.io.stage_manifest import (
        load_stage_artifact_manifest,
        write_stage_artifact_manifest,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    path = write_stage_artifact_manifest(
        stage,
        engine="lammps",
        lammps_units_style="electron",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    payload = json.loads(path.read_text())
    payload["native_source_units"]["length"] = "angstrom"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="native source units"):
        load_stage_artifact_manifest(path)


def test_stage_outcome_uses_verified_canonical_artifacts_without_raw_fallback(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from vitriflow.io.stage_manifest import write_stage_artifact_manifest
    from vitriflow.lammps_input import StageSpec
    from vitriflow.workflows.stage_runner import (
        StageArtifacts,
        stage_outcome_from_artifacts,
    )

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    thermo, msd_csv = _write_valid_stage_csvs(stage_dir)
    msd_csv.write_text(
        "Step,MSD\n" + "".join(f"{i},{0.006 * i:.16g}\n" for i in range(5))
    )
    manifest = write_stage_artifact_manifest(
        stage_dir,
        engine="cp2k",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd_csv,
    )
    raw_log = stage_dir / "raw.log"
    raw_log.write_text("this raw fallback must not be read\n")
    raw_msd = stage_dir / "raw.msd"
    raw_msd.write_text("malformed raw fallback\n")
    output = stage_dir / "output.data"
    output.write_text("vitriflow\n\n1 atoms\n")
    artifacts = StageArtifacts(
        stage_dir=stage_dir,
        input_local=stage_dir / "input.data",
        output_local=output,
        log_path=raw_log,
        msd_path=raw_msd,
        dump_path=None,
        neighbor_skin=float("nan"),
        neighbor_skin_retries=0,
        thermo_csv=thermo,
        msd_csv=msd_csv,
        traj_extxyz=None,
        final_extxyz=stage_dir / "final.extxyz",
        engine="cp2k",
        manifest_path=manifest,
    )
    stage = StageSpec(
        name="cp2k",
        input_data=stage_dir / "input.data",
        output_data=Path("output.data"),
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=1000.0,
        equil_steps=0,
        run_steps=4,
        seed=1,
    )

    outcome = stage_outcome_from_artifacts(
        artifacts,
        md_cfg=SimpleNamespace(timestep=1.0),
        stage=stage,
    )
    assert outcome.D == pytest.approx(1.0)
    assert outcome.pressure == pytest.approx(0.1)

    thermo.write_text(thermo.read_text() + "3,302,0.1,-0.8,102,2.3\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        stage_outcome_from_artifacts(
            artifacts,
            md_cfg=SimpleNamespace(timestep=1.0),
            stage=stage,
        )


def test_stage_manifest_does_not_mark_nonempty_malformed_placeholder_as_available(
    tmp_path: Path,
) -> None:
    from vitriflow.io.stage_manifest import (
        load_stage_artifact_manifest,
        write_stage_artifact_manifest,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    msd.write_text("placeholder: MSD parsing failed\n")
    path = write_stage_artifact_manifest(
        stage,
        engine="cp2k",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    record = load_stage_artifact_manifest(path)["artifacts"]["msd_csv"]
    assert record["size_bytes"] > 0
    assert record["available"] is False
    assert record["canonicalized"] is False
    assert len(record["sha256"]) == 64


def test_stage_manifest_finiteness_and_step_validation_preserves_nvt_missing_pressure(
    tmp_path: Path,
) -> None:
    from vitriflow.io.stage_manifest import (
        load_stage_artifact_manifest,
        write_stage_artifact_manifest,
    )

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)

    # CP2K NVT has no pressure samples; finite temperature/energy/volume data
    # remain useful and the canonical thermo artifact is valid.
    thermo.write_text(
        "Step,Temp,Press,PotEng,Volume,Density\n"
        "0,300,nan,-1.0,100.0,2.5\n"
        "2,301,nan,-0.9,101.0,2.4\n"
    )
    path = write_stage_artifact_manifest(
        stage,
        engine="cp2k",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    assert load_stage_artifact_manifest(path)["artifacts"]["thermo_csv"]["available"] is True

    bad_thermo_payloads = [
        # No finite metric evidence beyond Step.
        "Step,Temp,Press\n0,nan,nan\n1,nan,nan\n",
        # Infinite values are not missing-data markers.
        "Step,Temp,Press\n0,300,inf\n1,301,0\n",
        # Step is the physical-time index and cannot run backwards.
        "Step,Temp,Press\n2,300,0\n1,301,0\n",
        # Step is a count: fractional and duplicate coordinates are ambiguous.
        "Step,Temp,Press\n0.5,300,0\n1,301,0\n",
        "Step,Temp,Press\n0,300,0\n0,301,0\n",
    ]
    for payload in bad_thermo_payloads:
        thermo.write_text(payload)
        path = write_stage_artifact_manifest(
            stage,
            engine="cp2k",
            timestep_ps=0.001,
            thermo_csv=thermo,
            msd_csv=msd,
        )
        assert load_stage_artifact_manifest(path)["artifacts"]["thermo_csv"]["available"] is False

    thermo, _ = _write_valid_stage_csvs(stage)
    msd.write_text("Step,MSD\n0,0\n1,nan\n2,0.3\n")
    path = write_stage_artifact_manifest(
        stage,
        engine="cp2k",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    assert load_stage_artifact_manifest(path)["artifacts"]["msd_csv"]["available"] is False

    for payload in (
        "Step,MSD\n0,0\n1,-0.1\n2,0.3\n",
        "Step,MSD\n0,0\n1.5,0.1\n2,0.3\n",
        "Step,MSD\n0,0\n1,0.1\n1,0.3\n",
    ):
        msd.write_text(payload)
        path = write_stage_artifact_manifest(
            stage,
            engine="cp2k",
            timestep_ps=0.001,
            thermo_csv=thermo,
            msd_csv=msd,
        )
        record = load_stage_artifact_manifest(path)["artifacts"]["msd_csv"]
        assert record["available"] is False
        assert record["validation_error"]


def test_plot_stage_prefers_manifest_and_legacy_path_does_not_guess(tmp_path: Path) -> None:
    from vitriflow.io.stage_manifest import write_stage_artifact_manifest
    from vitriflow.plotting import _stage_plot_metadata

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    write_stage_artifact_manifest(
        stage,
        engine="cp2k",
        timestep_ps=0.002,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    conflicting_results = tmp_path / "results.json"
    conflicting_results.write_text(
        json.dumps(
            {
                "units": {"lammps_units": "metal", "time_unit_ps": 1.0},
                "recommendation": {"md": {"timestep": 99.0}},
            }
        )
    )

    metadata = _stage_plot_metadata(stage, results_json=conflicting_results)
    assert metadata["source"] == "stage_manifest"
    assert metadata["canonical_units"] is True
    assert metadata["dt"] == pytest.approx(0.002)
    assert metadata["time_unit_ps"] == pytest.approx(1.0)

    (stage / "stage_artifacts.json").unlink()
    legacy = _stage_plot_metadata(stage, results_json=None)
    assert legacy["source"] == "legacy"
    assert legacy["canonical_units"] is False
    assert legacy["dt"] is None
    assert legacy["time_unit_ps"] is None


def test_plot_stage_manifest_drives_canonical_axis_labels(
    tmp_path: Path, monkeypatch
) -> None:
    from vitriflow.io.stage_manifest import write_stage_artifact_manifest
    from vitriflow.plotting import plot_stage_timeseries

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    write_stage_artifact_manifest(
        stage,
        engine="lammps",
        lammps_units_style="electron",
        timestep_ps=0.0005,
        thermo_csv=thermo,
        msd_csv=msd,
    )
    captured: dict[str, object] = {}

    def capture(fig, _out_path, *, dpi, close=True):
        captured["ylabels"] = [axis.get_ylabel() for axis in fig.axes]
        captured["xlabel"] = fig.axes[-1].get_xlabel()

    monkeypatch.setattr("vitriflow.plotting._style_and_save_figure", capture)
    plot_stage_timeseries(
        stage,
        tmp_path / "unused.pdf",
        thermo_series=["Press"],
        include_msd=True,
        xaxis="time",
    )
    assert captured["ylabels"] == ["Press (GPa)", "MSD (Å²)"]
    assert captured["xlabel"] == "time (ps)"


def test_plot_stage_does_not_swallow_manifest_bound_msd_parse_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from vitriflow.io.stage_manifest import write_stage_artifact_manifest
    from vitriflow.plotting import plot_stage_timeseries

    stage = tmp_path / "stage"
    stage.mkdir()
    thermo, msd = _write_valid_stage_csvs(stage)
    write_stage_artifact_manifest(
        stage,
        engine="cp2k",
        timestep_ps=0.001,
        thermo_csv=thermo,
        msd_csv=msd,
    )

    def fail_strict_parse(_path):
        raise ValueError("synthetic strict parser failure")

    monkeypatch.setattr("vitriflow.io.thermo.parse_msd_csv", fail_strict_parse)
    with pytest.raises(ValueError, match="Manifest-bound MSD artifact failed strict parsing"):
        plot_stage_timeseries(
            stage,
            tmp_path / "unused.pdf",
            thermo_series=["Temp"],
            include_msd=True,
        )


def test_legacy_electron_plot_units_and_diffusion_scaling_are_dimensional() -> None:
    from vitriflow.lammps_units import BOHR_ANGSTROM
    from vitriflow.plotting import (
        _legacy_diffusion_plot_scale,
        _native_length_unit_label,
        _native_msd_unit_label,
        _thermo_unit_label,
    )

    assert _thermo_unit_label("Volume", "electron") == "bohr³"
    assert _thermo_unit_label("Density", "electron") == "amu/bohr³"
    assert _thermo_unit_label("PotEng", "electron") == "hartree"
    assert _thermo_unit_label("Press", "electron") == "Pa"
    assert _native_length_unit_label("electron") == "bohr"
    assert _native_msd_unit_label("electron") == "bohr²"

    electron_scale, label = _legacy_diffusion_plot_scale(
        units_style="electron", time_unit_ps=0.001
    )
    assert electron_scale == pytest.approx(BOHR_ANGSTROM**2 / 0.001)
    assert label == "D (Å²/ps)"

    metal_scale, _ = _legacy_diffusion_plot_scale(
        units_style="metal", time_unit_ps=1.0
    )
    real_scale, _ = _legacy_diffusion_plot_scale(
        units_style="real", time_unit_ps=0.001
    )
    assert metal_scale == pytest.approx(1.0)
    assert real_scale == pytest.approx(1000.0)
