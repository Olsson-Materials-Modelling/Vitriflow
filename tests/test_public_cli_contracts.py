from __future__ import annotations

import json
from pathlib import Path

import pytest

from vitriflow.cli import (
    _metrics_timeseries_cutoffs_from_results,
    _positive_int,
    _result_exit_code,
    _validate_metrics_timeseries_stage_contract,
)
from vitriflow.config import StructureMetricsConfig
from vitriflow.io.stage_manifest import build_stage_artifact_manifest


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ok", 0),
        ("planned", 0),
        ("incomplete", 2),
        ("not_converged", 2),
        ("error", 1),
        ("failed", 1),
        ("running", 1),
        ("unknown", 1),
    ],
)
def test_public_result_exit_policy(status: str, expected: int) -> None:
    assert _result_exit_code({"status": status}) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_positive_integer_cli_overrides_reject_invalid_values(value: str) -> None:
    with pytest.raises(Exception):
        _positive_int(value)


def _metrics_config() -> StructureMetricsConfig:
    return StructureMetricsConfig.model_validate(
        {
            "enabled": True,
            "type_to_species": ["Si", "O"],
            "pairs": [{"pair": ["Si", "O"]}],
        }
    )


def _results_payload(metrics: StructureMetricsConfig) -> dict:
    cutoffs = [{"pair": [1, 2], "cutoff": 2.25}]
    return {
        "status": "ok",
        "units": {
            "engine": "lammps",
            "lammps_units": "metal",
            "time_unit_ps": 1.0,
        },
        "production": {"cutoffs": json.loads(json.dumps(cutoffs))},
        "production_plan": {
            "engine": "lammps",
            "time_unit_ps": 1.0,
            "md_use": {"timestep": 0.001},
            "potential_config": {"user_units": "metal"},
            "metrics_cfg": metrics.model_dump(mode="json"),
            "type_to_species": ["Si", "O"],
            "preferred_cutoffs": json.loads(json.dumps(cutoffs)),
        },
    }


def test_metrics_timeseries_results_identity_and_cutoff_coverage(tmp_path: Path) -> None:
    metrics = _metrics_config()
    result_path = tmp_path / "autotune_results.json"
    stage = tmp_path / "production" / "box_001" / "relax"
    stage.mkdir(parents=True)
    payload = _results_payload(metrics)
    result_path.write_text(json.dumps(payload))

    cutoffs, timestep_ps = _metrics_timeseries_cutoffs_from_results(
        payload,
        results_path=result_path,
        stage_dir=stage,
        metrics_cfg=metrics,
        timestep_ps=0.001,
        type_to_species=["Si", "O"],
        units_style="metal",
        engine="lammps",
    )

    assert cutoffs == {(1, 2): 2.25}
    assert timestep_ps == pytest.approx(0.001)


def test_metrics_timeseries_uses_preflight_selected_results_timestep(
    tmp_path: Path,
) -> None:
    metrics = _metrics_config()
    payload = _results_payload(metrics)
    result_path = tmp_path / "autotune_results.json"
    stage = tmp_path / "production" / "box_001" / "relax"
    stage.mkdir(parents=True)

    _cutoffs, timestep_ps = _metrics_timeseries_cutoffs_from_results(
        payload,
        results_path=result_path,
        stage_dir=stage,
        metrics_cfg=metrics,
        # Source YAML requested 2 fs-equivalent; preflight selected 1 fs.
        timestep_ps=0.002,
        type_to_species=["Si", "O"],
        units_style="metal",
        engine="lammps",
    )

    assert timestep_ps == pytest.approx(0.001)


def test_metrics_timeseries_rejects_config_and_results_identity_mismatch(
    tmp_path: Path,
) -> None:
    metrics = _metrics_config()
    payload = _results_payload(metrics)
    payload["production_plan"]["metrics_cfg"]["pairs"] = []
    result_path = tmp_path / "autotune_results.json"
    stage = tmp_path / "production" / "box_001" / "relax"
    stage.mkdir(parents=True)

    with pytest.raises(ValueError, match="metrics_cfg identity mismatch"):
        _metrics_timeseries_cutoffs_from_results(
            payload,
            results_path=result_path,
            stage_dir=stage,
            metrics_cfg=metrics,
            timestep_ps=0.001,
            type_to_species=["Si", "O"],
            units_style="metal",
            engine="lammps",
        )


def test_metrics_timeseries_rejects_conflicting_or_incomplete_cutoffs(
    tmp_path: Path,
) -> None:
    metrics = _metrics_config()
    payload = _results_payload(metrics)
    payload["production"]["cutoffs"][0]["cutoff"] = 2.5
    result_path = tmp_path / "autotune_results.json"
    stage = tmp_path / "production" / "box_001" / "relax"
    stage.mkdir(parents=True)

    with pytest.raises(ValueError, match="conflicting cutoff identities"):
        _metrics_timeseries_cutoffs_from_results(
            payload,
            results_path=result_path,
            stage_dir=stage,
            metrics_cfg=metrics,
            timestep_ps=0.001,
            type_to_species=["Si", "O"],
            units_style="metal",
            engine="lammps",
        )

    payload = _results_payload(metrics)
    payload["production"]["cutoffs"] = []
    payload["production_plan"]["preferred_cutoffs"] = []
    with pytest.raises(ValueError, match="do not cover required metric pairs"):
        _metrics_timeseries_cutoffs_from_results(
            payload,
            results_path=result_path,
            stage_dir=stage,
            metrics_cfg=metrics,
            timestep_ps=0.001,
            type_to_species=["Si", "O"],
            units_style="metal",
            engine="lammps",
        )


def test_metrics_timeseries_validates_stage_manifest_time_and_units(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "relax"
    stage.mkdir()
    manifest = build_stage_artifact_manifest(
        engine="lammps",
        timestep_ps=0.002,
        thermo_csv=stage / "thermo.csv",
        msd_csv=stage / "msd.csv",
        lammps_units_style="metal",
    )
    (stage / "stage_artifacts.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="timestep mismatch"):
        _validate_metrics_timeseries_stage_contract(
            stage,
            engine="lammps",
            timestep_ps=0.001,
            units_style="metal",
        )
