from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest

# timeseries structure analysis
# skip ase
pytest.importorskip("ase")


def _write_extxyz(path: Path) -> None:
    from vitriflow.analysis.dump import DumpFrame
    from vitriflow.io.extxyz import write_extxyz_frames

    # simple frame cell
    cell = np.eye(3, dtype=float) * 10.0
    fr = DumpFrame(
        timestep=0,
        ids=np.asarray([1, 2], dtype=int),
        types=np.asarray([1, 2], dtype=int),
        positions=np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float),
        cell=cell,
        origin=np.zeros(3, dtype=float),
    )
    write_extxyz_frames(path, [fr], species=["Si", "N"], wrap=True)


def test_metrics_timeseries_stage_dir_resolution_accepts_outdir_prefixed_path(tmp_path: Path, monkeypatch):
    """Metrics timeseries stage."""
    from vitriflow.cli import main

    outdir = tmp_path / "Si3N4-MQ"
    stage = outdir / "production" / "box_001" / "melt"
    stage.mkdir(parents=True)

    _write_extxyz(stage / "traj.extxyz")

    # potential interactions selectors
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """potential:
  kind: mg2_sin
  user_units: metal
  interactions: [Si, N]
md:
  timestep: 0.001
autotune:
  metrics:
    enabled: true
    time_average_frames: 1
    time_average_stride: 1
    pairs:
      - {pair: [Si, N]}
"""
    )

    res = outdir / "autotune_results.json"
    res.write_text(
        json.dumps(
            {
                "production": {
                    "cutoffs": [
                        {"pair": [1, 1], "cutoff": 3.0},
                        {"pair": [1, 2], "cutoff": 2.5},
                        {"pair": [2, 2], "cutoff": 3.0},
                    ]
                }
            }
        )
    )

    out_csv = tmp_path / "mts.csv"

    # relative path
    monkeypatch.chdir(tmp_path)

    main(
        [
            "metrics-timeseries",
            "-c",
            str(cfg),
            "-d",
            "Si3N4-MQ/production/box_001/melt",
            "-o",
            str(out_csv),
            "--results",
            str(res),
        ]
    )

    assert out_csv.exists()
    txt = out_csv.read_text()
    assert "Step" in txt and "time" in txt
    # metric structure metrics
    assert "bondlen_Si-N_mean" in txt


def test_metrics_timeseries_stage_dir_resolution_strips_redundant_prefix(tmp_path: Path, monkeypatch):
    """Metrics timeseries stage."""
    from vitriflow.cli import main

    outdir = tmp_path / "Si3N4-MQ"
    stage = outdir / "production" / "box_001" / "melt"
    stage.mkdir(parents=True)

    _write_extxyz(stage / "traj.extxyz")

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """potential:
  kind: mg2_sin
  user_units: metal
  interactions: [Si, N]
md:
  timestep: 0.001
autotune:
  metrics:
    enabled: true
    time_average_frames: 1
    time_average_stride: 1
    pairs:
      - {pair: [Si, N]}
"""
    )

    res = outdir / "autotune_results.json"
    res.write_text(
        json.dumps(
            {
                "production": {
                    "cutoffs": [
                        {"pair": [1, 1], "cutoff": 3.0},
                        {"pair": [1, 2], "cutoff": 2.5},
                        {"pair": [2, 2], "cutoff": 3.0},
                    ]
                }
            }
        )
    )

    out_csv = tmp_path / "mts2.csv"

    # inside outdir redundantly
    monkeypatch.chdir(outdir)

    main(
        [
            "metrics-timeseries",
            "-c",
            str(cfg),
            "-d",
            "Si3N4-MQ/production/box_001/melt",
            "-o",
            str(out_csv),
            "--results",
            str(res),
        ]
    )

    assert out_csv.exists()
