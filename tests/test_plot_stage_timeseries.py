from __future__ import annotations

import json
from pathlib import Path


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def test_plot_stage_timeseries_writes_file(tmp_path: Path):
    from vitriflow.plotting import plot_stage_timeseries

    stage = tmp_path / "stage"
    stage.mkdir()

    _write_csv(
        stage / "thermo.csv",
        """
        Step,Temp,Press,Density,PotEng,Volume
        0,300,0,2.20,-1000,10000
        100,310,10,2.21,-995,10010
        200,305,5,2.205,-998,10005
        """,
    )
    _write_csv(
        stage / "msd.csv",
        """
        Step,MSD
        0,0.0
        100,0.5
        200,0.9
        """,
    )

    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "units": {"lammps_units": "metal", "time_unit_ps": 1.0},
                "recommendation": {"md": {"timestep": 0.001}},
            }
        )
    )

    out = tmp_path / "stage.pdf"
    plot_stage_timeseries(
        stage,
        out,
        results_json=results,
        thermo_series=["Temp", "Density"],
        include_msd=True,
        xaxis="time",
    )
    assert out.exists()
    assert out.stat().st_size > 1_000
