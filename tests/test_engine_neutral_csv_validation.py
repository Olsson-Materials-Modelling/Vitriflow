from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitriflow.io.thermo import (
    parse_msd_csv,
    parse_thermo_csv,
    write_msd_csv,
    write_thermo_csv,
)
from vitriflow.parse import ThermoTable


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("Step,Temp\n0,300\n1\n", "Ragged"),
        ("Step,Temp\n0,300,extra\n1,301\n", "Ragged"),
        ("Step,Temp\n0,300\n1,bad\n", "Non-numeric"),
        ("Step,Temp\n0.5,300\n1,301\n", "integer counts"),
        ("Step,Temp\n0,300\n0,301\n", "strictly increasing"),
        ("Step,Temp\n1,300\n0,301\n", "strictly increasing"),
        ("Step,Temp\n-1,300\n0,301\n", "nonnegative"),
        ("Step,Temp\n0,inf\n1,301\n", "infinite"),
        ("Step,Temp,Press\n0,nan,nan\n1,301,nan\n", "at least one finite metric"),
        ("Step,Step,Temp\n0,0,300\n1,1,301\n", "duplicate"),
    ],
)
def test_parse_thermo_csv_rejects_malformed_authoritative_rows(
    tmp_path: Path,
    payload: str,
    reason: str,
) -> None:
    path = tmp_path / "thermo.csv"
    path.write_text(payload)

    with pytest.raises(ValueError, match=reason):
        parse_thermo_csv(path)


def test_parse_thermo_csv_preserves_nan_missing_metric_but_legacy_is_opt_in(
    tmp_path: Path,
) -> None:
    path = tmp_path / "thermo.csv"
    path.write_text(
        "Step,Temp,Press\n"
        "0,300,nan\n"
        "2,301,nan\n"
    )
    table = parse_thermo_csv(path)
    assert table.columns == ["Step", "Temp", "Press"]
    assert np.isnan(table.as_dict()["Press"]).all()

    path.write_text("Step,Temp\n0,300\nmalformed,row\n1\n2,302\n")
    with pytest.raises(ValueError):
        parse_thermo_csv(path)
    legacy = parse_thermo_csv(path, legacy_tolerant=True)
    assert legacy.as_dict()["Step"].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(legacy.as_dict()["Temp"][1])


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("step,msd\n0,0\n1,0.1\n2,0.2\n", "header"),
        ("Step,MSD\n0,0\n1\n2,0.2\n", "expected 2 fields"),
        ("Step,MSD\n0,0\n1,0.1,extra\n2,0.2\n", "expected 2 fields"),
        ("Step,MSD\n0,0\n1,bad\n2,0.2\n", "Non-numeric"),
        ("Step,MSD\n0,0\n1.5,0.1\n2,0.2\n", "integer counts"),
        ("Step,MSD\n0,0\n1,0.1\n1,0.2\n", "strictly increasing"),
        ("Step,MSD\n-1,0\n0,0.1\n1,0.2\n", "nonnegative"),
        ("Step,MSD\n0,0\n1,-0.1\n2,0.2\n", "MSD must be nonnegative"),
        ("Step,MSD\n0,0\n1,nan\n2,0.2\n", "non-finite"),
    ],
)
def test_parse_msd_csv_rejects_malformed_or_unphysical_rows(
    tmp_path: Path,
    payload: str,
    reason: str,
) -> None:
    path = tmp_path / "msd.csv"
    path.write_text(payload)

    with pytest.raises(ValueError, match=reason):
        parse_msd_csv(path)


def test_engine_neutral_writers_reject_invalid_step_and_msd_physics(
    tmp_path: Path,
) -> None:
    duplicate_step_table = ThermoTable(
        columns=["Step", "Temp"],
        data=np.asarray([[0.0, 300.0], [0.0, 301.0]], dtype=float),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        write_thermo_csv(tmp_path / "thermo.csv", duplicate_step_table)

    with pytest.raises(ValueError, match="integer counts"):
        write_msd_csv(
            tmp_path / "msd.csv",
            step=[0.0, 1.5, 2.0],
            msd=[0.0, 0.1, 0.2],
        )
    with pytest.raises(ValueError, match="MSD must be nonnegative"):
        write_msd_csv(
            tmp_path / "msd.csv",
            step=[0.0, 1.0, 2.0],
            msd=[0.0, -0.1, 0.2],
        )

