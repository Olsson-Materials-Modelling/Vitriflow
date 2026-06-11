from __future__ import annotations

from pathlib import Path

import pytest


def test_csvr_rejected_for_lammps_engine(tmp_path: Path):
    from vitriflow.config import RunConfig

    data = tmp_path / "dummy.data"
    data.write_text("\n")

    cfg = {
        "engine": "lammps",
        "potential": {
            "kind": "kim",
            "model": "DUMMY_MODEL",
            "interactions": ["Si"],
        },
        "structure": {"lammps_data": str(data)},
        "md": {"thermostat": {"style": "csvr"}},
    }

    with pytest.raises(ValueError, match=r"csvr"):
        RunConfig.model_validate(cfg)


def test_csvr_allowed_for_cp2k_engine(tmp_path: Path):
    from vitriflow.config import RunConfig

    data = tmp_path / "dummy.data"
    data.write_text("\n")

    cfg = {
        "engine": "cp2k",
        "cp2k": {},
        "structure": {"lammps_data": str(data)},
        "md": {"thermostat": {"style": "csvr"}},
    }

    rc = RunConfig.model_validate(cfg)
    assert str(rc.engine) == "cp2k"
    assert str(rc.md.thermostat.style) == "csvr"
