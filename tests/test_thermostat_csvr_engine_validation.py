from __future__ import annotations

from pathlib import Path

import pytest


def test_csvr_allowed_for_lammps_engine(tmp_path: Path):
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

    rc = RunConfig.model_validate(cfg)
    assert str(rc.md.thermostat.style) == "csvr"


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


def test_cp2k_rejects_lammps_only_thermostat(tmp_path: Path):
    from vitriflow.config import RunConfig

    data = tmp_path / "dummy.data"
    data.write_text("\n")

    cfg = {
        "engine": "cp2k",
        "cp2k": {},
        "structure": {"lammps_data": str(data)},
        "md": {"thermostat": {"style": "langevin"}},
    }

    with pytest.raises(ValueError, match="engine='lammps'"):
        RunConfig.model_validate(cfg)


def test_cp2k_rejects_berendsen_barostat(tmp_path: Path):
    from vitriflow.config import RunConfig

    data = tmp_path / "dummy.data"
    data.write_text("\n")

    cfg = {
        "engine": "cp2k",
        "cp2k": {},
        "structure": {"lammps_data": str(data)},
        "md": {"barostat": {"style": "berendsen"}},
    }

    with pytest.raises(ValueError, match="barostat.style"):
        RunConfig.model_validate(cfg)
