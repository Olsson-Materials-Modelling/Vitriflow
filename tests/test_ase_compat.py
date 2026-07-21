from __future__ import annotations

import sys
import types

import pytest


@pytest.mark.parametrize(
    ("reader", "expected_keyword"),
    [
        (lambda fileobj, atom_style=None: None, "atom_style"),
        (lambda fileobj, style=None: None, "style"),
    ],
)
def test_lammps_data_reader_uses_supported_nondeprecated_keyword(
    monkeypatch, reader, expected_keyword
):
    from vitriflow.io import ase_compat

    calls = []
    ase_module = types.ModuleType("ase")
    io_module = types.ModuleType("ase.io")
    lammps_module = types.ModuleType("ase.io.lammpsdata")
    data_module = types.ModuleType("ase.data")
    data_module.atomic_numbers = {"Si": 14, "O": 8}
    io_module.read = lambda path, **kwargs: calls.append((path, kwargs)) or "atoms"
    lammps_module.read_lammps_data = reader
    ase_module.io = io_module
    monkeypatch.setitem(sys.modules, "ase", ase_module)
    monkeypatch.setitem(sys.modules, "ase.io", io_module)
    monkeypatch.setitem(sys.modules, "ase.io.lammpsdata", lammps_module)
    monkeypatch.setitem(sys.modules, "ase.data", data_module)
    ase_compat.lammps_data_atom_style_keyword.cache_clear()

    result = ase_compat.ase_read_lammps_data(
        "cell.data",
        atom_style="charge",
        specorder=["Si", "O"],
        units="metal",
    )

    assert result == "atoms"
    assert calls == [
        (
            "cell.data",
            {
                "format": "lammps-data",
                expected_keyword: "charge",
                "Z_of_type": {1: 14, 2: 8},
                "units": "metal",
            },
        )
    ]
    assert ({"style", "atom_style"} - {expected_keyword}).isdisjoint(
        calls[0][1]
    )
    ase_compat.lammps_data_atom_style_keyword.cache_clear()


def test_lammps_data_reader_fails_closed_for_unknown_ase_api(monkeypatch):
    from vitriflow.io import ase_compat

    ase_module = types.ModuleType("ase")
    io_module = types.ModuleType("ase.io")
    lammps_module = types.ModuleType("ase.io.lammpsdata")
    lammps_module.read_lammps_data = lambda fileobj: None
    ase_module.io = io_module
    monkeypatch.setitem(sys.modules, "ase", ase_module)
    monkeypatch.setitem(sys.modules, "ase.io", io_module)
    monkeypatch.setitem(sys.modules, "ase.io.lammpsdata", lammps_module)
    ase_compat.lammps_data_atom_style_keyword.cache_clear()

    with pytest.raises(RuntimeError, match="neither 'atom_style'.*legacy 'style'"):
        ase_compat.lammps_data_atom_style_keyword()
    ase_compat.lammps_data_atom_style_keyword.cache_clear()
