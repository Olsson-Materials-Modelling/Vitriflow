from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.datafile import read_datafile_frame
from vitriflow.analysis.dump import DumpFrame
from vitriflow.io.extxyz import (
    read_extxyz_frames,
    write_extxyz_frames,
    write_extxyz_iter,
    write_extxyz_single_with_species,
)


def _extxyz_text(*, step: object = 4, rows: str, n_atoms: int = 2) -> str:
    return (
        f"{n_atoms}\n"
        f'Lattice="8 0 0 0 8 0 0 0 8" '
        f'Properties=species:S:1:pos:R:3:type:I:1:id:I:1 pbc="T T T" Step={step}\n'
        + rows
    )


@pytest.mark.parametrize(
    "bad_step",
    ["1.5", "nan", "inf", "invalid", "-1", str(int(np.iinfo(np.intp).max) + 1)],
)
def test_extxyz_rejects_noninteger_or_nonfinite_declared_step(
    tmp_path: Path,
    bad_step: str,
):
    source = tmp_path / "bad-step.extxyz"
    source.write_text(
        _extxyz_text(
            step=bad_step,
            rows="Si 1 2 3 1 1\nSi 4 5 6 1 2\n",
        )
    )

    with pytest.raises(ValueError, match="EXTXYZ Step"):
        read_extxyz_frames(source)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("Si 1 2 3 1.5 1\nSi 4 5 6 1 2\n", "type column is not a valid integer"),
        ("Si 1 2 3 1 bad\nSi 4 5 6 1 2\n", "id column is not a valid positive integer"),
        ("Si 1 2 3 1 1\nSi 4 5 6 1 1\n", "IDs must be unique"),
        ("Si nan 2 3 1 1\nSi 4 5 6 1 2\n", "Non-finite pos"),
        ("Si 1 2 3 0 1\nSi 4 5 6 1 2\n", "type column is not a valid integer"),
        (
            f"Si 1 2 3 1 {int(np.iinfo(np.intp).max) + 1}\nSi 4 5 6 1 2\n",
            "id column is not a valid positive integer",
        ),
    ],
)
def test_extxyz_rejects_malformed_explicit_identity_or_geometry(
    tmp_path: Path,
    rows: str,
    message: str,
):
    source = tmp_path / "bad-evidence.extxyz"
    source.write_text(_extxyz_text(rows=rows))

    with pytest.raises(ValueError, match=message):
        read_extxyz_frames(source)


def test_extxyz_rejects_species_type_disagreement(tmp_path: Path):
    source = tmp_path / "mismatch.extxyz"
    source.write_text(
        _extxyz_text(rows="Si 1 2 3 2 1\nN 4 5 6 1 2\n")
    )

    with pytest.raises(ValueError, match="species/type mismatch"):
        read_extxyz_frames(source, type_to_species=["Si", "N"])


@pytest.mark.parametrize("duplicate_key", ["Lattice", "Properties", "pbc", "Step"])
def test_extxyz_rejects_duplicate_scientific_comment_keys(
    tmp_path: Path,
    duplicate_key: str,
):
    values = {
        "Lattice": '"8 0 0 0 8 0 0 0 8"',
        "Properties": "species:S:1:pos:R:3",
        "pbc": '"T T T"',
        "Step": "1",
    }
    comment = " ".join(f"{key}={value}" for key, value in values.items())
    comment += f" {duplicate_key}={values[duplicate_key]}"
    source = tmp_path / "duplicate-key.extxyz"
    source.write_text(f"1\n{comment}\nSi 1 2 3\n")

    with pytest.raises(ValueError, match=f"Duplicate EXTXYZ comment key '{duplicate_key}'"):
        read_extxyz_frames(source)


def test_extxyz_symbol_types_remain_stable_when_atom_order_changes(tmp_path: Path):
    source = tmp_path / "stable.extxyz"
    source.write_text(
        "2\n"
        'Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3 pbc="T T T" Step=1\n'
        "Si 1 1 1\nN 2 2 2\n"
        "2\n"
        'Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3 pbc="T T T" Step=2\n'
        "N 3 3 3\nSi 4 4 4\n"
    )

    frames = read_extxyz_frames(source)

    assert frames[0].types.tolist() == [1, 2]
    assert frames[1].types.tolist() == [2, 1]


def _data_text(*, rows: str, n_atoms: int = 2, n_types: int = 1, style: str = "atomic") -> str:
    return (
        "LAMMPS data\n\n"
        f"{n_atoms} atoms\n{n_types} atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        f"Atoms # {style}\n\n"
        + rows
    )


@pytest.mark.parametrize(
    "rows, n_atoms, n_types, message",
    [
        ("1.5 1 1 2 3\n2 1 4 5 6\n", 2, 1, "positive exact integer"),
        ("1 1 1 2 3\n1 1 4 5 6\n", 2, 1, "unique positive integers"),
        ("1 1 1 2 3\n", 2, 1, "Atom-count mismatch"),
        ("1 2 1 2 3\n2 1 4 5 6\n", 2, 1, "must lie in"),
        ("1 1 nan 2 3\n2 1 4 5 6\n", 2, 1, "Non-finite x coordinate"),
        ("1 1 1 2\n2 1 4 5 6\n", 2, 1, "Malformed atomic atom row"),
        (
            f"{int(np.iinfo(np.intp).max) + 1} 1 1 2 3\n2 1 4 5 6\n",
            2,
            1,
            "exceeds platform index range",
        ),
    ],
)
def test_datafile_frame_rejects_malformed_declared_atom_evidence(
    tmp_path: Path,
    rows: str,
    n_atoms: int,
    n_types: int,
    message: str,
):
    source = tmp_path / "bad.data"
    source.write_text(
        _data_text(rows=rows, n_atoms=n_atoms, n_types=n_types)
    )

    with pytest.raises(ValueError, match=message):
        read_datafile_frame(source)


def test_datafile_frame_validates_charge_rows_and_preserves_types(tmp_path: Path):
    source = tmp_path / "charge.data"
    source.write_text(
        _data_text(
            style="charge",
            n_types=2,
            rows="2 2 -0.5 4 5 6\n1 1 0.5 1 2 3\n",
        )
    )

    frame = read_datafile_frame(source, atom_style="charge", units_style="real")

    assert frame.ids.tolist() == [1, 2]
    assert frame.types.tolist() == [1, 2]
    np.testing.assert_allclose(frame.positions, [[1, 2, 3], [4, 5, 6]])

    source.write_text(
        _data_text(
            style="charge",
            n_types=2,
            rows="2 2 nan 4 5 6\n1 1 0.5 1 2 3\n",
        )
    )
    with pytest.raises(ValueError, match="Non-finite charge"):
        read_datafile_frame(source, atom_style="charge")


def _writer_frame(**overrides) -> DumpFrame:
    values = {
        "timestep": 3,
        "ids": np.asarray([1, 2], dtype=object),
        "types": np.asarray([1, 1], dtype=object),
        "positions": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        "cell": np.eye(3, dtype=float) * 10.0,
        "origin": np.zeros(3, dtype=float),
    }
    values.update(overrides)
    return DumpFrame(**values)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"timestep": 1.5}, "Step must be an exact integer"),
        ({"timestep": float("nan")}, "Step must be an exact integer"),
        ({"timestep": -1}, "Step must be a nonnegative exact integer"),
        (
            {"timestep": int(np.iinfo(np.intp).max) + 1},
            "outside platform index range",
        ),
        ({"ids": np.asarray([1.5, 2], dtype=object)}, "id at atom 1 must be an exact integer"),
        ({"ids": np.asarray([0, 2], dtype=object)}, "id at atom 1 must be a positive exact integer"),
        ({"ids": np.asarray([1, 1], dtype=object)}, "ids must be unique"),
        (
            {"ids": np.asarray([int(np.iinfo(np.intp).max) + 1, 2], dtype=object)},
            "outside platform index range",
        ),
        ({"types": np.asarray([1.5, 1], dtype=object)}, "type at atom 1 must be an exact integer"),
        ({"types": np.asarray([0, 1], dtype=object)}, "type at atom 1 must be a positive exact integer"),
        ({"ids": np.asarray([1], dtype=object)}, "ids must be a one-dimensional array matching"),
        (
            {"positions": np.asarray([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])},
            "positions must be finite",
        ),
        ({"origin": np.asarray([0.0, np.inf, 0.0])}, "origin must be finite"),
        ({"origin": np.asarray([0.0, 0.0])}, "origin must have shape"),
        ({"cell": np.zeros((3, 3))}, "cell must be nonsingular"),
    ],
)
def test_extxyz_writer_rejects_malformed_frame_evidence(
    tmp_path: Path,
    override: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        write_extxyz_frames(
            tmp_path / "invalid.extxyz",
            [_writer_frame(**override)],
            type_to_species=["Si"],
        )


def test_extxyz_writer_validates_mapping_and_per_atom_species_labels(tmp_path: Path):
    frame = _writer_frame(types=np.asarray([1, 2], dtype=object))
    with pytest.raises(ValueError, match="type exceeds supplied"):
        write_extxyz_frames(tmp_path / "short-map.extxyz", [frame], type_to_species=["Si"])
    with pytest.raises(ValueError, match="non-empty EXTXYZ token"):
        write_extxyz_frames(
            tmp_path / "bad-map.extxyz",
            [_writer_frame()],
            type_to_species=["Si N"],
        )
    with pytest.raises(ValueError, match="non-empty EXTXYZ token"):
        write_extxyz_single_with_species(
            tmp_path / "bad-species.extxyz",
            _writer_frame(),
            ["Si", ""],
        )
    with pytest.raises(ValueError, match="must be a sequence"):
        write_extxyz_frames(
            tmp_path / "string-map.extxyz",
            [_writer_frame()],
            type_to_species="Si",
        )


def test_extxyz_iter_writer_rejects_invalid_types_without_truncation(tmp_path: Path):
    with pytest.raises(ValueError, match="type at atom 1 must be an exact integer"):
        write_extxyz_iter(
            tmp_path / "iter.extxyz",
            iter([_writer_frame(types=np.asarray([1.25, 1], dtype=object))]),
            type_to_species=["Si"],
        )


def test_extxyz_writer_preserves_cell_relative_coordinate_convention(tmp_path: Path):
    frame = _writer_frame(
        positions=np.asarray([[6.0, -1.0, 4.0], [9.0, 3.0, 9.0]], dtype=float),
        origin=np.asarray([5.0, -2.0, 3.0], dtype=float),
    )
    source = tmp_path / "relative.extxyz"

    write_extxyz_frames(source, [frame], type_to_species=["Si"], wrap=False)
    loaded = read_extxyz_frames(source, type_to_species=["Si"])[0]

    np.testing.assert_allclose(loaded.positions, frame.positions - frame.origin)
    np.testing.assert_allclose(loaded.cell, frame.cell)


def test_extxyz_placeholder_species_preserves_multiple_explicit_types(tmp_path: Path):
    frame = _writer_frame(types=np.asarray([1, 2], dtype=object))
    source = tmp_path / "placeholder.extxyz"

    write_extxyz_frames(source, [frame])
    loaded = read_extxyz_frames(source)[0]

    assert loaded.types.tolist() == [1, 2]
