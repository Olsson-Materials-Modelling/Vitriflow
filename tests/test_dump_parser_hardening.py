from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vitriflow.analysis.dump import (
    read_dump_frames,
    read_last_dump_frame,
    read_last_dump_frames,
)


def _write_dump(
    path: Path,
    *,
    columns: str = "id type q x y z",
    rows: tuple[str, ...] = ("1 1 0 1 2 3",),
    natoms: int | None = None,
    box_header: str = "ITEM: BOX BOUNDS pp pp pp",
    bounds: tuple[str, str, str] = ("0 10", "0 10", "0 10"),
) -> Path:
    declared = len(rows) if natoms is None else natoms
    path.write_text(
        "ITEM: TIMESTEP\n7\n"
        f"ITEM: NUMBER OF ATOMS\n{declared}\n"
        f"{box_header}\n"
        + "\n".join(bounds)
        + f"\nITEM: ATOMS {columns}\n"
        + "\n".join(rows)
        + "\n"
    )
    return path


def test_dump_reader_preserves_integral_float_compatibility_and_sorts_ids(tmp_path: Path):
    source = _write_dump(
        tmp_path / "integral-floats.dump",
        rows=("2.0 1e0 -0.25 2 2 2", "1.000 2.000 0.25 1 1 1"),
    )

    frame = read_last_dump_frame(source, units_style=None)

    np.testing.assert_array_equal(frame.ids, [1, 2])
    np.testing.assert_array_equal(frame.types, [2, 1])
    np.testing.assert_allclose(frame.charges, [0.25, -0.25])


@pytest.mark.parametrize(
    ("columns", "row"),
    [
        ("id type x y z", "1.5 1 1 2 3"),
        ("id type x y z", "1 2.5 1 2 3"),
        ("id type xs ys zs", "1.5 1 0.1 0.2 0.3"),
        ("id type xs ys zs", "1 2.5 0.1 0.2 0.3"),
    ],
)
def test_dump_reader_rejects_fractional_ids_and_types_in_both_coordinate_paths(
    tmp_path: Path,
    columns: str,
    row: str,
):
    source = _write_dump(tmp_path / "fractional.dump", columns=columns, rows=(row,))
    with pytest.raises(ValueError, match="non-integral (id|type)"):
        read_last_dump_frame(source, units_style=None)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (("0 1 0 0 0",), "nonpositive id"),
        (("1 0 0 0 0",), "nonpositive type"),
        (("1 1 0 0 0", "1 2 1 1 1"), "duplicate atom ids"),
    ],
)
def test_dump_reader_rejects_nonpositive_identity_fields_and_duplicate_ids(
    tmp_path: Path,
    rows: tuple[str, ...],
    message: str,
):
    source = _write_dump(tmp_path / "identity.dump", columns="id type x y z", rows=rows)
    with pytest.raises(ValueError, match=message):
        read_last_dump_frame(source, units_style=None)


@pytest.mark.parametrize(
    ("columns", "row", "message"),
    [
        ("id type x y z", "1 1 nan 0 0", "nonfinite x"),
        ("id type xs ys zs", "1 1 0 inf 0", "nonfinite ys"),
        ("id type q x y z", "1 1 -inf 0 0 0", "nonfinite q"),
    ],
)
def test_dump_reader_rejects_nonfinite_positions_and_charges(
    tmp_path: Path,
    columns: str,
    row: str,
    message: str,
):
    source = _write_dump(tmp_path / "nonfinite-atom.dump", columns=columns, rows=(row,))
    with pytest.raises(ValueError, match=message):
        read_last_dump_frame(source, units_style=None)


@pytest.mark.parametrize(
    ("box_header", "bounds", "message"),
    [
        (
            "ITEM: BOX BOUNDS pp pp pp",
            ("0 nan", "0 10", "0 10"),
            "nonfinite BOX BOUNDS",
        ),
        (
            "ITEM: BOX BOUNDS pp pp pp",
            ("0 0", "0 10", "0 10"),
            "strictly positive cell lengths",
        ),
        (
            "ITEM: BOX BOUNDS pp pp pp",
            ("10 0", "10 0", "0 10"),
            "strictly positive cell lengths",
        ),
        (
            "ITEM: BOX BOUNDS pp pp pp",
            ("0 10", "0 10 0", "0 10"),
            "consistently contain exactly 2 or 3 values",
        ),
        (
            "ITEM: BOX BOUNDS pp pp pp",
            ("0 10 0", "0 10 0", "0 10 0"),
            "without the required 'xy xz yz' header",
        ),
        (
            "ITEM: BOX BOUNDS xy xz yz pp pp pp",
            ("0 1 1", "0 10 0", "0 10 0"),
            "strictly positive cell lengths",
        ),
    ],
)
def test_dump_reader_rejects_invalid_box_and_cell_evidence(
    tmp_path: Path,
    box_header: str,
    bounds: tuple[str, str, str],
    message: str,
):
    source = _write_dump(
        tmp_path / "bad-cell.dump",
        box_header=box_header,
        bounds=bounds,
    )
    with pytest.raises(ValueError, match=message):
        read_last_dump_frame(source, units_style=None)


@pytest.mark.parametrize(
    ("columns", "rows", "natoms", "message"),
    [
        ("id type x y z", ("1 1 0 0",), 1, "header declares 5 columns"),
        ("id type x y z", ("1 1 0 0 0 extra",), 1, "header declares 5 columns"),
        ("id id type x y z", ("1 1 1 0 0 0",), 1, "columns must be non-empty and unique"),
        ("id type x y z", ("1 1 0 0 0",), 2, "Truncated dump file"),
    ],
)
def test_dump_reader_rejects_malformed_atom_rows_and_short_counts(
    tmp_path: Path,
    columns: str,
    rows: tuple[str, ...],
    natoms: int,
    message: str,
):
    source = _write_dump(
        tmp_path / "bad-rows.dump",
        columns=columns,
        rows=rows,
        natoms=natoms,
    )
    with pytest.raises(ValueError, match=message):
        read_last_dump_frame(source, units_style=None)


def test_dump_reader_rejects_atom_rows_beyond_declared_count(tmp_path: Path):
    source = _write_dump(
        tmp_path / "too-many.dump",
        columns="id type x y z",
        rows=("1 1 0 0 0", "2 1 1 1 1"),
        natoms=1,
    )
    with pytest.raises(ValueError, match="count may be too small"):
        read_dump_frames(source, units_style=None)


def test_dump_reader_rejects_empty_atom_frames(tmp_path: Path):
    source = _write_dump(tmp_path / "empty.dump", rows=(), natoms=0)
    with pytest.raises(ValueError, match="nonpositive atom count"):
        read_last_dump_frame(source, units_style=None)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_dump_frame_count_limits_require_positive_integers(tmp_path: Path, value: object):
    source = _write_dump(tmp_path / "valid.dump")
    with pytest.raises(ValueError, match="integer >= 1"):
        read_dump_frames(source, last_n=value, units_style=None)
    with pytest.raises(ValueError, match="integer >= 1"):
        read_last_dump_frames(source, value, units_style=None)


def test_valid_restricted_triclinic_scaled_dump_is_preserved(tmp_path: Path):
    source = _write_dump(
        tmp_path / "triclinic.dump",
        columns="id type xs ys zs",
        rows=("1 2 0.25 0.5 0.75",),
        box_header="ITEM: BOX BOUNDS xy xz yz pp pp pp",
        bounds=("0.5 6 1", "2 7.25 -0.5", "3 9 0.25"),
    )

    frame = read_last_dump_frame(source, units_style=None)

    np.testing.assert_allclose(frame.origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        frame.cell,
        [[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [-0.5, 0.25, 6.0]],
    )
    np.testing.assert_allclose(frame.positions, [[2.125, 4.6875, 7.5]])
