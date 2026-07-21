from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..lammps_units import charge_to_elementary_factor, length_to_angstrom_factor


@dataclass(frozen=True)
class DumpFrame:
    """Dump frame."""

    timestep: int
    ids: np.ndarray  # n int
    types: np.ndarray  # n int
    positions: np.ndarray  # n float cartesian
    cell: np.ndarray  # float row vectors
    origin: np.ndarray  # float
    charges: Optional[np.ndarray] = None  # optional per-atom charge from dump column q
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pbc", normalize_pbc(self.pbc))

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    def cell_inv(self) -> np.ndarray:
        return np.linalg.inv(self.cell)


def normalize_pbc(value: Any) -> tuple[bool, bool, bool]:
    """Return three explicit periodic-boundary flags.

    Boundary conditions are scientific input, not a display hint.  Accept a
    scalar boolean for convenience, but reject missing, malformed, or string
    values instead of silently turning them into a fully periodic cell.
    """

    if isinstance(value, (bool, np.bool_)):
        flag = bool(value)
        return (flag, flag, flag)
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError("pbc must be a boolean or a sequence of exactly three booleans")
    try:
        raw = list(value)
    except Exception as exc:
        raise ValueError("pbc must be a boolean or a sequence of exactly three booleans") from exc
    if len(raw) != 3:
        raise ValueError(f"pbc must contain exactly three flags; got {len(raw)}")
    if not all(isinstance(x, (bool, np.bool_)) for x in raw):
        raise ValueError("pbc entries must be booleans")
    return tuple(bool(x) for x in raw)  # type: ignore[return-value]


def frame_pbc(frame: Any) -> tuple[bool, bool, bool]:
    """Read validated PBC from a frame, failing if the frame has no contract."""

    if not hasattr(frame, "pbc"):
        raise ValueError("structure frame has no periodic-boundary metadata")
    return normalize_pbc(getattr(frame, "pbc"))


def canonicalize_lammps_frame(frame: DumpFrame, *, units_style: str) -> DumpFrame:
    """Convert native LAMMPS geometry/charge fields to Angstrom and e."""

    length_factor = length_to_angstrom_factor(units_style)
    charge_factor = charge_to_elementary_factor(units_style)
    return DumpFrame(
        timestep=int(frame.timestep),
        ids=np.asarray(frame.ids, dtype=int),
        types=np.asarray(frame.types, dtype=int),
        positions=np.asarray(frame.positions, dtype=float) * float(length_factor),
        cell=np.asarray(frame.cell, dtype=float) * float(length_factor),
        origin=np.asarray(frame.origin, dtype=float) * float(length_factor),
        charges=(
            None
            if frame.charges is None
            else np.asarray(frame.charges, dtype=float) * float(charge_factor)
        ),
        pbc=frame.pbc,
    )


def _find_col(cols: list[str], names: Iterable[str]) -> Optional[int]:
    for nm in names:
        if nm in cols:
            return cols.index(nm)
    return None


def _parse_atom_integer(token: str, *, field: str, row: int, path: Path) -> int:
    """Parse an exactly integral, positive atom identifier field.

    LAMMPS normally writes ``id`` and ``type`` as decimal integers, but some
    custom dump formats render them as integral-valued floats (for example
    ``1.0``).  Preserve that compatibility without silently truncating a value
    such as ``1.5``.
    """

    try:
        value = Decimal(str(token))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"Dump {path} atom row {row} has invalid {field} value {token!r}; "
            "an exactly integral positive value is required"
        ) from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(
            f"Dump {path} atom row {row} has non-integral {field} value {token!r}; "
            "an exactly integral positive value is required"
        )
    int_info = np.iinfo(np.intp)
    if value <= 0:
        raise ValueError(f"Dump {path} atom row {row} has nonpositive {field} value {token!r}")
    if value > Decimal(int(int_info.max)):
        raise ValueError(f"Dump {path} atom row {row} {field} value {token!r} exceeds integer range")
    parsed = int(value)
    return parsed


def _parse_finite_atom_float(token: str, *, field: str, row: int, path: Path) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Dump {path} atom row {row} has invalid {field} value {token!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Dump {path} atom row {row} has nonfinite {field} value {token!r}")
    return value


def _validated_cell_and_origin(
    cell: np.ndarray,
    origin: np.ndarray,
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, right-handed, nonsingular 3-D cell evidence."""

    cell_array = np.asarray(cell, dtype=float)
    origin_array = np.asarray(origin, dtype=float)
    if cell_array.shape != (3, 3):
        raise ValueError(f"Dump {path} cell must have shape (3, 3); got {cell_array.shape}")
    if origin_array.shape != (3,):
        raise ValueError(f"Dump {path} origin must have shape (3,); got {origin_array.shape}")
    if not np.all(np.isfinite(cell_array)):
        raise ValueError(f"Dump {path} cell contains nonfinite values")
    if not np.all(np.isfinite(origin_array)):
        raise ValueError(f"Dump {path} origin contains nonfinite values")
    sign, logabsdet = np.linalg.slogdet(cell_array)
    if sign <= 0.0 or not np.isfinite(logabsdet):
        raise ValueError(f"Dump {path} cell must be finite, right-handed, and nonsingular")
    return cell_array, origin_array


def read_last_dump_frame(path: Path, *, units_style: Optional[str] = "metal") -> DumpFrame:
    """Last dump frame."""

    frames = read_dump_frames(path, last_n=1, units_style=units_style)
    if not frames:
        raise ValueError(f"Failed to parse last frame from dump: {path}")
    return frames[-1]


def read_last_dump_frames(path: Path, n: int, *, units_style: Optional[str] = "metal") -> list[DumpFrame]:
    """Last dump frames."""
    if isinstance(n, (bool, np.bool_)) or int(n) != n or int(n) < 1:
        raise ValueError("n must be an integer >= 1")
    return read_dump_frames(path, last_n=int(n), units_style=units_style)


def read_dump_frames(
    path: Path,
    *,
    last_n: Optional[int] = None,
    units_style: Optional[str] = "metal",
) -> list[DumpFrame]:
    """Dump frames."""

    if last_n is not None and (
        isinstance(last_n, (bool, np.bool_))
        or int(last_n) != last_n
        or int(last_n) < 1
    ):
        raise ValueError("last_n must be an integer >= 1")
    frames: "deque[DumpFrame]" = deque(
        maxlen=int(last_n) if last_n is not None else None
    )
    for fr in iter_dump_frames(path, units_style=units_style):
        frames.append(fr)
    return list(frames)


def iter_dump_frames(path: Path, *, units_style: Optional[str] = "metal") -> Iterable[DumpFrame]:
    """Iter dump frames."""

    with Path(path).open("r", errors="replace") as f:
        it = iter(f)
        parsed_frame = False
        while True:
            try:
                ln = next(it)
            except StopIteration:
                break
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith("ITEM: TIMESTEP"):
                if parsed_frame:
                    raise ValueError(
                        f"Unexpected content after a complete dump frame in {path}; "
                        "the NUMBER OF ATOMS count may be too small"
                    )
                continue

            # timestep
            try:
                timestep_token = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            try:
                timestep = int(timestep_token)
            except ValueError as exc:
                raise ValueError(f"Dump {path} has invalid integer timestep {timestep_token!r}") from exc

            # number atoms
            try:
                ln2 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln2.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected dump format in {path}: expected NUMBER OF ATOMS, got: {ln2}")
            try:
                natoms_token = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            try:
                natoms = int(natoms_token)
            except ValueError as exc:
                raise ValueError(f"Dump {path} has invalid integer atom count {natoms_token!r}") from exc
            if natoms < 1:
                raise ValueError(f"Dump {path} has nonpositive atom count {natoms}")

            # box bounds
            try:
                ln3 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln3.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump format in {path}: expected BOX BOUNDS, got: {ln3}")

            pbc = _pbc_from_box_bounds_header(ln3, path=Path(path))
            bounds: list[list[float]] = []
            try:
                for axis in range(3):
                    toks = next(it).split()
                    if not toks or toks[0] == "ITEM:":
                        raise ValueError(f"Dump {path} has a missing BOX BOUNDS row for axis {axis}")
                    try:
                        row = [float(x) for x in toks]
                    except (ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"Dump {path} has a nonnumeric BOX BOUNDS row for axis {axis}: {toks!r}"
                        ) from exc
                    if not all(np.isfinite(row)):
                        raise ValueError(f"Dump {path} has nonfinite BOX BOUNDS values for axis {axis}")
                    bounds.append(row)
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            row_widths = {len(row) for row in bounds}
            if len(row_widths) != 1 or next(iter(row_widths)) not in {2, 3}:
                raise ValueError(
                    f"Dump {path} BOX BOUNDS rows must consistently contain exactly 2 or 3 values; "
                    f"got {[len(row) for row in bounds]}"
                )
            bound_width = len(bounds[0])
            header_tokens = ln3.split()[3:]
            has_restricted_tilts = header_tokens[:3] == ["xy", "xz", "yz"]

            if bound_width == 2:
                if has_restricted_tilts:
                    raise ValueError(
                        f"Dump {path} declares triclinic tilt factors but BOX BOUNDS rows contain only 2 values"
                    )
                xlo, xhi = bounds[0]
                ylo, yhi = bounds[1]
                zlo, zhi = bounds[2]
                lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
                cell = np.array([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]], dtype=float)
                origin = np.array([xlo, ylo, zlo], dtype=float)
            else:
                if not has_restricted_tilts:
                    raise ValueError(
                        f"Dump {path} has 3-value BOX BOUNDS rows without the required 'xy xz yz' header"
                    )
                # triclinic bounding limits
                xlo_b, xhi_b, xy = float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])
                ylo_b, yhi_b, xz = float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])
                zlo_b, zhi_b, yz = float(bounds[2][0]), float(bounds[2][1]), float(bounds[2][2])

                x_corr_min = min(0.0, xy, xz, xy + xz)
                x_corr_max = max(0.0, xy, xz, xy + xz)
                y_corr_min = min(0.0, yz)
                y_corr_max = max(0.0, yz)

                xlo, xhi = (xlo_b - x_corr_min), (xhi_b - x_corr_max)
                ylo, yhi = (ylo_b - y_corr_min), (yhi_b - y_corr_max)
                zlo, zhi = zlo_b, zhi_b

                lx, ly, lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)
                cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)
                origin = np.array([xlo, ylo, zlo], dtype=float)

            if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
                raise ValueError(
                    f"Dump {path} BOX BOUNDS must define strictly positive cell lengths; "
                    f"got ({lx}, {ly}, {lz})"
                )
            cell, origin = _validated_cell_and_origin(cell, origin, path=Path(path))

            # atoms
            try:
                ln4 = next(it).strip()
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e
            if not ln4.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected dump format in {path}: expected ATOMS, got: {ln4}")
            cols = ln4.split()[2:]

            atom_lines: list[list[str]] = []
            try:
                for _ in range(natoms):
                    atom_lines.append(next(it).split())
            except StopIteration as e:
                raise ValueError(f"Truncated dump file: {path}") from e

            frame = _parse_dump_frame_from_table(
                timestep,
                natoms,
                cell,
                origin,
                cols,
                atom_lines,
                Path(path),
                pbc=pbc,
            )
            parsed_frame = True
            if units_style is None:
                yield frame
            else:
                yield canonicalize_lammps_frame(frame, units_style=units_style)


def _pbc_from_box_bounds_header(header: str, *, path: Path) -> tuple[bool, bool, bool]:
    """Parse LAMMPS boundary flags from an ``ITEM: BOX BOUNDS`` header."""

    tokens = str(header).split()[3:]
    boundary = [tok.lower() for tok in tokens if len(tok) == 2 and set(tok.lower()) <= set("pfsm")]
    if len(boundary) != 3:
        raise ValueError(
            f"Dump {path} BOX BOUNDS header must contain three LAMMPS boundary flags; got: {header!r}"
        )
    # LAMMPS periodic boundaries are encoded as pp.  Fixed, shrink-wrap, and
    # mixed face styles are non-periodic for minimum-image purposes.
    return normalize_pbc(tuple(tok == "pp" for tok in boundary))


def _parse_dump_frame_from_table(
    timestep: int,
    natoms: int,
    cell: np.ndarray,
    origin: np.ndarray,
    cols: list[str],
    atom_lines: list[list[str]],
    path: Path,
    *,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> DumpFrame:
    """Dump frame from."""

    if isinstance(natoms, (bool, np.bool_)) or not isinstance(natoms, (int, np.integer)):
        raise ValueError(f"Dump {path} atom count must be an integer; got {natoms!r}")
    natoms = int(natoms)
    if natoms < 1:
        raise ValueError(f"Dump {path} has nonpositive atom count {natoms}")
    if len(atom_lines) != natoms:
        raise ValueError(
            f"Dump {path} declares {natoms} atoms but provides {len(atom_lines)} atom rows"
        )
    if not cols or any(not str(col).strip() for col in cols) or len(set(cols)) != len(cols):
        raise ValueError(f"Dump {path} ATOMS columns must be non-empty and unique; got: {cols}")
    for row_number, toks in enumerate(atom_lines, start=1):
        if len(toks) != len(cols):
            raise ValueError(
                f"Dump {path} atom row {row_number} has {len(toks)} values, "
                f"but the ATOMS header declares {len(cols)} columns"
            )

    cell, origin = _validated_cell_and_origin(cell, origin, path=path)

    id_col = _find_col(cols, ["id"])
    type_col = _find_col(cols, ["type"])
    x_col = _find_col(cols, ["xu", "x"])
    y_col = _find_col(cols, ["yu", "y"])
    z_col = _find_col(cols, ["zu", "z"])
    xs_col = _find_col(cols, ["xs"])
    ys_col = _find_col(cols, ["ys"])
    zs_col = _find_col(cols, ["zs"])
    q_col = _find_col(cols, ["q", "charge"])

    if id_col is None or type_col is None:
        raise ValueError(f"Dump {path} missing required columns 'id' and/or 'type' (got: {cols})")

    use_scaled = (x_col is None or y_col is None or z_col is None) and (xs_col is not None and ys_col is not None and zs_col is not None)
    if not use_scaled and (x_col is None or y_col is None or z_col is None):
        raise ValueError(
            f"Dump {path} must contain cartesian columns x/y/z or xu/yu/zu, or scaled xs/ys/zs. Got: {cols}"
        )

    ids = np.empty((natoms,), dtype=int)
    types = np.empty((natoms,), dtype=int)
    pos = np.empty((natoms, 3), dtype=float)
    charges = np.empty((natoms,), dtype=float) if q_col is not None else None

    if use_scaled:
        H = cell
        for i, toks in enumerate(atom_lines):
            row_number = i + 1
            ids[i] = _parse_atom_integer(toks[id_col], field="id", row=row_number, path=path)
            types[i] = _parse_atom_integer(toks[type_col], field="type", row=row_number, path=path)
            if charges is not None and q_col is not None:
                charges[i] = _parse_finite_atom_float(
                    toks[q_col], field=cols[q_col], row=row_number, path=path
                )
            s = np.array(
                [
                    _parse_finite_atom_float(toks[xs_col], field=cols[xs_col], row=row_number, path=path),
                    _parse_finite_atom_float(toks[ys_col], field=cols[ys_col], row=row_number, path=path),
                    _parse_finite_atom_float(toks[zs_col], field=cols[zs_col], row=row_number, path=path),
                ],
                dtype=float,
            )
            pos[i, :] = origin + s @ H
    else:
        for i, toks in enumerate(atom_lines):
            row_number = i + 1
            ids[i] = _parse_atom_integer(toks[id_col], field="id", row=row_number, path=path)
            types[i] = _parse_atom_integer(toks[type_col], field="type", row=row_number, path=path)
            if charges is not None and q_col is not None:
                charges[i] = _parse_finite_atom_float(
                    toks[q_col], field=cols[q_col], row=row_number, path=path
                )
            pos[i, :] = [
                _parse_finite_atom_float(toks[x_col], field=cols[x_col], row=row_number, path=path),
                _parse_finite_atom_float(toks[y_col], field=cols[y_col], row=row_number, path=path),
                _parse_finite_atom_float(toks[z_col], field=cols[z_col], row=row_number, path=path),
            ]

    if not np.all(np.isfinite(pos)):
        raise ValueError(f"Dump {path} positions become nonfinite after coordinate conversion")
    if charges is not None and not np.all(np.isfinite(charges)):
        raise ValueError(f"Dump {path} charges contain nonfinite values")
    if np.unique(ids).size != natoms:
        raise ValueError(f"Dump {path} contains duplicate atom ids")

    order = np.argsort(ids)
    ids = ids[order]
    types = types[order]
    pos = pos[order]
    if charges is not None:
        charges = charges[order]

    return DumpFrame(
        timestep=int(timestep),
        ids=ids,
        types=types,
        positions=pos,
        cell=cell,
        origin=origin,
        charges=charges,
        pbc=normalize_pbc(pbc),
    )
