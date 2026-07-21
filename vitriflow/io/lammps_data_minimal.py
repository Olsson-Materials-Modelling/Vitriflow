from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..analysis.dump import DumpFrame
from ..lammps_units import (
    charge_to_elementary_factor,
    length_from_angstrom_factor,
    length_to_angstrom_factor,
    mass_to_amu_factor,
)


_RE_ATOMS = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.IGNORECASE)
_RE_TYPES = re.compile(r"^\s*(\d+)\s+atom\s+types\s*$", re.IGNORECASE)

# Section names that can legally follow ``Masses`` or ``Atoms``.  Recognising
# them explicitly is important: coefficient rows are numeric and must never be
# mistaken for masses or atoms by this deliberately small reader.
_SECTION_HEADS = {
    "masses",
    "atoms",
    "velocities",
    "ellipsoids",
    "lines",
    "triangles",
    "bodies",
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "pair coeffs",
    "pairij coeffs",
    "bond coeffs",
    "angle coeffs",
    "dihedral coeffs",
    "improper coeffs",
    "bondbond coeffs",
    "bondangle coeffs",
    "middlebondtorsion coeffs",
    "endbondtorsion coeffs",
    "angletorsion coeffs",
    "angleangletorsion coeffs",
    "bondbond13 coeffs",
    "angleangle coeffs",
}


def read_lammps_data_minimal(
    path: Path,
    *,
    atom_style: str = "atomic",
    specorder: Optional[Sequence[str]] = None,
    units_style: str = "metal",
):
    """Read native LAMMPS data into canonical ASE units (A, u, e)."""

    from ase import Atoms

    txt = Path(path).read_text(errors="replace")
    lines = txt.splitlines()

    n_atoms: Optional[int] = None
    n_types: Optional[int] = None

    xlo = ylo = zlo = 0.0
    xhi = yhi = zhi = None  # type: ignore[assignment]
    xy = xz = yz = 0.0

    def _strip(ln: str) -> str:
        return ln.split("#", 1)[0].strip()

    def _section_name(ln: str) -> Optional[str]:
        key = " ".join(ln.lower().split())
        return key if key in _SECTION_HEADS else None

    def _integer_field(token: str, *, field: str, line_number: int) -> int:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field} at {path}:{line_number}: {token!r}"
            ) from exc
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(
                f"Invalid {field} at {path}:{line_number}: expected an integer, "
                f"got {token!r}"
            )
        return int(value)

    # header scan
    for raw in lines[:400]:
        ln = _strip(raw)
        if not ln:
            continue
        m = _RE_ATOMS.match(ln)
        if m:
            n_atoms = int(m.group(1))
            continue
        m = _RE_TYPES.match(ln)
        if m:
            n_types = int(m.group(1))
            continue

        toks = ln.split()
        if len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["xlo", "xhi"]:
            xlo, xhi = float(toks[0]), float(toks[1])
        elif len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["ylo", "yhi"]:
            ylo, yhi = float(toks[0]), float(toks[1])
        elif len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["zlo", "zhi"]:
            zlo, zhi = float(toks[0]), float(toks[1])
        elif len(toks) >= 6 and [t.lower() for t in toks[-3:]] == ["xy", "xz", "yz"]:
            xy, xz, yz = float(toks[0]), float(toks[1]), float(toks[2])

    if n_atoms is None:
        raise ValueError(f"Failed to parse '<N> atoms' from {path}")
    if n_types is None:
        raise ValueError(f"Failed to parse '<N> atom types' from {path}")
    if n_atoms < 1 or n_types < 1:
        raise ValueError(f"LAMMPS data header in {path} must declare positive atom/type counts")
    if xhi is None or yhi is None or zhi is None:
        raise ValueError(f"Failed to parse box bounds (xlo/xhi etc.) from {path}")
    bounds_and_tilts = np.asarray(
        [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz], dtype=float
    )
    if not np.all(np.isfinite(bounds_and_tilts)):
        raise ValueError(f"LAMMPS box bounds and tilt factors in {path} must be finite")

    lx = float(xhi - xlo)
    ly = float(yhi - ylo)
    lz = float(zhi - zlo)
    if not (lx > 0 and ly > 0 and lz > 0):
        raise ValueError("Non-positive cell lengths parsed from data file")

    # lammps cell ase
    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )
    cell_scale = float(np.max(np.abs(cell)))
    det = float(np.linalg.det(cell))
    det_tol = 128.0 * np.finfo(float).eps * max(cell_scale**3, np.finfo(float).tiny)
    if not np.all(np.isfinite(cell)) or not math.isfinite(det) or abs(det) <= det_tol:
        raise ValueError(f"LAMMPS cell in {path} must be finite and nonsingular")

    # sections
    masses_by_type: dict[int, float] = {}

    idx_masses = None
    idx_atoms = None
    atoms_style_in_file: Optional[str] = None

    for i, raw in enumerate(lines):
        ln = _strip(raw)
        if not ln:
            continue
        head = ln.split()[0].lower()
        if head == "masses":
            idx_masses = i
        elif head == "atoms":
            idx_atoms = i
            if "#" in raw:
                style_comment = raw.split("#", 1)[1].strip().split()
                atoms_style_in_file = style_comment[0].lower() if style_comment else None

    if idx_masses is not None:
        for line_number, raw in enumerate(lines[idx_masses + 1 :], start=idx_masses + 2):
            ln = _strip(raw)
            if not ln:
                continue
            if _section_name(ln) is not None:
                break
            toks = ln.split()
            if len(toks) < 2:
                raise ValueError(
                    f"Malformed mass row at {path}:{line_number}: expected type and mass"
                )
            t = _integer_field(toks[0], field="mass atom type", line_number=line_number)
            if not 1 <= t <= n_types:
                raise ValueError(
                    f"Mass atom type out of range at {path}:{line_number}: "
                    f"{t} not in [1, {n_types}]"
                )
            try:
                mass = float(toks[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid mass at {path}:{line_number}: {toks[1]!r}"
                ) from exc
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(
                    f"Mass must be finite and positive at {path}:{line_number}; "
                    f"got {toks[1]!r} for atom type {t}"
                )
            if t in masses_by_type:
                raise ValueError(
                    f"Duplicate mass entry for atom type {t} at {path}:{line_number}"
                )
            masses_by_type[t] = mass

    if idx_atoms is None:
        raise ValueError(f"Failed to find 'Atoms' section in {path}")

    style = str(atom_style).strip().lower()
    if atoms_style_in_file in {"atomic", "charge"}:
        style = atoms_style_in_file

    ids: list[int] = []
    types: list[int] = []
    pos: list[list[float]] = []
    charges: list[float] = []

    for line_number, raw in enumerate(lines[idx_atoms + 1 :], start=idx_atoms + 2):
        ln = _strip(raw)
        if not ln:
            continue
        if _section_name(ln) is not None:
            break
        toks = ln.split()
        if style == "charge":
            if len(toks) < 6:
                raise ValueError(
                    f"Malformed charge atom row at {path}:{line_number}: expected "
                    "id, type, charge, x, y, z"
                )
            i0 = _integer_field(toks[0], field="atom id", line_number=line_number)
            t0 = _integer_field(toks[1], field="atom type", line_number=line_number)
            try:
                q0 = float(toks[2])
                x, y, z = float(toks[3]), float(toks[4]), float(toks[5])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric charge atom row at {path}:{line_number}"
                ) from exc
            if not np.isfinite(q0):
                raise ValueError(
                    f"Charge must be finite at {path}:{line_number}; got {toks[2]!r}"
                )
            ids.append(i0)
            types.append(t0)
            charges.append(q0)
            pos.append([x - xlo, y - ylo, z - zlo])
        else:
            if len(toks) < 5:
                raise ValueError(
                    f"Malformed atomic atom row at {path}:{line_number}: expected "
                    "id, type, x, y, z"
                )
            i0 = _integer_field(toks[0], field="atom id", line_number=line_number)
            t0 = _integer_field(toks[1], field="atom type", line_number=line_number)
            try:
                x, y, z = float(toks[2]), float(toks[3]), float(toks[4])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric atomic atom row at {path}:{line_number}"
                ) from exc
            ids.append(i0)
            types.append(t0)
            pos.append([x - xlo, y - ylo, z - zlo])

    if len(ids) != n_atoms:
        raise ValueError(
            f"Atom-count mismatch in {path}: header declares {n_atoms}, parsed {len(ids)}"
        )
    if len(set(ids)) != len(ids) or any(atom_id <= 0 for atom_id in ids):
        raise ValueError(f"Atom IDs in {path} must be unique positive integers")
    if any(t < 1 or t > n_types for t in types):
        raise ValueError(f"Atom types in {path} must lie in [1, {n_types}]")
    if not np.all(np.isfinite(np.asarray(pos, dtype=float))):
        raise ValueError(f"Atom positions in {path} must be finite")

    order = np.argsort(np.asarray(ids, dtype=int))
    types_arr = np.asarray(types, dtype=int)[order]
    pos_arr = np.asarray(pos, dtype=float)[order]

    if specorder is not None:
        spec = list(specorder)
        if len(spec) < int(np.max(types_arr)):
            raise ValueError("specorder shorter than max atom type in data file")
        symbols = [str(spec[t - 1]) for t in types_arr.tolist()]
    else:
        symbols = [f"X{int(t)}" for t in types_arr.tolist()]

    length_factor = float(length_to_angstrom_factor(units_style))
    atoms = Atoms(
        symbols=symbols,
        positions=pos_arr * length_factor,
        cell=cell * length_factor,
        pbc=True,
    )

    # Assign explicitly provided native masses.  Once a Masses section exists,
    # every atom type used by this structure must be covered; falling back to
    # periodic-table defaults would silently change isotopes or custom masses.
    if idx_masses is not None:
        used_types = set(int(t) for t in types_arr.tolist())
        missing = sorted(used_types.difference(masses_by_type))
        if missing:
            raise ValueError(
                f"Masses section in {path} does not cover used atom types {missing}"
            )
        mass_factor = float(mass_to_amu_factor(units_style))
        converted_masses = np.asarray(
            [masses_by_type[int(t)] * mass_factor for t in types_arr.tolist()],
            dtype=float,
        )
        if not np.all(np.isfinite(converted_masses)) or np.any(converted_masses <= 0.0):
            raise ValueError(f"Converted masses from {path} must be finite and positive")
        atoms.set_masses(converted_masses)

    if style == "charge":
        q = np.asarray(charges, dtype=float)[order] * float(
            charge_to_elementary_factor(units_style)
        )
        if q.size != n_atoms or not np.all(np.isfinite(q)):
            raise ValueError(f"Converted charges from {path} must be complete and finite")
        atoms.set_initial_charges(q)

    return atoms



def write_dumpframe_lammps_data(
    path: Path,
    frame: DumpFrame,
    *,
    atom_style: str = "atomic",
    masses_by_type: Optional[dict[int, float]] = None,
    charges_by_id: Optional[dict[int, float]] = None,
    canonical_to_lammps_units_style: Optional[str] = None,
) -> None:
    """Write a raw LAMMPS continuation/data frame.

    ``DumpFrame`` objects used by analysis are canonical (Angstrom).  Pass
    ``canonical_to_lammps_units_style`` at that boundary to convert geometry
    back to the native units consumed by ``read_data``.  Leave it unset only
    for frames deliberately parsed with ``units_style=None``.
    """

    if not isinstance(frame, DumpFrame):
        raise TypeError("frame must be a DumpFrame")

    try:
        ids_numeric = np.asarray(frame.ids, dtype=float).reshape(-1)
        types_numeric = np.asarray(frame.types, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("DumpFrame ids and types must be numeric integers") from exc
    if not np.all(np.isfinite(ids_numeric)) or not np.all(ids_numeric == np.floor(ids_numeric)):
        raise ValueError("DumpFrame ids must be finite integers")
    if not np.all(np.isfinite(types_numeric)) or not np.all(
        types_numeric == np.floor(types_numeric)
    ):
        raise ValueError("DumpFrame types must be finite integers")
    ids = ids_numeric.astype(int)
    types = types_numeric.astype(int)
    pos = np.asarray(frame.positions, dtype=float)
    if pos.shape != (ids.size, 3) or types.size != ids.size:
        raise ValueError("Inconsistent DumpFrame array sizes")
    if ids.size == 0:
        raise ValueError("DumpFrame must contain at least one atom")
    if np.any(ids <= 0) or np.unique(ids).size != ids.size:
        raise ValueError("DumpFrame ids must be unique positive integers")
    if np.any(types <= 0):
        raise ValueError("DumpFrame types must be positive integers")
    if not np.all(np.isfinite(pos)):
        raise ValueError("DumpFrame positions must be finite")

    order = np.argsort(ids)
    ids = ids[order]
    types = types[order]
    pos = pos[order]

    cell = np.asarray(frame.cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("frame.cell must be 3x3")
    if not np.all(np.isfinite(cell)):
        raise ValueError("frame.cell must be finite")
    cell_scale = float(np.max(np.abs(cell)))
    det = float(np.linalg.det(cell))
    det_tol = 128.0 * np.finfo(float).eps * max(cell_scale**3, np.finfo(float).tiny)
    if not math.isfinite(det) or abs(det) <= det_tol:
        raise ValueError("Invalid or degenerate cell for LAMMPS data output")

    origin = np.asarray(frame.origin, dtype=float).reshape(3)
    if not np.all(np.isfinite(origin)):
        raise ValueError("frame.origin must be finite")
    if canonical_to_lammps_units_style is not None:
        length_scale = float(length_from_angstrom_factor(canonical_to_lammps_units_style))
        if not math.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("canonical-to-LAMMPS length scale must be finite and > 0")
        pos = pos * length_scale
        cell = cell * length_scale
        origin = origin * length_scale
    if not (
        np.all(np.isfinite(pos))
        and np.all(np.isfinite(cell))
        and np.all(np.isfinite(origin))
    ):
        raise ValueError("Converted DumpFrame geometry must be finite")
    pos0 = pos - origin[None, :]

    # cell lammps representation
    B = cell.T
    Q, R = np.linalg.qr(B)
    for i in range(3):
        if R[i, i] < 0:
            R[i, :] *= -1.0
            Q[:, i] *= -1.0

    pos_rot = pos0 @ Q
    invR = np.linalg.inv(R)
    s = pos_rot @ invR.T
    s = s - np.floor(s)
    pos_wrap = s @ R.T

    lx = float(R[0, 0])
    ly = float(R[1, 1])
    lz = float(R[2, 2])
    xy = float(R[0, 1])
    xz = float(R[0, 2])
    yz = float(R[1, 2])
    if not (lx > 0.0 and ly > 0.0 and lz > 0.0):
        raise ValueError("Non-positive prism lengths after cell conversion")

    n_atoms = int(ids.size)
    n_types = int(np.max(types)) if n_atoms > 0 else 0
    masses: dict[int, float] = {}
    for raw_type, raw_mass in dict(masses_by_type or {}).items():
        try:
            type_float = float(raw_type)
            mass = float(raw_mass)
        except (TypeError, ValueError) as exc:
            raise ValueError("masses_by_type must map integer types to numeric masses") from exc
        if not math.isfinite(type_float) or not type_float.is_integer() or type_float <= 0:
            raise ValueError(f"Invalid mass atom type: {raw_type!r}")
        atom_type = int(type_float)
        if atom_type > n_types:
            raise ValueError(
                f"Mass atom type {atom_type} exceeds maximum frame type {n_types}"
            )
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"Mass for atom type {atom_type} must be finite and positive"
            )
        masses[atom_type] = mass

    normalized_atom_style = str(atom_style).strip().lower()
    if normalized_atom_style not in {"atomic", "charge"}:
        raise ValueError("atom_style must be 'atomic' or 'charge'")
    use_charge = normalized_atom_style == "charge"
    if use_charge:
        qmap = dict(charges_by_id or {})
        missing = [int(i) for i in ids.tolist() if int(i) not in qmap]
        if missing:
            raise ValueError(
                "write_dumpframe_lammps_data requires charges for all atoms when atom_style='charge'; "
                f"missing ids include {missing[:5]}"
            )
        invalid_charges = [
            int(i)
            for i in ids.tolist()
            if not math.isfinite(float(qmap[int(i)]))
        ]
        if invalid_charges:
            raise ValueError(
                "write_dumpframe_lammps_data requires finite charges; invalid ids "
                f"include {invalid_charges[:5]}"
            )
    else:
        qmap = {}

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write("vitriflow dump-frame data\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"0.0 {lx:.16g} xlo xhi\n")
        f.write(f"0.0 {ly:.16g} ylo yhi\n")
        f.write(f"0.0 {lz:.16g} zlo zhi\n")
        if abs(xy) > 0.0 or abs(xz) > 0.0 or abs(yz) > 0.0:
            f.write(f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz\n")
        f.write("\n")

        if masses:
            f.write("Masses\n\n")
            for it in range(1, n_types + 1):
                mv = masses.get(int(it), None)
                if mv is None:
                    continue
                f.write(f"{int(it)} {float(mv):.16g}\n")
            f.write("\n")

        sec = "charge" if use_charge else "atomic"
        f.write(f"Atoms # {sec}\n\n")
        for i, t, xyz in zip(ids.tolist(), types.tolist(), pos_wrap.tolist()):
            if use_charge:
                q = float(qmap[int(i)])
                f.write(
                    f"{int(i)} {int(t)} {q:.16g} {float(xyz[0]):.16g} {float(xyz[1]):.16g} {float(xyz[2]):.16g}\n"
                )
            else:
                f.write(
                    f"{int(i)} {int(t)} {float(xyz[0]):.16g} {float(xyz[1]):.16g} {float(xyz[2]):.16g}\n"
                )
