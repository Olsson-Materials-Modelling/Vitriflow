from __future__ import annotations

import math
import re
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from .dump import DumpFrame
from ..lammps_units import length_to_angstrom_factor


_ATOMS_RE = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.IGNORECASE)
_ATOM_TYPES_RE = re.compile(r"^\s*(\d+)\s+atom\s+types\s*$", re.IGNORECASE)


_LAMMPS_DATA_SECTION_HEADERS = {
    "atoms",
    "velocities",
    "masses",
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
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "ellipsoids",
    "lines",
    "triangles",
    "bodies",
}
_PAIR_COEFF_SECTION_HEADERS = {"pair coeffs", "pairij coeffs"}


def _normalized_lammps_data_section_header(raw: str) -> str | None:
    ln = raw.split("#", 1)[0].strip()
    if not ln:
        return None
    key = re.sub(r"\s+", " ", ln).strip().lower()
    if key in _LAMMPS_DATA_SECTION_HEADERS:
        return key
    return None


def _positive_exact_integer(
    token: object,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> int:
    try:
        value = Decimal(
            str(token).strip().replace("D", "E").replace("d", "e")
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"Invalid {field} at {path}:{line_number}: expected a positive exact integer, got {token!r}"
        ) from exc
    if not value.is_finite() or value != value.to_integral_value() or value <= 0:
        raise ValueError(
            f"Invalid {field} at {path}:{line_number}: expected a positive exact integer, got {token!r}"
        )
    if value > int(np.iinfo(np.intp).max):
        raise ValueError(
            f"Invalid {field} at {path}:{line_number}: integer {token!r} exceeds platform index range"
        )
    return int(value)


def strip_lammps_data_pair_coeff_sections(path: Path) -> int:
    """Remove Pair Coeffs / PairIJ Coeffs sections from a LAMMPS data file.

    Vitriflow treats the YAML potential block as the source of truth for pair
    interactions. When a prior LAMMPS stage writes a serialisable force field via
    ``write_data``, the resulting data file can gain a ``Pair Coeffs`` section.
    A later stage then fails at ``read_data`` because Vitriflow defines the pair
    style after reading the structure. Stripping embedded pair-coefficient
    sections keeps the structure file restart-safe without altering atoms,
    charges, masses, box bounds, or velocities.

    Returns the number of stripped pair-coefficient sections.
    """

    src = Path(path)
    tmp: Path | None = None
    removed = 0
    in_removed_section = False

    try:
        fd, raw_tmp = tempfile.mkstemp(
            dir=str(src.parent), prefix=f".{src.name}.", suffix=".paircoeff.tmp", text=True
        )
        tmp = Path(raw_tmp)
        with src.open("r", errors="replace") as fin, os.fdopen(fd, "w") as fout:
            for raw in fin:
                head = _normalized_lammps_data_section_header(raw)

                if in_removed_section:
                    if head is None:
                        continue
                    if head in _PAIR_COEFF_SECTION_HEADERS:
                        removed += 1
                        continue
                    in_removed_section = False

                if head in _PAIR_COEFF_SECTION_HEADERS:
                    removed += 1
                    in_removed_section = True
                    continue

                fout.write(raw)
            fout.flush()
            os.fsync(fout.fileno())

        if removed > 0:
            os.replace(tmp, src)
            tmp = None
        else:
            try:
                tmp.unlink()
                tmp = None
            except OSError:
                pass
        return int(removed)
    except Exception:
        try:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def datafile_has_velocities(path: Path, *, max_lines: int = 200000) -> bool:
    """Datafile has velocities."""

    try:
        n_atoms: int | None = None
        in_vel = False
        ids: set[int] = set()

        with Path(path).open("r", errors="replace") as f:
            lines_scanned = 0
            source = Path(path)
            for line_number, raw in enumerate(f, start=1):
                lines_scanned += 1
                if lines_scanned > int(max_lines):
                    break

                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue

                ln0 = ln.split("#", 1)[0].strip()
                if not ln0:
                    continue

                if not in_vel:
                    m = _ATOMS_RE.match(ln0)
                    if m:
                        try:
                            n_atoms = int(m.group(1))
                        except Exception:
                            n_atoms = None

                    head = ln0.split()[0].lower()
                    if head == "velocities":
                        in_vel = True
                    continue

                # velocities section
                toks = ln0.split()
                if len(toks) >= 4:
                    try:
                        i = _positive_exact_integer(
                            toks[0],
                            field="velocity atom id",
                            path=source,
                            line_number=line_number,
                        )
                        velocity = np.asarray(
                            [float(toks[1]), float(toks[2]), float(toks[3])],
                            dtype=float,
                        )
                        if np.all(np.isfinite(velocity)):
                            ids.add(i)
                            if n_atoms is not None and len(ids) >= int(n_atoms):
                                return True
                    except Exception:
                        pass

                # another section header
                head = toks[0].lower() if toks else ""
                if len(toks) == 1 and head in {
                    "atoms",
                    "masses",
                    "bonds",
                    "angles",
                    "dihedrals",
                    "impropers",
                    "pair",
                    "pairij",
                }:
                    break

        if n_atoms is not None:
            return len(ids) >= int(n_atoms) and int(n_atoms) > 0
        return len(ids) > 0

    except Exception:
        return False


def read_datafile_frame(
    path: Path,
    *,
    atom_style: str = "atomic",
    units_style: str = "metal",
) -> DumpFrame:
    """Strictly read a LAMMPS data frame in canonical Angstrom geometry."""

    source = Path(path)
    txt = source.read_text(errors="replace")
    lines = txt.splitlines()

    def _strip(ln: str) -> str:
        return ln.split("#", 1)[0].strip()

    def _native_float(token: str, *, field: str, line_number: int) -> float:
        try:
            value = float(str(token).replace("D", "E").replace("d", "e"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid {field} at {source}:{line_number}: {token!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite {field} at {source}:{line_number}: {token!r}"
            )
        return float(value)

    def _positive_integer(token: str, *, field: str, line_number: int) -> int:
        return _positive_exact_integer(
            token,
            field=field,
            path=source,
            line_number=line_number,
        )

    # Header counts and geometry are required evidence, not hints.
    n_atoms: int | None = None
    n_types: int | None = None
    xlo = ylo = zlo = 0.0
    xhi = yhi = zhi = None  # type: ignore[assignment]
    xy = xz = yz = 0.0
    seen_bounds: set[str] = set()
    seen_tilt = False

    for line_number, raw in enumerate(lines[:400], start=1):
        ln = _strip(raw)
        if not ln:
            continue
        match = _ATOMS_RE.match(ln)
        if match:
            if n_atoms is not None:
                raise ValueError(f"Duplicate atom-count header at {source}:{line_number}")
            n_atoms = int(match.group(1))
            if n_atoms > int(np.iinfo(np.intp).max):
                raise ValueError(f"Atom count exceeds platform index range at {source}:{line_number}")
            continue
        match = _ATOM_TYPES_RE.match(ln)
        if match:
            if n_types is not None:
                raise ValueError(f"Duplicate atom-type-count header at {source}:{line_number}")
            n_types = int(match.group(1))
            if n_types > int(np.iinfo(np.intp).max):
                raise ValueError(f"Atom-type count exceeds platform index range at {source}:{line_number}")
            continue
        toks = ln.split()
        if len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["xlo", "xhi"]:
            if "x" in seen_bounds:
                raise ValueError(f"Duplicate x bounds at {source}:{line_number}")
            xlo = _native_float(toks[0], field="xlo", line_number=line_number)
            xhi = _native_float(toks[1], field="xhi", line_number=line_number)
            seen_bounds.add("x")
        elif len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["ylo", "yhi"]:
            if "y" in seen_bounds:
                raise ValueError(f"Duplicate y bounds at {source}:{line_number}")
            ylo = _native_float(toks[0], field="ylo", line_number=line_number)
            yhi = _native_float(toks[1], field="yhi", line_number=line_number)
            seen_bounds.add("y")
        elif len(toks) >= 4 and [t.lower() for t in toks[-2:]] == ["zlo", "zhi"]:
            if "z" in seen_bounds:
                raise ValueError(f"Duplicate z bounds at {source}:{line_number}")
            zlo = _native_float(toks[0], field="zlo", line_number=line_number)
            zhi = _native_float(toks[1], field="zhi", line_number=line_number)
            seen_bounds.add("z")
        elif len(toks) >= 6 and [t.lower() for t in toks[-3:]] == ["xy", "xz", "yz"]:
            if seen_tilt:
                raise ValueError(f"Duplicate tilt-factor header at {source}:{line_number}")
            xy = _native_float(toks[0], field="xy", line_number=line_number)
            xz = _native_float(toks[1], field="xz", line_number=line_number)
            yz = _native_float(toks[2], field="yz", line_number=line_number)
            seen_tilt = True

    if n_atoms is None or n_atoms < 1:
        raise ValueError(f"LAMMPS data file must declare a positive '<N> atoms' header: {source}")
    if n_types is None or n_types < 1:
        raise ValueError(f"LAMMPS data file must declare a positive '<N> atom types' header: {source}")
    if xhi is None or yhi is None or zhi is None:
        raise ValueError(f"Failed to parse box bounds from data file: {source}")

    lx = float(xhi - xlo)
    ly = float(yhi - ylo)
    lz = float(zhi - zlo)
    if not (lx > 0.0 and ly > 0.0 and lz > 0.0):
        raise ValueError(f"Non-positive cell lengths parsed from data file: {source}")

    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )
    origin = np.array([float(xlo), float(ylo), float(zlo)], dtype=float)
    cell_scale = float(np.max(np.abs(cell)))
    determinant = float(np.linalg.det(cell))
    determinant_tol = 128.0 * np.finfo(float).eps * max(
        cell_scale**3,
        np.finfo(float).tiny,
    )
    if (
        not np.all(np.isfinite(cell))
        or not np.all(np.isfinite(origin))
        or not math.isfinite(determinant)
        or abs(determinant) <= determinant_tol
    ):
        raise ValueError(f"LAMMPS cell in {source} must be finite and nonsingular")

    # locate atoms section
    idx_atoms: int | None = None
    atoms_style_in_file: str | None = None
    for i, raw in enumerate(lines):
        if _normalized_lammps_data_section_header(raw) == "atoms":
            idx_atoms = i
            if "#" in raw:
                comment = raw.split("#", 1)[1].strip().split()
                atoms_style_in_file = comment[0].lower() if comment else None
            break

    if idx_atoms is None:
        raise ValueError(f"Failed to find 'Atoms' section in data file: {source}")

    style = str(atom_style).strip().lower()
    if atoms_style_in_file is not None:
        if atoms_style_in_file not in {"atomic", "charge"}:
            raise ValueError(
                f"Unsupported LAMMPS Atoms style {atoms_style_in_file!r} in {source}; "
                "only atomic and charge are supported"
            )
        style = atoms_style_in_file
    if style not in {"atomic", "charge"}:
        raise ValueError(
            f"Unsupported LAMMPS atom_style {atom_style!r}; only atomic and charge are supported"
        )

    ids: list[int] = []
    types: list[int] = []
    pos: list[list[float]] = []

    for line_number, raw in enumerate(lines[idx_atoms + 1 :], start=idx_atoms + 2):
        ln = _strip(raw)
        if not ln:
            continue
        if _normalized_lammps_data_section_header(raw) is not None:
            break

        toks = ln.split()
        if style == "charge":
            if len(toks) < 6:
                raise ValueError(
                    f"Malformed charge atom row at {source}:{line_number}: expected id, type, charge, x, y, z"
                )
            i0 = _positive_integer(toks[0], field="atom id", line_number=line_number)
            t0 = _positive_integer(toks[1], field="atom type", line_number=line_number)
            _native_float(toks[2], field="charge", line_number=line_number)
            x = _native_float(toks[3], field="x coordinate", line_number=line_number)
            y = _native_float(toks[4], field="y coordinate", line_number=line_number)
            z = _native_float(toks[5], field="z coordinate", line_number=line_number)
        else:
            if len(toks) < 5:
                raise ValueError(
                    f"Malformed atomic atom row at {source}:{line_number}: expected id, type, x, y, z"
                )
            i0 = _positive_integer(toks[0], field="atom id", line_number=line_number)
            t0 = _positive_integer(toks[1], field="atom type", line_number=line_number)
            x = _native_float(toks[2], field="x coordinate", line_number=line_number)
            y = _native_float(toks[3], field="y coordinate", line_number=line_number)
            z = _native_float(toks[4], field="z coordinate", line_number=line_number)

        ids.append(i0)
        types.append(t0)
        pos.append([x, y, z])

    if len(ids) != n_atoms:
        raise ValueError(
            f"Atom-count mismatch in {source}: header declares {n_atoms}, parsed {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Atom IDs in {source} must be unique positive integers")
    if any(atom_type > n_types for atom_type in types):
        raise ValueError(f"Atom types in {source} must lie in [1, {n_types}]")
    if not np.all(np.isfinite(np.asarray(pos, dtype=float))):
        raise ValueError(f"Atom positions in {source} must be finite")

    order = np.argsort(np.asarray(ids, dtype=int))
    ids_arr = np.asarray(ids, dtype=int)[order]
    types_arr = np.asarray(types, dtype=int)[order]
    pos_arr = np.asarray(pos, dtype=float)[order]

    length_factor = float(length_to_angstrom_factor(units_style))
    if not math.isfinite(length_factor) or length_factor <= 0.0:
        raise ValueError(f"Invalid length conversion for LAMMPS units_style={units_style!r}")
    return DumpFrame(
        timestep=0,
        ids=ids_arr,
        types=types_arr,
        positions=pos_arr * length_factor,
        cell=np.asarray(cell, dtype=float) * length_factor,
        origin=np.asarray(origin, dtype=float) * length_factor,
    )



def read_datafile_masses(path: Path, *, max_lines: int = 200000) -> dict[int, float]:
    """Datafile masses."""

    source = Path(path)
    masses: dict[int, float] = {}
    in_masses = False
    n_types: int | None = None
    with source.open("r", errors="replace") as f:
        for line_number, raw in enumerate(f, start=1):
            if line_number > int(max_lines):
                if in_masses:
                    raise ValueError(
                        f"Masses section in {source} exceeds max_lines={int(max_lines)}"
                    )
                break
            ln = raw.split("#", 1)[0].strip()
            if not ln:
                continue
            if not in_masses:
                match = _ATOM_TYPES_RE.match(ln)
                if match:
                    n_types = int(match.group(1))
                if _normalized_lammps_data_section_header(raw) == "masses":
                    in_masses = True
                continue
            section = _normalized_lammps_data_section_header(raw)
            if section is not None:
                break
            toks = ln.split()
            if len(toks) < 2:
                raise ValueError(
                    f"Malformed mass row at {source}:{line_number}: expected type and mass"
                )
            atom_type = _positive_exact_integer(
                toks[0],
                field="mass atom type",
                path=source,
                line_number=line_number,
            )
            try:
                mass = float(toks[1])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid mass at {source}:{line_number}: {toks[1]!r}"
                ) from exc
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(
                    f"Mass must be finite and positive at {source}:{line_number}"
                )
            if n_types is not None and atom_type > n_types:
                raise ValueError(
                    f"Mass atom type {atom_type} at {source}:{line_number} exceeds declared {n_types} atom types"
                )
            if atom_type in masses:
                raise ValueError(
                    f"Duplicate mass entry for atom type {atom_type} at {source}:{line_number}"
                )
            masses[atom_type] = float(mass)
    return masses


def read_datafile_charges(path: Path, *, atom_style: str = "atomic") -> dict[int, float]:
    """Datafile charges."""

    source = Path(path)
    txt = source.read_text(errors="replace")
    lines = txt.splitlines()

    def _strip(ln: str) -> str:
        return ln.split("#", 1)[0].strip()

    idx_atoms: int | None = None
    atoms_style_in_file: str | None = None
    for i, raw in enumerate(lines):
        ln = _strip(raw)
        if not ln:
            continue
        head = ln.split()[0].lower()
        if head == "atoms":
            idx_atoms = i
            if "#" in raw:
                try:
                    atoms_style_in_file = raw.split("#", 1)[1].strip().split()[0].lower()
                except Exception:
                    atoms_style_in_file = None
            break
    if idx_atoms is None:
        return {}

    style = str(atom_style).strip().lower()
    if atoms_style_in_file in {"atomic", "charge"}:
        style = str(atoms_style_in_file)
    if style != "charge":
        return {}

    # Validate complete atom/count/type/geometry evidence before extracting a
    # field that otherwise could appear plausible from only a subset of rows.
    frame = read_datafile_frame(source, atom_style="charge", units_style="metal")

    charges: dict[int, float] = {}
    for line_number, raw in enumerate(lines[idx_atoms + 1 :], start=idx_atoms + 2):
        ln = _strip(raw)
        if not ln:
            continue
        toks = ln.split()
        if _normalized_lammps_data_section_header(raw) is not None:
            break
        if len(toks) < 6:
            raise ValueError(
                f"Malformed charge atom row at {source}:{line_number}: expected id, type, charge, x, y, z"
            )
        try:
            i0 = _positive_exact_integer(
                toks[0],
                field="charge atom id",
                path=source,
                line_number=line_number,
            )
            q0 = float(toks[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid charge row at {source}:{line_number}") from exc
        if not math.isfinite(q0):
            raise ValueError(f"Charge must be finite at {source}:{line_number}")
        if i0 in charges:
            raise ValueError(f"Duplicate charge atom id {i0} at {source}:{line_number}")
        charges[i0] = float(q0)
    if set(charges) != set(int(value) for value in frame.ids.tolist()):
        raise ValueError(f"Charge rows in {source} do not cover exactly the declared atoms")
    return charges

def count_atoms_in_datafile(path: Path) -> int:
    """Count atoms in."""
    for ln in path.read_text(errors="replace").splitlines()[:200]:
        m = _ATOMS_RE.match(ln)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not find '<N> atoms' header in {path}")


def datafile_has_masses(path: Path, *, max_lines: int = 20000) -> bool:
    """Datafile has masses."""

    try:
        with path.open("r", errors="replace") as f:
            in_masses = False
            lines_scanned = 0
            ntypes: int | None = None
            mass_types: set[int] = set()

            for raw in f:
                lines_scanned += 1
                if lines_scanned > int(max_lines):
                    break

                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue

                # trailing comments tokenization
                ln0 = ln.split("#", 1)[0].strip()
                if not ln0:
                    continue

                if not in_masses:
                    m_types = _ATOM_TYPES_RE.match(ln0)
                    if m_types:
                        try:
                            ntypes = int(m_types.group(1))
                        except Exception:
                            ntypes = None

                    head = ln0.split()[0].lower()
                    if head == "masses":
                        in_masses = True
                    continue

                # masses section
                toks = ln0.split()
                if len(toks) >= 2:
                    try:
                        t = _positive_exact_integer(
                            toks[0],
                            field="mass atom type",
                            path=Path(path),
                            line_number=lines_scanned,
                        )
                        mass = float(toks[1])
                        if np.isfinite(mass) and mass > 0.0:
                            mass_types.add(t)
                            if ntypes is not None and len(mass_types) >= ntypes:
                                return True
                    except Exception:
                        pass

                # another section header
                head = toks[0].lower() if toks else ""
                if len(toks) == 1 and head in {
                    "atoms",
                    "velocities",
                    "bonds",
                    "angles",
                    "dihedrals",
                    "impropers",
                    "pair",
                    "pairij",
                }:
                    break

            # scan decide based
            if ntypes is not None:
                return len(mass_types) >= ntypes and ntypes > 0
            return len(mass_types) > 0

    except Exception:
        return False
