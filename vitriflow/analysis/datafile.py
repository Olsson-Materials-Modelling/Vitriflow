from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .dump import DumpFrame


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
    tmp = src.with_name(src.name + ".paircoeff_stripped.tmp")
    removed = 0
    in_removed_section = False

    try:
        with src.open("r", errors="replace") as fin, tmp.open("w") as fout:
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

        if removed > 0:
            tmp.replace(src)
        else:
            try:
                tmp.unlink()
            except OSError:
                pass
        return int(removed)
    except Exception:
        try:
            if tmp.exists():
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
            for raw in f:
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
                        i = int(float(toks[0]))
                        float(toks[1]); float(toks[2]); float(toks[3])
                        if i > 0:
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


def read_datafile_frame(path: Path, *, atom_style: str = "atomic") -> DumpFrame:
    """Datafile frame."""

    txt = Path(path).read_text(errors="replace")
    lines = txt.splitlines()

    # header scan
    xlo = ylo = zlo = 0.0
    xhi = yhi = zhi = None  # type: ignore[assignment]
    xy = xz = yz = 0.0

    def _strip(ln: str) -> str:
        return ln.split("#", 1)[0].strip()

    for raw in lines[:400]:
        ln = _strip(raw)
        if not ln:
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

    if xhi is None or yhi is None or zhi is None:
        raise ValueError(f"Failed to parse box bounds from data file: {path}")

    lx = float(xhi - xlo)
    ly = float(yhi - ylo)
    lz = float(zhi - zlo)
    if not (lx > 0.0 and ly > 0.0 and lz > 0.0):
        raise ValueError(f"Non-positive cell lengths parsed from data file: {path}")

    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )
    origin = np.array([float(xlo), float(ylo), float(zlo)], dtype=float)

    # locate atoms section
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
        raise ValueError(f"Failed to find 'Atoms' section in data file: {path}")

    style = str(atom_style).strip().lower()
    if atoms_style_in_file in {"atomic", "charge"}:
        style = str(atoms_style_in_file)

    ids: list[int] = []
    types: list[int] = []
    pos: list[list[float]] = []

    stop_heads = {
        "velocities",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "masses",
        "pair",
        "pairij",
    }

    for raw in lines[idx_atoms + 1 :]:
        ln = _strip(raw)
        if not ln:
            continue
        head = ln.split()[0].lower()
        if head in stop_heads:
            break

        toks = ln.split()
        if style == "charge":
            if len(toks) < 6:
                continue
            i0 = int(float(toks[0]))
            t0 = int(float(toks[1]))
            x, y, z = float(toks[3]), float(toks[4]), float(toks[5])
        else:
            if len(toks) < 5:
                continue
            i0 = int(float(toks[0]))
            t0 = int(float(toks[1]))
            x, y, z = float(toks[2]), float(toks[3]), float(toks[4])

        ids.append(i0)
        types.append(t0)
        pos.append([x, y, z])

    if len(ids) == 0:
        raise ValueError(f"Failed to parse any atoms from data file: {path}")

    order = np.argsort(np.asarray(ids, dtype=int))
    ids_arr = np.asarray(ids, dtype=int)[order]
    types_arr = np.asarray(types, dtype=int)[order]
    pos_arr = np.asarray(pos, dtype=float)[order]

    return DumpFrame(
        timestep=0,
        ids=ids_arr,
        types=types_arr,
        positions=pos_arr,
        cell=np.asarray(cell, dtype=float),
        origin=np.asarray(origin, dtype=float),
    )



def read_datafile_masses(path: Path, *, max_lines: int = 200000) -> dict[int, float]:
    """Datafile masses."""

    masses: dict[int, float] = {}
    try:
        with Path(path).open("r", errors="replace") as f:
            in_masses = False
            lines_scanned = 0
            for raw in f:
                lines_scanned += 1
                if lines_scanned > int(max_lines):
                    break
                ln = raw.split("#", 1)[0].strip()
                if not ln:
                    continue
                toks = ln.split()
                head = toks[0].lower()
                if not in_masses:
                    if head == "masses":
                        in_masses = True
                    continue

                if len(toks) >= 2:
                    try:
                        it = int(float(toks[0]))
                        mv = float(toks[1])
                        if it > 0 and np.isfinite(mv) and mv > 0.0:
                            masses[int(it)] = float(mv)
                    except Exception:
                        pass

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
    except Exception:
        return {}
    return masses


def read_datafile_charges(path: Path, *, atom_style: str = "atomic") -> dict[int, float]:
    """Datafile charges."""

    txt = Path(path).read_text(errors="replace")
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

    charges: dict[int, float] = {}
    stop_heads = {
        "velocities",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "masses",
        "pair",
        "pairij",
    }
    for raw in lines[idx_atoms + 1 :]:
        ln = _strip(raw)
        if not ln:
            continue
        toks = ln.split()
        if toks[0].lower() in stop_heads:
            break
        if len(toks) < 6:
            continue
        try:
            i0 = int(float(toks[0]))
            q0 = float(toks[2])
            if i0 > 0 and np.isfinite(q0):
                charges[int(i0)] = float(q0)
        except Exception:
            continue
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
                        t = int(float(toks[0]))
                        float(toks[1])
                        if t > 0:
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
