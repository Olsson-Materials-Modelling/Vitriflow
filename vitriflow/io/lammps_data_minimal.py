from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..analysis.dump import DumpFrame


_RE_ATOMS = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.IGNORECASE)
_RE_TYPES = re.compile(r"^\s*(\d+)\s+atom\s+types\s*$", re.IGNORECASE)


def read_lammps_data_minimal(
    path: Path,
    *,
    atom_style: str = "atomic",
    specorder: Optional[Sequence[str]] = None,
):
    """Lammps data minimal."""

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
    if xhi is None or yhi is None or zhi is None:
        raise ValueError(f"Failed to parse box bounds (xlo/xhi etc.) from {path}")

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
                try:
                    atoms_style_in_file = raw.split("#", 1)[1].strip().split()[0].lower()
                except Exception:
                    atoms_style_in_file = None

    if idx_masses is not None:
        for raw in lines[idx_masses + 1 :]:
            ln = _strip(raw)
            if not ln:
                continue
            head = ln.split()[0].lower()
            if head in {"atoms", "velocities", "bonds", "angles", "dihedrals", "impropers"}:
                break
            toks = ln.split()
            if len(toks) >= 2:
                try:
                    t = int(float(toks[0]))
                    m = float(toks[1])
                    if t >= 1:
                        masses_by_type[int(t)] = float(m)
                except Exception:
                    pass

    if idx_atoms is None:
        raise ValueError(f"Failed to find 'Atoms' section in {path}")

    style = str(atom_style).strip().lower()
    if atoms_style_in_file in {"atomic", "charge"}:
        style = atoms_style_in_file

    ids: list[int] = []
    types: list[int] = []
    pos: list[list[float]] = []
    charges: list[float] = []

    for raw in lines[idx_atoms + 1 :]:
        ln = _strip(raw)
        if not ln:
            continue
        head = ln.split()[0].lower()
        if head in {"velocities", "bonds", "angles", "dihedrals", "impropers", "masses"}:
            break
        toks = ln.split()
        if style == "charge":
            if len(toks) < 6:
                continue
            i0 = int(float(toks[0]))
            t0 = int(float(toks[1]))
            q0 = float(toks[2])
            x, y, z = float(toks[3]), float(toks[4]), float(toks[5])
            ids.append(i0)
            types.append(t0)
            charges.append(q0)
            pos.append([x - xlo, y - ylo, z - zlo])
        else:
            if len(toks) < 5:
                continue
            i0 = int(float(toks[0]))
            t0 = int(float(toks[1]))
            x, y, z = float(toks[2]), float(toks[3]), float(toks[4])
            ids.append(i0)
            types.append(t0)
            pos.append([x - xlo, y - ylo, z - zlo])

    if len(ids) == 0:
        raise ValueError(f"Failed to parse any atoms from {path}")

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

    atoms = Atoms(symbols=symbols, positions=pos_arr, cell=cell, pbc=True)

    # assign atom masses
    try:
        if masses_by_type:
            m = [float(masses_by_type.get(int(t), atoms[i].mass)) for i, t in enumerate(types_arr.tolist())]
            atoms.set_masses(m)
    except Exception:
        pass

    if style == "charge":
        try:
            q = np.asarray(charges, dtype=float)[order]
            atoms.set_initial_charges(q)
        except Exception:
            pass

    return atoms



def write_dumpframe_lammps_data(
    path: Path,
    frame: DumpFrame,
    *,
    atom_style: str = "atomic",
    masses_by_type: Optional[dict[int, float]] = None,
    charges_by_id: Optional[dict[int, float]] = None,
) -> None:
    """Dumpframe lammps data."""

    if not isinstance(frame, DumpFrame):
        raise TypeError("frame must be a DumpFrame")

    ids = np.asarray(frame.ids, dtype=int).reshape(-1)
    types = np.asarray(frame.types, dtype=int).reshape(-1)
    pos = np.asarray(frame.positions, dtype=float)
    if pos.shape != (ids.size, 3) or types.size != ids.size:
        raise ValueError("Inconsistent DumpFrame array sizes")

    order = np.argsort(ids)
    ids = ids[order]
    types = types[order]
    pos = pos[order]

    cell = np.asarray(frame.cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("frame.cell must be 3x3")
    if not np.all(np.isfinite(cell)):
        raise ValueError("frame.cell must be finite")
    if abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("Invalid or degenerate cell for LAMMPS data output")

    origin = np.asarray(frame.origin, dtype=float).reshape(3)
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
    masses = dict(masses_by_type or {})

    use_charge = str(atom_style).strip().lower() == "charge"
    if use_charge:
        qmap = dict(charges_by_id or {})
        missing = [int(i) for i in ids.tolist() if int(i) not in qmap]
        if missing:
            raise ValueError(
                "write_dumpframe_lammps_data requires charges for all atoms when atom_style='charge'; "
                f"missing ids include {missing[:5]}"
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
