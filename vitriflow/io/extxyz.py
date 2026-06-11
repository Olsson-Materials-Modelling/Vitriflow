from __future__ import annotations

"""Read/write extended XYZ (EXTXYZ) trajectories.

The goal is to provide a simple, engine-neutral trajectory/structure format that
preserves:
  - periodic cell (3x3 lattice matrix)
  - cartesian coordinates
  - atom ids and integer types (LAMMPS-style) when available
  - chemical symbols when available

This module intentionally implements a small, dependency-free subset of the EXTXYZ
convention used by ASE.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from ..analysis.dump import DumpFrame


_KV_RE = re.compile(r"(\w+)=((?:\"[^\"]*\")|(?:\S+))")


def _parse_kv(comment: str) -> Dict[str, str]:
    """Kv."""
    out: Dict[str, str] = {}
    for m in _KV_RE.finditer(str(comment).strip()):
        k = m.group(1)
        v = m.group(2)
        if v.startswith('"') and v.endswith('"') and len(v) >= 2:
            v = v[1:-1]
        out[str(k)] = str(v)
    return out


@dataclass(frozen=True)
class ExtXYZProperties:
    """Ext xyzproperties."""

    names: list[str]
    types: list[str]
    counts: list[int]

    @property
    def ncols(self) -> int:
        return int(sum(self.counts))

    def slices(self) -> Dict[str, slice]:
        """Slices."""
        m: Dict[str, slice] = {}
        j = 0
        for nm, n in zip(self.names, self.counts):
            m[str(nm)] = slice(j, j + int(n))
            j += int(n)
        return m


def _parse_properties(spec: str) -> ExtXYZProperties:
    parts = str(spec).strip()
    if parts == "":
        return ExtXYZProperties(names=[], types=[], counts=[])
    toks = parts.split(":")
    if len(toks) % 3 != 0:
        raise ValueError(f"Invalid EXTXYZ Properties spec: {spec!r}")
    names: list[str] = []
    types: list[str] = []
    counts: list[int] = []
    for i in range(0, len(toks), 3):
        nm = str(toks[i]).strip()
        tp = str(toks[i + 1]).strip()
        try:
            n = int(toks[i + 2])
        except Exception as e:
            raise ValueError(f"Invalid EXTXYZ Properties count in: {spec!r}") from e
        if nm == "" or tp == "" or n < 1:
            raise ValueError(f"Invalid EXTXYZ Properties field in: {spec!r}")
        names.append(nm)
        types.append(tp)
        counts.append(int(n))
    return ExtXYZProperties(names=names, types=types, counts=counts)


def _parse_lattice(v: str) -> np.ndarray:
    nums = [float(x) for x in str(v).split()]
    if len(nums) != 9:
        raise ValueError("EXTXYZ Lattice must have 9 floats")
    cell = np.asarray(nums, dtype=float).reshape((3, 3))
    if not np.all(np.isfinite(cell)):
        raise ValueError("Non-finite EXTXYZ Lattice")
    return cell


def _format_lattice(cell: np.ndarray) -> str:
    c = np.asarray(cell, dtype=float)
    if c.shape != (3, 3):
        raise ValueError("cell must be 3x3")
    if not np.all(np.isfinite(c)):
        raise ValueError("cell must be finite")
    flat = c.reshape(-1)
    return " ".join(f"{float(x):.16g}" for x in flat.tolist())


def _wrap_positions(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Wrap positions."""
    pos = np.asarray(positions, dtype=float)
    H = np.asarray(cell, dtype=float)
    org = np.asarray(origin, dtype=float)
    invH = np.linalg.inv(H)
    frac = (pos - org) @ invH
    frac = frac - np.floor(frac)
    return frac @ H


def _resolve_type_to_species(
    *,
    type_to_species: Optional[Sequence[str]],
    species: Optional[Sequence[str]],
    func_name: str,
) -> Optional[Sequence[str]]:
    """Type to species."""

    if type_to_species is not None and species is not None:
        left = tuple(str(x) for x in type_to_species)
        right = tuple(str(x) for x in species)
        if left != right:
            raise TypeError(
                f"{func_name} received both 'type_to_species' and legacy 'species' with different values"
            )
    if type_to_species is not None:
        return type_to_species
    if species is not None:
        return species
    return None


def write_extxyz_frames(
    path: Path,
    frames: Sequence[DumpFrame],
    *,
    type_to_species: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    wrap: bool = True,
) -> None:
    """Extxyz frames."""

    if not frames:
        raise ValueError("write_extxyz_frames requires at least one frame")

    resolved_species = _resolve_type_to_species(
        type_to_species=type_to_species,
        species=species,
        func_name="write_extxyz_frames",
    )

    # map integer symbol
    def _sym(t: int) -> str:
        if resolved_species is None:
            return "X"
        i = int(t) - 1
        if i < 0 or i >= len(resolved_species):
            return "X"
        return str(resolved_species[i])

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    pbc_str = " ".join(["T" if bool(x) else "F" for x in pbc])

    with Path(path).open("w") as f:
        for fr in frames:
            n = int(fr.n_atoms)
            cell = np.asarray(fr.cell, dtype=float)
            if cell.shape != (3, 3):
                raise ValueError("Frame cell must be 3x3")
            if not np.all(np.isfinite(cell)):
                raise ValueError("Frame cell must be finite")

            pos = np.asarray(fr.positions, dtype=float)
            if wrap:
                posw = _wrap_positions(positions=pos, cell=cell, origin=np.asarray(fr.origin, dtype=float))
            else:
                posw = pos - np.asarray(fr.origin, dtype=float)

            f.write(f"{n}\n")
            f.write(
                f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={int(fr.timestep)}\n"
            )
            for sym, r, t, i in zip([_sym(x) for x in fr.types.tolist()], posw, fr.types, fr.ids):
                f.write(
                    f"{sym} {float(r[0]):.12f} {float(r[1]):.12f} {float(r[2]):.12f} {int(t)} {int(i)}\n"
                )


def write_extxyz_iter(
    path: Path,
    frames: Iterable[DumpFrame],
    *,
    type_to_species: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    wrap: bool = True,
) -> DumpFrame:
    """Extxyz iter."""

    resolved_species = _resolve_type_to_species(
        type_to_species=type_to_species,
        species=species,
        func_name="write_extxyz_iter",
    )

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    pbc_str = " ".join(["T" if bool(x) else "F" for x in pbc])

    def _sym(t: int) -> str:
        if resolved_species is None:
            return "X"
        i = int(t) - 1
        if i < 0 or i >= len(resolved_species):
            return "X"
        return str(resolved_species[i])

    last: Optional[DumpFrame] = None
    with Path(path).open("w") as f:
        for fr in frames:
            last = fr
            n = int(fr.n_atoms)
            cell = np.asarray(fr.cell, dtype=float)
            if cell.shape != (3, 3):
                raise ValueError("Frame cell must be 3x3")
            if not np.all(np.isfinite(cell)):
                raise ValueError("Frame cell must be finite")

            pos = np.asarray(fr.positions, dtype=float)
            if wrap:
                posw = _wrap_positions(positions=pos, cell=cell, origin=np.asarray(fr.origin, dtype=float))
            else:
                posw = pos - np.asarray(fr.origin, dtype=float)

            f.write(f"{n}\n")
            f.write(
                f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={int(fr.timestep)}\n"
            )
            for sym, r, t, i in zip([_sym(x) for x in fr.types.tolist()], posw, fr.types, fr.ids):
                f.write(
                    f"{sym} {float(r[0]):.12f} {float(r[1]):.12f} {float(r[2]):.12f} {int(t)} {int(i)}\n"
                )

    if last is None:
        raise ValueError("No frames written to EXTXYZ")
    return last


def write_extxyz_single(
    path: Path,
    frame: DumpFrame,
    *,
    type_to_species: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    wrap: bool = True,
) -> None:
    """Extxyz single."""
    write_extxyz_frames(
        path,
        [frame],
        type_to_species=type_to_species,
        species=species,
        pbc=pbc,
        wrap=wrap,
    )


def write_extxyz_single_with_species(
    path: Path,
    frame: DumpFrame,
    species: Sequence[str],
    *,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    wrap: bool = True,
) -> None:
    """Extxyz single with."""

    n = int(frame.n_atoms)
    if len(species) != n:
        raise ValueError(f"species override length {len(species)} != n_atoms {n}")

    cell = np.asarray(frame.cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("Frame cell must be 3x3")
    if not np.all(np.isfinite(cell)):
        raise ValueError("Frame cell must be finite")

    pos = np.asarray(frame.positions, dtype=float)
    if wrap:
        posw = _wrap_positions(positions=pos, cell=cell, origin=np.asarray(frame.origin, dtype=float))
    else:
        posw = pos - np.asarray(frame.origin, dtype=float)

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    pbc_str = " ".join(["T" if bool(x) else "F" for x in pbc])

    with Path(path).open("w") as f:
        f.write(f"{n}\n")
        f.write(
            f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={int(frame.timestep)}\n"
        )
        for sym, r, t, i in zip([str(s) for s in species], posw, frame.types, frame.ids):
            f.write(
                f"{sym} {float(r[0]):.12f} {float(r[1]):.12f} {float(r[2]):.12f} {int(t)} {int(i)}\n"
            )


def _type_map_from_species_order(type_to_species: Optional[Sequence[str]]) -> Optional[Dict[str, int]]:
    """Build a case-tolerant species -> type index map from a configured order."""

    if type_to_species is None:
        return None
    out: Dict[str, int] = {}
    for idx, sym in enumerate(list(type_to_species), start=1):
        s = str(sym).strip()
        if not s:
            continue
        out[s] = int(idx)
        out[s.lower()] = int(idx)
    return out or None


def _type_from_symbol(
    sym: str,
    *,
    species_to_type: Optional[Dict[str, int]],
    fallback_map: Dict[str, int],
    next_type_ref: list[int],
) -> int:
    """Resolve an EXTXYZ symbol to an integer type.

    When an analysis config supplies ``type_to_species``, use that mapping so
    generic EXTXYZ files without a ``type`` property do not depend on the atom
    order inside each file.  Without a configured order, preserve the legacy
    first-seen mapping.
    """

    s = str(sym).strip()
    if species_to_type is not None:
        if s in species_to_type:
            return int(species_to_type[s])
        low = s.lower()
        if low in species_to_type:
            return int(species_to_type[low])
        raise ValueError(f"EXTXYZ species {s!r} is not present in configured type_to_species")
    if s not in fallback_map:
        fallback_map[s] = int(next_type_ref[0])
        next_type_ref[0] += 1
    return int(fallback_map[s])


def iter_extxyz_frames(path: Path, *, type_to_species: Optional[Sequence[str]] = None) -> Iterator[DumpFrame]:
    """Iter extxyz frames."""

    p = Path(path)
    species_to_type = _type_map_from_species_order(type_to_species)
    with p.open("r", errors="replace") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            if line == "":
                continue
            try:
                nat = int(line.split()[0])
            except Exception as e:
                raise ValueError(f"Invalid EXTXYZ atom count line in {p}: {line!r}") from e
            comment = f.readline()
            if comment == "":
                raise ValueError(f"Truncated EXTXYZ file {p}: missing comment line")
            kv = _parse_kv(comment)
            if "Lattice" not in kv:
                raise ValueError(f"EXTXYZ frame missing Lattice in {p}")
            cell = _parse_lattice(kv["Lattice"])
            step = int(float(kv.get("Step", "0")))

            props_spec = kv.get("Properties", "")
            props = _parse_properties(props_spec) if props_spec else ExtXYZProperties([], [], [])
            sl = props.slices()

            # determine expected token
            expected_cols = props.ncols
            have_props = expected_cols > 0
            if not have_props:
                expected_cols = 4  # xyz sym x

            syms: list[str] = []
            pos = np.empty((nat, 3), dtype=float)
            types = np.empty((nat,), dtype=int)
            ids = np.empty((nat,), dtype=int)

            # fallback for EXTXYZ files without an explicit type column.
            sym_to_type: Dict[str, int] = {}
            next_type_ref = [1]

            for i in range(nat):
                raw = f.readline()
                if raw == "":
                    raise ValueError(f"Truncated EXTXYZ file {p}: missing atom lines")
                toks = raw.split()
                if len(toks) != expected_cols:
                    raise ValueError(
                        f"Invalid EXTXYZ atom line (expected {expected_cols} tokens) in {p}: {raw!r}"
                    )

                if have_props:
                    # dense token vector
                    # extxyz corresponds flattened
                    # we species pos
                    # species
                    if "species" in sl:
                        sym = toks[sl["species"].start]
                    else:
                        sym = toks[0]
                    syms.append(str(sym))
                    # pos
                    if "pos" not in sl:
                        raise ValueError(f"EXTXYZ Properties missing 'pos' in {p}")
                    ps = sl["pos"]
                    try:
                        pos[i, :] = [float(toks[ps.start + j]) for j in range(3)]
                    except Exception as e:
                        raise ValueError(f"Invalid pos columns in EXTXYZ {p}") from e
                    # type
                    if "type" in sl:
                        try:
                            types[i] = int(float(toks[sl["type"].start]))
                        except Exception:
                            types[i] = 0
                    else:
                        types[i] = _type_from_symbol(
                            sym,
                            species_to_type=species_to_type,
                            fallback_map=sym_to_type,
                            next_type_ref=next_type_ref,
                        )
                    # id
                    if "id" in sl:
                        try:
                            ids[i] = int(float(toks[sl["id"].start]))
                        except Exception:
                            ids[i] = i + 1
                    else:
                        ids[i] = i + 1
                else:
                    # fallback
                    sym = toks[0]
                    syms.append(str(sym))
                    pos[i, :] = [float(toks[1]), float(toks[2]), float(toks[3])]
                    types[i] = _type_from_symbol(
                        sym,
                        species_to_type=species_to_type,
                        fallback_map=sym_to_type,
                        next_type_ref=next_type_ref,
                    )
                    ids[i] = i + 1

            # consistency dump conventions
            order = np.argsort(ids)
            ids = ids[order]
            types = types[order]
            pos = pos[order]

            yield DumpFrame(
                timestep=int(step),
                ids=np.asarray(ids, dtype=int),
                types=np.asarray(types, dtype=int),
                positions=np.asarray(pos, dtype=float),
                cell=np.asarray(cell, dtype=float),
                origin=np.zeros((3,), dtype=float),
            )


def read_extxyz_frames(
    path: Path,
    *,
    last_n: Optional[int] = None,
    type_to_species: Optional[Sequence[str]] = None,
) -> list[DumpFrame]:
    """Extxyz frames."""

    if last_n is not None:
        n = int(last_n)
        if n < 1:
            raise ValueError("last_n must be >= 1")
        from collections import deque

        dq: "deque[DumpFrame]" = deque(maxlen=n)
        for fr in iter_extxyz_frames(path, type_to_species=type_to_species):
            dq.append(fr)
        return list(dq)

    return list(iter_extxyz_frames(path, type_to_species=type_to_species))
