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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from ..analysis.dump import DumpFrame, frame_pbc, normalize_pbc


_KV_RE = re.compile(r"(\w+)=((?:\"[^\"]*\")|(?:\S+))")


def _parse_exact_integer(
    token: object,
    *,
    field: str,
    path: Path,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    """Parse an integer without float rounding or fractional truncation."""

    if isinstance(token, (bool, np.bool_)):
        raise ValueError(f"EXTXYZ {field} must be an exact integer in {path}; got {token!r}")
    try:
        value = Decimal(
            str(token).strip().replace("D", "E").replace("d", "e")
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"EXTXYZ {field} must be an exact integer in {path}; got {token!r}"
        ) from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(
            f"EXTXYZ {field} must be an exact integer in {path}; got {token!r}"
        )
    if positive and value <= 0:
        raise ValueError(
            f"EXTXYZ {field} must be a positive exact integer in {path}; got {token!r}"
        )
    if nonnegative and value < 0:
        raise ValueError(
            f"EXTXYZ {field} must be a nonnegative exact integer in {path}; got {token!r}"
        )
    intp = np.iinfo(np.intp)
    if value < int(intp.min) or value > int(intp.max):
        raise ValueError(
            f"EXTXYZ {field} integer {token!r} is outside platform index range in {path}"
        )
    return int(value)


def _parse_kv(comment: str) -> Dict[str, str]:
    """Kv."""
    out: Dict[str, str] = {}
    for m in _KV_RE.finditer(str(comment).strip()):
        k = m.group(1)
        v = m.group(2)
        if str(k) in out:
            raise ValueError(f"Duplicate EXTXYZ comment key {str(k)!r}")
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
            n = _parse_exact_integer(
                toks[i + 2],
                field="Properties count",
                path=Path("<Properties>"),
                positive=True,
            )
        except ValueError as e:
            raise ValueError(f"Invalid EXTXYZ Properties count in: {spec!r}") from e
        if nm == "" or tp == "" or n < 1:
            raise ValueError(f"Invalid EXTXYZ Properties field in: {spec!r}")
        if nm in names:
            raise ValueError(f"Duplicate EXTXYZ Properties field {nm!r} in: {spec!r}")
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
    scale = float(np.max(np.abs(cell)))
    det = float(np.linalg.det(cell))
    det_tol = 128.0 * np.finfo(float).eps * max(
        scale**3,
        np.finfo(float).tiny,
    )
    if not math.isfinite(det) or abs(det) <= det_tol:
        raise ValueError("EXTXYZ Lattice must be nonsingular")
    return cell


def _format_lattice(cell: np.ndarray) -> str:
    c = np.asarray(cell, dtype=float)
    if c.shape != (3, 3):
        raise ValueError("cell must be 3x3")
    if not np.all(np.isfinite(c)):
        raise ValueError("cell must be finite")
    flat = c.reshape(-1)
    return " ".join(f"{float(x):.16g}" for x in flat.tolist())


def _parse_pbc(value: str, *, path: Path) -> tuple[bool, bool, bool]:
    tokens = str(value).replace(",", " ").split()
    if len(tokens) != 3:
        raise ValueError(f"EXTXYZ pbc must contain exactly three flags in {path}")
    truth = {"t": True, "true": True, "1": True, "f": False, "false": False, "0": False}
    try:
        flags = tuple(truth[str(tok).strip().lower()] for tok in tokens)
    except KeyError as exc:
        raise ValueError(f"Invalid EXTXYZ pbc flag {exc.args[0]!r} in {path}") from exc
    return normalize_pbc(flags)


def _wrap_positions(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    origin: np.ndarray,
    pbc: Tuple[bool, bool, bool],
) -> np.ndarray:
    """Wrap positions."""
    pos = np.asarray(positions, dtype=float)
    H = np.asarray(cell, dtype=float)
    org = np.asarray(origin, dtype=float)
    invH = np.linalg.inv(H)
    frac = (pos - org) @ invH
    periodic = np.asarray(normalize_pbc(pbc), dtype=bool)
    frac[:, periodic] = frac[:, periodic] - np.floor(frac[:, periodic])
    return frac @ H


def _effective_pbc(frame: DumpFrame, override: Optional[Tuple[bool, bool, bool]]) -> tuple[bool, bool, bool]:
    return frame_pbc(frame) if override is None else normalize_pbc(override)


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


@dataclass(frozen=True)
class _ValidatedWriterFrame:
    timestep: int
    ids: np.ndarray
    types: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    origin: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])


def _validated_species_labels(
    labels: Sequence[object],
    *,
    field: str,
) -> list[str]:
    if isinstance(labels, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence of non-empty EXTXYZ tokens")
    out: list[str] = []
    for index, raw in enumerate(list(labels), start=1):
        if raw is None:
            raise ValueError(f"{field} label {index} must be a non-empty EXTXYZ token")
        label = str(raw).strip()
        if (
            not label
            or any(ch.isspace() for ch in label)
            or '"' in label
            or "'" in label
        ):
            raise ValueError(f"{field} label {index} must be a non-empty EXTXYZ token")
        out.append(label)
    return out


def _validated_writer_frame(
    frame: DumpFrame,
    *,
    path: Path,
    type_to_species: Optional[Sequence[str]],
) -> _ValidatedWriterFrame:
    """Validate all numeric evidence before serialising an EXTXYZ frame."""

    try:
        positions = np.asarray(frame.positions, dtype=float)
        cell = np.asarray(frame.cell, dtype=float)
        origin = np.asarray(frame.origin, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("EXTXYZ frame geometry must be numeric") from exc
    if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] < 1:
        raise ValueError("EXTXYZ frame positions must have shape (n_atoms, 3) with n_atoms >= 1")
    n_atoms = int(positions.shape[0])
    if cell.shape != (3, 3):
        raise ValueError("EXTXYZ frame cell must be 3x3")
    if origin.shape != (3,):
        raise ValueError("EXTXYZ frame origin must have shape (3,)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("EXTXYZ frame positions must be finite")
    if not np.all(np.isfinite(cell)):
        raise ValueError("EXTXYZ frame cell must be finite")
    if not np.all(np.isfinite(origin)):
        raise ValueError("EXTXYZ frame origin must be finite")
    scale = float(np.max(np.abs(cell)))
    determinant = float(np.linalg.det(cell))
    determinant_tol = 128.0 * np.finfo(float).eps * max(
        scale**3,
        np.finfo(float).tiny,
    )
    if not math.isfinite(determinant) or abs(determinant) <= determinant_tol:
        raise ValueError("EXTXYZ frame cell must be nonsingular")

    ids_raw = np.asarray(frame.ids, dtype=object)
    types_raw = np.asarray(frame.types, dtype=object)
    if ids_raw.ndim != 1 or ids_raw.size != n_atoms:
        raise ValueError("EXTXYZ frame ids must be a one-dimensional array matching positions")
    if types_raw.ndim != 1 or types_raw.size != n_atoms:
        raise ValueError("EXTXYZ frame types must be a one-dimensional array matching positions")
    ids = np.asarray(
        [
            _parse_exact_integer(
                value,
                field=f"id at atom {index}",
                path=Path(path),
                positive=True,
            )
            for index, value in enumerate(ids_raw.tolist(), start=1)
        ],
        dtype=np.intp,
    )
    types = np.asarray(
        [
            _parse_exact_integer(
                value,
                field=f"type at atom {index}",
                path=Path(path),
                positive=True,
            )
            for index, value in enumerate(types_raw.tolist(), start=1)
        ],
        dtype=np.intp,
    )
    if np.unique(ids).size != ids.size:
        raise ValueError("EXTXYZ frame ids must be unique positive integers")

    timestep = _parse_exact_integer(
        frame.timestep,
        field="Step",
        path=Path(path),
        nonnegative=True,
    )

    if type_to_species is not None:
        labels = _validated_species_labels(type_to_species, field="type_to_species")
        if len(set(label.lower() for label in labels)) != len(labels):
            raise ValueError("type_to_species labels must be unique ignoring case")
        if int(np.max(types)) > len(labels):
            raise ValueError("EXTXYZ frame type exceeds supplied type_to_species mapping")

    return _ValidatedWriterFrame(
        timestep=int(timestep),
        ids=ids,
        types=types,
        positions=positions,
        cell=cell,
        origin=origin,
    )


def write_extxyz_frames(
    path: Path,
    frames: Sequence[DumpFrame],
    *,
    type_to_species: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    wrap: bool = True,
) -> None:
    """Extxyz frames."""

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("write_extxyz_frames requires at least one frame")

    resolved_species = _resolve_type_to_species(
        type_to_species=type_to_species,
        species=species,
        func_name="write_extxyz_frames",
    )
    if resolved_species is not None:
        resolved_species = _validated_species_labels(
            resolved_species,
            field="type_to_species",
        )
    validated_frames = [
        _validated_writer_frame(
            frame,
            path=Path(path),
            type_to_species=resolved_species,
        )
        for frame in frame_list
    ]

    # map integer symbol
    def _sym(t: int) -> str:
        if resolved_species is None:
            return "X"
        i = int(t) - 1
        if i < 0 or i >= len(resolved_species):  # pragma: no cover - validated above
            raise ValueError("EXTXYZ frame type exceeds supplied type_to_species mapping")
        return str(resolved_species[i])

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    with Path(path).open("w") as f:
        for fr, validated in zip(frame_list, validated_frames):
            frame_flags = _effective_pbc(fr, pbc)
            pbc_str = " ".join(["T" if x else "F" for x in frame_flags])
            n = validated.n_atoms
            cell = validated.cell
            if wrap:
                posw = _wrap_positions(
                    positions=validated.positions,
                    cell=cell,
                    origin=validated.origin,
                    pbc=frame_flags,
                )
            else:
                posw = validated.positions - validated.origin

            f.write(f"{n}\n")
            f.write(
                f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={validated.timestep}\n"
            )
            for sym, r, t, i in zip(
                [_sym(x) for x in validated.types.tolist()],
                posw,
                validated.types,
                validated.ids,
            ):
                f.write(
                    f"{sym} {float(r[0]):.12f} {float(r[1]):.12f} {float(r[2]):.12f} {int(t)} {int(i)}\n"
                )


def write_extxyz_iter(
    path: Path,
    frames: Iterable[DumpFrame],
    *,
    type_to_species: Optional[Sequence[str]] = None,
    species: Optional[Sequence[str]] = None,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    wrap: bool = True,
) -> DumpFrame:
    """Extxyz iter."""

    resolved_species = _resolve_type_to_species(
        type_to_species=type_to_species,
        species=species,
        func_name="write_extxyz_iter",
    )
    if resolved_species is not None:
        resolved_species = _validated_species_labels(
            resolved_species,
            field="type_to_species",
        )

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    def _sym(t: int) -> str:
        if resolved_species is None:
            return "X"
        i = int(t) - 1
        if i < 0 or i >= len(resolved_species):  # pragma: no cover - validated below
            raise ValueError("EXTXYZ frame type exceeds supplied type_to_species mapping")
        return str(resolved_species[i])

    last: Optional[DumpFrame] = None
    with Path(path).open("w") as f:
        for fr in frames:
            last = fr
            validated = _validated_writer_frame(
                fr,
                path=Path(path),
                type_to_species=resolved_species,
            )
            frame_flags = _effective_pbc(fr, pbc)
            pbc_str = " ".join(["T" if x else "F" for x in frame_flags])
            n = validated.n_atoms
            cell = validated.cell
            if wrap:
                posw = _wrap_positions(
                    positions=validated.positions,
                    cell=cell,
                    origin=validated.origin,
                    pbc=frame_flags,
                )
            else:
                posw = validated.positions - validated.origin

            f.write(f"{n}\n")
            f.write(
                f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={validated.timestep}\n"
            )
            for sym, r, t, i in zip(
                [_sym(x) for x in validated.types.tolist()],
                posw,
                validated.types,
                validated.ids,
            ):
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
    pbc: Optional[Tuple[bool, bool, bool]] = None,
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
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    wrap: bool = True,
) -> None:
    """Extxyz single with."""

    validated = _validated_writer_frame(
        frame,
        path=Path(path),
        type_to_species=None,
    )
    n = validated.n_atoms
    species_labels = _validated_species_labels(species, field="species override")
    if len(species_labels) != n:
        raise ValueError(f"species override length {len(species_labels)} != n_atoms {n}")

    cell = validated.cell

    frame_flags = _effective_pbc(frame, pbc)
    if wrap:
        posw = _wrap_positions(
            positions=validated.positions,
            cell=cell,
            origin=validated.origin,
            pbc=frame_flags,
        )
    else:
        posw = validated.positions - validated.origin

    props = "species:S:1:pos:R:3:type:I:1:id:I:1"
    pbc_str = " ".join(["T" if x else "F" for x in frame_flags])

    with Path(path).open("w") as f:
        f.write(f"{n}\n")
        f.write(
            f"Lattice=\"{_format_lattice(cell)}\" Properties={props} pbc=\"{pbc_str}\" Step={validated.timestep}\n"
        )
        for sym, r, t, i in zip(
            species_labels,
            posw,
            validated.types,
            validated.ids,
        ):
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
            raise ValueError("type_to_species entries must be non-empty")
        if s in out or s.lower() in out:
            raise ValueError(f"type_to_species contains duplicate species {s!r}")
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
    # Preserve a stable symbol/type mapping across every frame when the file
    # omits explicit types.  Re-starting first-seen numbering per frame can
    # silently swap chemical identities when atom order changes.
    sym_to_type: Dict[str, int] = {}
    next_type_ref = [1]
    explicit_symbol_to_type: Dict[str, int] = {}
    explicit_type_to_symbol: Dict[int, str] = {}
    with p.open("r", errors="replace") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            line = line.strip()
            if line == "":
                continue
            count_tokens = line.split()
            if len(count_tokens) != 1:
                raise ValueError(f"Invalid EXTXYZ atom count line in {p}: {line!r}")
            try:
                nat = _parse_exact_integer(
                    count_tokens[0],
                    field="atom count",
                    path=p,
                    positive=True,
                )
            except ValueError as e:
                raise ValueError(f"Invalid EXTXYZ atom count line in {p}: {line!r}") from e
            comment = f.readline()
            if comment == "":
                raise ValueError(f"Truncated EXTXYZ file {p}: missing comment line")
            kv = _parse_kv(comment)
            if "Lattice" not in kv:
                raise ValueError(f"EXTXYZ frame missing Lattice in {p}")
            if "pbc" not in kv:
                raise ValueError(
                    f"EXTXYZ frame missing pbc in {p}; periodicity must be explicit for provenance-safe analysis"
                )
            cell = _parse_lattice(kv["Lattice"])
            pbc = _parse_pbc(kv["pbc"], path=p)
            step = _parse_exact_integer(
                kv.get("Step", "0"),
                field="Step",
                path=p,
                nonnegative=True,
            )

            props_spec = kv.get("Properties", "")
            props = _parse_properties(props_spec) if props_spec else ExtXYZProperties([], [], [])
            sl = props.slices()

            if props.ncols > 0:
                schema = {
                    str(name): (str(kind).upper(), int(count))
                    for name, kind, count in zip(props.names, props.types, props.counts)
                }
                if schema.get("pos") != ("R", 3):
                    raise ValueError(
                        f"EXTXYZ Properties must declare pos:R:3 in {p}; got {schema.get('pos')!r}"
                    )
                for name, expected in (
                    ("species", ("S", 1)),
                    ("type", ("I", 1)),
                    ("id", ("I", 1)),
                ):
                    if name in schema and schema[name] != expected:
                        raise ValueError(
                            f"EXTXYZ Properties field {name!r} must be {expected[0]}:{expected[1]} in {p}"
                        )
                if "species" not in schema and "type" not in schema:
                    raise ValueError(
                        f"EXTXYZ Properties in {p} must declare species or an explicit type"
                    )

            # determine expected token
            expected_cols = props.ncols
            have_props = expected_cols > 0
            if not have_props:
                expected_cols = 4  # xyz sym x

            syms: list[str] = []
            pos = np.empty((nat, 3), dtype=float)
            types = np.empty((nat,), dtype=int)
            ids = np.empty((nat,), dtype=int)

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
                        sym = "X"
                    syms.append(str(sym))
                    # pos
                    if "pos" not in sl:
                        raise ValueError(f"EXTXYZ Properties missing 'pos' in {p}")
                    ps = sl["pos"]
                    try:
                        pos[i, :] = [float(toks[ps.start + j]) for j in range(3)]
                    except Exception as e:
                        raise ValueError(f"Invalid pos columns in EXTXYZ {p}") from e
                    if not np.all(np.isfinite(pos[i, :])):
                        raise ValueError(f"Non-finite pos columns at atom {i + 1} in EXTXYZ {p}")
                    # type
                    if "type" in sl:
                        try:
                            types[i] = _parse_exact_integer(
                                toks[sl["type"].start],
                                field=f"type at atom {i + 1}",
                                path=p,
                                positive=True,
                            )
                        except ValueError as _e:
                            raise ValueError(
                                f"EXTXYZ type column is not a valid integer at atom {i} in {p}"
                            ) from _e
                        if type_to_species is not None and int(types[i]) > len(type_to_species):
                            raise ValueError(
                                f"EXTXYZ type {int(types[i])} at atom {i + 1} exceeds configured type_to_species"
                            )
                        if "species" in sl:
                            if species_to_type is not None:
                                # ``X`` is the writer's deliberate placeholder
                                # when only an explicit integer type is known.
                                # It carries no contradictory chemistry unless
                                # X itself is part of the configured mapping.
                                placeholder = (
                                    str(sym).strip().lower() == "x"
                                    and "x" not in species_to_type
                                )
                                if not placeholder:
                                    expected_type = _type_from_symbol(
                                        sym,
                                        species_to_type=species_to_type,
                                        fallback_map=sym_to_type,
                                        next_type_ref=next_type_ref,
                                    )
                                    if int(types[i]) != int(expected_type):
                                        raise ValueError(
                                            f"EXTXYZ species/type mismatch at atom {i + 1} in {p}: "
                                            f"species {sym!r} maps to type {expected_type}, got {int(types[i])}"
                                        )
                            elif str(sym).strip().lower() != "x":
                                previous_type = explicit_symbol_to_type.get(str(sym))
                                previous_symbol = explicit_type_to_symbol.get(int(types[i]))
                                if previous_type is not None and previous_type != int(types[i]):
                                    raise ValueError(
                                        f"EXTXYZ species {sym!r} changes type across frames in {p}"
                                    )
                                if previous_symbol is not None and previous_symbol != str(sym):
                                    raise ValueError(
                                        f"EXTXYZ type {int(types[i])} changes species across frames in {p}"
                                    )
                                explicit_symbol_to_type[str(sym)] = int(types[i])
                                explicit_type_to_symbol[int(types[i])] = str(sym)
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
                            ids[i] = _parse_exact_integer(
                                toks[sl["id"].start],
                                field=f"id at atom {i + 1}",
                                path=p,
                                positive=True,
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"EXTXYZ id column is not a valid positive integer at atom {i + 1} in {p}"
                            ) from exc
                    else:
                        ids[i] = i + 1
                else:
                    # fallback
                    sym = toks[0]
                    syms.append(str(sym))
                    pos[i, :] = [float(toks[1]), float(toks[2]), float(toks[3])]
                    if not np.all(np.isfinite(pos[i, :])):
                        raise ValueError(f"Non-finite position at atom {i + 1} in EXTXYZ {p}")
                    types[i] = _type_from_symbol(
                        sym,
                        species_to_type=species_to_type,
                        fallback_map=sym_to_type,
                        next_type_ref=next_type_ref,
                    )
                    ids[i] = i + 1

            if len(set(int(value) for value in ids.tolist())) != nat:
                raise ValueError(f"EXTXYZ atom IDs must be unique positive integers in {p}")

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
                pbc=pbc,
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
