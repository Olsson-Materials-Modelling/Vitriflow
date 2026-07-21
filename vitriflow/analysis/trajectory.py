from __future__ import annotations

"""Trajectory loading and stage-frame selection helpers."""

import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .dump import (
    DumpFrame,
    canonicalize_lammps_frame,
    normalize_pbc,
    read_dump_frames,
    read_last_dump_frames,
)
from ..io.extxyz import read_extxyz_frames

_FINAL_RESTART_SUFFIX = "-1.restart"
_FINAL_RESTART_KEYWORDS = ("final", "relaxed", "optimized", "optimised", "converged", "last", "endpoint")


def _has_delimited_token(name: str, token: str) -> bool:
    pattern = rf"(?:^|[^A-Za-z0-9]){re.escape(str(token).lower())}(?=$|[^A-Za-z0-9])"
    return re.search(pattern, str(name).lower()) is not None


def _is_strict_final_restart_name(path: Path | str) -> bool:
    """Return true only for restart filenames that explicitly mean final.

    CP2K final restarts conventionally end in ``*-1.restart``.  Do not treat
    missing-hyphen variants such as ``*1.restart`` as equivalent.
    """

    name = Path(str(path)).name.lower()
    if not name.endswith(".restart"):
        return False
    if name.endswith(_FINAL_RESTART_SUFFIX):
        return True
    stem = name[: -len(".restart")]
    return any(_has_delimited_token(stem, token) for token in _FINAL_RESTART_KEYWORDS)


def _read_strict_cp2k_restart(path: Path, *, type_to_species: Optional[Sequence[str]]) -> list[DumpFrame]:
    if not _is_strict_final_restart_name(path):
        raise ValueError(
            "automatic frame loading from .restart files is disabled unless the filename is an explicit final restart "
            "such as '*-1.restart' or contains a delimited final/relaxed/converged token"
        )
    try:
        from ..io.cp2k_restart import read_cp2k_restart_frame

        return [read_cp2k_restart_frame(Path(path), type_to_species=type_to_species)]
    except Exception as exc:
        raise ValueError(
            f"strict final restart could not be parsed as a CP2K text restart: {Path(path)}"
        ) from exc


def _read_cp2k_restart_frames(
    path: Path,
    *,
    last_n: Optional[int],
    type_to_species: Optional[Sequence[str]],
) -> list[DumpFrame]:
    # Only explicitly final restart names may be loaded automatically.  This
    # preserves the strict Si3N4 rule: ``*-1.restart`` is final, but missing
    # hyphen variants such as ``*1.restart`` and ordinary ``*.restart`` files
    # cannot be silently promoted to final-frame analysis inputs.
    return _read_strict_cp2k_restart(Path(path), type_to_species=type_to_species)


def _read_text_head(path: Path, *, max_lines: int = 80) -> list[str]:
    try:
        return Path(path).read_text(errors="replace").splitlines()[: int(max_lines)]
    except Exception:
        return []


def _looks_like_lammps_dump(path: Path) -> bool:
    p = Path(path)
    if p.suffix.lower() in {".lammpstrj", ".dump", ".trj"}:
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    up = [str(ln).strip().upper() for ln in head]
    return bool(any(ln.startswith("ITEM: TIMESTEP") for ln in up) and any(ln.startswith("ITEM: ATOMS") for ln in up))


def _looks_like_lammps_data(path: Path) -> bool:
    p = Path(path)
    name = p.name.lower()
    if name in {"relax.data", "output.data", "input.data", "structure.data"}:
        return True
    if p.suffix.lower() in {".data", ".lmp", ".dat"}:
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    low = [str(ln).lower() for ln in head]
    atoms_hdr = any(" atoms" in ln for ln in low)
    types_hdr = any(" atom types" in ln for ln in low)
    bounds_hdr = any("xlo xhi" in ln for ln in low)
    atoms_section = any(str(ln).strip().lower().startswith("atoms") for ln in low)
    return bool(atoms_hdr and bounds_hdr and (types_hdr or atoms_section))


def _float_token(token: str) -> float:
    return float(str(token).replace("D", "E").replace("d", "e"))


def _read_lammps_data_dumpframe_minimal(
    path: Path,
    *,
    atom_style: str = "atomic",
    units_style: str = "metal",
) -> DumpFrame:
    """Strictly read atomic/charge restricted-triclinic LAMMPS data.

    Raw LAMMPS data is a dimensional source boundary, so this parser validates
    the header evidence and every atom row before converting the complete
    frame to canonical units.  It intentionally supports only the two atom
    styles accepted by :class:`MDConfig`; guessing another layout would make a
    plausible-looking but physically wrong structure possible.
    """

    p = Path(path)
    lines = p.read_text(errors="replace").splitlines()
    atoms_re = re.compile(r"^\s*(\d+)\s+atoms\s*$", re.IGNORECASE)
    types_re = re.compile(r"^\s*(\d+)\s+atom\s+types\s*$", re.IGNORECASE)
    section_heads = {
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

    def _strip(raw: str) -> str:
        return str(raw).split("#", 1)[0].strip()

    def _section_name(raw: str) -> Optional[str]:
        key = " ".join(_strip(raw).lower().split())
        return key if key in section_heads else None

    def _exact_positive_int(token: str, *, field: str, line_number: int) -> int:
        try:
            value = Decimal(
                str(token).strip().replace("D", "E").replace("d", "e")
            )
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"Invalid {field} at {p}:{line_number}: {token!r}"
            ) from exc
        if not value.is_finite() or value != value.to_integral_value() or value <= 0:
            raise ValueError(
                f"Invalid {field} at {p}:{line_number}: expected a positive "
                f"integer, got {token!r}"
            )
        if value > int(np.iinfo(np.intp).max):
            raise ValueError(
                f"Invalid {field} at {p}:{line_number}: integer {token!r} exceeds platform index range"
            )
        return int(value)

    n_atoms: Optional[int] = None
    n_types: Optional[int] = None
    bounds: dict[str, tuple[float, float]] = {}
    tilt = (0.0, 0.0, 0.0)
    for line_number, raw in enumerate(lines[:400], start=1):
        line = _strip(raw)
        if not line:
            continue
        match = atoms_re.match(line)
        if match:
            if n_atoms is not None:
                raise ValueError(f"Duplicate atom-count header at {p}:{line_number}")
            n_atoms = int(match.group(1))
            continue
        match = types_re.match(line)
        if match:
            if n_types is not None:
                raise ValueError(f"Duplicate atom-type-count header at {p}:{line_number}")
            n_types = int(match.group(1))
            continue
        toks = line.split()
        tail2 = [str(t).lower() for t in toks[-2:]]
        tail3 = [str(t).lower() for t in toks[-3:]]
        try:
            if len(toks) >= 4 and tail2 == ["xlo", "xhi"]:
                if "x" in bounds:
                    raise ValueError(f"Duplicate x bounds at {p}:{line_number}")
                bounds["x"] = (_float_token(toks[0]), _float_token(toks[1]))
            elif len(toks) >= 4 and tail2 == ["ylo", "yhi"]:
                if "y" in bounds:
                    raise ValueError(f"Duplicate y bounds at {p}:{line_number}")
                bounds["y"] = (_float_token(toks[0]), _float_token(toks[1]))
            elif len(toks) >= 4 and tail2 == ["zlo", "zhi"]:
                if "z" in bounds:
                    raise ValueError(f"Duplicate z bounds at {p}:{line_number}")
                bounds["z"] = (_float_token(toks[0]), _float_token(toks[1]))
            elif len(toks) >= 6 and tail3 == ["xy", "xz", "yz"]:
                tilt = (_float_token(toks[0]), _float_token(toks[1]), _float_token(toks[2]))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid LAMMPS box geometry at {p}:{line_number}") from exc

    if n_atoms is None or n_atoms < 1:
        raise ValueError(f"LAMMPS data file must declare a positive '<N> atoms' header: {p}")
    if n_types is None or n_types < 1:
        raise ValueError(f"LAMMPS data file must declare a positive '<N> atom types' header: {p}")
    if not all(k in bounds for k in ("x", "y", "z")):
        raise ValueError(f"LAMMPS data file missing box bounds: {p}")

    atoms_start: Optional[int] = None
    style_in_file: Optional[str] = None
    for idx, raw in enumerate(lines):
        if _strip(raw).lower() == "atoms":
            atoms_start = idx + 1
            if "#" in str(raw):
                comment = str(raw).split("#", 1)[1].strip().split()
                style_in_file = comment[0].lower() if comment else None
            break
    if atoms_start is None:
        raise ValueError(f"LAMMPS data file missing Atoms section: {p}")

    style = str(atom_style or "atomic").strip().lower()
    if style_in_file is not None:
        if style_in_file not in {"atomic", "charge"}:
            raise ValueError(
                f"Unsupported LAMMPS Atoms style {style_in_file!r} in {p}; "
                "only atomic and charge are supported"
            )
        style = style_in_file
    if style not in {"atomic", "charge"}:
        raise ValueError(
            f"Unsupported LAMMPS atom_style {atom_style!r}; only atomic and charge are supported"
        )
    ids: list[int] = []
    types: list[int] = []
    positions: list[list[float]] = []
    for line_number, raw in enumerate(lines[atoms_start:], start=atoms_start + 1):
        line = _strip(raw)
        if not line:
            continue
        if _section_name(raw) is not None:
            break
        toks = line.split()
        required = 6 if style == "charge" else 5
        if len(toks) < required:
            raise ValueError(
                f"Malformed {style} atom row at {p}:{line_number}: expected at "
                f"least {required} fields"
            )
        try:
            if style == "atomic":
                atom_id = _exact_positive_int(toks[0], field="atom id", line_number=line_number)
                atom_type = _exact_positive_int(toks[1], field="atom type", line_number=line_number)
                xyz_idx = 2
            else:
                atom_id = _exact_positive_int(toks[0], field="atom id", line_number=line_number)
                atom_type = _exact_positive_int(toks[1], field="atom type", line_number=line_number)
                charge = _float_token(toks[2])
                if not math.isfinite(charge):
                    raise ValueError(
                        f"Non-finite charge at {p}:{line_number}: {toks[2]!r}"
                    )
                xyz_idx = 3
            xyz = [_float_token(toks[xyz_idx]), _float_token(toks[xyz_idx + 1]), _float_token(toks[xyz_idx + 2])]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid {style} atom row at {p}:{line_number}") from exc
        if not np.all(np.isfinite(np.asarray(xyz, dtype=float))):
            raise ValueError(f"Non-finite atom position at {p}:{line_number}")
        ids.append(atom_id)
        types.append(atom_type)
        positions.append(xyz)

    if len(positions) != n_atoms:
        raise ValueError(
            f"Atom-count mismatch in {p}: header declares {n_atoms}, parsed {len(positions)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Atom IDs in {p} must be unique positive integers")
    if any(atom_type > n_types for atom_type in types):
        raise ValueError(f"Atom types in {p} must lie in [1, {n_types}]")
    order = np.argsort(np.asarray(ids, dtype=int))
    ids_arr = np.asarray(ids, dtype=int)[order]
    types_arr = np.asarray(types, dtype=int)[order]
    xlo, xhi = bounds["x"]; ylo, yhi = bounds["y"]; zlo, zhi = bounds["z"]
    geometry = np.asarray([xlo, xhi, ylo, yhi, zlo, zhi, *tilt], dtype=float)
    if not np.all(np.isfinite(geometry)):
        raise ValueError(f"LAMMPS box geometry in {p} must be finite")
    if not (xhi > xlo and yhi > ylo and zhi > zlo):
        raise ValueError(f"LAMMPS box lengths in {p} must be positive")
    # DumpFrame positions are absolute Cartesian coordinates; origin is stored
    # separately and consumers form fractional coordinates from
    # ``positions-origin``.  Subtracting here as well would shift nonzero-origin
    # data files twice.
    pos_arr = np.asarray(positions, dtype=float)[order]
    xy, xz, yz = tilt
    cell = np.asarray(
        [
            [float(xhi - xlo), 0.0, 0.0],
            [float(xy), float(yhi - ylo), 0.0],
            [float(xz), float(yz), float(zhi - zlo)],
        ],
        dtype=float,
    )
    sv = np.linalg.svd(cell, compute_uv=False)
    if sv.size != 3 or float(sv[-1]) <= np.finfo(float).eps * max(1.0, float(sv[0])) * 16.0:
        raise ValueError(f"LAMMPS data file has zero-volume cell: {p}")
    frame = DumpFrame(
        timestep=0,
        ids=ids_arr,
        types=types_arr,
        positions=pos_arr,
        cell=np.asarray(cell, dtype=float),
        origin=np.asarray([xlo, ylo, zlo], dtype=float),
        # A LAMMPS data file does not encode boundary styles.  Within
        # VitriFlow this reader is the periodic simulation-cell reader; callers
        # needing non-periodic boundaries must use a format that records them
        # (EXTXYZ, LAMMPS dump, ASE-supported structure, or CP2K restart).
        pbc=(True, True, True),
    )
    return canonicalize_lammps_frame(frame, units_style=units_style)


def _atoms_to_dumpframe(atoms, *, type_to_species: Optional[Sequence[str]], timestep: int) -> DumpFrame:
    syms = [str(s) for s in atoms.get_chemical_symbols()]
    if len(syms) < 1:
        raise ValueError("Structure source produced zero atoms")

    if type_to_species is not None:
        mapping = {str(sym): i + 1 for i, sym in enumerate(list(type_to_species))}
    else:
        mapping = {str(sym): i + 1 for i, sym in enumerate(sorted(set(syms)))}
    try:
        types = np.asarray([int(mapping[str(sym)]) for sym in syms], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Structure contains symbol not present in type_to_species: {exc}") from exc

    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape != (3, 3):
        raise ValueError("Structure source has invalid cell shape")
    if abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("Structure source is missing a valid periodic cell")

    get_pbc = getattr(atoms, "get_pbc", None)
    if not callable(get_pbc):
        raise ValueError("Structure reader did not provide periodic-boundary metadata")
    pbc = normalize_pbc(get_pbc())

    ids = np.arange(1, int(len(syms)) + 1, dtype=int)
    return DumpFrame(
        timestep=int(timestep),
        ids=np.asarray(ids, dtype=int),
        types=np.asarray(types, dtype=int),
        positions=np.asarray(pos, dtype=float),
        cell=np.asarray(cell, dtype=float),
        origin=np.zeros((3,), dtype=float),
        pbc=pbc,
    )


def _frames_from_ase_source(
    path: Path,
    *,
    last_n: Optional[int],
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
    units_style: Optional[str] = "metal",
) -> list[DumpFrame]:
    p = Path(path)

    if _looks_like_lammps_data(p):
        source_units = _require_lammps_units_style(units_style, path=p)
        # Validate the raw file ourselves before any permissive third-party
        # reader can normalise malformed IDs, rows, or count mismatches.
        return [
            _read_lammps_data_dumpframe_minimal(
                p,
                atom_style=str(atom_style),
                units_style=source_units,
            )
        ]

    from ase.io import read as ase_read

    images = None
    if last_n is not None:
        try:
            images = ase_read(str(p), index=slice(-int(last_n), None))
        except Exception:
            images = None
    if images is None:
        try:
            images = ase_read(str(p), index=":")
        except Exception:
            images = ase_read(str(p))

    if isinstance(images, (list, tuple)):
        seq = list(images)
    else:
        seq = [images]
    if last_n is not None and len(seq) > int(last_n):
        seq = seq[-int(last_n) :]

    frames: list[DumpFrame] = []
    for i, atoms in enumerate(seq):
        info = getattr(atoms, "info", None)
        step = int(i)
        if isinstance(info, dict):
            for key in ("Step", "step", "timestep", "Timestep"):
                if key not in info:
                    continue
                val = info[key]
                try:
                    if isinstance(val, (bool, np.bool_)):
                        raise ValueError
                    numeric = Decimal(
                        str(val).strip().replace("D", "E").replace("d", "e")
                    )
                except (InvalidOperation, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {key} metadata in trajectory {p}: expected a finite exact integer, got {val!r}"
                    ) from exc
                if (
                    not numeric.is_finite()
                    or numeric != numeric.to_integral_value()
                    or numeric < 0
                    or numeric > int(np.iinfo(np.intp).max)
                ):
                    raise ValueError(
                        f"Invalid {key} metadata in trajectory {p}: expected a finite exact integer, got {val!r}"
                    )
                step = int(numeric)
                break
        frames.append(_atoms_to_dumpframe(atoms, type_to_species=type_to_species, timestep=step))
    return frames


def _require_lammps_units_style(units_style: Optional[str], *, path: Path) -> str:
    """Require explicit dimensional units at a raw LAMMPS source boundary."""

    if units_style is None or not str(units_style).strip():
        raise ValueError(
            "LAMMPS units_style is required to canonicalize raw dump/data source "
            f"{Path(path)}; canonical EXTXYZ/CP2K sources do not require it"
        )
    from ..lammps_units import normalize_lammps_units_style

    return normalize_lammps_units_style(str(units_style))


def read_frames_auto(
    path: Path,
    *,
    last_n: Optional[int] = None,
    type_to_species: Optional[Sequence[str]] = None,
    atom_style: str = "atomic",
    units_style: Optional[str] = "metal",
) -> list[DumpFrame]:
    """Frames auto."""

    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".restart":
        # Never pass restart files to ASE's generic format probe.  Allow only
        # recognised LAMMPS dump/data exports or explicitly final CP2K text
        # restarts, each through a deterministic reader.
        if _looks_like_lammps_dump(p):
            source_units = _require_lammps_units_style(units_style, path=p)
            return read_dump_frames(p, last_n=last_n, units_style=source_units)
        if _looks_like_lammps_data(p):
            return _frames_from_ase_source(
                p,
                last_n=last_n,
                type_to_species=type_to_species,
                atom_style=atom_style,
                units_style=units_style,
            )
        return _read_cp2k_restart_frames(p, last_n=last_n, type_to_species=type_to_species)
    if suf == ".extxyz":
        return read_extxyz_frames(p, last_n=last_n, type_to_species=type_to_species)
    if suf == ".xyz":
        try:
            return read_extxyz_frames(p, last_n=last_n, type_to_species=type_to_species)
        except Exception:
            return _frames_from_ase_source(
                p,
                last_n=last_n,
                type_to_species=type_to_species,
                atom_style=atom_style,
                units_style=units_style,
            )
    if _looks_like_lammps_dump(p):
        source_units = _require_lammps_units_style(units_style, path=p)
        return read_dump_frames(p, last_n=last_n, units_style=source_units)
    return _frames_from_ase_source(
        p,
        last_n=last_n,
        type_to_species=type_to_species,
        atom_style=atom_style,
        units_style=units_style,
    )


def read_last_frames_auto(
    path: Path,
    n: int,
    *,
    type_to_species: Optional[Sequence[str]] = None,
    atom_style: str = "atomic",
    units_style: Optional[str] = "metal",
) -> list[DumpFrame]:
    """Last frames auto."""

    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".restart":
        if _looks_like_lammps_dump(p):
            source_units = _require_lammps_units_style(units_style, path=p)
            return read_last_dump_frames(p, int(n), units_style=source_units)
        if _looks_like_lammps_data(p):
            return _frames_from_ase_source(
                p,
                last_n=int(n),
                type_to_species=type_to_species,
                atom_style=atom_style,
                units_style=units_style,
            )
        return _read_cp2k_restart_frames(p, last_n=int(n), type_to_species=type_to_species)
    if suf == ".extxyz":
        return read_extxyz_frames(p, last_n=int(n), type_to_species=type_to_species)
    if suf == ".xyz":
        try:
            return read_extxyz_frames(p, last_n=int(n), type_to_species=type_to_species)
        except Exception:
            return _frames_from_ase_source(
                p,
                last_n=int(n),
                type_to_species=type_to_species,
                atom_style=atom_style,
                units_style=units_style,
            )
    if _looks_like_lammps_dump(p):
        source_units = _require_lammps_units_style(units_style, path=p)
        return read_last_dump_frames(p, int(n), units_style=source_units)
    return _frames_from_ase_source(
        p,
        last_n=int(n),
        type_to_species=type_to_species,
        atom_style=atom_style,
        units_style=units_style,
    )


def stage_trajectory_path(stage_dir: Path) -> Optional[Path]:
    """Stage trajectory path."""
    d = Path(stage_dir)
    cand = d / "traj.extxyz"
    if cand.exists():
        return cand
    for nm in d.glob("*.lammpstrj"):
        return nm
    return None


def evenly_sample_indices(n: int, k: Optional[int]) -> list[int]:
    """Evenly sample indices."""

    n = int(n)
    if n <= 0:
        return []
    if k is None:
        return list(range(n))
    k = int(k)
    if k <= 0 or k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    idx = np.linspace(0, n - 1, k, dtype=int)
    out: list[int] = []
    seen: set[int] = set()
    for i in idx.tolist():
        ii = int(i)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    if out[0] != 0:
        out.insert(0, 0)
    if out[-1] != n - 1:
        out.append(n - 1)
    return sorted(set(out), key=out.index)


def quench_window_steps(
    *,
    T_start: float,
    T_stop: float,
    total_steps: int,
    T_upper: Optional[float],
    T_lower: Optional[float],
) -> Optional[Tuple[float, float]]:
    """Quench window steps."""

    if int(total_steps) <= 0:
        return None
    if T_upper is None or T_lower is None:
        return None
    Tu = float(T_upper)
    Tl = float(T_lower)
    if not (np.isfinite(Tu) and np.isfinite(Tl)):
        return None
    if float(T_start) == float(T_stop):
        return None
    loT = min(float(T_start), float(T_stop))
    hiT = max(float(T_start), float(T_stop))
    Tu = min(max(Tu, loT), hiT)
    Tl = min(max(Tl, loT), hiT)
    if Tu < Tl:
        Tu, Tl = Tl, Tu

    def _step_for_T(T: float) -> float:
        frac = (float(T) - float(T_start)) / (float(T_stop) - float(T_start))
        return float(frac * float(total_steps))

    s1 = _step_for_T(Tu)
    s2 = _step_for_T(Tl)
    a = max(0.0, min(float(total_steps), min(s1, s2)))
    b = max(0.0, min(float(total_steps), max(s1, s2)))
    if b <= a:
        return None
    return (float(a), float(b))


def _select_dense_window_indices(
    steps: np.ndarray,
    *,
    window: Tuple[float, float],
    quench_tail_min_frames: int,
    max_frames: Optional[int],
) -> list[int]:
    values = np.asarray(steps, dtype=float).reshape(-1)
    n_frames = int(values.size)
    if n_frames == 0:
        return []
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory steps must be finite")
    lo, hi = sorted((float(window[0]), float(window[1])))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("dense-window bounds must be finite")

    cap = n_frames
    if max_frames is not None and int(max_frames) > 0:
        cap = min(n_frames, int(max_frames))
    if cap == 1:
        return [n_frames - 1]

    # Endpoints are provenance anchors.  With only one slot the final state is
    # the more important physical endpoint; with >=2 retain both.
    chosen: list[int] = [0, n_frames - 1]
    chosen_set = set(chosen)
    dense_idx = [
        int(i) for i, value in enumerate(values.tolist()) if lo <= float(value) <= hi
    ]

    def _even_candidates(candidates: Sequence[int], count: int) -> list[int]:
        pool = [int(x) for x in candidates]
        wanted = min(max(int(count), 0), len(pool))
        if wanted <= 0:
            return []
        if wanted == len(pool):
            return pool
        if wanted == 1:
            return [pool[(len(pool) - 1) // 2]]
        positions = np.linspace(0, len(pool) - 1, wanted)
        selected: list[int] = []
        for raw in positions.tolist():
            candidate = pool[int(round(float(raw)))]
            if candidate not in selected:
                selected.append(candidate)
        # Rounding collisions are unlikely but cannot reduce the requested
        # count; fill deterministically from the remaining pool.
        for candidate in pool:
            if len(selected) >= wanted:
                break
            if candidate not in selected:
                selected.append(candidate)
        return selected

    remaining = cap - len(chosen)
    dense_candidates = [index for index in dense_idx if index not in chosen_set]
    for index in _even_candidates(dense_candidates, remaining):
        chosen.append(index)
        chosen_set.add(index)
    remaining = cap - len(chosen)
    if remaining > 0:
        uniform_candidates = [index for index in range(n_frames) if index not in chosen_set]
        for index in _even_candidates(uniform_candidates, remaining):
            chosen.append(index)
            chosen_set.add(index)
    return sorted(chosen)


def select_stage_frames(
    frames_all: Sequence[DumpFrame],
    *,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    stage_role: Optional[str] = None,
    quench_window_steps_range: Optional[Tuple[float, float]] = None,
    temperatures: Optional[Sequence[float]] = None,
    tm_temperature: Optional[float] = None,
    diffusion_freeze_temperature: Optional[float] = None,
    quench_tail_fraction: float = 0.67,
    quench_tail_min_frames: int = 8,
    quench_tail_fallback_fraction: float = 0.40,
) -> tuple[list[DumpFrame], dict[str, Any]]:
    """Stage frames."""

    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be >= 1")

    base = list(frames_all)[:: int(frame_stride)]
    meta: dict[str, Any] = {
        "selection": "uniform",
        "frame_stride": int(frame_stride),
        "max_frames": int(max_frames) if max_frames is not None else None,
    }
    if not base:
        return [], meta

    steps = np.asarray([float(fr.timestep) for fr in base], dtype=float)
    role = str(stage_role or "").strip().lower()

    dense_window: Optional[Tuple[float, float]] = None
    selection = "uniform"

    if role == "quench":
        if temperatures is not None and tm_temperature is not None and diffusion_freeze_temperature is not None:
            try:
                temps = np.asarray([float(x) for x in temperatures], dtype=float)
            except Exception:
                temps = np.full((len(base),), np.nan, dtype=float)
            if temps.shape[0] == len(base) and np.isfinite(temps).any():
                t_hi = max(float(tm_temperature), float(diffusion_freeze_temperature))
                t_lo = min(float(tm_temperature), float(diffusion_freeze_temperature))
                idx = [int(i) for i, T in enumerate(temps.tolist()) if t_lo <= float(T) <= t_hi]
                if idx:
                    dense_window = (float(steps[min(idx)]), float(steps[max(idx)]))
                    selection = "quench_tm_freeze_dense"
                    meta["tm_temperature"] = float(tm_temperature)
                    meta["diffusion_freeze_temperature"] = float(diffusion_freeze_temperature)
                    meta["dense_window_temperature"] = [float(t_lo), float(t_hi)]
        if dense_window is None and quench_window_steps_range is not None:
            dense_window = (float(quench_window_steps_range[0]), float(quench_window_steps_range[1]))
            selection = "quench_step_window_dense"
        if dense_window is None:
            frac = min(max(float(quench_tail_fallback_fraction), 0.05), 0.95)
            start_idx = int(max(0, math.floor((1.0 - frac) * len(base))))
            dense_window = (float(steps[start_idx]), float(steps[-1]))
            selection = "quench_tail_dense"
            meta["dense_window_fraction"] = float(frac)

        chosen_idx = _select_dense_window_indices(
            steps,
            window=dense_window,
            quench_tail_min_frames=max(int(quench_tail_min_frames), 1),
            max_frames=max_frames,
        )
        meta["selection"] = selection
        meta["dense_window_steps"] = [float(dense_window[0]), float(dense_window[1])]
        meta["quench_tail_fraction"] = float(quench_tail_fraction)
        meta["quench_tail_min_frames"] = int(quench_tail_min_frames)
        meta["quench_tail_fallback_fraction"] = float(quench_tail_fallback_fraction)
        dense_available = int(
            np.count_nonzero(
                (steps >= float(min(dense_window))) & (steps <= float(max(dense_window)))
            )
        )
        dense_selected = int(
            sum(
                float(min(dense_window)) <= float(steps[index]) <= float(max(dense_window))
                for index in chosen_idx
            )
        )
        requested_dense = max(int(quench_tail_min_frames), 1)
        cap = len(base)
        if max_frames is not None and int(max_frames) > 0:
            cap = min(cap, int(max_frames))
        reserved = [len(base) - 1] if cap == 1 else [0, len(base) - 1]
        reserved_dense = sum(
            float(min(dense_window)) <= float(steps[index]) <= float(max(dense_window))
            for index in set(reserved)
        )
        max_dense_under_cap = min(
            dense_available,
            int(reserved_dense) + max(0, cap - len(set(reserved))),
        )
        meta["dense_window_frames_available"] = dense_available
        meta["dense_window_frames_selected"] = dense_selected
        meta["dense_window_min_frames_requested"] = requested_dense
        meta["dense_window_minimum_satisfied"] = bool(dense_selected >= requested_dense)
        meta["dense_window_minimum_limited_by_cap"] = bool(
            dense_available >= requested_dense and max_dense_under_cap < requested_dense
        )
        meta["dense_window_minimum_limited_by_availability"] = bool(
            dense_available < requested_dense
        )
    else:
        chosen_idx = evenly_sample_indices(len(base), max_frames)

    selected = [base[int(i)] for i in chosen_idx]
    meta["n_selected"] = int(len(selected))
    meta["max_frames_hard_cap"] = True
    meta["selected_steps"] = [int(base[int(i)].timestep) for i in chosen_idx]
    return selected, meta
