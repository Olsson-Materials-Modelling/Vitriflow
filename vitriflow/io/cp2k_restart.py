from __future__ import annotations

"""Small, dependency-light CP2K restart structure reader.

CP2K ``*.restart`` files are text input/restart files, not generic
trajectories.  VitriFlow only uses this reader for explicitly final restart
snapshots such as ``*-1.restart``.  The parser extracts the final structure from
``&FORCE_EVAL / &SUBSYS / &CELL`` and ``&COORD`` sections and returns a single
:class:`~vitriflow.analysis.dump.DumpFrame`.
"""

import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..analysis.dump import DumpFrame, normalize_pbc

_BOHR_TO_ANGSTROM = 0.529177210903
_UNIT_SCALES = {
    "A": 1.0,
    "ANG": 1.0,
    "ANGSTROM": 1.0,
    "ANGSTROMS": 1.0,
    "BOHR": _BOHR_TO_ANGSTROM,
    "B": _BOHR_TO_ANGSTROM,
    "AU": _BOHR_TO_ANGSTROM,
    "A.U.": _BOHR_TO_ANGSTROM,
    "NM": 10.0,
    "NANOMETER": 10.0,
    "NANOMETERS": 10.0,
}

# Complete periodic-table symbols.  Keeping this local avoids making the CP2K
# restart reader depend on ASE just to canonicalise kind labels.
_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    # CP2K restart/input comments are normally introduced by ! or #.  Restart
    # structure lines do not need quoted comments for our purposes, so a simple
    # split is safer and predictable.
    out = str(line).rstrip("\n")
    for marker in ("!", "#"):
        pos = out.find(marker)
        if pos >= 0:
            out = out[:pos]
    return out.strip()


def _unit_token_to_scale(token: str) -> Optional[float]:
    t = str(token).strip().strip("[](){} ,;\t").upper()
    if t.startswith("UNIT="):
        t = t.split("=", 1)[1].strip().strip("[](){} ,;\t").upper()
    return _UNIT_SCALES.get(t)


def _line_unit_scale(tokens: Sequence[str], default: float) -> float:
    for tok in tokens:
        scale = _unit_token_to_scale(str(tok))
        if scale is not None:
            return float(scale)
    return float(default)


def _is_number_token(token: str) -> bool:
    t = str(token).strip().strip(",;[](){}")
    return _FLOAT_RE.match(t) is not None


def _float_from_token(token: str) -> float:
    return float(str(token).strip().strip(",;[](){}").replace("D", "E").replace("d", "e"))


def _numbers_from_tokens(tokens: Sequence[str]) -> list[float]:
    nums: list[float] = []
    for tok in tokens:
        if _is_number_token(str(tok)):
            nums.append(_float_from_token(str(tok)))
    return nums


def _canonical_symbol_from_kind(kind: str, type_to_species: Optional[Sequence[str]]) -> str:
    raw = str(kind).strip()
    if not raw:
        raise ValueError("CP2K coordinate line has an empty atom kind")

    if type_to_species is not None:
        species = [str(x) for x in type_to_species]
        raw_lower = raw.lower()
        for idx, sym in enumerate(species, start=1):
            if raw_lower == str(idx):
                return str(sym)
        for sym in species:
            s = str(sym)
            sl = s.lower()
            if raw_lower == sl:
                return s
            # Common CP2K KIND variants: Si_1, Si-core, N_shell, etc.  Require
            # a non-alphanumeric delimiter after the chemical symbol so labels
            # such as Sn are not confused with S.
            if raw_lower.startswith(sl) and len(raw_lower) > len(sl):
                nxt = raw_lower[len(sl)]
                if not nxt.isalnum():
                    return s

    cleaned = re.sub(r"^[^A-Za-z]+", "", raw)
    for length in (2, 1):
        cand = cleaned[:length]
        if not cand:
            continue
        sym = cand[0].upper() + cand[1:].lower()
        if sym in _ELEMENT_SYMBOLS:
            return sym
    raise ValueError(f"Could not infer chemical symbol from CP2K atom kind {kind!r}")


def _cell_from_lengths_angles(lengths: Sequence[float], angles_deg: Sequence[float]) -> np.ndarray:
    if len(lengths) != 3:
        raise ValueError("CP2K CELL ABC requires three lengths")
    if len(angles_deg) != 3:
        angles_deg = [90.0, 90.0, 90.0]
    a, b, c = [float(x) for x in lengths]
    alpha, beta, gamma = [math.radians(float(x)) for x in angles_deg]
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Invalid CP2K cell: gamma angle is singular")
    ax = np.array([a, 0.0, 0.0], dtype=float)
    bx = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)
    cx_x = c * math.cos(beta)
    cx_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = max(0.0, c * c - cx_x * cx_x - cx_y * cx_y)
    cx = np.array([cx_x, cx_y, math.sqrt(cz_sq)], dtype=float)
    return np.vstack([ax, bx, cx])


def _build_cell(cell_block: Mapping[str, Any]) -> np.ndarray:
    vectors = cell_block.get("vectors")
    if isinstance(vectors, Mapping) and all(k in vectors for k in ("A", "B", "C")):
        cell = np.asarray([vectors["A"], vectors["B"], vectors["C"]], dtype=float)
    else:
        lengths = cell_block.get("abc")
        if lengths is None:
            raise ValueError("CP2K restart has no parseable &CELL A/B/C or ABC definition")
        angles = cell_block.get("angles") or [90.0, 90.0, 90.0]
        cell = _cell_from_lengths_angles(lengths, angles)
    if cell.shape != (3, 3):
        raise ValueError("CP2K restart produced an invalid 3x3 cell")
    if not np.all(np.isfinite(cell)):
        raise ValueError("CP2K restart cell contains non-finite values")
    if abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("CP2K restart cell has zero volume")
    return cell


def _types_from_symbols(symbols: Sequence[str], type_to_species: Optional[Sequence[str]]) -> np.ndarray:
    if type_to_species is not None:
        mapping = {str(sym): i + 1 for i, sym in enumerate(list(type_to_species))}
        try:
            return np.asarray([int(mapping[str(sym)]) for sym in symbols], dtype=int)
        except KeyError as exc:
            raise ValueError(f"CP2K restart contains symbol not present in type_to_species: {exc}") from exc
    mapping = {str(sym): i + 1 for i, sym in enumerate(sorted(set(str(x) for x in symbols)))}
    return np.asarray([int(mapping[str(sym)]) for sym in symbols], dtype=int)


def _cp2k_periodic_flags(value: Any) -> tuple[bool, bool, bool]:
    """Translate CP2K ``CELL/PERIODIC`` syntax to Cartesian PBC flags."""

    label = str(value if value is not None else "XYZ").strip().upper()
    if label in {"", "XYZ"}:
        return (True, True, True)
    if label == "NONE":
        return (False, False, False)
    if not set(label) <= {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported CP2K CELL PERIODIC value: {value!r}")
    return normalize_pbc(tuple(axis in label for axis in "XYZ"))


def read_cp2k_restart_frame(path: Path, *, type_to_species: Optional[Sequence[str]] = None) -> DumpFrame:
    """Read one final structure from a CP2K text ``*.restart`` file.

    The returned cell is stored as row vectors in Angstrom and positions are
    Cartesian Angstrom coordinates.  Only the structural ``&CELL`` and
    ``&COORD`` data are read; velocities, thermostat state, and other restart
    state are intentionally ignored.
    """

    p = Path(path)
    cell_blocks: list[dict[str, Any]] = []
    coord_blocks: list[dict[str, Any]] = []
    current_cell: Optional[dict[str, Any]] = None
    current_coord: Optional[dict[str, Any]] = None
    stack: list[str] = []

    try:
        lines = p.read_text(errors="replace").splitlines()
    except Exception as exc:
        raise ValueError(f"Could not read CP2K restart file: {p}") from exc

    for raw_line in lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("@"):
            # CP2K preprocessor directives are not structural content here.
            continue
        if line.startswith("&"):
            rest = line[1:].strip()
            parts = rest.split()
            if not parts:
                continue
            head = parts[0].upper()
            if head == "END":
                end_name = parts[1].upper() if len(parts) > 1 else (stack[-1] if stack else "")
                if current_coord is not None and end_name == "COORD":
                    if current_coord.get("rows"):
                        coord_blocks.append(current_coord)
                    current_coord = None
                if current_cell is not None and end_name == "CELL":
                    cell_blocks.append(current_cell)
                    current_cell = None
                if stack:
                    if end_name in stack:
                        while stack:
                            popped = stack.pop()
                            if popped == end_name:
                                break
                    else:
                        stack.pop()
                continue
            section = head
            stack.append(section)
            if section == "CELL":
                current_cell = {
                    "vectors": {},
                    "unit_scale": 1.0,
                    "abc": None,
                    "angles": None,
                    # CP2K's CELL default is periodic in XYZ.  Store the
                    # explicit effective value so restart provenance never
                    # depends on an implicit all-periodic hash fallback.
                    "periodic": "XYZ",
                }
            elif section == "COORD":
                current_coord = {"rows": [], "unit_scale": 1.0, "scaled": False}
                # CP2K allows options on the section line: &COORD SCALED
                if any(tok.upper() == "SCALED" for tok in parts[1:]):
                    current_coord["scaled"] = True
            continue

        toks = line.split()
        if not toks:
            continue
        key = toks[0].upper()

        if current_cell is not None:
            if key == "PERIODIC":
                if len(toks) < 2:
                    raise ValueError(f"CP2K restart {p} has CELL PERIODIC without a value")
                current_cell["periodic"] = str(toks[1]).upper()
                continue
            if key == "UNIT" and len(toks) >= 2:
                scale = _line_unit_scale(toks[1:], current_cell.get("unit_scale", 1.0))
                current_cell["unit_scale"] = scale
                continue
            scale = _line_unit_scale(toks[1:], current_cell.get("unit_scale", 1.0))
            nums = _numbers_from_tokens(toks[1:])
            if key in {"A", "B", "C"} and len(nums) >= 3:
                current_cell.setdefault("vectors", {})[key] = [float(scale * x) for x in nums[:3]]
            elif key == "ABC" and len(nums) >= 3:
                current_cell["abc"] = [float(scale * x) for x in nums[:3]]
            elif key in {"ALPHA_BETA_GAMMA", "ALPHA_BETA_GAMMA_DEG"} and len(nums) >= 3:
                current_cell["angles"] = [float(x) for x in nums[:3]]
            elif key in {"ALPHA", "BETA", "GAMMA"} and nums:
                angles = list(current_cell.get("angles") or [90.0, 90.0, 90.0])
                idx = {"ALPHA": 0, "BETA": 1, "GAMMA": 2}[key]
                angles[idx] = float(nums[0])
                current_cell["angles"] = angles
            continue

        if current_coord is not None:
            if key == "UNIT" and len(toks) >= 2:
                current_coord["unit_scale"] = _line_unit_scale(toks[1:], current_coord.get("unit_scale", 1.0))
                continue
            if key == "SCALED":
                current_coord["scaled"] = True
                continue
            kind = toks[0]
            nums = _numbers_from_tokens(toks[1:])
            if len(nums) < 3:
                continue
            symbol = _canonical_symbol_from_kind(kind, type_to_species)
            scale = 1.0 if bool(current_coord.get("scaled", False)) else _line_unit_scale(toks[1:], current_coord.get("unit_scale", 1.0))
            xyz = [float(scale * x) for x in nums[:3]]
            current_coord.setdefault("rows", []).append((symbol, xyz))
            continue

    if current_coord is not None and current_coord.get("rows"):
        coord_blocks.append(current_coord)
    if current_cell is not None:
        cell_blocks.append(current_cell)

    if not cell_blocks:
        raise ValueError(f"CP2K restart {p} has no &CELL section")
    if not coord_blocks:
        raise ValueError(f"CP2K restart {p} has no &COORD section")

    final_cell = cell_blocks[-1]
    cell = _build_cell(final_cell)
    pbc = _cp2k_periodic_flags(final_cell.get("periodic", "XYZ"))
    coord = coord_blocks[-1]
    rows = list(coord.get("rows") or [])
    if not rows:
        raise ValueError(f"CP2K restart {p} contains no coordinates")
    symbols = [str(sym) for sym, _xyz in rows]
    arr = np.asarray([xyz for _sym, xyz in rows], dtype=float)
    if bool(coord.get("scaled", False)):
        arr = arr @ cell
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"CP2K restart {p} produced invalid coordinate array")

    ids = np.arange(1, int(arr.shape[0]) + 1, dtype=int)
    return DumpFrame(
        timestep=0,
        ids=ids,
        types=_types_from_symbols(symbols, type_to_species),
        positions=arr,
        cell=np.asarray(cell, dtype=float),
        origin=np.zeros((3,), dtype=float),
        pbc=pbc,
    )
