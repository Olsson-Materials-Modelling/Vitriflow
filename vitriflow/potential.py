from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .config import KimConfig, LammpsPotentialConfig, MG2SiNPotentialConfig, PotentialConfig


_CORE_TABLE_BEGIN = "# vitriflow_core_table_begin "
_CORE_TABLE_PAIR = "# vitriflow_core_table_pair "
_CORE_TABLE_END = "# vitriflow_core_table_end"
_ACCEL_SUFFIXES = ("gpu", "intel", "kk", "omp", "opt")
# fallback
_PERIODIC_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
_PERIODIC_NUMBERS = {sym: idx + 1 for idx, sym in enumerate(_PERIODIC_SYMBOLS)}


def kim_init_line(kim: KimConfig) -> str:
    """Kim init line."""
    if bool(getattr(kim, "unit_conversion_mode", False)):
        return f"kim init {kim.model} {kim.user_units} unit_conversion_mode"
    return f"kim init {kim.model} {kim.user_units}"


def kim_interactions_line(kim: KimConfig) -> str:
    """Kim interactions line."""
    if kim.interactions == "fixed_types":
        return "kim interactions fixed_types"
    species = " ".join(str(s) for s in kim.interactions)
    return f"kim interactions {species}"


def potential_init_lines(pot: PotentialConfig) -> List[str]:
    """Potential init lines."""
    if isinstance(pot, KimConfig):
        return [kim_init_line(pot)]
    # non kim
    units = str(getattr(pot, "user_units", "")).strip()
    if not units:
        units = "metal"
    return [f"units {units}"]


def potential_interactions_list(pot: PotentialConfig) -> list[str]:
    """Potential interactions list."""
    if isinstance(pot, KimConfig):
        if pot.interactions == "fixed_types":
            return []
        return [str(x) for x in pot.interactions]
    return [str(x) for x in getattr(pot, "interactions", [])]


def _localize_command_tokens(tokens: list[str], files: list[Path]) -> list[str]:
    """Localize command tokens."""
    if not files:
        return tokens
    repl: dict[str, str] = {}
    names: set[str] = set()
    for f in files:
        try:
            repl[str(f)] = str(f.name)
        except Exception:
            pass
        try:
            repl[str(Path(f).resolve(strict=False))] = str(Path(f).name)
        except Exception:
            pass
        try:
            names.add(str(Path(f).name))
        except Exception:
            pass

    out: list[str] = []
    for tok in tokens:
        if tok in repl:
            out.append(repl[tok])
            continue
        # localize tokens basename
        # relative path
        try:
            nm = str(Path(tok).name)
            if nm in names and tok != nm:
                out.append(nm)
                continue
        except Exception:
            pass
        out.append(tok)
    return out


def localized_lammps_commands(pot: LammpsPotentialConfig) -> list[str]:
    """Localized lammps commands."""
    files = [Path(x) for x in (pot.files or [])]
    out: list[str] = []
    for ln in pot.commands:
        s = str(ln).strip()
        if not s:
            continue
        toks = s.split()
        toks = _localize_command_tokens(toks, files)
        out.append(" ".join(toks))
    return out


def mg2_sin_commands(pot: MG2SiNPotentialConfig) -> list[str]:
    """Mg2 sin commands."""
    n = int(pot.table_points)
    rcut = float(pot.x0_A)
    fname = str(pot.table_filename).strip() or "mg2_sin.table"

    # pairs table sections
    inter = [str(x) for x in pot.interactions]

    def _section(spi: str, spj: str) -> str:
        s = {spi, spj}
        if s == {"Si"}:
            return "SiSi"
        if s == {"N"}:
            return "NN"
        if s == {"Si", "N"}:
            return "SiN"
        raise ValueError(f"MG2SiN: unsupported species pair: {spi}-{spj}")

    cmds = [f"pair_style table linear {n}"]
    # i j pairs
    for i in range(1, 3):
        for j in range(i, 3):
            sec = _section(inter[i - 1], inter[j - 1])
            cmds.append(f"pair_coeff {i} {j} {fname} {sec} {rcut:.6g}")
    return cmds


def potential_default_lines(pot: PotentialConfig) -> list[str]:
    """Potential default lines."""
    if isinstance(pot, KimConfig):
        return [kim_interactions_line(pot)]
    if isinstance(pot, MG2SiNPotentialConfig):
        return mg2_sin_commands(pot)
    if isinstance(pot, LammpsPotentialConfig):
        return localized_lammps_commands(pot)
    raise TypeError(f"Unsupported potential config type: {type(pot)}")


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _strip_accel_suffix(style: str) -> str:
    s = str(style).strip()
    for suffix in _ACCEL_SUFFIXES:
        tag = "/" + suffix
        if s.endswith(tag):
            return s[: -len(tag)]
    return s


def _atomic_number_from_symbol(symbol: str) -> int:
    sym = str(symbol).strip()
    try:
        from ase.data import atomic_numbers

        z = int(atomic_numbers[sym])
        if z > 0:
            return z
    except Exception:
        pass
    z = _PERIODIC_NUMBERS.get(sym)
    if z is None:
        raise KeyError(f"Unknown atomic number for symbol '{sym}'")
    return int(z)


def _distance_scale_from_angstrom(units_style: str) -> float:
    us = str(units_style).strip().lower()
    if us in {"metal", "real"}:
        return 1.0  #
    if us == "electron":
        return 1.0 / 0.529177210903  # bohr
    if us == "nano":
        return 0.1  # nm
    if us == "si":
        return 1.0e-10  # m
    if us == "cgs":
        return 1.0e-8  # cm
    raise ValueError(f"Unsupported LAMMPS units style for ZBL tabulation: {units_style!r}")


def _coulomb_prefactor_energy_distance(units_style: str) -> float:
    """Coulomb prefactor energy."""
    us = str(units_style).strip().lower()
    if us in {"metal", "nano"}:
        # metal differs distance
        base = 14.3996454784255
        return base if us == "metal" else base * 0.1
    if us == "real":
        return 332.06371329919216  # kcal mol e
    if us == "electron":
        return 1.0  # hartree bohr atomic
    if us == "si":
        return 2.3070775523517024e-28  # j m
    if us == "cgs":
        return 2.3070775523517024e-19  # erg cm
    raise ValueError(f"Unsupported LAMMPS units style for ZBL tabulation: {units_style!r}")


def _expand_type_selector(token: str, ntypes: int) -> list[int]:
    tok = str(token).strip()
    if tok == "*":
        return list(range(1, int(ntypes) + 1))
    if "*" not in tok:
        val = int(tok)
        if not (1 <= val <= int(ntypes)):
            raise ValueError(f"atom type selector {token!r} out of range 1..{ntypes}")
        return [val]
    left, right = tok.split("*", 1)
    lo = 1 if left == "" else int(left)
    hi = int(ntypes) if right == "" else int(right)
    if lo > hi:
        lo, hi = hi, lo
    lo = max(1, lo)
    hi = min(int(ntypes), hi)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _parse_buckingham_pair_style(base_cmds: Sequence[str]) -> dict[str, Any]:
    pair_style_line = None
    for ln in base_cmds:
        s = str(ln).strip()
        if s.startswith("pair_style"):
            pair_style_line = s
            break
    if pair_style_line is None:
        raise ValueError("No pair_style line found in potential command block")

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    raw_style = toks[1]
    style = _strip_accel_suffix(raw_style)
    args = toks[2:]

    if style == "buck":
        if len(args) < 1:
            raise ValueError(f"Buckingham style requires a cutoff: {pair_style_line}")
        return {
            "raw_style": raw_style,
            "style": style,
            "buck_cutoff": float(args[0]),
            "coul_style": None,
            "coul_cutoff": None,
        }

    if style in {"buck/coul/cut", "buck/coul/long", "buck/coul/msm"}:
        if len(args) < 1:
            raise ValueError(f"Buckingham/Coulomb style requires at least one cutoff: {pair_style_line}")
        buck_cut = float(args[0])
        coul_cut = float(args[0]) if len(args) == 1 else float(args[1])
        coul_style = {
            "buck/coul/cut": "coul/cut",
            "buck/coul/long": "coul/long",
            "buck/coul/msm": "coul/msm",
        }[style]
        return {
            "raw_style": raw_style,
            "style": style,
            "buck_cutoff": buck_cut,
            "coul_style": coul_style,
            "coul_cutoff": coul_cut,
        }

    raise ValueError(
        "Tabulated Buckingham+core conversion currently supports only "
        "'buck', 'buck/coul/cut', 'buck/coul/long', or 'buck/coul/msm'"
    )


def _extract_buckingham_coeffs(
    base_cmds: Sequence[str],
    *,
    ntypes: int,
    style_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    coeffs: dict[tuple[int, int], dict[str, Any]] = {}
    other: list[str] = []
    shift_enabled = False
    base_style = str(style_info["style"])
    buck_cut_global = float(style_info["buck_cutoff"])
    coul_cut_global = style_info.get("coul_cutoff", None)

    for ln in base_cmds:
        s = str(ln).strip()
        if not s:
            continue
        if s.startswith("pair_style"):
            continue
        if s.startswith("pair_coeff"):
            toks = s.split()
            if len(toks) < 6:
                raise ValueError(f"Malformed Buckingham pair_coeff line: {s}")
            sel_i = toks[1]
            sel_j = toks[2]
            args = toks[3:]
            if args and _strip_accel_suffix(args[0]) == base_style:
                args = args[1:]
            if len(args) < 3:
                raise ValueError(f"Malformed Buckingham pair_coeff line: {s}")
            A = float(args[0])
            rho = float(args[1])
            C = float(args[2])
            if rho <= 0.0:
                raise ValueError(f"Buckingham rho must be > 0 in line: {s}")
            rest = args[3:]
            buck_cut = buck_cut_global
            coul_cut = coul_cut_global
            if base_style == "buck":
                if len(rest) >= 1:
                    buck_cut = float(rest[0])
            elif base_style == "buck/coul/cut":
                if len(rest) >= 1:
                    buck_cut = float(rest[0])
                if len(rest) >= 2:
                    coul_cut = float(rest[1])
            elif base_style in {"buck/coul/long", "buck/coul/msm"}:
                if len(rest) >= 1:
                    buck_cut = float(rest[0])
                if len(rest) >= 2:
                    raise ValueError(f"Per-pair Coulombic cutoffs are not supported for {base_style}: {s}")
            else:  # pragma: no cover
                raise ValueError(f"Unsupported Buckingham style: {base_style}")

            ii = _expand_type_selector(sel_i, ntypes)
            jj = _expand_type_selector(sel_j, ntypes)
            if not ii or not jj:
                raise ValueError(f"Empty pair selector in line: {s}")
            for i in ii:
                for j in jj:
                    a, b = (i, j) if i <= j else (j, i)
                    coeffs[(a, b)] = {
                        "pair": [int(a), int(b)],
                        "A": float(A),
                        "rho": float(rho),
                        "C": float(C),
                        "buck_cutoff": float(buck_cut),
                        "coul_cutoff": (None if coul_cut is None else float(coul_cut)),
                    }
            continue

        if s.startswith("pair_modify"):
            toks = s.split()
            keep: list[str] = [toks[0]]
            i = 1
            while i < len(toks):
                key = toks[i].lower()
                if key == "shift" and i + 1 < len(toks):
                    shift_enabled = str(toks[i + 1]).strip().lower() in {"yes", "true", "1", "on"}
                    i += 2
                    continue
                keep.append(toks[i])
                i += 1
            if len(keep) > 1:
                other.append(" ".join(keep))
            continue

        other.append(s)

    missing = [(i, j) for i in range(1, ntypes + 1) for j in range(i, ntypes + 1) if (i, j) not in coeffs]
    if missing:
        raise ValueError(f"Missing Buckingham coefficients for pairs: {missing}")

    if base_style == "buck/coul/cut":
        coul_cuts = {float(v["coul_cutoff"]) for v in coeffs.values() if v.get("coul_cutoff") is not None}
        if len(coul_cuts) > 1:
            raise ValueError(
                "Tabulated Buckingham+ZBL conversion does not support pair-specific Coulombic cutoffs for buck/coul/cut"
            )

    pairs = [coeffs[(i, j)] for i in range(1, ntypes + 1) for j in range(i, ntypes + 1)]
    return pairs, other, bool(shift_enabled)


def _tabulated_core_metadata_lines(spec: dict[str, Any]) -> list[str]:
    head = {k: v for k, v in spec.items() if k != "pairs"}
    lines = [f"{_CORE_TABLE_BEGIN}{_json_dumps_compact(head)}"]
    for pair in spec.get("pairs", []):
        lines.append(f"{_CORE_TABLE_PAIR}{_json_dumps_compact(pair)}")
    lines.append(_CORE_TABLE_END)
    return lines


def _parse_tabulated_core_spec(potential_lines: Sequence[str]) -> Optional[dict[str, Any]]:
    head: Optional[dict[str, Any]] = None
    pairs: list[dict[str, Any]] = []
    for ln in potential_lines:
        s = str(ln).strip()
        if s.startswith(_CORE_TABLE_BEGIN):
            head = json.loads(s[len(_CORE_TABLE_BEGIN) :])
            pairs = []
            continue
        if s.startswith(_CORE_TABLE_PAIR):
            if head is not None:
                pairs.append(json.loads(s[len(_CORE_TABLE_PAIR) :]))
            continue
        if s == _CORE_TABLE_END and head is not None:
            head = dict(head)
            head["pairs"] = pairs
            return head
    return None


def update_tabulated_core_metadata_lines(
    potential_lines: Sequence[str],
    **updates: Any,
) -> list[str]:
    """Update tabulated core."""
    spec = _parse_tabulated_core_spec(potential_lines)
    if spec is None:
        raise ValueError("potential_lines do not contain vitriflow tabulated-core metadata")
    merged = dict(spec)
    merged.update({k: v for k, v in updates.items() if v is not None})
    table_block = _tabulated_core_metadata_lines(merged)
    out: list[str] = []
    in_block = False
    for ln in potential_lines:
        s = str(ln).strip()
        if s.startswith(_CORE_TABLE_BEGIN):
            if not in_block:
                out.extend(table_block)
            in_block = True
            continue
        if s == _CORE_TABLE_END and in_block:
            in_block = False
            continue
        if in_block:
            continue
        out.append(str(ln))
    if not out or not str(out[0]).startswith(_CORE_TABLE_BEGIN):
        out = table_block + out
    return out


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_validated_tabulated_core_source(stage_dir: Path, spec: Mapping[str, Any]) -> Optional[Path]:
    fname = Path(str(spec.get("filename", "")).strip() or "buckingham_core.table").name
    expected_sha = str(spec.get("sha256", "") or "").strip().lower() or None
    checked: set[Path] = set()
    mismatches: list[tuple[Path, str]] = []
    candidates: list[Path] = []
    stage_dir = Path(stage_dir)
    candidates.append(stage_dir / fname)
    for anc in [stage_dir] + list(stage_dir.parents):
        candidates.append(anc / "preflight" / "potential_override" / fname)
    for cand in candidates:
        cand = Path(cand)
        if cand in checked:
            continue
        checked.add(cand)
        if not cand.exists():
            continue
        if expected_sha is None:
            return cand
        got = _sha256_path(cand)
        if got.lower() == expected_sha:
            return cand
        mismatches.append((cand, got))
    if mismatches:
        parts = ", ".join(f"{path} ({digest[:12]})" for path, digest in mismatches)
        raise ValueError(
            f"validated tabulated-core source hash mismatch for {fname}: expected {expected_sha}, found {parts}"
        )
    return None


def _strip_kspace_modify_gewald(line: str) -> Optional[str]:
    toks = str(line).split()
    if len(toks) < 3 or toks[0] != "kspace_modify":
        return str(line).strip() or None
    keep = [toks[0]]
    i = 1
    while i < len(toks):
        key = toks[i].lower()
        if key == "gewald" and i + 1 < len(toks):
            i += 2
            continue
        keep.append(toks[i])
        i += 1
    return " ".join(keep) if len(keep) > 1 else None


def _explicit_gewald_from_commands(lines: Sequence[str]) -> Optional[float]:
    value: Optional[float] = None
    for ln in lines:
        toks = str(ln).split()
        if len(toks) < 3 or toks[0] != "kspace_modify":
            continue
        i = 1
        while i < len(toks) - 1:
            if toks[i].lower() == "gewald":
                value = float(toks[i + 1])
                i += 2
                continue
            i += 1
    return value


def _kspace_table_keyword_from_commands(lines: Sequence[str]) -> Optional[str]:
    """Kspace table keyword."""
    for ln in lines:
        toks = str(ln).split()
        if len(toks) < 2 or toks[0] != "kspace_style":
            continue
        style = _strip_accel_suffix(toks[1].strip().lower())
        if style.startswith("pppm"):
            return "pppm"
        if style.startswith("ewald"):
            return "ewald"
        if style.startswith("msm"):
            return "msm"
        return None
    return None


def _fixed_type_charges(
    species: Sequence[str],
    charges: Optional[Mapping[str, float]],
) -> dict[int, float]:
    if charges is None:
        raise ValueError(
            "tabulated Buckingham/Coulomb conversion requires fixed species charges; "
            "set structure.charges for all interacting species"
        )
    out: dict[int, float] = {}
    for idx, sym in enumerate(species, start=1):
        key = str(sym)
        if key not in charges:
            raise ValueError(
                "tabulated Buckingham/Coulomb conversion requires fixed species charges; "
                f"missing structure.charges entry for {key!r}"
            )
        out[int(idx)] = float(charges[key])
    return out


def build_tabulated_buckingham_core_lines(
    base_cmds: Sequence[str],
    *,
    species: Sequence[str],
    units_style: str,
    r_in: float,
    r_out: float,
    table_points: int,
    table_filename: str,
    table_r_min: float,
    charges: Optional[Mapping[str, float]] = None,
    gewald: Optional[float] = None,
) -> list[str]:
    """Tabulated buckingham core."""
    species_list = [str(x) for x in species]
    if len(species_list) == 0:
        raise ValueError("species ordering is required for Buckingham+ZBL tabulation")

    style_info = _parse_buckingham_pair_style(base_cmds)
    pair_specs, other, shift_enabled = _extract_buckingham_coeffs(
        base_cmds,
        ntypes=len(species_list),
        style_info=style_info,
    )
    fname = Path(str(table_filename).strip() or "buckingham_core.table").name
    n = int(table_points)
    rlo = float(table_r_min)
    if not (math.isfinite(rlo) and rlo > 0.0):
        raise ValueError("table_r_min must be finite and > 0")

    z_numbers = {i + 1: _atomic_number_from_symbol(sym) for i, sym in enumerate(species_list)}
    charge_by_type: Optional[dict[int, float]] = None
    coul_mode: Optional[str] = None
    gewald_value: Optional[float] = None
    table_kspace_keyword: Optional[str] = None
    base_style = str(style_info["style"])
    if base_style == "buck":
        coul_mode = None
    elif base_style == "buck/coul/cut":
        coul_mode = "cut"
        charge_by_type = _fixed_type_charges(species_list, charges)
    elif base_style == "buck/coul/long":
        coul_mode = "long"
        charge_by_type = _fixed_type_charges(species_list, charges)
        gewald_value = _explicit_gewald_from_commands(base_cmds) if gewald is None else float(gewald)
        if not (gewald_value is not None and math.isfinite(float(gewald_value)) and float(gewald_value) > 0.0):
            raise ValueError(
                "tabulated buck/coul/long requires a fixed G-ewald value; set "
                "potential.core_repulsion.table_gewald or provide kspace_modify gewald"
            )
        gewald_value = float(gewald_value)
        table_kspace_keyword = _kspace_table_keyword_from_commands(base_cmds)
        if table_kspace_keyword not in {"pppm", "ewald"}:
            raise ValueError(
                "tabulated buck/coul/long requires kspace_style pppm* or ewald* so the runtime "
                "can use pair_style table with the matching compatibility keyword"
            )
    elif base_style == "buck/coul/msm":
        raise ValueError(
            "tabulated buck/coul/msm is not supported: vitriflow only tabulates Coulombic Buckingham "
            "for Ewald/PPPM-compatible real-space splitting"
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Buckingham style: {base_style}")

    other_clean: list[str] = []
    has_kspace_style = False
    for ln in other:
        s = str(ln).strip()
        if not s:
            continue
        if s.startswith("kspace_style"):
            has_kspace_style = True
            other_clean.append(s)
            continue
        if s.startswith("kspace_modify") and coul_mode == "long":
            stripped = _strip_kspace_modify_gewald(s)
            if stripped is not None:
                other_clean.append(stripped)
            continue
        other_clean.append(s)

    if coul_mode == "long" and not has_kspace_style:
        raise ValueError("tabulated buck/coul/long requires a kspace_style command in the base potential block")

    spec_pairs: list[dict[str, Any]] = []
    pair_lines: list[str] = []
    for entry in pair_specs:
        i, j = (int(entry["pair"][0]), int(entry["pair"][1]))
        buck_cut = float(entry["buck_cutoff"])
        coul_cut = entry.get("coul_cutoff", None)
        pair_cut = max(float(buck_cut), float(r_out))
        if coul_mode in {"cut", "long"} and coul_cut is not None:
            pair_cut = max(float(pair_cut), float(coul_cut))
        if rlo >= pair_cut:
            raise ValueError(
                f"table_r_min ({rlo:g}) must be smaller than the pair cutoff ({pair_cut:g}) for pair ({i},{j})"
            )
        section = f"P{i}_{j}"
        q_i = None if charge_by_type is None else float(charge_by_type[i])
        q_j = None if charge_by_type is None else float(charge_by_type[j])
        spec_pairs.append(
            {
                "section": section,
                "pair": [i, j],
                "species": [species_list[i - 1], species_list[j - 1]],
                "A": float(entry["A"]),
                "rho": float(entry["rho"]),
                "C": float(entry["C"]),
                "buck_cutoff": float(buck_cut),
                "pair_cutoff": float(pair_cut),
                "z_i": int(z_numbers[i]),
                "z_j": int(z_numbers[j]),
                "r_in": float(r_in),
                "r_out": float(r_out),
                "shift_buck": bool(shift_enabled),
                "coul_mode": coul_mode,
                "coul_cutoff": (None if coul_cut is None else float(coul_cut)),
                "q_i": q_i,
                "q_j": q_j,
                "gewald": gewald_value,
            }
        )
        pair_lines.append(f"pair_coeff {i} {j} {fname} {section} {pair_cut:.6g}")

    metadata = {
        "version": 5,
        "kind": "buckingham_zbl_table",
        "filename": fname,
        "points": int(n),
        "r_min": float(rlo),
        "units": str(units_style).strip() or "metal",
        "base_style": str(style_info["style"]),
        "coul_style": style_info.get("coul_style", None),
        "coul_cutoff": style_info.get("coul_cutoff", None),
        "coul_mode": coul_mode,
        "gewald": gewald_value,
        "kspace_keyword": table_kspace_keyword,
        "force_mode": "analytic",
        "include_fprime": True,
    }
    metadata["pairs"] = spec_pairs

    lines = _tabulated_core_metadata_lines(metadata)
    if coul_mode in {None, "cut"}:
        lines.append(f"pair_style table linear {n}")
    else:
        lines.append(f"pair_style table linear {n} {str(table_kspace_keyword)}")
    lines.extend(pair_lines)
    lines.extend(other_clean)
    if coul_mode == "long":
        lines.append(f"kspace_modify gewald {float(gewald_value):.16g}")
    return lines


def _buckingham_energy_derivatives(
    r: np.ndarray,
    *,
    A: float,
    rho: float,
    C: float,
    cutoff: float,
    shift: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    U = np.zeros_like(r)
    dU = np.zeros_like(r)
    d2U = np.zeros_like(r)
    m = r < float(cutoff)
    if not np.any(m):
        return U, dU, d2U
    rr = r[m]
    expm = np.exp(-rr / float(rho))
    U_m = float(A) * expm - float(C) / (rr**6)
    dU_m = -float(A) * expm / float(rho) + 6.0 * float(C) / (rr**7)
    d2U_m = float(A) * expm / (float(rho) ** 2) - 42.0 * float(C) / (rr**8)
    if bool(shift):
        U_m = U_m - (float(A) * math.exp(-float(cutoff) / float(rho)) - float(C) / (float(cutoff) ** 6))
    U[m] = U_m
    dU[m] = dU_m
    d2U[m] = d2U_m
    return U, dU, d2U


def _zbl_base_energy_derivatives(
    r: np.ndarray,
    *,
    z_i: int,
    z_j: int,
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    pref = _coulomb_prefactor_energy_distance(units_style) * float(z_i) * float(z_j)
    a = 0.46850 * _distance_scale_from_angstrom(units_style) / (
        float(z_i) ** 0.23 + float(z_j) ** 0.23
    )
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError(f"Invalid ZBL screening length for Z={z_i}, {z_j}")
    x = r / float(a)
    coeff = np.asarray([0.18175, 0.50986, 0.28022, 0.02817], dtype=float)
    decay = np.asarray([3.19980, 0.94229, 0.40290, 0.20162], dtype=float)
    ex = np.exp(-np.outer(x, decay))
    phi = ex @ coeff
    phi_p = ex @ (-coeff * decay)
    phi_pp = ex @ (coeff * decay * decay)
    U = pref * phi / r
    dU = pref * (phi_p / (float(a) * r) - phi / (r * r))
    d2U = pref * (phi_pp / ((float(a) ** 2) * r) - 2.0 * phi_p / (float(a) * r * r) + 2.0 * phi / (r * r * r))
    return U, dU, d2U


def _zbl_switched_energy_derivative(
    r: np.ndarray,
    *,
    z_i: int,
    z_j: int,
    units_style: str,
    r_in: float,
    r_out: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zbl switched energy."""
    r = np.asarray(r, dtype=float)
    if not (float(r_in) > 0.0 and float(r_out) > float(r_in)):
        raise ValueError(f"Invalid ZBL switching radii: r_in={r_in}, r_out={r_out}")

    U0, dU0, d2U0 = _zbl_base_energy_derivatives(r, z_i=z_i, z_j=z_j, units_style=units_style)
    U = np.zeros_like(r)
    dU = np.zeros_like(r)

    inner = float(r_in)
    outer = float(r_out)
    delta = outer - inner
    Uc, dUc, d2Uc = _zbl_base_energy_derivatives(
        np.asarray([outer], dtype=float), z_i=z_i, z_j=z_j, units_style=units_style
    )
    Uc = float(Uc[0])
    dUc = float(dUc[0])
    d2Uc = float(d2Uc[0])
    A = (-3.0 * dUc + delta * d2Uc) / (delta * delta)
    B = (2.0 * dUc - delta * d2Uc) / (delta * delta * delta)
    C = -Uc + 0.5 * delta * dUc - (delta * delta * d2Uc) / 12.0

    m_inner = r < inner
    if np.any(m_inner):
        U[m_inner] = U0[m_inner] + C
        dU[m_inner] = dU0[m_inner]

    m_mid = (r >= inner) & (r < outer)
    if np.any(m_mid):
        dr = r[m_mid] - inner
        switch = (A / 3.0) * (dr**3) + (B / 4.0) * (dr**4) + C
        d_switch = A * (dr**2) + B * (dr**3)
        U[m_mid] = U0[m_mid] + switch
        dU[m_mid] = dU0[m_mid] + d_switch

    # r outer already
    return U, dU


def _coul_cut_energy_derivatives(
    r: np.ndarray,
    *,
    q_i: float,
    q_j: float,
    units_style: str,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    U = np.zeros_like(r)
    dU = np.zeros_like(r)
    if float(q_i) == 0.0 or float(q_j) == 0.0:
        return U, dU
    m = r < float(cutoff)
    if not np.any(m):
        return U, dU
    rr = r[m]
    pref = _coulomb_prefactor_energy_distance(units_style) * float(q_i) * float(q_j)
    U[m] = pref / rr
    dU[m] = -pref / (rr * rr)
    return U, dU


def _coul_long_short_energy_derivatives(
    r: np.ndarray,
    *,
    q_i: float,
    q_j: float,
    units_style: str,
    cutoff: float,
    gewald: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    U = np.zeros_like(r)
    dU = np.zeros_like(r)
    if float(q_i) == 0.0 or float(q_j) == 0.0:
        return U, dU
    if not (math.isfinite(float(gewald)) and float(gewald) > 0.0):
        raise ValueError("tabulated coul/long contribution requires a finite G-ewald > 0")
    m = r < float(cutoff)
    if not np.any(m):
        return U, dU
    rr = r[m]
    pref = _coulomb_prefactor_energy_distance(units_style) * float(q_i) * float(q_j)
    g = float(gewald)
    gr = g * rr
    erfc_vals = np.asarray([math.erfc(float(x)) for x in gr], dtype=float)
    exp_vals = np.exp(-(gr * gr))
    U[m] = pref * erfc_vals / rr
    dU[m] = pref * (-(2.0 * g / math.sqrt(math.pi)) * exp_vals / rr - erfc_vals / (rr * rr))
    return U, dU



def _tabulated_buckingham_section_arrays(
    pair: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray | float]:
    """Tabulated buckingham section."""
    n = int(spec["points"])
    rmin = float(spec["r_min"])
    units_style = str(spec.get("units", "metal")).strip() or "metal"
    force_mode = str(spec.get("force_mode", "analytic") or "analytic").strip().lower()
    pair_cut = float(pair["pair_cutoff"])
    buck_cut = float(pair["buck_cutoff"])
    if int(n) < 2:
        raise ValueError("Tabulated Buckingham core requires at least 2 points")
    if not (pair_cut > rmin):
        raise ValueError(f"Tabulated Buckingham core requires pair_cutoff > r_min for section {pair.get('section', '?')}")
    r = np.sqrt(np.linspace(rmin * rmin, pair_cut * pair_cut, n, dtype=float))
    U_b, dU_b, _d2U_b = _buckingham_energy_derivatives(
        r,
        A=float(pair["A"]),
        rho=float(pair["rho"]),
        C=float(pair["C"]),
        cutoff=float(buck_cut),
        shift=bool(pair.get("shift_buck", False)),
    )
    U_z, dU_z = _zbl_switched_energy_derivative(
        r,
        z_i=int(pair["z_i"]),
        z_j=int(pair["z_j"]),
        units_style=units_style,
        r_in=float(pair["r_in"]),
        r_out=float(pair["r_out"]),
    )
    coul_mode = pair.get("coul_mode", spec.get("coul_mode", None))
    if coul_mode == "cut":
        U_c, dU_c = _coul_cut_energy_derivatives(
            r,
            q_i=float(pair.get("q_i", 0.0) or 0.0),
            q_j=float(pair.get("q_j", 0.0) or 0.0),
            units_style=units_style,
            cutoff=float(pair.get("coul_cutoff", pair_cut)),
        )
    elif coul_mode == "long":
        U_c, dU_c = _coul_long_short_energy_derivatives(
            r,
            q_i=float(pair.get("q_i", 0.0) or 0.0),
            q_j=float(pair.get("q_j", 0.0) or 0.0),
            units_style=units_style,
            cutoff=float(pair.get("coul_cutoff", pair_cut)),
            gewald=float(pair.get("gewald", spec.get("gewald", 0.0) or 0.0)),
        )
    else:
        U_c = np.zeros_like(r)
        dU_c = np.zeros_like(r)
    U = U_b + U_z + U_c
    if force_mode == "analytic":
        F = -(dU_b + dU_z + dU_c)
    elif force_mode == "fd_consistent":
        if r.size < 3:
            F = -(dU_b + dU_z + dU_c)
        else:
            F = -np.asarray(np.gradient(U, r, edge_order=2), dtype=float)
    else:  # pragma: no cover - guarded by preflight table refinement
        raise ValueError(f"Unsupported tabulated Buckingham force_mode: {force_mode}")
    if r.size >= 3:
        dF = np.asarray(np.gradient(F, r, edge_order=2), dtype=float)
    elif r.size == 2:
        slope = (float(F[1]) - float(F[0])) / (float(r[1]) - float(r[0]))
        dF = np.asarray([slope, slope], dtype=float)
    else:  # pragma: no cover
        dF = np.zeros_like(F)
    return {
        "r": np.asarray(r, dtype=float),
        "energy": np.asarray(U, dtype=float),
        "force": np.asarray(F, dtype=float),
        "fprime_lo": float(dF[0]) if dF.size else 0.0,
        "fprime_hi": float(dF[-1]) if dF.size else 0.0,
    }


def write_tabulated_buckingham_core_table(path: Path, spec: dict[str, Any]) -> None:
    """Tabulated buckingham core."""
    path = Path(path)
    n = int(spec["points"])
    rmin = float(spec["r_min"])
    units_style = str(spec.get("units", "metal")).strip() or "metal"
    pairs = list(spec.get("pairs", []))
    include_fprime = bool(spec.get("include_fprime", True))
    if int(n) < 2:
        raise ValueError("Tabulated Buckingham core requires at least 2 points")
    if not (math.isfinite(rmin) and rmin > 0.0):
        raise ValueError("Tabulated Buckingham core requires r_min > 0")
    if len(pairs) == 0:
        raise ValueError("Tabulated Buckingham core spec contains no pair sections")

    header = [
        f"# vitriflow Buckingham tabulated real-space potential generated by vitriflow",
        f"# UNITS: {units_style}",
        "# columns: index r energy force (force = -dU/dr)",
        "",
    ]
    lines: list[str] = []
    lines.extend(header)

    for pair in pairs:
        section = str(pair["section"])
        data = _tabulated_buckingham_section_arrays(pair, spec=spec)
        r = np.asarray(data["r"], dtype=float)
        U = np.asarray(data["energy"], dtype=float)
        F = np.asarray(data["force"], dtype=float)
        pair_cut = float(pair["pair_cutoff"])
        lines.append(section)
        hdr = f"N {n} RSQ {rmin:.16g} {pair_cut:.16g}"
        if include_fprime:
            hdr += f" FPRIME {float(data['fprime_lo']):.16g} {float(data['fprime_hi']):.16g}"
        lines.append(hdr)
        lines.append("")
        for idx in range(n):
            lines.append(f"{idx+1:d} {r[idx]:.16g} {U[idx]:.16g} {F[idx]:.16g}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def prepare_potential_files(
    pot: PotentialConfig,
    stage_dir: Path,
    potential_lines: Optional[Sequence[str]] = None,
) -> None:
    """Potential files."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pot, MG2SiNPotentialConfig):
        out = stage_dir / (str(pot.table_filename).strip() or "mg2_sin.table")
        write_mg2_sin_table(out, pot)

    if isinstance(pot, LammpsPotentialConfig):
        for f in pot.files or []:
            src = Path(f)
            if not src.exists():
                raise FileNotFoundError(f"Potential auxiliary file not found: {src}")
            dst = stage_dir / src.name
            if dst.resolve() == src.resolve():
                continue
            shutil.copy2(src, dst)

    if potential_lines is not None:
        spec = _parse_tabulated_core_spec(potential_lines)
        if spec is not None:
            out = stage_dir / (str(spec.get("filename", "")).strip() or "buckingham_core.table")
            src = _find_validated_tabulated_core_source(stage_dir, spec)
            if src is not None:
                if out.exists():
                    try:
                        if out.resolve() != src.resolve():
                            shutil.copy2(src, out)
                    except Exception:
                        shutil.copy2(src, out)
                else:
                    shutil.copy2(src, out)
            elif str(spec.get("sha256", "")).strip():
                raise FileNotFoundError(
                    "validated tabulated-core source file not found; expected a preflight-generated table "
                    f"for {out.name} with sha256={spec.get('sha256', '')}"
                )
            else:
                write_tabulated_buckingham_core_table(out, spec)

    # kim nothing prepare
    return


def _p5_taper(r: np.ndarray, *, x1: float, x0: float) -> tuple[np.ndarray, np.ndarray]:
    """P5 taper."""
    r = np.asarray(r, dtype=float)
    g = np.ones_like(r)
    gp = np.zeros_like(r)

    # r x0
    m0 = r >= float(x0)
    if np.any(m0):
        g[m0] = 0.0
        gp[m0] = 0.0

    # x1 x0 polynomial
    m = (r > float(x1)) & (r < float(x0))
    if np.any(m):
        t = (r[m] - float(x1)) / (float(x0) - float(x1))
        g[m] = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        dgdt = -30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4
        gp[m] = dgdt / (float(x0) - float(x1))

    # r x1 already
    return g, gp


def _tang_toennies_f6(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tang toennies f6."""
    x = np.asarray(x, dtype=float)
    exp_mx = np.exp(-x)
    # sum k x
    s = np.zeros_like(x)
    fact = 1.0
    xpow = np.ones_like(x)
    for k in range(0, 7):
        if k > 0:
            fact *= float(k)
            xpow = xpow * x
        s = s + xpow / fact
    f = 1.0 - exp_mx * s
    # derivative identity exp
    df = exp_mx * (x**6) / float(math.factorial(6))
    return f, df


def _mg2_morse(r: np.ndarray, *, D0: float, alpha: float, r0: float) -> tuple[np.ndarray, np.ndarray]:
    """Mg2 morse."""
    r = np.asarray(r, dtype=float)
    dr = r - float(r0)
    e1 = np.exp(-float(alpha) * dr)
    e2 = e1 * e1
    U = float(D0) * (e2 - 2.0 * e1)
    dUdr = 2.0 * float(alpha) * float(D0) * (e1 - e2)
    return U, dUdr


def _mg2_general(r: np.ndarray, *, A: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Mg2 general."""
    r = np.asarray(r, dtype=float)
    expm = np.exp(-r / float(rho))
    U = float(A) * expm / r
    dUdr = -float(A) * expm * (1.0 / (float(rho) * r) + 1.0 / (r * r))
    return U, dUdr


def _mg2_dispersion_nn(r: np.ndarray, *, C6: float, b6: float) -> tuple[np.ndarray, np.ndarray]:
    """Mg2 dispersion nn."""
    r = np.asarray(r, dtype=float)
    x = float(b6) * r
    f6, df6dx = _tang_toennies_f6(x)
    df6dr = float(b6) * df6dx

    r6 = r**6
    r7 = r6 * r

    U = -float(C6) * f6 / r6
    dUdr = (-float(C6) * df6dr / r6) + (6.0 * float(C6) * f6 / r7)
    return U, dUdr


def write_mg2_sin_table(path: Path, pot: MG2SiNPotentialConfig) -> None:
    """Mg2 sin table."""

    path = Path(path)
    n = int(pot.table_points)
    rmin = float(pot.r_min_A)
    rmax = float(pot.x0_A)

    r = np.linspace(rmin, rmax, n, dtype=float)
    # taper derivative
    g, gp = _p5_taper(r, x1=float(pot.x1_A), x0=float(pot.x0_A))

    # si n
    U0_sin, dU0_sin = _mg2_morse(r, D0=float(pot.D0_eV), alpha=float(pot.alpha_invA), r0=float(pot.r0_A))
    U_sin = g * U0_sin
    dU_sin = gp * U0_sin + g * dU0_sin
    F_sin = -dU_sin

    # si
    U0_sisi, dU0_sisi = _mg2_general(r, A=float(pot.A_SiSi_eVA), rho=float(pot.rho_SiSi_A))
    U_sisi = g * U0_sisi
    dU_sisi = gp * U0_sisi + g * dU0_sisi
    F_sisi = -dU_sisi

    # nn
    U0_nn_rep, dU0_nn_rep = _mg2_general(r, A=float(pot.A_NN_eVA), rho=float(pot.rho_NN_A))
    U0_nn_disp, dU0_nn_disp = _mg2_dispersion_nn(r, C6=float(pot.C6_NN_eVA6), b6=float(pot.b6_NN_invA))
    U0_nn = U0_nn_rep + U0_nn_disp
    dU0_nn = dU0_nn_rep + dU0_nn_disp
    U_nn = g * U0_nn
    dU_nn = gp * U0_nn + g * dU0_nn
    F_nn = -dU_nn

    # enforce exact cutoff
    for arr in (U_sin, F_sin, U_sisi, F_sisi, U_nn, F_nn):
        arr[-1] = 0.0

    def _section_lines(name: str, U: np.ndarray, F: np.ndarray) -> list[str]:
        sec: list[str] = []
        sec.append(str(name))
        sec.append(f"N {n} R {rmin:.16g} {rmax:.16g}")
        sec.append("")
        for i in range(n):
            sec.append(f"{i+1:d} {r[i]:.16g} {U[i]:.16g} {F[i]:.16g}")
        sec.append("")
        return sec

    header = [
        "# MG2 Si-N tabulated potential generated by vitriflow",
        "# columns: index r energy force (force = -dU/dr)",
        "",
    ]

    text_lines: list[str] = []
    text_lines.extend(header)
    text_lines.extend(_section_lines("SiSi", U_sisi, F_sisi))
    text_lines.extend(_section_lines("SiN", U_sin, F_sin))
    text_lines.extend(_section_lines("NN", U_nn, F_nn))

    path.write_text("\n".join(text_lines) + "\n")
