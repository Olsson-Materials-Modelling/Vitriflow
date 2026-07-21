from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from .config import (
    KimConfig,
    LammpsPotentialConfig,
    MG2SiNPotentialConfig,
    PotentialConfig,
    validated_lammps_localized_filename,
)
from .lammps_units import (
    charge_from_elementary_factor,
    energy_from_ev_factor,
    lammps_charge_coulomb_prefactor,
    length_from_angstrom_factor,
    normalize_lammps_units_style,
    zbl_coulomb_prefactor,
)
from .utils import stable_file_identity


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


def _lammps_rsq_grid(rlo: float, rhi: float, points: int) -> np.ndarray:
    """Return the exact radius grid reconstructed by LAMMPS for ``RSQ``.

    ``pair_table.cpp`` does not retain the radius column supplied in an RSQ
    table.  It recomputes every radius using the operation ordering below::

        sqrt(rlo*rlo + ((rhi*rhi-rlo*rlo) * i) / (N-1))

    ``sqrt(linspace(rlo**2, rhi**2, N))`` is mathematically equivalent but is
    not floating-point equivalent: NumPy forms one step and then multiplies
    it, whereas LAMMPS multiplies before dividing.  The one-ULP differences
    are physically negligible but can change LAMMPS's force/secant predicate
    exactly at a genuine analytic inflection.  Keep generation, reference
    evaluation, and warning auditing on this single execution grid.
    """

    n = int(points)
    lower = float(rlo)
    upper = float(rhi)
    if n < 2:
        raise ValueError("LAMMPS RSQ grid requires at least two points")
    if not (
        math.isfinite(lower)
        and math.isfinite(upper)
        and 0.0 < lower < upper
    ):
        raise ValueError(
            "LAMMPS RSQ grid requires finite bounds with 0 < rlo < rhi"
        )

    # Use separate NumPy operations to preserve the C++ double operation
    # order and avoid an implementation-dependent fused multiply-add.
    lower_sq = np.float64(lower) * np.float64(lower)
    delta_sq = (
        np.float64(upper) * np.float64(upper) - lower_sq
    )
    indices = np.arange(n, dtype=np.float64)
    scaled = np.multiply(delta_sq, indices)
    scaled = np.divide(scaled, np.float64(n - 1))
    radius_sq = np.add(lower_sq, scaled)
    radii = np.sqrt(radius_sq)
    # Finite input bounds are not sufficient here: squaring an extreme bound
    # can overflow, while a sub-ULP interval or too many knots can collapse
    # adjacent reconstructed radii.  LAMMPS table interpolation requires a
    # finite, strictly increasing coordinate grid, so reject an unrepresentable
    # RSQ table before energies or endpoint derivatives are attached to it.
    if not np.all(np.isfinite(radii)) or not np.all(np.diff(radii) > 0.0):
        raise ValueError(
            "LAMMPS RSQ grid is not representable as finite, strictly "
            "increasing binary64 radii for the requested bounds and point count"
        )
    return radii


def _lammps_double_token(value: float) -> str:
    """Serialize one binary64 value as a concise exact round-trip token."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("LAMMPS numeric tokens must be finite")
    token = repr(number)
    # LAMMPS accepts integral-looking floating literals.  Removing only the
    # redundant suffix preserves Python's shortest-roundtrip guarantee while
    # retaining the established readable command/table form (``10``, not
    # ``10.0``; ``0.1``, not ``0.10000000000000001``).
    if token.endswith(".0"):
        token = token[:-2]
    return token


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
    rcut = float(pot.x0_A) * length_from_angstrom_factor(pot.user_units)
    raw_name = pot.table_filename
    fname = validated_lammps_localized_filename(
        "mg2_sin.table" if raw_name == "" else raw_name,
        field_name="MG2 table_filename",
    )

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
    return float(length_from_angstrom_factor(units_style))


def _coulomb_prefactor_energy_distance(units_style: str) -> float:
    """LAMMPS Coulomb coefficient for charges in the native charge unit.

    This is engine-facing: combined KSpace-compatible tables must reproduce
    LAMMPS's serialized ``qqr2e`` constant exactly.  ZBL continues to use the
    independent physical-unit prefactor.
    """

    return float(lammps_charge_coulomb_prefactor(units_style))


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
    pair_style_lines: list[str] = []
    for ln in base_cmds:
        s = str(ln).split("#", 1)[0].strip()
        toks = s.split()
        if toks and toks[0] == "pair_style":
            pair_style_lines.append(s)
    if not pair_style_lines:
        raise ValueError("No pair_style line found in potential command block")
    if len(pair_style_lines) != 1:
        raise ValueError(
            "Buckingham table conversion requires exactly one effective pair_style "
            f"command, found {len(pair_style_lines)}: {pair_style_lines}"
        )
    pair_style_line = pair_style_lines[0]

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    raw_style = toks[1]
    style = _strip_accel_suffix(raw_style)
    args = toks[2:]

    if style == "hybrid/overlay":
        # This parser is deliberately exact.  A hybrid command has no general
        # machine-readable grammar: every pair style owns a variable number of
        # arguments.  Supporting a small, closed set is the only way to prove
        # that the generated single table contains every additive term.
        supported = {
            "buck": (1, 1),
            "buck/coul/cut": (1, 2),
            "buck/coul/long": (1, 2),
            "morse": (1, 1),
            "coul/cut": (1, 1),
            "coul/long": (1, 1),
        }
        components: dict[str, list[float]] = {}
        idx = 0
        while idx < len(args):
            raw_component = args[idx]
            component = _strip_accel_suffix(raw_component).lower()
            if component not in supported:
                raise ValueError(
                    "Tabulated Buckingham hybrid/overlay conversion supports only "
                    "one Buckingham substyle plus optional morse and one optional "
                    f"coul/cut or coul/long substyle; got {raw_component!r} in: "
                    f"{pair_style_line}"
                )
            if component in components:
                raise ValueError(
                    "Tabulated Buckingham hybrid/overlay conversion rejects multiple "
                    f"instances of substyle {component!r}: {pair_style_line}"
                )
            idx += 1
            minimum, maximum = supported[component]
            values: list[float] = []
            while idx < len(args) and len(values) < maximum:
                next_component = _strip_accel_suffix(args[idx]).lower()
                if next_component in supported:
                    break
                try:
                    value = float(args[idx])
                except ValueError as exc:
                    raise ValueError(
                        "Unsupported or ambiguous hybrid/overlay substyle argument "
                        f"{args[idx]!r} after {component!r}: {pair_style_line}"
                    ) from exc
                if not (math.isfinite(value) and value > 0.0):
                    raise ValueError(
                        f"{component} cutoff must be finite and > 0: {pair_style_line}"
                    )
                values.append(value)
                idx += 1
            if len(values) < minimum:
                raise ValueError(
                    f"{component} requires {minimum} cutoff argument(s): {pair_style_line}"
                )
            components[component] = values

        buck_components = [
            name
            for name in ("buck", "buck/coul/cut", "buck/coul/long")
            if name in components
        ]
        if len(buck_components) != 1:
            raise ValueError(
                "Tabulated Buckingham hybrid/overlay conversion requires exactly one "
                f"Buckingham substyle, found {buck_components}: {pair_style_line}"
            )
        buck_style = buck_components[0]
        buck_values = components[buck_style]
        buck_cut = float(buck_values[0])

        embedded_coul_style = {
            "buck": None,
            "buck/coul/cut": "coul/cut",
            "buck/coul/long": "coul/long",
        }[buck_style]
        embedded_coul_cut = (
            None
            if embedded_coul_style is None
            else float(buck_values[0] if len(buck_values) == 1 else buck_values[1])
        )
        separate_coul = [
            name for name in ("coul/cut", "coul/long") if name in components
        ]
        if embedded_coul_style is not None and separate_coul:
            raise ValueError(
                "hybrid/overlay cannot combine a Buckingham/Coulomb substyle with a "
                f"second Coulomb substyle {separate_coul}; that would double count Coulomb"
            )
        if len(separate_coul) > 1:
            raise ValueError(
                "hybrid/overlay conversion supports at most one Coulomb substyle; "
                f"found {separate_coul}"
            )
        coul_style = embedded_coul_style
        coul_cut = embedded_coul_cut
        if separate_coul:
            coul_style = separate_coul[0]
            coul_cut = float(components[coul_style][0])

        return {
            "raw_style": raw_style,
            "style": style,
            "hybrid": True,
            "buck_style": buck_style,
            "buck_cutoff": buck_cut,
            "coul_style": coul_style,
            "coul_cutoff": coul_cut,
            "morse_style": "morse" if "morse" in components else None,
            "morse_cutoff": (
                float(components["morse"][0]) if "morse" in components else None
            ),
            "hybrid_components": sorted(components),
        }

    if style == "buck":
        if len(args) != 1:
            raise ValueError(
                "Buckingham pair_style requires exactly one cutoff argument: "
                f"{pair_style_line}"
            )
        buck_cut = float(args[0])
        if not (math.isfinite(buck_cut) and buck_cut > 0.0):
            raise ValueError(f"Buckingham cutoff must be finite and > 0: {pair_style_line}")
        return {
            "raw_style": raw_style,
            "style": style,
            "hybrid": False,
            "buck_style": style,
            "buck_cutoff": buck_cut,
            "coul_style": None,
            "coul_cutoff": None,
            "morse_style": None,
            "morse_cutoff": None,
        }

    if style in {"buck/coul/cut", "buck/coul/long", "buck/coul/msm"}:
        if len(args) not in {1, 2}:
            raise ValueError(
                "Buckingham/Coulomb pair_style requires exactly one or two cutoff "
                f"arguments: {pair_style_line}"
            )
        buck_cut = float(args[0])
        coul_cut = float(args[0]) if len(args) == 1 else float(args[1])
        if not all(math.isfinite(x) and x > 0.0 for x in (buck_cut, coul_cut)):
            raise ValueError(
                f"Buckingham/Coulomb cutoffs must be finite and > 0: {pair_style_line}"
            )
        coul_style = {
            "buck/coul/cut": "coul/cut",
            "buck/coul/long": "coul/long",
            "buck/coul/msm": "coul/msm",
        }[style]
        return {
            "raw_style": raw_style,
            "style": style,
            "hybrid": False,
            "buck_style": style,
            "buck_cutoff": buck_cut,
            "coul_style": coul_style,
            "coul_cutoff": coul_cut,
            "morse_style": None,
            "morse_cutoff": None,
        }

    raise ValueError(
        "Tabulated Buckingham+core conversion currently supports only "
        "'buck', 'buck/coul/cut', 'buck/coul/long', 'buck/coul/msm', or an "
        "exact supported 'hybrid/overlay' composition"
    )


def inspect_buckingham_core_compatibility(
    base_cmds: Sequence[str],
) -> dict[str, Any]:
    """Fail-closed classification for autocore dispatch.

    A genuinely non-Buckingham pair style returns ``is_buckingham=False``.
    Any command block that lexically contains a Buckingham style is instead
    passed through the exact parser: supported blocks return their parsed
    representation, while malformed or lossy hybrids raise ``ValueError``.
    This prevents a Buckingham-containing hybrid from being silently treated
    as an unrelated potential with autocore skipped.
    """

    pair_style_lines: list[str] = []
    for line in base_cmds:
        stripped = str(line).split("#", 1)[0].strip()
        tokens = stripped.split()
        if tokens and tokens[0] == "pair_style":
            pair_style_lines.append(stripped)
    if not pair_style_lines:
        return {
            "is_buckingham": False,
            "supported": False,
            "pair_style": None,
            "parsed": None,
        }

    contains_buckingham = any(
        _strip_accel_suffix(token).lower().startswith("buck")
        for line in pair_style_lines
        for token in line.split()[1:]
    )
    if not contains_buckingham:
        return {
            "is_buckingham": False,
            "supported": False,
            "pair_style": pair_style_lines[0],
            "parsed": None,
        }

    parsed = _parse_buckingham_pair_style(base_cmds)
    return {
        "is_buckingham": True,
        "supported": True,
        "pair_style": pair_style_lines[0],
        "parsed": parsed,
    }


def _validated_equal_special_bonds_weights(
    base_cmds: Sequence[str],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return LJ/Coulomb weights, rejecting mappings a single table cannot represent."""

    lj = (0.0, 0.0, 0.0)
    coul = (0.0, 0.0, 0.0)
    for line in base_cmds:
        toks = str(line).strip().split()
        if not toks or toks[0] != "special_bonds":
            continue
        args = toks[1:]
        if not args:
            raise ValueError("Malformed special_bonds command")
        # LAMMPS resets all special weights to zero for every special_bonds
        # command before applying that command's keywords.
        lj = (0.0, 0.0, 0.0)
        coul = (0.0, 0.0, 0.0)
        if len(args) == 1 and args[0].lower() == "default":
            continue
        preset = args[0].lower()
        if preset == "amber":
            lj = (0.0, 0.0, 0.5)
            coul = (0.0, 0.0, 5.0 / 6.0)
            i = 1
        elif preset == "charmm":
            lj = coul = (0.0, 0.0, 0.0)
            i = 1
        elif preset == "dreiding":
            lj = coul = (0.0, 0.0, 1.0)
            i = 1
        elif preset == "fene":
            lj = coul = (0.0, 1.0, 1.0)
            i = 1
        else:
            i = 0
        while i < len(args):
            key = args[i].lower()
            if key in {"angle", "dihedral", "one/five"} and i + 1 < len(args):
                i += 2
                continue
            if key not in {"lj", "coul", "lj/coul"} or i + 3 >= len(args):
                raise ValueError(
                    "Buckingham+Coulomb table conversion requires an explicit special_bonds "
                    "lj, coul, or lj/coul triplet"
                )
            vals = tuple(float(x) for x in args[i + 1 : i + 4])
            if not all(math.isfinite(x) for x in vals):
                raise ValueError("special_bonds weights must be finite")
            if key in {"lj", "lj/coul"}:
                lj = vals  # type: ignore[assignment]
            if key in {"coul", "lj/coul"}:
                coul = vals  # type: ignore[assignment]
            i += 4
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-15) for a, b in zip(lj, coul)):
        raise ValueError(
            "Buckingham+Coulomb table conversion requires identical special_lj and "
            f"special_coul weights, got lj={lj} and coul={coul}"
        )
    return lj, coul


def _extract_hybrid_buckingham_coeffs(
    base_cmds: Sequence[str],
    *,
    ntypes: int,
    style_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Extract a closed, provably additive Buckingham hybrid/overlay model."""

    buck_style = str(style_info["buck_style"])
    coul_style = style_info.get("coul_style", None)
    morse_style = style_info.get("morse_style", None)
    separate_coul = coul_style in {"coul/cut", "coul/long"} and buck_style == "buck"
    buck: dict[tuple[int, int], dict[str, Any]] = {}
    coul_pairs: set[tuple[int, int]] = set()
    morse: dict[tuple[int, int], dict[str, Any]] = {}
    other: list[str] = []
    shift_enabled = False

    def _explicit_pair(select_i: str, select_j: str, line: str) -> tuple[int, int]:
        if "*" in select_i or "*" in select_j:
            raise ValueError(
                "hybrid/overlay Buckingham conversion requires explicit integer pair_coeff "
                f"selectors; wildcard/range selectors cannot be proven unambiguous: {line}"
            )
        try:
            i = int(select_i)
            j = int(select_j)
        except ValueError as exc:
            raise ValueError(f"Invalid hybrid/overlay pair_coeff selectors: {line}") from exc
        if not (1 <= i <= int(ntypes) and 1 <= j <= int(ntypes)):
            raise ValueError(
                f"hybrid/overlay pair_coeff selector outside 1..{ntypes}: {line}"
            )
        return (i, j) if i <= j else (j, i)

    for ln in base_cmds:
        s = str(ln).split("#", 1)[0].strip()
        if not s:
            continue
        toks = s.split()
        command = toks[0]
        if command == "pair_style":
            continue
        if command == "pair_coeff":
            if len(toks) < 4:
                raise ValueError(f"Malformed hybrid/overlay pair_coeff line: {s}")
            key = _explicit_pair(toks[1], toks[2], s)
            substyle = _strip_accel_suffix(toks[3]).lower()
            args = toks[4:]
            allowed_substyles = {buck_style}
            if separate_coul:
                allowed_substyles.add(str(coul_style))
            if morse_style is not None:
                allowed_substyles.add("morse")
            if substyle not in allowed_substyles:
                raise ValueError(
                    "hybrid/overlay pair_coeff names an unsupported, absent, or ambiguous "
                    f"substyle {toks[3]!r}: {s}"
                )

            if substyle == buck_style:
                if key in buck:
                    raise ValueError(
                        f"Duplicate hybrid/overlay Buckingham pair_coeff for pair {key}: {s}"
                    )
                allowed_lengths = {
                    "buck": {3, 4},
                    "buck/coul/cut": {3, 4, 5},
                    "buck/coul/long": {3, 4},
                }[buck_style]
                if len(args) not in allowed_lengths:
                    raise ValueError(
                        "Malformed hybrid/overlay Buckingham pair_coeff; expected "
                        f"{sorted(allowed_lengths)} values after the substyle: {s}"
                    )
                A, rho, C = (float(args[0]), float(args[1]), float(args[2]))
                if not all(math.isfinite(x) for x in (A, rho, C)) or rho <= 0.0:
                    raise ValueError(
                        f"Buckingham A/rho/C must be finite and rho > 0: {s}"
                    )
                rest = [float(x) for x in args[3:]]
                buck_cut = float(style_info["buck_cutoff"])
                pair_coul_cut = style_info.get("coul_cutoff", None)
                if rest:
                    buck_cut = rest[0]
                if buck_style == "buck/coul/cut" and len(rest) == 2:
                    pair_coul_cut = rest[1]
                if not (math.isfinite(buck_cut) and buck_cut > 0.0):
                    raise ValueError(f"Buckingham pair cutoff must be finite and > 0: {s}")
                if pair_coul_cut is not None and not (
                    math.isfinite(float(pair_coul_cut)) and float(pair_coul_cut) > 0.0
                ):
                    raise ValueError(f"Coulomb pair cutoff must be finite and > 0: {s}")
                buck[key] = {
                    "pair": [int(key[0]), int(key[1])],
                    "A": float(A),
                    "rho": float(rho),
                    "C": float(C),
                    "buck_cutoff": float(buck_cut),
                    "coul_cutoff": (
                        None if pair_coul_cut is None else float(pair_coul_cut)
                    ),
                }
                continue

            if substyle in {"coul/cut", "coul/long"}:
                if args:
                    raise ValueError(
                        f"{substyle} hybrid pair_coeff takes no coefficient arguments: {s}"
                    )
                if key in coul_pairs:
                    raise ValueError(
                        f"Duplicate hybrid/overlay {substyle} pair_coeff for pair {key}: {s}"
                    )
                coul_pairs.add(key)
                continue

            if substyle == "morse":
                if key in morse:
                    raise ValueError(
                        f"Duplicate hybrid/overlay Morse pair_coeff for pair {key}: {s}"
                    )
                if len(args) not in {3, 4}:
                    raise ValueError(
                        "Morse pair_coeff requires D0, alpha, r0, and optional cutoff: "
                        f"{s}"
                    )
                D0, alpha, r0 = (float(args[0]), float(args[1]), float(args[2]))
                if not all(math.isfinite(x) for x in (D0, alpha, r0)):
                    raise ValueError(f"Morse coefficients must be finite: {s}")
                if D0 < 0.0 or alpha <= 0.0 or r0 <= 0.0:
                    raise ValueError(
                        "Morse D0 must be >= 0 and alpha/r0 must be > 0: " + s
                    )
                morse_cut = (
                    float(style_info["morse_cutoff"])
                    if len(args) == 3
                    else float(args[3])
                )
                if not (math.isfinite(morse_cut) and morse_cut > 0.0):
                    raise ValueError(f"Morse cutoff must be finite and > 0: {s}")
                morse[key] = {
                    "D0": float(D0),
                    "alpha": float(alpha),
                    "r0": float(r0),
                    "cutoff": float(morse_cut),
                }
                continue

            raise AssertionError(f"unhandled supported hybrid substyle {substyle}")

        if command == "pair_modify":
            lowered = [token.lower() for token in toks[1:]]
            if "pair" in lowered:
                raise ValueError(
                    "style-targeted pair_modify commands are ambiguous after replacing a "
                    f"hybrid/overlay model by one table: {s}"
                )
            keep: list[str] = [toks[0]]
            i = 1
            while i < len(toks):
                key = toks[i].lower()
                if key == "shift" and i + 1 < len(toks):
                    shift_enabled = toks[i + 1].lower() in {"yes", "true", "1", "on"}
                    if shift_enabled:
                        raise ValueError(
                            "hybrid/overlay pair_modify shift yes cannot be mapped "
                            "unambiguously across Buckingham, Morse, and Coulomb substyles"
                        )
                    i += 2
                    continue
                if key == "tail" and i + 1 < len(toks):
                    if toks[i + 1].lower() in {"yes", "true", "1", "on"}:
                        raise ValueError(
                            "Buckingham hybrid table conversion cannot preserve "
                            "pair_modify tail yes"
                        )
                    i += 2
                    continue
                if key in {"table", "tabinner"}:
                    if i + 1 >= len(toks):
                        raise ValueError(f"Malformed pair_modify {key} command: {s}")
                    # These tune the analytic coul/long lookup that is removed
                    # by the combined KSpace-compatible replacement table.
                    # Retaining them would misleadingly suggest that the
                    # accepted runtime still uses the source Coulomb lookup.
                    i += 2
                    continue
                keep.append(toks[i])
                i += 1
            if len(keep) > 1:
                other.append(" ".join(keep))
            continue

        if command == "dielectric":
            if len(toks) != 2:
                raise ValueError(f"Malformed dielectric command: {s}")
            dielectric = float(toks[1])
            if not math.isclose(dielectric, 1.0, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    "Buckingham hybrid table conversion currently requires dielectric 1.0"
                )
            continue

        other.append(s)

    required = {
        (i, j) for i in range(1, int(ntypes) + 1) for j in range(i, int(ntypes) + 1)
    }
    missing_buck = sorted(required.difference(buck))
    if missing_buck:
        raise ValueError(
            "hybrid/overlay requires explicit Buckingham pair_coeff coverage for every "
            f"unordered type pair; missing {missing_buck}"
        )
    if separate_coul:
        missing_coul = sorted(required.difference(coul_pairs))
        if missing_coul:
            raise ValueError(
                f"hybrid/overlay requires explicit {coul_style} pair_coeff coverage for "
                f"every unordered type pair; missing {missing_coul}"
            )
    if morse_style is not None and not morse:
        raise ValueError(
            "hybrid/overlay declares morse but no explicit Morse pair_coeff was supplied"
        )

    pairs: list[dict[str, Any]] = []
    for key in sorted(required):
        entry = dict(buck[key])
        if separate_coul:
            entry["coul_cutoff"] = float(style_info["coul_cutoff"])
        term = morse.get(key)
        entry["morse_terms"] = [] if term is None else [dict(term)]
        pairs.append(entry)
    return pairs, other, bool(shift_enabled)


def _extract_buckingham_coeffs(
    base_cmds: Sequence[str],
    *,
    ntypes: int,
    style_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    if bool(style_info.get("hybrid", False)):
        return _extract_hybrid_buckingham_coeffs(
            base_cmds, ntypes=ntypes, style_info=style_info
        )

    coeffs: dict[tuple[int, int], dict[str, Any]] = {}
    other: list[str] = []
    shift_enabled = False
    base_style = str(style_info["style"])
    buck_cut_global = float(style_info["buck_cutoff"])
    coul_cut_global = style_info.get("coul_cutoff", None)

    for ln in base_cmds:
        s = str(ln).split("#", 1)[0].strip()
        if not s:
            continue
        command = s.split()[0]
        if command == "pair_style":
            continue
        if command == "pair_coeff":
            toks = s.split()
            if len(toks) < 6:
                raise ValueError(f"Malformed Buckingham pair_coeff line: {s}")
            sel_i = toks[1]
            sel_j = toks[2]
            args = toks[3:]
            if args and _strip_accel_suffix(args[0]) == base_style:
                args = args[1:]
            allowed_lengths = {
                "buck": {3, 4},
                "buck/coul/cut": {3, 4, 5},
                "buck/coul/long": {3, 4},
                "buck/coul/msm": {3, 4},
            }[base_style]
            if len(args) not in allowed_lengths:
                expected = ", ".join(str(x) for x in sorted(allowed_lengths))
                raise ValueError(
                    "Malformed or surplus Buckingham pair_coeff arguments: expected "
                    f"{expected} coefficient/cutoff values after selectors/style, got "
                    f"{len(args)} in: {s}"
                )
            if len(args) < 3:  # defensive; all allowed sets contain >=3
                raise ValueError(f"Malformed Buckingham pair_coeff line: {s}")
            A = float(args[0])
            rho = float(args[1])
            C = float(args[2])
            if not all(math.isfinite(x) for x in (A, rho, C)):
                raise ValueError(f"Buckingham A, rho, and C must be finite in line: {s}")
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
            cutoffs = [float(buck_cut)]
            if coul_cut is not None:
                cutoffs.append(float(coul_cut))
            if not all(math.isfinite(x) and x > 0.0 for x in cutoffs):
                raise ValueError(f"Buckingham pair cutoffs must be finite and > 0 in line: {s}")

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
                if key == "tail" and i + 1 < len(toks):
                    tail_enabled = str(toks[i + 1]).strip().lower() in {
                        "yes", "true", "1", "on"
                    }
                    if tail_enabled:
                        raise ValueError(
                            "Buckingham table conversion cannot preserve pair_modify tail yes; "
                            "analytic long-range tail corrections are not represented by pair_style table"
                        )
                    # Explicit tail=no is the default and is not meaningful for
                    # the replacement table style, so omit it.
                    i += 2
                    continue
                if key in {"table", "tabinner"}:
                    if i + 1 >= len(toks):
                        raise ValueError(f"Malformed pair_modify {key} command: {s}")
                    # Source-only coul/long interpolation controls have no
                    # place in the combined table/KSpace runtime.
                    i += 2
                    continue
                keep.append(toks[i])
                i += 1
            if len(keep) > 1:
                other.append(" ".join(keep))
            continue

        if s.startswith("dielectric"):
            toks = s.split()
            if len(toks) != 2:
                raise ValueError(f"Malformed dielectric command: {s}")
            dielectric = float(toks[1])
            if not math.isclose(dielectric, 1.0, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    "Buckingham table conversion currently requires dielectric 1.0; "
                    "a non-unity dielectric is not incorporated consistently into both "
                    "the tabulated real-space and reciprocal-space reference terms"
                )
            # Unity is the LAMMPS default; omit the redundant command.
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


def tabulated_core_metadata(
    potential_lines: Optional[Sequence[str]],
) -> Optional[dict[str, Any]]:
    """Return a detached copy of protected autocore table metadata, if any."""

    if potential_lines is None:
        return None
    spec = _parse_tabulated_core_spec(potential_lines)
    return None if spec is None else dict(spec)


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
    return str(stable_file_identity(Path(path))["sha256"])


def _atomic_write_generated_potential_text(path: Path, text: str) -> Path:
    """Publish a generated potential file without following an old alias."""

    destination = Path(path)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise RuntimeError(
            f"Cannot publish generated potential file safely: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    stable_file_identity(destination, reject_final_symlink=True)
    return destination


def _atomic_copy_verified_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_size_bytes: Optional[int] = None,
) -> Path:
    """Atomically stage one regular file and prove the copied bytes.

    The source identity is obtained with the stable-inode reader.  The copy is
    written to a same-directory temporary file and independently hashed before
    ``os.replace``.  Reopening a concurrently replaced source can therefore
    never place bytes under an identity established for an earlier inode.
    """

    src = Path(source)
    dst = Path(destination)
    source_identity = stable_file_identity(src)
    expected_sha = str(
        expected_sha256
        if expected_sha256 is not None
        else source_identity["sha256"]
    ).strip().lower()
    expected_size = int(
        expected_size_bytes
        if expected_size_bytes is not None
        else source_identity["size_bytes"]
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("verified file copy requires a valid SHA-256 digest")
    if expected_size < 0:
        raise ValueError("verified file copy requires a non-negative size")
    if (
        str(source_identity["sha256"]).lower() != expected_sha
        or int(source_identity["size_bytes"]) != expected_size
    ):
        raise ValueError(
            f"source file does not match its protected identity: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        raise ValueError(f"verified file destination must not be a symbolic link: {dst}")
    try:
        if dst.exists() and dst.resolve(strict=True) == Path(
            str(source_identity["resolved_path"])
        ):
            return dst
    except OSError:
        pass

    tmp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{dst.name}.",
            suffix=".tmp",
            dir=dst.parent,
            delete=False,
        ) as handle:
            tmp_name = handle.name
            with src.open("rb") as src_handle:
                shutil.copyfileobj(src_handle, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        tmp = Path(tmp_name)
        copied = stable_file_identity(tmp, reject_final_symlink=True)
        if (
            str(copied["sha256"]).lower() != expected_sha
            or int(copied["size_bytes"]) != expected_size
        ):
            raise ValueError(
                f"staged file bytes do not match the protected source identity: {src}"
            )
        os.replace(tmp, dst)
        tmp_name = None
        try:
            dir_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
            dir_fd = os.open(str(dst.parent), dir_flags)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some non-POSIX/network filesystems do not expose directory
            # fsync.  Atomic replace plus post-replacement content validation
            # remains the enforced runtime contract there.
            pass
        final = stable_file_identity(dst, reject_final_symlink=True)
        if (
            str(final["sha256"]).lower() != expected_sha
            or int(final["size_bytes"]) != expected_size
        ):
            raise ValueError(
                f"staged destination failed post-replacement verification: {dst}"
            )
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    return dst


def _assert_regular_path_below_root(path: Path, *, root: Path) -> Path:
    """Resolve a regular file without permitting symlink escape below root."""

    root_path = Path(root).resolve(strict=False)
    candidate = Path(path)
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"protected path escapes its result tree: {candidate}") from exc
    cursor = root_path
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"protected path must not contain symbolic links: {cursor}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"protected path escapes its result tree: {candidate}") from exc
    return resolved


def validated_tabulated_core_path(
    potential_lines: Optional[Sequence[str]],
    *,
    root: Path,
) -> Optional[Path]:
    """Resolve and authenticate a generated autocore table below ``root``.

    Production-plan metadata carries the expected SHA-256.  A replay must find
    the exact regular file in the source result tree; it may never silently
    regenerate a table whose realized LAMMPS interpolation was already
    verified during preflight.
    """

    spec = tabulated_core_metadata(potential_lines)
    if spec is None:
        return None
    expected = str(spec.get("sha256", "") or "").strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(
            "protected tabulated-core metadata is missing a valid SHA-256 digest"
        )
    expected_size_raw = spec.get("size_bytes")
    expected_size = (
        None if expected_size_raw is None else int(expected_size_raw)
    )
    if expected_size is not None and expected_size < 0:
        raise ValueError(
            "protected tabulated-core metadata has an invalid size_bytes value"
        )
    raw_filename = spec.get("filename", "")
    filename = validated_lammps_localized_filename(
        "buckingham_core.table" if raw_filename == "" else raw_filename,
        field_name="protected tabulated-core filename",
    )

    root_path = Path(root).expanduser().resolve(strict=False)
    candidates: list[Path] = []
    source_relpath = spec.get("source_relpath")
    if source_relpath is not None and str(source_relpath).strip():
        rel = Path(str(source_relpath))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(
                "protected tabulated-core source_relpath must remain inside its result tree"
            )
        candidates.append(root_path / rel)
    candidates.extend(
        [
            root_path / "preflight" / "potential_override" / filename,
            root_path / filename,
        ]
    )

    checked: set[Path] = set()
    mismatches: list[tuple[Path, str]] = []
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not candidate.exists() and not candidate.is_symlink():
            continue
        try:
            resolved_candidate = _assert_regular_path_below_root(
                candidate, root=root_path
            )
        except FileNotFoundError:
            continue
        identity = stable_file_identity(
            resolved_candidate,
            reject_final_symlink=True,
        )
        got = str(identity["sha256"]).lower()
        size_matches = expected_size is None or int(identity["size_bytes"]) == expected_size
        if got == expected and size_matches:
            return resolved_candidate
        mismatches.append((resolved_candidate, got))
    if mismatches:
        found = ", ".join(f"{path} ({digest[:12]})" for path, digest in mismatches)
        raise ValueError(
            "protected tabulated-core source hash mismatch: "
            f"expected {expected}, found {found}"
        )
    raise FileNotFoundError(
        "protected tabulated-core source was not found below "
        f"{root_path}; expected {filename} with sha256={expected}"
    )


def stage_validated_tabulated_core_for_replay(
    potential_lines: Optional[Sequence[str]],
    *,
    source_root: Path,
    target_root: Path,
) -> Optional[Path]:
    """Copy a preflight-verified table into a new production result tree."""

    source = validated_tabulated_core_path(potential_lines, root=source_root)
    if source is None:
        return None
    spec = tabulated_core_metadata(potential_lines)
    assert spec is not None
    expected = str(spec["sha256"]).strip().lower()
    expected_size_raw = spec.get("size_bytes")
    expected_size = (
        None if expected_size_raw is None else int(expected_size_raw)
    )
    raw_filename = spec.get("filename", "")
    filename = validated_lammps_localized_filename(
        source.name if raw_filename == "" else raw_filename,
        field_name="protected tabulated-core filename",
    )
    target_root = Path(target_root).expanduser().resolve(strict=False)
    target_dir = target_root / "preflight" / "potential_override"
    for path in (target_root, target_root / "preflight", target_dir):
        if path.exists() and path.is_symlink():
            raise ValueError(
                f"tabulated-core replay destination must not contain symbolic links: {path}"
            )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.is_symlink():
        raise ValueError(
            f"tabulated-core replay destination must not be a symbolic link: {target}"
        )
    if target.exists():
        current = stable_file_identity(target, reject_final_symlink=True)
        if (
            str(current["sha256"]).lower() != expected
            or (
                expected_size is not None
                and int(current["size_bytes"]) != expected_size
            )
        ):
            raise ValueError(
                "existing tabulated-core replay destination does not match "
                f"the protected identity: {target}"
            )
        return target
    return _atomic_copy_verified_regular_file(
        source,
        target,
        expected_sha256=expected,
        expected_size_bytes=expected_size,
    )


def _find_validated_tabulated_core_source(stage_dir: Path, spec: Mapping[str, Any]) -> Optional[Path]:
    raw_filename = spec.get("filename", "")
    fname = validated_lammps_localized_filename(
        "buckingham_core.table" if raw_filename == "" else raw_filename,
        field_name="tabulated core filename",
    )
    expected_sha = str(spec.get("sha256", "") or "").strip().lower() or None
    checked: set[Path] = set()
    mismatches: list[tuple[Path, str]] = []
    candidates: list[tuple[Path, Path]] = []
    # Canonicalise the stage entry point before walking its ancestors.  A
    # user-facing outdir may legitimately be a symlink to an HPC scratch
    # filesystem; that alias is not an escape from the actual result tree.
    # Symlinks *inside* the canonical tree remain forbidden below.
    stage_dir = Path(stage_dir).expanduser().resolve(strict=False)
    candidates.append((stage_dir / fname, stage_dir))
    for anc in [stage_dir] + list(stage_dir.parents):
        candidates.append(
            (anc / "preflight" / "potential_override" / fname, anc)
        )
    for cand, search_root in candidates:
        cand = Path(cand)
        if cand in checked:
            continue
        checked.add(cand)
        if not cand.exists() and not cand.is_symlink():
            continue
        try:
            resolved_candidate = _assert_regular_path_below_root(
                cand, root=search_root
            )
        except FileNotFoundError:
            continue
        if expected_sha is None:
            return resolved_candidate
        got = _sha256_path(resolved_candidate)
        if got.lower() == expected_sha:
            return resolved_candidate
        mismatches.append((resolved_candidate, got))
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
    *,
    units_style: str,
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
        configured = float(charges[key])
        if not math.isfinite(configured):
            raise ValueError(
                "tabulated Buckingham/Coulomb conversion requires finite fixed "
                f"species charges; got {charges[key]!r} for {key!r}"
            )
        # Public configuration stores formal/partial charges as multiples of
        # e.  LAMMPS SI/CGS/micro styles instead consume C/statC/pC.
        native = configured * charge_from_elementary_factor(units_style)
        if not math.isfinite(native):
            raise ValueError(f"Native charge conversion overflowed for species {key!r}")
        out[int(idx)] = native
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
    has_bonded_topology: Optional[bool] = None,
    table_style: str = "linear",
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
    raw_table_name = str(table_filename)
    fname = validated_lammps_localized_filename(
        "buckingham_core.table" if raw_table_name == "" else raw_table_name,
        field_name="tabulated core filename",
    )
    n = int(table_points)
    rlo = float(table_r_min)
    interpolation_style = str(table_style).strip().lower()
    if interpolation_style not in {"linear", "spline"}:
        raise ValueError(
            "tabulated Buckingham interpolation style must be 'linear' or 'spline'"
        )
    if not (math.isfinite(rlo) and rlo > 0.0):
        raise ValueError("table_r_min must be finite and > 0")

    z_numbers = {i + 1: _atomic_number_from_symbol(sym) for i, sym in enumerate(species_list)}
    charge_by_type: Optional[dict[int, float]] = None
    coul_mode: Optional[str] = None
    gewald_value: Optional[float] = None
    table_kspace_keyword: Optional[str] = None
    base_style = str(style_info["style"])
    buck_style = str(style_info.get("buck_style", base_style))
    coul_style = style_info.get("coul_style", None)
    is_hybrid = bool(style_info.get("hybrid", False))
    if buck_style == "buck" and coul_style is None:
        coul_mode = None
    elif coul_style == "coul/cut":
        coul_mode = "cut"
        charge_by_type = _fixed_type_charges(species_list, charges, units_style=units_style)
    elif coul_style == "coul/long":
        coul_mode = "long"
        charge_by_type = _fixed_type_charges(species_list, charges, units_style=units_style)
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
    elif coul_style == "coul/msm" or buck_style == "buck/coul/msm":
        raise ValueError(
            "tabulated buck/coul/msm is not supported: vitriflow only tabulates Coulombic Buckingham "
            "for Ewald/PPPM-compatible real-space splitting"
        )
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported Buckingham/Coulomb composition: {base_style} ({buck_style}, {coul_style})"
        )

    special_lj = (0.0, 0.0, 0.0)
    special_coul = (0.0, 0.0, 0.0)
    if coul_mode in {"cut", "long"}:
        special_lj, special_coul = _validated_equal_special_bonds_weights(base_cmds)
    if coul_mode == "long":
        if has_bonded_topology is None:
            raise ValueError(
                "buck/coul/long table conversion requires an explicit bonded-topology "
                "determination; pass has_bonded_topology=True or False"
            )
        if bool(has_bonded_topology) and any(
            not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-15)
            for value in special_lj
        ):
            raise ValueError(
                "buck/coul/long with bonded topology cannot be represented by one combined "
                "pair table unless all special_lj and special_coul weights are 1.0; otherwise "
                "the KSpace (1-factor_coul)/r bonded correction is missing"
            )

    other_clean: list[str] = []
    has_kspace_style = False
    kspace_style_count = 0
    for ln in other:
        s = str(ln).strip()
        if not s:
            continue
        if s.startswith("kspace_style"):
            has_kspace_style = True
            kspace_style_count += 1
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
    if coul_mode == "long" and kspace_style_count != 1:
        raise ValueError(
            "Buckingham coul/long table conversion requires exactly one effective "
            f"kspace_style command, found {kspace_style_count}"
        )
    if coul_mode != "long" and has_kspace_style:
        raise ValueError(
            "Buckingham conversion without coul/long cannot retain a kspace_style command "
            "after conversion to a non-KSpace pair table"
        )

    common_long_cutoff: Optional[float] = None
    if coul_mode == "long":
        common_long_cutoff = float(style_info["coul_cutoff"])
        if not (math.isfinite(common_long_cutoff) and common_long_cutoff > 0.0):
            raise ValueError("tabulated buck/coul/long requires one finite positive common cutoff")
        # LAMMPS KSpace-compatible pair tables require every pair-table outer
        # cutoff to be identical.  A shorter Buckingham cutoff is not safe
        # either: it would put a hard energy/force discontinuity inside the
        # table, where one value at one strictly increasing radius cannot
        # represent both one-sided limits.  Linear/spline preprocessing would
        # smear that discontinuity across a finite bin.  Require one exact
        # component cutoff instead of silently changing the Hamiltonian.
        mismatched = [
            tuple(int(x) for x in entry["pair"])
            for entry in pair_specs
            if not math.isclose(
                float(entry["buck_cutoff"]),
                common_long_cutoff,
                rel_tol=0.0,
                abs_tol=1.0e-14 * common_long_cutoff,
            )
        ]
        mismatched_morse = [
            tuple(int(x) for x in entry["pair"])
            for entry in pair_specs
            for term in entry.get("morse_terms", [])
            if not math.isclose(
                float(term["cutoff"]),
                common_long_cutoff,
                rel_tol=0.0,
                abs_tol=1.0e-14 * common_long_cutoff,
            )
        ]
        mismatched = sorted(set(mismatched + mismatched_morse))
        if mismatched:
            raise ValueError(
                "tabulated buck/coul/long cannot preserve the KSpace split: pair-specific "
                "Buckingham and Morse cutoffs must exactly match the common Coulomb/KSpace cutoff "
                f"{common_long_cutoff:g}; mismatched pairs {mismatched} would create an "
                "unrepresentable internal hard cutoff"
            )
    elif coul_mode == "cut":
        mismatched = [
            tuple(int(x) for x in entry["pair"])
            for entry in pair_specs
            if entry.get("coul_cutoff", None) is None
            or not math.isclose(
                float(entry["buck_cutoff"]),
                float(entry["coul_cutoff"]),
                rel_tol=0.0,
                abs_tol=1.0e-14
                * max(float(entry["buck_cutoff"]), float(entry["coul_cutoff"])),
            )
        ]
        if mismatched:
            raise ValueError(
                "tabulated buck/coul/cut requires each Buckingham cutoff to exactly "
                "match its Coulomb cutoff; mismatched pairs "
                f"{mismatched} would create an unrepresentable internal hard cutoff"
            )

    # A single table cannot exactly represent an internal hard cutoff of one
    # additive component.  Morse is optional pair-by-pair, but wherever it is
    # active its cutoff must coincide with the full pair cutoff.
    mismatched_hybrid_components: list[tuple[int, int]] = []
    for entry in pair_specs:
        component_cutoffs = [float(entry["buck_cutoff"])]
        if entry.get("coul_cutoff", None) is not None:
            component_cutoffs.append(float(entry["coul_cutoff"]))
        component_cutoffs.extend(
            float(term["cutoff"]) for term in entry.get("morse_terms", [])
        )
        if component_cutoffs and any(
            not math.isclose(
                value,
                component_cutoffs[0],
                rel_tol=0.0,
                abs_tol=1.0e-14 * max(value, component_cutoffs[0]),
            )
            for value in component_cutoffs[1:]
        ):
            mismatched_hybrid_components.append(
                tuple(int(x) for x in entry["pair"])
            )
    if mismatched_hybrid_components:
        raise ValueError(
            "tabulated additive hybrid/overlay conversion requires coincident "
            "Buckingham, Morse, and Coulomb cutoffs for each active pair; mismatched "
            f"pairs {sorted(set(mismatched_hybrid_components))} would create an "
            "unrepresentable internal hard cutoff"
        )

    spec_pairs: list[dict[str, Any]] = []
    pair_lines: list[str] = []
    for entry in pair_specs:
        i, j = (int(entry["pair"][0]), int(entry["pair"][1]))
        buck_cut = float(entry["buck_cutoff"])
        coul_cut = entry.get("coul_cutoff", None)
        morse_terms = [dict(term) for term in entry.get("morse_terms", [])]
        component_cutoffs = [float(buck_cut)] + [
            float(term["cutoff"]) for term in morse_terms
        ]
        pair_cut = max(component_cutoffs + [float(r_out)])
        if coul_mode == "long":
            assert common_long_cutoff is not None
            if float(r_out) >= common_long_cutoff:
                raise ValueError(
                    "Buckingham+ZBL requested r_out must lie strictly below the common "
                    f"buck/coul/long KSpace cutoff ({r_out:g} >= {common_long_cutoff:g})"
                )
            pair_cut = float(common_long_cutoff)
        elif coul_mode == "cut" and coul_cut is not None:
            pair_cut = max(float(pair_cut), float(coul_cut))
        if float(r_out) >= pair_cut:
            raise ValueError(
                "Buckingham+ZBL requested r_out must lie strictly below every "
                f"component cutoff ({r_out:g} >= {pair_cut:g}) for pair ({i},{j})"
            )
        if rlo >= pair_cut:
            raise ValueError(
                f"table_r_min ({rlo:g}) must be smaller than the pair cutoff ({pair_cut:g}) for pair ({i},{j})"
            )
        section = f"P{i}_{j}"
        q_i = None if charge_by_type is None else float(charge_by_type[i])
        q_j = None if charge_by_type is None else float(charge_by_type[j])
        pair_spec: dict[str, Any] = {
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
            "requested_r_in": float(r_in),
            "requested_r_out": float(r_out),
            "shift_buck": bool(shift_enabled),
            "morse_terms": morse_terms,
            "shift_morse": bool(shift_enabled),
            "coul_mode": coul_mode,
            "coul_cutoff": (None if coul_cut is None else float(coul_cut)),
            "q_i": q_i,
            "q_j": q_j,
            "gewald": gewald_value,
        }
        resolved_in, resolved_out, join_energy, resolution = _resolve_repulsive_core_join(
            pair_spec,
            units_style=units_style,
            r_min=rlo,
        )
        pair_spec["r_in"] = resolved_in
        pair_spec["r_out"] = resolved_out
        pair_spec["join_energy"] = join_energy
        pair_spec["join_energy_component"] = "buckingham"
        pair_spec["join_resolution"] = resolution
        pair_spec["validation"] = _validate_repulsive_core_pair(
            pair_spec,
            units_style=units_style,
            r_min=rlo,
        )
        spec_pairs.append(pair_spec)
        # Preserve the parsed model cutoff at round-trip double precision.
        # Coarser formatting can silently move an otherwise valid non-integer
        # literature cutoff even though the table metadata retains its exact
        # value.
        pair_lines.append(
            f"pair_coeff {i} {j} {fname} {section} {pair_cut:.17g}"
        )

    has_morse = any(bool(pair.get("morse_terms")) for pair in spec_pairs)
    metadata = {
        "version": 10,
        "kind": "additive_hybrid_buckingham_zbl_table" if is_hybrid else "buckingham_zbl_table",
        "short_range_regularization": "c2_force_blended_buckingham_to_zbl",
        "regularized_components": ["buckingham"],
        "preserved_components": (
            (["morse"] if has_morse else [])
            + (
                ["full_coulomb_via_real_plus_kspace"]
                if coul_mode == "long"
                else (["direct_coulomb"] if coul_mode == "cut" else [])
            )
        ),
        "morse_policy": (
            "preserved_unchanged_at_all_r" if has_morse else "not_present"
        ),
        "join_selection_hamiltonian": "buckingham_plus_unchanged_morse_plus_full_coulomb",
        "repulsion_validation_hamiltonian": "regularized_buckingham_plus_unchanged_morse_plus_full_coulomb",
        "ewald_split_invariant_regularization": True,
        "filename": fname,
        "points": int(n),
        "r_min": float(rlo),
        "units": str(units_style).strip() or "metal",
        "base_style": str(style_info["style"]),
        "buck_style": buck_style,
        "coul_style": style_info.get("coul_style", None),
        "coul_cutoff": style_info.get("coul_cutoff", None),
        "coul_mode": coul_mode,
        "gewald": gewald_value,
        "kspace_keyword": table_kspace_keyword,
        "common_kspace_cutoff": common_long_cutoff,
        "special_lj": list(special_lj),
        "special_coul": list(special_coul),
        "has_bonded_topology": (
            None if has_bonded_topology is None else bool(has_bonded_topology)
        ),
        "force_mode": "analytic",
        "table_style": interpolation_style,
        "include_fprime": True,
        "source_hybrid_components": list(style_info.get("hybrid_components", [])),
        "source_pair_style": next(
            (
                str(line).split("#", 1)[0].strip()
                for line in base_cmds
                if str(line).split("#", 1)[0].strip().startswith("pair_style ")
            ),
            None,
        ),
    }
    metadata["pairs"] = spec_pairs

    lines = _tabulated_core_metadata_lines(metadata)
    if coul_mode in {None, "cut"}:
        lines.append(f"pair_style table {interpolation_style} {n}")
    else:
        lines.append(
            f"pair_style table {interpolation_style} {n} "
            f"{str(table_kspace_keyword)}"
        )
    lines.extend(pair_lines)
    lines.extend(other_clean)
    if coul_mode == "long":
        lines.append(f"kspace_modify gewald {_lammps_double_token(gewald_value)}")
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


def _morse_energy_derivatives(
    r: np.ndarray,
    *,
    D0: float,
    alpha: float,
    r0: float,
    cutoff: float,
    shift: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LAMMPS Morse energy and its first two radial derivatives."""

    rr_all = np.asarray(r, dtype=float)
    U = np.zeros_like(rr_all)
    dU = np.zeros_like(rr_all)
    d2U = np.zeros_like(rr_all)
    mask = rr_all < float(cutoff)
    if not np.any(mask):
        return U, dU, d2U
    rr = rr_all[mask]
    x = np.exp(-float(alpha) * (rr - float(r0)))
    U_m = float(D0) * (x * x - 2.0 * x)
    dU_m = 2.0 * float(D0) * float(alpha) * (x - x * x)
    d2U_m = (
        2.0
        * float(D0)
        * float(alpha)
        * float(alpha)
        * (2.0 * x * x - x)
    )
    if bool(shift):
        x_cut = math.exp(-float(alpha) * (float(cutoff) - float(r0)))
        U_m = U_m - float(D0) * (x_cut * x_cut - 2.0 * x_cut)
    U[mask] = U_m
    dU[mask] = dU_m
    d2U[mask] = d2U_m
    return U, dU, d2U


def _zbl_base_energy_derivatives(
    r: np.ndarray,
    *,
    z_i: int,
    z_j: int,
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    # Z_i and Z_j are counts of elementary charges, not values in the LAMMPS
    # unit style's native charge unit.
    pref = zbl_coulomb_prefactor(units_style) * float(z_i) * float(z_j)
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


def _quintic_partition_derivatives(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable decreasing P5 partition and derivatives with respect to ``x``.

    The expanded polynomial ``1 - 10*x**3 + 15*x**4 - 6*x**5`` suffers
    catastrophic cancellation near ``x=1``; its expanded derivative can even
    acquire the wrong sign there.  These exactly factorized forms preserve the
    non-negative partition and non-positive slope throughout the binary64
    transition interval.
    """

    xx = np.asarray(x, dtype=float)
    one_minus = 1.0 - xx
    weight = one_minus**3 * (1.0 + 3.0 * xx + 6.0 * xx**2)
    first = -30.0 * xx**2 * one_minus**2
    second = -60.0 * xx * one_minus * (1.0 - 2.0 * xx)
    return weight, first, second


def _regularized_buckingham_zbl_energy_derivatives(
    r: np.ndarray,
    *,
    A: float,
    rho: float,
    C: float,
    cutoff: float,
    shift: bool,
    z_i: int,
    z_j: int,
    units_style: str,
    r_in: float,
    r_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """C2 replacement of the divergent Buckingham core by ZBL.

    Adding ZBL to Buckingham is not a regularization when ``C > 0`` because
    ``-C/r^6`` dominates the ``+const/r`` ZBL repulsion as ``r -> 0``.  This
    routine instead uses pure ZBL for ``r <= r_in``, pure Buckingham for
    ``r >= r_out``, and a quintic partition of unity between them.  The
    switching function and its first two derivatives vanish at both joins, so
    energy, force, and force derivative are continuous while the short-range
    limit is positive infinity.
    """

    rr = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(rr)) or np.any(rr <= 0.0):
        raise ValueError("Buckingham-ZBL regularization requires finite r > 0")
    inner = float(r_in)
    outer = float(r_out)
    if not (math.isfinite(inner) and math.isfinite(outer) and 0.0 < inner < outer):
        raise ValueError(f"Invalid Buckingham-ZBL join radii: r_in={r_in}, r_out={r_out}")

    U_b, dU_b, d2U_b = _buckingham_energy_derivatives(
        rr,
        A=float(A),
        rho=float(rho),
        C=float(C),
        cutoff=float(cutoff),
        shift=bool(shift),
    )
    U_z, dU_z, d2U_z = _zbl_base_energy_derivatives(
        rr,
        z_i=int(z_i),
        z_j=int(z_j),
        units_style=units_style,
    )

    U = np.array(U_b, copy=True)
    dU = np.array(dU_b, copy=True)
    d2U = np.array(d2U_b, copy=True)

    low = rr <= inner
    U[low] = U_z[low]
    dU[low] = dU_z[low]
    d2U[low] = d2U_z[low]

    mid = (rr > inner) & (rr < outer)
    if np.any(mid):
        delta = outer - inner
        x = (rr[mid] - inner) / delta
        w, wp_x, wpp_x = _quintic_partition_derivatives(x)
        wp = wp_x / delta
        wpp = wpp_x / (delta * delta)
        delta_u = U_z[mid] - U_b[mid]
        delta_du = dU_z[mid] - dU_b[mid]
        U[mid] = U_b[mid] + w * delta_u
        dU[mid] = dU_b[mid] + w * delta_du + wp * delta_u
        d2U[mid] = (
            d2U_b[mid]
            + w * (d2U_z[mid] - d2U_b[mid])
            + 2.0 * wp * delta_du
            + wpp * delta_u
        )

    return U, dU, d2U


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


def _pair_noncoulomb_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buckingham plus every explicitly parsed additive Morse contribution."""

    rr = np.asarray(r, dtype=float)
    u, du, d2u = _buckingham_energy_derivatives(
        rr,
        A=float(pair["A"]),
        rho=float(pair["rho"]),
        C=float(pair["C"]),
        cutoff=float(pair["buck_cutoff"]),
        shift=bool(pair.get("shift_buck", False)),
    )
    um, dum, d2um = _pair_morse_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    return u + um, du + dum, d2u + d2um


def _pair_morse_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parsed Morse overlays unchanged.

    ``units_style`` is accepted for a uniform pair-component interface; Morse
    coefficients are already in native LAMMPS energy/distance units.
    """

    del units_style
    rr = np.asarray(r, dtype=float)
    u = np.zeros_like(rr)
    du = np.zeros_like(rr)
    d2u = np.zeros_like(rr)
    for term in pair.get("morse_terms", []):
        um, dum, d2um = _morse_energy_derivatives(
            rr,
            D0=float(term["D0"]),
            alpha=float(term["alpha"]),
            r0=float(term["r0"]),
            cutoff=float(term["cutoff"]),
            shift=bool(pair.get("shift_morse", False)),
        )
        u += um
        du += dum
        d2u += d2um
    return u, du, d2u


def _pair_coulomb_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return either full or runtime-real-space Coulomb derivatives.

    ``full`` is :math:`k q_i q_j/r` and is independent of the Ewald splitting
    parameter.  ``runtime`` is the term consumed by the pair style: it is the
    same direct Coulomb term for ``coul/cut`` and
    :math:`k q_i q_j\,\mathrm{erfc}(G r)/r` for ``coul/long``.  In the latter
    case KSpace supplies the complementary ``erf`` contribution.
    """

    rr = np.asarray(r, dtype=float)
    u = np.zeros_like(rr)
    du = np.zeros_like(rr)
    d2u = np.zeros_like(rr)
    mode = pair.get("coul_mode", None)
    if mode not in {"cut", "long"}:
        return u, du, d2u
    rep = str(representation).strip().lower()
    if rep not in {"full", "runtime"}:
        raise ValueError(f"Unknown Coulomb representation {representation!r}")
    cutoff = float(pair.get("coul_cutoff", pair["pair_cutoff"]))
    mask = rr < cutoff
    if not np.any(mask):
        return u, du, d2u
    q_i = float(pair.get("q_i", 0.0) or 0.0)
    q_j = float(pair.get("q_j", 0.0) or 0.0)
    pref = _coulomb_prefactor_energy_distance(units_style) * q_i * q_j
    if pref == 0.0:
        return u, du, d2u
    rm = rr[mask]
    if rep == "full" or mode == "cut":
        u[mask] = pref / rm
        du[mask] = -pref / (rm * rm)
        d2u[mask] = 2.0 * pref / (rm * rm * rm)
        return u, du, d2u

    g = float(pair.get("gewald", 0.0) or 0.0)
    if not (math.isfinite(g) and g > 0.0):
        raise ValueError("tabulated coul/long contribution requires a finite G-ewald > 0")
    gr = g * rm
    exp_vals = np.exp(-(gr * gr))
    erfc_vals = np.asarray([math.erfc(float(x)) for x in gr], dtype=float)
    a = 2.0 * g / math.sqrt(math.pi)
    u[mask] = pref * erfc_vals / rm
    du[mask] = pref * (-a * exp_vals / rm - erfc_vals / (rm * rm))
    d2u[mask] = pref * (
        2.0 * a * g * g * exp_vals
        + 2.0 * a * exp_vals / (rm * rm)
        + 2.0 * erfc_vals / (rm * rm * rm)
    )
    return u, du, d2u


def _pair_base_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unregularized source pair-style curve (non-Coulomb + real Coulomb)."""

    rr = np.asarray(r, dtype=float)
    u, du, d2u = _pair_noncoulomb_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    uc, duc, d2uc = _pair_coulomb_energy_derivatives(
        rr,
        pair=pair,
        units_style=units_style,
        representation="runtime",
    )
    return u + uc, du + duc, d2u + d2uc


def _pair_total_target_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G-independent physical pair target used to resolve and validate joins."""

    rr = np.asarray(r, dtype=float)
    u, du, d2u = _pair_noncoulomb_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    uc, duc, d2uc = _pair_coulomb_energy_derivatives(
        rr,
        pair=pair,
        units_style=units_style,
        representation="full",
    )
    return u + uc, du + duc, d2u + d2uc


def _pair_base_energy_derivative(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning the base energy and first derivative."""

    u, du, _ = _pair_base_energy_derivatives(r, pair=pair, units_style=units_style)
    return u, du


def _resolve_repulsive_core_join(
    pair: Mapping[str, Any],
    *,
    units_style: str,
    r_min: float,
) -> tuple[float, float, float, dict[str, Any]]:
    """Resolve a join on a repulsive branch of the intended total pair model.

    For ``coul/long`` this deliberately uses the full :math:`q_iq_j/r`
    interaction rather than its G-dependent real-space partition.  The join
    is therefore invariant to the numerical Ewald split.  The complete
    regularized total is independently sampled after construction; a join
    whose unchanged Morse/Coulomb terms defeat the Buckingham/ZBL core is
    rejected.
    """

    lo = max(float(r_min), np.finfo(float).tiny)
    requested_inner = float(pair["requested_r_in"])
    requested_outer = float(pair["requested_r_out"])
    component_cutoffs = [float(pair["buck_cutoff"])]
    component_cutoffs.extend(
        float(term["cutoff"]) for term in pair.get("morse_terms", [])
    )
    if pair.get("coul_mode", None) in {"cut", "long"}:
        component_cutoffs.append(float(pair.get("coul_cutoff", pair["pair_cutoff"])))
    # Never blend across a component cutoff: buck/coul/cut and the tabulated
    # real-space terms are not force-continuous there.  A splice that straddled
    # one could not satisfy the advertised C2 contract.
    branch_limit = min(component_cutoffs)
    hi = branch_limit * (1.0 - 1.0e-10)
    if not (lo < requested_inner < requested_outer <= float(pair["pair_cutoff"])):
        raise ValueError(
            f"pair {pair.get('section', '?')} has invalid requested core interval "
            f"({requested_inner:g}, {requested_outer:g})"
        )

    grid = np.linspace(lo, hi, 65537, dtype=float)
    _u, du, _d2u = _pair_total_target_energy_derivatives(
        grid, pair=pair, units_style=units_style
    )
    force = -du
    finite = np.isfinite(force)
    positive = finite & (force > 0.0)
    if not np.any(positive):
        raise ValueError(
            f"pair {pair.get('section', '?')} has no resolved repulsive base branch "
            f"at or below requested r_out={requested_outer:g}; refusing an unsafe core table"
        )

    # Prefer the requested outer join when it lies in a repulsive connected
    # component.  A common Buckingham catastrophe has not yet exited its
    # unphysical attractive branch at the requested radius; in that case use
    # the first physical repulsive component's force maximum.
    requested_is_inside_branch = requested_outer < branch_limit
    i_requested = int(np.searchsorted(grid, min(requested_outer, hi), side="left"))
    i_requested = min(max(i_requested, 0), len(grid) - 1)
    if requested_is_inside_branch and positive[i_requested]:
        i_hi = i_requested
    elif (not requested_is_inside_branch) and positive[-1]:
        i_hi = len(grid) - 1
    else:
        first = int(np.flatnonzero(positive)[0])
        first_stop = first
        while first_stop + 1 < len(grid) and positive[first_stop + 1]:
            first_stop += 1
        component = np.arange(first, first_stop + 1, dtype=int)
        i_hi = int(component[np.argmax(force[component])])
    start = i_hi
    while start > 0 and positive[start - 1]:
        start -= 1
    stop = i_hi
    while stop + 1 < len(grid) and positive[stop + 1]:
        stop += 1
    if requested_is_inside_branch and i_hi == i_requested and positive[i_requested]:
        r_outer = requested_outer
    elif i_hi == len(grid) - 1:
        r_outer = float(grid[-1])
    else:
        r_outer = float(grid[i_hi])

    branch_start = float(grid[start])
    # Keep the requested width where possible, but never place the transition
    # on the attractive side of the original total pair model.
    r_inner = max(requested_inner, branch_start)
    min_width = 1024.0 * np.finfo(float).eps * max(1.0, abs(r_outer))
    if not (r_inner + min_width < r_outer):
        # Move to the middle of the known-positive component if a requested
        # endpoint left no numerically meaningful transition width.
        r_inner = 0.5 * (branch_start + r_outer)
    if not (lo < r_inner < r_outer):
        raise ValueError(
            f"pair {pair.get('section', '?')} has no finite-width repulsive transition interval"
        )

    u_outer_total, du_outer_total, _ = _pair_total_target_energy_derivatives(
        np.asarray([r_outer]), pair=pair, units_style=units_style
    )
    f_outer = float(-du_outer_total[0])
    if not (math.isfinite(f_outer) and f_outer > 0.0):
        raise ValueError(f"pair {pair.get('section', '?')} resolved a non-repulsive outer join")
    u_outer_buck, _du_outer_buck, _ = _buckingham_energy_derivatives(
        np.asarray([r_outer]),
        A=float(pair["A"]),
        rho=float(pair["rho"]),
        C=float(pair["C"]),
        cutoff=float(pair["buck_cutoff"]),
        shift=bool(pair.get("shift_buck", False)),
    )
    transition_mask = (grid >= r_inner) & (grid <= r_outer)
    if not np.any(transition_mask):
        raise ValueError(
            f"pair {pair.get('section', '?')} has no numerically resolved "
            "repulsive transition samples; increase the core interval width"
        )
    details = {
        "probe_points": int(grid.size),
        "total_branch_start": branch_start,
        "total_branch_stop": float(grid[stop]),
        "component_cutoff_limit": float(branch_limit),
        "original_total_force_min_transition": float(
            np.min(force[transition_mask])
        ),
        "original_total_force_at_outer": f_outer,
        "original_total_energy_at_outer": float(u_outer_total[0]),
        "join_selection_coulomb": "full_qiqj_over_r",
        "join_selection_ewald_split_invariant": True,
    }
    return float(r_inner), float(r_outer), float(u_outer_buck[0]), details


def _regularized_buckingham_component_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace only Buckingham's unbounded core by a C2 ZBL force splice."""

    _validate_regularized_buckingham_join_metadata(pair)

    rr = np.asarray(r, dtype=float)
    u_base, du_base, d2_base = _buckingham_energy_derivatives(
        rr,
        A=float(pair["A"]),
        rho=float(pair["rho"]),
        C=float(pair["C"]),
        cutoff=float(pair["buck_cutoff"]),
        shift=bool(pair.get("shift_buck", False)),
    )
    rin = float(pair["r_in"])
    rout = float(pair["r_out"])
    uout = float(pair["join_energy"])
    uz, duz, d2uz = _zbl_base_energy_derivatives(
        rr, z_i=int(pair["z_i"]), z_j=int(pair["z_j"]), units_style=units_style
    )
    u = np.array(u_base, copy=True)
    du = np.array(du_base, copy=True)
    d2 = np.zeros_like(rr)

    # Fixed Gauss-Legendre quadrature makes U the integral of the explicit
    # Buckingham/ZBL force blend; dU and d2U remain analytic.  The complete
    # Buckingham-core + unchanged Morse + full-Coulomb force is separately
    # required to be repulsive below.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    def _mid_integral(lower: np.ndarray) -> np.ndarray:
        x = 0.5 * (rout - lower[:, None]) * nodes[None, :] + 0.5 * (rout + lower[:, None])
        uz_x, duz_x, _ = _zbl_base_energy_derivatives(
            x.reshape(-1), z_i=int(pair["z_i"]), z_j=int(pair["z_j"]), units_style=units_style
        )
        fz_x = -duz_x.reshape(x.shape)
        t = (x - rin) / (rout - rin)
        w, _wp_t, _wpp_t = _quintic_partition_derivatives(t)
        _ub_x, dub_x, _d2ub_x = _buckingham_energy_derivatives(
            x.reshape(-1),
            A=float(pair["A"]),
            rho=float(pair["rho"]),
            C=float(pair["C"]),
            cutoff=float(pair["buck_cutoff"]),
            shift=bool(pair.get("shift_buck", False)),
        )
        fb_x = -dub_x.reshape(x.shape)
        f = w * fz_x + (1.0 - w) * fb_x
        return 0.5 * (rout - lower) * np.sum(weights[None, :] * f, axis=1)

    mid = (rr > rin) & (rr < rout)
    if np.any(mid):
        x = (rr[mid] - rin) / (rout - rin)
        w, wp_x, _wpp_x = _quintic_partition_derivatives(x)
        wp = wp_x / (rout - rin)
        fz = -duz[mid]
        fpz = -d2uz[mid]
        fb = -du_base[mid]
        fpb = -d2_base[mid]
        force = w * fz + (1.0 - w) * fb
        force_p = w * fpz + (1.0 - w) * fpb + wp * (fz - fb)
        u[mid] = uout + _mid_integral(rr[mid])
        du[mid] = -force
        d2[mid] = -force_p

    low = rr <= rin
    if np.any(low):
        irin = float(_mid_integral(np.asarray([rin]))[0])
        uzrin = float(_zbl_base_energy_derivatives(
            np.asarray([rin]), z_i=int(pair["z_i"]), z_j=int(pair["z_j"]), units_style=units_style
        )[0][0])
        shift = uout + irin - uzrin
        u[low] = uz[low] + shift
        du[low] = duz[low]
        d2[low] = d2uz[low]

    high = rr >= rout
    if np.any(high):
        d2[high] = d2_base[high]
    return u, du, d2


def _validate_regularized_buckingham_join_metadata(
    pair: Mapping[str, Any],
) -> None:
    """Authenticate the meaning of the stored core-splice energy anchor.

    Metadata versions before v10 used ``join_energy`` for a different
    Hamiltonian.  The v10 evaluator anchors only the Buckingham component at
    ``r_out`` and preserves Morse/Coulomb separately.  Reinterpreting an old
    or hand-edited anchor would introduce an energy discontinuity while the
    force could still look plausible, so validate both its semantic label and
    its independently recomputed value before any analytic evaluation.
    """

    if str(pair.get("join_energy_component", "")).strip().lower() != "buckingham":
        raise ValueError(
            "Tabulated Buckingham autocore requires "
            "join_energy_component='buckingham' (metadata schema v10)"
        )
    try:
        r_out = float(pair["r_out"])
        recorded = float(pair["join_energy"])
        expected = float(
            _buckingham_energy_derivatives(
                np.asarray([r_out], dtype=float),
                A=float(pair["A"]),
                rho=float(pair["rho"]),
                C=float(pair["C"]),
                cutoff=float(pair["buck_cutoff"]),
                shift=bool(pair.get("shift_buck", False)),
            )[0][0]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid Buckingham autocore join metadata") from exc
    if not (math.isfinite(r_out) and r_out > 0.0 and math.isfinite(recorded)):
        raise ValueError("Buckingham autocore join metadata must be finite with r_out > 0")
    atol = 4096.0 * np.finfo(float).eps * max(1.0, abs(expected))
    if not math.isclose(recorded, expected, rel_tol=2.0e-13, abs_tol=atol):
        raise ValueError(
            "Buckingham autocore join_energy is inconsistent with the "
            f"Buckingham component at r_out: {recorded:.16g} != {expected:.16g}"
        )


def _validate_tabulated_core_generation_metadata(spec: Mapping[str, Any]) -> None:
    """Require the only metadata schema understood by analytic regeneration."""

    if type(spec.get("version")) is not int or spec.get("version") != 10:
        raise ValueError(
            "Analytic Buckingham autocore generation requires metadata version 10; "
            "older authenticated table bytes may be copied but must not be regenerated"
        )
    kind = str(spec.get("kind", "")).strip()
    if kind not in {
        "buckingham_zbl_table",
        "additive_hybrid_buckingham_zbl_table",
    }:
        raise ValueError(f"Unsupported Buckingham autocore metadata kind: {kind!r}")
    pairs = spec.get("pairs", None)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Buckingham autocore metadata requires at least one pair")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("Buckingham autocore pair metadata must be a mapping")
        _validate_regularized_buckingham_join_metadata(pair)


def _repulsive_regularized_total_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularized Buckingham plus unchanged Morse and full Coulomb.

    This is the physical pair target used for safety validation.  For
    ``coul/long`` it contains :math:`q_iq_j/r`, never the G-dependent Ewald
    real-space partition.
    """

    rr = np.asarray(r, dtype=float)
    ub, dub, d2ub = _regularized_buckingham_component_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    um, dum, d2um = _pair_morse_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    uc, duc, d2uc = _pair_coulomb_energy_derivatives(
        rr,
        pair=pair,
        units_style=units_style,
        representation="full",
    )
    return ub + um + uc, dub + dum + duc, d2ub + d2um + d2uc


def _repulsive_regularized_pair_energy_derivatives(
    r: np.ndarray,
    *,
    pair: Mapping[str, Any],
    units_style: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runtime table curve: Buck-core + unchanged Morse + real Coulomb.

    With ``coul/long``, KSpace supplies the complementary ``erf`` term, so the
    executed total is exactly `_repulsive_regularized_total_energy_derivatives`
    and is independent of the chosen Ewald splitting parameter.
    """

    rr = np.asarray(r, dtype=float)
    ub, dub, d2ub = _regularized_buckingham_component_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    um, dum, d2um = _pair_morse_energy_derivatives(
        rr, pair=pair, units_style=units_style
    )
    uc, duc, d2uc = _pair_coulomb_energy_derivatives(
        rr,
        pair=pair,
        units_style=units_style,
        representation="runtime",
    )
    return ub + um + uc, dub + dum + duc, d2ub + d2um + d2uc


def _validate_repulsive_core_pair(
    pair: Mapping[str, Any],
    *,
    units_style: str,
    r_min: float,
) -> dict[str, Any]:
    """Densely assert finiteness, repulsion, monotonicity, and C2 joins."""

    rin = float(pair["r_in"])
    rout = float(pair["r_out"])
    lo = float(r_min)
    # A geometric probe resolves the singular short-range end; a linear probe
    # resolves both joins and the force blend.  Explicit one-sided points make
    # continuity checks insensitive to grid coincidence.
    geometric = np.geomspace(lo, rout, 32769, dtype=float)
    linear = np.linspace(lo, rout, 32769, dtype=float)
    eps_in = max(1.0e-8 * max(1.0, rin), 64.0 * np.spacing(rin))
    eps_out = max(1.0e-8 * max(1.0, rout), 64.0 * np.spacing(rout))
    joins = np.asarray(
        [rin - eps_in, rin, rin + eps_in, rout - eps_out, rout, rout + eps_out],
        dtype=float,
    )
    probe = np.unique(np.concatenate((geometric, linear, joins)))
    probe = probe[(probe > 0.0) & (probe <= float(pair["pair_cutoff"]))]
    u, du, d2u = _repulsive_regularized_total_energy_derivatives(
        probe, pair=pair, units_style=units_style
    )
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(du)) and np.all(np.isfinite(d2u))):
        raise ValueError(f"pair {pair.get('section', '?')} produced non-finite regularized core values")
    core = probe <= rout
    force = -du[core]
    force_tol = 2048.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(force))
    if np.any(force < -force_tol):
        worst = int(np.argmin(force + force_tol))
        raise ValueError(
            f"pair {pair.get('section', '?')} regularized core is not everywhere repulsive: "
            f"force={float(force[worst]):.16g} at r={float(probe[core][worst]):.16g}"
        )
    # Positive force implies U is non-increasing with separation.  Test the
    # evaluated energy as an independent guard against a sign/integration bug.
    order = np.argsort(probe[core])
    ordered_u = u[core][order]
    delta_u = np.diff(ordered_u)
    adjacent_scale = np.maximum(
        1.0, np.maximum(np.abs(ordered_u[:-1]), np.abs(ordered_u[1:]))
    )
    energy_tol = 4096.0 * np.finfo(float).eps * adjacent_scale
    if np.any(delta_u > energy_tol):
        raise ValueError(f"pair {pair.get('section', '?')} regularized core energy is not monotone")

    # Analytic endpoint identities: pure shifted ZBL at r_in and exact base at
    # r_out, including the second derivative of energy (C2 continuity).
    u_join, du_join, d2_join = _repulsive_regularized_total_energy_derivatives(
        np.asarray([rin, rout]), pair=pair, units_style=units_style
    )
    uz, duz, d2z = _zbl_base_energy_derivatives(
        np.asarray([rin]), z_i=int(pair["z_i"]), z_j=int(pair["z_j"]), units_style=units_style
    )
    unchanged_inner_u, unchanged_inner_du, unchanged_inner_d2 = (
        _pair_morse_energy_derivatives(
            np.asarray([rin]), pair=pair, units_style=units_style
        )
    )
    inner_coul_u, inner_coul_du, inner_coul_d2 = _pair_coulomb_energy_derivatives(
        np.asarray([rin]),
        pair=pair,
        units_style=units_style,
        representation="full",
    )
    expected_inner_u = uz + unchanged_inner_u + inner_coul_u
    expected_inner_du = duz + unchanged_inner_du + inner_coul_du
    expected_inner_d2 = d2z + unchanged_inner_d2 + inner_coul_d2
    ub, dub, d2b = _pair_total_target_energy_derivatives(
        np.asarray([rout]), pair=pair, units_style=units_style
    )
    for actual, expected, name in (
        (du_join[0], expected_inner_du[0], "inner total force"),
        (d2_join[0], expected_inner_d2[0], "inner total force derivative"),
        (u_join[1], ub[0], "outer energy"),
        (du_join[1], dub[0], "outer force"),
        (d2_join[1], d2b[0], "outer force derivative"),
    ):
        atol = 4096.0 * np.finfo(float).eps * max(1.0, abs(float(expected)))
        if not math.isclose(float(actual), float(expected), rel_tol=2.0e-12, abs_tol=atol):
            raise ValueError(
                f"pair {pair.get('section', '?')} failed C2 join validation for {name}: "
                f"{float(actual):.16g} != {float(expected):.16g}"
            )
    # The regularized Buckingham component may differ from bare ZBL only by a
    # constant.  Morse and Coulomb are deliberately unchanged.
    inner_shift = float(u_join[0] - expected_inner_u[0])
    return {
        "probe_points": int(probe.size),
        "minimum_total_core_force": float(np.min(force)),
        "minimum_core_force": float(np.min(force)),
        "validated_core_domain": [float(lo), float(rout)],
        "repulsive_on_validated_core_domain": True,
        "buckingham_zbl_energy_shift": inner_shift,
        "inner_zbl_energy_shift": inner_shift,
        "c2_join_validated": True,
        "repulsive_through_r_out": True,
        "repulsion_validation_scope": "regularized_buckingham_plus_unchanged_morse_plus_full_coulomb",
        "coulomb_validation_form": "full_qiqj_over_r",
        "ewald_split_invariant": True,
        "morse_policy": (
            "preserved_unchanged_at_all_r"
            if bool(pair.get("morse_terms"))
            else "not_present"
        ),
    }



def _tabulated_buckingham_section_arrays(
    pair: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray | float]:
    """Tabulated buckingham section."""
    _validate_tabulated_core_generation_metadata(spec)
    n = int(spec["points"])
    rmin = float(spec["r_min"])
    units_style = str(spec.get("units", "metal")).strip() or "metal"
    force_mode = str(spec.get("force_mode", "analytic") or "analytic").strip().lower()
    pair_cut = float(pair["pair_cutoff"])
    if int(n) < 2:
        raise ValueError("Tabulated Buckingham core requires at least 2 points")
    if not (pair_cut > rmin):
        raise ValueError(f"Tabulated Buckingham core requires pair_cutoff > r_min for section {pair.get('section', '?')}")
    r = _lammps_rsq_grid(rmin, pair_cut, n)
    # LAMMPS applies the pair cutoff before evaluating a table at exactly
    # ``pair_cut``.  The last table knot must therefore contain the *left-hand
    # limit*, not the zero value on the excluded side of the hard cutoff.
    # Supplying zero here makes linear table interpolation smear the cutoff
    # discontinuity across the final bin and creates a spurious force spike
    # just inside the physical interaction range.
    evaluation_r = np.array(r, copy=True)
    cutoff_left = float(np.nextafter(pair_cut, rmin))
    if not (rmin < cutoff_left < pair_cut):
        raise ValueError(
            "Tabulated Buckingham core cutoff has no representable interior "
            f"left-limit radius for section {pair.get('section', '?')}"
        )
    evaluation_r[-1] = cutoff_left
    U, dU, d2U = _repulsive_regularized_pair_energy_derivatives(
        evaluation_r, pair=pair, units_style=units_style
    )
    if force_mode == "analytic":
        F = -dU
    elif force_mode == "fd_consistent":
        if r.size < 3:
            F = -dU
        else:
            # LAMMPS checks each supplied table force against finite
            # differences of the supplied table energy.  Differentiate that
            # same energy on the actual (nonuniform-in-r) RSQ knot grid; a
            # spline derivative is not the derivative of LAMMPS's table
            # interpolant and produces false inconsistency warnings.  Clamp
            # only the endpoints to the independently validated analytic
            # one-sided slopes, where a finite-difference stencil cannot see
            # both sides (and the upper knot intentionally stores the
            # left-hand limit at the hard cutoff).
            F = -np.asarray(np.gradient(U, r, edge_order=2), dtype=float)
            F[0] = -float(dU[0])
            F[-1] = -float(dU[-1])
    else:  # pragma: no cover - guarded by preflight table refinement
        raise ValueError(f"Unsupported tabulated Buckingham force_mode: {force_mode}")
    # FPRIME is dF/dr at the table endpoints.  Use the exact one-sided
    # derivative of the physical force, -d2U/dr2, instead of estimating it
    # from the tabulated force.  This remains correct for both force modes and
    # at the upper hard cutoff because ``evaluation_r`` stores its left limit.
    dF = -np.asarray(d2U, dtype=float)
    return {
        "r": np.asarray(r, dtype=float),
        "energy": np.asarray(U, dtype=float),
        "force": np.asarray(F, dtype=float),
        "fprime_lo": float(dF[0]) if dF.size else 0.0,
        "fprime_hi": float(dF[-1]) if dF.size else 0.0,
    }


def tabulated_buckingham_reference_sections(
    spec: Mapping[str, Any],
    *,
    npoints: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate the analytic regularized potential on the pair-write RSQ grid."""

    n = int(npoints)
    if n < 2:
        raise ValueError("Buckingham reference evaluation requires npoints >= 2")
    ref_spec = dict(spec)
    ref_spec["points"] = n
    # Analytic derivatives are the physical reference even when a candidate
    # table elects finite-difference-consistent forces for interpolation.
    ref_spec["force_mode"] = "analytic"
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in list(spec.get("pairs", [])):
        data = _tabulated_buckingham_section_arrays(pair, spec=ref_spec)
        # ``pair_write`` reports the excluded-side value (zero) at the exact
        # cutoff even though the source table's final knot is the interior
        # left limit.  Preserve that runtime convention in the comparison
        # reference while continuing to test all points immediately below it.
        energy = np.asarray(data["energy"], dtype=float).copy()
        force = np.asarray(data["force"], dtype=float).copy()
        energy[-1] = 0.0
        force[-1] = 0.0
        out[str(pair["section"])] = {
            "r": np.asarray(data["r"], dtype=float),
            "energy": energy,
            "force": force,
        }
    if not out:
        raise ValueError("Buckingham reference spec contains no pair sections")
    return out


def tabulated_buckingham_unregularized_reference_sections(
    spec: Mapping[str, Any],
    *,
    npoints: int,
    r_min: Optional[float] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate the parsed full source Hamiltonian above each resolved join.

    The returned grid uses the same RSQ spacing and exact-cutoff zero
    convention as LAMMPS ``pair_write``.  It is intended to audit that a
    generated table retains *all* parsed Buckingham, Morse, and real-space
    Coulomb contributions outside the core splice.
    """

    n = int(npoints)
    if n < 2:
        raise ValueError("Buckingham source reference evaluation requires npoints >= 2")
    units_style = str(spec.get("units", "metal")).strip() or "metal"
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in list(spec.get("pairs", [])):
        lower = float(pair["r_out"])
        if r_min is not None:
            lower = max(lower, float(r_min))
        cutoff = float(pair["pair_cutoff"])
        if not (math.isfinite(lower) and 0.0 < lower < cutoff):
            raise ValueError(
                f"Invalid source-reference interval [{lower:g}, {cutoff:g}] for "
                f"section {pair.get('section', '?')}"
            )
        radius = _lammps_rsq_grid(lower, cutoff, n)
        evaluation_radius = np.array(radius, copy=True)
        evaluation_radius[-1] = np.nextafter(cutoff, lower)
        energy, derivative, _ = _pair_base_energy_derivatives(
            evaluation_radius, pair=pair, units_style=units_style
        )
        force = -np.asarray(derivative, dtype=float)
        energy = np.asarray(energy, dtype=float)
        # pair_write reports the excluded side at exactly the hard cutoff.
        energy[-1] = 0.0
        force[-1] = 0.0
        out[str(pair["section"])] = {
            "r": radius,
            "energy": energy,
            "force": force,
        }
    if not out:
        raise ValueError("Buckingham source reference spec contains no pair sections")
    return out


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
        # LAMMPS compares the token immediately following ``UNITS:``
        # literally with the active unit style.  Punctuation belongs on a
        # separate comment line: ``metal;`` is not the ``metal`` unit tag.
        f"# UNITS: {units_style}",
        "# vitriflow Buckingham regularized real-space potential",
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
        # Shortest-roundtrip binary64 tokens keep the bounds read by LAMMPS
        # identical to the bounds used to evaluate the table above.
        hdr = (
            f"N {n} RSQ {_lammps_double_token(rmin)} "
            f"{_lammps_double_token(pair_cut)}"
        )
        if include_fprime:
            hdr += (
                f" FPRIME {_lammps_double_token(data['fprime_lo'])} "
                f"{_lammps_double_token(data['fprime_hi'])}"
            )
        lines.append(hdr)
        lines.append("")
        for idx in range(n):
            lines.append(
                f"{idx+1:d} {_lammps_double_token(r[idx])} "
                f"{_lammps_double_token(U[idx])} {_lammps_double_token(F[idx])}"
            )
        lines.append("")

    _atomic_write_generated_potential_text(path, "\n".join(lines) + "\n")


def prepare_potential_files(
    pot: PotentialConfig,
    stage_dir: Path,
    potential_lines: Optional[Sequence[str]] = None,
) -> None:
    """Potential files."""
    stage_dir = Path(stage_dir)
    if stage_dir.is_symlink():
        raise ValueError(
            f"Potential staging directory must not be a symbolic link: {stage_dir}"
        )

    # Validate every destination basename and basename-to-source mapping before
    # creating the staging directory.  Callers can therefore probe a malformed
    # configuration without leaving a partially initialized result tree.
    if isinstance(pot, MG2SiNPotentialConfig):
        validated_lammps_localized_filename(
            pot.table_filename,
            field_name="MG2 table_filename",
        )
    if isinstance(pot, LammpsPotentialConfig):
        planned_sources: dict[str, Path] = {}
        for index, raw_source in enumerate(pot.files or []):
            source = Path(raw_source)
            name = validated_lammps_localized_filename(
                source.name,
                field_name=f"potential.files[{index}] basename",
            )
            prior = planned_sources.get(name)
            if prior is not None and prior.resolve(strict=False) != source.resolve(strict=False):
                raise ValueError(
                    "potential.files contains distinct sources with the same "
                    f"localized basename {name!r}: {prior} and {source}"
                )
            planned_sources[name] = source
    if potential_lines is not None:
        planned_core = _parse_tabulated_core_spec(potential_lines)
        if planned_core is not None:
            raw_core_name = planned_core.get("filename", "")
            validated_lammps_localized_filename(
                "buckingham_core.table" if raw_core_name == "" else raw_core_name,
                field_name="tabulated core filename",
            )

    stage_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(pot, MG2SiNPotentialConfig):
        table_name = validated_lammps_localized_filename(
            pot.table_filename,
            field_name="MG2 table_filename",
        )
        out = stage_dir / table_name
        write_mg2_sin_table(out, pot)

    if isinstance(pot, LammpsPotentialConfig):
        # Localization is by basename. Distinct sources sharing a basename
        # would silently overwrite one another; refuse instead so the user
        # renames or re-organises explicitly.
        seen_by_name: dict[str, Path] = {}
        for f in pot.files or []:
            src = Path(f)
            if not src.exists():
                raise FileNotFoundError(f"Potential auxiliary file not found: {src}")
            name = validated_lammps_localized_filename(
                src.name,
                field_name="potential.files basename",
            )
            prior = seen_by_name.get(name)
            if prior is not None:
                try:
                    same = prior.resolve(strict=False) == src.resolve(strict=False)
                except OSError:
                    same = False
                if not same:
                    raise ValueError(
                        f"potential.files contains two distinct sources with basename {name!r}: "
                        f"{prior} and {src}. Localization is by basename, so one would silently "
                        "overwrite the other. Rename one of them."
                    )
            seen_by_name[name] = src
            dst = stage_dir / name
            try:
                same_dst = dst.resolve() == src.resolve()
            except OSError:
                same_dst = False
            if same_dst:
                continue
            _atomic_copy_verified_regular_file(src, dst)

    if potential_lines is not None:
        spec = _parse_tabulated_core_spec(potential_lines)
        if spec is not None:
            raw_out_name = spec.get("filename", "")
            out_name = validated_lammps_localized_filename(
                "buckingham_core.table" if raw_out_name == "" else raw_out_name,
                field_name="tabulated core filename",
            )
            out = stage_dir / out_name
            src = _find_validated_tabulated_core_source(stage_dir, spec)
            if src is not None:
                protected_sha = str(spec.get("sha256", "") or "").strip().lower()
                protected_size_raw = spec.get("size_bytes")
                _atomic_copy_verified_regular_file(
                    src,
                    out,
                    expected_sha256=(protected_sha or None),
                    expected_size_bytes=(
                        None
                        if protected_size_raw is None
                        else int(protected_size_raw)
                    ),
                )
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
        g[m], dgdt, _d2gdt2 = _quintic_partition_derivatives(t)
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

    # Evaluate the published parameterisation in its defining eV/Angstrom
    # system, then convert the independent variable, energy and derivative to
    # the selected LAMMPS unit style.  This keeps one physical potential across
    # all dimensional unit styles.
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

    units = normalize_lammps_units_style(pot.user_units)
    length_factor = length_from_angstrom_factor(units)
    energy_factor = energy_from_ev_factor(units)
    force_factor = energy_factor / length_factor
    r_native = r * float(length_factor)
    rmin_native = rmin * float(length_factor)
    rmax_native = rmax * float(length_factor)

    U_sin = U_sin * float(energy_factor)
    U_sisi = U_sisi * float(energy_factor)
    U_nn = U_nn * float(energy_factor)
    F_sin = F_sin * float(force_factor)
    F_sisi = F_sisi * float(force_factor)
    F_nn = F_nn * float(force_factor)

    def _section_lines(name: str, U: np.ndarray, F: np.ndarray) -> list[str]:
        sec: list[str] = []
        sec.append(str(name))
        sec.append(f"N {n} R {rmin_native:.16g} {rmax_native:.16g}")
        sec.append("")
        for i in range(n):
            sec.append(f"{i+1:d} {r_native[i]:.16g} {U[i]:.16g} {F[i]:.16g}")
        sec.append("")
        return sec

    header = [
        # Keep the unit style as a standalone LAMMPS tag token.
        f"# UNITS: {units}",
        "# MG2 Si-N tabulated potential generated by vitriflow",
        "# columns: index r energy force (force = -dU/dr)",
        "",
    ]

    text_lines: list[str] = []
    text_lines.extend(header)
    text_lines.extend(_section_lines("SiSi", U_sisi, F_sisi))
    text_lines.extend(_section_lines("SiN", U_sin, F_sin))
    text_lines.extend(_section_lines("NN", U_nn, F_nn))

    _atomic_write_generated_potential_text(path, "\n".join(text_lines) + "\n")
