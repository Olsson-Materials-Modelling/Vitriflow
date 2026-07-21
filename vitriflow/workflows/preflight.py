from __future__ import annotations

import ast
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..analysis.datafile import strip_lammps_data_pair_coeff_sections
from ..io.lammps_data_minimal import read_lammps_data_minimal
from ..analysis.stats import window_mean_stderr
from ..config import (
    BarostatConfig,
    MDConfig,
    RunConfig,
    ThermostatConfig,
    KimConfig,
    validated_lammps_localized_filename,
)
from ..lammps_input import StageSpec, render_stage
from ..potential import (
    _lammps_double_token,
    _lammps_rsq_grid,
    _pair_base_energy_derivatives,
    _pair_coulomb_energy_derivatives,
    _parse_tabulated_core_spec,
    _repulsive_regularized_pair_energy_derivatives,
    build_tabulated_buckingham_core_lines,
    inspect_buckingham_core_compatibility,
    kim_init_line,
    kim_interactions_line,
    potential_default_lines,
    potential_init_lines,
    prepare_potential_files,
    tabulated_buckingham_reference_sections,
    update_tabulated_core_metadata_lines,
    write_tabulated_buckingham_core_table,
)
from ..lammps_units import (
    boltzmann_constant_native,
    charge_to_elementary_factor,
    length_from_angstrom_factor,
)
from ..parse import parse_last_thermo_table
from ..runner import Cp2kRunner, LammpsRunner, RunResult
from .stage_runner import run_stage_local
from ..utils import (
    ExternalCommandError,
    ensure_dir,
    scale_steps_for_timestep,
    stable_file_identity,
)
from .autoskin import run_with_neighbor_skin_autotune
from .progress import CondensedProgressLog


class PreflightError(RuntimeError):
    """Preflight error."""

    def __init__(self, message: str, candidates: Optional[list[Any]] = None):
        super().__init__(message)
        self.candidates = candidates or []


def _duration_ps_to_lammps_time(duration_ps: float, units_style: str) -> float:
    """Convert a physical duration in ps to one LAMMPS native time unit."""

    from .quench_rates import lammps_timeunit_ps

    value_ps = float(duration_ps)
    if not (math.isfinite(value_ps) and value_ps > 0.0):
        raise ValueError("duration_ps must be finite and > 0")
    time_unit_ps = lammps_timeunit_ps(str(units_style))
    if time_unit_ps is None:
        raise ValueError(
            f"Cannot convert a physical duration for LAMMPS units style {units_style!r}"
        )
    return value_ps / float(time_unit_ps)


def _screen_cp2k_pressure_samples(
    samples_bar: Sequence[float],
    *,
    target_bar: float,
    tail_window: int,
    max_abs_bar: float,
    tolerance_bar: float,
    require_tolerance: bool,
) -> tuple[bool, dict[str, float]]:
    """Evaluate the pressure portion of a CP2K NPT preflight candidate."""

    values = np.asarray(samples_bar, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False, {
            "P_mean": float("nan"),
            "P_tail_std": float("nan"),
            "P_max_abs": float("nan"),
        }
    n_tail = min(max(1, int(tail_window)), int(values.size))
    tail = values[-n_tail:]
    mean = float(np.mean(tail))
    std = float(np.std(tail))
    max_abs = float(np.max(np.abs(values)))
    ok = bool(
        math.isfinite(mean)
        and math.isfinite(max_abs)
        and max_abs <= float(max_abs_bar)
    )
    if ok and bool(require_tolerance):
        ok = abs(mean - float(target_bar)) <= float(tolerance_bar)
    return ok, {
        "P_mean": mean,
        "P_tail_std": std,
        "P_max_abs": max_abs,
    }

@dataclass(frozen=True)
class CoreRepulsionResult:
    enabled: bool
    applied: bool
    style: str
    base_pair_style: Optional[str]
    r_inner: Optional[float]
    r_outer: Optional[float]
    attempts: int
    success: bool
    note: str
    # ``r_inner``/``r_outer`` are retained as the historical global
    # calibration interval.  A safe Buckingham table may resolve a different
    # interval for each pair so that the join lies on a repulsive base branch.
    # The additive fields below make that distinction machine-readable without
    # changing existing constructors or result consumers.
    r_inner_r_outer_role: str = "not_applicable"
    requested_r_inner: Optional[float] = None
    requested_r_outer: Optional[float] = None
    join_radii_lammps_units_style: Optional[str] = None
    resolved_pair_joins: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ThermoCandidateResult:
    timestep: float
    ensemble: str
    tdamp: float
    pdamp: Optional[float]
    ok: bool
    score: float
    details: dict[str, Any]


@dataclass(frozen=True)
class PreflightResult:
    selected_ensemble: str
    selected_timestep: float
    selected_tdamp: float
    selected_pdamp: Optional[float]
    # override scripts
    potential_lines: Optional[list[str]]
    core_repulsion: CoreRepulsionResult
    candidates: list[ThermoCandidateResult]


def _clean_dt_candidates(values: Optional[Sequence[Any]]) -> list[float]:
    out: list[float] = []
    if values is None:
        return out
    seen: set[float] = set()
    for x in values:
        try:
            v = float(x)
        except Exception:
            continue
        if v > 0.0 and math.isfinite(v) and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _implicit_dt_fallbacks(dt0: float, *, cp2k: bool = False) -> list[float]:
    vals = [dt0, dt0 / 2.0, dt0 / 4.0] if cp2k else [dt0, dt0 / 2.0, dt0 / 4.0, dt0 / 8.0]
    return _clean_dt_candidates(vals)


def _core_dt_candidates(config: RunConfig) -> tuple[list[float], str]:
    pf = config.autotune.preflight
    core = getattr(config.kim, "core_repulsion", None)
    md0 = float(config.md.timestep)
    pf_explicit = _clean_dt_candidates(getattr(pf, "dt_candidates", None))
    core_explicit = _clean_dt_candidates(getattr(core, "dt_candidates", None) if core is not None else None)

    if pf_explicit and core_explicit:
        inter = [x for x in core_explicit if x in set(pf_explicit)]
        if not inter:
            raise ValueError(
                "Explicit autotune.preflight.dt_candidates and potential.core_repulsion.dt_candidates are disjoint; "
                "core preflight cannot select an undeclared timestep."
            )
        return sorted(inter, reverse=True), "intersection(autotune.preflight.dt_candidates, potential.core_repulsion.dt_candidates)"
    if core_explicit:
        return sorted(core_explicit, reverse=True), "potential.core_repulsion.dt_candidates"
    if pf_explicit:
        return sorted(pf_explicit, reverse=True), "autotune.preflight.dt_candidates"
    if bool(getattr(pf, "allow_implicit_dt_fallback", False)):
        return sorted(_implicit_dt_fallbacks(md0), reverse=True), "implicit_dt_fallback"
    return [md0], "md.timestep"


def _preflight_dt_candidates(config: RunConfig, *, dt_cap: Optional[float] = None, cp2k: bool = False) -> tuple[list[float], str]:
    pf = config.autotune.preflight
    core = getattr(config.kim, "core_repulsion", None)
    md0 = float(config.md.timestep)
    pf_explicit = _clean_dt_candidates(getattr(pf, "dt_candidates", None))
    core_explicit = _clean_dt_candidates(getattr(core, "dt_candidates", None) if (core is not None and not cp2k) else None)

    if pf_explicit:
        vals = pf_explicit
        source = "autotune.preflight.dt_candidates"
    elif core_explicit:
        vals = core_explicit
        source = "potential.core_repulsion.dt_candidates"
    elif bool(getattr(pf, "allow_implicit_dt_fallback", False)):
        vals = _implicit_dt_fallbacks(md0, cp2k=cp2k)
        source = "implicit_dt_fallback"
    else:
        vals = [md0]
        source = "md.timestep"

    vals = _clean_dt_candidates(vals)
    if dt_cap is not None:
        cap = float(dt_cap)
        vals = [x for x in vals if x <= cap + 1.0e-15]
    vals = sorted(vals, reverse=True)
    return vals, source


def _atomic_number(symbol: str) -> int:
    sym = str(symbol).strip()
    try:
        from ase.data import atomic_numbers

        return int(atomic_numbers[sym])
    except Exception:  # pragma: no cover
        _fallback = {"H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Si": 14, "Fe": 26}
        if sym not in _fallback:
            raise KeyError(f"Unknown atomic number for symbol '{sym}'")
        return _fallback[sym]


def _read_nn_median_from_datafile(
    data_path: Path,
    *,
    atom_style: str,
    units_style: str,
) -> float:
    """Nearest-neighbour median in the native LAMMPS length unit."""
    try:
        from ..io.ase_compat import ase_read_lammps_data

        atoms = ase_read_lammps_data(
            data_path,
            atom_style=str(atom_style),
            units=str(units_style),
        )
    except Exception as e:  # pragma: no cover - dependency/parser failure
        raise RuntimeError(
            f"ASE failed to read LAMMPS data for preflight distance estimation: {data_path}"
        ) from e

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    nn = nn[np.isfinite(nn) & (nn > 1.0e-8)]
    if len(nn) == 0:
        raise ValueError(f"Failed to compute nearest-neighbour distances from {data_path}")
    # ASE always exposes geometry in Angstrom; the caller combines this value
    # with native table radii, so convert back exactly once.
    return float(np.median(nn)) * float(length_from_angstrom_factor(units_style))


def _datafile_has_bonded_topology(data_path: Path) -> bool:
    """Determine whether a LAMMPS data header declares bonded topology."""

    topology_kinds = {"bonds", "angles", "dihedrals", "impropers"}
    counts = {kind: 0 for kind in topology_kinds}
    for raw in Path(data_path).read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0].lower() in {
            "masses", "atoms", "velocities", "bonds", "angles", "dihedrals", "impropers"
        } and not tokens[0].replace(".", "", 1).isdigit():
            break
        if len(tokens) >= 2 and tokens[1].lower() in topology_kinds:
            try:
                value = int(tokens[0])
            except ValueError as exc:
                raise ValueError(f"Malformed bonded-topology count in {data_path}: {raw}") from exc
            if value < 0:
                raise ValueError(f"Negative bonded-topology count in {data_path}: {raw}")
            counts[tokens[1].lower()] = value
    return any(value > 0 for value in counts.values())


def _validate_tabulated_coulomb_runtime_charges(
    data_path: Path,
    *,
    atom_style: str,
    species: Sequence[str],
    configured_charges: Optional[Mapping[str, float]],
    units_style: str,
    potential_commands: Sequence[str],
    require_explicit_set_for_present_types: bool = False,
) -> dict[str, Any]:
    """Resolve and prove the fixed charges used by the table and KSpace.

    Table construction folds fixed type charges into the real-space energy.
    The effective runtime state is the charge-style data followed, in command
    order, by any KIM-generated literal ``set type N charge Q`` assignments.
    Optional ``structure.charges`` values are assertions on that state, not a
    second mandatory source of truth.  Per-atom/group/variable assignments and
    charge-equilibration fixes remain unsupported and fail closed.
    """

    style = str(atom_style).strip().lower()
    if style != "charge":
        raise ValueError(
            "tabulated buck/coul/* requires md.atom_style='charge'; atomic-style "
            "input does not carry the runtime charges used to build the table"
        )
    species_list = [str(x) for x in species]
    if not species_list or len(set(species_list)) != len(species_list):
        raise ValueError(
            "tabulated buck/coul/* charge validation requires a non-empty, unique "
            "species ordering"
        )
    configured: Optional[dict[str, float]] = None
    if configured_charges is not None:
        configured = {}
        for symbol in species_list:
            if symbol not in configured_charges:
                raise ValueError(
                    "tabulated buck/coul/* charge audit is missing configured "
                    f"charge for {symbol!r}"
                )
            value = float(configured_charges[symbol])
            if not math.isfinite(value):
                raise ValueError(
                    f"tabulated buck/coul/* configured charge for {symbol!r} is not finite"
                )
            configured[symbol] = value

    # Simulator Models may establish fixed charges with commands emitted by
    # ``kim interactions``.  Accept only the deterministic type-wide literal
    # form.  The last assignment for a type wins, matching LAMMPS command
    # order.  Values in commands are native to the active LAMMPS unit style.
    set_by_type: dict[int, float] = {}
    set_evidence: list[dict[str, Any]] = []
    native_to_e = float(charge_to_elementary_factor(units_style))
    ntypes = len(species_list)
    for command_index, raw in enumerate(potential_commands):
        line = str(raw).split("#", 1)[0].strip()
        toks_raw = line.split()
        toks = [token.lower() for token in toks_raw]
        if not toks:
            continue
        if toks[0] == "set":
            if not (
                len(toks_raw) == 5
                and toks[1] == "type"
                and toks[3] == "charge"
                and re.fullmatch(r"[1-9][0-9]*", toks_raw[2]) is not None
            ):
                raise ValueError(
                    "tabulated buck/coul/* accepts only fixed literal "
                    "'set type <integer> charge <number>' assignments; "
                    f"unsupported charge command: {line}"
                )
            atom_type = int(toks_raw[2])
            if atom_type < 1 or atom_type > ntypes:
                raise ValueError(
                    "tabulated buck/coul/* fixed charge command selects atom type "
                    f"{atom_type}, outside [1, {ntypes}]"
                )
            try:
                native_charge = float(
                    toks_raw[4].replace("D", "E").replace("d", "e")
                )
            except ValueError as exc:
                raise ValueError(
                    "tabulated buck/coul/* fixed type charge must be a finite "
                    f"numeric literal: {line}"
                ) from exc
            charge_e = native_charge * native_to_e
            if not (math.isfinite(native_charge) and math.isfinite(charge_e)):
                raise ValueError(
                    "tabulated buck/coul/* fixed type charge must be finite: "
                    f"{line}"
                )
            set_by_type[atom_type] = charge_e
            set_evidence.append(
                {
                    "command_index": int(command_index),
                    "command": line,
                    "atom_type": int(atom_type),
                    "species": species_list[atom_type - 1],
                    "charge_native": native_charge,
                    "charge_e": charge_e,
                }
            )
        if toks[0] == "fix":
            raise ValueError(
                "tabulated buck/coul/* cannot prove fixed type charges with a "
                f"KIM-generated fix command: {line}"
            )

    # If the file explicitly declares its Atoms style, it must agree with the
    # configured charge style.  With no comment, the configured style controls
    # the standard LAMMPS column interpretation.
    for raw in Path(data_path).read_text(errors="replace").splitlines():
        before_comment = raw.split("#", 1)[0].strip().lower()
        if before_comment == "atoms":
            if "#" in raw:
                declared = raw.split("#", 1)[1].strip().split()
                if declared and declared[0].lower() != "charge":
                    raise ValueError(
                        "tabulated buck/coul/* requires an Atoms # charge section; "
                        f"input declares Atoms # {declared[0]}"
                    )
            break

    atoms = read_lammps_data_minimal(
        Path(data_path),
        atom_style="charge",
        specorder=species_list,
        units_style=str(units_style),
    )
    symbols = [str(x) for x in atoms.get_chemical_symbols()]
    actual = np.asarray(atoms.get_initial_charges(), dtype=float).reshape(-1)
    if actual.size != len(symbols) or actual.size == 0:
        raise ValueError(
            "tabulated buck/coul/* input charge audit found missing/incomplete atom charges"
        )
    if not np.all(np.isfinite(actual)):
        raise ValueError("tabulated buck/coul/* input charges must all be finite")

    counts: dict[str, int] = {symbol: 0 for symbol in species_list}
    input_by_species: dict[str, list[float]] = {
        symbol: [] for symbol in species_list
    }
    for atom_index, (symbol, observed) in enumerate(zip(symbols, actual.tolist()), start=1):
        if symbol not in input_by_species:
            raise ValueError(
                f"input atom {atom_index} maps to unexpected species/type {symbol!r}"
            )
        counts[symbol] += 1
        input_by_species[symbol].append(float(observed))

    def _same_charge(left: float, right: float) -> bool:
        return math.isclose(
            float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-13
        )

    effective: dict[str, float] = {}
    source_by_species: dict[str, str] = {}
    input_ranges: dict[str, Optional[list[float]]] = {}
    overwritten_species: list[str] = []
    for atom_type, symbol in enumerate(species_list, start=1):
        values = input_by_species[symbol]
        if values:
            input_ranges[symbol] = [float(min(values)), float(max(values))]
        else:
            input_ranges[symbol] = None

        if (
            bool(require_explicit_set_for_present_types)
            and values
            and atom_type not in set_by_type
        ):
            raise ValueError(
                "tabulated buck/coul/* requires an explicit fixed KIM "
                f"'set type {atom_type} charge <number>' assignment for present "
                f"species {symbol!r}: the structure was generated without "
                "structure.charges, so its data-file value is not an independent "
                "charge authority"
            )

        if atom_type in set_by_type:
            resolved = float(set_by_type[atom_type])
            source = "kim_fixed_set_type_charge"
            if any(not _same_charge(value, resolved) for value in values):
                overwritten_species.append(symbol)
        elif values:
            resolved = float(values[0])
            if any(not _same_charge(value, resolved) for value in values[1:]):
                raise ValueError(
                    "tabulated buck/coul/* input charges vary within uncovered "
                    f"species/type {symbol!r}; a fixed type charge cannot be proven"
                )
            source = "input.data Atoms # charge"
        elif configured is not None:
            # An interaction type absent from this structure cannot be audited
            # against atoms, but an explicit complete configuration still
            # supplies the table coefficient should that type later be used.
            resolved = float(configured[symbol])
            source = "structure.charges (type absent from input)"
        else:
            raise ValueError(
                "tabulated buck/coul/* cannot resolve a fixed charge for absent "
                f"species/type {symbol!r}"
            )

        if configured is not None and not _same_charge(resolved, configured[symbol]):
            raise ValueError(
                "tabulated buck/coul/* effective runtime charge conflicts with "
                f"structure.charges for {symbol!r}: runtime={resolved:.16g} e, "
                f"configured={configured[symbol]:.16g} e"
            )
        effective[symbol] = resolved
        source_by_species[symbol] = source

    return {
        "passed": True,
        "source": "input.data followed by fixed KIM type-charge assignments",
        "comparison_units": "elementary_charge",
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance_e": 1.0e-13,
        "n_atoms": int(actual.size),
        "counts_by_species": counts,
        "configured_charges_e": configured,
        "effective_charges_e": effective,
        "source_by_species": source_by_species,
        "input_charge_range_e_by_species": input_ranges,
        "fixed_set_type_charges_e": {
            species_list[atom_type - 1]: float(value)
            for atom_type, value in sorted(set_by_type.items())
        },
        "fixed_set_commands": set_evidence,
        "input_charges_overwritten_for_species": overwritten_species,
        "explicit_set_required_for_present_types": bool(
            require_explicit_set_for_present_types
        ),
        "variable_charge_commands": False,
    }


def _extract_kim_interactions_block(log_path: Path) -> list[str]:
    """Extract one complete, closed KIM interaction command block.

    Silently dropping an unfamiliar material command can change the
    Hamiltonian before the Buckingham parser ever sees it.  Real KIM Simulator
    Models also emit scalar ``variable``/``if`` control commands and LAMMPS
    status text inside the marked block.  Resolve that small, deterministic
    control language, retain only the branch which fired, and reject unknown
    command-like input while ignoring identifiable output text.
    """
    lines = log_path.read_text(errors="replace").splitlines()
    low = [ln.lower() for ln in lines]

    def clean(ln: str) -> str:
        # A branch command may legitimately contain a quoted ``#``.  Strip
        # comments only when the marker is outside quotes.
        quote: Optional[str] = None
        escaped = False
        kept: list[str] = []
        for char in ln:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quote is not None:
                kept.append(char)
                escaped = True
                continue
            if char in {"'", '"'}:
                if quote is None:
                    quote = char
                elif quote == char:
                    quote = None
                kept.append(char)
                continue
            if char == "#" and quote is None:
                break
            kept.append(char)
        return "".join(kept).strip()

    begins = [i for i, ln in enumerate(low) if "begin kim interactions" in ln]
    ends = [i for i, ln in enumerate(low) if "end kim interactions" in ln]
    if len(begins) != 1 or len(ends) != 1 or not begins[0] < ends[0]:
        raise ValueError(
            "KIM command extraction requires exactly one ordered BEGIN/END "
            f"KIM INTERACTIONS block in {log_path}; found "
            f"begin={len(begins)}, end={len(ends)}"
        )
    b, e = begins[0], ends[0]
    allowed_heads = {
        "pair_style",
        "pair_coeff",
        "kspace_style",
        "kspace_modify",
        "pair_modify",
        "special_bonds",
        "neighbor",
        "neigh_modify",
        "dielectric",
        "set",
        "fix",
    }

    # These are output records produced by commands in known KIM Simulator
    # Models.  Keep this deliberately narrow: lower-case, command-shaped text
    # which is not listed here continues to fail closed.
    output_patterns = tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"^setting\s+atom\s+values\b",
            r"^\d+\s+settings?\s+made(?:\s+for\b|\s*$)",
            r"^generated\s+\d+\s+of\s+\d+\s+mixed\s+pair_coeff\s+terms\b",
            r"^reading\s+.+\s+potential\s+file\b",
            r"^(?:warning|note):\s+",
        )
    )

    def is_output_record(command: str) -> bool:
        if any(pattern.search(command) for pattern in output_patterns):
            return True
        # LAMMPS commands are case-sensitive and use lower-case command names;
        # capitalised prose is screen/log output, not executable input.
        head = command.split(maxsplit=1)[0]
        return bool(head and head[0].isupper())

    variables: dict[str, str] = {}

    def substitute_variables(text: str, *, require_all: bool) -> str:
        missing: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                missing.add(name)
                return match.group(0)
            return variables[name]

        resolved = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, text)
        if missing and require_all:
            raise ValueError(
                "Cannot resolve KIM-generated variable reference(s) "
                f"{sorted(missing)!r} in interaction command: {text}"
            )
        return resolved

    def scalar_condition(expression: str) -> bool:
        expression = substitute_variables(expression, require_all=True).strip()
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"(?<![<>=!])!(?!=)", " not ", expression)
        expression = re.sub(r"\btrue\b", "True", expression, flags=re.IGNORECASE)
        expression = re.sub(r"\bfalse\b", "False", expression, flags=re.IGNORECASE)
        expression = re.sub(r"\byes\b", "True", expression, flags=re.IGNORECASE)
        expression = re.sub(r"\bno\b", "False", expression, flags=re.IGNORECASE)
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Unsupported KIM-generated if condition: {expression!r}"
            ) from exc

        def value(node: ast.AST) -> Union[bool, int, float, str]:
            if isinstance(node, ast.Expression):
                return value(node.body)
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (bool, int, float, str)
            ):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not bool(value(node.operand))
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                operand = value(node.operand)
                if not isinstance(operand, (int, float)) or isinstance(operand, bool):
                    raise ValueError("non-numeric unary operand")
                return +operand if isinstance(node.op, ast.UAdd) else -operand
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                return all(bool(value(item)) for item in node.values)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                return any(bool(value(item)) for item in node.values)
            if isinstance(node, ast.Compare):
                # Python gives chained comparisons special mathematical
                # semantics, whereas LAMMPS variable expressions apply
                # relational operators left-to-right.  Rather than risk
                # selecting a different interaction branch, accept only one
                # comparison per scalar condition; nested/boolean combinations
                # of individual comparisons remain supported.
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise ValueError("chained comparisons are not supported")
                left = value(node.left)
                for operator, comparator in zip(node.ops, node.comparators):
                    right = value(comparator)
                    if isinstance(operator, ast.Eq):
                        passed = left == right
                    elif isinstance(operator, ast.NotEq):
                        passed = left != right
                    elif isinstance(operator, ast.Lt):
                        passed = left < right  # type: ignore[operator]
                    elif isinstance(operator, ast.LtE):
                        passed = left <= right  # type: ignore[operator]
                    elif isinstance(operator, ast.Gt):
                        passed = left > right  # type: ignore[operator]
                    elif isinstance(operator, ast.GtE):
                        passed = left >= right  # type: ignore[operator]
                    else:
                        raise ValueError("unsupported comparison operator")
                    if not passed:
                        return False
                    left = right
                return True
            raise ValueError(
                f"unsupported expression element {type(node).__name__}"
            )

        try:
            return bool(value(tree))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported KIM-generated if condition: {expression!r}"
            ) from exc

    block = [clean(ln) for ln in lines[b + 1 : e]]
    block = [command for command in block if command]
    out: list[str] = []
    # LAMMPS may echo the command selected by an ``if`` immediately after the
    # control line.  Suppress only that syntactic echo.  A global de-duplicate
    # is not safe: repeated commands are order-sensitive in LAMMPS (notably a
    # repeated ``pair_style`` resets pair coefficients), and preserving them
    # lets the downstream exactly-one-style audit reject ambiguous blocks.
    expected_immediate_echoes: list[tuple[str, ...]] = []

    def append_interaction(command: str) -> tuple[str, ...]:
        resolved = substitute_variables(command, require_all=True).strip()
        try:
            tokens = shlex.split(resolved, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"Malformed KIM-generated interaction command: {command}"
            ) from exc
        if not tokens or tokens[0].lower() not in allowed_heads:
            raise ValueError(
                "Unsupported KIM-generated material command in interaction "
                f"block: {resolved}"
            )
        out.append(resolved)
        return tuple(tokens)

    for c in block:
        try:
            tokens = shlex.split(c, posix=True)
        except ValueError as exc:
            raise ValueError(f"Malformed KIM interaction-block line: {c}") from exc
        if not tokens:
            continue
        if expected_immediate_echoes:
            if tuple(tokens) == expected_immediate_echoes[0]:
                expected_immediate_echoes.pop(0)
                continue
            # Only a contiguous, token-identical line immediately following
            # the conditional is identifiable as its echo.  Anything else is
            # independent material input and must retain its original order.
            expected_immediate_echoes.clear()
        head = tokens[0].lower()
        if head in allowed_heads:
            append_interaction(c)
            continue
        if head == "variable":
            if len(tokens) < 3:
                raise ValueError(f"Malformed KIM-generated variable command: {c}")
            name, style = tokens[1], tokens[2].lower()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Malformed KIM-generated variable name: {name!r}")
            if style == "delete" and len(tokens) == 3:
                variables.pop(name, None)
                continue
            if style not in {"equal", "internal", "string", "index"} or len(tokens) < 4:
                raise ValueError(
                    f"Unsupported KIM-generated variable command: {c}"
                )
            raw_value = " ".join(tokens[3:]) if style == "equal" else tokens[3]
            variables[name] = substitute_variables(raw_value, require_all=True)
            continue
        if head == "if":
            if len(tokens) < 4 or "then" not in [token.lower() for token in tokens]:
                raise ValueError(f"Malformed KIM-generated if command: {c}")
            lower_tokens = [token.lower() for token in tokens]
            then_index = lower_tokens.index("then")
            if then_index != 2:
                raise ValueError(f"Unsupported KIM-generated if command: {c}")
            if "elif" in lower_tokens:
                raise ValueError(f"Unsupported KIM-generated elif command: {c}")
            else_index = (
                lower_tokens.index("else", then_index + 1)
                if "else" in lower_tokens[then_index + 1 :]
                else len(tokens)
            )
            then_commands = tokens[then_index + 1 : else_index]
            else_commands = tokens[else_index + 1 :] if else_index < len(tokens) else []
            # Do not infer a branch merely because one candidate command is
            # echoed elsewhere in the block: an unrelated command with the
            # same spelling could occur before the conditional and make that
            # inference order-incorrect.  Every accepted conditional must be
            # resolved from its bounded scalar variables; otherwise extraction
            # fails closed.
            fired = then_commands if scalar_condition(tokens[1]) else else_commands
            branch_tokens: list[tuple[str, ...]] = []
            for command in fired:
                branch_tokens.append(append_interaction(command))
            expected_immediate_echoes = branch_tokens
            continue
        if is_output_record(c):
            continue
        raise ValueError(
            "Unsupported KIM-generated material command in interaction "
            f"block: {c}"
        )

    if not out:
        raise ValueError(f"Could not extract KIM-generated commands from log: {log_path}")
    return out


def _localize_input_data_for_preflight(*, source: Path, destination: Path) -> Path:
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(source).read_bytes())
    strip_lammps_data_pair_coeff_sections(dst)
    return dst


def _kim_extract_commands(runner: LammpsRunner, config: RunConfig, input_data: Path, outdir: Path) -> list[str]:
    """Kim extract commands."""
    stage_dir = outdir / "preflight" / "kim_extract"
    ensure_dir(stage_dir)

    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    script = f"""# vitriflow preflight: extract KIM-generated interaction commands
{kim_init_line(config.kim)}
atom_style {config.md.atom_style}
boundary p p p
atom_modify map array

read_data input.data

{kim_interactions_line(config.kim)}
"""
    runner.run(script, stage_dir, "log.lammps", timeout_sec=float(config.autotune.preflight.timeout_sec))
    cmds = _extract_kim_interactions_block(stage_dir / "log.lammps")
    (stage_dir / "kim_generated_commands.txt").write_text("\n".join(cmds) + "\n")
    return cmds


def _rewrite_for_hybrid_overlay_zbl(
    base_cmds: list[str],
    *,
    species: list[str],
    r_in: float,
    r_out: float,
) -> tuple[list[str], str]:
    """Rewrite for hybrid."""
    pair_style_line = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            pair_style_line = ln.strip()
            break
    if pair_style_line is None:
        raise ValueError("No pair_style line found in KIM-generated command block")

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    base_style = toks[1]
    base_args = toks[2:]

    if "buck" not in base_style.lower():
        raise ValueError(f"Base pair_style is not Buckingham-like: {base_style}")

    base_coeff: list[str] = []
    other: list[str] = []
    for ln in base_cmds:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("pair_style"):
            continue
        if s.startswith("pair_coeff"):
            tt = s.split()
            # hybrid syntax inserted
            if len(tt) >= 4 and tt[3] == base_style:
                base_coeff.append(s)
            else:
                base_coeff.append(" ".join(tt[:3] + [base_style] + tt[3:]))
            continue
        other.append(s)

    hybrid_pair_style = "pair_style hybrid/overlay " + " ".join(
        [base_style] + base_args + ["zbl", f"{r_in:.6g}", f"{r_out:.6g}"]
    )

    zbl_coeff: list[str] = []
    Z = [_atomic_number(s) for s in species]
    ntypes = len(Z)
    for i in range(1, ntypes + 1):
        for j in range(i, ntypes + 1):
            zbl_coeff.append(f"pair_coeff {i} {j} zbl {Z[i-1]} {Z[j-1]}")

    # order pair remaining
    return [hybrid_pair_style] + base_coeff + zbl_coeff + other, base_style


def _rewrite_for_hybrid_overlay_lj_repulsive(
    base_cmds: list[str],
    *,
    species: list[str],
    r_in: float,
    r_out: float,
    u_target_eV: float,
) -> tuple[list[str], str]:
    """Rewrite for hybrid."""
    pair_style_line = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            pair_style_line = ln.strip()
            break
    if pair_style_line is None:
        raise ValueError("No pair_style line found in KIM-generated command block")

    toks = pair_style_line.split()
    if len(toks) < 2:
        raise ValueError(f"Malformed pair_style line: {pair_style_line}")
    base_style = toks[1]
    base_args = toks[2:]

    if "buck" not in base_style.lower():
        raise ValueError(f"Base pair_style is not Buckingham-like: {base_style}")

    # prepare coefficients hybrid
    base_coeff: list[str] = []
    other: list[str] = []
    for ln in base_cmds:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("pair_style"):
            continue
        if s.startswith("pair_coeff"):
            tt = s.split()
            if len(tt) >= 4 and tt[3] == base_style:
                base_coeff.append(s)
            else:
                base_coeff.append(" ".join(tt[:3] + [base_style] + tt[3:]))
            continue
        other.append(s)

    # repulsive minimum force
    if r_out <= 0 or r_in <= 0 or r_in >= r_out:
        raise ValueError(f"Invalid LJ core radii: r_in={r_in}, r_out={r_out}")
    sigma = float(r_out) / (2.0 ** (1.0 / 6.0))
    x = float(sigma) / float(r_in)
    denom = 4.0 * (x ** 12 - x ** 6) + 1.0
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("Failed to compute LJ calibration denominator")
    eps = float(u_target_eV) / float(denom)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid LJ epsilon computed")

    hybrid_pair_style = "pair_style hybrid/overlay " + " ".join(
        [base_style] + base_args + ["lj/cut", f"{r_out:.6g}"]
    )

    lj_coeff: list[str] = []
    ntypes = len(species)
    for i in range(1, ntypes + 1):
        for j in range(i, ntypes + 1):
            lj_coeff.append(f"pair_coeff {i} {j} lj/cut {eps:.6g} {sigma:.6g}")

    # order pair remaining
    return [hybrid_pair_style, "pair_modify shift yes"] + base_coeff + lj_coeff + other, base_style


def _run_stability_test(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    *,
    outdir: Path,
    potential_lines: list[str],
    temperature: float,
    timestep: float,
    label: str = "",
) -> bool:
    """Stability test."""

    # unique directory preserve
    stage_dir = _core_stability_stage_dir(outdir, temperature=temperature, label=label)
    ensure_dir(stage_dir)
    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    dt = float(timestep)
    Te = float(temperature)

    core = config.kim.core_repulsion
    pf = config.autotune.preflight

    # thermostat time constants
    tdamp = max(float(config.md.thermostat.tdamp), 100.0 * dt)
    langevin_damp = float(core.langevin_damp) if core.langevin_damp is not None else max(100.0 * dt, 0.05)

    # conservative metal distance
    limit_max_disp = float(getattr(core, "limit_max_disp", 0.02))
    ramp_steps = int(getattr(core, "ramp_steps", 5000))
    limit_hold_steps = int(getattr(core, "limit_hold_steps", 2000))

    # important preserve physical
    # otherwise smaller candidates
    # appear spuriously stable
    dt_ref = float(config.md.timestep)
    ramp_steps = scale_steps_for_timestep(ramp_steps, dt_ref, dt, min_steps=1)
    limit_hold_steps = scale_steps_for_timestep(limit_hold_steps, dt_ref, dt, min_steps=1)

    # stability highest scan
    nvt_steps = int(core.test_equil_steps) + int(core.test_run_steps)
    try:
        nvt_steps = max(nvt_steps, int(config.autotune.tm_scan.equil_steps) + int(config.autotune.tm_scan.sample_steps))
    except Exception:
        pass
    nvt_steps = max(2000, int(nvt_steps))
    nvt_steps = scale_steps_for_timestep(int(nvt_steps), dt_ref, dt, min_steps=2000)

    pot_block = "\n".join([str(x).strip() for x in potential_lines if str(x).strip()])
    T0 = float(getattr(pf, "T_low", None) or float(config.md.temperature) or 300.0)
    if T0 <= 0.0 or not math.isfinite(T0):
        T0 = 300.0
    T0 = min(T0, Te)

    # deterministic decouples timestep
    # pathologies volume neighbor
    # thermo scan confirmation
    final_hold = f"""# Nose-Hoover NVT hold
fix int all nvt temp {Te} {Te} {tdamp}
run {nvt_steps}
unfix int"""

    # auxiliary potential present
    prepare_potential_files(config.kim, stage_dir, potential_lines)

    init_block = "\n".join(potential_init_lines(config.kim))

    def _build_script(md_use: MDConfig) -> str:
        return f"""# vitriflow preflight: core stability (high T)

    {init_block}
    atom_style {config.md.atom_style}
    boundary p p p
    atom_modify map array

    read_data input.data

    # potential force field
    {pot_block}

    # conservative neighbour stability
    neighbor {md_use.neighbor_skin} bin
    neigh_modify every 1 delay 0 check yes

    timestep {dt}

    thermo_style custom step time temp press pe ke etotal vol density lx ly lz
    thermo {max(50, int(config.md.thermo_every))}
    thermo_modify flush yes

    # minimisation relieve contacts
    min_style cg
    minimize 1.0e-6 1.0e-8 500 5000

    # velocities low ramp
    velocity all create {T0} 12345 mom yes rot yes dist gaussian

    # langevin displacement limiting
    fix lang all langevin {T0} {Te} {langevin_damp} 12345
    fix lim all nve/limit {limit_max_disp}
    run {ramp_steps}
    unfix lim
    unfix lang

    # short limited langevin
    fix lang2 all langevin {Te} {Te} {langevin_damp} 54321
    fix lim2 all nve/limit {limit_max_disp}
    run {limit_hold_steps}
    unfix lim2
    unfix lang2

    {final_hold}

    write_data output.data nocoeff
    """

    try:
        run_with_neighbor_skin_autotune(
            runner,
            _build_script,
            stage_dir,
            config.md,
            log_name="log.lammps",
            timeout_sec=float(config.autotune.preflight.timeout_sec),
            cleanup_paths=[stage_dir / "output.data"],
        )
    except Exception:
        return False

    # sanity thermo table
    try:
        tbl = parse_last_thermo_table(stage_dir / "log.lammps").as_dict()
        T = np.asarray(tbl.get("Temp", []), dtype=float)
        if len(T) == 0 or not np.all(np.isfinite(T)):
            return False
        if float(np.nanmax(T)) > float(pf.max_temp_factor) * float(temperature):
            return False
        P = np.asarray(tbl.get("Press", []), dtype=float)
        if len(P) > 0 and np.any(np.isfinite(P)):
            if float(np.nanmax(np.abs(P))) > float(pf.max_press_abs):
                return False
    except Exception:
        return False
    return True


def _core_stability_stage_dir(outdir: Path, *, temperature: float, label: str = "") -> Path:
    safe = str(label).strip()
    if safe:
        safe = safe.replace(".", "p").replace("-", "m").replace("+", "")
        safe = "_" + safe
    return Path(outdir) / "preflight" / f"core_stability_T{int(round(float(temperature)))}{safe}"


def _parse_gewald_from_log(log_path: Path) -> Optional[float]:
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    patterns = [
        r"G\s+vector(?:\s*\([^)]*\))?\s*=\s*([0-9Ee+\-.]+)",
        r"g[_-]?ewald\s*=\s*([0-9Ee+\-.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _resolve_tabulated_gewald(
    config: RunConfig,
    base_cmds: Sequence[str],
    *,
    stage_log: Optional[Path] = None,
) -> Optional[float]:
    core = config.kim.core_repulsion
    explicit = getattr(core, "table_gewald", None)
    if explicit is not None:
        return float(explicit)
    for ln in base_cmds:
        toks = str(ln).split()
        if len(toks) < 3 or toks[0] != "kspace_modify":
            continue
        i = 1
        while i < len(toks) - 1:
            if toks[i].lower() == "gewald":
                return float(toks[i + 1])
            i += 1
    if stage_log is not None:
        val = _parse_gewald_from_log(stage_log)
        if val is not None:
            return val
        try:
            screen = Path(stage_log).with_name("screen.out")
        except Exception:
            screen = None
        if screen is not None:
            return _parse_gewald_from_log(screen)
    return None


def _pair_style_tokens(base_cmds: Sequence[str]) -> list[str]:
    """Return the effective pair-style tokens without inline comments.

    Autocore must inspect hybrid substyles as well as the top-level style.  In
    particular, ``pair_style hybrid/overlay ... buck ...`` is Buckingham-like
    even though token 2 is not.  This deliberately conservative token view is
    only used for safety gating; the potential parser remains authoritative
    for whether a particular additive Hamiltonian can be represented.
    """

    lines: list[list[str]] = []
    for raw in base_cmds:
        stripped = str(raw).split("#", 1)[0].strip()
        tokens = stripped.split()
        if tokens and tokens[0].lower() == "pair_style":
            lines.append(tokens[1:])
    if len(lines) != 1:
        return []
    return lines[0]


def _potential_contains_buckingham(base_cmds: Sequence[str]) -> bool:
    """Conservatively detect Buckingham direct and hybrid substyles."""

    return any(str(token).strip().lower().startswith("buck") for token in _pair_style_tokens(base_cmds))


def _potential_contains_coulomb(base_cmds: Sequence[str]) -> bool:
    """Return whether the pair-style declaration contains a Coulomb substyle."""

    return any("coul/" in str(token).strip().lower() for token in _pair_style_tokens(base_cmds))


def _potential_requires_gewald(base_cmds: Sequence[str]) -> bool:
    """Return whether tabulation needs LAMMPS's actual Ewald splitting value."""

    return any("coul/long" in str(token).strip().lower() for token in _pair_style_tokens(base_cmds))


def _probe_original_potential_gewald(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    *,
    outdir: Path,
    potential_lines: Sequence[str],
) -> float:
    """Initialize KSpace with ``run 0`` and return the actual G-Ewald value.

    This is the sole system-bearing execution of the original Buckingham
    potential in autocore.  It intentionally contains no minimization,
    integration fix, velocity creation, timestep scan, or write operation.
    If a fixed value was configured, it is applied to this probe and checked
    against what LAMMPS reports; otherwise the LAMMPS-selected value is frozen
    into the generated table and all subsequent runtime potential blocks.
    """

    if not _potential_requires_gewald(potential_lines):
        raise ValueError("G-Ewald probe requested for a potential without coul/long")

    stage_dir = Path(outdir) / "preflight" / "core_kspace_probe"
    ensure_dir(stage_dir)
    _localize_input_data_for_preflight(
        source=input_data,
        destination=stage_dir / "input.data",
    )
    for name in ("log.lammps", "screen.out", "stdout.txt", "stderr.txt"):
        try:
            (stage_dir / name).unlink()
        except FileNotFoundError:
            pass

    prepare_potential_files(config.kim, stage_dir, potential_lines)
    init_block = "\n".join(potential_init_lines(config.kim))
    potential_block = "\n".join(
        str(line).strip() for line in potential_lines if str(line).strip()
    )
    requested = _resolve_tabulated_gewald(config, potential_lines)
    fixed_gewald = (
        f"kspace_modify gewald {float(requested):.16g}" if requested is not None else ""
    )

    def _build_script(md_use: MDConfig) -> str:
        return f"""# vitriflow preflight: original-potential KSpace run-0 probe
{init_block}
atom_style {config.md.atom_style}
boundary p p p
atom_modify map array

read_data input.data

{potential_block}
{fixed_gewald}

neighbor {md_use.neighbor_skin} bin
neigh_modify every 1 delay 0 check yes

thermo_style custom step temp press pe etotal vol density lx ly lz
thermo 1
thermo_modify flush yes

run 0
"""

    selected_skin, skin_retries = run_with_neighbor_skin_autotune(
        runner,
        _build_script,
        stage_dir,
        config.md,
        log_name="log.lammps",
        timeout_sec=float(config.autotune.preflight.timeout_sec),
    )
    resolved = _parse_gewald_from_log(stage_dir / "log.lammps")
    if resolved is None:
        resolved = _parse_gewald_from_log(stage_dir / "screen.out")
    if resolved is None or not (math.isfinite(float(resolved)) and float(resolved) > 0.0):
        raise PreflightError(
            "Original-potential run-0 probe did not report a finite positive G-Ewald; "
            "refusing to construct a Coulombic autocore table"
        )

    resolved = float(resolved)
    if requested is not None and not math.isclose(
        resolved,
        float(requested),
        rel_tol=5.0e-6,
        abs_tol=5.0e-12 * max(1.0, abs(float(requested))),
    ):
        raise PreflightError(
            "Original-potential run-0 probe reported a G-Ewald value inconsistent "
            f"with the fixed request: reported={resolved:.16g}, requested={float(requested):.16g}"
        )

    # Preserve the exact configured value after checking a potentially rounded
    # LAMMPS log rendering.  For an automatic choice, the reported value is the
    # only reproducible value available and becomes authoritative.
    selected = float(requested) if requested is not None else resolved
    report = {
        "operation": "run_0_only",
        "dynamics_performed": False,
        "requested_gewald": None if requested is None else float(requested),
        "reported_gewald": resolved,
        "selected_gewald": selected,
        "neighbor_skin": float(selected_skin),
        "neighbor_skin_retries": int(skin_retries),
    }
    (stage_dir / "gewald_probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return selected


def _parse_pair_table_file(path: Path) -> dict[str, dict[str, np.ndarray]]:
    lines = Path(path).read_text(errors="ignore").splitlines()
    out: dict[str, dict[str, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        section = s.split()[0]
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        hdr = lines[i].split()
        if len(hdr) < 2 or hdr[0] != "N":
            raise ValueError(f"Malformed pair_write/table section header for {section!r} in {path}")
        n_expected = int(hdr[1])
        i += 1
        rows_r: list[float] = []
        rows_u: list[float] = []
        rows_f: list[float] = []
        while i < len(lines):
            row = lines[i].strip()
            if not row:
                i += 1
                if len(rows_r) >= n_expected:
                    break
                continue
            if row.startswith("#"):
                i += 1
                continue
            toks = row.split()
            if len(toks) >= 4 and toks[0].isdigit():
                rows_r.append(float(toks[1]))
                rows_u.append(float(toks[2]))
                rows_f.append(float(toks[3]))
                i += 1
                if len(rows_r) >= n_expected:
                    break
                continue
            break
        if len(rows_r) != n_expected:
            raise ValueError(
                f"Expected {n_expected} rows for pair-table section {section!r} in {path}, found {len(rows_r)}"
            )
        out[section] = {
            "r": np.asarray(rows_r, dtype=float),
            "energy": np.asarray(rows_u, dtype=float),
            "force": np.asarray(rows_f, dtype=float),
        }
    if not out:
        raise ValueError(f"No pair-table sections found in {path}")
    return out


_TABLE_WARNING_PATTERNS = (
    "force values in table",
    "distance values in table",
    "dE/dr",
    "tabulated",
)


def _table_warning_lines(
    stage_dir: Path,
    *,
    run_result: RunResult,
    log_name: str,
) -> list[str]:
    """Return table warnings emitted by one specific ``pair_write`` run.

    Pair-write verification directories are deliberately reused across table
    refinement candidates.  Reading the conventional diagnostic filenames in
    the directory can therefore attribute a warning from an earlier candidate
    to the current one, especially when the user configures ``-screen none``.
    The captured streams on ``RunResult`` and the exact requested log are the
    only authoritative outputs for this invocation.
    """

    expected_log = Path(stage_dir) / str(log_name)
    actual_log_raw = getattr(run_result, "log_file", None)
    if actual_log_raw is None:
        raise PreflightError(
            "pair_write runner did not report the log file for warning audit"
        )
    actual_log = Path(actual_log_raw)
    if actual_log.resolve() != expected_log.resolve():
        raise PreflightError(
            "pair_write runner reported an unexpected log file for warning "
            f"audit: expected {expected_log}, got {actual_log}"
        )

    msgs: list[str] = []
    seen: set[str] = set()
    sources = [
        str(getattr(run_result, "stdout", "") or ""),
        str(getattr(run_result, "stderr", "") or ""),
    ]
    if actual_log.exists():
        sources.append(actual_log.read_text(errors="ignore"))
    for source in sources:
        for raw in source.splitlines():
            s = raw.strip()
            low = s.lower()
            if "warning" not in low:
                continue
            if not any(pat.lower() in low for pat in _TABLE_WARNING_PATTERNS):
                continue
            if s not in seen:
                seen.add(s)
                msgs.append(s)
    return msgs


_LAMMPS_FORCE_TABLE_WARNING_RE = re.compile(
    r"(?P<count>[0-9]+)\s+of\s+(?P<points>[0-9]+)\s+force values "
    r"in table\s+(?P<section>\S+)\s+are inconsistent with\s+-dE/dr",
    flags=re.IGNORECASE,
)


def _audit_lammps_inflection_warnings(
    *,
    spec: Mapping[str, Any],
    table_path: Path,
    observed_warnings: Sequence[str],
) -> dict[str, Any]:
    """Classify only LAMMPS's proven inflection warnings as advisory.

    LAMMPS compares each interior force knot with the two adjacent energy
    secants.  A mathematically correct derivative can lie outside both at a
    true energy inflection; LAMMPS's warning itself documents that exception.
    Reproduce the predicate on the serialized file values and require every
    flagged three-knot interval to bracket a zero of the independent analytic
    second derivative.  The observed warning section, count, and table length
    must then match exactly.  Anything else remains blocking.
    """

    sections = _parse_pair_table_file(Path(table_path))
    points = int(spec["points"])
    r_min = float(spec["r_min"])
    units_style = str(spec.get("units", "metal")).strip() or "metal"
    pair_reports: dict[str, Any] = {}
    expected: dict[str, int] = {}
    blocking: list[str] = []

    for pair_raw in list(spec.get("pairs", []) or []):
        pair = dict(pair_raw)
        section = str(pair["section"])
        if section not in sections:
            blocking.append(
                f"serialized source table is missing section {section!r}"
            )
            continue
        data = sections[section]
        serialized_radius = np.asarray(data.get("r", []), dtype=float)
        energy = np.asarray(data.get("energy", []), dtype=float)
        force = np.asarray(data.get("force", []), dtype=float)
        if (
            serialized_radius.size != points
            or energy.size != points
            or force.size != points
        ):
            blocking.append(
                f"serialized table section {section!r} has "
                f"r/E/F sizes {serialized_radius.size}/{energy.size}/{force.size}; "
                f"expected {points}"
            )
            continue
        _require_finite_pair_curve_values(
            source="serialized source",
            section=section,
            values={"energy": energy, "force": force},
        )
        pair_cutoff = float(pair["pair_cutoff"])
        radius = _lammps_rsq_grid(r_min, pair_cutoff, points)
        if not np.array_equal(serialized_radius, radius):
            blocking.append(
                f"serialized table section {section!r} does not use the exact "
                "LAMMPS RSQ runtime knot grid"
            )
            continue
        left = -(energy[1:-1] - energy[:-2]) / (
            radius[1:-1] - radius[:-2]
        )
        right = -(energy[2:] - energy[1:-1]) / (
            radius[2:] - radius[1:-1]
        )
        interior = force[1:-1]
        mask = (interior < np.minimum(left, right)) | (
            interior > np.maximum(left, right)
        )
        indices = np.flatnonzero(mask) + 1
        expected[section] = int(indices.size)

        flagged: list[dict[str, Any]] = []
        all_inflections = True
        for index_raw in indices:
            index = int(index_raw)
            probe = radius[index - 1 : index + 2].copy()
            # The serialized cutoff knot stores the regularized potential's
            # one-sided (included) value, whereas evaluating the analytic
            # piecewise function at the exact cutoff returns the excluded-side
            # zero.  Keep the independent curvature audit on the same physical
            # side of that discontinuous hard cutoff.  Otherwise a warning at
            # N-2 can appear to bracket an inflection solely because its final
            # probe samples the excluded-side zero.
            if index + 1 == points - 1:
                probe[-1] = np.nextafter(pair_cutoff, r_min)
            _u, _du, d2u = _repulsive_regularized_pair_energy_derivatives(
                probe,
                pair=pair,
                units_style=units_style,
            )
            curvature_scale = max(1.0, float(np.max(np.abs(d2u))))
            zero_tol = 4096.0 * np.finfo(float).eps * curvature_scale
            brackets_zero = bool(
                float(np.min(d2u)) <= zero_tol
                and float(np.max(d2u)) >= -zero_tol
            )
            all_inflections = all_inflections and brackets_zero
            flagged.append(
                {
                    "index": index,
                    "r": float(radius[index]),
                    "left_secant": float(left[index - 1]),
                    "right_secant": float(right[index - 1]),
                    "force": float(force[index]),
                    "curvature": [float(value) for value in d2u],
                    "brackets_analytic_inflection": brackets_zero,
                }
            )
        if not all_inflections:
            blocking.append(
                f"LAMMPS force/energy predicate flags a non-inflection knot "
                f"in table section {section!r}"
            )
        pair_reports[section] = {
            "points": points,
            "predicted_warning_count": int(indices.size),
            "all_flagged_knots_bracket_analytic_inflections": all_inflections,
            "flagged_knots": flagged,
        }

    observed_by_section: dict[str, tuple[int, int, str]] = {}
    unclassified: list[str] = []
    for warning in [str(value) for value in observed_warnings]:
        match = _LAMMPS_FORCE_TABLE_WARNING_RE.search(warning)
        if match is None:
            unclassified.append(warning)
            continue
        section = str(match.group("section")).rstrip(".,;:")
        row = (
            int(match.group("count")),
            int(match.group("points")),
            warning,
        )
        if section in observed_by_section:
            blocking.append(
                f"duplicate LAMMPS force/energy warning for table section {section!r}"
            )
        else:
            observed_by_section[section] = row

    blocking.extend(unclassified)
    advisory: list[str] = []
    for section, predicted_count in expected.items():
        observed = observed_by_section.pop(section, None)
        if predicted_count == 0:
            if observed is not None:
                blocking.append(observed[2])
            continue
        if observed is None:
            blocking.append(
                f"LAMMPS omitted the predicted force/energy warning for table "
                f"section {section!r} ({predicted_count} of {points})"
            )
            continue
        observed_count, observed_points, warning = observed
        pair_ok = bool(
            pair_reports.get(section, {}).get(
                "all_flagged_knots_bracket_analytic_inflections", False
            )
        )
        if (
            observed_count == predicted_count
            and observed_points == points
            and pair_ok
        ):
            advisory.append(warning)
        else:
            blocking.append(warning)
    for _section, (_count, _points, warning) in observed_by_section.items():
        blocking.append(warning)

    return {
        "passed": not blocking,
        "advisory_warnings": advisory,
        "blocking_warnings": blocking,
        "pairs": pair_reports,
    }


def _atomless_pair_write_potential_lines(
    config: RunConfig,
    *,
    potential_lines: Sequence[str],
    spec: Mapping[str, Any],
) -> list[str]:
    """Remove only audited atom-dependent charge assignments for pair_write.

    LAMMPS ``pair_write`` evaluates one type pair in an empty simulation box;
    its optional ``qi qj`` arguments supply the charges used by Coulombic pair
    styles.  A real stage, in contrast, must retain KIM's fixed
    ``set type N charge Q`` assignments so they can update the atoms read from
    the data file.  Replaying those assignments in the empty box is both
    unnecessary and invalid on current LAMMPS versions.

    The omission is deliberately tied to the successful runtime-charge audit
    embedded in the table metadata.  Any unaudited ``set`` command, stale
    audit evidence, missing pair charge, or disagreement between the audited
    runtime state and ``pair_write`` arguments fails closed.
    """

    def canonical_command(raw: Any) -> str:
        return " ".join(str(raw).split("#", 1)[0].split())

    rendered = [str(line) for line in potential_lines]
    set_lines = [
        canonical_command(line)
        for line in rendered
        if canonical_command(line).split(maxsplit=1)[:1]
        and canonical_command(line).split(maxsplit=1)[0].lower() == "set"
    ]

    audit_raw = spec.get("runtime_charge_audit", None)
    audit = dict(audit_raw) if isinstance(audit_raw, Mapping) else None
    evidence_raw = list((audit or {}).get("fixed_set_commands", []) or [])

    # Metadata declaring charge-setting commands must agree with the execution
    # block even if the latter is unexpectedly missing them.
    if not set_lines and not evidence_raw:
        return rendered
    if audit is None or not bool(audit.get("passed", False)):
        raise ValueError(
            "atomless pair_write encountered a set command without a passed "
            "runtime-charge audit"
        )

    ntypes = len(list(config.kim.interactions))
    audited_commands: list[str] = []
    audited_counts: dict[str, int] = {}
    last_native_charge_by_type: dict[int, float] = {}
    for row_raw in evidence_raw:
        if not isinstance(row_raw, Mapping):
            raise ValueError(
                "atomless pair_write runtime-charge audit contains malformed "
                "fixed-set evidence"
            )
        row = dict(row_raw)
        command = canonical_command(row.get("command", ""))
        tokens = command.split()
        if not (
            len(tokens) == 5
            and tokens[0].lower() == "set"
            and tokens[1].lower() == "type"
            and tokens[3].lower() == "charge"
            and re.fullmatch(r"[1-9][0-9]*", tokens[2]) is not None
        ):
            raise ValueError(
                "atomless pair_write runtime-charge audit contains an unsupported "
                f"fixed-set command: {command or '<empty>'}"
            )
        atom_type = int(tokens[2])
        if not 1 <= atom_type <= ntypes:
            raise ValueError(
                "atomless pair_write audited charge command selects atom type "
                f"{atom_type}, outside [1, {ntypes}]"
            )
        try:
            native_charge = float(tokens[4].replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise ValueError(
                "atomless pair_write audited charge is not a numeric literal: "
                f"{command}"
            ) from exc
        if not math.isfinite(native_charge):
            raise ValueError(
                f"atomless pair_write audited charge is not finite: {command}"
            )
        if int(row.get("atom_type", atom_type)) != atom_type:
            raise ValueError(
                "atomless pair_write fixed-set audit disagrees with its command "
                f"for atom type {atom_type}"
            )
        if row.get("charge_native", None) is not None:
            recorded_native = float(row["charge_native"])
            scale = max(abs(native_charge), abs(recorded_native), np.finfo(float).tiny)
            if not math.isclose(
                native_charge,
                recorded_native,
                rel_tol=1.0e-12,
                abs_tol=64.0 * np.finfo(float).eps * scale,
            ):
                raise ValueError(
                    "atomless pair_write fixed-set audit charge disagrees with "
                    f"its command for atom type {atom_type}"
                )
        audited_counts[command] = audited_counts.get(command, 0) + 1
        audited_commands.append(command)
        last_native_charge_by_type[atom_type] = native_charge

    # Charge setters have last-assignment-wins semantics.  A multiset match is
    # insufficient: stale metadata with the same commands in a different
    # order could validate one effective charge while production executes
    # another.  Require the exact audited sequence before omitting it from the
    # atomless script.
    if set_lines != audited_commands:
        raise ValueError(
            "atomless pair_write runtime-charge audit is stale: ordered fixed "
            "charge commands differ from the potential block"
        )

    filtered: list[str] = []
    remaining = dict(audited_counts)
    for raw in rendered:
        command = canonical_command(raw)
        head = command.split(maxsplit=1)[:1]
        if not head or head[0].lower() != "set":
            filtered.append(raw)
            continue
        if remaining.get(command, 0) <= 0:
            raise ValueError(
                "atomless pair_write refuses an unaudited or duplicate set "
                f"command: {command}"
            )
        remaining[command] -= 1

    missing = [
        command
        for command, count in remaining.items()
        for _ in range(max(0, int(count)))
    ]
    if missing:
        raise ValueError(
            "atomless pair_write runtime-charge audit is stale: audited command "
            f"is absent from the potential block: {missing[0]}"
        )

    # The explicit pair_write charges must reproduce the effective runtime
    # charge state whose atom-mutating commands were omitted above.  Values in
    # both the LAMMPS command and table pair metadata are native-unit charges.
    species = [str(symbol) for symbol in config.kim.interactions]
    effective_raw = audit.get("effective_charges_e", None)
    if not isinstance(effective_raw, Mapping):
        raise ValueError(
            "atomless pair_write runtime-charge audit is missing effective charges"
        )
    native_to_e = float(charge_to_elementary_factor(config.kim.user_units))
    if not (math.isfinite(native_to_e) and native_to_e > 0.0):
        raise ValueError("atomless pair_write charge-unit conversion is invalid")
    pairs = list(spec.get("pairs", []) or [])
    if not pairs:
        raise ValueError(
            "atomless pair_write cannot omit fixed charge assignments because "
            "the table metadata contains no pair charge arguments"
        )
    for pair in pairs:
        pair_types = list(pair.get("pair", []) or [])
        if len(pair_types) != 2:
            raise ValueError("atomless pair_write table metadata has a malformed pair")
        for endpoint, charge_key in enumerate(("q_i", "q_j")):
            atom_type = int(pair_types[endpoint])
            if not 1 <= atom_type <= ntypes:
                raise ValueError(
                    f"atomless pair_write pair selects atom type {atom_type}, "
                    f"outside [1, {ntypes}]"
                )
            value_raw = pair.get(charge_key, None)
            if value_raw is None or not math.isfinite(float(value_raw)):
                raise ValueError(
                    "atomless pair_write cannot omit fixed charge assignments "
                    f"because pair {pair_types} lacks a finite {charge_key}"
                )
            symbol = species[atom_type - 1]
            if symbol not in effective_raw:
                raise ValueError(
                    "atomless pair_write runtime-charge audit is missing effective "
                    f"charge for {symbol!r}"
                )
            expected_native = float(effective_raw[symbol]) / native_to_e
            actual_native = float(value_raw)
            scale = max(
                abs(expected_native), abs(actual_native), np.finfo(float).tiny
            )
            if not math.isclose(
                actual_native,
                expected_native,
                rel_tol=1.0e-12,
                abs_tol=64.0 * np.finfo(float).eps * scale,
            ):
                raise ValueError(
                    "atomless pair_write charge argument disagrees with the "
                    f"audited runtime charge for {symbol!r}: "
                    f"pair_write={actual_native:.16g}, audited={expected_native:.16g}"
                )
            if atom_type in last_native_charge_by_type:
                set_native = float(last_native_charge_by_type[atom_type])
                set_scale = max(
                    abs(set_native), abs(actual_native), np.finfo(float).tiny
                )
                if not math.isclose(
                    actual_native,
                    set_native,
                    rel_tol=1.0e-12,
                    abs_tol=64.0 * np.finfo(float).eps * set_scale,
                ):
                    raise ValueError(
                        "atomless pair_write charge argument disagrees with the "
                        f"omitted fixed charge command for atom type {atom_type}"
                    )
    return filtered


def _render_pair_write_script(
    config: RunConfig,
    *,
    potential_lines: Sequence[str],
    spec: dict[str, Any],
    npoints: int,
    output_name: str,
) -> str:
    ntypes = len(list(config.kim.interactions))
    units_style = str(config.kim.user_units).strip() or "metal"
    atom_style = str(config.md.atom_style).strip() or "atomic"
    lines: list[str] = [
        "# vitriflow preflight: pair_write potential verification",
        f"units {units_style}",
        "dimension 3",
        f"atom_style {atom_style}",
        "boundary p p p",
        "region box block 0 20 0 20 0 20",
        f"create_box {ntypes} box",
    ]
    for i in range(1, ntypes + 1):
        lines.append(f"mass {i} 1.0")
    lines.append("")
    lines.extend(
        _atomless_pair_write_potential_lines(
            config,
            potential_lines=potential_lines,
            spec=spec,
        )
    )
    gewald = spec.get("gewald", None)
    if gewald is not None:
        # pair charge box
        # fixed defines tabulated
        # reference generated identical
        expected_gewald = float(gewald)
        rendered_gewald: list[float] = []
        for raw in lines:
            tokens = str(raw).split()
            if len(tokens) < 3 or tokens[0].lower() != "kspace_modify":
                continue
            for index in range(1, len(tokens) - 1):
                if tokens[index].lower() == "gewald":
                    rendered_gewald.append(float(tokens[index + 1]))
        if rendered_gewald:
            if any(
                not math.isclose(
                    actual,
                    expected_gewald,
                    rel_tol=1.0e-12,
                    abs_tol=64.0
                    * np.finfo(float).eps
                    * max(abs(actual), abs(expected_gewald), 1.0),
                )
                for actual in rendered_gewald
            ):
                raise ValueError(
                    "pair_write potential block has a G-ewald value that "
                    "disagrees with the tabulated-core metadata"
                )
        else:
            lines.append(
                f"kspace_modify gewald {_lammps_double_token(expected_gewald)}"
            )
    lines.append("")
    for pair in list(spec.get("pairs", [])):
        i, j = int(pair["pair"][0]), int(pair["pair"][1])
        pair_cut = float(pair["pair_cutoff"])
        rmin = float(pair.get("source_audit_r_min", spec["r_min"]))
        if not (math.isfinite(rmin) and 0.0 < rmin < pair_cut):
            raise ValueError(
                f"pair_write interval [{rmin:g}, {pair_cut:g}] is invalid for "
                f"section {pair.get('section', '?')}"
            )
        cmd = (
            f"pair_write {i} {j} {int(npoints)} rsq "
            f"{_lammps_double_token(rmin)} {_lammps_double_token(pair_cut)} "
            f"{output_name} {pair['section']}"
        )
        if pair.get("q_i", None) is not None and pair.get("q_j", None) is not None:
            cmd += (
                f" {_lammps_double_token(pair['q_i'])} "
                f"{_lammps_double_token(pair['q_j'])}"
            )
        lines.append(cmd)
    lines.append("")
    return "\n".join(lines) + "\n"



def _pair_write_potential_curves(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    stage_dir: Path,
    potential_lines: Sequence[str],
    spec: dict[str, Any],
    npoints: int,
    output_name: str,
    log_name: str,
) -> dict[str, Any]:
    stage_dir = Path(stage_dir)
    ensure_dir(stage_dir)
    out_path = stage_dir / output_name
    # The directory is reused during refinement.  Remove every conventional
    # runner artifact before invoking LAMMPS so neither a custom runner nor a
    # disabled screen stream can leave stale diagnostics available for audit.
    # Warning collection below is additionally limited to RunResult and this
    # invocation's exact log_name.
    stale_paths = {
        out_path,
        stage_dir / str(log_name),
        stage_dir / "screen.out",
        stage_dir / "stdout.txt",
        stage_dir / "stderr.txt",
    }
    for stale_path in stale_paths:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
        except IsADirectoryError as exc:
            raise PreflightError(
                f"pair_write output path is unexpectedly a directory: {stale_path}"
            ) from exc
    prepare_potential_files(config.kim, stage_dir, potential_lines)
    script = _render_pair_write_script(
        config,
        potential_lines=potential_lines,
        spec=spec,
        npoints=int(npoints),
        output_name=str(Path(output_name).name),
    )
    pairwrite_runner = runner
    try:
        if getattr(runner.cfg, "mpi_cmd", None):
            pairwrite_runner = LammpsRunner(runner.cfg.model_copy(update={"nprocs": 1}))
    except Exception:
        pairwrite_runner = runner
    run_result = pairwrite_runner.run(
        script,
        stage_dir,
        log_name,
        timeout_sec=float(config.autotune.preflight.timeout_sec),
    )
    if not out_path.exists():
        raise PreflightError(f"pair_write did not create expected output file: {out_path}")
    return {
        "path": out_path,
        "sections": _parse_pair_table_file(out_path),
        "warnings": _table_warning_lines(
            stage_dir,
            run_result=run_result,
            log_name=log_name,
        ),
    }




def _local_reference_error_allowance(
    reference: np.ndarray,
    *,
    rel_tol: float,
    abs_tol_frac: float,
    auxiliary_magnitude: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a scale-aware pointwise tolerance immune to a singular core.

    A global maximum is unsafe for repulsive tables: the first ZBL point can
    be many orders of magnitude above the physical tail.  Use the smaller of
    the left/right neighbour scales (so a cutoff jump does not inflate the
    tolerance on its zero side), with a median/roundoff floor.
    """

    ref = np.asarray(reference, dtype=float)
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference table values must be finite")
    magnitude = np.abs(ref)
    if auxiliary_magnitude is None:
        scale_magnitude = magnitude
        auxiliary_max = 0.0
    else:
        auxiliary = np.asarray(auxiliary_magnitude, dtype=float)
        if auxiliary.shape != ref.shape:
            raise ValueError("auxiliary tolerance scale must match the reference shape")
        if not np.all(np.isfinite(auxiliary)) or np.any(auxiliary < 0.0):
            raise ValueError("auxiliary tolerance scale must be finite and non-negative")
        scale_magnitude = np.maximum(magnitude, auxiliary)
        auxiliary_max = float(np.max(auxiliary)) if auxiliary.size else 0.0
    if magnitude.size == 0:
        return magnitude.copy(), {"robust_scale": 0.0, "roundoff_scale": 0.0}
    left = np.maximum(
        scale_magnitude,
        np.concatenate(([scale_magnitude[0]], scale_magnitude[:-1])),
    )
    right = np.maximum(
        scale_magnitude,
        np.concatenate((scale_magnitude[1:], [scale_magnitude[-1]])),
    )
    local = np.minimum(left, right)
    robust = float(np.median(scale_magnitude))
    max_scale = float(np.max(scale_magnitude))
    roundoff = max(float(np.finfo(float).tiny), 64.0 * np.finfo(float).eps * max_scale)
    floor = max(robust, roundoff)
    scale = np.maximum(local, floor)
    allow = float(rel_tol) * magnitude + float(abs_tol_frac) * scale
    allow = np.maximum(allow, float(np.finfo(float).tiny))
    return allow, {
        "robust_scale": robust,
        "roundoff_scale": roundoff,
        "absolute_floor_scale": floor,
        "max_local_scale": float(np.max(local)),
        "auxiliary_max_scale": auxiliary_max,
    }


def _require_finite_pair_curve_values(
    *,
    source: str,
    section: str,
    values: Mapping[str, np.ndarray],
) -> None:
    """Reject non-finite pair-curve data before computing error ratios.

    NumPy comparisons involving NaN can make a failed curve appear to have a
    small (even zero) aggregate error.  Treat non-finite radii, energies, or
    forces as a verification execution error and preserve the affected field
    and indices in the diagnostic.
    """

    for field_name, raw in values.items():
        array = np.asarray(raw, dtype=float)
        nonfinite = np.flatnonzero(~np.isfinite(array))
        if nonfinite.size:
            preview = ", ".join(str(int(index)) for index in nonfinite[:8])
            suffix = ", ..." if int(nonfinite.size) > 8 else ""
            raise ValueError(
                f"{source} pair table section {section!r} contains non-finite "
                f"{field_name} value(s) at indices [{preview}{suffix}]"
            )


def _compare_pair_table_sections(
    reference: dict[str, dict[str, np.ndarray]],
    realized: dict[str, dict[str, np.ndarray]],
    *,
    rel_tol: float,
    abs_tol_frac: float,
    critical_radii_by_section: Optional[dict[str, list[dict[str, Any]]]] = None,
    auxiliary_scale_sections: Optional[dict[str, dict[str, np.ndarray]]] = None,
    subtraction_roundoff_scale_sections: Optional[
        dict[str, dict[str, np.ndarray]]
    ] = None,
) -> dict[str, Any]:
    pair_reports: dict[str, Any] = {}
    overall_energy_ratio = 0.0
    overall_force_ratio = 0.0
    overall_r_error = 0.0
    worst_energy_section: Optional[str] = None
    worst_force_section: Optional[str] = None
    passed = True
    missing = sorted(set(reference) ^ set(realized))
    if missing:
        return {"passed": False, "missing_sections": missing, "pairs": {}, "overall": {}}
    for section in sorted(reference):
        ref = reference[section]
        got = realized[section]
        r_ref = np.asarray(ref["r"], dtype=float)
        r_got = np.asarray(got["r"], dtype=float)
        if r_ref.shape != r_got.shape:
            return {
                "passed": False,
                "missing_sections": [],
                "pairs": pair_reports,
                "overall": {"shape_mismatch": {section: [int(r_ref.size), int(r_got.size)]}},
            }
        r_err = float(np.max(np.abs(r_got - r_ref))) if r_ref.size else 0.0
        overall_r_error = max(overall_r_error, r_err)
        u_ref = np.asarray(ref["energy"], dtype=float)
        u_got = np.asarray(got["energy"], dtype=float)
        f_ref = np.asarray(ref["force"], dtype=float)
        f_got = np.asarray(got["force"], dtype=float)
        _require_finite_pair_curve_values(
            source="reference",
            section=section,
            values={"radius": r_ref, "energy": u_ref, "force": f_ref},
        )
        _require_finite_pair_curve_values(
            source="realized",
            section=section,
            values={"radius": r_got, "energy": u_got, "force": f_got},
        )
        du = u_got - u_ref
        df = f_got - f_ref
        auxiliary = (auxiliary_scale_sections or {}).get(section, {})
        u_allow, u_scale_report = _local_reference_error_allowance(
            u_ref,
            rel_tol=rel_tol,
            abs_tol_frac=abs_tol_frac,
            auxiliary_magnitude=auxiliary.get("energy", None),
        )
        f_allow, f_scale_report = _local_reference_error_allowance(
            f_ref,
            rel_tol=rel_tol,
            abs_tol_frac=abs_tol_frac,
            auxiliary_magnitude=auxiliary.get("force", None),
        )
        # A component obtained by subtracting two independently serialized
        # ``pair_write`` curves is conditionally resolvable only down to the
        # precision of those two operands.  LAMMPS writes 15 significant
        # decimal digits; 64 binary64 epsilons conservatively cover both
        # serialization errors, parsing, and the final subtraction.  Apply
        # this bound directly (never through the user absolute-tolerance
        # multiplier) and only when the caller supplies the actual operand
        # magnitudes.  This keeps ordinary source/table comparisons strict.
        subtraction_scale = (subtraction_roundoff_scale_sections or {}).get(
            section, {}
        )

        def _apply_subtraction_roundoff_bound(
            allowance: np.ndarray,
            raw_scale: Any,
            *,
            field_name: str,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            if raw_scale is None:
                return allowance, {
                    "subtraction_roundoff_factor_eps": 0.0,
                    "subtraction_operand_max_scale": 0.0,
                    "subtraction_roundoff_max_allowance": 0.0,
                    "subtraction_roundoff_limited_points": 0,
                }
            operand_scale = np.asarray(raw_scale, dtype=float)
            if operand_scale.shape != allowance.shape:
                raise ValueError(
                    f"{field_name} subtraction roundoff scale must match the "
                    "reference shape"
                )
            if not np.all(np.isfinite(operand_scale)) or np.any(operand_scale < 0.0):
                raise ValueError(
                    f"{field_name} subtraction roundoff scale must be finite and "
                    "non-negative"
                )
            factor = 64.0
            roundoff_allowance = factor * np.finfo(float).eps * operand_scale
            limited = roundoff_allowance > allowance
            return np.maximum(allowance, roundoff_allowance), {
                "subtraction_roundoff_factor_eps": factor,
                "subtraction_operand_max_scale": (
                    float(np.max(operand_scale)) if operand_scale.size else 0.0
                ),
                "subtraction_roundoff_max_allowance": (
                    float(np.max(roundoff_allowance))
                    if roundoff_allowance.size
                    else 0.0
                ),
                "subtraction_roundoff_limited_points": int(np.count_nonzero(limited)),
            }

        u_allow, u_roundoff_report = _apply_subtraction_roundoff_bound(
            u_allow,
            subtraction_scale.get("energy", None),
            field_name="energy",
        )
        f_allow, f_roundoff_report = _apply_subtraction_roundoff_bound(
            f_allow,
            subtraction_scale.get("force", None),
            field_name="force",
        )
        u_scale_report.update(u_roundoff_report)
        f_scale_report.update(f_roundoff_report)
        u_ratio_arr = np.abs(du) / u_allow if u_ref.size else np.zeros(0, dtype=float)
        f_ratio_arr = np.abs(df) / f_allow if f_ref.size else np.zeros(0, dtype=float)
        u_ratio = float(np.max(u_ratio_arr)) if u_ratio_arr.size else 0.0
        f_ratio = float(np.max(f_ratio_arr)) if f_ratio_arr.size else 0.0
        overall_energy_ratio = max(overall_energy_ratio, u_ratio)
        overall_force_ratio = max(overall_force_ratio, f_ratio)
        if u_ratio >= overall_energy_ratio:
            worst_energy_section = section
        if f_ratio >= overall_force_ratio:
            worst_force_section = section
        u_mask = np.abs(u_ref) > u_scale_report["absolute_floor_scale"] * float(abs_tol_frac)
        f_mask = np.abs(f_ref) > f_scale_report["absolute_floor_scale"] * float(abs_tol_frac)
        max_rel_u = float(np.max(np.abs(du[u_mask]) / np.abs(u_ref[u_mask]))) if np.any(u_mask) else 0.0
        max_rel_f = float(np.max(np.abs(df[f_mask]) / np.abs(f_ref[f_mask]))) if np.any(f_mask) else 0.0
        idx_u = int(np.argmax(u_ratio_arr)) if u_ratio_arr.size else 0
        idx_f = int(np.argmax(f_ratio_arr)) if f_ratio_arr.size else 0
        n_energy_fail = int(np.count_nonzero(u_ratio_arr > 1.0))
        n_force_fail = int(np.count_nonzero(f_ratio_arr > 1.0))
        pair_ok = bool((u_ratio <= 1.0) and (f_ratio <= 1.0) and (r_err <= 1.0e-12))
        passed = passed and pair_ok
        critical_reports: list[dict[str, Any]] = []
        for critical in (critical_radii_by_section or {}).get(section, []):
            radius = float(critical["r"])
            center = int(np.argmin(np.abs(r_ref - radius))) if r_ref.size else 0
            lo = max(0, center - 2)
            hi = min(int(r_ref.size), center + 3)
            local_u_ratio = float(np.max(u_ratio_arr[lo:hi])) if hi > lo else 0.0
            local_f_ratio = float(np.max(f_ratio_arr[lo:hi])) if hi > lo else 0.0
            critical_ok = bool(local_u_ratio <= 1.0 and local_f_ratio <= 1.0)
            pair_ok = pair_ok and critical_ok
            passed = passed and critical_ok
            critical_reports.append(
                {
                    "name": str(critical.get("name", "critical_radius")),
                    "r": radius,
                    "nearest_index": center,
                    "window_indices": [lo, max(lo, hi - 1)],
                    "max_energy_ratio": local_u_ratio,
                    "max_force_ratio": local_f_ratio,
                    "passed": critical_ok,
                }
            )
        pair_reports[section] = {
            "passed": pair_ok,
            "max_abs_energy_error": float(np.max(np.abs(du))) if du.size else 0.0,
            "max_abs_force_error": float(np.max(np.abs(df))) if df.size else 0.0,
            "max_rel_energy_error": max_rel_u,
            "max_rel_force_error": max_rel_f,
            "max_energy_ratio": u_ratio,
            "max_force_ratio": f_ratio,
            "max_r_error": r_err,
            "n_points": int(r_ref.size),
            "n_energy_fail": n_energy_fail,
            "n_force_fail": n_force_fail,
            "energy_fail_fraction": (float(n_energy_fail) / float(r_ref.size)) if r_ref.size else 0.0,
            "force_fail_fraction": (float(n_force_fail) / float(r_ref.size)) if r_ref.size else 0.0,
            "energy_tolerance_scale": u_scale_report,
            "force_tolerance_scale": f_scale_report,
            "critical_radius_neighborhoods": critical_reports,
            "worst_energy_point": {
                "index": idx_u,
                "r": float(r_ref[idx_u]) if r_ref.size else 0.0,
                "ref_energy": float(u_ref[idx_u]) if u_ref.size else 0.0,
                "got_energy": float(u_got[idx_u]) if u_got.size else 0.0,
                "abs_error": float(abs(du[idx_u])) if du.size else 0.0,
                "allow": float(u_allow[idx_u]) if u_allow.size else 0.0,
            },
            "worst_force_point": {
                "index": idx_f,
                "r": float(r_ref[idx_f]) if r_ref.size else 0.0,
                "ref_force": float(f_ref[idx_f]) if f_ref.size else 0.0,
                "got_force": float(f_got[idx_f]) if f_got.size else 0.0,
                "abs_error": float(abs(df[idx_f])) if df.size else 0.0,
                "allow": float(f_allow[idx_f]) if f_allow.size else 0.0,
            },
        }
    return {
        "passed": bool(passed),
        "missing_sections": [],
        "pairs": pair_reports,
        "overall": {
            "max_energy_ratio": overall_energy_ratio,
            "max_force_ratio": overall_force_ratio,
            "max_r_error": overall_r_error,
            "worst_energy_section": worst_energy_section,
            "worst_force_section": worst_force_section,
        },
    }



def _table_force_energy_consistency_report(
    sections: dict[str, dict[str, np.ndarray]],
    *,
    rel_tol: float,
    abs_tol_frac: float,
) -> dict[str, Any]:
    """Check discrete work/energy consistency without differentiating noise.

    A pointwise numerical derivative is ill-conditioned on the deliberately
    nonuniform RSQ grid, especially near the steep ZBL core and a force zero.
    Instead compare each energy increment with trapezoidal force work.  The
    half-range term is the conservative bound between the two endpoint-force
    rectangles.  The final interval is excluded because a LAMMPS hard cutoff
    is intentionally discontinuous there; its endpoint and interior
    neighbourhood are independently checked against the analytic reference.
    """
    pair_reports: dict[str, Any] = {}
    overall_ratio = 0.0
    worst_section: Optional[str] = None
    passed = True
    for section in sorted(sections):
        data = sections[section]
        r = np.asarray(data.get("r", []), dtype=float)
        u = np.asarray(data.get("energy", []), dtype=float)
        f = np.asarray(data.get("force", []), dtype=float)
        _require_finite_pair_curve_values(
            source="realized",
            section=section,
            values={"radius": r, "energy": u, "force": f},
        )
        if r.size < 3 or u.size != r.size or f.size != r.size:
            pair_ok = bool(r.size == u.size == f.size and r.size >= 2)
            pair_reports[section] = {
                "passed": pair_ok,
                "n_points": int(r.size),
                "max_force_ratio": 0.0,
                "max_abs_force_error": 0.0,
                "n_fail": 0,
                "fail_fraction": 0.0,
                "note": "insufficient points for interval work check",
            }
            passed = passed and pair_ok
            continue
        h_all = np.diff(r)
        if np.any(~np.isfinite(h_all)) or np.any(h_all <= 0.0):
            pair_reports[section] = {
                "passed": False,
                "n_points": int(r.size),
                "note": "radius grid is not finite and strictly increasing",
            }
            passed = False
            continue
        # Omit only the deliberate hard-cutoff jump from the work identity.
        h = h_all[:-1]
        delta_u = np.diff(u)[:-1]
        f_left = f[:-2]
        f_right = f[1:-1]
        trapezoid_work = 0.5 * h * (f_left + f_right)
        residual = delta_u + trapezoid_work
        force_scale = max(1.0, float(np.median(np.abs(f[:-1]))))
        variation_bound = 0.5 * h * np.abs(f_right - f_left)
        magnitude = np.abs(delta_u) + 0.5 * h * (
            np.abs(f_left) + np.abs(f_right)
        )
        allow = (
            variation_bound
            + float(rel_tol) * magnitude
            + float(abs_tol_frac) * h * force_scale
        )
        allow = np.maximum(allow, float(np.finfo(float).tiny))
        ratio_arr = np.abs(residual) / allow if residual.size else np.zeros(0, dtype=float)
        ratio = float(np.max(ratio_arr)) if ratio_arr.size else 0.0
        overall_ratio = max(overall_ratio, ratio)
        if ratio >= overall_ratio:
            worst_section = section
        idx = int(np.argmax(ratio_arr)) if ratio_arr.size else 0
        n_fail = int(np.count_nonzero(ratio_arr > 1.0))
        pair_ok = bool(ratio <= 1.0)
        passed = passed and pair_ok
        pair_reports[section] = {
            "passed": pair_ok,
            "n_points": int(r.size),
            "n_intervals_checked": int(ratio_arr.size),
            "max_abs_work_residual": float(np.max(np.abs(residual))) if residual.size else 0.0,
            "max_force_ratio": ratio,
            "n_fail": n_fail,
            "fail_fraction": (float(n_fail) / float(ratio_arr.size)) if ratio_arr.size else 0.0,
            "force_tolerance_scale": {
                "median_force_scale": force_scale,
                "variation_bound_included": True,
            },
            "worst_interval": {
                "index": idx,
                "r_left": float(r[idx]) if r.size else 0.0,
                "r_right": float(r[idx + 1]) if r.size > idx + 1 else 0.0,
                "energy_increment": float(delta_u[idx]) if delta_u.size else 0.0,
                "trapezoid_force_work": float(trapezoid_work[idx]) if trapezoid_work.size else 0.0,
                "abs_residual": float(abs(residual[idx])) if residual.size else 0.0,
                "allow": float(allow[idx]) if allow.size else 0.0,
            },
        }
    return {
        "passed": bool(passed),
        "pairs": pair_reports,
        "overall": {"max_force_ratio": overall_ratio, "worst_section": worst_section},
    }


def _materialize_pairwrite_tabulated_core_source(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    outdir: Path,
    source_potential_lines: Sequence[str],
    spec: dict[str, Any],
) -> Path:
    potdir = Path(outdir) / "preflight" / "potential_override"
    ensure_dir(potdir)
    raw_filename = spec.get("filename", "")
    fname = validated_lammps_localized_filename(
        "buckingham_core.table" if raw_filename == "" else raw_filename,
        field_name="tabulated core filename",
    )
    data = _pair_write_potential_curves(
        runner,
        config,
        stage_dir=potdir,
        potential_lines=source_potential_lines,
        spec=spec,
        npoints=int(spec["points"]),
        output_name=fname,
        log_name="log_pairwrite_source.lammps",
    )
    if data["warnings"]:
        raise PreflightError(
            "analytic source pair_write emitted table-related warnings: " + "; ".join(data["warnings"])
        )
    return Path(data["path"])


def _source_equivalence_component_references(
    spec: Mapping[str, Any],
    *,
    npoints: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    """Return exact full, non-Coulomb, Coulomb, and component-scale curves.

    The source-equivalence domain lies at or above every resolved outer join,
    so the unregularized parsed Hamiltonian is the relevant reference.  Split
    it into non-Coulomb and Coulomb pieces to avoid judging a small numerical
    realization error against a near-zero cancellation of large components.
    """

    units_style = str(spec.get("units", "metal") or "metal")
    pairs_by_section = {
        str(pair["section"]): dict(pair) for pair in list(spec.get("pairs", []))
    }
    full: dict[str, dict[str, np.ndarray]] = {}
    noncoul: dict[str, dict[str, np.ndarray]] = {}
    coul: dict[str, dict[str, np.ndarray]] = {}
    scale: dict[str, dict[str, np.ndarray]] = {}
    for section, pair in pairs_by_section.items():
        lower = float(pair.get("source_audit_r_min", pair["r_out"]))
        cutoff = float(pair["pair_cutoff"])
        if not (math.isfinite(lower) and 0.0 < lower < cutoff):
            raise ValueError(
                f"invalid source-equivalence interval [{lower:g}, {cutoff:g}] "
                f"for section {section!r}"
            )
        r = _lammps_rsq_grid(lower, cutoff, int(npoints))
        evaluation_r = np.array(r, copy=True)
        evaluation_r[-1] = np.nextafter(cutoff, lower)
        noncoul_pair = dict(pair)
        noncoul_pair["coul_mode"] = None
        u_noncoul, du_noncoul, _ = _pair_base_energy_derivatives(
            evaluation_r,
            pair=noncoul_pair,
            units_style=units_style,
        )
        # Evaluate the screened real-space Coulomb term directly.  Forming it
        # as ``(noncoul + coul) - noncoul`` destroys the very tail that this
        # component audit is intended to diagnose and makes its reference
        # depend on cancellation roundoff.
        u_coul, du_coul, _ = _pair_coulomb_energy_derivatives(
            evaluation_r,
            pair=pair,
            units_style=units_style,
            representation="runtime",
        )
        u_full = np.asarray(u_noncoul, dtype=float) + np.asarray(
            u_coul, dtype=float
        )
        du_full = np.asarray(du_noncoul, dtype=float) + np.asarray(
            du_coul, dtype=float
        )
        f_full = -du_full
        f_noncoul = -np.asarray(du_noncoul, dtype=float)
        f_coul = -np.asarray(du_coul, dtype=float)
        # pair_write reports the excluded side at the exact hard cutoff.
        for values in (u_full, f_full, u_noncoul, f_noncoul, u_coul, f_coul):
            values[-1] = 0.0
        full[section] = {"r": r.copy(), "energy": np.asarray(u_full), "force": f_full}
        noncoul[section] = {
            "r": r.copy(),
            "energy": np.asarray(u_noncoul),
            "force": f_noncoul,
        }
        coul[section] = {"r": r.copy(), "energy": u_coul, "force": f_coul}
        scale[section] = {
            "energy": np.abs(u_noncoul) + np.abs(u_coul),
            "force": np.abs(f_noncoul) + np.abs(f_coul),
        }
    return full, noncoul, coul, scale


def _subtract_pair_table_sections(
    minuend: Mapping[str, Mapping[str, np.ndarray]],
    subtrahend: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    """Subtract pair curves and return exact per-point operand magnitudes.

    The magnitude result is not a physical tolerance.  It is solely the
    conditioning scale needed to bound floating/serialization roundoff in the
    subtraction itself.
    """

    if set(minuend) != set(subtrahend):
        raise ValueError("pair-curve subtraction requires identical sections")
    result: dict[str, dict[str, np.ndarray]] = {}
    operand_scale: dict[str, dict[str, np.ndarray]] = {}
    for section in sorted(minuend):
        left = minuend[section]
        right = subtrahend[section]
        r_left = np.asarray(left["r"], dtype=float)
        r_right = np.asarray(right["r"], dtype=float)
        if r_left.shape != r_right.shape or not np.allclose(
            r_left, r_right, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"pair-curve subtraction found incompatible radius grids in {section!r}"
            )
        left_energy = np.asarray(left["energy"], dtype=float)
        right_energy = np.asarray(right["energy"], dtype=float)
        left_force = np.asarray(left["force"], dtype=float)
        right_force = np.asarray(right["force"], dtype=float)
        result[section] = {
            "r": r_left.copy(),
            "energy": left_energy - right_energy,
            "force": left_force - right_force,
        }
        operand_scale[section] = {
            "energy": np.abs(left_energy) + np.abs(right_energy),
            "force": np.abs(left_force) + np.abs(right_force),
        }
    return result, operand_scale


def _zero_pair_write_charges(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Copy table metadata while setting only pair_write probe charges to zero."""

    zero_spec = dict(spec)
    zero_pairs: list[dict[str, Any]] = []
    for raw_pair in list(spec.get("pairs", [])):
        pair = dict(raw_pair)
        if pair.get("coul_mode", None) == "long":
            pair["q_i"] = 0.0
            pair["q_j"] = 0.0
        zero_pairs.append(pair)
    zero_spec["pairs"] = zero_pairs
    zero_spec.pop("runtime_charge_audit", None)
    return zero_spec


def _audit_original_potential_above_core_joins(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    outdir: Path,
    source_potential_lines: Sequence[str],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Compare LAMMPS's additive source potential with the analytic parser.

    ``pair_write`` is atomless and performs no minimization or integration.
    Every pair is sampled from its own resolved outer join.  This audits the
    whole preserved source interval for pairs whose joins differ, without ever
    evaluating the singular Buckingham branch that autocore exists to
    replace.  Optional Qi/Qj arguments are rendered from the table metadata,
    making Coulombic direct and hybrid models auditable in an empty box.
    Agreement here is required before any generated table is materialized or
    used for stability dynamics.
    """

    pairs = [dict(pair) for pair in list(spec.get("pairs", []))]
    if not pairs:
        raise ValueError("source-equivalence audit requires at least one pair")
    resolved_outer = [float(pair["r_out"]) for pair in pairs]
    if not all(math.isfinite(value) and value > 0.0 for value in resolved_outer):
        raise ValueError("source-equivalence audit found an invalid resolved outer join")
    cutoffs = [float(pair["pair_cutoff"]) for pair in pairs]
    if not all(
        math.isfinite(cutoff) and cutoff > outer
        for cutoff, outer in zip(cutoffs, resolved_outer)
    ):
        raise ValueError(
            "source-equivalence audit requires every pair cutoff to lie above "
            "that pair's resolved outer join"
        )

    audit_spec = dict(spec)
    for pair, lower in zip(pairs, resolved_outer):
        pair["source_audit_r_min"] = float(lower)
    audit_spec["pairs"] = pairs
    audit_spec["r_min"] = float(min(resolved_outer))
    npoints = max(4097, int(config.kim.core_repulsion.table_verify_points))
    audit_spec["points"] = int(npoints)
    audit_dir = Path(outdir) / "preflight" / "source_equivalence"
    ensure_dir(audit_dir)
    rel_tol = float(config.kim.core_repulsion.table_verify_rel_tol)
    abs_tol_frac = float(config.kim.core_repulsion.table_verify_abs_tol_frac)

    coul_modes = {pair.get("coul_mode", None) for pair in pairs}
    if coul_modes == {"long"}:
        # Validate and remove atom-mutating fixed charge commands once, then
        # perform two otherwise identical pair_write probes.  Explicit Qi/Qj
        # are the only difference: the charged curve is the original total;
        # the zero-charge curve is its exact Buckingham/Morse realization.
        # Their numerical difference isolates LAMMPS's own Coulomb component.
        atomless_source_lines = _atomless_pair_write_potential_lines(
            config,
            potential_lines=source_potential_lines,
            spec=audit_spec,
        )
        probe_spec = dict(audit_spec)
        probe_spec.pop("runtime_charge_audit", None)
        zero_spec = _zero_pair_write_charges(probe_spec)
        full_reference, noncoul_reference, coul_reference, component_scale = (
            _source_equivalence_component_references(
                probe_spec,
                npoints=int(npoints),
            )
        )
        tabinner = max(
            float(np.finfo(float).eps), 0.5 * float(min(resolved_outer))
        )
        candidates: list[dict[str, Any]] = []
        previous_coul: Optional[dict[str, dict[str, np.ndarray]]] = None
        accepted: Optional[dict[str, Any]] = None
        # 16 and 18 bits establish convergence.  20 bits is a bounded fallback
        # for unusually stiff splits.  These overrides exist only in atomless
        # pair_write audit scripts and are never returned as runtime commands.
        for bits in (16, 18, 20):
            probe_lines = list(atomless_source_lines) + [
                f"pair_modify table {bits} tabinner {tabinner:.16g}"
            ]
            charged = _pair_write_potential_curves(
                runner,
                config,
                stage_dir=audit_dir / f"table_bits_{bits}" / "charged",
                potential_lines=probe_lines,
                spec=probe_spec,
                npoints=int(npoints),
                output_name="original_charged.table",
                log_name="log_pairwrite_charged.lammps",
            )
            zero = _pair_write_potential_curves(
                runner,
                config,
                stage_dir=audit_dir / f"table_bits_{bits}" / "zero_charge",
                potential_lines=probe_lines,
                spec=zero_spec,
                npoints=int(npoints),
                output_name="original_zero_charge.table",
                log_name="log_pairwrite_zero_charge.lammps",
            )
            isolated_coul, isolated_coul_roundoff_scale = _subtract_pair_table_sections(
                charged["sections"], zero["sections"]
            )
            noncoul_comparison = _compare_pair_table_sections(
                noncoul_reference,
                zero["sections"],
                rel_tol=rel_tol,
                abs_tol_frac=abs_tol_frac,
            )
            coul_comparison = _compare_pair_table_sections(
                coul_reference,
                isolated_coul,
                rel_tol=rel_tol,
                abs_tol_frac=abs_tol_frac,
                subtraction_roundoff_scale_sections=isolated_coul_roundoff_scale,
            )
            total_comparison = _compare_pair_table_sections(
                full_reference,
                charged["sections"],
                rel_tol=rel_tol,
                abs_tol_frac=abs_tol_frac,
                auxiliary_scale_sections=component_scale,
            )
            convergence = None
            if previous_coul is not None:
                convergence = _compare_pair_table_sections(
                    isolated_coul,
                    previous_coul,
                    rel_tol=rel_tol,
                    abs_tol_frac=abs_tol_frac,
                    auxiliary_scale_sections=component_scale,
                )
            candidate_warnings = list(charged.get("warnings", []) or []) + list(
                zero.get("warnings", []) or []
            )
            candidate_passed = bool(
                noncoul_comparison.get("passed", False)
                and coul_comparison.get("passed", False)
                and total_comparison.get("passed", False)
                and convergence is not None
                and convergence.get("passed", False)
                and not candidate_warnings
            )
            candidate = {
                "table_bits": int(bits),
                "tabinner": float(tabinner),
                "passed": candidate_passed,
                "warnings": candidate_warnings,
                "noncoulomb": noncoul_comparison,
                "coulomb": coul_comparison,
                "total_component_scaled": total_comparison,
                "coulomb_resolution_convergence": convergence,
            }
            candidates.append(candidate)
            if candidate_passed:
                accepted = candidate
                break
            previous_coul = isolated_coul

        comparison = (
            dict(accepted["total_component_scaled"])
            if accepted is not None
            else dict(candidates[-1]["total_component_scaled"])
        )
        warnings = (
            list(accepted.get("warnings", []) or [])
            if accepted is not None
            else [
                warning
                for candidate in candidates
                for warning in list(candidate.get("warnings", []) or [])
            ]
        )
        passed = accepted is not None
        component_audit: Optional[dict[str, Any]] = {
            "method": "same_source_charged_minus_zero_charge",
            "audit_table_bits_schedule": [16, 18, 20],
            "accepted_table_bits": (
                None if accepted is None else int(accepted["table_bits"])
            ),
            "audit_only_pair_modify": True,
            "runtime_potential_commands_modified": False,
            "candidates": candidates,
        }
    else:
        if "long" in coul_modes:
            raise ValueError(
                "source-equivalence audit found inconsistent coul/long metadata across pairs"
            )
        reference, _noncoul, _coul, _scale = (
            _source_equivalence_component_references(
                audit_spec,
                npoints=int(npoints),
            )
        )
        realized = _pair_write_potential_curves(
            runner,
            config,
            stage_dir=audit_dir,
            potential_lines=source_potential_lines,
            spec=audit_spec,
            npoints=int(npoints),
            output_name="original_above_joins.table",
            log_name="log_pairwrite_original.lammps",
        )
        comparison = _compare_pair_table_sections(
            reference,
            realized["sections"],
            rel_tol=rel_tol,
            abs_tol_frac=abs_tol_frac,
        )
        warnings = list(realized.get("warnings", []) or [])
        passed = bool(comparison.get("passed", False)) and not warnings
        component_audit = None
    report = {
        "passed": passed,
        "dynamics_performed": False,
        "sampling_domain": "each pair: r >= that pair's resolved r_out",
        "r_min": float(min(resolved_outer)),
        "min_resolved_r_out": float(min(resolved_outer)),
        "max_resolved_r_out": float(max(resolved_outer)),
        "r_min_by_section": {
            str(pair.get("section", "?")): float(pair["r_out"])
            for pair in pairs
        },
        "points": int(npoints),
        "explicit_pair_charges": {
            str(pair.get("section", "?")): {
                "q_i": pair.get("q_i", None),
                "q_j": pair.get("q_j", None),
            }
            for pair in pairs
        },
        "warnings": warnings,
        "comparison": comparison,
        "component_audit": component_audit,
        "rel_tol": rel_tol,
        "abs_tol_frac": abs_tol_frac,
    }
    (audit_dir / "source_equivalence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        overall = dict(comparison.get("overall", {}) or {})
        if component_audit is not None:
            last = component_audit["candidates"][-1]
            component_ratios = []
            for key in (
                "noncoulomb",
                "coulomb",
                "total_component_scaled",
                "coulomb_resolution_convergence",
            ):
                value = last.get(key, None)
                if isinstance(value, Mapping):
                    component_ratios.append(dict(value.get("overall", {}) or {}))
            if component_ratios:
                overall = {
                    "max_energy_ratio": max(
                        float(value.get("max_energy_ratio", 0.0) or 0.0)
                        for value in component_ratios
                    ),
                    "max_force_ratio": max(
                        float(value.get("max_force_ratio", 0.0) or 0.0)
                        for value in component_ratios
                    ),
                }
        raise PreflightError(
            "Original additive LAMMPS potential fails component-wise agreement or "
            "Coulomb-resolution convergence against the parsed analytic Hamiltonian "
            "above every resolved autocore join; refusing table generation "
            f"(max_energy_ratio={_format_candidate_metric(overall, 'max_energy_ratio')}, "
            f"max_force_ratio={_format_candidate_metric(overall, 'max_force_ratio')}, "
            f"warnings={len(warnings)})"
        )
    return report



def _verify_tabulated_core_against_source(
    runner: LammpsRunner,
    config: RunConfig,
    *,
    outdir: Path,
    table_potential_lines: Sequence[str],
    reference_sections: dict[str, dict[str, np.ndarray]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    verify_dir = Path(outdir) / "preflight" / "table_verify"
    ensure_dir(verify_dir)
    core = config.kim.core_repulsion
    verify_points = _tabulated_verification_points(
        configured_points=int(core.table_verify_points),
        table_points=int(spec["points"]),
    )
    expected_sections = reference_sections
    if not expected_sections or any(
        int(np.asarray(section.get("r", []), dtype=float).size) != verify_points
        for section in expected_sections.values()
    ):
        expected_sections = tabulated_buckingham_reference_sections(
            spec,
            npoints=verify_points,
        )
    table_stage_dir = verify_dir / "table"
    realized = _pair_write_potential_curves(
        runner,
        config,
        stage_dir=table_stage_dir,
        potential_lines=table_potential_lines,
        spec=spec,
        npoints=verify_points,
        output_name="realized.table",
        log_name="log_pairwrite_table.lammps",
    )
    critical_radii: dict[str, list[dict[str, Any]]] = {}
    for pair in spec.get("pairs", []):
        section = str(pair.get("section", ""))
        entries: list[dict[str, Any]] = []
        for name in ("r_in", "r_out", "buck_cutoff", "coul_cutoff", "pair_cutoff"):
            value = pair.get(name, None)
            if value is None:
                continue
            radius = float(value)
            if math.isfinite(radius):
                entries.append({"name": name, "r": radius})
        if section:
            critical_radii[section] = entries
    cmp = _compare_pair_table_sections(
        expected_sections,
        realized["sections"],
        rel_tol=float(core.table_verify_rel_tol),
        abs_tol_frac=float(core.table_verify_abs_tol_frac),
        critical_radii_by_section=critical_radii,
    )
    warnings = list(realized["warnings"])
    raw_filename = spec.get("filename", "")
    source_table = table_stage_dir / validated_lammps_localized_filename(
        "buckingham_core.table" if raw_filename == "" else raw_filename,
        field_name="tabulated core filename",
    )
    warning_audit = _audit_lammps_inflection_warnings(
        spec=spec,
        table_path=source_table,
        observed_warnings=warnings,
    )
    advisory_warnings = list(warning_audit.get("advisory_warnings", []) or [])
    blocking_warnings = list(warning_audit.get("blocking_warnings", []) or [])
    self_consistency = _table_force_energy_consistency_report(
        realized["sections"],
        rel_tol=float(core.table_verify_rel_tol),
        abs_tol_frac=float(core.table_verify_abs_tol_frac),
    )
    ok = bool(cmp.get("passed", False)) and bool(self_consistency.get("passed", False))
    if (
        bool(getattr(core, "table_require_warning_free", True))
        and blocking_warnings
    ):
        ok = False
    report = {
        "passed": ok,
        "warnings": warnings,
        "advisory_warnings": advisory_warnings,
        "blocking_warnings": blocking_warnings,
        "warning_audit": warning_audit,
        "comparison": cmp,
        "self_consistency": self_consistency,
        "verify_points": verify_points,
        "configured_verify_points": int(core.table_verify_points),
        "coverage": "all_table_knots_and_interval_midpoints",
        "coverage_proof": {
            "table_knots": int(spec["points"]),
            "verification_intervals": int(verify_points - 1),
            "required_interval_divisor": int(2 * (int(spec["points"]) - 1)),
            "divides_exactly": bool(
                (verify_points - 1) % (2 * (int(spec["points"]) - 1)) == 0
            ),
            "configured_value_rounded_up": bool(
                verify_points != int(core.table_verify_points)
            ),
        },
        "rel_tol": float(core.table_verify_rel_tol),
        "abs_tol_frac": float(core.table_verify_abs_tol_frac),
    }
    (verify_dir / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report



def _materialize_generated_tabulated_core_source(
    *,
    outdir: Path,
    spec: dict[str, Any],
) -> Path:
    potdir = Path(outdir) / "preflight" / "potential_override"
    ensure_dir(potdir)
    raw_filename = spec.get("filename", "")
    fname = validated_lammps_localized_filename(
        "buckingham_core.table" if raw_filename == "" else raw_filename,
        field_name="tabulated core filename",
    )
    out = potdir / fname
    write_tabulated_buckingham_core_table(out, spec)
    return out


def _tabulation_candidate_stability_status(candidate: Mapping[str, Any]) -> str:
    """Return the explicit tri-state stability outcome for one candidate."""

    status = str(candidate.get("stability_status", "") or "").strip().lower()
    if status in {"not_run", "pass", "fail"}:
        return status
    # Backward-compatible rendering for reports produced before the explicit
    # status field existed.
    stability_value = candidate.get("stability_ok", None)
    if stability_value is None:
        return "not_run"
    return "pass" if bool(stability_value) else "fail"


def _tabulation_candidate_verification_status(candidate: Mapping[str, Any]) -> str:
    """Return an explicit verification outcome, including execution errors."""

    status = str(candidate.get("verification_status", "") or "").strip().lower()
    if status in {"not_run", "pass", "fail", "error"}:
        return status
    if bool(candidate.get("verify_passed", False)):
        return "pass"
    if candidate.get("comparison") is not None or candidate.get("self_consistency") is not None:
        return "fail"
    if str(candidate.get("error", "") or "").strip():
        return "error"
    return "not_run"


def _finite_candidate_metric(value: Any) -> float:
    """Return a finite candidate metric, or +inf for absent/invalid values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if math.isfinite(number) else float("inf")


def _format_candidate_metric(metrics: Mapping[str, Any], key: str) -> str:
    """Distinguish unavailable metrics from explicitly non-finite results."""

    if key not in metrics or metrics.get(key) is None:
        return "n/a"
    try:
        number = float(metrics[key])
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(number):
        return "nonfinite"
    return f"{number:.3g}"


def _tabulation_candidate_error_headline(candidate: Mapping[str, Any]) -> str:
    """Extract a concise root-error line while retaining full details in JSON."""

    raw = str(candidate.get("error", "") or "").strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    preferred = next(
        (
            line
            for line in lines
            if re.match(r"^(?:error|fatal)(?::|\b)", line, flags=re.IGNORECASE)
        ),
        lines[0],
    )
    headline = re.sub(r"\s+", " ", preferred).strip()
    limit = 320
    if len(headline) > limit:
        headline = headline[: limit - 3].rstrip() + "..."
    return headline


def _record_tabulation_candidate_error(
    candidate: dict[str, Any],
    error: Exception,
    *,
    failure_stage: str,
    stability_attempted: bool,
) -> None:
    """Record a candidate exception without conflating it with a metric fail."""

    stage = str(failure_stage).strip() or "unknown"
    candidate["error"] = str(error)
    candidate["error_type"] = type(error).__name__
    candidate["failure_stage"] = stage
    if stage == "verification":
        candidate["verification_status"] = "error"
    if bool(stability_attempted):
        candidate["stability_ok"] = False
        candidate["stability_status"] = "fail"
    candidate["passed"] = False


def _tabulation_candidate_sort_key(
    candidate: dict[str, Any],
) -> tuple[float, ...]:
    comparison = dict((candidate.get("comparison") or {}).get("overall", {}) or {})
    warnings = list(
        candidate.get("blocking_warnings", candidate.get("warnings", [])) or []
    )
    verification_status = _tabulation_candidate_verification_status(candidate)
    verify_ok = verification_status == "pass"
    stability_ok = _tabulation_candidate_stability_status(candidate) == "pass"
    comparison_ok = bool(
        isinstance(candidate.get("comparison"), Mapping)
        and bool((candidate.get("comparison") or {}).get("passed", False))
    )
    self_consistency_ok = bool(
        isinstance(candidate.get("self_consistency"), Mapping)
        and bool((candidate.get("self_consistency") or {}).get("passed", False))
    )
    max_energy = _finite_candidate_metric(comparison.get("max_energy_ratio"))
    max_force = _finite_candidate_metric(comparison.get("max_force_ratio"))
    max_self = _finite_candidate_metric(
        ((candidate.get("self_consistency") or {}).get("overall", {}) or {}).get(
            "max_force_ratio"
        )
    )
    # Diagnose the candidate closest to physical acceptance.  A table that
    # passes the independently evaluated curve and work-consistency gates but
    # has one warning-classification failure is more informative than a
    # warning-free fallback that fails both numerical gates.  The previous
    # scalar warning penalty inverted that ordering and described the coarse
    # linear fallback as "best" in otherwise valid hybrid cases.
    numerical_gate_failures = float(
        int(not comparison_ok) + int(not self_consistency_ok)
    )
    return (
        0.0 if bool(candidate.get("passed", False)) else 1.0,
        0.0 if verify_ok else 1.0,
        0.0 if not str(candidate.get("error", "") or "").strip() else 1.0,
        numerical_gate_failures,
        0.0 if not warnings else 1.0,
        0.0 if stability_ok else 1.0,
        max(max_energy, max_force, max_self),
        max_energy + max_force + max_self,
        _finite_candidate_metric(candidate.get("table_points")),
        {
            "analytic": 0.0,
            "fd_consistent": 1.0,
        }.get(str(candidate.get("force_mode", "")), 2.0),
        0.0 if str(candidate.get("table_style", "")) == "spline" else 1.0,
    )


def _format_tabulation_candidate_summary(candidate: Mapping[str, Any]) -> str:
    comparison = dict((candidate.get("comparison") or {}).get("overall", {}) or {})
    warnings = list(
        candidate.get("blocking_warnings", candidate.get("warnings", [])) or []
    )
    advisory_warnings = list(candidate.get("advisory_warnings", []) or [])
    verification_status = _tabulation_candidate_verification_status(candidate)
    stability_status = _tabulation_candidate_stability_status(candidate)
    comparison_raw = candidate.get("comparison")
    consistency_raw = candidate.get("self_consistency")
    curve_status = (
        "not_run"
        if not isinstance(comparison_raw, Mapping)
        else ("pass" if bool(comparison_raw.get("passed", False)) else "fail")
    )
    work_status = (
        "not_run"
        if not isinstance(consistency_raw, Mapping)
        else ("pass" if bool(consistency_raw.get("passed", False)) else "fail")
    )
    warning_status = "pass" if not warnings else "fail"
    parts = [
        f"mode={candidate.get('force_mode', '?')}",
        f"interpolation={candidate.get('table_style', '?')}",
        f"table_points={candidate.get('table_points', '?')}",
        f"verify_points={candidate.get('verify_points', '?')}",
        f"verify={verification_status}",
        f"stability={stability_status}",
        f"curve={curve_status}",
        f"work={work_status}",
        f"warning_audit={warning_status}",
        f"max_energy_ratio={_format_candidate_metric(comparison, 'max_energy_ratio')}",
        f"max_force_ratio={_format_candidate_metric(comparison, 'max_force_ratio')}",
        f"warnings={len(warnings)}",
        f"advisory_warnings={len(advisory_warnings)}",
    ]
    failure_stage = str(candidate.get("failure_stage", "") or "").strip()
    if failure_stage:
        parts.append(f"failure_stage={failure_stage}")
    error_type = str(candidate.get("error_type", "") or "").strip()
    if error_type:
        parts.append(f"error_type={error_type}")
    headline = _tabulation_candidate_error_headline(candidate)
    if headline:
        parts.append(f"error={headline}")
    return ", ".join(parts)


def _write_tabulation_refinement_report(
    *,
    outdir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    verify_dir = Path(outdir) / "preflight" / "table_verify"
    ensure_dir(verify_dir)
    json_path = verify_dir / "refinement_report.json"
    txt_path = verify_dir / "refinement_summary.txt"
    json_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    summary_lines: list[str] = []
    summary_lines.append(f"status: {report.get('status', 'unknown')}")
    summary_lines.append(f"fallback_to_analytic: {bool(report.get('fallback_to_analytic', False))}")
    source_equivalence = report.get("source_equivalence", None)
    if isinstance(source_equivalence, Mapping):
        source_overall = dict(
            (source_equivalence.get("comparison") or {}).get("overall", {}) or {}
        )
        summary_lines.append(
            "source_equivalence: "
            f"passed={bool(source_equivalence.get('passed', False))}, "
            f"r_min={source_equivalence.get('r_min', 'n/a')}, "
            f"max_energy_ratio={_format_candidate_metric(source_overall, 'max_energy_ratio')}, "
            f"max_force_ratio={_format_candidate_metric(source_overall, 'max_force_ratio')}"
        )
    best = report.get("best_candidate", None)
    if isinstance(best, Mapping):
        summary_lines.append("best_candidate: " + _format_tabulation_candidate_summary(best))
    accepted = report.get("accepted_candidate", None)
    if isinstance(accepted, Mapping):
        summary_lines.append("accepted_candidate: " + _format_tabulation_candidate_summary(accepted))
    reason = str(report.get("reason", "") or "").strip()
    if reason:
        summary_lines.append("reason: " + reason)
    cand_list = list(report.get("candidates", []) or [])
    if cand_list:
        summary_lines.append("candidates:")
        for cand in cand_list:
            if isinstance(cand, Mapping):
                summary_lines.append("  - " + _format_tabulation_candidate_summary(cand))
                error_headline = _tabulation_candidate_error_headline(cand)
                if error_headline:
                    summary_lines.append("    error: " + error_headline)
    txt_path.write_text("\n".join(summary_lines) + "\n")
    return json_path, txt_path


def _append_condensed_preflight_warning(outdir: Path, message: str) -> None:
    try:
        CondensedProgressLog(Path(outdir) / "condensed.log").warn("preflight", str(message))
    except Exception:
        pass


def _append_condensed_preflight_info(outdir: Path, message: str) -> None:
    try:
        CondensedProgressLog(Path(outdir) / "condensed.log").info("preflight", str(message))
    except Exception:
        pass


def _append_preflight_warning(outdir: Path, warnings_list: list[str], message: str) -> None:
    msg = str(message).strip()
    if not msg:
        return
    if msg not in warnings_list:
        warnings_list.append(msg)
    _append_condensed_preflight_warning(outdir, msg)


def _fmt_dt_values(values: Sequence[float]) -> str:
    vals = [float(x) for x in values]
    return "[" + ", ".join(f"{v:g}" for v in vals) + "]"


def _make_timestep_fallback_warning(
    *,
    context: str,
    source: str,
    tried: Sequence[float],
    selected: Optional[float],
) -> str:
    tried_vals = [float(x) for x in tried]
    tried_txt = _fmt_dt_values(tried_vals)
    if selected is None:
        return (
            f"{context}: timestep fallback activated; source={source}; tried={tried_txt}; "
            "no stable candidate passed"
        )
    return (
        f"{context}: timestep fallback activated; source={source}; tried={tried_txt}; "
        f"selected dt={float(selected):g} after larger candidate(s) failed"
    )


def _tabulated_realizations() -> tuple[tuple[str, str], ...]:
    """Ordered interpolation/force candidates for safe table refinement.

    Cubic interpolation in LAMMPS's native r-squared coordinate resolves the
    singularly curved ZBL end without requiring a million-knot linear table.
    Analytic knot forces preserve the intended Hamiltonian.  Linear and the
    legacy all-knot finite-difference realization remain fail-closed
    compatibility fallbacks and must pass the same realized-curve verification
    before use.
    """

    return (
        ("spline", "analytic"),
        ("linear", "analytic"),
        ("linear", "fd_consistent"),
    )


def _tabulated_verification_points(
    *, configured_points: int, table_points: int
) -> int:
    """Cover every RSQ table knot and every interval midpoint.

    A verification grid contains all runtime knots and all interval midpoints
    iff its number of RSQ intervals is an integer multiple of
    ``2*(N-1)``.  Round a larger user request *up* to the next such grid;
    retaining an arbitrary larger value would generally lose exact alignment
    while falsely claiming complete coverage.
    """

    configured = int(configured_points)
    knots = int(table_points)
    if configured < 2:
        raise ValueError("table verification requires at least two points")
    if knots < 2:
        raise ValueError("tabulated Buckingham tables require at least two knots")
    base_intervals = 2 * (knots - 1)
    requested_intervals = max(1, configured - 1)
    multiplier = max(1, int(math.ceil(requested_intervals / base_intervals)))
    return multiplier * base_intervals + 1


def _resolved_pair_joins_from_tabulated_metadata(
    potential_lines: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Return the requested and realized join interval for every table pair.

    The table metadata is the execution authority: the safe join resolver may
    move one pair's interval beyond the global calibration request in order to
    splice onto a connected repulsive branch.  Reporting only that global
    request would therefore understate how much of the original potential was
    regularized.
    """

    spec = _parse_tabulated_core_spec(potential_lines)
    if spec is None:
        raise ValueError("accepted tabulated core is missing its metadata block")
    raw_pairs = spec.get("pairs", [])
    if not isinstance(raw_pairs, Sequence) or isinstance(
        raw_pairs, (str, bytes, bytearray)
    ):
        raise ValueError("accepted tabulated core metadata has a malformed pair list")

    rows: list[dict[str, Any]] = []
    required_radii = (
        "requested_r_in",
        "requested_r_out",
        "r_in",
        "r_out",
    )
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, Mapping):
            raise ValueError(
                f"accepted tabulated core pair metadata at index {index} is not an object"
            )
        pair_raw = raw_pair.get("pair")
        species_raw = raw_pair.get("species")
        if (
            not isinstance(pair_raw, Sequence)
            or isinstance(pair_raw, (str, bytes, bytearray))
            or len(pair_raw) != 2
        ):
            raise ValueError(
                f"accepted tabulated core pair metadata at index {index} has no two-type pair"
            )
        if (
            not isinstance(species_raw, Sequence)
            or isinstance(species_raw, (str, bytes, bytearray))
            or len(species_raw) != 2
        ):
            raise ValueError(
                f"accepted tabulated core pair metadata at index {index} has no two-species label"
            )

        radii: dict[str, float] = {}
        for key in required_radii:
            value = float(raw_pair[key])
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"accepted tabulated core pair metadata {raw_pair.get('section', index)!r} "
                    f"has invalid {key}={raw_pair.get(key)!r}"
                )
            radii[key] = value
        if not (
            radii["requested_r_in"] < radii["requested_r_out"]
            and radii["r_in"] < radii["r_out"]
        ):
            raise ValueError(
                f"accepted tabulated core pair metadata {raw_pair.get('section', index)!r} "
                "has a non-increasing requested or resolved join interval"
            )

        rows.append(
            {
                "section": str(raw_pair.get("section", f"pair_{index}")),
                "pair": [int(pair_raw[0]), int(pair_raw[1])],
                "species": [str(species_raw[0]), str(species_raw[1])],
                "requested_r_in": radii["requested_r_in"],
                "requested_r_out": radii["requested_r_out"],
                "resolved_r_in": radii["r_in"],
                "resolved_r_out": radii["r_out"],
                "resolver_adjusted": not (
                    math.isclose(
                        radii["requested_r_in"],
                        radii["r_in"],
                        rel_tol=0.0,
                        abs_tol=1.0e-14 * max(1.0, radii["r_in"]),
                    )
                    and math.isclose(
                        radii["requested_r_out"],
                        radii["r_out"],
                        rel_tol=0.0,
                        abs_tol=1.0e-14 * max(1.0, radii["r_out"]),
                    )
                ),
            }
        )
    if not rows:
        raise ValueError("accepted tabulated core metadata contains no pair joins")
    return tuple(rows)


def _core_join_report_from_execution_lines(
    potential_lines: Sequence[str],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Classify legacy radii from the actual accepted potential representation."""

    if _parse_tabulated_core_spec(potential_lines) is None:
        # Analytic hybrid/overlay paths apply the one calibrated interval
        # globally, so the historical fields are the realized join.
        return "applied_global_join", ()
    return (
        "global_calibration_request",
        _resolved_pair_joins_from_tabulated_metadata(potential_lines),
    )


def _maybe_apply_core_repulsion(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    outdir: Path,
    *,
    T_test: float,
) -> tuple[Optional[list[str]], CoreRepulsionResult, float]:
    """Maybe apply core."""
    core = config.kim.core_repulsion
    dt0 = float(config.md.timestep)

    # disabled return immediately
    if not core.enabled:
        return (
            None,
            CoreRepulsionResult(
                enabled=False,
                applied=False,
                style=str(core.style),
                base_pair_style=None,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=True,
                note="disabled",
            ),
            dt0,
        )

    try:
        if isinstance(config.kim, KimConfig):
            base_cmds = _kim_extract_commands(runner, config, input_data, outdir)
        else:
            # kim potentials directly
            base_cmds = potential_default_lines(config.kim)
    except Exception as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=None,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=f"failed to obtain base potential commands: {e}",
            ),
            dt0,
        )

    base_pair_style = None
    for ln in base_cmds:
        if ln.strip().startswith("pair_style"):
            tt = ln.split()
            if len(tt) >= 2:
                base_pair_style = tt[1]
            break

    # Buckingham may be a direct style or an additive substyle under
    # hybrid/overlay.  Dispatch through the same exact parser used to construct
    # the replacement table.  In particular, a command block containing
    # multiple pair_style definitions, an unsupported hybrid composition, or
    # any other Buckingham-like but lossy representation must fail here rather
    # than being misclassified as a non-Buckingham potential with autocore
    # silently skipped.
    try:
        compatibility = inspect_buckingham_core_compatibility(base_cmds)
    except Exception as e:
        reason = (
            "Buckingham autocore compatibility audit failed before any "
            f"potential execution: {e}"
        )
        report = {
            "status": "rejected_fail_closed",
            "fallback_to_analytic": False,
            "reason": reason,
            "accepted_candidate": None,
            "best_candidate": None,
            "candidates": [],
        }
        _write_tabulation_refinement_report(outdir=outdir, report=report)
        raise PreflightError(reason) from e
    contains_buckingham = bool(compatibility.get("is_buckingham", False))
    if base_pair_style is None or not contains_buckingham:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=True,
                note="base potential is not Buckingham-like; core overlay not applied",
            ),
            dt0,
        )

    species = (
        list(config.kim.interactions)
        if isinstance(config.kim.interactions, list) and len(config.kim.interactions) > 0
        else []
    )
    if len(species) == 0:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note="kim.interactions must be a non-empty list for core overlay parameterisation",
            ),
            dt0,
        )

    # candidate timesteps declared
    # fallback
    # explicitly enabled
    try:
        dt_list, dt_source = _core_dt_candidates(config)
    except ValueError as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=str(e),
            ),
            dt0,
        )

    try:
        r_nn = _read_nn_median_from_datafile(
            input_data,
            atom_style=str(config.md.atom_style),
            units_style=str(config.kim.user_units),
        )
    except Exception as e:
        return (
            None,
            CoreRepulsionResult(
                enabled=True,
                applied=False,
                style=str(core.style),
                base_pair_style=base_pair_style,
                r_inner=None,
                r_outer=None,
                attempts=0,
                success=False,
                note=f"failed to estimate nearest-neighbour distance: {e}",
            ),
            dt0,
        )

    # cutoff guess structure
    style = str(core.style).strip().lower()
    # dimensionless target configured
    # lammps style repulsive
    units_style = str(config.kim.user_units).strip().lower()
    distance_factor = length_from_angstrom_factor(units_style)
    r_out_min = float(core.r_out_min) * float(distance_factor)
    r_out_max = float(core.r_out_max) * float(distance_factor)
    table_r_min = float(core.table_r_min) * float(distance_factor)
    r_out = float(core.r_out_factor) * float(r_nn)
    r_out = max(r_out_min, min(r_out_max, r_out))

    kB = boltzmann_constant_native(units_style)
    u_target_native = float(core.u_target_kT) * float(kB) * float(T_test)

    charges = (
        None
        if getattr(config.structure, "charges", None) is None
        else dict(config.structure.charges)
    )
    runtime_charge_audit: Optional[dict[str, Any]] = None
    has_bonded_topology = _datafile_has_bonded_topology(input_data)
    if style == "zbl" and _potential_contains_coulomb(base_cmds):
        try:
            runtime_charge_audit = _validate_tabulated_coulomb_runtime_charges(
                input_data,
                atom_style=str(config.md.atom_style),
                species=species,
                configured_charges=charges,
                units_style=units_style,
                potential_commands=base_cmds,
                require_explicit_set_for_present_types=bool(
                    isinstance(config.kim, KimConfig)
                    and config.structure.generate is not None
                    and charges is None
                ),
            )
            charges = {
                str(symbol): float(value)
                for symbol, value in dict(
                    runtime_charge_audit["effective_charges_e"]
                ).items()
            }
        except Exception as e:
            reason = f"failed fixed-charge audit before Coulombic autocore initialization: {e}"
            report = {
                "status": "rejected_fail_closed",
                "fallback_to_analytic": False,
                "reason": reason,
                "accepted_candidate": None,
                "best_candidate": None,
                "candidates": [],
            }
            _write_tabulation_refinement_report(outdir=outdir, report=report)
            raise PreflightError(reason) from e

    tabulated_gewald: Optional[float] = None
    if style == "zbl" and _potential_requires_gewald(base_cmds):
        try:
            tabulated_gewald = _probe_original_potential_gewald(
                runner,
                config,
                input_data,
                outdir=outdir,
                potential_lines=base_cmds,
            )
        except Exception as e:
            reason = f"failed to resolve G-Ewald from the original-potential run-0 probe: {e}"
            report = {
                "status": "rejected_fail_closed",
                "fallback_to_analytic": False,
                "reason": reason,
                "accepted_candidate": None,
                "best_candidate": None,
                "candidates": [],
            }
            _write_tabulation_refinement_report(outdir=outdir, report=report)
            if isinstance(e, PreflightError):
                raise PreflightError(reason) from e
            raise PreflightError(reason) from e

    # loop
    for k in range(1, int(core.max_attempts) + 1):
        r_in = float(core.r_in_factor) * float(r_out)

        try:
            if style == "zbl":
                # Never execute the analytic Buckingham+ZBL sum.  It remains
                # unbounded below as r -> 0 regardless of a finite ZBL overlay.
                # Construction below goes directly to the C2 replacement.
                pot_lines = []
                base_style = str(base_pair_style)
            elif style == "lj_repulsive":
                pot_lines, base_style = _rewrite_for_hybrid_overlay_lj_repulsive(
                    base_cmds, species=species, r_in=r_in, r_out=r_out, u_target_eV=u_target_native
                )
            else:
                raise ValueError(f"Unsupported core_repulsion.style: {style}")
        except Exception as e:
            return (
                None,
                CoreRepulsionResult(
                    enabled=True,
                    applied=False,
                    style=str(core.style),
                    base_pair_style=base_pair_style,
                    r_inner=None,
                    r_outer=None,
                    attempts=k,
                    success=False,
                    note=f"failed to rewrite KIM commands: {e}",
                ),
                dt0,
            )

        # Legacy bounded LJ-overlay calibration retains its direct timestep
        # scan.  ZBL candidates are scanned only after table materialization
        # and strict pair-write verification below.
        dt_selected: Optional[float] = None
        dt_tried: list[float] = []
        if style == "zbl":
            # Construction sentinel only; no timestep has passed dynamics yet.
            dt_selected = float(dt_list[0])
        else:
            for dt in dt_list:
                dt_tried.append(float(dt))
                ok = _run_stability_test(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=pot_lines,
                    temperature=float(T_test),
                    timestep=float(dt),
                    label=f"try{k}_dt{dt:.6g}",
                )
                if ok:
                    dt_selected = float(dt)
                    break

        if dt_selected is not None:
            final_lines = pot_lines
            note = (
                "constructing bounded C2 candidate before timestep calibration"
                if style == "zbl"
                else f"calibrated at T={T_test:g} K with dt={dt_selected:g}"
            )
            if style != "zbl" and len(dt_tried) > 1:
                fb_msg = _make_timestep_fallback_warning(
                    context="core-repulsion calibration",
                    source=dt_source,
                    tried=dt_tried,
                    selected=float(dt_selected),
                )
                _append_condensed_preflight_warning(outdir, fb_msg)
                note += f" ({fb_msg})"
            # A Buckingham+ZBL sum is still unbounded below as r -> 0.  ZBL
            # production therefore always uses the regularized replacement
            # table, irrespective of the legacy ``tabulate`` toggle.
            if style == "zbl":
                gewald = tabulated_gewald
                refinement_candidates: list[dict[str, Any]] = []
                best_candidate: Optional[dict[str, Any]] = None
                accepted_candidate: Optional[dict[str, Any]] = None
                accepted_lines: Optional[list[str]] = None
                source_equivalence_report: Optional[dict[str, Any]] = None
                summary_reason = ""
                point_schedule: list[int] = []
                tp = int(core.table_points)
                tp_max = int(core.table_points_max)
                while True:
                    point_schedule.append(int(tp))
                    if tp >= tp_max:
                        break
                    tp = min(tp_max, max(int(tp) + 1, int(math.ceil(float(tp) * 2.0))))
                try:
                    seed_lines = build_tabulated_buckingham_core_lines(
                        base_cmds,
                        species=species,
                        units_style=units_style,
                        r_in=float(r_in),
                        r_out=float(r_out),
                        table_points=int(core.table_points),
                        table_filename=str(core.table_filename),
                        table_r_min=float(table_r_min),
                        charges=charges,
                        gewald=gewald,
                        has_bonded_topology=has_bonded_topology,
                    )
                    if runtime_charge_audit is not None:
                        seed_lines = update_tabulated_core_metadata_lines(
                            seed_lines,
                            runtime_charge_audit=runtime_charge_audit,
                        )
                    seed_spec = _parse_tabulated_core_spec(seed_lines)
                    if seed_spec is None:
                        raise ValueError("failed to parse tabulated-core metadata after construction")
                    reference_data = {
                        "sections": tabulated_buckingham_reference_sections(
                            seed_spec,
                            npoints=int(core.table_verify_points),
                        ),
                        "warnings": [],
                    }
                    source_equivalence_report = _audit_original_potential_above_core_joins(
                        runner,
                        config,
                        outdir=outdir,
                        source_potential_lines=base_cmds,
                        spec=seed_spec,
                    )
                except Exception as e:
                    summary_reason = f"failed to initialise tabulated Buckingham verification: {e}"
                    report = {
                        "status": "rejected_fail_closed",
                        "fallback_to_analytic": False,
                        "reason": summary_reason,
                        "accepted_candidate": None,
                        "best_candidate": None,
                        "candidates": [],
                        "source_equivalence": source_equivalence_report,
                    }
                    _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                    try:
                        summary_rel = str(summary_path.relative_to(outdir))
                    except Exception:
                        summary_rel = str(summary_path)
                    raise PreflightError(
                        "C2-regularized Buckingham-ZBL table initialization failed; refusing the "
                        "unbounded analytic hybrid/overlay fallback. "
                        f"{summary_reason}; see {summary_rel}"
                    ) from e
                else:
                    for table_style, force_mode in _tabulated_realizations():
                        for table_points in point_schedule:
                            candidate: dict[str, Any] = {
                                "force_mode": str(force_mode),
                                "table_style": str(table_style),
                                "table_points": int(table_points),
                                "verify_points": _tabulated_verification_points(
                                    configured_points=int(core.table_verify_points),
                                    table_points=int(table_points),
                                ),
                                "verify_passed": False,
                                "verification_status": "not_run",
                                # ``None`` means verification did not permit a
                                # stability run.  Do not report that as a
                                # dynamical failure: False is reserved for a
                                # stability test that was actually attempted.
                                "stability_ok": None,
                                "stability_status": "not_run",
                                "passed": False,
                                "warnings": [],
                                "advisory_warnings": [],
                                "blocking_warnings": [],
                            }
                            candidate_lines: Optional[list[str]] = None
                            last_report: Optional[dict[str, Any]] = None
                            stability_attempted = False
                            candidate_stage = "construction"
                            try:
                                candidate_lines = build_tabulated_buckingham_core_lines(
                                    base_cmds,
                                    species=species,
                                    units_style=units_style,
                                    r_in=float(r_in),
                                    r_out=float(r_out),
                                    table_points=int(table_points),
                                    table_filename=str(core.table_filename),
                                    table_r_min=float(table_r_min),
                                    charges=charges,
                                    gewald=gewald,
                                    has_bonded_topology=has_bonded_topology,
                                    table_style=str(table_style),
                                )
                                candidate_lines = update_tabulated_core_metadata_lines(
                                    candidate_lines,
                                    force_mode=str(force_mode),
                                    table_style=str(table_style),
                                    include_fprime=True,
                                    runtime_charge_audit=runtime_charge_audit,
                                )
                                spec = _parse_tabulated_core_spec(candidate_lines)
                                if spec is None:
                                    raise ValueError("failed to parse tabulated-core metadata after construction")
                                candidate_stage = "materialization"
                                table_path = _materialize_generated_tabulated_core_source(outdir=outdir, spec=spec)
                                table_identity = stable_file_identity(
                                    table_path,
                                    reject_final_symlink=True,
                                )
                                candidate_lines = update_tabulated_core_metadata_lines(
                                    candidate_lines,
                                    generated_by="vitriflow_generated",
                                    sha256=str(table_identity["sha256"]),
                                    size_bytes=int(table_identity["size_bytes"]),
                                    source_relpath=str(
                                        Path("preflight")
                                        / "potential_override"
                                        / table_path.name
                                    ),
                                    force_mode=str(force_mode),
                                    table_style=str(table_style),
                                    include_fprime=True,
                                )
                                spec = _parse_tabulated_core_spec(candidate_lines)
                                if spec is None:
                                    raise ValueError("failed to parse tabulated-core metadata after source materialisation")
                                candidate_stage = "verification"
                                last_report = _verify_tabulated_core_against_source(
                                    runner,
                                    config,
                                    outdir=outdir,
                                    table_potential_lines=candidate_lines,
                                    reference_sections=reference_data["sections"],
                                    spec=spec,
                                )
                                candidate["warnings"] = list(last_report.get("warnings", []))
                                candidate["advisory_warnings"] = list(
                                    last_report.get("advisory_warnings", []) or []
                                )
                                candidate["blocking_warnings"] = list(
                                    last_report.get("blocking_warnings", []) or []
                                )
                                candidate["warning_audit"] = dict(
                                    last_report.get("warning_audit", {}) or {}
                                )
                                candidate["comparison"] = dict(last_report.get("comparison", {}) or {})
                                candidate["self_consistency"] = dict(last_report.get("self_consistency", {}) or {})
                                candidate["verify_points"] = int(
                                    last_report.get(
                                        "verify_points", candidate["verify_points"]
                                    )
                                )
                                candidate["verify_passed"] = bool(last_report.get("passed", False))
                                candidate["verification_status"] = (
                                    "pass" if candidate["verify_passed"] else "fail"
                                )
                                if bool(candidate["verify_passed"]):
                                    stability_attempted = True
                                    candidate_stage = "stability"
                                    candidate_dt_tried: list[float] = []
                                    candidate["stability_ok"] = False
                                    candidate["selected_timestep"] = None
                                    for candidate_dt in dt_list:
                                        candidate_dt_tried.append(float(candidate_dt))
                                        stable = bool(_run_stability_test(
                                            runner,
                                            config,
                                            input_data,
                                            outdir=outdir,
                                            potential_lines=candidate_lines,
                                            temperature=float(T_test),
                                            timestep=float(candidate_dt),
                                            label=(
                                                f"try{k}_dt{candidate_dt:.6g}_table_"
                                                f"{str(table_style)}_"
                                                f"{str(force_mode).replace('-', '_')}_{int(table_points)}"
                                            ),
                                        ))
                                        if stable:
                                            candidate["stability_ok"] = True
                                            candidate["selected_timestep"] = float(candidate_dt)
                                            break
                                    candidate["dt_candidates_tried"] = candidate_dt_tried
                                    candidate["stability_status"] = (
                                        "pass" if candidate["stability_ok"] else "fail"
                                    )
                                candidate["passed"] = bool(candidate["verify_passed"] and candidate["stability_ok"])
                            except Exception as e:
                                _record_tabulation_candidate_error(
                                    candidate,
                                    e,
                                    failure_stage=candidate_stage,
                                    stability_attempted=stability_attempted,
                                )
                            refinement_candidates.append(candidate)
                            if best_candidate is None or _tabulation_candidate_sort_key(candidate) < _tabulation_candidate_sort_key(best_candidate):
                                best_candidate = dict(candidate)
                            if bool(candidate.get("passed", False)) and candidate_lines is not None:
                                accepted_candidate = dict(candidate)
                                accepted_lines = list(candidate_lines)
                                break
                        if accepted_candidate is not None:
                            break

                    if accepted_candidate is not None and accepted_lines is not None:
                        final_lines = accepted_lines
                        selected_candidate_dt = accepted_candidate.get("selected_timestep", None)
                        if selected_candidate_dt is None:
                            raise PreflightError(
                                "accepted tabulated autocore candidate has no verified stability timestep"
                            )
                        dt_selected = float(selected_candidate_dt)
                        dt_tried = [
                            float(value)
                            for value in list(accepted_candidate.get("dt_candidates_tried", []) or [])
                        ]
                        note = (
                            f"bounded C2 table calibrated at T={T_test:g} K with "
                            f"dt={dt_selected:g}"
                        )
                        if len(dt_tried) > 1:
                            fb_msg = _make_timestep_fallback_warning(
                                context="bounded core-repulsion calibration",
                                source=dt_source,
                                tried=dt_tried,
                                selected=float(dt_selected),
                            )
                            _append_condensed_preflight_warning(outdir, fb_msg)
                            note += f" ({fb_msg})"
                        overall = dict((accepted_candidate.get("comparison") or {}).get("overall", {}) or {})
                        summary_reason = "accepted verified tabulated real-space table"
                        report = {
                            "status": "tabulated_verified",
                            "fallback_to_analytic": False,
                            "reason": summary_reason,
                            "accepted_candidate": accepted_candidate,
                            "best_candidate": best_candidate,
                            "candidates": refinement_candidates,
                            "source_equivalence": source_equivalence_report,
                        }
                        _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                        try:
                            summary_rel = str(summary_path.relative_to(outdir))
                        except Exception:
                            summary_rel = str(summary_path)
                        _append_condensed_preflight_info(
                            outdir,
                            "tabulated Buckingham real-space pair potential accepted after refinement; "
                            + _format_tabulation_candidate_summary(accepted_candidate)
                            + f"; see {summary_rel}",
                        )
                        note += (
                            " (tabulated Buckingham real-space pair potential accepted after refinement; "
                            f"pair_write-verified on {int(accepted_candidate.get('verify_points', core.table_verify_points))} RSQ points, "
                            f"interpolation={accepted_candidate.get('table_style', '?')}, "
                            f"force_mode={accepted_candidate.get('force_mode', '?')}, "
                            f"table_points={int(accepted_candidate.get('table_points', core.table_points))}, "
                            f"max_energy_ratio={float(overall.get('max_energy_ratio', 0.0)):.3g}, "
                            f"max_force_ratio={float(overall.get('max_force_ratio', 0.0)):.3g}; "
                            f"see {summary_rel})"
                        )
                    else:
                        all_errored_before_comparison = bool(refinement_candidates) and all(
                            bool(str(candidate.get("error", "") or "").strip())
                            and candidate.get("comparison") is None
                            for candidate in refinement_candidates
                        )
                        if all_errored_before_comparison:
                            summary_reason = (
                                "all tabulated Buckingham real-space pair-potential candidates "
                                "errored before energy/force comparison"
                            )
                        else:
                            summary_reason = (
                                "tabulated Buckingham real-space pair-potential refinement did not find a "
                                "candidate that passed strict energy/force verification and calibrated stability"
                            )
                        report = {
                            "status": "rejected_fail_closed",
                            "fallback_to_analytic": False,
                            "reason": summary_reason,
                            "accepted_candidate": None,
                            "best_candidate": best_candidate,
                            "candidates": refinement_candidates,
                            "source_equivalence": source_equivalence_report,
                        }
                        _json_path, summary_path = _write_tabulation_refinement_report(outdir=outdir, report=report)
                        try:
                            summary_rel = str(summary_path.relative_to(outdir))
                        except Exception:
                            summary_rel = str(summary_path)
                        best_txt = _format_tabulation_candidate_summary(best_candidate) if isinstance(best_candidate, dict) else "no viable candidate"
                        raise PreflightError(
                            "No C2-regularized Buckingham-ZBL table passed strict energy/force "
                            "verification and stability; refusing the unbounded analytic "
                            f"hybrid/overlay fallback. best candidate: {best_txt}; see {summary_rel}"
                        )

            pdir = outdir / "preflight" / "potential_override"
            ensure_dir(pdir)
            (pdir / "potential_lines.lmp").write_text("\n".join(final_lines) + "\n")
            # The accepted execution lines, not the legacy ``tabulate`` toggle,
            # are authoritative.  This remains compatible with analytic
            # hybrid/overlay callers while reporting every actual pair splice
            # whenever the accepted representation is a generated table.
            radii_role, resolved_pair_joins = (
                _core_join_report_from_execution_lines(final_lines)
            )
            return (
                final_lines,
                CoreRepulsionResult(
                    enabled=True,
                    applied=True,
                    style=str(core.style),
                    base_pair_style=str(base_style),
                    r_inner=float(r_in),
                    r_outer=float(r_out),
                    attempts=k,
                    success=True,
                    note=note,
                    r_inner_r_outer_role=radii_role,
                    requested_r_inner=float(r_in),
                    requested_r_outer=float(r_out),
                    join_radii_lammps_units_style=str(units_style),
                    resolved_pair_joins=resolved_pair_joins,
                ),
                float(dt_selected),
            )

        # calibration update
        # strengthen repulsion widen
        if style == "lj_repulsive":
            u_target_native = float(u_target_native) * float(getattr(core, "strength_grow_factor", 2.0))
        r_out = max(r_out_min, min(r_out_max, float(r_out) * float(core.grow_factor)))

    return (
        None,
        CoreRepulsionResult(
            enabled=True,
            applied=True,
            style=str(core.style),
            base_pair_style=str(base_pair_style),
            r_inner=None,
            r_outer=float(r_out),
            attempts=int(core.max_attempts),
            success=False,
            note=f"failed to find stable {style} core settings after {int(core.max_attempts)} attempts",
        ),
        float(dt_list[-1]) if dt_list else dt0,
    )


def _evaluate_thermo_table(
    tbl: dict[str, np.ndarray],
    *,
    T_target: float,
    P_target: float,
    ensemble: str,
    config: RunConfig,
) -> tuple[bool, dict[str, Any], float]:
    pf = config.autotune.preflight

    if "Temp" not in tbl:
        return False, {"reason": "Temp column missing"}, float("inf")
    T = np.asarray(tbl["Temp"], dtype=float)
    if len(T) < 3 or not np.all(np.isfinite(T)):
        return False, {"reason": "Temp invalid"}, float("inf")

    Tw = window_mean_stderr(T, start_fraction=0.5)
    eT = abs(Tw.mean - float(T_target)) / max(1.0, float(T_target))

    if float(np.nanmax(T)) > float(pf.max_temp_factor) * float(T_target):
        return False, {"reason": "Temp exploded", "T_max": float(np.nanmax(T))}, float("inf")

    ok = eT <= float(pf.temp_rel_tol)
    score = float(eT)
    details: dict[str, Any] = {
        "T_target": float(T_target),
        "T_mean": float(Tw.mean),
        "T_stderr": float(Tw.stderr),
        "T_rel_err": float(eT),
    }

    if ensemble == "npt":
        # pressure
        if "Press" in tbl:
            P = np.asarray(tbl["Press"], dtype=float)
            Pw = window_mean_stderr(P, start_fraction=0.5)
            eP = abs(Pw.mean - float(P_target))
            details.update(
                {
                    "P_target": float(P_target),
                    "P_mean": float(Pw.mean),
                    "P_stderr": float(Pw.stderr),
                    "P_abs_err": float(eP),
                }
            )
            if not np.isfinite(eP) or abs(float(Pw.mean)) > float(pf.max_press_abs):
                return False, {**details, "reason": "Press invalid/exploded"}, float("inf")
            # pressure convergence objective
            # score hard pass
            # rejecting otherwise stable
            # pressure relaxes slowly
            # cells enable hard
            # preflight pressure tolerance
            if bool(getattr(pf, "require_pressure_tolerance", False)) and eP > float(pf.press_abs_tol):
                ok = False
            score += float(eP) / max(1.0, float(pf.press_abs_tol))
        else:
            ok = False
            details["reason"] = "Press column missing"

        # volume runaway barostat
        V = None
        if "Vol" in tbl:
            V = np.asarray(tbl["Vol"], dtype=float)
        elif "Density" in tbl:
            den = np.asarray(tbl["Density"], dtype=float)
            if np.all(np.isfinite(den)) and float(np.nanmin(den)) > 0:
                V = 1.0 / den

        if V is not None and len(V) >= 3 and np.all(np.isfinite(V)):
            V0 = float(V[0])
            Vw = window_mean_stderr(V, start_fraction=0.5)
            Vmean = float(Vw.mean)
            if V0 > 0 and Vmean > 0:
                ratio = max(Vmean / V0, V0 / Vmean)
                details.update(
                    {
                        "V0": float(V0),
                        "V_mean": float(Vmean),
                        "V_ratio_tail_vs_start": float(ratio),
                    }
                )
                if ratio > float(pf.max_vol_ratio):
                    return False, {**details, "reason": "Volume drift too large"}, float("inf")
                score += float(max(0.0, ratio - 1.0)) / max(1.0, float(pf.max_vol_ratio) - 1.0)

    return ok, details, float(score)


def _run_candidate(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    *,
    outdir: Path,
    potential_lines: Optional[list[str]],
    md: MDConfig,
    temperature: float,
    name: str,
    equil_steps: int,
    run_steps: int,
) -> tuple[bool, dict[str, Any], float]:
    stage_dir = outdir / "preflight" / "thermo_scan" / name / f"T{int(round(temperature))}"
    ensure_dir(stage_dir)
    _localize_input_data_for_preflight(source=input_data, destination=stage_dir / "input.data")

    # preserve candidate timestep
    # preflight selection smaller
    dt_ref = float(config.md.timestep)
    equil_steps_s = scale_steps_for_timestep(int(equil_steps), dt_ref, float(md.timestep), min_steps=0)
    run_steps_s = scale_steps_for_timestep(int(run_steps), dt_ref, float(md.timestep), min_steps=1)

    stage = StageSpec(
        name=f"preflight_{name}",
        input_data=stage_dir / "input.data",
        output_data=stage_dir / "output.data",
        temperature_start=float(temperature),
        temperature_stop=float(temperature),
        pressure=float(md.pressure),
        equil_steps=int(equil_steps_s),
        run_steps=int(run_steps_s),
        seed=12345,
        replicate=(1, 1, 1),
        write_dump=False,
        dump_every=1000,
        msd_every=200,
        potential_lines=potential_lines,
    )
    try:
        prepare_potential_files(config.kim, stage_dir, potential_lines)
        cleanup = [stage_dir / "output.data", stage_dir / f"{stage.name}.msd.dat"]
        run_with_neighbor_skin_autotune(
            runner,
            lambda md_use: render_stage(config.kim, md_use, stage),
            stage_dir,
            md,
            log_name="log.lammps",
            timeout_sec=float(config.autotune.preflight.timeout_sec),
            cleanup_paths=cleanup,
        )
    except (ExternalCommandError, Exception) as e:
        return False, {"reason": f"lammps failed: {e}"}, float("inf")

    try:
        tbl = parse_last_thermo_table(stage_dir / "log.lammps").as_dict()
    except Exception as e:
        return False, {"reason": f"failed to parse thermo: {e}"}, float("inf")

    ok, details, score = _evaluate_thermo_table(
        tbl,
        T_target=float(temperature),
        P_target=float(md.pressure),
        ensemble=str(md.ensemble),
        config=config,
    )
    return ok, details, score


def _select_md_settings(
    runner: LammpsRunner,
    config: RunConfig,
    input_data: Path,
    outdir: Path,
    *,
    potential_lines: Optional[list[str]],
    timestep: Optional[float] = None,
) -> tuple[MDConfig, list[ThermoCandidateResult]]:
    pf = config.autotune.preflight
    T_low = float(pf.T_low) if pf.T_low is not None else float(config.autotune.tm_scan.t_min)
    T_high = float(pf.T_high) if pf.T_high is not None else float(config.autotune.tm_scan.t_max)

    ensembles = pf.ensembles if pf.ensembles is not None else [config.md.ensemble]
    ensembles = [str(e) for e in ensembles]

    dt = float(timestep) if timestep is not None else float(config.md.timestep)
    dt_tag = f"dt{dt:.6g}".replace(".", "p")
    tdamps = [float(f) * dt for f in pf.tdamp_factors]
    pdamps = [float(f) * dt for f in pf.pdamp_factors]

    candidates: list[ThermoCandidateResult] = []

    for ens in ensembles:
        for tdamp in tdamps:
            if ens == "npt":
                for pdamp in pdamps:
                    md = config.md.model_copy(
                        deep=True,
                        update={
                            "timestep": float(dt),
                            "ensemble": "npt",
                            "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(tdamp)),
                            "barostat": BarostatConfig(style=config.md.barostat.style, pdamp=float(pdamp)),
                        },
                    )
                    name = f"{dt_tag}_{ens}_td{tdamp:.3g}_pd{pdamp:.3g}".replace(".", "p")
                    okL, detL, scL = _run_candidate(
                        runner,
                        config,
                        input_data,
                        outdir=outdir,
                        potential_lines=potential_lines,
                        md=md,
                        temperature=T_low,
                        name=name,
                        equil_steps=pf.equil_steps,
                        run_steps=pf.run_steps,
                    )
                    okH, detH, scH = _run_candidate(
                        runner,
                        config,
                        input_data,
                        outdir=outdir,
                        potential_lines=potential_lines,
                        md=md,
                        temperature=T_high,
                        name=name,
                        equil_steps=pf.equil_steps,
                        run_steps=pf.run_steps,
                    )
                    ok = bool(okL and okH)
                    score = float(scL + scH)
                    candidates.append(
                        ThermoCandidateResult(
                            timestep=float(dt),
                            ensemble=ens,
                            tdamp=float(tdamp),
                            pdamp=float(pdamp),
                            ok=ok,
                            score=score,
                            details={"low": detL, "high": detH},
                        )
                    )
            else:
                md = config.md.model_copy(
                    deep=True,
                    update={
                        "timestep": float(dt),
                        "ensemble": "nvt",
                        "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(tdamp)),
                    },
                )
                name = f"{dt_tag}_{ens}_td{tdamp:.3g}".replace(".", "p")
                okL, detL, scL = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md,
                    temperature=T_low,
                    name=name,
                    equil_steps=pf.equil_steps,
                    run_steps=pf.run_steps,
                )
                okH, detH, scH = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md,
                    temperature=T_high,
                    name=name,
                    equil_steps=pf.equil_steps,
                    run_steps=pf.run_steps,
                )
                ok = bool(okL and okH)
                score = float(scL + scH)
                candidates.append(
                    ThermoCandidateResult(
                        timestep=float(dt),
                        ensemble=ens,
                        tdamp=float(tdamp),
                        pdamp=None,
                        ok=ok,
                        score=score,
                        details={"low": detL, "high": detH},
                    )
                )

    ok_cands = [c for c in candidates if c.ok]
    if not ok_cands:
        raise PreflightError(
            "Preflight failed: no thermostat/barostat candidates passed. "
            "All probe runs either crashed or violated stability/accuracy tolerances.",
            candidates=candidates,
        )

    # ensemble consistency production
    # candidate quench volume
    # transient introduce rate
    prod_ens = str(config.md.ensemble).strip().lower()
    ok_use = ok_cands
    if prod_ens == "npt":
        ok_npt = [c for c in ok_cands if str(c.ensemble) == "npt"]
        if ok_npt:
            ok_use = ok_npt
        else:
            if bool(getattr(pf, "allow_nvt_fallback", False)):
                ok_use = ok_cands
            else:
                raise PreflightError(
                    "Preflight failed: production ensemble is NPT but no NPT thermostat/barostat candidates passed.",
                    candidates=candidates,
                )

    # screening instabilities barostat
    # conservative candidates selecting
    # selection melt quench
    # selection melt quench
    # production ensemble conservative
    # aggressive volume neighbor
    # candidates comparable stability

    prod_ens = str(config.md.ensemble).strip().lower()
    chosen = None

    if prod_ens == "npt":
        # prefer conservative pdamp
        min_pdamp_ps = float(getattr(pf, "min_pdamp_ps_highT", 0.0) or 0.0)
        min_pdamp = (
            _duration_ps_to_lammps_time(min_pdamp_ps, str(config.kim.user_units))
            if min_pdamp_ps > 0.0
            else 0.0
        )
        ok_use_npt = [c for c in ok_use if str(c.ensemble)=="npt" and c.pdamp is not None]
        if min_pdamp > 0.0:
            ok_ge = [c for c in ok_use_npt if float(c.pdamp) >= min_pdamp]
            if not ok_ge:
                raise PreflightError(
                    "Preflight failed: no stable NPT candidate satisfies "
                    f"min_pdamp_ps_highT={min_pdamp_ps:g} ps "
                    f"({min_pdamp:g} in LAMMPS {config.kim.user_units} time units).",
                    candidates=candidates,
                )
            ok_use_npt = ok_ge

        # order candidates stability
        ok_use_npt = sorted(ok_use_npt, key=lambda c: (-float(c.pdamp or 0.0), -float(c.tdamp), float(c.score)))

        topk = int(getattr(pf, "confirm_topk", 5) or 5)
        topk = max(1, topk)

        # confirm temperatures include
        confirm_temps = list(getattr(pf, "confirm_temps", None) or [])
        if not confirm_temps:
            confirm_temps = [T_high]
        else:
            if all(abs(float(x) - float(T_high)) > 1.0e-6 for x in confirm_temps):
                confirm_temps.append(T_high)
        confirm_temps = [float(x) for x in confirm_temps]

        # confirmation length scan
        tm_total = int(getattr(config.autotune.tm_scan, "equil_steps", 0)) + int(getattr(config.autotune.tm_scan, "sample_steps", 0))
        confirm_run = int(getattr(pf, "confirm_run_steps", max(pf.run_steps, 10000)))
        confirm_run = max(confirm_run, tm_total)
        confirm_equil = int(getattr(pf, "confirm_equil_steps", pf.equil_steps))

        passed = []
        for c in ok_use_npt[:topk]:
            all_ok = True
            for Tconf in confirm_temps:
                md_c = config.md.model_copy(
                    deep=True,
                    update={
                        "timestep": float(dt),
                        "ensemble": "npt",
                        "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(c.tdamp)),
                        "barostat": BarostatConfig(style=config.md.barostat.style, pdamp=float(c.pdamp)),
                    },
                )
                okC, detC, scC = _run_candidate(
                    runner,
                    config,
                    input_data,
                    outdir=outdir,
                    potential_lines=potential_lines,
                    md=md_c,
                    temperature=float(Tconf),
                    name=f"{dt_tag}_confirm_{int(round(Tconf))}K_td{float(c.tdamp):.3g}_pd{float(c.pdamp):.3g}".replace(".", "p"),
                    equil_steps=confirm_equil,
                    run_steps=confirm_run,
                )
                try:
                    c.details[f"confirm_{int(round(Tconf))}K"] = detC
                    c.details[f"confirm_{int(round(Tconf))}K_score"] = float(scC)
                    c.details[f"confirm_{int(round(Tconf))}K_ok"] = bool(okC)
                except Exception:
                    pass
                if not okC:
                    all_ok = False
                    break
            if all_ok:
                passed.append(c)

        if not passed:
            raise PreflightError(
                "Preflight failed: screening found stable NPT candidates, but none passed the long confirmation run(s). "
                "This indicates timestep and/or barostat damping is too aggressive for high-T NPT (or the system is too small).",
                candidates=candidates,
            )

        # choose conservative confirm
        passed = sorted(passed, key=lambda c: (-float(c.pdamp or 0.0), -float(c.tdamp), float(c.score)))
        chosen = passed[0]

    else:
        ok_use = sorted(ok_use, key=lambda c: float(c.score))
        chosen = ok_use[0]

    md_sel = config.md.model_copy(
        deep=True,
        update={
            "timestep": float(dt),
            "ensemble": str(chosen.ensemble),
            "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(chosen.tdamp)),
            "barostat": (
                BarostatConfig(style=config.md.barostat.style, pdamp=float(chosen.pdamp))
                if str(chosen.ensemble) == "npt" and chosen.pdamp is not None
                else config.md.barostat
            ),
        },
    )
    return md_sel, candidates


def run_preflight(runner: Union[LammpsRunner, Cp2kRunner], config: RunConfig, input_data: Path, outdir: Path) -> PreflightResult:
    """Preflight."""
    ensure_dir(outdir / "preflight")

    pf = config.autotune.preflight

    # preflight timestep thermostat
    if getattr(config, "engine", "lammps") == "cp2k" or isinstance(runner, Cp2kRunner):
        return _run_preflight_cp2k(runner, config, input_data, outdir)
    if not pf.enabled:
        core_res = CoreRepulsionResult(
            enabled=bool(config.kim.core_repulsion.enabled),
            applied=False,
            style=str(config.kim.core_repulsion.style),
            base_pair_style=None,
            r_inner=None,
            r_outer=None,
            attempts=0,
            success=True,
            note="preflight disabled",
        )
        return PreflightResult(
            selected_ensemble=str(config.md.ensemble),
            selected_timestep=float(config.md.timestep),
            selected_tdamp=float(config.md.thermostat.tdamp),
            selected_pdamp=float(config.md.barostat.pdamp) if str(config.md.ensemble) == "npt" else None,
            potential_lines=None,
            core_repulsion=core_res,
            candidates=[],
        )

    T_test = float(pf.T_high) if pf.T_high is not None else float(config.autotune.tm_scan.t_max)
    pot_lines, core_res, dt_sel = _maybe_apply_core_repulsion(runner, config, input_data, outdir, T_test=T_test)

    # requested buckingham overlay
    # parameterisation continuing misleading
    if bool(config.kim.core_repulsion.enabled) and (not bool(core_res.success)):
        fail = {
            "success": False,
            "reason": f"Core repulsion preflight failed: {core_res.note}",
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(f"Core repulsion preflight failed: {core_res.note}")

    # timestep thermo preflight
    # fallback
    # explicitly enabled
    dt_base = float(dt_sel) if dt_sel is not None else float(config.md.timestep)
    dt_list, dt_source = _preflight_dt_candidates(config, dt_cap=(dt_base if (bool(core_res.applied) and bool(core_res.success)) else None))
    preflight_warnings: list[str] = []
    if len(dt_list) == 0:
        msg = (
            f"Preflight cannot continue: the calibrated core-repulsion timestep ({dt_base:g}) is smaller than every allowed "
            f"preflight timestep from {dt_source}. Declare a compatible autotune.preflight.dt_candidates list."
        )
        fail = {
            "success": False,
            "reason": msg,
            "timestep_candidate_source": dt_source,
            "timestep_candidates_tried": [],
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(msg)

    md_sel = None
    candidates_all: list[ThermoCandidateResult] = []
    last_err: Optional[PreflightError] = None
    dt_tried: list[float] = []
    for dt in dt_list:
        dt_tried.append(float(dt))
        try:
            md_sel_i, cands_i = _select_md_settings(
                runner,
                config,
                input_data,
                outdir,
                potential_lines=pot_lines,
                timestep=float(dt),
            )
            md_sel = md_sel_i
            candidates_all.extend(cands_i)
            break
        except PreflightError as e:
            last_err = e
            try:
                candidates_all.extend(list(getattr(e, "candidates", []) or []))
            except Exception:
                pass
            continue

    if len(dt_tried) > 1:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            _make_timestep_fallback_warning(
                context="thermo preflight",
                source=dt_source,
                tried=dt_tried,
                selected=(None if md_sel is None else float(getattr(md_sel, "timestep", dt_tried[-1]))),
            ),
        )

    if md_sel is None:
        prod_ens = str(config.md.ensemble).strip().lower()

        # summarise misleading diagnoses
        ok_total = 0
        ok_by_ens: dict[str, int] = {}
        scanned_ens: set[str] = set()
        for c in candidates_all:
            try:
                ens = str(getattr(c, "ensemble", "")).strip().lower() or "unknown"
            except Exception:
                ens = "unknown"
            scanned_ens.add(ens)
            if bool(getattr(c, "ok", False)):
                ok_total += 1
                ok_by_ens[ens] = int(ok_by_ens.get(ens, 0)) + 1

        ok_npt = int(ok_by_ens.get("npt", 0))
        ok_nvt = int(ok_by_ens.get("nvt", 0))
        scanned_sorted = sorted(scanned_ens)

        # actionable production scan
        # unstable preflight scan
        # configured test nvt
        if prod_ens == "npt" and ok_total > 0 and ok_npt == 0 and ok_nvt > 0:
            msg = (
                "Preflight failed: production ensemble is NPT but no stable NPT thermostat/barostat candidates were found. "
                f"Found {ok_nvt} stable NVT candidates. "
                "Ensure autotune.preflight.ensembles includes 'npt' (or leave it unset to follow md.ensemble), "
                "or set autotune.preflight.allow_nvt_fallback=true to permit NVT selection."
            )
        else:
            # otherwise informative underlying
            if last_err is not None:
                msg = f"Preflight failed after timestep fallback: {last_err}"
            else:
                msg = (
                    "Preflight failed: no stable thermostat/barostat candidates passed, even after timestep fallback. "
                    f"Requested production ensemble: {prod_ens}."
                )

        fail = {
            "success": False,
            "reason": msg,
            "requested_production_ensemble": prod_ens,
            "ensembles_scanned": scanned_sorted,
            "ok_counts_by_ensemble": ok_by_ens,
            "timestep_candidate_source": dt_source,
            "timestep_candidates_tried": [float(x) for x in dt_list],
            "warnings": list(preflight_warnings),
            "potential_lines": pot_lines,
            "core_repulsion": asdict(core_res),
            "candidates": [asdict(c) for c in candidates_all],
        }
        (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(fail, indent=2))
        raise PreflightError(msg, candidates=candidates_all) from last_err

    candidates = candidates_all

    requested_ensemble = str(config.md.ensemble).strip().lower()
    if str(md_sel.ensemble).strip().lower() != requested_ensemble:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            f"thermo preflight: ensemble fallback activated; requested ensemble={requested_ensemble}; "
            f"selected ensemble={str(md_sel.ensemble).strip().lower()}",
        )

    res = PreflightResult(
        selected_ensemble=str(md_sel.ensemble),
        selected_timestep=float(md_sel.timestep),
        selected_tdamp=float(md_sel.thermostat.tdamp),
        selected_pdamp=float(md_sel.barostat.pdamp) if str(md_sel.ensemble) == "npt" else None,
        potential_lines=pot_lines,
        core_repulsion=core_res,
        candidates=candidates,
    )
    report = asdict(res)
    report["timestep_candidate_source"] = dt_source
    report["timestep_candidates_tried"] = [float(x) for x in dt_list]
    report["selected_timestep_reason"] = "largest stable user-declared candidate"
    report["warnings"] = list(preflight_warnings)
    (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(report, indent=2))
    return res



def _run_preflight_cp2k(
    runner: Union[LammpsRunner, Cp2kRunner],
    config: RunConfig,
    input_data: Path,
    outdir: Path,
) -> PreflightResult:
    """Preflight cp2k."""

    if not isinstance(runner, Cp2kRunner):
        raise ValueError("engine='cp2k' requires a Cp2kRunner")

    pf = config.autotune.preflight
    md_ref = config.md

    import numpy as np

    # mapping reading lammps
    type_to_species = config.autotune.metrics.type_to_species
    if type_to_species is None:
        raise ValueError("CP2K preflight requires autotune.metrics.type_to_species")

    # preflight disabled stability
    if not pf.enabled:
        core_res = CoreRepulsionResult(
            enabled=False,
            applied=False,
            style="none",
            base_pair_style=None,
            r_inner=None,
            r_outer=None,
            attempts=0,
            success=True,
            note="preflight disabled (cp2k backend)",
        )
        return PreflightResult(
            selected_ensemble=str(md_ref.ensemble),
            selected_timestep=float(md_ref.timestep),
            selected_tdamp=float(md_ref.thermostat.tdamp),
            selected_pdamp=float(md_ref.barostat.pdamp) if str(md_ref.ensemble).lower() == "npt" else None,
            potential_lines=None,
            core_repulsion=core_res,
            candidates=[],
        )

    # temperatures scan conservative
    T_low = float(config.autotune.tm_scan.t_min)
    T_high = float(config.autotune.tm_scan.t_max)
    if not (math.isfinite(T_low) and math.isfinite(T_high) and T_low > 0 and T_high > T_low):
        raise ValueError("Invalid tm_scan temperature range for preflight")

    # candidate timesteps permitted
    # fallback
    dt_list, dt_source = _preflight_dt_candidates(config, cp2k=True)
    preflight_warnings: list[str] = []
    dt_attempted: list[float] = []

    td_factors = [float(x) for x in pf.tdamp_factors]
    td_factors = [x for x in td_factors if math.isfinite(x) and x > 0]
    if len(td_factors) == 0:
        td_factors = [100.0]

    pd_factors = [float(x) for x in pf.pdamp_factors]
    pd_factors = [x for x in pd_factors if math.isfinite(x) and x > 0]
    if len(pd_factors) == 0:
        pd_factors = [1000.0]

    min_pdamp_fs_highT = float(pf.min_pdamp_ps_highT) * 1000.0

    # evaluation helpers
    def _tail_stats(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        n = int(min(int(pf.tail_window), int(x.size)))
        tail = x[-n:]
        return float(np.mean(tail)), float(np.std(tail))

    def _ok_temp(temp: np.ndarray, target: float) -> bool:
        temp = np.asarray(temp, dtype=float)
        temp = temp[np.isfinite(temp)]
        if temp.size == 0:
            return False
        meanT, _ = _tail_stats(temp)
        if not math.isfinite(meanT):
            return False
        # consistent lammps preflight
        if abs(meanT - target) / target > float(pf.temp_rel_tol):
            return False
        if float(np.max(temp)) > float(pf.max_temp_factor) * target:
            return False
        return True

    def _ok_volume(vol: np.ndarray) -> bool:
        vol = np.asarray(vol, dtype=float)
        vol = vol[np.isfinite(vol)]
        if vol.size == 0:
            return False
        n = int(min(int(pf.tail_window), int(vol.size)))
        tail = vol[-n:]
        vmin = float(np.min(tail))
        vmax = float(np.max(tail))
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin > 0.0):
            return False
        if vmax / vmin > float(pf.max_vol_ratio):
            return False
        return True

    candidates: list[ThermoCandidateResult] = []

    # ensembles configured order
    ens_order = [str(e).lower() for e in (pf.ensembles or [md_ref.ensemble])]
    ens_order = [e for e in ens_order if e in ("nvt", "npt")]
    if len(ens_order) == 0:
        ens_order = [str(md_ref.ensemble).lower()]

    best_overall: Optional[ThermoCandidateResult] = None
    best_overall_hard: Optional[ThermoCandidateResult] = None

    for ens in ens_order:
        best: Optional[ThermoCandidateResult] = None
        best_hard: Optional[ThermoCandidateResult] = None

        for dt in dt_list:
            if not dt_attempted or float(dt_attempted[-1]) != float(dt):
                dt_attempted.append(float(dt))
            for fac in td_factors:
                tdamp = fac * dt

                if ens == "npt":
                    for p_fac in pd_factors:
                        pdamp = max(p_fac * dt, min_pdamp_fs_highT)

                        md_use = md_ref.model_copy(deep=True)
                        md_use.timestep = float(dt)
                        md_use.ensemble = "npt"
                        md_use.thermostat.tdamp = float(tdamp)
                        md_use.barostat.pdamp = float(pdamp)

                        stage = StageSpec(
                            name="preflight_highT",
                            input_data=input_data,
                            output_data=outdir / "preflight" / "highT.out.data",
                            sample_ensemble=None,
                            temperature_start=T_high,
                            temperature_stop=T_high,
                            pressure=float(md_ref.pressure),
                            equil_steps=int(pf.equil_steps),
                            run_steps=int(pf.run_steps),
                            seed=int(config.random_seed),
                            replicate=None,
                            write_dump=False,
                            dump_every=md_use.dump_every,
                            tail_dump_frames=0,
                            tail_dump_stride=1,
                            # preflight diffusion estimates
                            # msd sampling thermo
                            msd_every=int(md_use.thermo_every),
                            potential_lines=None,
                        )

                        cand_dir = outdir / "preflight" / f"npt_dt_{dt:g}_td_{tdamp:g}_pd_{pdamp:g}"
                        ok = False
                        last_err: Optional[Exception] = None
                        meanT = float("nan")
                        maxT = float("nan")
                        tstd = float("nan")
                        vmean = float("nan")
                        vstd = float("nan")
                        vratio = float("nan")
                        pmean = float("nan")
                        pstd = float("nan")
                        pmax_abs = float("nan")
                        score = float("inf")
                        vol_ok = False
                        pressure_ok = False
                        hard_temp_ok = False
                        strict_temp_ok = False
                        hard_ok = False
                        try:
                            art = run_stage_local(
                                runner,
                                None,
                                md_use,
                                stage,
                                cand_dir,
                                log_name="log.lammps",
                                type_to_species=list(type_to_species),
                            )
                            # CP2K's synthetic native log retains pressure in
                            # bar for the configured preflight thresholds.  The
                            # engine-neutral CSV uses GPa, so convert back only
                            # when the native log cannot be parsed.
                            pressure_scale_to_bar = 1.0
                            try:
                                table = parse_last_thermo_table(art.log_path)
                            except Exception:
                                from ..io.thermo import parse_thermo_csv

                                table = parse_thermo_csv(art.thermo_csv)
                                pressure_scale_to_bar = 1.0e4
                            # thermo table dict
                            # mapping column access
                            tdict = table.as_dict()
                            temp = np.asarray(tdict.get("Temp", []), dtype=float)
                            vol = np.asarray(tdict.get("Volume", []), dtype=float)
                            pressure = (
                                np.asarray(tdict.get("Press", []), dtype=float)
                                * pressure_scale_to_bar
                            )

                            if temp.size > 0:
                                meanT, tstd = _tail_stats(temp)
                                maxT = float(np.max(temp[np.isfinite(temp)])) if np.any(np.isfinite(temp)) else float("nan")

                            if vol.size > 0:
                                vmean, vstd = _tail_stats(vol)
                                n = int(min(int(pf.tail_window), int(np.isfinite(vol).sum())))
                                if n > 0:
                                    tail = vol[np.isfinite(vol)][-n:]
                                    vmin = float(np.min(tail))
                                    vmax = float(np.max(tail))
                                    vratio = vmax / vmin if vmin > 0 else float("nan")

                            vol_ok = _ok_volume(vol)
                            pressure_ok, pressure_details = _screen_cp2k_pressure_samples(
                                pressure,
                                target_bar=float(md_ref.pressure),
                                tail_window=int(pf.tail_window),
                                max_abs_bar=float(pf.max_press_abs),
                                tolerance_bar=float(pf.press_abs_tol),
                                require_tolerance=bool(
                                    getattr(pf, "require_pressure_tolerance", False)
                                ),
                            )
                            pmean = float(pressure_details["P_mean"])
                            pstd = float(pressure_details["P_tail_std"])
                            pmax_abs = float(pressure_details["P_max_abs"])
                            # strict stability tolerance
                            # enforces bounded statistics
                            hard_temp_ok = bool(temp.size > 0 and math.isfinite(meanT) and math.isfinite(maxT) and maxT <= float(pf.max_temp_factor) * T_high)
                            strict_temp_ok = bool(hard_temp_ok and abs(meanT - T_high) / T_high <= float(pf.temp_rel_tol))
                            hard_ok = bool(hard_temp_ok and vol_ok and pressure_ok)
                            ok = bool(strict_temp_ok and vol_ok and pressure_ok)

                            # temperature volume fluctuation
                            vol_pen = 0.0
                            if math.isfinite(vratio):
                                vol_pen = max(0.0, vratio - 1.0)
                            pressure_pen = (
                                abs(pmean - float(md_ref.pressure)) / max(1.0, float(pf.press_abs_tol))
                                if math.isfinite(pmean)
                                else float("inf")
                            )
                            score = (
                                -float(dt)
                                + abs(meanT - T_high) / T_high
                                + 0.1 * vol_pen
                                + pressure_pen
                            )

                        except Exception as e:
                            last_err = e
                            ok = False
                            # debug
                            try:
                                import traceback

                                ensure_dir(cand_dir)
                                (cand_dir / "vitriflow_error.txt").write_text(traceback.format_exc())
                            except Exception:
                                pass

                        details: dict[str, Any] = {
                            "T_target": float(T_high),
                            "mean_temp": float(meanT),
                            "temp_tail_std": float(tstd),
                            "max_temp": float(maxT),
                            "mean_volume": float(vmean),
                            "volume_tail_std": float(vstd),
                            "volume_ratio": float(vratio),
                            "P_target": float(md_ref.pressure),
                            "P_mean": float(pmean),
                            "P_tail_std": float(pstd),
                            "P_max_abs": float(pmax_abs),
                        }
                        details.update(
                            {
                                "temp_rel_err": float(abs(meanT - T_high) / T_high) if math.isfinite(meanT) else float("nan"),
                                "strict_temp_ok": bool(strict_temp_ok),
                                "hard_temp_ok": bool(hard_temp_ok),
                                "vol_ok": bool(vol_ok),
                                "pressure_ok": bool(pressure_ok),
                                "hard_ok": bool(hard_ok),
                            }
                        )
                        if last_err is not None:
                            details["last_error"] = str(last_err)

                        cand = ThermoCandidateResult(
                            timestep=float(dt),
                            ensemble="npt",
                            tdamp=float(tdamp),
                            pdamp=float(pdamp),
                            ok=bool(ok),
                            score=float(score),
                            details=details,
                        )
                        candidates.append(cand)

                        if ok and (best is None or cand.score < best.score):
                            best = cand
                        if hard_ok and (best_hard is None or cand.score < best_hard.score):
                            best_hard = cand

                else:  # nvt
                    md_use = md_ref.model_copy(deep=True)
                    md_use.timestep = float(dt)
                    md_use.ensemble = "nvt"
                    md_use.thermostat.tdamp = float(tdamp)

                    stage = StageSpec(
                        name="preflight_highT",
                        input_data=input_data,
                        output_data=outdir / "preflight" / "highT.out.data",
                        sample_ensemble=None,
                        temperature_start=T_high,
                        temperature_stop=T_high,
                        pressure=float(md_ref.pressure),
                        equil_steps=int(pf.equil_steps),
                        run_steps=int(pf.run_steps),
                        seed=int(config.random_seed),
                        replicate=None,
                        write_dump=False,
                        dump_every=md_use.dump_every,
                        tail_dump_frames=0,
                        tail_dump_stride=1,
                        msd_every=int(md_use.thermo_every),
                        potential_lines=None,
                    )

                    cand_dir = outdir / "preflight" / f"nvt_dt_{dt:g}_td_{tdamp:g}"
                    ok = False
                    last_err: Optional[Exception] = None
                    meanT = float("nan")
                    maxT = float("nan")
                    tstd = float("nan")
                    score = float("inf")
                    hard_temp_ok = False
                    strict_temp_ok = False
                    hard_ok = False
                    try:
                        art = run_stage_local(
                            runner,
                            None,
                            md_use,
                            stage,
                            cand_dir,
                            log_name="log.lammps",
                            type_to_species=list(type_to_species),
                        )
                        try:
                            from ..io.thermo import parse_thermo_csv

                            table = parse_thermo_csv(art.thermo_csv)
                        except Exception:
                            table = parse_last_thermo_table(art.log_path)
                        tdict = table.as_dict()
                        temp = np.asarray(tdict.get("Temp", []), dtype=float)
                        if temp.size > 0:
                            meanT, tstd = _tail_stats(temp)
                            maxT = float(np.max(temp[np.isfinite(temp)])) if np.any(np.isfinite(temp)) else float("nan")
                        # strict stability volume
                        hard_temp_ok = bool(temp.size > 0 and math.isfinite(meanT) and math.isfinite(maxT) and maxT <= float(pf.max_temp_factor) * T_high)
                        strict_temp_ok = bool(hard_temp_ok and abs(meanT - T_high) / T_high <= float(pf.temp_rel_tol))
                        hard_ok = bool(hard_temp_ok)
                        ok = bool(strict_temp_ok)
                        score = -float(dt) + abs(meanT - T_high) / T_high
                    except Exception as e:
                        last_err = e
                        ok = False
                        try:
                            import traceback

                            ensure_dir(cand_dir)
                            (cand_dir / "vitriflow_error.txt").write_text(traceback.format_exc())
                        except Exception:
                            pass

                    details: dict[str, Any] = {
                        "T_target": float(T_high),
                        "mean_temp": float(meanT),
                        "temp_tail_std": float(tstd),
                        "max_temp": float(maxT),
                    }
                    details.update(
                        {
                            "temp_rel_err": float(abs(meanT - T_high) / T_high) if math.isfinite(meanT) else float("nan"),
                            "strict_temp_ok": bool(strict_temp_ok),
                            "hard_temp_ok": bool(hard_temp_ok),
                            "hard_ok": bool(hard_ok),
                        }
                    )
                    if last_err is not None:
                        details["last_error"] = str(last_err)

                    cand = ThermoCandidateResult(
                        timestep=float(dt),
                        ensemble="nvt",
                        tdamp=float(tdamp),
                        pdamp=None,
                        ok=bool(ok),
                        score=float(score),
                        details=details,
                    )
                    candidates.append(cand)

                    if ok and (best is None or cand.score < best.score):
                        best = cand
                    if hard_ok and (best_hard is None or cand.score < best_hard.score):
                        best_hard = cand

            # candidate largest searching
            if best is not None and best.timestep == float(dt):
                break

        if best is not None:
            best_overall = best
            break
        if best_overall_hard is None and best_hard is not None:
            best_overall_hard = best_hard

    core_res = CoreRepulsionResult(
        enabled=False,
        applied=False,
        style="none",
        base_pair_style=None,
        r_inner=None,
        r_outer=None,
        attempts=0,
        success=True,
        note="core repulsion not applicable for cp2k backend",
    )
    if best_overall is None and best_overall_hard is not None:
        # fallback
        # candidate bounded volume
        d = dict(best_overall_hard.details or {})
        d["selected_by"] = "hard_fallback"
        d["note"] = "No candidate met strict mean-T tolerance; selected best hard-stable candidate."
        best_overall = ThermoCandidateResult(
            timestep=float(best_overall_hard.timestep),
            ensemble=str(best_overall_hard.ensemble),
            tdamp=float(best_overall_hard.tdamp),
            pdamp=float(best_overall_hard.pdamp) if best_overall_hard.pdamp is not None else None,
            ok=True,
            score=float(best_overall_hard.score),
            details=d,
        )

    if len(dt_attempted) > 1:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            _make_timestep_fallback_warning(
                context="cp2k preflight",
                source=dt_source,
                tried=dt_attempted,
                selected=(None if best_overall is None else float(best_overall.timestep)),
            ),
        )

    if best_overall is None:
        # candidate exception structure
        # representative misleading diagnosis
        errs = []
        for c in candidates:
            try:
                if isinstance(c, ThermoCandidateResult) and isinstance(c.details, dict) and "last_error" in c.details:
                    errs.append(str(c.details.get("last_error")))
            except Exception:
                pass
        if errs and len(errs) == len(candidates):
            msg = (
                "CP2K preflight failed: all candidates errored before producing usable output; "
                f"example error: {errs[0]}"
            )
        else:
            msg = "CP2K preflight failed: no stable thermostat/barostat candidates passed at high temperature"
        raise PreflightError(msg, candidates=candidates)

    requested_ensemble = str(md_ref.ensemble).lower()
    if str(best_overall.ensemble).lower() != requested_ensemble:
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            f"cp2k preflight: ensemble fallback activated; requested ensemble={requested_ensemble}; "
            f"selected ensemble={str(best_overall.ensemble).lower()}",
        )
    if str((best_overall.details or {}).get("selected_by", "")).strip().lower() == "hard_fallback":
        _append_preflight_warning(
            outdir,
            preflight_warnings,
            "cp2k preflight: hard stability fallback activated; no candidate met strict mean-temperature tolerance; "
            "selected best hard-stable candidate",
        )

    res = PreflightResult(
        selected_ensemble=str(best_overall.ensemble),
        selected_timestep=float(best_overall.timestep),
        selected_tdamp=float(best_overall.tdamp),
        selected_pdamp=float(best_overall.pdamp) if str(best_overall.ensemble).lower() == "npt" else None,
        potential_lines=None,
        core_repulsion=core_res,
        candidates=candidates,
    )
    report = asdict(res)
    report["timestep_candidate_source"] = dt_source
    report["timestep_candidates_tried"] = [float(x) for x in dt_list]
    report["selected_timestep_reason"] = "largest stable user-declared candidate"
    report["warnings"] = list(preflight_warnings)
    (outdir / "preflight" / "preflight_results.json").write_text(json.dumps(report, indent=2))
    return res
