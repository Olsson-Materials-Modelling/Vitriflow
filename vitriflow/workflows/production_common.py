
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import TypeAdapter

from ..config import (
    ConvergenceConfig,
    MDConfig,
    PotentialConfig,
    ProductionEnsembleConfig,
    StructureMetricsConfig,
)
from ..analysis.amorphous import analyse_amorphous_state, reduced_formula_from_frame
from ..analysis.gr import compute_gr
from ..analysis.structure import (
    compute_coordination_defect_details,
    compute_coordination_defects,
    compute_structure_distributions_timeavg,
    compute_structure_metrics,
    compute_structure_metrics_timeavg,
    estimate_pair_cutoffs,
)
from ..analysis.provenance import json_sanitize, write_json_strict
from ..analysis.trajectory import read_last_frames_auto
from ..io.extxyz import write_extxyz_single_with_species
from .elastic_screen import should_collect_elastic_stage_timeseries
from .stage_metrics import collect_stage_metrics_timeseries, should_collect_stage_metrics_timeseries
from .step_counts import recommended_quench_dump_every, resolve_lammps_units_style
from .quench_rates import lammps_timeunit_ps, quench_steps_for_rate
from ..analysis.trajectory import quench_window_steps


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionPlan:
    engine: str
    structure_data: Path
    T_high: float
    high_total_steps: int
    t_final: float
    chosen_rate: float
    cooling_rate_ps: Optional[float]
    replicate: tuple[int, int, int]
    pressure: float
    md_use: dict[str, Any]
    potential_config: Optional[dict[str, Any]]
    potential_lines: Optional[list[str]]
    core_repulsion: Optional[dict[str, Any]]
    type_to_species: Optional[list[str]]
    metrics_cfg: dict[str, Any]
    effective_metrics: dict[str, Any]
    production_cfg: dict[str, Any]
    convergence_cfg: dict[str, Any]
    cutoffs_rate: list[dict[str, Any]]
    cutoffs_size: list[dict[str, Any]]
    preferred_cutoffs: list[dict[str, Any]]
    quench_steps: int
    relax_steps: int
    msd_every: int
    seed_base: int
    time_unit_ps: Optional[float]
    sampling_hint: Optional[dict[str, float]]
    execution_mode: str = "adaptive"
    source_kind: str = "plan"


def _cutoffs_list_from_dict(obj: Optional[Mapping[Tuple[int, int], float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(obj, Mapping):
        return out
    for (a, b), c in sorted(obj.items()):
        out.append({"pair": [int(a), int(b)], "cutoff": float(c)})
    return out


def cutoffs_dict_from_any(obj: Any) -> dict[Tuple[int, int], float]:
    if isinstance(obj, Mapping):
        out: dict[Tuple[int, int], float] = {}
        for k, v in obj.items():
            if isinstance(k, tuple) and len(k) == 2:
                out[(int(min(k[0], k[1])), int(max(k[0], k[1])))] = float(v)
        return out
    if isinstance(obj, list):
        out: dict[Tuple[int, int], float] = {}
        for ent in obj:
            if not isinstance(ent, Mapping):
                continue
            pair = ent.get("pair", None)
            cutoff = ent.get("cutoff", None)
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and cutoff is not None:
                a, b = int(pair[0]), int(pair[1])
                out[(min(a, b), max(a, b))] = float(cutoff)
        return out
    return {}


def _validated_plan_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"production plan {name} must be a mapping")
    return dict(value)


def _validated_plan_integer(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"production plan {name} must be an integer >= {minimum}")
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"production plan {name} must be an integer >= {minimum}"
        ) from exc
    if not math.isfinite(numeric) or numeric != float(integer) or integer < minimum:
        raise ValueError(f"production plan {name} must be an integer >= {minimum}")
    return integer


def _validated_plan_cutoffs(name: str, value: Any) -> dict[Tuple[int, int], float]:
    if value is None:
        return {}
    entries: list[tuple[Any, Any]] = []
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            if not isinstance(item, Mapping) or "pair" not in item or "cutoff" not in item:
                raise ValueError(
                    f"production plan {name}[{idx}] requires a pair and cutoff"
                )
            entries.append((item.get("pair"), item.get("cutoff")))
    else:
        raise ValueError(f"production plan {name} must be a mapping or list")

    out: dict[Tuple[int, int], float] = {}
    for idx, (pair_raw, cutoff_raw) in enumerate(entries):
        if not isinstance(pair_raw, Sequence) or isinstance(
            pair_raw, (str, bytes, bytearray)
        ):
            raise ValueError(f"production plan {name}[{idx}].pair must contain two type ids")
        pair = list(pair_raw)
        if len(pair) != 2:
            raise ValueError(
                f"production plan {name}[{idx}].pair must contain exactly two type ids"
            )
        a = _validated_plan_integer(
            f"{name}[{idx}].pair[0]", pair[0], minimum=1
        )
        b = _validated_plan_integer(
            f"{name}[{idx}].pair[1]", pair[1], minimum=1
        )
        try:
            cutoff = float(cutoff_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"production plan {name}[{idx}] contains non-numeric data") from exc
        if not (math.isfinite(cutoff) and cutoff > 0.0):
            raise ValueError(f"production plan {name}[{idx}].cutoff must be finite and > 0")
        key = (min(a, b), max(a, b))
        if key in out and not math.isclose(
            out[key], cutoff, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"production plan {name} contains conflicting cutoffs for pair {key}"
            )
        out[key] = cutoff
    return out


def validate_production_plan(
    plan: ProductionPlan,
    *,
    require_structure_file: bool = False,
) -> ProductionPlan:
    """Validate all execution-critical fields of a production plan."""

    engine = str(plan.engine).strip().lower()
    if engine not in {"lammps", "cp2k"}:
        raise ValueError("production plan engine must be 'lammps' or 'cp2k'")
    structure_text = str(plan.structure_data).strip()
    if not structure_text or structure_text == ".":
        raise ValueError("production plan structure_data must be a non-empty file path")
    if require_structure_file and not Path(plan.structure_data).is_file():
        raise FileNotFoundError(
            f"production plan structure_data is not a readable file: {plan.structure_data}"
        )

    finite_fields = {
        "T_high": plan.T_high,
        "t_final": plan.t_final,
        "chosen_rate": plan.chosen_rate,
        "pressure": plan.pressure,
    }
    for name, raw in finite_fields.items():
        if not math.isfinite(float(raw)):
            raise ValueError(f"production plan {name} must be finite")
    if float(plan.T_high) <= 0.0:
        raise ValueError("production plan T_high must be > 0")
    if float(plan.t_final) < 0.0 or float(plan.t_final) >= float(plan.T_high):
        raise ValueError("production plan requires 0 <= t_final < T_high")
    if float(plan.chosen_rate) <= 0.0:
        raise ValueError("production plan chosen_rate must be > 0")
    if plan.cooling_rate_ps is not None and not (
        math.isfinite(float(plan.cooling_rate_ps)) and float(plan.cooling_rate_ps) > 0.0
    ):
        raise ValueError("production plan cooling_rate_ps must be finite and > 0")
    if plan.time_unit_ps is not None and not (
        math.isfinite(float(plan.time_unit_ps)) and float(plan.time_unit_ps) > 0.0
    ):
        raise ValueError("production plan time_unit_ps must be finite and > 0")
    if plan.cooling_rate_ps is not None and plan.time_unit_ps is not None:
        expected = float(plan.chosen_rate) / float(plan.time_unit_ps)
        if not math.isclose(
            float(plan.cooling_rate_ps), expected, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise ValueError(
                "production plan cooling_rate_ps is inconsistent with "
                "chosen_rate/time_unit_ps"
            )
    if engine == "cp2k" and plan.time_unit_ps is not None and not math.isclose(
        float(plan.time_unit_ps), 0.001, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("CP2K production plans require time_unit_ps=0.001")

    for name, raw, minimum in (
        ("high_total_steps", plan.high_total_steps, 1),
        ("quench_steps", plan.quench_steps, 1),
        ("relax_steps", plan.relax_steps, 0),
        ("msd_every", plan.msd_every, 1),
        ("seed_base", plan.seed_base, 0),
    ):
        _validated_plan_integer(name, raw, minimum=minimum)

    rep = tuple(plan.replicate)
    if len(rep) != 3 or any(isinstance(x, bool) or int(x) < 1 for x in rep):
        raise ValueError("production plan replicate must contain three integers >= 1")

    md_validated = MDConfig.model_validate(plan.md_use)
    metrics_validated = StructureMetricsConfig.model_validate(plan.metrics_cfg)
    ProductionEnsembleConfig.model_validate(plan.production_cfg)
    ConvergenceConfig.model_validate(plan.convergence_cfg)
    if not math.isclose(
        float(plan.pressure),
        float(md_validated.pressure),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "production plan pressure is inconsistent with md_use.pressure"
        )
    expected_quench_steps = quench_steps_for_rate(
        float(plan.T_high) - float(plan.t_final),
        float(plan.chosen_rate),
        float(md_validated.timestep),
        min_steps=1,
    )
    if int(plan.quench_steps) != int(expected_quench_steps):
        raise ValueError(
            "production plan quench_steps is inconsistent with "
            "(T_high-t_final)/(chosen_rate*md_use.timestep): "
            f"stored={int(plan.quench_steps)} expected={int(expected_quench_steps)}"
        )
    if engine == "lammps":
        if not isinstance(plan.potential_config, Mapping):
            raise ValueError("LAMMPS production plans require potential_config")
        TypeAdapter(PotentialConfig).validate_python(plan.potential_config)

    metrics_species = getattr(metrics_validated, "type_to_species", None)
    interactions = (
        plan.potential_config.get("interactions")
        if isinstance(plan.potential_config, Mapping)
        else None
    )
    interaction_species = (
        [str(x) for x in interactions]
        if isinstance(interactions, Sequence)
        and not isinstance(interactions, (str, bytes, bytearray))
        else None
    )
    if plan.type_to_species is None:
        sources: list[str] = []
        if metrics_species is not None:
            sources.append("metrics_cfg.type_to_species")
        if interaction_species is not None:
            sources.append("potential_config.interactions")
        if sources:
            raise ValueError(
                "production plan type_to_species is required when species ordering is "
                "defined by " + " and ".join(sources)
            )
    else:
        if metrics_species is not None and [str(x) for x in metrics_species] != [
            str(x) for x in plan.type_to_species
        ]:
            raise ValueError(
                "production plan type_to_species is inconsistent with metrics_cfg.type_to_species"
            )
        if interaction_species is not None:
            if interaction_species != [str(x) for x in plan.type_to_species]:
                raise ValueError(
                    "production plan type_to_species must match potential interaction ordering"
                )

    if plan.type_to_species is not None:
        species = [str(x).strip() for x in plan.type_to_species]
        if any(not x for x in species):
            raise ValueError(
                "production plan type_to_species must contain non-empty symbols"
            )
    if str(plan.execution_mode).strip().lower() not in {"adaptive", "fixed"}:
        raise ValueError("production plan execution_mode must be 'adaptive' or 'fixed'")
    if not str(plan.source_kind).strip():
        raise ValueError("production plan source_kind must be non-empty")

    for name, values in (
        ("cutoffs_rate", plan.cutoffs_rate),
        ("cutoffs_size", plan.cutoffs_size),
        ("preferred_cutoffs", plan.preferred_cutoffs),
    ):
        _validated_plan_cutoffs(name, values)
    if plan.sampling_hint is not None:
        if not isinstance(plan.sampling_hint, Mapping):
            raise ValueError("production plan sampling_hint must be a mapping or null")
        for key, value in plan.sampling_hint.items():
            if not str(key).strip() or not math.isfinite(float(value)):
                raise ValueError(
                    "production plan sampling_hint keys must be non-empty and values finite"
                )
    return plan


def make_production_plan(
    *,
    engine: str,
    structure_data: Path,
    T_high: float,
    high_total_steps: int,
    t_final: float,
    chosen_rate: float,
    cooling_rate_ps: Optional[float],
    replicate: Sequence[int],
    pressure: float,
    md_use: Mapping[str, Any],
    potential_config: Optional[Mapping[str, Any]],
    potential_lines: Optional[Sequence[str]],
    core_repulsion: Optional[Mapping[str, Any]],
    type_to_species: Optional[Sequence[str]],
    metrics_cfg: Mapping[str, Any],
    effective_metrics: Mapping[str, Any],
    production_cfg: Mapping[str, Any],
    convergence_cfg: Mapping[str, Any],
    cutoffs_rate: Any,
    cutoffs_size: Any,
    preferred_cutoffs: Any,
    quench_steps: int,
    relax_steps: int,
    msd_every: int,
    seed_base: int,
    time_unit_ps: Optional[float],
    sampling_hint: Optional[Mapping[str, float]],
    execution_mode: str,
    source_kind: str = "plan",
) -> ProductionPlan:
    if not isinstance(replicate, Sequence) or isinstance(
        replicate, (str, bytes, bytearray)
    ):
        raise ValueError("replicate must contain three integers >= 1")
    rep_raw = list(replicate)
    if len(rep_raw) != 3:
        raise ValueError("replicate must contain three integers >= 1")
    rep: list[int] = []
    for value in rep_raw:
        if isinstance(value, bool):
            raise ValueError("replicate must contain three integers >= 1")
        numeric = float(value)
        integer = int(numeric)
        if not math.isfinite(numeric) or numeric != float(integer) or integer < 1:
            raise ValueError("replicate must contain three integers >= 1")
        rep.append(integer)

    engine_norm = str(engine).strip().lower()
    md_data = _validated_plan_mapping("md_use", md_use)
    metrics_data = _validated_plan_mapping("metrics_cfg", metrics_cfg)
    effective_data = _validated_plan_mapping("effective_metrics", effective_metrics)
    production_data = _validated_plan_mapping("production_cfg", production_cfg)
    convergence_data = _validated_plan_mapping("convergence_cfg", convergence_cfg)
    potential_data = (
        None
        if potential_config is None
        else _validated_plan_mapping("potential_config", potential_config)
    )
    core_data = (
        None
        if core_repulsion is None
        else _validated_plan_mapping("core_repulsion", core_repulsion)
    )
    potential_line_list: Optional[list[str]] = None
    if potential_lines is not None:
        if not isinstance(potential_lines, Sequence) or isinstance(
            potential_lines, (str, bytes, bytearray)
        ):
            raise ValueError("production plan potential_lines must be a sequence or null")
        potential_line_list = [str(x).strip() for x in potential_lines]
        if any(not x for x in potential_line_list):
            raise ValueError("production plan potential_lines cannot contain empty lines")

    rate_cutoffs = _validated_plan_cutoffs("cutoffs_rate", cutoffs_rate)
    size_cutoffs = _validated_plan_cutoffs("cutoffs_size", cutoffs_size)
    preferred = _validated_plan_cutoffs("preferred_cutoffs", preferred_cutoffs)
    sampling_data: Optional[dict[str, float]] = None
    if sampling_hint is not None:
        if not isinstance(sampling_hint, Mapping):
            raise ValueError("production plan sampling_hint must be a mapping or null")
        sampling_data = {}
        for key, value in sampling_hint.items():
            if value is None:
                raise ValueError(
                    f"production plan sampling_hint[{str(key)!r}] must be finite, not null"
                )
            numeric = float(value)
            if not str(key).strip() or not math.isfinite(numeric):
                raise ValueError(
                    "production plan sampling_hint keys must be non-empty and values finite"
                )
            sampling_data[str(key)] = numeric

    species_data: Optional[list[str]] = None
    if type_to_species is not None:
        if not isinstance(type_to_species, Sequence) or isinstance(
            type_to_species, (str, bytes, bytearray)
        ):
            raise ValueError(
                "production plan type_to_species must be a non-string sequence or null"
            )
        species_data = [str(x).strip() for x in type_to_species]
        if any(not x for x in species_data):
            raise ValueError(
                "production plan type_to_species must contain non-empty symbols"
            )

    high_steps_value = _validated_plan_integer(
        "high_total_steps", high_total_steps, minimum=1
    )
    quench_steps_value = _validated_plan_integer("quench_steps", quench_steps, minimum=1)
    relax_steps_value = _validated_plan_integer("relax_steps", relax_steps, minimum=0)
    msd_every_value = _validated_plan_integer("msd_every", msd_every, minimum=1)
    seed_base_value = _validated_plan_integer("seed_base", seed_base, minimum=0)

    plan = ProductionPlan(
        engine=engine_norm,
        structure_data=Path(structure_data),
        T_high=float(T_high),
        high_total_steps=high_steps_value,
        t_final=float(t_final),
        chosen_rate=float(chosen_rate),
        cooling_rate_ps=(None if cooling_rate_ps is None else float(cooling_rate_ps)),
        replicate=(rep[0], rep[1], rep[2]),
        pressure=float(pressure),
        md_use=md_data,
        potential_config=potential_data,
        potential_lines=potential_line_list,
        core_repulsion=core_data,
        type_to_species=species_data,
        metrics_cfg=metrics_data,
        effective_metrics=effective_data,
        production_cfg=production_data,
        convergence_cfg=convergence_data,
        cutoffs_rate=_cutoffs_list_from_dict(rate_cutoffs),
        cutoffs_size=_cutoffs_list_from_dict(size_cutoffs),
        preferred_cutoffs=_cutoffs_list_from_dict(preferred),
        quench_steps=quench_steps_value,
        relax_steps=relax_steps_value,
        msd_every=msd_every_value,
        seed_base=seed_base_value,
        time_unit_ps=(None if time_unit_ps is None else float(time_unit_ps)),
        sampling_hint=sampling_data,
        execution_mode=str(execution_mode),
        source_kind=str(source_kind),
    )
    return validate_production_plan(plan, require_structure_file=True)


def production_plan_to_dict(plan: ProductionPlan, *, relative_to: Optional[Path] = None) -> dict[str, Any]:
    structure_data = Path(plan.structure_data).expanduser()
    if relative_to is not None:
        # A public output directory may be supplied as a relative path.  A
        # fresh run then commonly carries ``out/structure/input.data`` while
        # resume has resolved the same stored path to an absolute filename.
        # Compare canonical operands so both representations serialize to the
        # same output-relative plan.  If the structure is outside the output
        # tree, persist its resolved absolute path; otherwise a relative path
        # would be reinterpreted beneath ``relative_to`` on replay.
        structure_resolved = structure_data.resolve(strict=False)
        relative_root = Path(relative_to).expanduser().resolve(strict=False)
        try:
            structure_data = structure_resolved.relative_to(relative_root)
        except ValueError:
            structure_data = structure_resolved
    return {
        "schema": "vitriflow.production_plan.v1",
        "engine": str(plan.engine),
        "structure_data": str(structure_data),
        "T_high": float(plan.T_high),
        "high_total_steps": int(plan.high_total_steps),
        "t_final": float(plan.t_final),
        "chosen_rate": float(plan.chosen_rate),
        "cooling_rate_ps": (None if plan.cooling_rate_ps is None else float(plan.cooling_rate_ps)),
        "replicate": [int(plan.replicate[0]), int(plan.replicate[1]), int(plan.replicate[2])],
        "pressure": float(plan.pressure),
        "md_use": dict(plan.md_use),
        "potential_config": (None if plan.potential_config is None else dict(plan.potential_config)),
        "potential_lines": (None if plan.potential_lines is None else list(plan.potential_lines)),
        "core_repulsion": (None if plan.core_repulsion is None else dict(plan.core_repulsion)),
        "type_to_species": (None if plan.type_to_species is None else list(plan.type_to_species)),
        "metrics_cfg": dict(plan.metrics_cfg),
        "effective_metrics": dict(plan.effective_metrics),
        "production_cfg": dict(plan.production_cfg),
        "convergence_cfg": dict(plan.convergence_cfg),
        "cutoffs_rate": list(plan.cutoffs_rate),
        "cutoffs_size": list(plan.cutoffs_size),
        "preferred_cutoffs": list(plan.preferred_cutoffs),
        "quench_steps": int(plan.quench_steps),
        "relax_steps": int(plan.relax_steps),
        "msd_every": int(plan.msd_every),
        "seed_base": int(plan.seed_base),
        "time_unit_ps": (None if plan.time_unit_ps is None else float(plan.time_unit_ps)),
        "sampling_hint": (None if plan.sampling_hint is None else dict(plan.sampling_hint)),
        "execution_mode": str(plan.execution_mode),
        "source_kind": str(plan.source_kind),
    }


def production_plan_from_dict(data: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> ProductionPlan:
    if not isinstance(data, Mapping):
        raise ValueError("production plan must be a mapping")
    schema = str(data.get("schema", "") or "").strip().lower()
    if schema != "vitriflow.production_plan.v1":
        raise ValueError(
            f"Unsupported production plan schema {data.get('schema')!r}; "
            "expected 'vitriflow.production_plan.v1'"
        )
    allowed = {
        "schema", "engine", "structure_data", "T_high", "high_total_steps",
        "t_final", "chosen_rate", "cooling_rate_ps", "replicate", "pressure",
        "md_use", "potential_config", "potential_lines", "core_repulsion",
        "type_to_species", "metrics_cfg", "effective_metrics", "production_cfg",
        "convergence_cfg", "cutoffs_rate", "cutoffs_size", "preferred_cutoffs",
        "quench_steps", "relax_steps", "msd_every", "seed_base", "time_unit_ps",
        "sampling_hint", "execution_mode", "source_kind",
    }
    unknown = sorted(str(key) for key in data.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(
            "production plan contains unknown field(s): " + ", ".join(unknown)
        )
    # A versioned production plan is an exact replay contract.  Every field
    # emitted by production_plan_to_dict must be present, including nullable
    # fields; silently filling defaults here can change the engine, pressure,
    # seeds, cutoffs, metric behavior, or stopping semantics.
    missing = sorted(key for key in allowed if key not in data)
    if missing:
        raise ValueError(
            "production plan is missing required field(s): " + ", ".join(missing)
        )
    structure_data = Path(str(data.get("structure_data", "")))
    if not structure_data.is_absolute() and base_dir is not None:
        structure_data = (Path(base_dir) / structure_data).resolve(strict=False)
    potential_config = data.get("potential_config", None)
    if isinstance(potential_config, Mapping):
        potential_config = dict(potential_config)
        raw_files = potential_config.get("files", None)
        if isinstance(raw_files, Sequence) and not isinstance(
            raw_files, (str, bytes, bytearray)
        ):
            resolved_files: list[str] = []
            for index, raw in enumerate(raw_files):
                if raw is None or not str(raw).strip():
                    raise ValueError(
                        f"production plan potential_config.files entry at index {index} "
                        "must be a non-empty path"
                    )
                p = Path(str(raw)).expanduser()
                if not p.is_absolute() and base_dir is not None:
                    p = Path(base_dir) / p
                if not p.is_file():
                    raise ValueError(
                        f"production plan potential_config.files entry is not a file: {p}"
                    )
                resolved_files.append(str(p.resolve(strict=False)))
            potential_config["files"] = resolved_files
    return make_production_plan(
        engine=str(data["engine"]),
        structure_data=structure_data,
        T_high=float(data["T_high"]),
        high_total_steps=data["high_total_steps"],
        t_final=float(data["t_final"]),
        chosen_rate=float(data["chosen_rate"]),
        cooling_rate_ps=(None if data["cooling_rate_ps"] is None else float(data["cooling_rate_ps"])),
        replicate=data["replicate"],
        pressure=float(data["pressure"]),
        md_use=data["md_use"],
        potential_config=potential_config,
        potential_lines=data["potential_lines"],
        core_repulsion=data["core_repulsion"],
        type_to_species=data["type_to_species"],
        metrics_cfg=data["metrics_cfg"],
        effective_metrics=data["effective_metrics"],
        production_cfg=data["production_cfg"],
        convergence_cfg=data["convergence_cfg"],
        cutoffs_rate=data["cutoffs_rate"],
        cutoffs_size=data["cutoffs_size"],
        preferred_cutoffs=data["preferred_cutoffs"],
        quench_steps=data["quench_steps"],
        relax_steps=data["relax_steps"],
        msd_every=data["msd_every"],
        seed_base=data["seed_base"],
        time_unit_ps=(None if data["time_unit_ps"] is None else float(data["time_unit_ps"])),
        sampling_hint=data["sampling_hint"],
        execution_mode=str(data["execution_mode"]),
        source_kind=str(data["source_kind"]),
    )


def run_production_ensemble(**kwargs: Any) -> dict[str, Any]:
    """TRANSITIONAL SHIM — architectural finding remains OPEN.

    The project design guide requires that run/autotune/run-schedule control flow must be
    separate and that shared production logic belongs in `production_common`.
    Today the implementation (`_ProductionEnsembleRunner` and
    `_run_production_ensemble`) still lives in `autotune.py`; this function
    only redirects `run.py` away from importing autotune internals directly.

    DO NOT describe this as an architecture fix. It is a name-aliasing shim:
    `run.py` calls `production_common.run_production_ensemble(...)` instead
    of `autotune._run_production_ensemble(...)`, but the call still lands in
    autotune via the lazy import below. The cross-runner dependency persists
    and a real fix requires physically migrating the runner class to this
    module (or to a dedicated `production_runner` module).

    Why we kept the shim instead of resolving the finding:
      * 0.4.29.4 is a regression-protection release; moving ~1300 LOC of
        ensemble runner is not surgical.
      * Renaming the call site now means the eventual migration only edits
        this one file, not every caller.
      * The `_run_production_ensemble` reference is still discoverable to
        anyone grepping autotune, which is the honest state of affairs.

    The targeted test in `tests/test_production_runner_shim_is_transitional.py`
    pins this status so a future PR cannot silently rebrand the shim as a
    completed fix.
    """

    # Lazy import: required because autotune already imports from
    # production_common at module top, so a top-level import here would
    # cycle. NOT a design feature -- another reason to migrate the impl.
    from .autotune import _run_production_ensemble

    return _run_production_ensemble(**kwargs)


def production_plan_from_source(source: Optional[Mapping[str, Any]], *, base_dir: Optional[Path] = None) -> Optional[ProductionPlan]:
    if source is None:
        return None
    if isinstance(source.get("production_plan", None), Mapping):
        return production_plan_from_dict(source.get("production_plan", {}), base_dir=base_dir)
    if str(source.get("schema", "")).strip().lower() == "vitriflow.production_plan.v1":
        return production_plan_from_dict(source, base_dir=base_dir)
    required = {"structure_data", "T_high", "high_total_steps", "t_final", "chosen_rate", "replicate", "md_use", "production_cfg", "convergence_cfg", "quench_steps", "relax_steps"}
    if required.issubset(set(source.keys())):
        return production_plan_from_dict(source, base_dir=base_dir)
    return None


def resolve_production_time_unit_ps(
    *,
    config=None,
    engine: Optional[str] = None,
    pot_cfg=None,
    time_unit_ps: Optional[float] = None,
) -> Optional[float]:
    """Production time unit."""

    eng = str(engine if engine is not None else getattr(config, "engine", "lammps") or "lammps").strip().lower()
    if eng == "cp2k":
        return 0.001

    try:
        val = float(time_unit_ps) if time_unit_ps is not None else None
    except Exception:
        val = None
    if val is not None and math.isfinite(val) and val > 0.0:
        return float(val)

    units_style = resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal")
    return lammps_timeunit_ps(units_style)


def resolve_production_warmup_start_temperature(*, prod_cfg, T_high: Optional[float] = None) -> float:
    """Production warmup start."""

    raw = getattr(prod_cfg, "warmup_start_temperature", 300.0)
    try:
        T0 = float(raw)
    except Exception as exc:  # pragma: no cover - defensive against malformed ad-hoc configs
        raise ValueError("production warm-up start temperature must be a real number") from exc
    if not math.isfinite(T0) or T0 <= 0.0:
        raise ValueError("production warm-up start temperature must be finite and > 0")
    if T_high is not None:
        Thi = float(T_high)
        if math.isfinite(Thi) and T0 > Thi:
            raise ValueError(
                "production warm-up start temperature must be <= T_high "
                f"(got {T0:g} K > {Thi:g} K)"
            )
    return T0


def resolve_production_warmup_duration_ps(*, prod_cfg) -> float:
    """Production warmup duration."""

    raw = getattr(prod_cfg, "warmup_duration_ps", 5.0)
    try:
        dur_ps = float(raw)
    except Exception as exc:  # pragma: no cover - defensive against malformed ad-hoc configs
        raise ValueError("production warm-up duration must be a real number") from exc
    if not math.isfinite(dur_ps) or dur_ps <= 0.0:
        raise ValueError("production warm-up duration must be finite and > 0")
    return dur_ps


def resolve_production_warmup_steps(
    *,
    prod_cfg,
    md_timestep: float,
    time_unit_ps: Optional[float],
) -> int:
    """Production warmup steps."""

    dt = float(md_timestep)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("production warm-up timestep must be finite and > 0")

    tu_ps = None if time_unit_ps is None else float(time_unit_ps)
    if tu_ps is None or not math.isfinite(tu_ps) or tu_ps <= 0.0:
        raise ValueError(
            "production warm-up duration is specified in ps, but the MD time unit could not be resolved"
        )

    dur_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    step_ps = float(dt) * float(tu_ps)
    if not math.isfinite(step_ps) or step_ps <= 0.0:
        raise ValueError("production warm-up step size must be finite and > 0 ps")
    return max(1, int(math.ceil(float(dur_ps) / float(step_ps))))


def _slug(value: Any) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "metric"


def _json_float_vector(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _json_float_matrix(values: Any) -> list[list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        arr = np.asarray(arr.reshape((-1, 3)), dtype=float)
    return [[float(x) for x in row] for row in arr.tolist()]


def _frame_lattice_json(frame: Any) -> dict[str, Any]:
    cell = np.asarray(getattr(frame, "cell"), dtype=float)
    origin = np.asarray(getattr(frame, "origin", np.zeros(3, dtype=float)), dtype=float)
    volume = None
    try:
        volume = abs(float(np.linalg.det(cell)))
    except Exception:
        volume = None
    return {
        "cell": _json_float_matrix(cell),
        "origin": _json_float_vector(origin),
        "pbc": [bool(x) for x in getattr(frame, "pbc", (True, True, True))],
        "volume": volume,
        "vectors_are_rows": True,
        "units": "angstrom",
    }


def _frame_structure_json(frame: Any, *, type_to_species: Optional[Sequence[str]]) -> dict[str, Any]:
    ids = np.asarray(getattr(frame, "ids"), dtype=int).reshape(-1)
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    positions = np.asarray(getattr(frame, "positions"), dtype=float)
    type_map = [str(x) for x in type_to_species] if type_to_species is not None else None
    species: Optional[list[str | None]] = None
    if type_map is not None:
        species = []
        for t in types.tolist():
            ti = int(t)
            species.append(type_map[ti - 1] if 1 <= ti <= len(type_map) else None)
    lattice = _frame_lattice_json(frame)
    return {
        "schema": "vitriflow.structure_snapshot.v1",
        "frame_role": "final",
        "coordinate_system": "cartesian",
        "position_units": "angstrom",
        "timestep": int(getattr(frame, "timestep", 0)),
        "n_atoms": int(positions.shape[0]),
        "type_to_species": type_map,
        "ids": [int(x) for x in ids.tolist()],
        "types": [int(x) for x in types.tolist()],
        "species": species,
        "positions": _json_float_matrix(positions),
        "lattice": lattice,
    }

def plan_production_stage_diagnostics(
    *,
    prod_cfg,
    metrics_cfg,
    runner,
    force_isotropic: bool,
    total_quench_steps: int,
    temperature_start: float,
    temperature_stop: float,
    sampling_hint: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Production stage diagnostics."""

    dump_traj = bool(getattr(prod_cfg, "dump_trajectory", True))
    dump_every = int(getattr(prod_cfg, "dump_every_steps", 5000) or 5000)
    if dump_every < 1:
        dump_every = 1

    collect_stage_metric_series = should_collect_stage_metrics_timeseries(metrics_cfg)
    collect_elastic_series = {
        role: should_collect_elastic_stage_timeseries(
            metrics_cfg,
            runner=runner,
            stage_role=role,
            force_isotropic=bool(force_isotropic),
        )[0]
        for role in ("melt", "quench", "relax")
    }
    need_stage_dump = {
        role: bool(dump_traj or collect_stage_metric_series or collect_elastic_series[role])
        for role in ("melt", "quench", "relax")
    }

    quench_min_window_frames = 1
    if bool(collect_stage_metric_series):
        quench_min_window_frames = max(
            quench_min_window_frames,
            int(getattr(metrics_cfg, "quench_tail_min_frames", 24) or 24),
        )
    if bool(collect_elastic_series.get("quench", False)):
        quench_min_window_frames = max(
            quench_min_window_frames,
            int(getattr(metrics_cfg.elastic, "quench_tail_min_frames", 12) or 12),
        )

    quench_dump_every = recommended_quench_dump_every(
        total_steps=int(total_quench_steps),
        temperature_start=float(temperature_start),
        temperature_stop=float(temperature_stop),
        base_dump_every=int(dump_every),
        sampling_hint=sampling_hint,
        min_window_frames=int(quench_min_window_frames),
    )
    quench_window_steps_range = quench_window_steps(
        T_start=float(temperature_start),
        T_stop=float(temperature_stop),
        total_steps=int(total_quench_steps),
        T_upper=(sampling_hint or {}).get("Tm") if sampling_hint is not None else None,
        T_lower=(sampling_hint or {}).get("freeze_temperature") if sampling_hint is not None else None,
    )

    return {
        "dump_traj": bool(dump_traj),
        "dump_every": int(dump_every),
        "collect_stage_metric_series": bool(collect_stage_metric_series),
        "collect_elastic_series": dict(collect_elastic_series),
        "need_stage_dump": dict(need_stage_dump),
        "quench_min_window_frames": int(quench_min_window_frames),
        "quench_dump_every": int(quench_dump_every),
        "quench_window_steps_range": quench_window_steps_range,
    }


def resolve_production_relax_dump_settings(*, stage_diag: Mapping[str, Any], metrics_cfg) -> dict[str, Any]:
    """Production relax dump."""

    need_stage_dump = bool((stage_diag.get("need_stage_dump", {}) or {}).get("relax", False))
    dump_every = int(stage_diag.get("dump_every", 0) or 0)
    metrics_enabled = bool(getattr(metrics_cfg, "enabled", False))
    tail_frames = int(getattr(metrics_cfg, "time_average_frames", 0) or 0)
    tail_stride = int(getattr(metrics_cfg, "time_average_stride", 0) or 0)

    if need_stage_dump:
        return {
            "write_dump": True,
            "dump_every": (dump_every if dump_every > 0 else None),
            "tail_dump_frames": None,
            "tail_dump_stride": None,
            "mode": "full",
        }

    if metrics_enabled and tail_frames > 0 and tail_stride > 0:
        return {
            "write_dump": True,
            "dump_every": None,
            "tail_dump_frames": int(tail_frames),
            "tail_dump_stride": int(tail_stride),
            "mode": "tail_only",
        }

    return {
        "write_dump": False,
        "dump_every": None,
        "tail_dump_frames": None,
        "tail_dump_stride": None,
        "mode": "none",
    }





def configured_graph_rules(metrics_cfg: Any) -> tuple[Any, ...]:
    """Return only graph rules explicitly supplied by the user.

    This is the authoritative opt-in boundary for enhanced graph analysis.
    Legacy cutoff-driven analysis must not infer a graph request merely because
    it has a non-empty cutoff map.
    """

    if isinstance(metrics_cfg, Mapping):
        raw_rules = metrics_cfg.get("graph_rules", None)
    else:
        raw_rules = getattr(metrics_cfg, "graph_rules", None)
    if raw_rules is None:
        return ()
    if isinstance(raw_rules, (str, bytes, Mapping)):
        # Invalid shapes remain explicit requests and are rejected by the graph
        # resolver.  Treating them as disabled would hide a user error.
        return (raw_rules,)
    try:
        return tuple(raw_rules)
    except TypeError:
        return (raw_rules,)


def graph_analysis_requested(metrics_cfg: Any) -> bool:
    """Whether enhanced graph analysis was explicitly requested.

    Callers must use this predicate before importing graph-analysis machinery,
    streaming graph chunks, or finalising graph sidecars.  An absent or empty
    ``metrics.graph_rules`` value is the historical cutoff-only mode.
    """

    return bool(configured_graph_rules(metrics_cfg))


def entry_has_graph_analysis(entry: Any) -> bool:
    """Return whether an entry contains an active enhanced-graph payload.

    This companion predicate is safe for current, streamed, and resumed entry
    schemas and gives finalisers a payload-level guard in addition to
    :func:`graph_analysis_requested`.
    """

    if not isinstance(entry, Mapping):
        return False
    payload = entry.get("graph_analysis", None)
    if not isinstance(payload, Mapping) or payload.get("enabled", True) is False:
        return False
    return bool(
        payload.get("graph_rules")
        or payload.get("adaptive_graph_rule_records")
        or payload.get("graph_metric_rows")
        or payload.get("streamed_sidecars")
    )


def write_graph_analysis_outputs(*args: Any, **kwargs: Any) -> dict[str, str]:
    """Lazily dispatch the enhanced graph-sidecar writer.

    This preserves the shared workflow API without importing graph-metric
    dependencies during cutoff-only runs.  Callers must gate this function with
    :func:`graph_analysis_requested` or :func:`entry_has_graph_analysis`.
    """

    from ..analysis.graph_metrics import write_graph_analysis_outputs as _write

    return _write(*args, **kwargs)


def _primary_hard_graph_rule_from_analysis(graph_analysis: Mapping[str, Any]) -> Optional[Any]:
    """Return the primary single hard graph rule for backward-compatible fields.

    Adaptive analysis evaluates many graph rules for the CSV/sensitivity outputs.
    The JSON summary and defect artefacts still need a single explicitly logged
    graph to remain useful.  Prefer the non-sweep hard rule derived by the
    adaptive resolver; fall back to the first hard rule if only sweep rules were
    requested.
    """

    # Lazy by design: importing this module must not import the enhanced graph
    # stack for a legacy/default production run.
    from ..analysis.graph import GraphRule, graph_family_from_rule

    records = list((graph_analysis.get("graph_rules", []) if isinstance(graph_analysis, Mapping) else []) or [])
    fallback: Optional[Any] = None
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        try:
            rule = GraphRule.from_any(rec)
        except Exception:
            continue
        if str(rule.kind) != "hard_cutoff":
            continue
        params = dict(rule.parameters or {})
        if bool(params.get("legacy", False)):
            continue
        fam = graph_family_from_rule(rule)
        if fam == "network_graph" and params.get("rdf_sweep_fraction", None) is None:
            return rule
        if fallback is None or (fam == "network_graph" and graph_family_from_rule(fallback) != "network_graph"):
            fallback = rule
        if params.get("rdf_sweep_fraction", None) is None and graph_family_from_rule(rule) not in {"candidate_contact_graph", "soft_ambiguity_graph"}:
            return rule
    return fallback


def _metrics_have_adaptive_graph_rules(metrics_cfg: Any) -> bool:
    adaptive = {
        "rdf_adaptive",
        "rdf_adaptive_hard_cutoff",
        "rdf_adaptive_hard_cutoff_sweep",
        "rdf_adaptive_hard_cutoff_interval",
        "rdf_adaptive_soft_logistic",
        # Compatibility aliases accepted by early 0.4.29.9 drafts.
        "hard_cutoff_auto_rdf",
        "hard_cutoff_sweep_auto_rdf",
        "hard_cutoff_interval_auto_rdf",
        "soft_logistic_auto_rdf",
    }
    for rule in configured_graph_rules(metrics_cfg):
        if isinstance(rule, Mapping):
            params = rule.get("parameters", {}) or {}
            kind = str(rule.get("kind", ""))
        else:
            params = getattr(rule, "parameters", {}) or {}
            kind = str(getattr(rule, "kind", ""))
        derive = str(params.get("derive_from", "")).strip().lower() if isinstance(params, dict) else ""
        source = str(params.get("source", params.get("cutoff_source", ""))).strip().lower() if isinstance(params, dict) else ""
        if kind in adaptive or derive in {"rdf", "rdf_minimum", "rdf_first_minimum", "pair_distribution", "pair_distribution_function", "shell_separability"} or source in {"rdf", "rdf_minimum", "rdf_first_minimum", "pair_distribution", "pair_distribution_function", "shell_separability"}:
            return True
    return False

def analyse_production_box(
    *,
    box_id: int,
    outdir: Path,
    melt_stage_dir: Path,
    quench_stage_dir: Path,
    relax_stage_dir: Path,
    relax_data_path: Path,
    density_mean: float,
    density_stderr: float,
    metrics_cfg,
    cutoffs: Optional[Mapping[Tuple[int, int], float]],
    required_pairs: Sequence[tuple[Any, Any]],
    fixed_cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]],
    md_timestep: float,
    quench_window_steps_range: Optional[Tuple[float, float]] = None,
    sampling_hint: Optional[Mapping[str, float]] = None,
    bondlen_cdf_points: int = 200,
    angle_cdf_points: int = 180,
    seeds: Optional[Mapping[str, int]] = None,
    melt_elastic: Any = None,
    relax_elastic: Any = None,
    elastic_timeseries: Any = None,
    exclude_coordination_defects: bool = False,
    rejects_dir: Optional[Path] = None,
    relax_dump_path: Optional[Path] = None,
    relax_traj_path: Optional[Path] = None,
    analysis_source_path: Optional[Path] = None,
    analysis_source_role: Optional[str] = None,
    atom_style: str = "atomic",
    embed_structures: bool = True,
    lammps_units_style: Optional[str] = "metal",
    engine: str = "lammps",
) -> tuple[dict[str, Any], dict[Tuple[int, int], float]]:
    """Production box."""

    outdir = Path(outdir)
    melt_stage_dir = Path(melt_stage_dir)
    quench_stage_dir = Path(quench_stage_dir)
    relax_stage_dir = Path(relax_stage_dir)
    relax_data_path = Path(relax_data_path)
    relax_dump_path = Path(relax_dump_path) if relax_dump_path is not None else (relax_stage_dir / "relax.lammpstrj")
    relax_traj_path = Path(relax_traj_path) if relax_traj_path is not None else (
        (relax_stage_dir / "traj.extxyz") if (relax_stage_dir / "traj.extxyz").exists() else relax_dump_path
    )
    analysis_source_path = Path(analysis_source_path) if analysis_source_path is not None else None

    explicit_analysis_source = analysis_source_path is not None
    preferred_frames_source = analysis_source_path if explicit_analysis_source else relax_traj_path
    preferred_source_role = (
        str(analysis_source_role)
        if analysis_source_role not in (None, "")
        else ("explicit_analysis_source" if explicit_analysis_source else "relax_trajectory")
    )
    frames_source = Path(preferred_frames_source)
    selected_source_role = str(preferred_source_role)
    source_fallback_reason: Optional[str] = None

    if explicit_analysis_source:
        # A caller-selected scientific input is a contract. Silently replacing
        # it with a different trajectory/final structure would make the
        # recorded role and content provenance false.
        if not frames_source.is_file():
            raise FileNotFoundError(
                "Explicit production analysis source not found or not a regular file: "
                f"{frames_source}"
            )
    elif not frames_source.is_file():
        dump_candidate = Path(relax_dump_path)
        if dump_candidate != frames_source and dump_candidate.is_file():
            LOG.warning(
                "Preferred production relax trajectory missing for box %s; "
                "falling back to dump trajectory %s",
                int(box_id),
                str(dump_candidate),
            )
            frames_source = dump_candidate
            selected_source_role = "relax_trajectory"
            source_fallback_reason = "preferred_relax_trajectory_missing_or_not_file"
        elif Path(relax_data_path).is_file():
            LOG.warning(
                "Production relax trajectory missing for box %s; falling back to final structure %s",
                int(box_id),
                str(relax_data_path),
            )
            frames_source = Path(relax_data_path)
            selected_source_role = "final_structure"
            source_fallback_reason = "relax_trajectory_missing_or_not_file"
        else:
            raise FileNotFoundError(
                "Production analysis source and fallbacks were not found as regular files: "
                f"preferred={preferred_frames_source} dump={relax_dump_path} final={relax_data_path}"
            )

    # Parse an immutable copy whose bytes are locked to the public source
    # identity.  Otherwise a replacement between parsing and manifest creation
    # can pair an in-memory A frame with the SHA-256 of file B.
    from ..analysis.graph import manifest_row_from_frame, source_file_identity

    locked_source_identity = source_file_identity(frames_source)
    if not bool(locked_source_identity.get("exists", False)):
        raise FileNotFoundError(f"Production analysis source is unavailable: {frames_source}")

    def _source_identity_key(identity: Mapping[str, Any]) -> tuple[int, str]:
        try:
            size = int(identity.get("size_bytes", -1))
        except Exception:
            size = -1
        return size, str(identity.get("sha256", ""))

    snapshot_handle = tempfile.NamedTemporaryFile(
        prefix=f".vitriflow_analysis_box_{int(box_id):03d}_",
        suffix="".join(frames_source.suffixes),
        dir=str(relax_stage_dir),
        delete=False,
    )
    snapshot_path = Path(snapshot_handle.name)
    snapshot_handle.close()
    try:
        shutil.copyfile(frames_source, snapshot_path)
        snapshot_identity = source_file_identity(snapshot_path)
        source_after_copy = source_file_identity(frames_source)
        locked_key = _source_identity_key(locked_source_identity)
        if _source_identity_key(snapshot_identity) != locked_key or _source_identity_key(source_after_copy) != locked_key:
            raise RuntimeError(
                f"Production analysis source changed while creating immutable snapshot: {frames_source}"
            )
        frames = read_last_frames_auto(
            snapshot_path,
            int(metrics_cfg.time_average_frames),
            type_to_species=type_to_species,
            atom_style=str(atom_style),
            units_style=lammps_units_style,
        )
        snapshot_structure_manifest = manifest_row_from_frame(
            frames[-1],
            box_id=int(box_id),
            source_path=snapshot_path,
            source_role=selected_source_role,
            type_to_species=type_to_species,
            density=float(density_mean) if math.isfinite(float(density_mean)) else None,
        )
    finally:
        try:
            snapshot_path.unlink(missing_ok=True)
        except Exception:
            LOG.warning("Failed to remove temporary analysis snapshot %s", snapshot_path)
    cut_map: dict[Tuple[int, int], float] = dict(cutoffs or {})
    normalized_required_pairs = sorted(
        {
            (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
            for pair in required_pairs
        }
    )
    missing_required_pairs = [pair for pair in normalized_required_pairs if pair not in cut_map]
    if missing_required_pairs and not _metrics_have_adaptive_graph_rules(metrics_cfg):
        cut_map = estimate_pair_cutoffs(
            frames,
            normalized_required_pairs,
            auto=metrics_cfg.auto_cutoff,
            fixed_cutoffs={**dict(fixed_cutoffs or {}), **cut_map},
        )

    graph_requested = graph_analysis_requested(metrics_cfg)
    graph_analysis: Optional[dict[str, Any]] = None
    primary_graph_rule = None
    primary_graph = None
    primary_graph_family: Optional[str] = None
    primary_graph_cutoffs: dict[Tuple[int, int], float] = {}
    if graph_requested:
        # Enhanced graph analysis is an explicit feature and is imported only
        # inside its opt-in branch.  Missing optional graph dependencies can
        # therefore never abort the historical cutoff-only path.
        from ..analysis.graph import (
            build_graph,
            graph_family_from_rule,
            pair_cutoffs_from_parameters,
            verify_manifest_row,
        )
        from ..analysis.graph_metrics import graph_analysis_for_frame

        graph_analysis = graph_analysis_for_frame(
            frames[-1],
            metrics_cfg,
            box_id=int(box_id),
            type_to_species=type_to_species,
            legacy_cutoffs=cut_map,
            # The frame came from the verified immutable snapshot.  The
            # original public path/identity is injected only after its final
            # unchanged check below.
            source_path=None,
            source_role=selected_source_role,
            density=float(density_mean) if math.isfinite(float(density_mean)) else None,
        )
        verify_manifest_row(frames[-1], graph_analysis["structure_manifest"], type_to_species=type_to_species)

        primary_graph_rule = _primary_hard_graph_rule_from_analysis(graph_analysis)
        if primary_graph_rule is not None:
            primary_graph = build_graph(frames[-1], primary_graph_rule, type_to_species=type_to_species)
            primary_graph_cutoffs = pair_cutoffs_from_parameters(primary_graph_rule.parameters)
            primary_graph_family = str(graph_family_from_rule(primary_graph.graph_rule))

    # Coordinate-only scalars such as RDF peak descriptors and void summaries are
    # still produced by the time-average path.  Graph-derived scalars are then
    # overlaid from the explicit primary graph so adaptive analysis never leaves
    # the legacy JSON summary empty.
    sm = compute_structure_metrics_timeavg(frames, metrics_cfg, cutoffs=cut_map, type_to_species=type_to_species)
    struct_vals = dict(sm.values)
    if primary_graph is not None:
        graph_sm = compute_structure_metrics(frames[-1], metrics_cfg, cutoffs={}, type_to_species=type_to_species, graph=primary_graph)
        struct_vals.update(dict(graph_sm.values))

    stage_metrics = None
    if should_collect_stage_metrics_timeseries(metrics_cfg):
        stage_metrics = {
            "melt": collect_stage_metrics_timeseries(
                stage_dir=melt_stage_dir,
                metrics_cfg=metrics_cfg,
                cutoffs=cut_map,
                md_timestep=float(md_timestep),
                type_to_species=type_to_species,
                outdir=outdir,
                stage_role="melt",
                lammps_units_style=lammps_units_style,
                engine=str(engine),
            ),
            "quench": collect_stage_metrics_timeseries(
                stage_dir=quench_stage_dir,
                metrics_cfg=metrics_cfg,
                cutoffs=cut_map,
                md_timestep=float(md_timestep),
                type_to_species=type_to_species,
                outdir=outdir,
                stage_role="quench",
                quench_window_steps_range=quench_window_steps_range,
                sampling_hint=sampling_hint,
                lammps_units_style=lammps_units_style,
                engine=str(engine),
            ),
            "relax": collect_stage_metrics_timeseries(
                stage_dir=relax_stage_dir,
                metrics_cfg=metrics_cfg,
                cutoffs=cut_map,
                md_timestep=float(md_timestep),
                type_to_species=type_to_species,
                outdir=outdir,
                stage_role="relax",
                lammps_units_style=lammps_units_style,
                engine=str(engine),
            ),
        }

    if primary_graph is not None:
        from ..analysis.structure import compute_structure_distributions_for_graph

        dist = compute_structure_distributions_for_graph(
            frames[-1],
            metrics_cfg,
            graph=primary_graph,
            type_to_species=type_to_species,
            bondlen_cdf_points=int(bondlen_cdf_points),
            angle_cdf_points=int(angle_cdf_points),
        )
    else:
        dist = compute_structure_distributions_timeavg(
            frames,
            metrics_cfg,
            cutoffs=cut_map,
            type_to_species=type_to_species,
            bondlen_cdf_points=int(bondlen_cdf_points),
            angle_cdf_points=int(angle_cdf_points),
        )

    gr_curves: dict[str, Any] = {}
    for gm in list(metrics_cfg.gr):
        label = "all" if gm.pair is None else f"{gm.pair[0]}-{gm.pair[1]}"
        key = f"gr_{_slug(label)}"
        if key in gr_curves:
            previous_label = str(gr_curves[key].get("label", ""))
            raise ValueError(
                "Duplicate generated g(r) curve key "
                f"'{key}' for labels '{previous_label}' and '{label}'; "
                "metric labels must be unique after slug normalization"
            )
        r, g, _l = compute_gr(
            frames,
            r_max=float(gm.r_max),
            nbins=int(gm.nbins),
            pair=gm.pair,
            type_to_species=type_to_species,
        )
        gr_curves[key] = {"label": str(label), "r": [float(v) for v in r.tolist()], "g": [float(v) for v in g.tolist()]}

    sq_curves: dict[str, Any] = {}
    if hasattr(metrics_cfg, "sq"):
        from ..analysis.sq import compute_sq

        for sm_cfg in list(getattr(metrics_cfg, "sq", [])):
            label = "all" if sm_cfg.pair is None else f"{sm_cfg.pair[0]}-{sm_cfg.pair[1]}"
            key = f"sq_{_slug(label)}"
            if key in sq_curves:
                previous_label = str(sq_curves[key].get("label", ""))
                raise ValueError(
                    "Duplicate generated S(q) curve key "
                    f"'{key}' for labels '{previous_label}' and '{label}'; "
                    "metric labels must be unique after slug normalization"
                )
            q, s, representation = compute_sq(
                frames,
                q_max=float(sm_cfg.q_max),
                nq=int(sm_cfg.nq),
                r_max=float(sm_cfg.r_max),
                nbins=int(sm_cfg.nbins),
                pair=sm_cfg.pair,
                type_to_species=type_to_species,
                window=str(getattr(sm_cfg, "window", "lorch")),
                return_metadata=True,
            )
            sq_curves[key] = {
                "label": str(label),
                "q": [float(v) for v in q.tolist()],
                "s": [float(v) for v in s.tolist()],
                "representation": dict(representation),
            }

    dist_all = {
        "bondlen": dist.get("bondlen", {}),
        "angle": dist.get("angle", {}),
        "coord": dist.get("coord", {}),
        "void": dist.get("void", {}),
        "gr": gr_curves,
        "sq": sq_curves,
    }

    coord_defects = compute_coordination_defects(
        frames[-1],
        metrics_cfg,
        cutoffs=(primary_graph_cutoffs if primary_graph is not None else cut_map),
        type_to_species=type_to_species,
        graph=primary_graph,
    )
    has_coord_defects = any(
        bool((v or {}).get("has_defect", False)) for v in (coord_defects or {}).values()
    )

    coord_defect_details: dict[str, Any] = {}
    coord_defect_artifacts: dict[str, Any] = {}
    coord_artifact_paths: list[Path] = []
    coordination_sweep_enabled = bool(
        getattr(getattr(metrics_cfg, "coordination_sweep", None), "enabled", False)
    )
    collect_coordination_details = bool(
        metrics_cfg.enabled
        and len(list(metrics_cfg.coordinations)) > 0
        and (
            coordination_sweep_enabled
            or exclude_coordination_defects
            or has_coord_defects
        )
    )
    if collect_coordination_details:
        try:
            coord_defect_details = compute_coordination_defect_details(
                frames[-1],
                metrics_cfg,
                cutoffs=(primary_graph_cutoffs if primary_graph is not None else cut_map),
                type_to_species=type_to_species,
                graph=primary_graph,
            )
            for _detail in (coord_defect_details or {}).values():
                if isinstance(_detail, dict) and graph_requested:
                    for _prov in (coord_defects or {}).values():
                        if isinstance(_prov, dict) and _prov.get("graph_rule", None) is not None:
                            _detail.setdefault("graph_rule", _prov.get("graph_rule"))
                            _detail.setdefault("structure_hash", _prov.get("structure_hash"))
                            break

            cd_json = relax_stage_dir / "coordination_defects_detail.json"
            write_json_strict(cd_json, coord_defect_details)
            coord_defect_artifacts["detail_json"] = _relpath_or_str(cd_json, outdir)
            coord_defect_artifacts["status"] = "ok"
            coord_defect_artifacts["coordination_sweep_enabled"] = bool(
                coordination_sweep_enabled
            )
            coord_artifact_paths.append(cd_json)

            defect_idx: set[int] = set()
            shell_idx: set[int] = set()
            for vv in (coord_defect_details or {}).values():
                for i in (vv or {}).get("defective_idx", []) or []:
                    defect_idx.add(int(i))
                for i in (vv or {}).get("shell_idx", []) or []:
                    shell_idx.add(int(i))

            fr_final = frames[-1]
            n_atoms = int(fr_final.n_atoms)
            if type_to_species is not None:
                sp: list[str] = []
                for t in fr_final.types.tolist():
                    ti = int(t)
                    if 1 <= ti <= len(type_to_species):
                        sp.append(str(type_to_species[ti - 1]))
                    else:
                        sp.append("X")
            else:
                sp = ["X"] * n_atoms

            for idx in sorted(defect_idx):
                if 0 <= int(idx) < n_atoms:
                    if sp[int(idx)] == "Si":
                        sp[int(idx)] = "Sm"
                    elif sp[int(idx)] == "N":
                        sp[int(idx)] = "O"

            marked_path = relax_stage_dir / "coordination_defects_marked.extxyz"
            write_extxyz_single_with_species(marked_path, fr_final, sp, wrap=True)
            coord_defect_artifacts["marked_extxyz"] = _relpath_or_str(marked_path, outdir)
            coord_artifact_paths.append(marked_path)

            if shell_idx:
                from ..analysis.dump import DumpFrame

                idxs = sorted([int(i) for i in shell_idx if 0 <= int(i) < n_atoms])
                fr_shell = DumpFrame(
                    timestep=int(fr_final.timestep),
                    ids=np.asarray(fr_final.ids[idxs], dtype=int),
                    types=np.asarray(fr_final.types[idxs], dtype=int),
                    positions=np.asarray(fr_final.positions[idxs], dtype=float),
                    cell=np.asarray(fr_final.cell, dtype=float),
                    origin=np.asarray(fr_final.origin, dtype=float),
                    pbc=tuple(bool(x) for x in getattr(fr_final, "pbc", (True, True, True))),
                )
                sp_shell = [sp[int(i)] for i in idxs]
                shell_path = relax_stage_dir / "coordination_defects_shell.extxyz"
                write_extxyz_single_with_species(shell_path, fr_shell, sp_shell, wrap=True)
                coord_defect_artifacts["shell_extxyz"] = _relpath_or_str(shell_path, outdir)
                coord_artifact_paths.append(shell_path)
        except Exception as e:
            coord_defect_artifacts["error"] = str(e)
            coord_defect_artifacts["status"] = "failed"
            coord_defect_artifacts["coordination_sweep_enabled"] = bool(
                coordination_sweep_enabled
            )
            if coordination_sweep_enabled:
                raise RuntimeError(
                    "Configured coordination_sweep analysis failed"
                ) from e

    amorphous_report: dict[str, Any] = {}
    amorphous_artifacts: dict[str, Any] = {}
    is_amorphous = True
    if bool(getattr(getattr(metrics_cfg, "amorphous", None), "enabled", False)):
        formula_override = reduced_formula_from_frame(frames[0], type_to_species=type_to_species)
        amorphous_report = analyse_amorphous_state(
            frames,
            metrics_cfg=metrics_cfg,
            cutoffs=cut_map,
            type_to_species=type_to_species,
            cache_dir=(outdir / "amorphous_references"),
            formula_override=formula_override,
        )
        struct_vals.update({str(k): float(v) for k, v in dict(amorphous_report.get("scalar_metrics", {})).items()})
        ar_json = relax_stage_dir / "amorphous_state.json"
        write_json_strict(ar_json, amorphous_report)
        amorphous_artifacts["state_json"] = _relpath_or_str(ar_json, outdir)
        is_amorphous = bool(amorphous_report.get("passed", True))
        coord_artifact_paths.append(ar_json)

    # A structure manifest is a mandatory provenance record, not a request for
    # enhanced graph analysis.  Reuse the graph path's row when graph analysis
    # was explicitly requested; otherwise hash the exact frame analysed here.
    # This is pure serialization/hashing and does not build a graph or change the
    # legacy cutoff-driven default analysis path.
    # This row was generated while the verified immutable snapshot still
    # existed, so its frame hashes, reader metadata and file bytes form one
    # atomic provenance fact.  Replace only its public source path/identity
    # after checking the original remained unchanged.
    structure_manifest = dict(snapshot_structure_manifest)

    source_identity_final = source_file_identity(frames_source)
    if _source_identity_key(source_identity_final) != _source_identity_key(locked_source_identity):
        raise RuntimeError(
            "Production analysis source changed after its immutable snapshot was parsed; "
            f"refusing a mixed-content manifest for {frames_source}"
        )

    # Public paths must remain meaningful if the result directory is moved.  The
    # source content identity still proves which artifact supplied the frame.
    source_rel = _relpath_or_str(Path(frames_source), outdir)
    preferred_source_rel = _relpath_or_str(Path(preferred_frames_source), outdir)
    source_selection = {
        "schema": "vitriflow.analysis_source_selection.v1",
        "explicit_source_requested": bool(explicit_analysis_source),
        "preferred_path": preferred_source_rel,
        "preferred_role": str(preferred_source_role),
        "selected_path": source_rel,
        "selected_role": str(selected_source_role),
        "fallback_used": bool(source_fallback_reason is not None),
        "fallback_reason": source_fallback_reason,
    }
    structure_manifest["source_path"] = source_rel
    structure_manifest["source_role"] = str(selected_source_role)
    structure_manifest["source_selection"] = dict(source_selection)
    structure_manifest["file_size"] = int(locked_source_identity["size_bytes"])
    # mtime from the temporary snapshot is not metadata for the public source;
    # content SHA-256 and size are the authoritative identity.
    structure_manifest["mtime"] = None
    structure_manifest["source_identity_lock"] = "immutable_snapshot_sha256_size"
    structure_manifest["source_file_identity"] = {
        **dict(locked_source_identity),
        "path": source_rel,
    }
    if isinstance(graph_analysis, dict):
        # Keep the graph payload's manifest byte-for-byte consistent with the
        # mandatory per-box manifest after paths are made result-relative.
        graph_analysis["structure_manifest"] = dict(structure_manifest)
    final_lattice_snapshot = _frame_lattice_json(frames[-1])
    if bool(embed_structures):
        final_structure_payload: dict[str, Any] = _frame_structure_json(frames[-1], type_to_species=type_to_species)
        final_lattice_payload: dict[str, Any] = dict(final_structure_payload.get("lattice", final_lattice_snapshot))
    else:
        final_structure_payload = {
            "schema": "vitriflow.structure_snapshot_ref.v1",
            "embedded": False,
            "frame_role": "final",
            "structure_hash": structure_manifest.get("structure_hash"),
            "cell_hash": structure_manifest.get("cell_hash"),
            "positions_hash": structure_manifest.get("positions_hash"),
            "symbols_hash": structure_manifest.get("symbols_hash"),
            "n_atoms": int(getattr(frames[-1], "n_atoms", len(getattr(frames[-1], "ids", [])))),
            "source_path": structure_manifest.get("source_path"),
            "source_role": structure_manifest.get("source_role"),
            "source_file_identity": structure_manifest.get("source_file_identity"),
            "note": "Coordinates are intentionally not embedded; reload source_path and verify its size/SHA-256 and the reconstructed structure hashes before use.",
        }
        final_lattice_payload = dict(final_lattice_snapshot)
        final_lattice_payload.setdefault("structure_hash", structure_manifest.get("structure_hash"))
        final_lattice_payload.setdefault("embedded_structure", False)

    # Every completed box receives concrete, strict-JSON provenance sidecars.
    # The snapshot is either the full structure or an explicit verified
    # reference, according to embed_structures; the manifest is always present.
    structure_snapshot_path = relax_stage_dir / "structure_snapshot.json"
    structure_manifest_path = relax_stage_dir / "structure_manifest.json"
    write_json_strict(structure_snapshot_path, final_structure_payload)
    write_json_strict(
        structure_manifest_path,
        {
            "schema": "vitriflow.structure_manifest.v2",
            "structures": [structure_manifest],
        },
    )

    entry = {
        "box": int(box_id),
        "density": float(density_mean),
        "density_stderr": float(density_stderr),
        "metrics": struct_vals,
        "distributions": dist_all,
        "structure_embedded": bool(embed_structures),
        "structure": final_structure_payload,
        "lattice": final_lattice_payload,
        "has_coordination_defects": bool(has_coord_defects),
        "coordination_defects": coord_defects,
        "coordination_defect_details": coord_defect_details,
        "is_amorphous": bool(is_amorphous),
        "amorphous": amorphous_report,
        "paths": {
            "melt_dir": _relpath_or_str(melt_stage_dir, outdir),
            "quench_dir": _relpath_or_str(quench_stage_dir, outdir),
            "relax_dir": _relpath_or_str(relax_stage_dir, outdir),
            "relax_data": _relpath_or_str(relax_data_path, outdir),
            "relax_dump": (_relpath_or_str(relax_dump_path, outdir) if relax_dump_path.exists() else None),
            "relax_traj": (_relpath_or_str(relax_traj_path, outdir) if relax_traj_path.exists() else None),
            "analysis_source": (_relpath_or_str(frames_source, outdir) if Path(frames_source).exists() else None),
            "structure_snapshot": _relpath_or_str(structure_snapshot_path, outdir),
            "structure_manifest": _relpath_or_str(structure_manifest_path, outdir),
            "coord_defects": coord_defect_artifacts,
            "amorphous": amorphous_artifacts,
        },
        "elastic_melt": melt_elastic,
        "elastic_relax": relax_elastic,
        "elastic_timeseries": elastic_timeseries,
        "stage_metrics": stage_metrics,
        "structure_manifest": structure_manifest,
        "analysis_source_role": str(selected_source_role),
        "analysis_source_selection": source_selection,
    }
    if graph_analysis is not None:
        entry.update(
            {
                "graph_analysis": graph_analysis,
                "single_rule_output": {
                    "present": bool(primary_graph is not None),
                    "label": (
                        "adaptive_primary_network_graph_rule"
                        if primary_graph is not None and primary_graph_family == "network_graph"
                        else ("adaptive_primary_graph_rule" if primary_graph is not None else None)
                    ),
                    "graph_rule": (primary_graph.graph_rule.to_json() if primary_graph is not None else None),
                    "structure_hash": (str(primary_graph.structure_hash) if primary_graph is not None else None),
                    "cutoffs": _cutoffs_list_from_dict(primary_graph_cutoffs),
                    "note": (
                        "Backward-compatible JSON metrics, distributions and coordination_defects were evaluated with the explicit primary graph rule; "
                        "per-rule descriptors are separated in graph_metric_by_rule.csv."
                        if primary_graph is not None
                        else "Enhanced graph analysis was explicitly requested, but no primary hard graph rule was selected."
                    ),
                },
            }
        )
    if seeds is not None:
        for k, v in dict(seeds).items():
            entry[f"seed_{k}"] = int(v)

    reject_reasons: list[str] = []
    if exclude_coordination_defects and bool(has_coord_defects):
        reject_reasons.append("coordination_defects")
    if bool(getattr(getattr(metrics_cfg, "amorphous", None), "enabled", False)) and bool(getattr(getattr(metrics_cfg, "amorphous", None), "enforce_during_production", False)) and not bool(is_amorphous):
        reject_reasons.append("non_amorphous")
    if len(reject_reasons) > 0:
        entry["reject"] = _materialise_reject_payload(
            outdir=outdir,
            box_id=int(box_id),
            relax_stage_dir=relax_stage_dir,
            relax_data_path=relax_data_path,
            relax_dump_path=relax_dump_path,
            rejects_dir=rejects_dir,
            reasons=reject_reasons,
            extra_files=coord_artifact_paths,
        )
        if graph_requested:
            for _prov in (coord_defects or {}).values():
                if isinstance(_prov, dict) and _prov.get("graph_rule", None) is not None:
                    entry["reject"].setdefault("graph_rule", _prov.get("graph_rule"))
                    entry["reject"].setdefault("structure_hash", _prov.get("structure_hash"))
                    break
        if graph_requested and "non_amorphous" in reject_reasons and isinstance(amorphous_report, dict):
            if amorphous_report.get("graph_rule", None) is not None:
                entry["reject"].setdefault("graph_rule", amorphous_report.get("graph_rule"))
            if amorphous_report.get("structure_hash", None) is not None:
                entry["reject"].setdefault("structure_hash", amorphous_report.get("structure_hash"))
            entry["reject"].setdefault("graph_filter", "legacy_single_cutoff_amorphous_local_order")

    return json_sanitize(entry), cut_map


_CONVERGENCE_FAMILY_TO_GROUP: dict[str, str] = {
    "density": "long",
    "bondlen_scalar": "short",
    "bondlen_cdf": "short",
    "angle_scalar": "short",
    "angle_cdf": "short",
    "coord_scalar": "short",
    "coord_cdf": "short",
    "ring": "medium",
    "gr_peak": "short",
    "gr_curve": "long",
    "sq_curve": "long",
    "void_cdf": "long",
}


def _configured_convergence_families(metrics_cfg: Any) -> list[str]:
    """Return every configured family with a defined convergence criterion.

    This is deliberately derived from configuration rather than from the first
    observed box.  Otherwise a wholly missing family can disappear from the
    criterion after seeing the data and produce a false pass.
    """

    def _value(name: str, default: Any) -> Any:
        if metrics_cfg is None:
            return default
        if isinstance(metrics_cfg, Mapping):
            return metrics_cfg.get(name, default)
        return getattr(metrics_cfg, name, default)

    families = ["density"]
    if list(_value("pairs", []) or []):
        families.extend(["bondlen_scalar", "bondlen_cdf"])
    if list(_value("angles", []) or []):
        families.extend(["angle_scalar", "angle_cdf"])
    if list(_value("coordinations", []) or []):
        families.extend(["coord_scalar", "coord_cdf"])
    rings = _value("rings", None)
    rings_enabled = (
        bool((rings or {}).get("enabled", False))
        if isinstance(rings, Mapping)
        else bool(getattr(rings, "enabled", False))
    )
    if rings_enabled:
        families.append("ring")
    if list(_value("gr", []) or []):
        families.extend(["gr_peak", "gr_curve"])
    if list(_value("sq", []) or []):
        families.append("sq_curve")
    voids = _value("voids", None)
    voids_enabled = (
        bool((voids or {}).get("enabled", False))
        if isinstance(voids, Mapping)
        else bool(getattr(voids, "enabled", False))
    )
    if voids_enabled:
        families.append("void_cdf")
    return list(dict.fromkeys(str(name) for name in families))


def _convergence_scalar_family(name: str) -> str | None:
    key = str(name)
    if key.startswith("bondlen_") and key.endswith(("_mean", "_std")):
        return "bondlen_scalar"
    if key.startswith("angle_") and key.endswith(("_mean", "_std")):
        return "angle_scalar"
    if key.startswith("coord_") and key.endswith(("_mean", "_std")):
        return "coord_scalar"
    if key.startswith("gr_") and key.endswith(
        ("_peak_r", "_peak_height", "_peak_fwhm")
    ):
        return "gr_peak"
    return None


def _classify_emitted_scalar_metric(name: str, value: Any) -> dict[str, Any]:
    """Classify every emitted scalar without inventing unit-incompatible tolerances.

    Production writes a mixture of convergence descriptors and useful
    diagnostics into ``box["metrics"]``.  The distinction must be explicit:
    silently ignoring a newly emitted key makes it impossible for release
    validation to prove that the complete analysis path was exercised.  Only
    descriptors with an existing, dimensionally defined tolerance family are
    eligible for ensemble convergence.
    """

    key = str(name)
    try:
        finite = bool(math.isfinite(float(value)))
    except Exception:
        finite = False

    family = _convergence_scalar_family(key)
    if family is not None:
        return {
            "name": key,
            "role": "convergence",
            "family": str(family),
            "group": str(_CONVERGENCE_FAMILY_TO_GROUP[family]),
            "convergence_item": f"scalar:{key}",
            "value_status": "finite" if finite else "nonfinite",
            "reason": "dimensionally defined ensemble tolerance family",
        }
    if key.startswith("ring_frac_"):
        return {
            "name": key,
            "role": "convergence",
            "family": "ring",
            "group": "medium",
            "convergence_item": f"ring:{key}",
            "value_status": "finite" if finite else "nonfinite",
            "reason": "normalized ring-fraction convergence descriptor",
        }
    if key == "ring_mean_size":
        return {
            "name": key,
            "role": "convergence",
            "family": "ring",
            "group": "medium",
            "convergence_item": "ring:ring_mean_size",
            "value_status": "finite" if finite else "nonfinite",
            "reason": "dimensionless ring-size convergence descriptor",
        }

    if key.startswith("bond_incidence_") and key.endswith("_count"):
        reason = (
            "extensive sample-incidence diagnostic; normalized bond-length "
            "scalars and CDFs carry the ensemble criterion"
        )
    elif key == "ring_count":
        reason = (
            "extensive ring-incidence diagnostic; normalized ring fractions "
            "and conditional mean size carry the ensemble criterion"
        )
    elif key.startswith("sq_"):
        reason = (
            "derived S(q) peak diagnostic; the complete S(q) curve carries "
            "the dimensionless long-range ensemble criterion"
        )
    elif key.startswith("void_clearance_"):
        reason = (
            "derived void-summary diagnostic with mixed physical units; the "
            "void-clearance CDF carries the dimensionless ensemble criterion"
        )
    elif key.startswith("amorphous_"):
        reason = (
            "amorphous-state classification/filter diagnostic; no independent "
            "ensemble tolerance family is defined"
        )
    else:
        reason = (
            "diagnostic-only scalar: no dimensionally valid ensemble tolerance "
            "family is defined"
        )
    return {
        "name": key,
        "role": "diagnostic_only",
        "family": None,
        "group": None,
        "convergence_item": None,
        "value_status": "finite" if finite else "nonfinite",
        "reason": reason,
    }


def build_production_convergence_spec(
    entry: Mapping[str, Any],
    metrics_cfg: Any = None,
) -> dict[str, Any]:
    struct_vals = dict(entry.get("metrics", {}) or {})
    dist_all = dict(entry.get("distributions", {}) or {})
    skipped: list[dict[str, Any]] = []

    def _skip(
        kind: str,
        name: str,
        reason: str,
        *,
        blocking: bool,
        status: str,
    ) -> None:
        skipped.append(
            {
                "kind": str(kind),
                "name": str(name),
                "reason": str(reason),
                "blocking": bool(blocking),
                "status": str(status),
            }
        )

    def _finite_names(family: str, value_key: str, grid_key: str, report_kind: str) -> list[str]:
        names: list[str] = []
        for nm in sorted(list((dist_all.get(family, {}) or {}).keys())):
            payload = ((dist_all.get(family, {}) or {}).get(nm, {}) or {})
            ok, reason, _nval, _ngrid = _finite_array_from_curve_payload(
                payload, value_key, grid_key=grid_key
            )
            if ok:
                names.append(str(nm))
            elif _is_explicit_zero_incidence_curve(payload, value_key, grid_key=grid_key):
                # Retain the requested name so ensemble sanitisation can prove
                # that *every* box has zero incidence.  Dropping it here would
                # make a mixed zero/non-zero population depend on box order.
                names.append(str(nm))
                _skip(
                    report_kind,
                    str(nm),
                    "conditional distribution is undefined because measured incidence is zero",
                    blocking=False,
                    status="valid_zero_incidence",
                )
            else:
                _skip(
                    report_kind,
                    str(nm),
                    reason,
                    blocking=True,
                    status="invalid_or_missing",
                )
        return names

    bond_names = _finite_names("bondlen", "cdf", "x", "bondlen_cdf")
    angle_names = _finite_names("angle", "cdf", "x", "angle_cdf")
    coord_names = _finite_names("coord", "cdf", "x", "coord_cdf")
    gr_labels = _finite_names("gr", "g", "r", "gr_curve")
    sq_labels = _finite_names("sq", "s", "q", "sq_curve")
    void_names = _finite_names("void", "cdf", "x", "void_cdf")

    scalar_metric_classification = {
        str(key): _classify_emitted_scalar_metric(str(key), struct_vals.get(key))
        for key in sorted(str(name) for name in struct_vals)
    }
    scalar_names: list[str] = []
    for key in sorted(str(name) for name in struct_vals):
        if _convergence_scalar_family(key) is None:
            continue
        scalar_names.append(str(key))
        try:
            value = float(struct_vals.get(key, float("nan")))
        except Exception:
            value = float("nan")
        if not math.isfinite(value):
            _skip(
                "scalar_metric",
                str(key),
                "configured convergence scalar contains a non-finite value",
                blocking=True,
                status="invalid_or_missing",
            )

    ring_keys: list[str] = []
    for k in sorted([kk for kk in struct_vals.keys() if str(kk).startswith("ring_frac_")]):
        try:
            val = float(struct_vals.get(k, float("nan")))
        except Exception:
            val = float("nan")
        if math.isfinite(val):
            ring_keys.append(str(k))
        else:
            _skip(
                "ring_metric",
                str(k),
                "ring metric contains non-finite values",
                blocking=True,
                status="invalid_or_missing",
            )

    ring_has_mean_size = False
    if "ring_mean_size" in struct_vals:
        try:
            rms = float(struct_vals.get("ring_mean_size", float("nan")))
        except Exception:
            rms = float("nan")
        if math.isfinite(rms):
            ring_has_mean_size = True
        else:
            try:
                ring_count = float(struct_vals.get("ring_count", float("nan")))
            except Exception:
                ring_count = float("nan")
            zero_rings = math.isfinite(ring_count) and ring_count == 0.0
            _skip(
                "ring_mean_size",
                "ring_mean_size",
                (
                    "ring mean size is undefined because measured ring incidence is zero"
                    if zero_rings
                    else "ring mean size contains non-finite values"
                ),
                blocking=bool(not zero_rings),
                status=("valid_zero_incidence" if zero_rings else "invalid_or_missing"),
            )

    return {
        "bondlen_names": bond_names,
        "angle_names": angle_names,
        "coord_names": coord_names,
        "ring_keys": ring_keys,
        "ring_has_mean_size": bool(ring_has_mean_size),
        "gr_labels": gr_labels,
        "sq_labels": sq_labels,
        "void_names": void_names,
        "scalar_names": scalar_names,
        "scalar_metric_classification": scalar_metric_classification,
        "configured_metric_families": _configured_convergence_families(metrics_cfg),
        "skipped_metrics_initial": skipped,
    }

def validate_production_entry_against_spec(entry: Mapping[str, Any], spec: Mapping[str, Any], *, box_label: Any) -> None:
    dist_all = dict(entry.get("distributions", {}) or {})
    for nm in spec.get("bondlen_names", []):
        if nm not in (dist_all.get("bondlen", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing bond-length distribution '{nm}'")
    for nm in spec.get("angle_names", []):
        if nm not in (dist_all.get("angle", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing angle distribution '{nm}'")
    for nm in spec.get("coord_names", []):
        if nm not in (dist_all.get("coord", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing coordination distribution '{nm}'")
    for lab in spec.get("gr_labels", []):
        if lab not in (dist_all.get("gr", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing g(r) curve '{lab}'")
    for nm in spec.get("void_names", []):
        if nm not in (dist_all.get("void", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing void distribution '{nm}'")
    for lab in spec.get("sq_labels", []):
        if lab not in (dist_all.get("sq", {}) or {}):
            raise RuntimeError(f"Production box {box_label} missing S(q) curve '{lab}'")
    metrics = dict(entry.get("metrics", {}) or {})
    for name in spec.get("scalar_names", []):
        if name not in metrics:
            raise RuntimeError(
                f"Production box {box_label} missing convergence scalar '{name}'"
            )


def metrics_checked_from_conv_spec(spec: Optional[Mapping[str, Any]]) -> Optional[list[str]]:
    if spec is None:
        return None
    return [
        "density",
        *list(spec.get("scalar_names", [])),
        *list(spec.get("ring_keys", [])),
        *(["ring_mean_size"] if bool(spec.get("ring_has_mean_size", False)) else []),
        *list(spec.get("bondlen_names", [])),
        *list(spec.get("angle_names", [])),
        *list(spec.get("coord_names", [])),
        *list(spec.get("gr_labels", [])),
        *list(spec.get("sq_labels", [])),
        *list(spec.get("void_names", [])),
    ]


def _relpath_or_str(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def _copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    try:
        if Path(src).exists():
            shutil.copy2(src, dst)
            return Path(dst)
    except Exception:
        return None
    return None


def _materialise_reject_payload(
    *,
    outdir: Path,
    box_id: int,
    relax_stage_dir: Path,
    relax_data_path: Path,
    relax_dump_path: Path,
    rejects_dir: Optional[Path],
    reasons: Sequence[str],
    extra_files: Optional[Sequence[Path]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"reason": (str(reasons[0]) if len(reasons) == 1 else "+".join([str(x) for x in reasons])), "reasons": [str(x) for x in reasons]}
    if rejects_dir is None:
        return payload
    try:
        rejects_dir = Path(rejects_dir)
        rejects_dir.mkdir(parents=True, exist_ok=True)
        rdir = rejects_dir / f"box_{int(box_id):03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        dst_data = _copy_if_exists(Path(relax_data_path), rdir / Path(relax_data_path).name)
        dst_dump = _copy_if_exists(Path(relax_dump_path), rdir / Path(relax_dump_path).name)
        for src in list(extra_files or []):
            _copy_if_exists(Path(src), rdir / Path(src).name)
        payload.update({
            "reject_dir": _relpath_or_str(rdir, outdir),
            "relax_data": (_relpath_or_str(dst_data, outdir) if dst_data is not None else None),
            "relax_dump": (_relpath_or_str(dst_dump, outdir) if dst_dump is not None else None),
        })
    except Exception as e:
        payload["error"] = str(e)
    return payload


def _tol_for_metric(name: str, conv) -> tuple[float, float]:
    """Tol for metric."""
    if name == "density":
        return float(conv.density_rel_tol), float(conv.density_abs_tol)
    if name.startswith("coord_"):
        return float(conv.coord_rel_tol), float(conv.coord_abs_tol)
    if name.startswith("bondlen_"):
        return float(conv.bondlen_rel_tol), float(conv.bondlen_abs_tol)
    if name.startswith("angle_"):
        return float(conv.angle_rel_tol), float(conv.angle_abs_tol)
    if name.startswith("ring_frac_"):
        return float(conv.ring_rel_tol), float(conv.ring_abs_tol)
    if name == "ring_mean_size":
        return float(conv.ring_size_rel_tol), float(conv.ring_size_abs_tol)
    if name.startswith("gr_") and name.endswith("_peak_r"):
        return conv.gr_peak_r_rel_tol, conv.gr_peak_r_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_height"):
        return conv.gr_peak_height_rel_tol, conv.gr_peak_height_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_fwhm"):
        return conv.gr_peak_fwhm_rel_tol, conv.gr_peak_fwhm_abs_tol
    raise ValueError(
        f"Metric {name!r} has no defined ensemble-convergence tolerance family "
        "and is not eligible for convergence selection"
    )


def _alpha_from_z(z: float) -> float:
    p = float(NormalDist().cdf(abs(float(z))))
    a = 2.0 * max(0.0, 1.0 - float(p))
    return float(min(1.0, max(0.0, a)))


def _critical_value(n: int, alpha: float) -> tuple[float, str]:
    a = float(min(1.0, max(0.0, alpha)))
    if int(n) < 2:
        return float("inf"), "n<2"
    try:
        from scipy.stats import t as _t  # type: ignore

        crit = float(_t.ppf(1.0 - a / 2.0, df=int(n) - 1))
        if math.isfinite(crit):
            return crit, "t"
    except Exception:
        pass
    crit = float(NormalDist().inv_cdf(1.0 - a / 2.0))
    return crit, "z"



_CDF_ROUNDOFF_ATOL = 1.0e-12
_CDF_COUNT_ALIASES = ("sample_count", "n_samples", "count", "n", "n_effective")


def _validated_cdf_count_aliases(
    payload: Mapping[str, Any],
    *,
    allow_zero: bool,
) -> int | None:
    """Return exact, internally consistent CDF count metadata.

    Count metadata controls pooled-CDF weights.  It therefore cannot be
    rounded, selected from the first usable alias, or allowed to disagree
    across aliases without changing the represented empirical measure.
    """

    parsed: list[tuple[str, int]] = []
    qualifier = "nonnegative" if allow_zero else "positive"
    for key in _CDF_COUNT_ALIASES:
        if key not in payload:
            continue
        raw = payload.get(key)
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(
                f"CDF count alias '{key}' must be a finite exact {qualifier} integer; "
                "booleans are not counts"
            )
        try:
            text = str(raw).strip()
            if not text:
                raise InvalidOperation
            value = Decimal(text)
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                f"CDF count alias '{key}' must be a finite exact {qualifier} integer"
            ) from exc
        if (
            not value.is_finite()
            or value != value.to_integral_value()
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(
                f"CDF count alias '{key}' must be a finite exact {qualifier} integer"
            )
        parsed.append((str(key), int(value)))

    if not parsed:
        return None
    reference = int(parsed[0][1])
    if any(int(value) != reference for _key, value in parsed[1:]):
        rendered = ", ".join(f"{key}={value}" for key, value in parsed)
        raise ValueError(f"CDF count aliases disagree: {rendered}")
    return reference


def _finite_array_from_curve_payload(payload: Mapping[str, Any], value_key: str, *, grid_key: str | None = None) -> tuple[bool, str, int, int]:
    """Return whether a stored curve payload is usable for ensemble convergence."""
    if not isinstance(payload, Mapping):
        return False, "payload is not a mapping", 0, 0
    try:
        vals = np.asarray(payload.get(value_key, []), dtype=float)
    except Exception:
        return False, f"{value_key} is not numeric", 0, 0
    if vals.ndim != 1 or vals.size == 0:
        return False, f"{value_key} is empty", int(vals.size), 0
    if not np.all(np.isfinite(vals)):
        return False, f"{value_key} contains non-finite values", int(vals.size), 0
    grid_size = int(vals.size)
    if grid_key is not None:
        try:
            grid = np.asarray(payload.get(grid_key, []), dtype=float)
        except Exception:
            return False, f"{grid_key} is not numeric", int(vals.size), 0
        grid_size = int(grid.size)
        if grid.ndim != 1 or grid.size != vals.size:
            return False, f"{grid_key}/{value_key} lengths differ", int(vals.size), int(grid.size)
        if not np.all(np.isfinite(grid)):
            return False, f"{grid_key} contains non-finite values", int(vals.size), int(grid.size)
        if grid.size > 1 and not np.all(np.diff(grid) > 0.0):
            return False, f"{grid_key} is not strictly increasing", int(vals.size), int(grid.size)
    if str(value_key) == "cdf":
        tol = float(_CDF_ROUNDOFF_ATOL)
        if float(np.min(vals)) < -tol or float(np.max(vals)) > 1.0 + tol:
            return False, "cdf values lie outside [0,1] beyond roundoff tolerance", int(vals.size), int(grid_size)
        if vals.size > 1 and np.any(np.diff(vals) < -tol):
            return False, "cdf is not nondecreasing", int(vals.size), int(grid_size)
    return True, "", int(vals.size), int(grid_size)


def _is_explicit_zero_incidence_curve(
    payload: Mapping[str, Any],
    value_key: str,
    *,
    grid_key: str | None = None,
) -> bool:
    """Recognise a measured zero-incidence descriptor, not missing data.

    A conditional CDF is mathematically undefined when its conditioning event
    never occurred.  Producers encode that state explicitly with
    ``sample_count=0`` and ``available=false``.  Empty arrays without both
    markers remain malformed/missing and must not be silently accepted.
    """
    if not isinstance(payload, Mapping):
        return False
    if payload.get("available", None) is not False:
        return False
    if "sample_count" not in payload:
        return False
    try:
        count = _validated_cdf_count_aliases(payload, allow_zero=True)
    except ValueError:
        return False
    if count != 0:
        return False
    vals = np.asarray(payload.get(value_key, []), dtype=float)
    if vals.ndim != 1 or vals.size != 0:
        return False
    if grid_key is not None:
        grid = np.asarray(payload.get(grid_key, []), dtype=float)
        if grid.ndim != 1 or grid.size != 0:
            return False
    return True




def _prepare_cdf_curve_payload(
    payload: Mapping[str, Any],
    *,
    xkey: str = "x",
    cdf_key: str = "cdf",
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Return a validated CDF curve from a stored per-box payload.

    Per-box adaptive graph descriptors can legitimately produce box-specific
    CDF grids.  The ensemble object must therefore be induced by evaluating
    every per-box CDF on an explicit common grid rather than assuming the first
    box's grid is universal.  Scientific inputs are validated, not sorted,
    clipped, or monotonised after the fact; only <=1e-12 floating-point
    roundoff accepted by the validator is normalised at the boundary.
    """
    ok, reason, _nval, _ngrid = _finite_array_from_curve_payload(
        payload, cdf_key, grid_key=xkey
    )
    if not ok:
        raise ValueError(reason)
    x = np.asarray(payload.get(xkey, []), dtype=float)
    cdf = np.asarray(payload.get(cdf_key, []), dtype=float)
    # Normalise only validator-accepted roundoff, never a scientifically
    # malformed descent or out-of-range probability.
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)
    sample_count = _validated_cdf_count_aliases(payload, allow_zero=False)
    return x.astype(float), cdf.astype(float), sample_count


def _common_cdf_grid(curves: Sequence[tuple[np.ndarray, np.ndarray, int | None]], *, x_ref: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Return a deterministic common CDF grid and whether source grids matched."""
    if x_ref is not None:
        xr = np.asarray(x_ref, dtype=float)
        if (
            xr.ndim != 1
            or xr.size < 1
            or not np.all(np.isfinite(xr))
            or (xr.size > 1 and not np.all(np.diff(xr) > 0.0))
        ):
            raise ValueError("reference CDF grid must be finite and strictly increasing")
        return xr.astype(float), False
    if not curves:
        raise ValueError("no CDF curves supplied")
    first = np.asarray(curves[0][0], dtype=float)
    same = True
    for x, _cdf, _n in curves[1:]:
        xx = np.asarray(x, dtype=float)
        if xx.shape != first.shape or (xx.size and float(np.max(np.abs(xx - first))) > 1.0e-10):
            same = False
            break
    if same:
        return first.astype(float), True
    grids = [np.asarray(x, dtype=float) for x, _cdf, _n in curves]
    common = np.unique(np.concatenate(grids))
    common = common[np.isfinite(common)]
    common.sort()
    if common.size < 1:
        raise ValueError("cannot construct finite common CDF grid")
    return common.astype(float), False


def _stack_continuous_cdfs_for_boxes(
    boxes: Sequence[Mapping[str, Any]],
    family: str,
    name: str,
    *,
    xkey: str = "x",
    x_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate per-box CDFs on an explicit common ensemble grid.

    Returns ``x_common, matrix, metadata`` where matrix has one row per box.
    Values below a box's source grid are evaluated as zero; values above it
    retain that box's stored terminal probability.  This preserves an
    explicitly truncated empirical CDF rather than silently completing it to
    one.
    """
    curves: list[tuple[np.ndarray, np.ndarray, int | None]] = []
    lengths: list[int] = []
    box_ids: list[Any] = []
    for idx, b in enumerate(boxes):
        dist_all = (b.get("distributions", {}) or {}) if isinstance(b, Mapping) else {}
        fam = (dist_all.get(family, {}) or {}) if isinstance(dist_all, Mapping) else {}
        payload = fam.get(name, None) if isinstance(fam, Mapping) else None
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Missing {family} CDF '{name}' in box {idx + 1}")
        x, cdf, n = _prepare_cdf_curve_payload(payload, xkey=xkey, cdf_key="cdf")
        curves.append((x, cdf, n))
        lengths.append(int(x.size))
        box_ids.append(b.get("box", b.get("box_id", idx + 1)) if isinstance(b, Mapping) else idx + 1)
    x_common, source_grids_same = _common_cdf_grid(curves, x_ref=x_ref)
    mat = np.zeros((len(curves), int(x_common.size)), dtype=float)
    for i, (x, cdf, _n) in enumerate(curves):
        idx = np.searchsorted(x, x_common, side="right") - 1
        row = np.zeros_like(x_common, dtype=float)
        mask = idx >= 0
        if np.any(mask):
            row[mask] = cdf[idx[mask]]
        row[x_common >= x[-1]] = cdf[-1]
        row = np.maximum.accumulate(np.clip(row, 0.0, 1.0))
        mat[i, :] = row
    counts = [n for _x, _cdf, n in curves]
    if all(n is not None and int(n) > 0 for n in counts):
        weights = np.asarray([float(n) for n in counts if n is not None], dtype=float)
        weighting = "sample_count"
    else:
        weights = np.ones((len(curves),), dtype=float)
        weighting = "equal_box"
    weights = weights / float(np.sum(weights))
    pooled = np.sum(mat * weights[:, None], axis=0)
    meta = {
        "schema": "vitriflow.ensemble_cdf.v1",
        "family": str(family),
        "name": str(name),
        "status": "ok",
        "numerical_status": "ok",
        "common_grid": True,
        "source_grids_same": bool(source_grids_same),
        "regridded_from_box_specific_grids": bool(not source_grids_same or x_ref is not None),
        "grid_source": "native_common_grid" if source_grids_same and x_ref is None else "ensemble_union_support_grid",
        "grid_alignment_method": "right_continuous_cdf_evaluation",
        "x_min": float(x_common[0]),
        "x_max": float(x_common[-1]),
        "n_points": int(x_common.size),
        "n_boxes": int(len(curves)),
        "box_ids": list(box_ids),
        "source_grid_lengths": list(lengths),
        "grid_mismatch_count": 0 if source_grids_same else int(len(curves)),
        "normalization": "per_box_cdf_then_ensemble_mean",
        "pooled_normalization": "sample_count_weighted_cdf" if weighting == "sample_count" else "equal_box_weighted_cdf",
        "sample_weighting": weighting,
        "sample_counts": [None if n is None else int(n) for n in counts],
        "n_samples_total": (int(sum(int(n) for n in counts if n is not None)) if all(n is not None for n in counts) else None),
        "pooled_cdf": [float(v) for v in pooled.tolist()],
    }
    return x_common.astype(float), mat.astype(float), meta


def _prepare_sampled_curve_payload(
    payload: Mapping[str, Any],
    *,
    xkey: str,
    ykey: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one sampled physical curve without silently repairing its grid."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload is not a mapping")
    x = np.asarray(payload.get(xkey, []), dtype=float)
    y = np.asarray(payload.get(ykey, []), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError(
            f"{xkey}/{ykey} must be one-dimensional arrays of equal length >= 2"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"{xkey}/{ykey} contain non-finite values")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError(f"{xkey} must be strictly increasing")
    return x.astype(float), y.astype(float)


_SQ_REPRESENTATION_SCHEMA = "vitriflow.sq_representation.v1"
_SQ_REPRESENTATION_BASE_FIELDS = frozenset(
    {
        "schema",
        "observable",
        "estimator",
        "normalization",
        "normalization_family",
        "normalization_formula",
        "self_term",
        "pair",
        "rdf_normalization",
        "scattering_weights",
        "scattering_weighted",
        "dimensionless",
        "q_unit",
        "r_unit",
        "termination_window",
        "termination_window_definition",
        "radial_transform_kernel",
        "radial_quadrature",
        "r_support_requested_A",
        "r_support_effective_A",
        "r_support_clipped_to_unique_image_radius",
        "r_support_policy",
        "n_r_bins",
        "q_min_A^-1",
        "q_max_A^-1",
        "n_q_points",
        "q_zero_semantics",
        "frame_aggregation",
        "density_handling",
        "density_prefactor_unit",
        "density_prefactors_A^-3",
        "n_frames_requested",
        "n_frames_used",
    }
)
_SQ_REPRESENTATION_PARTIAL_FIELDS = frozenset(
    {"partial_kind", "resolved_type_sets"}
)
_SQ_REPRESENTATION_INVARIANTS = (
    "schema",
    "observable",
    "estimator",
    "normalization",
    "normalization_family",
    "normalization_formula",
    "self_term",
    "pair",
    "resolved_type_sets",
    "scattering_weights",
    "scattering_weighted",
    "dimensionless",
    "q_unit",
    "r_unit",
    "termination_window",
    "rdf_normalization",
    "termination_window_definition",
    "radial_transform_kernel",
    "radial_quadrature",
    "r_support_requested_A",
    "r_support_policy",
    "n_r_bins",
    "q_min_A^-1",
    "q_max_A^-1",
    "n_q_points",
    "q_zero_semantics",
    "frame_aggregation",
    "density_handling",
    "density_prefactor_unit",
)


def _sq_metadata_exact_integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"S(q) representation {field} must be an integer >= {minimum}")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"S(q) representation {field} must be an integer >= {minimum}")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"S(q) representation {field} must be an integer >= {minimum}"
        ) from exc
    if not math.isfinite(numeric) or numeric != float(integer) or integer < minimum:
        raise ValueError(f"S(q) representation {field} must be an integer >= {minimum}")
    return int(integer)


def _sq_metadata_finite_float(
    value: Any,
    *,
    field: str,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        qualifier = "finite and > 0" if positive else "finite"
        raise ValueError(f"S(q) representation {field} must be {qualifier}")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        qualifier = "finite and > 0" if positive else "finite"
        raise ValueError(f"S(q) representation {field} must be {qualifier}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        qualifier = "finite and > 0" if positive else "finite"
        raise ValueError(f"S(q) representation {field} must be {qualifier}") from exc
    if not math.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "finite and > 0" if positive else "finite"
        raise ValueError(f"S(q) representation {field} must be {qualifier}")
    return float(numeric)


def _sq_metadata_selector(value: Any, *, field: str) -> int | str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"S(q) representation {field} must be a valid atom selector")
    if isinstance(value, (int, np.integer)):
        if int(value) < 1:
            raise ValueError(f"S(q) representation {field} type id must be >= 1")
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"S(q) representation {field} must be a valid atom selector")


def _sq_metadata_json_token(value: Any, *, field: str) -> str:
    """Return a deterministic token for already validated JSON metadata."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"S(q) representation field {field} is not strict JSON data"
        ) from exc


def _validate_sq_representation_payload(
    representation: Any,
    *,
    q: np.ndarray,
    box_label: Any,
) -> dict[str, Any]:
    """Validate the exact v1 RDF-transform representation contract."""
    prefix = f"S(q) representation in box {box_label}"
    if not isinstance(representation, Mapping):
        raise ValueError(f"{prefix} must be a mapping")
    rep = dict(representation)
    if any(not isinstance(field, str) for field in rep):
        raise ValueError(f"{prefix} field names must be strings")
    if rep.get("schema") != _SQ_REPRESENTATION_SCHEMA:
        raise ValueError(
            f"{prefix} schema must be {_SQ_REPRESENTATION_SCHEMA!r}"
        )

    normalization_family = rep.get("normalization_family")
    if normalization_family == "number_number":
        expected_fields = _SQ_REPRESENTATION_BASE_FIELDS
    elif normalization_family == "ashcroft_langreth":
        expected_fields = (
            _SQ_REPRESENTATION_BASE_FIELDS | _SQ_REPRESENTATION_PARTIAL_FIELDS
        )
    else:
        raise ValueError(f"{prefix} has an unsupported normalization_family")
    actual_fields = set(rep)
    missing = sorted(expected_fields - actual_fields)
    unknown = sorted(actual_fields - expected_fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields {missing}")
        if unknown:
            details.append(f"unknown fields {unknown}")
        raise ValueError(f"{prefix} does not match its schema: {'; '.join(details)}")

    exact_constants = {
        "observable": "static_structure_factor",
        "estimator": "isotropic_rdf_fourier_transform",
        "rdf_normalization": "finite_population_unordered_pair_shell_volume",
        "scattering_weights": "none",
        "q_unit": "angstrom^-1",
        "r_unit": "angstrom",
        "radial_transform_kernel": "4*pi*r^2*sinc(q*r)",
        "radial_quadrature": "uniform_bin_midpoint",
        "r_support_policy": (
            "minimum_half_shortest_lattice_translation_across_frames"
        ),
        "q_zero_semantics": (
            "finite_r_windowed_rdf_transform_extrapolation_not_thermodynamic_compressibility"
        ),
        "frame_aggregation": (
            "equal_frame_mean_after_per_frame_density_transform"
        ),
        "density_handling": "per_frame_number_density_prefactor",
        "density_prefactor_unit": "angstrom^-3",
    }
    for field, expected in exact_constants.items():
        if rep[field] != expected:
            raise ValueError(f"{prefix} field {field} has an unsupported value")
    if not isinstance(rep["scattering_weighted"], (bool, np.bool_)) or bool(
        rep["scattering_weighted"]
    ):
        raise ValueError(f"{prefix} scattering_weighted must be false")
    if not isinstance(rep["dimensionless"], (bool, np.bool_)) or not bool(
        rep["dimensionless"]
    ):
        raise ValueError(f"{prefix} dimensionless must be true")

    window = rep["termination_window"]
    window_definitions = {
        "lorch": "sinc(pi*r/r_support_effective)",
        "hann": "0.5*(1+cos(pi*r/r_support_effective))",
        "none": "1",
    }
    if not isinstance(window, str) or window not in window_definitions:
        raise ValueError(f"{prefix} termination_window is unsupported")
    if rep["termination_window_definition"] != window_definitions[window]:
        raise ValueError(
            f"{prefix} termination_window_definition disagrees with termination_window"
        )

    if normalization_family == "number_number":
        expected_normalization = {
            "normalization": "unweighted_number_number_total",
            "normalization_formula": (
                "S_NN(q) = 1 + 4*pi*rho*integral[r^2*(g_NN(r)-1)*sinc(q*r) dr]"
            ),
        }
        if rep["pair"] is not None:
            raise ValueError(f"{prefix} total normalization requires pair=null")
        self_term_expected = 1.0
    else:
        expected_normalization = {
            "normalization": "ashcroft_langreth_partial",
            "normalization_formula": (
                "S_ab(q) = delta_ab + 4*pi*sqrt(rho_a*rho_b)*"
                "integral[r^2*(g_ab(r)-1)*sinc(q*r) dr]"
            ),
        }
        pair = rep["pair"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{prefix} partial pair must be a two-item JSON array")
        for idx, selector in enumerate(pair):
            _sq_metadata_selector(selector, field=f"pair[{idx}]")
        resolved = rep["resolved_type_sets"]
        if not isinstance(resolved, list) or len(resolved) != 2:
            raise ValueError(
                f"{prefix} resolved_type_sets must be a two-item JSON array"
            )
        resolved_sets: list[list[int]] = []
        for outer_idx, values in enumerate(resolved):
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"{prefix} resolved_type_sets[{outer_idx}] must be a non-empty JSON array"
                )
            parsed = [
                _sq_metadata_exact_integer(
                    value,
                    field=f"resolved_type_sets[{outer_idx}][{inner_idx}]",
                    minimum=1,
                )
                for inner_idx, value in enumerate(values)
            ]
            if parsed != sorted(set(parsed)):
                raise ValueError(
                    f"{prefix} resolved_type_sets[{outer_idx}] must be sorted and unique"
                )
            resolved_sets.append(parsed)
        partial_kind = rep["partial_kind"]
        if partial_kind == "self":
            if resolved_sets[0] != resolved_sets[1]:
                raise ValueError(f"{prefix} self partial must resolve to equal type sets")
            self_term_expected = 1.0
        elif partial_kind == "cross":
            if set(resolved_sets[0]) & set(resolved_sets[1]):
                raise ValueError(f"{prefix} cross partial type sets must be disjoint")
            self_term_expected = 0.0
        else:
            raise ValueError(f"{prefix} partial_kind must be 'self' or 'cross'")
    for field, expected in expected_normalization.items():
        if rep[field] != expected:
            raise ValueError(f"{prefix} field {field} has an unsupported value")
    self_term = _sq_metadata_finite_float(rep["self_term"], field="self_term")
    if self_term != self_term_expected:
        raise ValueError(f"{prefix} self_term disagrees with its normalization")

    r_requested = _sq_metadata_finite_float(
        rep["r_support_requested_A"],
        field="r_support_requested_A",
        positive=True,
    )
    r_effective = _sq_metadata_finite_float(
        rep["r_support_effective_A"],
        field="r_support_effective_A",
        positive=True,
    )
    r_scale = max(1.0, abs(r_requested), abs(r_effective))
    r_tol = 128.0 * np.finfo(float).eps * r_scale
    if r_effective > r_requested + r_tol:
        raise ValueError(f"{prefix} effective radial support exceeds requested support")
    clipped_expected = bool(
        r_effective
        < r_requested - 64.0 * np.finfo(float).eps * r_scale
    )
    clipped = rep["r_support_clipped_to_unique_image_radius"]
    if not isinstance(clipped, (bool, np.bool_)) or bool(clipped) != clipped_expected:
        raise ValueError(f"{prefix} radial-support clipping flag is inconsistent")

    n_r_bins = _sq_metadata_exact_integer(rep["n_r_bins"], field="n_r_bins", minimum=50)
    q_min = _sq_metadata_finite_float(rep["q_min_A^-1"], field="q_min_A^-1")
    q_max = _sq_metadata_finite_float(
        rep["q_max_A^-1"], field="q_max_A^-1", positive=True
    )
    n_q_points = _sq_metadata_exact_integer(
        rep["n_q_points"], field="n_q_points", minimum=10
    )
    if q_min != 0.0 or q_max <= q_min:
        raise ValueError(f"{prefix} q support must start at zero and have positive width")
    if int(q.size) != n_q_points:
        raise ValueError(f"{prefix} n_q_points disagrees with the stored q array")
    expected_q = np.linspace(q_min, q_max, n_q_points, dtype=float)
    q_scale = max(1.0, abs(q_min), abs(q_max))
    q_tol = 128.0 * np.finfo(float).eps * q_scale
    if not np.allclose(q, expected_q, rtol=0.0, atol=q_tol):
        raise ValueError(
            f"{prefix} stored q array is not the represented uniform q grid"
        )

    n_frames_requested = _sq_metadata_exact_integer(
        rep["n_frames_requested"], field="n_frames_requested", minimum=1
    )
    n_frames_used = _sq_metadata_exact_integer(
        rep["n_frames_used"], field="n_frames_used", minimum=1
    )
    if n_frames_requested != n_frames_used:
        raise ValueError(
            f"{prefix} n_frames_requested must equal n_frames_used"
        )
    density_prefactors = rep["density_prefactors_A^-3"]
    if not isinstance(density_prefactors, list):
        raise ValueError(
            f"{prefix} density_prefactors_A^-3 must be a JSON array"
        )
    if len(density_prefactors) != n_frames_used:
        raise ValueError(
            f"{prefix} density_prefactors_A^-3 must contain one value per used frame"
        )
    densities = [
        _sq_metadata_finite_float(
            value,
            field=f"density_prefactors_A^-3[{idx}]",
            positive=True,
        )
        for idx, value in enumerate(density_prefactors)
    ]

    # Prove that invariant values are JSON-comparable now, rather than failing
    # later while serializing the convergence report.
    for field in _SQ_REPRESENTATION_INVARIANTS:
        if field in rep:
            _sq_metadata_json_token(rep[field], field=field)
    if "partial_kind" in rep:
        _sq_metadata_json_token(rep["partial_kind"], field="partial_kind")
    return {
        "representation": rep,
        "r_support_requested_A": r_requested,
        "r_support_effective_A": r_effective,
        "n_r_bins": n_r_bins,
        "q_min_A^-1": q_min,
        "q_max_A^-1": q_max,
        "n_q_points": n_q_points,
        "n_frames_requested": n_frames_requested,
        "n_frames_used": n_frames_used,
        "density_prefactor_min_A^-3": float(min(densities)),
        "density_prefactor_max_A^-3": float(max(densities)),
    }


def _sq_representation_validation_for_curves(
    payloads: Sequence[Mapping[str, Any]],
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    box_ids: Sequence[Any],
) -> dict[str, Any]:
    """Validate representation compatibility before any q interpolation."""
    present = ["representation" in payload for payload in payloads]
    if not any(present):
        return {
            "status": "legacy_unavailable",
            "schema": None,
            "reason": "all S(q) payloads omit representation metadata",
            "r_support_requested_A": None,
            "r_support_effective_A": None,
            "r_support_effective_by_box_A": [],
            "r_support_match_tolerance_A": None,
            "n_frames_requested": None,
            "n_frames_used": None,
        }
    if not all(present):
        with_metadata = [box_ids[i] for i, value in enumerate(present) if value]
        without_metadata = [box_ids[i] for i, value in enumerate(present) if not value]
        raise ValueError(
            "mixed S(q) representation metadata availability is not comparable: "
            f"present in boxes {with_metadata}, absent in boxes {without_metadata}"
        )

    validated = [
        _validate_sq_representation_payload(
            payload["representation"],
            q=curves[idx][0],
            box_label=box_ids[idx],
        )
        for idx, payload in enumerate(payloads)
    ]
    reps = [item["representation"] for item in validated]
    reference = reps[0]
    invariant_fields = list(_SQ_REPRESENTATION_INVARIANTS)
    if "partial_kind" in reference:
        invariant_fields.append("partial_kind")
    for idx, rep in enumerate(reps[1:], start=1):
        for field in invariant_fields:
            if _sq_metadata_json_token(
                rep.get(field), field=field
            ) != _sq_metadata_json_token(reference.get(field), field=field):
                raise ValueError(
                    "S(q) representation invariant mismatch for "
                    f"{field!r} between boxes {box_ids[0]} and {box_ids[idx]}"
                )

    frame_counts = [int(item["n_frames_used"]) for item in validated]
    if any(value != frame_counts[0] for value in frame_counts[1:]):
        raise ValueError(
            "S(q) representation n_frames_requested/n_frames_used must be equal "
            f"across boxes; got {frame_counts}"
        )

    effective_supports = [
        float(item["r_support_effective_A"]) for item in validated
    ]
    support_ref = effective_supports[0]
    support_tolerance = (
        128.0
        * np.finfo(float).eps
        * max(1.0, *(abs(value) for value in effective_supports))
    )
    support_min_idx = int(np.argmin(np.asarray(effective_supports, dtype=float)))
    support_max_idx = int(np.argmax(np.asarray(effective_supports, dtype=float)))
    if (
        effective_supports[support_max_idx] - effective_supports[support_min_idx]
        > support_tolerance
    ):
        raise ValueError(
            "S(q) effective radial support mismatch between boxes "
            f"{box_ids[support_min_idx]} "
            f"({effective_supports[support_min_idx]:.17g} A) and "
            f"{box_ids[support_max_idx]} "
            f"({effective_supports[support_max_idx]:.17g} A); q-grid "
            "interpolation cannot repair a different real-space estimator support"
        )

    return {
        "status": "validated",
        "schema": _SQ_REPRESENTATION_SCHEMA,
        "r_support_requested_A": float(validated[0]["r_support_requested_A"]),
        "r_support_effective_A": float(support_ref),
        "r_support_effective_by_box_A": list(effective_supports),
        "r_support_match_tolerance_A": float(support_tolerance),
        "n_frames_requested": int(frame_counts[0]),
        "n_frames_used": int(frame_counts[0]),
        "density_prefactor_ranges_A^-3": [
            [
                float(item["density_prefactor_min_A^-3"]),
                float(item["density_prefactor_max_A^-3"]),
            ]
            for item in validated
        ],
        "validated_invariants": list(invariant_fields),
    }


def _stack_sampled_curves_for_boxes(
    boxes: Sequence[Mapping[str, Any]],
    family: str,
    name: str,
    *,
    xkey: str,
    ykey: str,
    x_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interpolate sampled physical curves onto their common support.

    Unlike a CDF, g(r) and S(q) do not have a mathematically defined constant
    extension outside their sampled support.  The ensemble grid is therefore
    restricted to the intersection of every source grid, and extrapolation is
    forbidden.  This prevents array indices from being compared as though they
    represented the same physical radius or wavevector.
    """
    if not boxes:
        raise ValueError(f"no boxes supplied for {family} curve {name!r}")
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    payloads: list[Mapping[str, Any]] = []
    lengths: list[int] = []
    box_ids: list[Any] = []
    for idx, b in enumerate(boxes):
        dist_all = (b.get("distributions", {}) or {}) if isinstance(b, Mapping) else {}
        fam = (dist_all.get(family, {}) or {}) if isinstance(dist_all, Mapping) else {}
        payload = fam.get(name, None) if isinstance(fam, Mapping) else None
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Missing {family} curve '{name}' in box {idx + 1}")
        x, y = _prepare_sampled_curve_payload(payload, xkey=xkey, ykey=ykey)
        curves.append((x, y))
        payloads.append(payload)
        lengths.append(int(x.size))
        box_ids.append(
            b.get("box", b.get("box_id", idx + 1))
            if isinstance(b, Mapping)
            else idx + 1
        )

    sq_representation_validation: dict[str, Any] | None = None
    if family == "sq":
        sq_representation_validation = _sq_representation_validation_for_curves(
            payloads,
            curves,
            box_ids,
        )

    overlap_min = max(float(x[0]) for x, _y in curves)
    overlap_max = min(float(x[-1]) for x, _y in curves)
    if not (
        math.isfinite(overlap_min)
        and math.isfinite(overlap_max)
        and overlap_max > overlap_min
    ):
        raise ValueError(
            f"{family} curve '{name}' grids have no common interval of support"
        )

    first = curves[0][0]
    source_grids_same = all(
        x.shape == first.shape
        and (x.size == 0 or float(np.max(np.abs(x - first))) <= 1.0e-10)
        for x, _y in curves[1:]
    )
    if x_ref is not None:
        x_common = np.asarray(x_ref, dtype=float)
        if (
            x_common.ndim != 1
            or x_common.size < 2
            or not np.all(np.isfinite(x_common))
            or not np.all(np.diff(x_common) > 0.0)
        ):
            raise ValueError("reference sampled-curve grid must be finite and strictly increasing")
        scale = max(1.0, abs(overlap_min), abs(overlap_max))
        eps = 32.0 * np.finfo(float).eps * scale
        if float(x_common[0]) < overlap_min - eps or float(x_common[-1]) > overlap_max + eps:
            raise ValueError("reference sampled-curve grid lies outside common support")
        grid_source = "provided_ensemble_common_grid"
    elif source_grids_same:
        x_common = first.copy()
        grid_source = "native_common_grid"
    else:
        within = [
            x[(x >= overlap_min) & (x <= overlap_max)]
            for x, _y in curves
        ]
        x_common = np.unique(
            np.concatenate(
                [np.asarray([overlap_min, overlap_max], dtype=float), *within]
            )
        )
        x_common.sort()
        grid_source = "ensemble_union_common_support_grid"
    if x_common.size < 2:
        raise ValueError(f"cannot construct a common grid for {family} curve '{name}'")

    mat = np.zeros((len(curves), int(x_common.size)), dtype=float)
    for idx, (x, y) in enumerate(curves):
        # x_common is proven to lie within every curve's support, so np.interp
        # performs interpolation only (never its constant extrapolation mode).
        mat[idx, :] = np.interp(x_common, x, y)
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"interpolated {family} curve '{name}' is non-finite")

    meta = {
        "schema": "vitriflow.ensemble_sampled_curve.v1",
        "family": str(family),
        "name": str(name),
        "status": "ok",
        "numerical_status": "ok",
        "common_grid": True,
        "source_grids_same": bool(source_grids_same),
        "regridded_from_box_specific_grids": bool(
            not source_grids_same or x_ref is not None
        ),
        "grid_source": str(grid_source),
        "grid_alignment_method": "linear_interpolation_on_common_support",
        "support_policy": "intersection_no_extrapolation",
        "x_min": float(x_common[0]),
        "x_max": float(x_common[-1]),
        "n_points": int(x_common.size),
        "n_boxes": int(len(curves)),
        "box_ids": list(box_ids),
        "source_grid_lengths": list(lengths),
        "grid_mismatch_count": 0 if source_grids_same else int(len(curves)),
    }
    if sq_representation_validation is not None:
        meta["representation_validation"] = sq_representation_validation
        meta["representation_validation_status"] = sq_representation_validation[
            "status"
        ]
        meta["representation_effective_r_support_A"] = (
            sq_representation_validation["r_support_effective_A"]
        )
    return x_common.astype(float), mat.astype(float), meta


def _stack_coord_cdfs_for_boxes(
    boxes: Sequence[Mapping[str, Any]],
    name: str,
    *,
    x_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stack coordination CDFs on their stored physical support.

    Hard graphs normally have the integer support ``0,1,...``.  Soft graph
    coordination is fractional and its explicit x-grid must not be discarded
    or reinterpreted as array indices.  The general right-continuous CDF
    alignment is valid for both cases, including a legitimate singleton
    ``x=[0], cdf=[1]`` zero-coordination distribution.
    """
    x_common, mat, meta = _stack_continuous_cdfs_for_boxes(
        boxes,
        "coord",
        name,
        xkey="x",
        x_ref=x_ref,
    )
    integer_grid = np.arange(int(x_common.size), dtype=float)
    is_integer_support = bool(
        x_common.shape == integer_grid.shape
        and np.allclose(x_common, integer_grid, rtol=0.0, atol=_CDF_ROUNDOFF_ATOL)
    )
    meta = {
        **dict(meta),
        "family": "coord",
        "grid_source": (
            "ensemble_common_integer_grid"
            if is_integer_support
            else "ensemble_common_explicit_coordination_grid"
        ),
        "coordination_support": "integer" if is_integer_support else "fractional_or_nonuniform",
        "grid_alignment_method": "right_continuous_cdf_evaluation",
    }
    return x_common.astype(float), mat.astype(float), meta


def build_ensemble_cdf_sidecar(boxes: Sequence[Mapping[str, Any]], spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build explicit ensemble-level CDF sidecar from per-box distributions."""
    raw = dict(spec or {})
    families: list[tuple[str, str, str]] = [
        ("bondlen", "bondlen_names", "x"),
        ("angle", "angle_names", "x"),
        ("void", "void_names", "x"),
    ]
    out: dict[str, Any] = {
        "schema": "vitriflow.ensemble_cdfs.v1",
        "n_boxes": int(len(boxes or [])),
        "normalization": "per_box_cdf_mean_plus_optional_sample_count_weighted_pooled_cdf",
        "families": {"bondlen": {}, "angle": {}, "coord": {}, "void": {}},
    }
    for family, key, xkey in families:
        for nm in sorted(set(str(x) for x in list(raw.get(key, []) or []))):
            try:
                x, mat, meta = _stack_continuous_cdfs_for_boxes(boxes, family, nm, xkey=xkey)
                mu, sd_vec, se_vec, n = _vector_stats(mat)
                out["families"][family][nm] = json_sanitize({
                    **meta,
                    "x": [float(v) for v in x.tolist()],
                    "mean_cdf": [float(v) for v in mu.tolist()],
                    "stderr": [float(v) for v in se_vec.tolist()],
                    "std": [float(v) for v in sd_vec.tolist()],
                    "n_effective_boxes": int(n),
                })
            except Exception as exc:
                out["families"][family][nm] = {"status": "unavailable", "numerical_status": "failed", "reason": str(exc)}
    for nm in sorted(set(str(x) for x in list(raw.get("coord_names", []) or []))):
        try:
            x, mat, meta = _stack_coord_cdfs_for_boxes(boxes, nm)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            out["families"]["coord"][nm] = json_sanitize({
                **meta,
                "x": [float(v) for v in x.tolist()],
                "mean_cdf": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "std": [float(v) for v in sd_vec.tolist()],
                "n_effective_boxes": int(n),
            })
        except Exception as exc:
            out["families"]["coord"][nm] = {"status": "unavailable", "numerical_status": "failed", "reason": str(exc)}
    return json_sanitize(out)

def _sanitize_production_convergence_spec(
    boxes: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the requested convergence matrix without failing open.

    Explicit all-box zero incidence is a valid measured state for a
    conditional distribution.  Missing/corrupt payloads and mixed
    present/zero-incidence populations are blocking because deleting those
    descriptors would mutate the convergence criterion after observing data.
    """
    raw = dict(spec or {})
    skipped: list[dict[str, Any]] = list(raw.get("skipped_metrics_initial", []) or [])

    def _skip(
        kind: str,
        name: str,
        reason: str,
        *,
        box: Any = None,
        blocking: bool = True,
        status: str = "invalid_or_missing",
    ) -> None:
        item: dict[str, Any] = {
            "kind": str(kind),
            "name": str(name),
            "reason": str(reason),
            "blocking": bool(blocking),
            "status": str(status),
        }
        if box is not None:
            item["box"] = box
        skipped.append(item)

    def _box_label(b: Mapping[str, Any], idx: int) -> Any:
        return b.get("box", idx + 1)

    def _usable_payload_for_all(
        family: str,
        name: str,
        value_key: str,
        *,
        grid_key: str | None = None,
        require_same_length: bool = True,
        require_same_grid: bool = True,
        report_kind: str | None = None,
    ) -> bool:
        ref_len: int | None = None
        ref_grid: np.ndarray | None = None
        states: list[str] = []
        for idx, b in enumerate(boxes):
            dist_all = b.get("distributions", {}) if isinstance(b, Mapping) else {}
            fam = (dist_all.get(family, {}) or {}) if isinstance(dist_all, Mapping) else {}
            payload = fam.get(name, None) if isinstance(fam, Mapping) else None
            ok, reason, nval, _ngrid = _finite_array_from_curve_payload(payload or {}, value_key, grid_key=grid_key)
            if not ok:
                if _is_explicit_zero_incidence_curve(
                    payload or {}, value_key, grid_key=grid_key
                ):
                    states.append("zero_incidence")
                    continue
                _skip(
                    report_kind or f"{family}_{value_key}",
                    name,
                    reason,
                    box=_box_label(b, idx),
                    blocking=True,
                    status="invalid_or_missing",
                )
                return False
            states.append("finite")
            if require_same_length:
                if ref_len is None:
                    ref_len = int(nval)
                elif int(nval) != int(ref_len):
                    _skip(report_kind or f"{family}_{value_key}", name, "curve length differs across boxes", box=_box_label(b, idx))
                    return False
            if require_same_grid and grid_key is not None:
                grid = np.asarray((payload or {}).get(grid_key, []), dtype=float)
                if ref_grid is None:
                    ref_grid = grid
                elif grid.shape != ref_grid.shape or (grid.size and float(np.max(np.abs(grid - ref_grid))) > 1e-10):
                    _skip(report_kind or f"{family}_{value_key}", name, "curve grid differs across boxes", box=_box_label(b, idx))
                    return False
        if states and all(state == "zero_incidence" for state in states):
            _skip(
                report_kind or f"{family}_{value_key}",
                name,
                "conditional distribution is undefined because measured incidence is zero in every box",
                blocking=False,
                status="valid_zero_incidence",
            )
            return False
        if any(state == "zero_incidence" for state in states):
            first_zero = states.index("zero_incidence")
            _skip(
                report_kind or f"{family}_{value_key}",
                name,
                "descriptor incidence is zero in only part of the ensemble; conditional CDF convergence is not defined",
                box=_box_label(boxes[first_zero], first_zero),
                blocking=True,
                status="mixed_zero_and_nonzero_incidence",
            )
            return False
        return True

    clean: dict[str, Any] = dict(raw)

    for idx, box in enumerate(boxes):
        try:
            density_value = float(box.get("density", float("nan")))
        except Exception:
            density_value = float("nan")
        if not math.isfinite(density_value):
            _skip(
                "density",
                "density",
                "density is missing or non-finite",
                box=_box_label(box, idx),
                blocking=True,
                status="invalid_or_missing",
            )

    requested_scalar_names = {str(name) for name in list(raw.get("scalar_names", []) or [])}
    for box in boxes:
        metrics_row = ((box.get("metrics", {}) or {}) if isinstance(box, Mapping) else {})
        requested_scalar_names.update(
            str(name)
            for name in metrics_row
            if _convergence_scalar_family(str(name)) is not None
        )
    scalar_names: list[str] = []
    for name in sorted(requested_scalar_names):
        usable = True
        for idx, box in enumerate(boxes):
            metrics_row = ((box.get("metrics", {}) or {}) if isinstance(box, Mapping) else {})
            try:
                value = float(metrics_row.get(name, float("nan")))
            except Exception:
                value = float("nan")
            if not math.isfinite(value):
                _skip(
                    "scalar_metric",
                    str(name),
                    "configured convergence scalar is missing or non-finite",
                    box=_box_label(box, idx),
                    blocking=True,
                    status="invalid_or_missing",
                )
                usable = False
                break
        if usable:
            scalar_names.append(str(name))
    clean["scalar_names"] = scalar_names

    # Preserve an auditable classification for *every* emitted scalar, not
    # just the tolerance-backed subset.  This catches newly added analysis
    # outputs in release validation while keeping unit-incompatible
    # diagnostics out of the numerical stopping criterion.
    classified_names = {
        str(name)
        for name in dict(raw.get("scalar_metric_classification", {}) or {})
    }
    for box in boxes:
        metrics_row = ((box.get("metrics", {}) or {}) if isinstance(box, Mapping) else {})
        classified_names.update(str(name) for name in metrics_row)
    scalar_classification: dict[str, dict[str, Any]] = {}
    for name in sorted(classified_names):
        present_boxes: list[Any] = []
        finite_boxes: list[Any] = []
        representative: Any = None
        for idx, box in enumerate(boxes):
            metrics_row = ((box.get("metrics", {}) or {}) if isinstance(box, Mapping) else {})
            if name not in metrics_row:
                continue
            label = _box_label(box, idx)
            present_boxes.append(label)
            representative = metrics_row.get(name)
            try:
                if math.isfinite(float(representative)):
                    finite_boxes.append(label)
            except Exception:
                pass
        classification = _classify_emitted_scalar_metric(name, representative)
        classification.update(
            {
                "n_boxes_total": int(len(boxes)),
                "n_boxes_present": int(len(present_boxes)),
                "n_boxes_finite": int(len(finite_boxes)),
                "present_boxes": present_boxes,
                "missing_boxes": [
                    _box_label(box, idx)
                    for idx, box in enumerate(boxes)
                    if name not in (((box.get("metrics", {}) or {}) if isinstance(box, Mapping) else {}))
                ],
                "value_status": (
                    "finite_all_boxes"
                    if len(finite_boxes) == len(boxes)
                    else (
                        "nonfinite_or_missing"
                        if present_boxes
                        else "missing_all_boxes"
                    )
                ),
            }
        )
        scalar_classification[name] = classification
    clean["scalar_metric_classification"] = scalar_classification

    def _union_distribution_names(family: str, raw_key: str) -> list[str]:
        names = {str(nm) for nm in list(raw.get(raw_key, []) or [])}
        for b in boxes:
            dist_all = (b.get("distributions", {}) or {}) if isinstance(b, Mapping) else {}
            fam = (dist_all.get(family, {}) or {}) if isinstance(dist_all, Mapping) else {}
            if isinstance(fam, Mapping):
                names.update(str(k) for k in fam.keys())
        return sorted(names)

    def _usable_sampled_curve_for_all(
        family: str,
        name: str,
        *,
        grid_key: str,
        value_key: str,
        report_kind: str,
    ) -> bool:
        if not _usable_payload_for_all(
            family,
            name,
            value_key,
            grid_key=grid_key,
            require_same_length=False,
            require_same_grid=False,
            report_kind=report_kind,
        ):
            return False
        try:
            _stack_sampled_curves_for_boxes(
                boxes,
                family,
                name,
                xkey=grid_key,
                ykey=value_key,
            )
        except Exception as exc:
            _skip(
                report_kind,
                name,
                f"sampled curves cannot be aligned on common physical support: {exc}",
                blocking=True,
                status="incompatible_curve_grids",
            )
            return False
        return True

    # Continuous CDFs are allowed to use box-specific grids.  The convergence
    # pass builds an explicit ensemble grid and evaluates every per-box CDF on
    # that grid, so sanitization should require finite payloads but not identical
    # x arrays.
    clean["bondlen_names"] = [
        str(nm) for nm in _union_distribution_names("bondlen", "bondlen_names")
        if _usable_payload_for_all(
            "bondlen", str(nm), "cdf", grid_key="x",
            require_same_length=False, require_same_grid=False, report_kind="bondlen_cdf",
        )
    ]
    clean["angle_names"] = [
        str(nm) for nm in _union_distribution_names("angle", "angle_names")
        if _usable_payload_for_all(
            "angle", str(nm), "cdf", grid_key="x",
            require_same_length=False, require_same_grid=False, report_kind="angle_cdf",
        )
    ]
    clean["coord_names"] = [
        str(nm) for nm in _union_distribution_names("coord", "coord_names")
        if _usable_payload_for_all(
            "coord", str(nm), "cdf", grid_key="x",
            require_same_length=False, require_same_grid=False, report_kind="coord_cdf",
        )
    ]
    clean["void_names"] = [
        str(nm) for nm in _union_distribution_names("void", "void_names")
        if _usable_payload_for_all(
            "void", str(nm), "cdf", grid_key="x",
            require_same_length=False, require_same_grid=False, report_kind="void_cdf",
        )
    ]
    clean["gr_labels"] = [
        str(lab) for lab in _union_distribution_names("gr", "gr_labels")
        if _usable_sampled_curve_for_all(
            "gr", str(lab), grid_key="r", value_key="g", report_kind="gr_curve",
        )
    ]
    clean["sq_labels"] = [
        str(lab) for lab in _union_distribution_names("sq", "sq_labels")
        if _usable_sampled_curve_for_all(
            "sq", str(lab), grid_key="q", value_key="s", report_kind="sq_curve",
        )
    ]

    requested_ring_keys = {str(key) for key in list(raw.get("ring_keys", []) or [])}
    for b in boxes:
        metrics_row = ((b.get("metrics", {}) or {}) if isinstance(b, Mapping) else {})
        requested_ring_keys.update(
            str(key) for key in metrics_row if str(key).startswith("ring_frac_")
        )
    ring_keys: list[str] = []
    for key in sorted(requested_ring_keys):
        ok = True
        for idx, b in enumerate(boxes):
            val = ((b.get("metrics", {}) or {}) if isinstance(b, Mapping) else {}).get(key, float("nan"))
            try:
                fval = float(val)
            except Exception:
                fval = float("nan")
            if not math.isfinite(fval):
                _skip("ring_metric", str(key), "ring metric contains non-finite values", box=_box_label(b, idx))
                ok = False
                break
        if ok:
            ring_keys.append(str(key))
    clean["ring_keys"] = ring_keys

    ring_mean_present = bool(raw.get("ring_has_mean_size", False)) or any(
        "ring_mean_size" in (((b.get("metrics", {}) or {}) if isinstance(b, Mapping) else {}))
        for b in boxes
    )
    if ring_mean_present:
        ok = True
        zero_ring_boxes: list[Any] = []
        for idx, b in enumerate(boxes):
            metrics_row = ((b.get("metrics", {}) or {}) if isinstance(b, Mapping) else {})
            val = metrics_row.get("ring_mean_size", float("nan"))
            try:
                fval = float(val)
            except Exception:
                fval = float("nan")
            if not math.isfinite(fval):
                try:
                    ring_count = float(metrics_row.get("ring_count", float("nan")))
                except Exception:
                    ring_count = float("nan")
                if math.isfinite(ring_count) and ring_count == 0.0:
                    zero_ring_boxes.append(_box_label(b, idx))
                    continue
                _skip(
                    "ring_mean_size",
                    "ring_mean_size",
                    "ring mean size contains non-finite values without an explicit zero ring count",
                    box=_box_label(b, idx),
                    blocking=True,
                    status="invalid_or_missing",
                )
                ok = False
                break
        if ok and zero_ring_boxes:
            all_zero = len(zero_ring_boxes) == len(boxes)
            _skip(
                "ring_mean_size",
                "ring_mean_size",
                (
                    "ring mean size is undefined because measured ring incidence is zero in every box"
                    if all_zero
                    else "ring incidence is zero in only part of the ensemble; conditional mean-size convergence is not defined"
                ),
                box=(None if all_zero else zero_ring_boxes[0]),
                blocking=bool(not all_zero),
                status=("valid_zero_incidence" if all_zero else "mixed_zero_and_nonzero_incidence"),
            )
            ok = False
        clean["ring_has_mean_size"] = bool(ok)
    else:
        clean["ring_has_mean_size"] = False

    configured_families = list(
        dict.fromkeys(str(name) for name in list(raw.get("configured_metric_families", []) or []))
    )
    clean["configured_metric_families"] = configured_families

    effective_family_counts = {
        "density": 1,
        "bondlen_scalar": sum(
            _convergence_scalar_family(name) == "bondlen_scalar"
            for name in clean["scalar_names"]
        ),
        "angle_scalar": sum(
            _convergence_scalar_family(name) == "angle_scalar"
            for name in clean["scalar_names"]
        ),
        "coord_scalar": sum(
            _convergence_scalar_family(name) == "coord_scalar"
            for name in clean["scalar_names"]
        ),
        "gr_peak": sum(
            _convergence_scalar_family(name) == "gr_peak"
            for name in clean["scalar_names"]
        ),
        "bondlen_cdf": len(clean["bondlen_names"]),
        "angle_cdf": len(clean["angle_names"]),
        "coord_cdf": len(clean["coord_names"]),
        "ring": len(clean["ring_keys"]) + int(bool(clean["ring_has_mean_size"])),
        "gr_curve": len(clean["gr_labels"]),
        "sq_curve": len(clean["sq_labels"]),
        "void_cdf": len(clean["void_names"]),
    }
    zero_incidence_kinds = {
        str(item.get("kind", ""))
        for item in skipped
        if str(item.get("status", "")) == "valid_zero_incidence"
    }
    for family in configured_families:
        if int(effective_family_counts.get(family, 0)) > 0:
            continue
        # An all-box zero-incidence conditional descriptor is measured
        # evidence, not missing output.  It is represented explicitly in the
        # coverage report and never replaced by an artificial CDF.
        if family in zero_incidence_kinds or (
            family == "ring" and "ring_mean_size" in zero_incidence_kinds
        ):
            continue
        _skip(
            "configured_metric_family",
            str(family),
            "configured convergence-capable metric family produced no usable payload",
            blocking=True,
            status="configured_family_missing",
        )
    return clean, skipped

def _tol_scalar(name: str, conv) -> tuple[float, float]:
    if name == "density":
        return float(conv.density_rel_tol), float(conv.density_abs_tol)
    return _tol_for_metric(name, conv)


def _tol_curve(kind: str, conv) -> tuple[float, float]:
    if kind == "bondlen_cdf":
        return float(conv.bondlen_cdf_rel_tol), float(conv.bondlen_cdf_abs_tol)
    if kind == "angle_cdf":
        return float(conv.angle_cdf_rel_tol), float(conv.angle_cdf_abs_tol)
    if kind == "coord_cdf":
        return float(conv.coord_cdf_rel_tol), float(conv.coord_cdf_abs_tol)
    if kind == "gr_curve":
        return float(conv.gr_curve_rel_tol), float(conv.gr_curve_abs_tol)
    if kind == "sq_curve":
        return float(conv.sq_curve_rel_tol), float(conv.sq_curve_abs_tol)
    if kind == "void_cdf":
        return float(conv.void_cdf_rel_tol), float(conv.void_cdf_abs_tol)
    return 0.0, 0.0


def _metric_group(kind: str) -> str:
    if kind in ("density", "gr_curve", "sq_curve", "void_cdf"):
        return "long"
    if kind == "ring":
        return "medium"
    if kind in ("bondlen_cdf", "angle_cdf", "coord_cdf"):
        return "short"
    if kind == "ring_mean_size":
        return "medium"
    if kind in {"bondlen_scalar", "angle_scalar", "coord_scalar", "gr_peak"}:
        return "short"
    return "other"


def _vector_stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError("mat must be 2D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values in convergence matrix")
    n = int(arr.shape[0])
    mu = np.mean(arr, axis=0)
    if n < 2:
        sd = np.full_like(mu, np.nan, dtype=float)
        se = np.full_like(mu, np.nan, dtype=float)
        return mu, sd, se, n
    sd = np.std(arr, axis=0, ddof=1)
    se = sd / math.sqrt(float(n))
    return mu, sd, se, n



class _ProductionConvergenceChecker:
    """Production convergence checker."""

    def __init__(self, boxes: list[dict[str, Any]], spec: dict[str, Any], conv_cfg) -> None:
        self.boxes = boxes
        self.spec = spec
        self.conv_cfg = conv_cfg

        self.z = float(conv_cfg.zscore)
        self.alpha_family = _alpha_from_z(self.z)

        mode = str(getattr(conv_cfg, "mode", "both")).strip().lower()
        self.mode = mode if mode in {"ci", "stability", "both"} else "both"

        self.n_boxes = int(len(boxes))
        self.ring_keys: list[str] = list(spec.get("ring_keys", []))
        self.scalar_names: list[str] = list(spec.get("scalar_names", []))
        self.bond_names: list[str] = list(spec.get("bondlen_names", []))
        self.angle_names: list[str] = list(spec.get("angle_names", []))
        self.coord_names: list[str] = list(spec.get("coord_names", []))
        self.gr_labels: list[str] = list(spec.get("gr_labels", []))
        self.sq_labels: list[str] = list(spec.get("sq_labels", []))
        self.void_names: list[str] = list(spec.get("void_names", []))

        self.m_tests = self._count_tests()
        self.familywise = str(getattr(conv_cfg, "familywise", "none"))
        if self.familywise == "bonferroni" and self.m_tests > 1:
            self.alpha_test = float(self.alpha_family) / float(self.m_tests)
        else:
            self.alpha_test = float(self.alpha_family)

        self.crit, self.crit_method = _critical_value(self.n_boxes, self.alpha_test)

        bounded_ci_method = str(getattr(conv_cfg, "bounded_ci_method", "t")).strip().lower()
        if bounded_ci_method not in ("t", "empirical_bernstein", "hoeffding"):
            bounded_ci_method = "t"
        self.bounded_ci_method = bounded_ci_method

        self.report: dict[str, Any] = {
            "zscore": float(self.z),
            "mode": str(self.mode),
            "n_boxes": int(self.n_boxes),
            "familywise": {
                "method": self.familywise,
                "alpha_family": float(self.alpha_family),
                "m_tests": int(self.m_tests),
                "alpha_per_test": float(self.alpha_test),
                "crit": float(self.crit),
                "crit_method": str(self.crit_method),
                "bounded_ci_method": str(self.bounded_ci_method),
            },
            "scalars": {},
            "distributions": {},
            "groups": {},
            "stability": {},
            "inference_contract": {
                "interval_method": "fixed_n_per_look",
                "sequentially_valid": False,
                "optional_stopping_coverage_guaranteed": False,
                "interpretation": (
                    "precision/stability diagnostic at the current ensemble size; "
                    "repeated batch looks are not a confidence sequence or alpha-spending design"
                ),
            },
        }
        self.ok_ci = True
        self.group_ok: dict[str, bool] = {}
        self.group_items: dict[str, list[str]] = {}
        self.ok_stab = True

    def run(self) -> tuple[bool, dict[str, Any]]:
        self._evaluate_ci_checks()
        self.report["groups"] = {
            g: {"passed": bool(self.group_ok.get(g, True)), "items": self.group_items.get(g, [])}
            for g in sorted(self.group_items.keys())
        }
        self._evaluate_stability()
        ci_items = {
            **{
                f"scalar:{name}": bool((payload or {}).get("passed", False))
                for name, payload in dict(self.report.get("scalars", {}) or {}).items()
            },
            **{
                f"distribution:{name}": bool((payload or {}).get("passed", False))
                for name, payload in dict(self.report.get("distributions", {}) or {}).items()
            },
        }
        stability_items = {
            str(name): bool((payload or {}).get("passed", False))
            for name, payload in dict(
                (self.report.get("stability", {}) or {}).get("checks", {}) or {}
            ).items()
        }
        self.report["convergence_degree"] = {
            "ci": {
                "n_checked": int(len(ci_items)),
                "n_passed": int(sum(1 for passed in ci_items.values() if passed)),
                "pass_fraction": (
                    float(sum(1 for passed in ci_items.values() if passed)) / float(len(ci_items))
                    if ci_items
                    else None
                ),
                "failed_items": sorted(name for name, passed in ci_items.items() if not passed),
            },
            "stability": {
                "n_checked": int(len(stability_items)),
                "n_passed": int(sum(1 for passed in stability_items.values() if passed)),
                "pass_fraction": (
                    float(sum(1 for passed in stability_items.values() if passed)) / float(len(stability_items))
                    if stability_items
                    else None
                ),
                "failed_items": sorted(
                    name for name, passed in stability_items.items() if not passed
                ),
            },
        }
        self.report["ci_converged"] = bool(self.ok_ci)
        self.report["stability_converged"] = bool(self.ok_stab)
        if self.mode == "ci":
            ok_final = bool(self.ok_ci)
        elif self.mode == "stability":
            ok_final = bool(self.ok_stab)
        else:
            ok_final = bool(self.ok_ci and self.ok_stab)
        self.report["converged"] = bool(ok_final)
        self.report["passed"] = bool(ok_final)
        return bool(ok_final), self.report

    def _count_tests(self) -> int:
        m_tests = 0
        m_tests += 1  # density
        m_tests += int(len(self.scalar_names))
        m_tests += int(len(self.ring_keys))
        if bool(self.spec.get("ring_has_mean_size", False)):
            m_tests += 1

        if self.bond_names:
            for nm in self.bond_names:
                try:
                    x_ref, _mat, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "bondlen", nm, xkey="x")
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        if self.angle_names:
            for nm in self.angle_names:
                try:
                    x_ref, _mat, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "angle", nm, xkey="x")
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        if self.coord_names:
            for nm in self.coord_names:
                try:
                    x_ref, _mat, _meta = _stack_coord_cdfs_for_boxes(self.boxes, nm)
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        if self.gr_labels:
            for lab in self.gr_labels:
                try:
                    x_ref, _mat, _meta = _stack_sampled_curves_for_boxes(
                        self.boxes, "gr", lab, xkey="r", ykey="g"
                    )
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        if self.sq_labels:
            for lab in self.sq_labels:
                try:
                    x_ref, _mat, _meta = _stack_sampled_curves_for_boxes(
                        self.boxes, "sq", lab, xkey="q", ykey="s"
                    )
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        if self.void_names:
            for nm in self.void_names:
                try:
                    x_ref, _mat, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "void", nm, xkey="x")
                    m_tests += int(x_ref.size)
                except Exception:
                    pass
        return int(m_tests)

    def _update_group(self, kind: str, name: str, passed: bool) -> None:
        g = _metric_group(kind)
        self.group_items.setdefault(g, []).append(name)
        self.group_ok[g] = bool(self.group_ok.get(g, True) and bool(passed))

    def _halfwidth_bounded(self, sd_vec: np.ndarray, n: int) -> np.ndarray:
        """Halfwidth bounded."""
        if int(n) < 2:
            return np.full_like(sd_vec, np.inf, dtype=float)
        a = float(self.alpha_test)
        if (not math.isfinite(a)) or a <= 0.0 or a >= 1.0:
            return np.full_like(sd_vec, np.inf, dtype=float)
        if self.bounded_ci_method == "t":
            se = sd_vec / math.sqrt(float(n))
            return float(self.crit) * se
        if self.bounded_ci_method == "hoeffding":
            hw = math.sqrt(math.log(2.0 / a) / (2.0 * float(n)))
            return np.full_like(sd_vec, float(hw), dtype=float)
        v = np.square(sd_vec)
        v = np.minimum(v, 0.25)
        L = math.log(3.0 / a)
        return np.sqrt(2.0 * v * L / float(n)) + 3.0 * L / float(n)

    @staticmethod
    def _w1_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return float("nan")
        xs = np.sort(x)
        ys = np.sort(y)
        z = np.sort(np.concatenate([xs, ys]))
        fx = np.searchsorted(xs, z, side="right") / float(xs.size)
        fy = np.searchsorted(ys, z, side="right") / float(ys.size)
        dz = np.diff(z)
        if dz.size == 0:
            return 0.0
        return float(np.sum(np.abs(fx[:-1] - fy[:-1]) * dz))

    @staticmethod
    def _ks_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return float("nan")
        xs = np.sort(x)
        ys = np.sort(y)
        z = np.sort(np.concatenate([xs, ys]))
        fx = np.searchsorted(xs, z, side="right") / float(xs.size)
        fy = np.searchsorted(ys, z, side="right") / float(ys.size)
        return float(np.max(np.abs(fx - fy)))

    def _bootstrap_upper(self, dist_fn, x: np.ndarray, y: np.ndarray, n_boot: int, q: float) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return float("nan")
        d0 = float(dist_fn(x, y))
        if int(n_boot) <= 0:
            return float(d0)
        rng = np.random.default_rng(0)
        ds = []
        for _ in range(int(n_boot)):
            xb = rng.choice(x, size=x.size, replace=True)
            yb = rng.choice(y, size=y.size, replace=True)
            ds.append(float(dist_fn(xb, yb)))
        return float(np.nanquantile(np.asarray(ds, dtype=float), float(q)))

    def _evaluate_ci_checks(self) -> None:
        self._evaluate_density_scalar()
        self._evaluate_configured_scalars()
        self._evaluate_ring_statistics()
        self._evaluate_ring_mean_size()
        self._evaluate_bondlen_cdfs()
        self._evaluate_angle_cdfs()
        self._evaluate_coord_cdfs()
        self._evaluate_gr_curves()
        self._evaluate_sq_curves()
        self._evaluate_void_cdfs()

    def _evaluate_density_scalar(self) -> None:
        dens_values: list[float] = []
        for box in self.boxes:
            try:
                dens_values.append(float(box.get("density", float("nan"))))
            except Exception:
                dens_values.append(float("nan"))
        dens = np.asarray(dens_values, dtype=float)
        if not np.all(np.isfinite(dens)):
            passed = False
            mu = float("nan")
            sd = float("nan")
            se = float("nan")
            half = float("inf")
        else:
            mu = float(np.mean(dens))
            sd = float(np.std(dens, ddof=1)) if self.n_boxes >= 2 else float("nan")
            se = float(sd / math.sqrt(self.n_boxes)) if self.n_boxes >= 2 else float("nan")
            rel_tol, abs_tol = _tol_scalar("density", self.conv_cfg)
            tol = float(max(abs_tol, rel_tol * abs(mu)))
            half = float(self.crit * se) if (math.isfinite(self.crit) and math.isfinite(se)) else float("inf")
            passed = bool(self.n_boxes >= 2 and math.isfinite(tol) and (half <= tol))
        rel_tol, abs_tol = _tol_scalar("density", self.conv_cfg)
        tol = float(max(abs_tol, rel_tol * abs(mu))) if math.isfinite(mu) else float("nan")
        self.report["scalars"]["density"] = {
            "group": _metric_group("density"),
            "mean": float(mu),
            "std": float(sd),
            "stderr": float(se),
            "ci_halfwidth": float(half),
            "rel_tol": float(rel_tol),
            "abs_tol": float(abs_tol),
            "tol": float(tol),
            "passed": bool(passed),
            "n_effective_boxes": int(np.count_nonzero(np.isfinite(dens))),
        }
        self._update_group("density", "density", passed)
        self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_configured_scalars(self) -> None:
        for name in self.scalar_names:
            values: list[float] = []
            for box in self.boxes:
                try:
                    values.append(
                        float((box.get("metrics", {}) or {}).get(name, float("nan")))
                    )
                except Exception:
                    values.append(float("nan"))
            arr = np.asarray(values, dtype=float)
            finite = bool(np.all(np.isfinite(arr)))
            mu = float(np.mean(arr)) if finite and arr.size else float("nan")
            sd = (
                float(np.std(arr, ddof=1))
                if finite and int(arr.size) >= 2
                else float("nan")
            )
            se = (
                float(sd / math.sqrt(float(arr.size)))
                if math.isfinite(sd) and int(arr.size) >= 2
                else float("nan")
            )
            rel_tol, abs_tol = _tol_scalar(name, self.conv_cfg)
            tol = (
                float(max(abs_tol, rel_tol * abs(mu)))
                if math.isfinite(mu)
                else float("nan")
            )
            half = (
                float(self.crit * se)
                if math.isfinite(self.crit) and math.isfinite(se)
                else float("inf")
            )
            passed = bool(
                int(arr.size) >= 2
                and finite
                and math.isfinite(tol)
                and half <= tol
            )
            family = _convergence_scalar_family(name)
            assert family is not None
            self.report["scalars"][str(name)] = {
                "group": _metric_group(family),
                "family": str(family),
                "mean": float(mu),
                "std": float(sd),
                "stderr": float(se),
                "ci_halfwidth": float(half),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": float(tol),
                "passed": bool(passed),
                "n_effective_boxes": int(arr.size),
            }
            self._update_group(str(family), str(name), passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_ring_statistics(self) -> None:
        if not self.ring_keys:
            return
        mat = np.zeros((self.n_boxes, len(self.ring_keys)), dtype=float)
        for i, b in enumerate(self.boxes):
            m = b.get("metrics", {}) or {}
            for j, k in enumerate(self.ring_keys):
                mat[i, j] = float(m.get(k, float("nan")))
        if not np.all(np.isfinite(mat)):
            passed = False
            mu = np.full((len(self.ring_keys),), np.nan, dtype=float)
            se = np.full((len(self.ring_keys),), np.nan, dtype=float)
            half = np.full((len(self.ring_keys),), np.inf, dtype=float)
            tol_arr = np.full((len(self.ring_keys),), np.nan, dtype=float)
        else:
            mu, sd_vec, se, n = _vector_stats(mat)
            rel_tol_r, abs_tol_r = float(self.conv_cfg.ring_rel_tol), float(self.conv_cfg.ring_abs_tol)
            tol_arr = np.maximum(float(abs_tol_r), float(rel_tol_r) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(np.isfinite(tol_arr)) and np.all(half <= tol_arr))
        worst = int(np.nanargmax(half - tol_arr)) if np.all(np.isfinite(half)) and np.all(np.isfinite(tol_arr)) else None
        self.report["distributions"]["ring"] = {
            "group": _metric_group("ring"),
            "kind": "pmf",
            "keys": self.ring_keys,
            "mean": [float(v) for v in np.asarray(mu, dtype=float).tolist()],
            "stderr": [float(v) for v in np.asarray(se, dtype=float).tolist()],
            "ci_halfwidth": [float(v) for v in np.asarray(half, dtype=float).tolist()],
            "rel_tol": float(self.conv_cfg.ring_rel_tol),
            "abs_tol": float(self.conv_cfg.ring_abs_tol),
            "tol": [float(v) for v in np.asarray(tol_arr, dtype=float).tolist()],
            "passed": bool(passed),
            "worst_index": int(worst) if worst is not None else None,
            "worst_key": self.ring_keys[int(worst)] if worst is not None else None,
            "n_effective_boxes": int(self.n_boxes),
        }
        self._update_group("ring", "ring", passed)
        self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_ring_mean_size(self) -> None:
        if not bool(self.spec.get("ring_has_mean_size", False)):
            return
        arr = np.asarray(
            [float((b.get("metrics", {}) or {}).get("ring_mean_size", float("nan"))) for b in self.boxes],
            dtype=float,
        )
        mu = float(np.mean(arr)) if np.all(np.isfinite(arr)) else float("nan")
        sd = float(np.std(arr, ddof=1)) if (self.n_boxes >= 2 and np.all(np.isfinite(arr))) else float("nan")
        se = float(sd / math.sqrt(self.n_boxes)) if (self.n_boxes >= 2 and math.isfinite(sd)) else float("nan")
        rel_tol_s, abs_tol_s = _tol_for_metric("ring_mean_size", self.conv_cfg)
        tol = float(max(abs_tol_s, rel_tol_s * abs(mu))) if math.isfinite(mu) else float("nan")
        half = float(self.crit * se) if (math.isfinite(self.crit) and math.isfinite(se)) else float("inf")
        passed = bool(self.n_boxes >= 2 and math.isfinite(tol) and (half <= tol))
        self.report["scalars"]["ring_mean_size"] = {
            "group": _metric_group("ring_mean_size"),
            "mean": float(mu),
            "std": float(sd),
            "stderr": float(se),
            "ci_halfwidth": float(half),
            "rel_tol": float(rel_tol_s),
            "abs_tol": float(abs_tol_s),
            "tol": float(tol),
            "passed": bool(passed),
            "n_effective_boxes": int(self.n_boxes),
        }
        self._update_group("ring_mean_size", "ring_mean_size", passed)
        self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_bondlen_cdfs(self) -> None:
        if not self.bond_names:
            return
        rel_tol_b, abs_tol_b = _tol_curve("bondlen_cdf", self.conv_cfg)
        for nm in self.bond_names:
            x_ref, mat, emeta = _stack_continuous_cdfs_for_boxes(self.boxes, "bondlen", nm, xkey="x")
            p = int(x_ref.size)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_b), float(rel_tol_b) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(np.isfinite(half)) and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = json_sanitize({
                **emeta,
                "group": _metric_group("bondlen_cdf"),
                "kind": "bondlen_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "ensemble_cdf": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_b),
                "abs_tol": float(abs_tol_b),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            })
            self._update_group("bondlen_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_angle_cdfs(self) -> None:
        if not self.angle_names:
            return
        rel_tol_a, abs_tol_a = _tol_curve("angle_cdf", self.conv_cfg)
        for nm in self.angle_names:
            x_ref, mat, emeta = _stack_continuous_cdfs_for_boxes(self.boxes, "angle", nm, xkey="x")
            p = int(x_ref.size)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_a), float(rel_tol_a) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(np.isfinite(half)) and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = json_sanitize({
                **emeta,
                "group": _metric_group("angle_cdf"),
                "kind": "angle_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "ensemble_cdf": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_a),
                "abs_tol": float(abs_tol_a),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            })
            self._update_group("angle_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_coord_cdfs(self) -> None:
        if not self.coord_names:
            return
        rel_tol_c, abs_tol_c = _tol_curve("coord_cdf", self.conv_cfg)
        for nm in self.coord_names:
            x_ref, mat, emeta = _stack_coord_cdfs_for_boxes(self.boxes, nm)
            p = int(x_ref.size)
            if p < 1:
                raise RuntimeError(f"Empty coordination CDF for '{nm}'")
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_c), float(rel_tol_c) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(np.isfinite(half)) and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = json_sanitize({
                **emeta,
                "group": _metric_group("coord_cdf"),
                "kind": "coord_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "ensemble_cdf": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_c),
                "abs_tol": float(abs_tol_c),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            })
            self._update_group("coord_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_gr_curves(self) -> None:
        if not self.gr_labels:
            return
        rel_tol_g, abs_tol_g = _tol_curve("gr_curve", self.conv_cfg)
        for lab in self.gr_labels:
            r_ref, mat, emeta = _stack_sampled_curves_for_boxes(
                self.boxes, "gr", lab, xkey="r", ykey="g"
            )
            nb = int(r_ref.size)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_g), float(rel_tol_g) * np.abs(mu))
            half = float(self.crit) * se_vec
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if nb > 0 else None
            self.report["distributions"][lab] = {
                **emeta,
                "group": _metric_group("gr_curve"),
                "kind": "gr_curve",
                "r": [float(v) for v in r_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_g),
                "abs_tol": float(abs_tol_g),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_r": float(r_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            }
            self._update_group("gr_curve", lab, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_sq_curves(self) -> None:
        if not self.sq_labels:
            return
        rel_tol_s, abs_tol_s = _tol_curve("sq_curve", self.conv_cfg)
        for lab in self.sq_labels:
            q_ref, mat, emeta = _stack_sampled_curves_for_boxes(
                self.boxes, "sq", lab, xkey="q", ykey="s"
            )
            nb = int(q_ref.size)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_s), float(rel_tol_s) * np.abs(mu))
            half = float(self.crit) * se_vec
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if nb > 0 else None
            self.report["distributions"][lab] = {
                **emeta,
                "group": _metric_group("sq_curve"),
                "kind": "sq_curve",
                "q": [float(v) for v in q_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_s),
                "abs_tol": float(abs_tol_s),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_q": float(q_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            }
            self._update_group("sq_curve", lab, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_void_cdfs(self) -> None:
        if not self.void_names:
            return
        rel_tol_v, abs_tol_v = _tol_curve("void_cdf", self.conv_cfg)
        for nm in self.void_names:
            x_ref, mat, emeta = _stack_continuous_cdfs_for_boxes(self.boxes, "void", nm, xkey="x")
            p = int(x_ref.size)
            if p < 2:
                raise RuntimeError(f"Invalid void CDF grid for '{nm}'")
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_v), float(rel_tol_v) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(np.isfinite(half)) and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = json_sanitize({
                **emeta,
                "group": _metric_group("void_cdf"),
                "kind": "void_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "ensemble_cdf": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_v),
                "abs_tol": float(abs_tol_v),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
                "n_effective_boxes": int(n),
            })
            self._update_group("void_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_stability(self) -> None:
        stab: dict[str, Any] = {
            "enabled": bool(self.mode in {"stability", "both"}),
            "split": str(getattr(self.conv_cfg, "stability_split", "half")),
            "distance": str(getattr(self.conv_cfg, "stability_distance", "wasserstein")),
            "bootstrap": int(getattr(self.conv_cfg, "stability_bootstrap", 0)),
            "quantile": float(getattr(self.conv_cfg, "stability_quantile", 0.95)),
            "checks": {},
        }

        if self.mode not in {"stability", "both"}:
            self.report["stability"] = stab
            self.ok_stab = True
            return

        split = str(getattr(self.conv_cfg, "stability_split", "half")).strip().lower()
        n_boot = int(getattr(self.conv_cfg, "stability_bootstrap", 0))
        q = float(getattr(self.conv_cfg, "stability_quantile", 0.95))
        dist_kind = str(getattr(self.conv_cfg, "stability_distance", "wasserstein")).strip().lower()
        if dist_kind not in {"wasserstein", "ks"}:
            dist_kind = "wasserstein"

        if split == "last_batch":
            nb = max(1, self.n_boxes // 4)
            if self.n_boxes >= 2 * nb:
                g1 = self.boxes[-2 * nb : -nb]
                g2 = self.boxes[-nb:]
            else:
                split = "half"
                nb = self.n_boxes // 2
                g1 = self.boxes[:nb]
                g2 = self.boxes[nb:]
        else:
            nb = self.n_boxes // 2
            g1 = self.boxes[:nb]
            g2 = self.boxes[nb:]
        stab["n_group1"] = int(len(g1))
        stab["n_group2"] = int(len(g2))

        if len(g1) < 2 or len(g2) < 2:
            self.ok_stab = False
            self.report["stability"] = stab
            return

        def _dist_scalar(x: np.ndarray, y: np.ndarray) -> float:
            if dist_kind == "ks":
                return self._ks_distance_1d(x, y)
            return self._w1_distance_1d(x, y)

        def _curve_distance(mat1: np.ndarray, mat2: np.ndarray) -> float:
            mat1 = np.asarray(mat1, dtype=float)
            mat2 = np.asarray(mat2, dtype=float)
            if mat1.ndim != 2 or mat2.ndim != 2 or mat1.shape[1] != mat2.shape[1]:
                return float("nan")
            p = int(mat1.shape[1])
            dmax = 0.0
            for j in range(p):
                dj = float(_dist_scalar(mat1[:, j], mat2[:, j]))
                if math.isfinite(dj):
                    dmax = max(dmax, dj)
            return float(dmax)

        def _bootstrap_curve_upper(mat1: np.ndarray, mat2: np.ndarray) -> float:
            d0 = _curve_distance(mat1, mat2)
            if int(n_boot) <= 0:
                return float(d0)
            rng = np.random.default_rng(0)
            ds = []
            for _ in range(int(n_boot)):
                i1 = rng.integers(0, mat1.shape[0], size=mat1.shape[0])
                i2 = rng.integers(0, mat2.shape[0], size=mat2.shape[0])
                ds.append(_curve_distance(mat1[i1, :], mat2[i2, :]))
            return float(np.nanquantile(np.asarray(ds, dtype=float), float(q)))

        def _record_scalar_check(name: str, x: np.ndarray, y: np.ndarray, rel_tol: float, abs_tol: float) -> None:
            ref = float(np.nanmedian(np.concatenate([x, y])))
            if dist_kind == "ks":
                # KS is a dimensionless empirical-CDF separation.  Comparing
                # it to density/length/angle absolute tolerances is
                # dimensionally invalid, so it has its own dimensionless
                # effect-size tolerance.
                tol = float(getattr(self.conv_cfg, "stability_ks_tol", 0.10))
                tolerance_basis = "dimensionless_stability_ks_tol"
            else:
                tol = float(max(abs_tol, rel_tol * abs(ref)))
                tolerance_basis = "descriptor_units_max_abs_or_relative"
            d = float(_dist_scalar(x, y))
            upper = float(self._bootstrap_upper(_dist_scalar, x, y, n_boot, q))
            passed = bool(math.isfinite(tol) and math.isfinite(upper) and (upper <= tol))
            stab["checks"][name] = {
                "kind": "scalar",
                "group": _metric_group(
                    _convergence_scalar_family(name) or (
                        "ring_mean_size" if name == "ring_mean_size" else (
                            "ring" if name.startswith("ring_frac_") else name
                        )
                    )
                ),
                "distance": float(d),
                "upper": float(upper),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": float(tol),
                "tolerance_basis": str(tolerance_basis),
                "passed": bool(passed),
                "n_effective_boxes": int(
                    np.count_nonzero(np.isfinite(x))
                    + np.count_nonzero(np.isfinite(y))
                ),
            }
            self.ok_stab = self.ok_stab and bool(passed)

        def _record_curve_check(name: str, mat1: np.ndarray, mat2: np.ndarray, rel_tol: float, abs_tol: float, grid: dict[str, Any]) -> None:
            ref = float(np.nanmax(np.abs(np.nanmean(np.vstack([mat1, mat2]), axis=0))))
            if dist_kind == "ks":
                tol = float(getattr(self.conv_cfg, "stability_ks_tol", 0.10))
                tolerance_basis = "dimensionless_stability_ks_tol"
            else:
                tol = float(max(abs_tol, rel_tol * abs(ref)))
                tolerance_basis = "descriptor_units_max_abs_or_relative"
            d = float(_curve_distance(mat1, mat2))
            upper = float(_bootstrap_curve_upper(mat1, mat2))
            passed = bool(math.isfinite(tol) and math.isfinite(upper) and (upper <= tol))
            stab["checks"][name] = {
                "kind": "curve",
                "group": _metric_group(str(name).split(":", 1)[0]),
                "distance": float(d),
                "upper": float(upper),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": float(tol),
                "tolerance_basis": str(tolerance_basis),
                "passed": bool(passed),
                "grid": dict(grid),
                "n_effective_boxes": int(mat1.shape[0] + mat2.shape[0]),
            }
            self.ok_stab = self.ok_stab and bool(passed)

        x = np.asarray([float(b["density"]) for b in g1], dtype=float)
        y = np.asarray([float(b["density"]) for b in g2], dtype=float)
        rel_tol, abs_tol = _tol_scalar("density", self.conv_cfg)
        _record_scalar_check("density", x, y, rel_tol, abs_tol)

        for name in self.scalar_names:
            family = _convergence_scalar_family(name)
            assert family is not None
            rel_tol_scalar, abs_tol_scalar = _tol_scalar(name, self.conv_cfg)
            x = np.asarray(
                [float((b.get("metrics", {}) or {}).get(name, float("nan"))) for b in g1],
                dtype=float,
            )
            y = np.asarray(
                [float((b.get("metrics", {}) or {}).get(name, float("nan"))) for b in g2],
                dtype=float,
            )
            _record_scalar_check(
                str(name), x, y, rel_tol_scalar, abs_tol_scalar
            )

        if self.ring_keys:
            rel_tol_r, abs_tol_r = float(self.conv_cfg.ring_rel_tol), float(self.conv_cfg.ring_abs_tol)
            for k in self.ring_keys:
                x = np.asarray([float((b.get("metrics", {}) or {}).get(k, float("nan"))) for b in g1], dtype=float)
                y = np.asarray([float((b.get("metrics", {}) or {}).get(k, float("nan"))) for b in g2], dtype=float)
                _record_scalar_check(str(k), x, y, rel_tol_r, abs_tol_r)

        if bool(self.spec.get("ring_has_mean_size", False)):
            rel_tol_s, abs_tol_s = _tol_for_metric("ring_mean_size", self.conv_cfg)
            x = np.asarray([float((b.get("metrics", {}) or {}).get("ring_mean_size", float("nan"))) for b in g1], dtype=float)
            y = np.asarray([float((b.get("metrics", {}) or {}).get("ring_mean_size", float("nan"))) for b in g2], dtype=float)
            _record_scalar_check("ring_mean_size", x, y, rel_tol_s, abs_tol_s)

        if self.bond_names:
            rel_tol_b, abs_tol_b = _tol_curve("bondlen_cdf", self.conv_cfg)
            for nm in self.bond_names:
                x_ref, _mall, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "bondlen", nm, xkey="x")
                _x1, m1, _m1 = _stack_continuous_cdfs_for_boxes(g1, "bondlen", nm, xkey="x", x_ref=x_ref)
                _x2, m2, _m2 = _stack_continuous_cdfs_for_boxes(g2, "bondlen", nm, xkey="x", x_ref=x_ref)
                _record_curve_check(
                    f"bondlen_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_b,
                    abs_tol_b,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(x_ref.size), "grid_source": "ensemble_common_grid"},
                )

        if self.angle_names:
            rel_tol_a, abs_tol_a = _tol_curve("angle_cdf", self.conv_cfg)
            for nm in self.angle_names:
                x_ref, _mall, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "angle", nm, xkey="x")
                _x1, m1, _m1 = _stack_continuous_cdfs_for_boxes(g1, "angle", nm, xkey="x", x_ref=x_ref)
                _x2, m2, _m2 = _stack_continuous_cdfs_for_boxes(g2, "angle", nm, xkey="x", x_ref=x_ref)
                _record_curve_check(
                    f"angle_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_a,
                    abs_tol_a,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(x_ref.size), "grid_source": "ensemble_common_grid"},
                )

        if self.coord_names:
            rel_tol_c, abs_tol_c = _tol_curve("coord_cdf", self.conv_cfg)
            for nm in self.coord_names:
                x_ref, _mall, coord_meta = _stack_coord_cdfs_for_boxes(self.boxes, nm)
                _x1, m1, _m1 = _stack_coord_cdfs_for_boxes(g1, nm, x_ref=x_ref)
                _x2, m2, _m2 = _stack_coord_cdfs_for_boxes(g2, nm, x_ref=x_ref)
                _record_curve_check(
                    f"coord_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_c,
                    abs_tol_c,
                    {
                        "x0": float(x_ref[0]),
                        "x1": float(x_ref[-1]),
                        "p": int(x_ref.size),
                        "grid_source": str(coord_meta.get("grid_source")),
                        "grid_alignment_method": "right_continuous_cdf_evaluation",
                    },
                )

        if self.void_names:
            rel_tol_v, abs_tol_v = _tol_curve("void_cdf", self.conv_cfg)
            for nm in self.void_names:
                x_ref, _mall, _meta = _stack_continuous_cdfs_for_boxes(self.boxes, "void", nm, xkey="x")
                _x1, m1, _m1 = _stack_continuous_cdfs_for_boxes(g1, "void", nm, xkey="x", x_ref=x_ref)
                _x2, m2, _m2 = _stack_continuous_cdfs_for_boxes(g2, "void", nm, xkey="x", x_ref=x_ref)
                _record_curve_check(
                    f"void_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_v,
                    abs_tol_v,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(x_ref.size), "grid_source": "ensemble_common_grid"},
                )

        if self.gr_labels:
            rel_tol_g, abs_tol_g = _tol_curve("gr_curve", self.conv_cfg)
            for lab in self.gr_labels:
                r_ref, _mall, _meta = _stack_sampled_curves_for_boxes(
                    self.boxes, "gr", lab, xkey="r", ykey="g"
                )
                _x1, m1, _m1 = _stack_sampled_curves_for_boxes(
                    g1, "gr", lab, xkey="r", ykey="g", x_ref=r_ref
                )
                _x2, m2, _m2 = _stack_sampled_curves_for_boxes(
                    g2, "gr", lab, xkey="r", ykey="g", x_ref=r_ref
                )
                _record_curve_check(
                    f"gr_curve:{lab}",
                    m1,
                    m2,
                    rel_tol_g,
                    abs_tol_g,
                    {
                        "x0": float(r_ref[0]),
                        "x1": float(r_ref[-1]),
                        "p": int(r_ref.size),
                        "grid_source": "ensemble_common_support_grid",
                        "grid_alignment_method": "linear_interpolation_on_common_support",
                    },
                )

        if self.sq_labels:
            rel_tol_s, abs_tol_s = _tol_curve("sq_curve", self.conv_cfg)
            for lab in self.sq_labels:
                q_ref, _mall, _meta = _stack_sampled_curves_for_boxes(
                    self.boxes, "sq", lab, xkey="q", ykey="s"
                )
                _x1, m1, _m1 = _stack_sampled_curves_for_boxes(
                    g1, "sq", lab, xkey="q", ykey="s", x_ref=q_ref
                )
                _x2, m2, _m2 = _stack_sampled_curves_for_boxes(
                    g2, "sq", lab, xkey="q", ykey="s", x_ref=q_ref
                )
                _record_curve_check(
                    f"sq_curve:{lab}",
                    m1,
                    m2,
                    rel_tol_s,
                    abs_tol_s,
                    {
                        "x0": float(q_ref[0]),
                        "x1": float(q_ref[-1]),
                        "p": int(q_ref.size),
                        "grid_source": "ensemble_common_support_grid",
                        "grid_alignment_method": "linear_interpolation_on_common_support",
                    },
                )


        self.report["stability"] = stab


def _achieved_convergence_degree(
    report: Mapping[str, Any],
    *,
    n_boxes: int,
) -> dict[str, Any]:
    """Summarise how close the worst assessed check is to its tolerance.

    Ratios are dimensionless (observed uncertainty or stability upper bound
    divided by its own tolerance), so heterogeneous descriptor families can be
    ranked without comparing unlike physical units.  The signed margin is
    retained only with the identity of that worst check and is therefore in
    that check's native units.
    """

    def _finite_vector(value: Any) -> np.ndarray:
        try:
            return np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)

    def _add_candidates(
        target: list[dict[str, Any]],
        *,
        section: str,
        name: str,
        observed: Any,
        tolerance: Any,
    ) -> None:
        obs = _finite_vector(observed)
        tol = _finite_vector(tolerance)
        if obs.size == 0 or tol.size == 0:
            return
        if obs.size == 1 and tol.size > 1:
            obs = np.full_like(tol, float(obs[0]), dtype=float)
        elif tol.size == 1 and obs.size > 1:
            tol = np.full_like(obs, float(tol[0]), dtype=float)
        if obs.size != tol.size:
            return
        for index, (oval, tval) in enumerate(zip(obs.tolist(), tol.tolist())):
            o = float(oval)
            t = float(tval)
            if not (math.isfinite(o) and math.isfinite(t) and t >= 0.0):
                continue
            if t > 0.0:
                ratio = o / t
            elif o == 0.0:
                ratio = 0.0
            else:
                continue
            if not math.isfinite(ratio):
                continue
            target.append(
                {
                    "section": str(section),
                    "name": str(name),
                    "component_index": int(index),
                    "observed": float(o),
                    "tolerance": float(t),
                    "tolerance_utilization_ratio": float(ratio),
                    "signed_margin": float(t - o),
                    "passed": bool(o <= t),
                }
            )

    ci_candidates: list[dict[str, Any]] = []
    for name, payload in dict(report.get("scalars", {}) or {}).items():
        if isinstance(payload, Mapping):
            _add_candidates(
                ci_candidates,
                section="ci",
                name=f"scalar:{name}",
                observed=payload.get("ci_halfwidth"),
                tolerance=payload.get("tol"),
            )
    for name, payload in dict(report.get("distributions", {}) or {}).items():
        if isinstance(payload, Mapping):
            _add_candidates(
                ci_candidates,
                section="ci",
                name=f"distribution:{name}",
                observed=payload.get("ci_halfwidth"),
                tolerance=payload.get("tol"),
            )

    stability_candidates: list[dict[str, Any]] = []
    stability = report.get("stability", {}) or {}
    checks = stability.get("checks", {}) if isinstance(stability, Mapping) else {}
    for name, payload in dict(checks or {}).items():
        if isinstance(payload, Mapping):
            _add_candidates(
                stability_candidates,
                section="stability",
                name=str(name),
                observed=payload.get("upper"),
                tolerance=payload.get("tol"),
            )

    def _summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {
                "assessed": False,
                "n_components": 0,
                "worst_tolerance_utilization_ratio": None,
                "worst_signed_margin": None,
                "worst_check": None,
            }
        worst = max(
            candidates,
            key=lambda row: float(row.get("tolerance_utilization_ratio", -math.inf)),
        )
        return {
            "assessed": True,
            "n_components": int(len(candidates)),
            "worst_tolerance_utilization_ratio": float(
                worst["tolerance_utilization_ratio"]
            ),
            "worst_signed_margin": float(worst["signed_margin"]),
            "worst_check": dict(worst),
        }

    mode = str(report.get("mode", "both"))
    active = (
        ci_candidates
        if mode == "ci"
        else (
            stability_candidates
            if mode == "stability"
            else [*ci_candidates, *stability_candidates]
        )
    )
    return {
        "n_boxes": int(n_boxes),
        "mode": str(mode),
        "ratio_definition": "observed_uncertainty_or_stability_upper_bound / tolerance",
        "passing_ratio_max": 1.0,
        "ci": _summary(ci_candidates),
        "stability": _summary(stability_candidates),
        "overall_active": _summary(active),
    }


def _convergence_family_for_item(
    name: str,
    payload: Mapping[str, Any],
) -> str | None:
    if str(name) == "density":
        return "density"
    scalar_family = _convergence_scalar_family(str(name))
    if scalar_family is not None:
        return scalar_family
    if str(name) in {"ring", "ring_mean_size"} or str(name).startswith("ring_frac_"):
        return "ring"
    kind = str(payload.get("kind", ""))
    if kind in _CONVERGENCE_FAMILY_TO_GROUP:
        return kind
    if kind == "pmf":
        return "ring"
    return None


def _build_convergence_evidence_coverage(
    *,
    report: Mapping[str, Any],
    requested_spec: Mapping[str, Any],
    skipped: Sequence[Mapping[str, Any]],
    conv_cfg: Any,
    n_boxes: int,
) -> dict[str, Any]:
    """Audit descriptor-family and majority-box coverage for a convergence claim."""

    fraction = float(getattr(conv_cfg, "minimum_evidence_fraction", 0.5))
    minimum_required = int(math.floor(float(fraction) * float(n_boxes)) + 1)
    mode = str(report.get("mode", getattr(conv_cfg, "mode", "both")))
    active_sections = (
        ["ci"]
        if mode == "ci"
        else (["stability"] if mode == "stability" else ["ci", "stability"])
    )
    items: dict[str, dict[str, Any]] = {}

    for name, payload_raw in dict(report.get("scalars", {}) or {}).items():
        payload = dict(payload_raw or {})
        family = _convergence_family_for_item(str(name), payload)
        n_effective = int(payload.get("n_effective_boxes", 0) or 0)
        key = f"ci:scalar:{name}"
        items[key] = {
            "section": "ci",
            "name": f"scalar:{name}",
            "family": family,
            "group": str(payload.get("group", _CONVERGENCE_FAMILY_TO_GROUP.get(str(family), "other"))),
            "n_contributing_boxes": n_effective,
            "minimum_boxes_required": minimum_required,
            "strict_majority_supported": bool(n_effective >= minimum_required),
        }
    for name, payload_raw in dict(report.get("distributions", {}) or {}).items():
        payload = dict(payload_raw or {})
        family = _convergence_family_for_item(str(name), payload)
        n_effective = int(payload.get("n_effective_boxes", 0) or 0)
        key = f"ci:distribution:{name}"
        items[key] = {
            "section": "ci",
            "name": f"distribution:{name}",
            "family": family,
            "group": str(payload.get("group", _CONVERGENCE_FAMILY_TO_GROUP.get(str(family), "other"))),
            "n_contributing_boxes": n_effective,
            "minimum_boxes_required": minimum_required,
            "strict_majority_supported": bool(n_effective >= minimum_required),
        }
    stability = dict(report.get("stability", {}) or {})
    for name, payload_raw in dict(stability.get("checks", {}) or {}).items():
        payload = dict(payload_raw or {})
        family = _convergence_family_for_item(str(name), payload)
        if family is None and ":" in str(name):
            family = str(name).split(":", 1)[0]
        n_effective = int(payload.get("n_effective_boxes", 0) or 0)
        key = f"stability:{name}"
        items[key] = {
            "section": "stability",
            "name": str(name),
            "family": family,
            "group": str(payload.get("group", _CONVERGENCE_FAMILY_TO_GROUP.get(str(family), "other"))),
            "n_contributing_boxes": n_effective,
            "minimum_boxes_required": minimum_required,
            "strict_majority_supported": bool(n_effective >= minimum_required),
        }

    zero_incidence_families: set[str] = set()
    zero_incidence_items: list[dict[str, Any]] = []
    for issue_raw in skipped:
        issue = dict(issue_raw or {})
        if str(issue.get("status", "")) != "valid_zero_incidence":
            continue
        kind = str(issue.get("kind", ""))
        family = "ring" if kind == "ring_mean_size" else kind
        if family not in _CONVERGENCE_FAMILY_TO_GROUP:
            continue
        zero_incidence_families.add(family)
        zero_incidence_items.append(
            {
                "name": str(issue.get("name", family)),
                "family": str(family),
                "group": _CONVERGENCE_FAMILY_TO_GROUP[family],
                "n_contributing_boxes": int(n_boxes),
                "minimum_boxes_required": minimum_required,
                "strict_majority_supported": bool(n_boxes >= minimum_required),
                "status": "measured_zero_incidence_in_every_box",
            }
        )

    configured_families = list(
        dict.fromkeys(
            str(name)
            for name in list(requested_spec.get("configured_metric_families", []) or [])
        )
    )
    active_items = {
        key: value
        for key, value in items.items()
        if str(value.get("section")) in active_sections
    }
    family_coverage: dict[str, Any] = {}
    for family in configured_families:
        family_items = [
            dict(value)
            for value in active_items.values()
            if str(value.get("family")) == str(family)
        ]
        all_majority = bool(family_items) and all(
            bool(value.get("strict_majority_supported", False))
            for value in family_items
        )
        zero_measured = family in zero_incidence_families
        covered = bool(all_majority or zero_measured)
        family_coverage[family] = {
            "group": _CONVERGENCE_FAMILY_TO_GROUP.get(family, "other"),
            "n_active_assessments": int(len(family_items)),
            "active_assessments": [str(value.get("name")) for value in family_items],
            "all_assessments_strict_majority_supported": bool(all_majority),
            "measured_zero_incidence_in_every_box": bool(zero_measured),
            "covered": bool(covered),
            "status": (
                "measured_zero_incidence"
                if zero_measured and not family_items
                else ("assessed" if covered else "missing_or_insufficient_evidence")
            ),
        }

    groups: dict[str, Any] = {}
    for group in ("short", "medium", "long"):
        required_families = [
            family
            for family in configured_families
            if _CONVERGENCE_FAMILY_TO_GROUP.get(family) == group
        ]
        covered = bool(required_families) and all(
            bool((family_coverage.get(family, {}) or {}).get("covered", False))
            for family in required_families
        )
        groups[group] = {
            "configured": bool(required_families),
            "required_families": required_families,
            "covered": bool(covered) if required_families else None,
            "status": (
                "covered"
                if covered
                else ("not_configured" if not required_families else "missing_or_insufficient_evidence")
            ),
        }

    insufficient_items = [
        dict(value)
        for value in active_items.values()
        if not bool(value.get("strict_majority_supported", False))
    ]
    missing_families = [
        family
        for family, payload in family_coverage.items()
        if not bool((payload or {}).get("covered", False))
    ]
    passed = bool(not insufficient_items and not missing_families)
    return json_sanitize(
        {
            "schema": "vitriflow.convergence_evidence_coverage.v1",
            "policy": "n_contributing_boxes_strictly_greater_than_fraction_of_accepted_ensemble",
            "minimum_evidence_fraction_exclusive": float(fraction),
            "n_boxes_total": int(n_boxes),
            "minimum_boxes_required": int(minimum_required),
            "active_sections": active_sections,
            "configured_metric_families": configured_families,
            "items": items,
            "zero_incidence_items": zero_incidence_items,
            "families": family_coverage,
            "groups": groups,
            "insufficient_items": insufficient_items,
            "missing_families": missing_families,
            "passed": bool(passed),
        }
    )


_CANONICAL_DISTRIBUTION_FIELDS = {
    "group", "kind", "family", "keys", "x", "r", "q", "mean",
    "ensemble_cdf", "stderr", "ci_halfwidth", "rel_tol", "abs_tol",
    "tol", "passed", "worst_index", "worst_x", "worst_r", "worst_q",
    "worst_key", "n_effective_boxes", "representation", "grid_source",
    "grid_alignment_method", "common_support", "support_policy",
    "source_grids_same", "source_grid_lengths", "grid_mismatch_count",
    "x_min", "x_max", "n_points", "box_ids",
    "representation_validation", "representation_validation_status",
    "representation_effective_r_support_A",
}


def canonical_convergence_assessment(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the context-independent numerical convergence contract.

    Stopping/advisory labels describe where an assessment was consumed, not a
    different mathematical result.  They are intentionally excluded.  All
    criterion-defining inputs and outputs remain, including inference
    qualification, effective/requested specs, groups, CI/stability evidence,
    criteria integrity, achieved degree and the final pass result.
    """

    source = dict(report or {})
    inference = dict(source.get("inference_contract", {}) or {})
    distributions = {
        str(name): {
            str(key): value
            for key, value in dict(payload or {}).items()
            if str(key) in _CANONICAL_DISTRIBUTION_FIELDS
        }
        for name, payload in sorted(dict(source.get("distributions", {}) or {}).items())
    }
    groups = {
        str(name): {
            "passed": (payload or {}).get("passed"),
            "items": sorted(str(item) for item in list((payload or {}).get("items", []) or [])),
        }
        for name, payload in sorted(dict(source.get("groups", {}) or {}).items())
    }
    return json_sanitize(
        {
            "mode": source.get("mode"),
            "n_boxes": source.get("n_boxes"),
            "familywise": source.get("familywise"),
            "convergence_spec_requested": source.get("convergence_spec_requested"),
            "convergence_spec_effective": source.get("convergence_spec_effective"),
            "groups": groups,
            "scalars": source.get("scalars", {}),
            "distributions": distributions,
            "stability": source.get("stability", {}),
            "criteria_integrity": source.get("criteria_integrity", {}),
            "evidence_coverage": source.get("evidence_coverage", {}),
            "metric_plumbing_coverage": source.get("metric_plumbing_coverage", {}),
            "convergence_degree": source.get("convergence_degree", {}),
            "achieved_convergence_degree": source.get("achieved_convergence_degree", {}),
            "ci_converged": bool(source.get("ci_converged", False)),
            "stability_converged": bool(source.get("stability_converged", False)),
            "converged": bool(source.get("converged", source.get("passed", False))),
            "passed": bool(source.get("passed", source.get("converged", False))),
            "inference_qualification": {
                "interval_method": inference.get("interval_method"),
                "sequentially_valid": inference.get("sequentially_valid"),
                "optional_stopping_coverage_guaranteed": inference.get(
                    "optional_stopping_coverage_guaranteed"
                ),
            },
        }
    )


def compare_convergence_assessments(
    reference: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two public convergence reports under the canonical contract."""

    ref = canonical_convergence_assessment(reference)
    got = canonical_convergence_assessment(replay)
    differences: list[dict[str, Any]] = []

    def _walk(a: Any, b: Any, path: str) -> None:
        if len(differences) >= 64:
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in a:
                    differences.append({"path": child, "reference": "<missing>", "replay": b[key]})
                elif key not in b:
                    differences.append({"path": child, "reference": a[key], "replay": "<missing>"})
                else:
                    _walk(a[key], b[key], child)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({"path": path, "reference_length": len(a), "replay_length": len(b)})
                return
            for index, (left, right) in enumerate(zip(a, b)):
                _walk(left, right, f"{path}[{index}]")
            return
        if a != b:
            differences.append({"path": path, "reference": a, "replay": b})

    _walk(ref, got, "")
    return json_sanitize(
        {
            "schema": "vitriflow.convergence_parity.v1",
            "equivalent": bool(not differences),
            "comparison": "exact_canonical_numerical_contract",
            "n_differences": int(len(differences)),
            "differences": differences,
            "reference": ref,
            "replay": got,
        }
    )


def _metric_plumbing_coverage(
    requested_spec: Mapping[str, Any],
    effective_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarise which emitted descriptors do and do not enter convergence.

    This is intentionally diagnostic rather than a universal convergence
    blocker: a scientifically valid workflow may request only an S(q) curve
    plus several derived peak annotations.  Release/application validation can
    nevertheless require a strict majority for its deliberately broad test
    configurations without any silent metric omission.
    """

    requested = dict(requested_spec or {})
    effective = dict(effective_spec or {})
    classifications = dict(
        effective.get(
            "scalar_metric_classification",
            requested.get("scalar_metric_classification", {}),
        )
        or {}
    )

    emitted: list[dict[str, Any]] = [
        {
            "name": "density",
            "kind": "scalar",
            "role": "convergence",
            "family": "density",
            "group": "long",
        }
    ]
    for name, payload_raw in sorted(classifications.items()):
        payload = dict(payload_raw or {})
        emitted.append(
            {
                "name": str(name),
                "kind": "scalar",
                "role": str(payload.get("role", "diagnostic_only")),
                "family": payload.get("family"),
                "group": payload.get("group"),
                "reason": payload.get("reason"),
                "value_status": payload.get("value_status"),
            }
        )

    distribution_families = (
        ("bondlen_names", "bondlen_cdf", "short"),
        ("angle_names", "angle_cdf", "short"),
        ("coord_names", "coord_cdf", "short"),
        ("gr_labels", "gr_curve", "long"),
        ("sq_labels", "sq_curve", "long"),
        ("void_names", "void_cdf", "long"),
    )
    for spec_key, family, group in distribution_families:
        names = {
            str(name) for name in list(requested.get(spec_key, []) or [])
        }
        names.update(str(name) for name in list(effective.get(spec_key, []) or []))
        for name in sorted(names):
            emitted.append(
                {
                    "name": str(name),
                    "kind": "distribution",
                    "role": "convergence",
                    "family": str(family),
                    "group": str(group),
                }
            )

    convergence_items = [
        item for item in emitted if str(item.get("role")) == "convergence"
    ]
    diagnostic_items = [
        item for item in emitted if str(item.get("role")) != "convergence"
    ]
    total = int(len(emitted))
    n_convergence = int(len(convergence_items))
    fraction = float(n_convergence / total) if total else None
    return json_sanitize(
        {
            "schema": "vitriflow.metric_plumbing_coverage.v1",
            "n_emitted_descriptors": total,
            "n_convergence_descriptors": n_convergence,
            "n_diagnostic_only_descriptors": int(len(diagnostic_items)),
            "fraction_entering_convergence": fraction,
            "strict_majority_enters_convergence": bool(
                fraction is not None and fraction > 0.5
            ),
            "convergence_descriptors": convergence_items,
            "diagnostic_only_descriptors": diagnostic_items,
        }
    )


def check_production_convergence(boxes: list[dict[str, Any]], spec: dict[str, Any], conv_cfg) -> tuple[bool, dict[str, Any]]:
    """Production convergence.

    Conditional distributions with explicit zero incidence in every box are
    retained as valid undefined diagnostics.  Requested descriptors that are
    missing, corrupt, or disappear in only part of the ensemble fail the
    convergence decision instead of silently changing its metric set.
    """
    requested_spec = dict(spec or {})
    clean_spec, skipped = _sanitize_production_convergence_spec(boxes, requested_spec)
    ok, report = _ProductionConvergenceChecker(boxes, clean_spec, conv_cfg).run()
    report["metric_plumbing_coverage"] = _metric_plumbing_coverage(
        requested_spec,
        clean_spec,
    )
    # Initial-spec and ensemble sanitisation may describe the same payload.
    # De-duplicate for stable public diagnostics without weakening any blocker.
    unique_issues: list[dict[str, Any]] = []
    seen_issues: set[tuple[Any, ...]] = set()
    for item in skipped:
        row = dict(item or {})
        key = (
            row.get("kind"),
            row.get("name"),
            row.get("box"),
            row.get("status"),
            bool(row.get("blocking", True)),
        )
        if key not in seen_issues:
            seen_issues.add(key)
            unique_issues.append(row)
    skipped = unique_issues
    blocking_issues = [dict(item) for item in skipped if bool(item.get("blocking", True))]
    zero_incidence = [
        dict(item) for item in skipped
        if str(item.get("status", "")) == "valid_zero_incidence"
    ]
    if blocking_issues:
        report["statistical_convergence_before_criteria_integrity"] = {
            "ci_converged": bool(report.get("ci_converged", False)),
            "stability_converged": bool(report.get("stability_converged", False)),
        }
        mode = str(report.get("mode", getattr(conv_cfg, "mode", "both")))
        if mode in {"ci", "both"}:
            report["ci_converged"] = False
        if mode in {"stability", "both"}:
            report["stability_converged"] = False
        ok = False
        report["converged"] = False
        report["passed"] = False
    report["convergence_spec_requested"] = requested_spec
    report["convergence_spec_effective"] = clean_spec
    report["ensemble_cdfs"] = build_ensemble_cdf_sidecar(boxes, clean_spec)
    report["skipped_metrics"] = list(skipped)
    report["criteria_integrity"] = {
        "passed": bool(not blocking_issues),
        "blocking_issue_count": int(len(blocking_issues)),
        "blocking_issues": blocking_issues,
        "valid_zero_incidence_count": int(len(zero_incidence)),
        "valid_zero_incidence": zero_incidence,
    }
    evidence_coverage = _build_convergence_evidence_coverage(
        report=report,
        requested_spec=requested_spec,
        skipped=skipped,
        conv_cfg=conv_cfg,
        n_boxes=len(boxes),
    )
    report["evidence_coverage"] = evidence_coverage
    if not bool(evidence_coverage.get("passed", False)):
        report["statistical_convergence_before_evidence_coverage"] = {
            "ci_converged": bool(report.get("ci_converged", False)),
            "stability_converged": bool(report.get("stability_converged", False)),
        }
        mode_for_coverage = str(report.get("mode", getattr(conv_cfg, "mode", "both")))
        if mode_for_coverage in {"ci", "both"}:
            report["ci_converged"] = False
        if mode_for_coverage in {"stability", "both"}:
            report["stability_converged"] = False
        ok = False
        report["converged"] = False
        report["passed"] = False

    report_groups = dict(report.get("groups", {}) or {})
    coverage_groups = dict(evidence_coverage.get("groups", {}) or {})
    for group_name in ("short", "medium", "long"):
        group = dict(report_groups.get(group_name, {}) or {})
        coverage_group = dict(coverage_groups.get(group_name, {}) or {})
        group.setdefault("items", [])
        group["configured"] = bool(coverage_group.get("configured", False))
        group["evidence_covered"] = coverage_group.get("covered")
        group["evidence_status"] = coverage_group.get("status")
        if bool(group.get("configured", False)) and not bool(
            coverage_group.get("covered", False)
        ):
            group["passed"] = False
        elif "passed" not in group:
            group["passed"] = None
        report_groups[group_name] = group
    report["groups"] = report_groups
    degree = dict(report.get("convergence_degree", {}) or {})
    degree["criteria_integrity"] = {
        "n_checked": int(len(blocking_issues)),
        "n_passed": 0,
        "passed": bool(not blocking_issues),
        "valid_zero_incidence_count": int(len(zero_incidence)),
    }
    coverage_items = dict(evidence_coverage.get("items", {}) or {})
    active_coverage_items = [
        dict(item)
        for item in coverage_items.values()
        if str(item.get("section"))
        in set(str(x) for x in list(evidence_coverage.get("active_sections", []) or []))
    ]
    configured_family_coverage = dict(evidence_coverage.get("families", {}) or {})
    degree["evidence_coverage"] = {
        "n_checked": int(len(active_coverage_items) + len(configured_family_coverage)),
        "n_passed": int(
            sum(
                bool(item.get("strict_majority_supported", False))
                for item in active_coverage_items
            )
            + sum(
                bool(item.get("covered", False))
                for item in configured_family_coverage.values()
            )
        ),
        "passed": bool(evidence_coverage.get("passed", False)),
        "failed_items": [
            str(item.get("name"))
            for item in active_coverage_items
            if not bool(item.get("strict_majority_supported", False))
        ]
        + [
            f"configured_family:{name}"
            for name, item in configured_family_coverage.items()
            if not bool((item or {}).get("covered", False))
        ],
    }
    mode = str(report.get("mode", getattr(conv_cfg, "mode", "both")))
    active_sections = ["ci"] if mode == "ci" else (["stability"] if mode == "stability" else ["ci", "stability"])
    n_stat_checked = int(
        sum(int((degree.get(section, {}) or {}).get("n_checked", 0) or 0) for section in active_sections)
    )
    n_stat_passed = int(
        sum(int((degree.get(section, {}) or {}).get("n_passed", 0) or 0) for section in active_sections)
    )
    unassessed_active_sections = [
        section
        for section in active_sections
        if int((degree.get(section, {}) or {}).get("n_checked", 0) or 0) == 0
    ]
    # An active convergence mode with no evaluable check is an unassessed
    # failure gate, not a zero-denominator success.
    evidence_degree = dict(degree.get("evidence_coverage", {}) or {})
    n_overall = int(
        n_stat_checked
        + len(unassessed_active_sections)
        + len(blocking_issues)
        + int(evidence_degree.get("n_checked", 0) or 0)
    )
    n_overall_passed = int(
        n_stat_passed + int(evidence_degree.get("n_passed", 0) or 0)
    )
    degree["overall"] = {
        "assessed": bool(n_overall > 0),
        "n_checked": int(n_overall),
        "n_passed": int(n_overall_passed),
        "pass_fraction": (
            float(n_overall_passed) / float(n_overall) if n_overall > 0 else None
        ),
        "converged": bool(ok),
        "unassessed_active_sections": unassessed_active_sections,
        "status": (
            "unassessed"
            if n_overall == 0
            else ("converged" if bool(ok) else "not_converged")
        ),
    }
    report["convergence_degree"] = degree
    report["achieved_convergence_degree"] = _achieved_convergence_degree(
        report,
        n_boxes=len(boxes),
    )
    if skipped:
        notes = list(report.get("notes", []) or [])
        if blocking_issues:
            notes.append(
                f"Convergence failed closed because {len(blocking_issues)} requested metric payload(s) "
                "were missing, corrupt, or had mixed zero/nonzero incidence."
            )
        if zero_incidence:
            notes.append(
                f"Recorded {len(zero_incidence)} conditional metric(s) as valid zero-incidence "
                "diagnostics; no artificial CDF was constructed."
            )
        report["notes"] = notes
    return bool(ok), report


def assess_fixed_count_convergence_posthoc(
    boxes: Sequence[Mapping[str, Any]],
    spec: Optional[Mapping[str, Any]],
    conv_cfg: Any,
    *,
    execution_target_met: bool,
    min_boxes: int,
) -> dict[str, Any]:
    """Assess a terminal fixed-count ensemble without turning it into a stop rule.

    The numerical checks are exactly the production convergence checks used by
    adaptive production, but they are evaluated once, after fixed-count
    execution has ended.  The returned labels deliberately distinguish the
    fixed-n, post-hoc diagnostic from an adaptive or sequentially valid stopping
    decision.
    """

    box_rows = [dict(box) for box in boxes]
    mode_raw = str(getattr(conv_cfg, "mode", "both")).strip().lower()
    mode = mode_raw if mode_raw in {"ci", "stability", "both"} else "both"

    if box_rows:
        effective_spec = (
            dict(spec)
            if isinstance(spec, Mapping)
            else build_production_convergence_spec(box_rows[0])
        )
        criterion_met, report = check_production_convergence(
            box_rows,
            effective_spec,
            conv_cfg,
        )
        out = dict(report)
        assessment_performed = True
        status = "fixed_n_terminal_posthoc_assessed"
    else:
        active_sections = (
            ["ci"]
            if mode == "ci"
            else (["stability"] if mode == "stability" else ["ci", "stability"])
        )
        criterion_met = None
        assessment_performed = False
        status = "fixed_n_terminal_posthoc_unassessed"
        out = {
            "mode": str(mode),
            "n_boxes": 0,
            "scalars": {},
            "distributions": {},
            "groups": {},
            "stability": {},
            "ci_converged": False,
            "stability_converged": False,
            "converged": False,
            "passed": False,
            "criteria_integrity": {
                "passed": True,
                "blocking_issue_count": 0,
                "blocking_issues": [],
                "valid_zero_incidence_count": 0,
                "valid_zero_incidence": [],
            },
            "convergence_degree": {
                "ci": {
                    "n_checked": 0,
                    "n_passed": 0,
                    "pass_fraction": None,
                    "failed_items": [],
                },
                "stability": {
                    "n_checked": 0,
                    "n_passed": 0,
                    "pass_fraction": None,
                    "failed_items": [],
                },
                "criteria_integrity": {
                    "n_checked": 0,
                    "n_passed": 0,
                    "passed": True,
                    "valid_zero_incidence_count": 0,
                },
                "overall": {
                    "assessed": False,
                    "n_checked": 0,
                    "n_passed": 0,
                    "pass_fraction": None,
                    "converged": False,
                    "unassessed_active_sections": list(active_sections),
                    "status": "unassessed",
                },
            },
            "notes": [
                "The fixed-count terminal post-hoc diagnostic is unassessed because "
                "there are no accepted boxes."
            ],
        }
        out["achieved_convergence_degree"] = _achieved_convergence_degree(
            out,
            n_boxes=0,
        )

    degree = dict(out.get("convergence_degree", {}) or {})
    active_sections = (
        ["ci"]
        if mode == "ci"
        else (["stability"] if mode == "stability" else ["ci", "stability"])
    )
    failed_items: list[dict[str, Any]] = []
    for section in active_sections:
        section_degree = degree.get(section, {})
        if not isinstance(section_degree, Mapping):
            continue
        for name in list(section_degree.get("failed_items", []) or []):
            failed_items.append(
                {
                    "section": str(section),
                    "name": str(name),
                    "reason": "tolerance_not_met",
                }
            )
    overall_degree = degree.get("overall", {})
    if isinstance(overall_degree, Mapping):
        for section in list(overall_degree.get("unassessed_active_sections", []) or []):
            failed_items.append(
                {
                    "section": str(section),
                    "name": str(section),
                    "reason": "active_section_unassessed",
                }
            )
    criteria_integrity = out.get("criteria_integrity", {})
    if isinstance(criteria_integrity, Mapping):
        for issue in list(criteria_integrity.get("blocking_issues", []) or []):
            issue_row = dict(issue) if isinstance(issue, Mapping) else {"name": str(issue)}
            failed_items.append(
                {
                    "section": "criteria_integrity",
                    "name": str(issue_row.get("name", issue_row.get("kind", "unknown"))),
                    "reason": str(issue_row.get("status", "blocking_issue")),
                    "details": issue_row,
                }
            )
    if not box_rows:
        failed_items.append(
            {
                "section": "ensemble",
                "name": "accepted_boxes",
                "reason": "no_accepted_boxes",
            }
        )

    inference_contract = dict(out.get("inference_contract", {}) or {})
    inference_contract.update(
        {
            "assessment_design": "fixed_n_terminal_posthoc",
            "sequentially_valid": False,
            "optional_stopping_coverage_guaranteed": False,
            "used_for_stopping": False,
            "interpretation": (
                "One terminal fixed-n precision/stability diagnostic evaluated after "
                "execution; it is post-hoc, is not a sequential stopping decision, and "
                "does not imply optional-stopping coverage."
            ),
        }
    )
    out.update(
        {
            "status": str(status),
            "sampling_design": "fixed_n",
            "assessment_role": "terminal_posthoc_diagnostic",
            "assessment_performed": bool(assessment_performed),
            "check_convergence": False,
            "used_for_stopping": False,
            "stopping_assessment_performed": False,
            "stopping_status": "fixed_count_unassessed",
            "sequential_inference_status": "not_sequentially_valid",
            "posthoc_criterion_met": (
                None if criterion_met is None else bool(criterion_met)
            ),
            "posthoc_failed_items": failed_items,
            "execution_target_met": bool(execution_target_met),
            "n_boxes_accepted": int(len(box_rows)),
            "min_boxes": int(min_boxes),
            "inference_contract": inference_contract,
        }
    )
    return out
