
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..analysis.amorphous import analyse_amorphous_state, reduced_formula_from_frame
from ..analysis.gr import compute_gr
from ..analysis.structure import (
    compute_coordination_defect_details,
    compute_coordination_defects,
    compute_structure_distributions_timeavg,
    compute_structure_metrics_timeavg,
    estimate_pair_cutoffs,
)
from ..analysis.trajectory import read_last_frames_auto
from ..io.extxyz import write_extxyz_single_with_species
from .elastic_screen import should_collect_elastic_stage_timeseries
from .stage_metrics import collect_stage_metrics_timeseries, should_collect_stage_metrics_timeseries
from .step_counts import recommended_quench_dump_every, resolve_lammps_units_style
from .quench_rates import lammps_timeunit_ps
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
    rep = tuple(int(x) for x in replicate)
    if len(rep) != 3:
        raise ValueError("replicate must have length 3")
    return ProductionPlan(
        engine=str(engine),
        structure_data=Path(structure_data),
        T_high=float(T_high),
        high_total_steps=int(high_total_steps),
        t_final=float(t_final),
        chosen_rate=float(chosen_rate),
        cooling_rate_ps=(None if cooling_rate_ps is None else float(cooling_rate_ps)),
        replicate=(int(rep[0]), int(rep[1]), int(rep[2])),
        pressure=float(pressure),
        md_use=dict(md_use),
        potential_config=(None if potential_config is None else dict(potential_config)),
        potential_lines=(None if potential_lines is None else [str(x) for x in potential_lines]),
        core_repulsion=(None if core_repulsion is None else dict(core_repulsion)),
        type_to_species=(None if type_to_species is None else [str(x) for x in type_to_species]),
        metrics_cfg=dict(metrics_cfg),
        effective_metrics=dict(effective_metrics),
        production_cfg=dict(production_cfg),
        convergence_cfg=dict(convergence_cfg),
        cutoffs_rate=_cutoffs_list_from_dict(cutoffs_dict_from_any(cutoffs_rate)),
        cutoffs_size=_cutoffs_list_from_dict(cutoffs_dict_from_any(cutoffs_size)),
        preferred_cutoffs=_cutoffs_list_from_dict(cutoffs_dict_from_any(preferred_cutoffs)),
        quench_steps=int(quench_steps),
        relax_steps=int(relax_steps),
        msd_every=int(msd_every),
        seed_base=int(seed_base),
        time_unit_ps=(None if time_unit_ps is None else float(time_unit_ps)),
        sampling_hint=(None if sampling_hint is None else {str(k): float(v) for k, v in sampling_hint.items() if v is not None}),
        execution_mode=str(execution_mode),
        source_kind=str(source_kind),
    )


def production_plan_to_dict(plan: ProductionPlan, *, relative_to: Optional[Path] = None) -> dict[str, Any]:
    structure_data = Path(plan.structure_data)
    if relative_to is not None:
        try:
            structure_data = structure_data.relative_to(relative_to)
        except Exception:
            pass
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
    structure_data = Path(str(data.get("structure_data", "")))
    if not structure_data.is_absolute() and base_dir is not None:
        structure_data = (Path(base_dir) / structure_data).resolve(strict=False)
    return make_production_plan(
        engine=str(data.get("engine", "lammps")),
        structure_data=structure_data,
        T_high=float(data.get("T_high")),
        high_total_steps=int(data.get("high_total_steps")),
        t_final=float(data.get("t_final")),
        chosen_rate=float(data.get("chosen_rate")),
        cooling_rate_ps=(None if data.get("cooling_rate_ps", None) is None else float(data.get("cooling_rate_ps"))),
        replicate=data.get("replicate", [1, 1, 1]),
        pressure=float(data.get("pressure", 0.0)),
        md_use=data.get("md_use", {}),
        potential_config=data.get("potential_config", None),
        potential_lines=data.get("potential_lines", None),
        core_repulsion=data.get("core_repulsion", None),
        type_to_species=data.get("type_to_species", None),
        metrics_cfg=data.get("metrics_cfg", {}),
        effective_metrics=data.get("effective_metrics", {}),
        production_cfg=data.get("production_cfg", {}),
        convergence_cfg=data.get("convergence_cfg", {}),
        cutoffs_rate=data.get("cutoffs_rate", []),
        cutoffs_size=data.get("cutoffs_size", []),
        preferred_cutoffs=data.get("preferred_cutoffs", []),
        quench_steps=int(data.get("quench_steps")),
        relax_steps=int(data.get("relax_steps")),
        msd_every=int(data.get("msd_every", 100)),
        seed_base=int(data.get("seed_base", 12345 + 13579)),
        time_unit_ps=(None if data.get("time_unit_ps", None) is None else float(data.get("time_unit_ps"))),
        sampling_hint=data.get("sampling_hint", None),
        execution_mode=str(data.get("execution_mode", "adaptive")),
        source_kind=str(data.get("source_kind", "plan")),
    )


def run_production_ensemble(**kwargs: Any) -> dict[str, Any]:
    """TRANSITIONAL SHIM — architectural finding remains OPEN.

    CLAUDE.md states that run/autotune/run-schedule control flow must be
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

    frames_source = analysis_source_path if analysis_source_path is not None else relax_traj_path
    if not Path(frames_source).exists():
        if Path(relax_traj_path).exists():
            LOG.warning(
                "Production analysis source missing for box %s; falling back to relax trajectory %s",
                int(box_id),
                str(relax_traj_path),
            )
            frames_source = relax_traj_path
        elif Path(relax_data_path).exists():
            LOG.warning(
                "Production relax trajectory missing for box %s; falling back to final structure %s",
                int(box_id),
                str(relax_data_path),
            )
            frames_source = relax_data_path
        else:
            raise FileNotFoundError(f"Production analysis source not found: {frames_source}")

    frames = read_last_frames_auto(
        frames_source,
        int(metrics_cfg.time_average_frames),
        type_to_species=type_to_species,
        atom_style=str(atom_style),
    )
    cut_map: dict[Tuple[int, int], float] = dict(cutoffs or {})
    if len(cut_map) == 0 and len(list(required_pairs)) > 0:
        cut_map = estimate_pair_cutoffs(
            frames,
            required_pairs,
            auto=metrics_cfg.auto_cutoff,
            fixed_cutoffs=fixed_cutoffs,
        )

    sm = compute_structure_metrics_timeavg(frames, metrics_cfg, cutoffs=cut_map, type_to_species=type_to_species)
    struct_vals = dict(sm.values)

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
            ),
            "relax": collect_stage_metrics_timeseries(
                stage_dir=relax_stage_dir,
                metrics_cfg=metrics_cfg,
                cutoffs=cut_map,
                md_timestep=float(md_timestep),
                type_to_species=type_to_species,
                outdir=outdir,
                stage_role="relax",
            ),
        }

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
            q, s = compute_sq(
                frames,
                q_max=float(sm_cfg.q_max),
                nq=int(sm_cfg.nq),
                r_max=float(sm_cfg.r_max),
                nbins=int(sm_cfg.nbins),
                pair=sm_cfg.pair,
                type_to_species=type_to_species,
                window=str(getattr(sm_cfg, "window", "lorch")),
            )
            sq_curves[key] = {"label": str(label), "q": [float(v) for v in q.tolist()], "s": [float(v) for v in s.tolist()]}

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
        cutoffs=cut_map,
        type_to_species=type_to_species,
    )
    has_coord_defects = any(
        bool((v or {}).get("has_defect", False)) for v in (coord_defects or {}).values()
    )

    coord_defect_details: dict[str, Any] = {}
    coord_defect_artifacts: dict[str, Any] = {}
    coord_artifact_paths: list[Path] = []
    if metrics_cfg.enabled and len(list(metrics_cfg.coordinations)) > 0 and (exclude_coordination_defects or has_coord_defects):
        try:
            coord_defect_details = compute_coordination_defect_details(
                frames[-1],
                metrics_cfg,
                cutoffs=cut_map,
                type_to_species=type_to_species,
            )

            cd_json = relax_stage_dir / "coordination_defects_detail.json"
            cd_json.write_text(json.dumps(coord_defect_details, indent=2))
            coord_defect_artifacts["detail_json"] = _relpath_or_str(cd_json, outdir)
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
                )
                sp_shell = [sp[int(i)] for i in idxs]
                shell_path = relax_stage_dir / "coordination_defects_shell.extxyz"
                write_extxyz_single_with_species(shell_path, fr_shell, sp_shell, wrap=True)
                coord_defect_artifacts["shell_extxyz"] = _relpath_or_str(shell_path, outdir)
                coord_artifact_paths.append(shell_path)
        except Exception as e:
            coord_defect_artifacts["error"] = str(e)

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
        ar_json.write_text(json.dumps(amorphous_report, indent=2))
        amorphous_artifacts["state_json"] = _relpath_or_str(ar_json, outdir)
        is_amorphous = bool(amorphous_report.get("passed", True))
        coord_artifact_paths.append(ar_json)

    entry = {
        "box": int(box_id),
        "density": float(density_mean),
        "density_stderr": float(density_stderr),
        "metrics": struct_vals,
        "distributions": dist_all,
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
            "coord_defects": coord_defect_artifacts,
            "amorphous": amorphous_artifacts,
        },
        "elastic_melt": melt_elastic,
        "elastic_relax": relax_elastic,
        "elastic_timeseries": elastic_timeseries,
        "stage_metrics": stage_metrics,
    }
    if analysis_source_role is not None:
        entry["analysis_source_role"] = str(analysis_source_role)
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

    return entry, cut_map


def build_production_convergence_spec(entry: Mapping[str, Any]) -> dict[str, Any]:
    struct_vals = dict(entry.get("metrics", {}) or {})
    dist_all = dict(entry.get("distributions", {}) or {})
    bond_names = sorted(list((dist_all.get("bondlen", {}) or {}).keys()))
    angle_names = sorted(list((dist_all.get("angle", {}) or {}).keys()))
    coord_names = sorted(list((dist_all.get("coord", {}) or {}).keys()))
    ring_keys = sorted([k for k in struct_vals.keys() if str(k).startswith("ring_frac_")])

    for nm in bond_names:
        if not ((dist_all.get("bondlen", {}) or {}).get(nm, {}) or {}).get("cdf"):
            raise RuntimeError(f"Bond-length distribution '{nm}' is empty; check selectors and cutoffs")
    for nm in angle_names:
        if not ((dist_all.get("angle", {}) or {}).get(nm, {}) or {}).get("cdf"):
            raise RuntimeError(f"Angle distribution '{nm}' is empty; check selectors")
    for nm in coord_names:
        if not ((dist_all.get("coord", {}) or {}).get(nm, {}) or {}).get("cdf"):
            raise RuntimeError(f"Coordination distribution '{nm}' is empty; check selectors and cutoffs")
    for lab in (dist_all.get("gr", {}) or {}).keys():
        if not ((dist_all.get("gr", {}) or {}).get(lab, {}) or {}).get("g"):
            raise RuntimeError(f"g(r) curve '{lab}' is empty; check g(r) configuration")
    for nm in (dist_all.get("void", {}) or {}).keys():
        if not ((dist_all.get("void", {}) or {}).get(nm, {}) or {}).get("cdf"):
            raise RuntimeError(f"Void distribution '{nm}' is empty; check voids configuration")
    for lab in (dist_all.get("sq", {}) or {}).keys():
        if not ((dist_all.get("sq", {}) or {}).get(lab, {}) or {}).get("s"):
            raise RuntimeError(f"S(q) curve '{lab}' is empty; check sq configuration")

    return {
        "bondlen_names": bond_names,
        "angle_names": angle_names,
        "coord_names": coord_names,
        "ring_keys": ring_keys,
        "ring_has_mean_size": bool("ring_mean_size" in struct_vals),
        "gr_labels": sorted(list((dist_all.get("gr", {}) or {}).keys())),
        "sq_labels": sorted(list((dist_all.get("sq", {}) or {}).keys())),
        "void_names": sorted(list((dist_all.get("void", {}) or {}).keys())),
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


def metrics_checked_from_conv_spec(spec: Optional[Mapping[str, Any]]) -> Optional[list[str]]:
    if spec is None:
        return None
    return [
        "density",
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
        f"No convergence tolerance defined for metric {name!r}; "
        "add a case to _tol_for_metric in autotune.py and production_common.py"
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
        m_tests += int(len(self.ring_keys))
        if bool(self.spec.get("ring_has_mean_size", False)):
            m_tests += 1

        if self.bond_names:
            for nm in self.bond_names:
                m_tests += int(len(self.boxes[0]["distributions"]["bondlen"][nm]["cdf"]))
        if self.angle_names:
            for nm in self.angle_names:
                m_tests += int(len(self.boxes[0]["distributions"]["angle"][nm]["cdf"]))
        if self.coord_names:
            for nm in self.coord_names:
                kmax = 0
                for b in self.boxes:
                    kmax = max(kmax, int(len(b["distributions"]["coord"][nm]["cdf"])))
                m_tests += int(kmax)
        if self.gr_labels:
            for lab in self.gr_labels:
                m_tests += int(len(self.boxes[0]["distributions"]["gr"][lab]["g"]))
        if self.sq_labels:
            for lab in self.sq_labels:
                m_tests += int(len(self.boxes[0]["distributions"]["sq"][lab]["s"]))
        if self.void_names:
            for nm in self.void_names:
                m_tests += int(len(self.boxes[0]["distributions"]["void"][nm]["cdf"]))
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
        self._evaluate_ring_statistics()
        self._evaluate_ring_mean_size()
        self._evaluate_bondlen_cdfs()
        self._evaluate_angle_cdfs()
        self._evaluate_coord_cdfs()
        self._evaluate_gr_curves()
        self._evaluate_sq_curves()
        self._evaluate_void_cdfs()

    def _evaluate_density_scalar(self) -> None:
        dens = np.asarray([float(b["density"]) for b in self.boxes], dtype=float)
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
        }
        self._update_group("density", "density", passed)
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
        }
        self._update_group("ring_mean_size", "ring_mean_size", passed)
        self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_bondlen_cdfs(self) -> None:
        if not self.bond_names:
            return
        rel_tol_b, abs_tol_b = _tol_curve("bondlen_cdf", self.conv_cfg)
        for nm in self.bond_names:
            x_ref = np.asarray(self.boxes[0]["distributions"]["bondlen"][nm]["x"], dtype=float)
            p = int(len(self.boxes[0]["distributions"]["bondlen"][nm]["cdf"]))
            mat = np.zeros((self.n_boxes, p), dtype=float)
            for i, b in enumerate(self.boxes):
                d = b["distributions"]["bondlen"].get(nm, None)
                if d is None:
                    raise RuntimeError(f"Missing bondlen distribution '{nm}' in box {i+1}")
                x = np.asarray(d.get("x", []), dtype=float)
                if x.shape != x_ref.shape or (np.max(np.abs(x - x_ref)) > 1e-10):
                    raise RuntimeError(f"Inconsistent bondlen CDF grid for '{nm}' across boxes")
                mat[i, :] = np.asarray(d.get("cdf", []), dtype=float)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_b), float(rel_tol_b) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = {
                "group": _metric_group("bondlen_cdf"),
                "kind": "bondlen_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_b),
                "abs_tol": float(abs_tol_b),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
            }
            self._update_group("bondlen_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_angle_cdfs(self) -> None:
        if not self.angle_names:
            return
        rel_tol_a, abs_tol_a = _tol_curve("angle_cdf", self.conv_cfg)
        for nm in self.angle_names:
            x_ref = np.asarray(self.boxes[0]["distributions"]["angle"][nm]["x"], dtype=float)
            p = int(len(self.boxes[0]["distributions"]["angle"][nm]["cdf"]))
            mat = np.zeros((self.n_boxes, p), dtype=float)
            for i, b in enumerate(self.boxes):
                d = b["distributions"]["angle"].get(nm, None)
                if d is None:
                    raise RuntimeError(f"Missing angle distribution '{nm}' in box {i+1}")
                x = np.asarray(d.get("x", []), dtype=float)
                if x.shape != x_ref.shape or (np.max(np.abs(x - x_ref)) > 1e-10):
                    raise RuntimeError(f"Inconsistent angle CDF grid for '{nm}' across boxes")
                mat[i, :] = np.asarray(d.get("cdf", []), dtype=float)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_a), float(rel_tol_a) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = {
                "group": _metric_group("angle_cdf"),
                "kind": "angle_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_a),
                "abs_tol": float(abs_tol_a),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
            }
            self._update_group("angle_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_coord_cdfs(self) -> None:
        if not self.coord_names:
            return
        rel_tol_c, abs_tol_c = _tol_curve("coord_cdf", self.conv_cfg)
        for nm in self.coord_names:
            kmax = 0
            for b in self.boxes:
                kmax = max(kmax, int(len(b["distributions"]["coord"][nm]["cdf"])) - 1)
            if kmax < 0:
                raise RuntimeError(f"Empty coordination CDF for '{nm}'")
            x_ref = np.arange(0, kmax + 1, dtype=float)
            p = int(x_ref.size)
            mat = np.zeros((self.n_boxes, p), dtype=float)
            for i, b in enumerate(self.boxes):
                d = b["distributions"]["coord"].get(nm, None)
                if d is None:
                    raise RuntimeError(f"Missing coord distribution '{nm}' in box {i+1}")
                x = np.asarray(d.get("x", []), dtype=float)
                cdf = np.asarray(d.get("cdf", []), dtype=float)
                if x.size != cdf.size:
                    raise RuntimeError(f"Malformed coord CDF for '{nm}' in box {i+1}")
                if cdf.size < p:
                    pad = np.ones((p - cdf.size,), dtype=float)
                    cdf2 = np.concatenate([cdf, pad])
                else:
                    cdf2 = cdf[:p]
                mat[i, :] = cdf2
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_c), float(rel_tol_c) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = {
                "group": _metric_group("coord_cdf"),
                "kind": "coord_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_c),
                "abs_tol": float(abs_tol_c),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
            }
            self._update_group("coord_cdf", nm, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_gr_curves(self) -> None:
        if not self.gr_labels:
            return
        rel_tol_g, abs_tol_g = _tol_curve("gr_curve", self.conv_cfg)
        for lab in self.gr_labels:
            r0 = np.asarray(self.boxes[0]["distributions"]["gr"][lab]["r"], dtype=float)
            nb = int(r0.size)
            if nb < 2:
                raise RuntimeError(f"Invalid g(r) grid for '{lab}'")
            rmax_eff = float("inf")
            for b in self.boxes:
                r = np.asarray(b["distributions"]["gr"][lab]["r"], dtype=float)
                if r.size != nb:
                    raise RuntimeError(f"Inconsistent g(r) length for '{lab}'")
                dr = float(r[1] - r[0])
                rmax_eff = min(rmax_eff, float(r[-1] + 0.5 * dr))
            edges = np.linspace(0.0, float(rmax_eff), nb + 1)
            r_ref = 0.5 * (edges[:-1] + edges[1:])
            mat = np.zeros((self.n_boxes, nb), dtype=float)
            for i, b in enumerate(self.boxes):
                r = np.asarray(b["distributions"]["gr"][lab]["r"], dtype=float)
                g = np.asarray(b["distributions"]["gr"][lab]["g"], dtype=float)
                if not (np.all(np.isfinite(r)) and np.all(np.isfinite(g))):
                    raise RuntimeError(f"Non-finite g(r) data for '{lab}'")
                mat[i, :] = np.interp(r_ref, r, g)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_g), float(rel_tol_g) * np.abs(mu))
            half = float(self.crit) * se_vec
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if nb > 0 else None
            self.report["distributions"][lab] = {
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
            }
            self._update_group("gr_curve", lab, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_sq_curves(self) -> None:
        if not self.sq_labels:
            return
        rel_tol_s, abs_tol_s = _tol_curve("sq_curve", self.conv_cfg)
        for lab in self.sq_labels:
            q0 = np.asarray(self.boxes[0]["distributions"]["sq"][lab]["q"], dtype=float)
            nb = int(q0.size)
            if nb < 2:
                raise RuntimeError(f"Invalid S(q) grid for '{lab}'")
            qmax_eff = float("inf")
            for b in self.boxes:
                qv = np.asarray(b["distributions"]["sq"][lab]["q"], dtype=float)
                if qv.size != nb:
                    raise RuntimeError(f"Inconsistent S(q) length for '{lab}'")
                qmax_eff = min(qmax_eff, float(qv[-1]))
            q_ref = np.linspace(0.0, float(qmax_eff), nb)
            mat = np.zeros((self.n_boxes, nb), dtype=float)
            for i, b in enumerate(self.boxes):
                qv = np.asarray(b["distributions"]["sq"][lab]["q"], dtype=float)
                sv = np.asarray(b["distributions"]["sq"][lab]["s"], dtype=float)
                if not (np.all(np.isfinite(qv)) and np.all(np.isfinite(sv))):
                    raise RuntimeError(f"Non-finite S(q) data for '{lab}'")
                mat[i, :] = np.interp(q_ref, qv, sv)
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_s), float(rel_tol_s) * np.abs(mu))
            half = float(self.crit) * se_vec
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if nb > 0 else None
            self.report["distributions"][lab] = {
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
            }
            self._update_group("sq_curve", lab, passed)
            self.ok_ci = self.ok_ci and bool(passed)

    def _evaluate_void_cdfs(self) -> None:
        if not self.void_names:
            return
        rel_tol_v, abs_tol_v = _tol_curve("void_cdf", self.conv_cfg)
        for nm in self.void_names:
            x_ref = np.asarray(self.boxes[0]["distributions"]["void"][nm].get("x", []), dtype=float)
            p = int(x_ref.size)
            if p < 2:
                raise RuntimeError(f"Invalid void CDF grid for '{nm}'")
            mat = np.zeros((self.n_boxes, p), dtype=float)
            for i, b in enumerate(self.boxes):
                d = b["distributions"]["void"][nm]
                x = np.asarray(d.get("x", []), dtype=float)
                cdf = np.asarray(d.get("cdf", []), dtype=float)
                if x.size != cdf.size:
                    raise RuntimeError(f"Malformed void CDF for '{nm}' in box {i+1}")
                if x.size != p:
                    raise RuntimeError(f"Inconsistent void CDF length for '{nm}'")
                mat[i, :] = cdf
            mu, sd_vec, se_vec, n = _vector_stats(mat)
            tol_arr = np.maximum(float(abs_tol_v), float(rel_tol_v) * np.abs(mu))
            half = self._halfwidth_bounded(sd_vec, n)
            passed = bool(n >= 2 and np.all(half <= tol_arr))
            worst = int(np.argmax(half - tol_arr)) if p > 0 else None
            self.report["distributions"][nm] = {
                "group": _metric_group("void_cdf"),
                "kind": "void_cdf",
                "x": [float(v) for v in x_ref.tolist()],
                "mean": [float(v) for v in mu.tolist()],
                "stderr": [float(v) for v in se_vec.tolist()],
                "ci_halfwidth": [float(v) for v in half.tolist()],
                "rel_tol": float(rel_tol_v),
                "abs_tol": float(abs_tol_v),
                "tol": [float(v) for v in tol_arr.tolist()],
                "passed": bool(passed),
                "worst_index": int(worst) if worst is not None else None,
                "worst_x": float(x_ref[int(worst)]) if worst is not None else None,
            }
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
            tol = float(max(abs_tol, rel_tol * abs(ref)))
            d = float(_dist_scalar(x, y))
            upper = float(self._bootstrap_upper(_dist_scalar, x, y, n_boot, q))
            passed = bool(math.isfinite(tol) and math.isfinite(upper) and (upper <= tol))
            stab["checks"][name] = {
                "kind": "scalar",
                "distance": float(d),
                "upper": float(upper),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": float(tol),
                "passed": bool(passed),
            }
            self.ok_stab = self.ok_stab and bool(passed)

        def _record_curve_check(name: str, mat1: np.ndarray, mat2: np.ndarray, rel_tol: float, abs_tol: float, grid: dict[str, Any]) -> None:
            ref = float(np.nanmax(np.abs(np.nanmean(np.vstack([mat1, mat2]), axis=0))))
            tol = float(max(abs_tol, rel_tol * abs(ref)))
            d = float(_curve_distance(mat1, mat2))
            upper = float(_bootstrap_curve_upper(mat1, mat2))
            passed = bool(math.isfinite(tol) and math.isfinite(upper) and (upper <= tol))
            stab["checks"][name] = {
                "kind": "curve",
                "distance": float(d),
                "upper": float(upper),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": float(tol),
                "passed": bool(passed),
                "grid": dict(grid),
            }
            self.ok_stab = self.ok_stab and bool(passed)

        x = np.asarray([float(b["density"]) for b in g1], dtype=float)
        y = np.asarray([float(b["density"]) for b in g2], dtype=float)
        rel_tol, abs_tol = _tol_scalar("density", self.conv_cfg)
        _record_scalar_check("density", x, y, rel_tol, abs_tol)

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
                x_ref = np.asarray(self.boxes[0]["distributions"]["bondlen"][nm]["x"], dtype=float)
                p = int(len(self.boxes[0]["distributions"]["bondlen"][nm]["cdf"]))
                m1 = np.zeros((len(g1), p), dtype=float)
                m2 = np.zeros((len(g2), p), dtype=float)
                for i, b in enumerate(g1):
                    dct = b["distributions"]["bondlen"][nm]
                    m1[i, :] = np.asarray(dct["cdf"], dtype=float)
                for i, b in enumerate(g2):
                    dct = b["distributions"]["bondlen"][nm]
                    m2[i, :] = np.asarray(dct["cdf"], dtype=float)
                _record_curve_check(
                    f"bondlen_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_b,
                    abs_tol_b,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(p)},
                )

        if self.angle_names:
            rel_tol_a, abs_tol_a = _tol_curve("angle_cdf", self.conv_cfg)
            for nm in self.angle_names:
                x_ref = np.asarray(self.boxes[0]["distributions"]["angle"][nm]["x"], dtype=float)
                p = int(len(self.boxes[0]["distributions"]["angle"][nm]["cdf"]))
                m1 = np.zeros((len(g1), p), dtype=float)
                m2 = np.zeros((len(g2), p), dtype=float)
                for i, b in enumerate(g1):
                    m1[i, :] = np.asarray(b["distributions"]["angle"][nm]["cdf"], dtype=float)
                for i, b in enumerate(g2):
                    m2[i, :] = np.asarray(b["distributions"]["angle"][nm]["cdf"], dtype=float)
                _record_curve_check(
                    f"angle_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_a,
                    abs_tol_a,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(p)},
                )

        if self.coord_names:
            rel_tol_c, abs_tol_c = _tol_curve("coord_cdf", self.conv_cfg)
            for nm in self.coord_names:
                kmax = 0
                for b in self.boxes:
                    kmax = max(kmax, int(len(b["distributions"]["coord"][nm]["cdf"])) - 1)
                p = int(kmax + 1)
                m1 = np.zeros((len(g1), p), dtype=float)
                m2 = np.zeros((len(g2), p), dtype=float)
                for i, b in enumerate(g1):
                    cdf = np.asarray(b["distributions"]["coord"][nm]["cdf"], dtype=float)
                    if cdf.size < p:
                        cdf = np.concatenate([cdf, np.ones((p - cdf.size,), dtype=float)])
                    m1[i, :] = cdf[:p]
                for i, b in enumerate(g2):
                    cdf = np.asarray(b["distributions"]["coord"][nm]["cdf"], dtype=float)
                    if cdf.size < p:
                        cdf = np.concatenate([cdf, np.ones((p - cdf.size,), dtype=float)])
                    m2[i, :] = cdf[:p]
                _record_curve_check(f"coord_cdf:{nm}", m1, m2, rel_tol_c, abs_tol_c, {"p": int(p)})

        if self.gr_labels:
            rel_tol_g, abs_tol_g = _tol_curve("gr_curve", self.conv_cfg)
            for lab in self.gr_labels:
                r0 = np.asarray(self.boxes[0]["distributions"]["gr"][lab]["r"], dtype=float)
                nbins = int(r0.size)
                m1 = np.zeros((len(g1), nbins), dtype=float)
                m2 = np.zeros((len(g2), nbins), dtype=float)
                for i, b in enumerate(g1):
                    m1[i, :] = np.asarray(b["distributions"]["gr"][lab]["g"], dtype=float)
                for i, b in enumerate(g2):
                    m2[i, :] = np.asarray(b["distributions"]["gr"][lab]["g"], dtype=float)
                _record_curve_check(f"gr_curve:{lab}", m1, m2, rel_tol_g, abs_tol_g, {"p": int(nbins)})

        if self.sq_labels:
            rel_tol_s, abs_tol_s = _tol_curve("sq_curve", self.conv_cfg)
            for lab in self.sq_labels:
                q0 = np.asarray(self.boxes[0]["distributions"]["sq"][lab]["q"], dtype=float)
                nbins = int(q0.size)
                m1 = np.zeros((len(g1), nbins), dtype=float)
                m2 = np.zeros((len(g2), nbins), dtype=float)
                for i, b in enumerate(g1):
                    m1[i, :] = np.asarray(b["distributions"]["sq"][lab]["s"], dtype=float)
                for i, b in enumerate(g2):
                    m2[i, :] = np.asarray(b["distributions"]["sq"][lab]["s"], dtype=float)
                _record_curve_check(f"sq_curve:{lab}", m1, m2, rel_tol_s, abs_tol_s, {"p": int(nbins)})

        if self.void_names:
            rel_tol_v, abs_tol_v = _tol_curve("void_cdf", self.conv_cfg)
            for nm in self.void_names:
                x_ref = np.asarray(self.boxes[0]["distributions"]["void"][nm].get("x", []), dtype=float)
                p = int(x_ref.size)
                m1 = np.zeros((len(g1), p), dtype=float)
                m2 = np.zeros((len(g2), p), dtype=float)
                for i, b in enumerate(g1):
                    m1[i, :] = np.asarray(b["distributions"]["void"][nm]["cdf"], dtype=float)
                for i, b in enumerate(g2):
                    m2[i, :] = np.asarray(b["distributions"]["void"][nm]["cdf"], dtype=float)
                _record_curve_check(
                    f"void_cdf:{nm}",
                    m1,
                    m2,
                    rel_tol_v,
                    abs_tol_v,
                    {"x0": float(x_ref[0]) if x_ref.size else None, "x1": float(x_ref[-1]) if x_ref.size else None, "p": int(p)},
                )

        self.report["stability"] = stab


def check_production_convergence(boxes: list[dict[str, Any]], spec: dict[str, Any], conv_cfg) -> tuple[bool, dict[str, Any]]:
    """Production convergence."""
    return _ProductionConvergenceChecker(boxes, spec, conv_cfg).run()
