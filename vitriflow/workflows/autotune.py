from __future__ import annotations

import json
import math
import random
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import RunConfig, MDConfig, ThermostatConfig, BarostatConfig, KimConfig
from ..kim import ensure_model_installed
from ..lammps_input import StageSpec
from ..runner import Cp2kRunner, LammpsRunner
from ..utils import ensure_dir, scale_steps_for_timestep
from ..parse import parse_all_thermo_tables, parse_last_thermo_table, parse_msd_file
from ..io.thermo import parse_thermo_csv
from ..io.extxyz import write_extxyz_single_with_species
from ..analysis.tm import estimate_tm, estimate_tm_from_diffusion
from ..analysis.stats import early_late_change

from ..analysis.datafile import count_atoms_in_datafile
from ..analysis.amorphous import analyse_amorphous_state, summarize_rate_amorphous_acceptance
from ..analysis.motif_summary import summarize_production_crystal_motifs
from ..analysis.convergence import allowed_delta, choose_fastest_converged
from ..analysis.trajectory import read_last_frames_auto, quench_window_steps
from ..analysis.gr import compute_first_peak_gr, compute_gr
from ..analysis.structure import (
    compute_structure_metrics,
    compute_structure_metrics_timeavg,
    compute_structure_distributions_timeavg,
    compute_coordination_defects,
    compute_coordination_defect_details,
    estimate_pair_cutoffs,
    fixed_cutoffs_from_metrics,
    required_pairs_from_metrics,
)

from ..cp2k_driver import density_g_cm3_from_atoms, render_cp2k_cell_opt_input

from ..structuregen import prepare_initial_structure, prepare_size_scan_base_structure
from .elastic_screen import (
    build_elastic_sampling_hint,
    estimate_diffusion_freeze_temperature,
    run_elastic_screen_lammps,
    run_elastic_screen_timeseries_lammps,
    should_collect_elastic_stage_timeseries,
    should_run_elastic_screen,
)
from .stage_metrics import collect_stage_metrics_timeseries, should_collect_stage_metrics_timeseries
from .metrics_policy import resolve_effective_metrics_config
from .production_common import (
    analyse_production_box,
    build_production_convergence_spec,
    check_production_convergence,
    make_production_plan,
    metrics_checked_from_conv_spec,
    plan_production_stage_diagnostics,
    production_plan_to_dict,
    resolve_production_relax_dump_settings,
    resolve_production_time_unit_ps,
    resolve_production_warmup_duration_ps,
    resolve_production_warmup_start_temperature,
    resolve_production_warmup_steps,
    validate_production_entry_against_spec,
)
from .progress import CondensedProgressLog, write_autotune_outputs
from .preflight import run_preflight
from .quench_rates import quench_steps_for_rate, resolve_quench_rates_K_per_time
from .stage_runner import StageOutcome, run_stage_local, run_stages_continuous_lammps, stage_outcome_from_artifacts
from .step_counts import extend_highT_steps_for_force_isotropic, resolve_lammps_units_style, resolve_md_pressure


def _production_resume_seed_draw_count(entry: Any, *, default_draws: int) -> int:
    """Production resume seed."""

    try:
        fallback = max(1, int(default_draws))
    except Exception:
        fallback = 1

    count = 0
    if isinstance(entry, Mapping):
        count += sum(1 for k, v in entry.items() if str(k).startswith("seed_") and v is not None)
        nested = entry.get("seeds", None)
        if isinstance(nested, Mapping):
            nested_count = sum(1 for _k, v in nested.items() if v is not None)
            count = max(count, nested_count)

    return int(count) if count > 0 else int(fallback)


def _count_production_resume_seed_draws(entries: Sequence[Any], *, default_draws: int) -> int:
    total = 0
    for ent in entries:
        total += _production_resume_seed_draw_count(ent, default_draws=default_draws)
    return int(total)


def _format_rate_amorphous_criteria_summary(summary: Mapping[str, Any]) -> str:
    crit = dict(summary.get("criteria_summary", {}) or {})
    if not crit:
        return ""
    parts: list[str] = []
    for name in sorted(crit.keys()):
        payload = dict(crit.get(name, {}) or {})
        n_failed = int(payload.get("n_failed", 0) or 0)
        n_eval = int(payload.get("n_evaluated", 0) or 0)
        if n_eval <= 0 or n_failed <= 0:
            continue
        thr = payload.get("threshold", None)
        mean = payload.get("mean", None)
        vmax = payload.get("max", None)
        seg = f"{name}: failed={n_failed}/{n_eval}"
        if mean is not None:
            try:
                seg += f", mean={float(mean):.4g}"
            except Exception:
                pass
        if vmax is not None:
            try:
                seg += f", max={float(vmax):.4g}"
            except Exception:
                pass
        if thr is not None:
            try:
                seg += f", thr={float(thr):.4g}"
            except Exception:
                pass
        parts.append(seg)
    return "; ".join(parts)


def _write_rate_scan_failure_snapshot(
    *,
    outdir: Path,
    config: RunConfig,
    pot_cfg: Any,
    kim_install: Any,
    preflight: Any,
    T: Sequence[float],
    D: Sequence[float],
    D_mu: Sequence[float],
    D_se: Sequence[float],
    D_med: Sequence[float],
    tm_cfg: Any,
    tm_summary: Any,
    tm_outcomes_all: Sequence[Any],
    tm_est: Any,
    time_unit_ps: Optional[float],
    T_high: float,
    high_total_steps: int,
    force_iso_active: bool,
    high_cfg: Any,
    high_stationarity_summary: Any,
    high_rep_summaries: Sequence[Any],
    high_outcomes: Sequence[Any],
    melt_pool: Sequence[Any],
    melt_data: Any,
    rate_results: Sequence[Mapping[str, Any]],
    cutoffs_rate: Mapping[tuple[int, int], float],
    metric_warnings: Sequence[str],
    metrics_summary: Mapping[str, Any],
    failure_message: str,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    results = {
        "status": "failed",
        "failure": {
            "stage": "rate_scan",
            "reason": str(failure_message),
            "kind": "amorphous_gate",
        },
        "units": {
            "engine": str(getattr(config, "engine", "lammps")),
            "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
            "time_unit_ps": float(time_unit_ps) if time_unit_ps is not None else None,
        },
        "kim_install": asdict(kim_install) if kim_install is not None else None,
        "preflight": asdict(preflight),
        "tm_scan": {
            "temps": [float(t) for t in T],
            "replicates_per_temp": int(getattr(tm_cfg, 'replicates_per_temp', 1)),
            "D": [float(x) for x in D],
            "D_mean": [float(x) for x in D_mu],
            "D_stderr": [float(x) for x in D_se],
            "D_median": [float(x) for x in D_med],
            "summary": tm_summary,
            "outcomes": [asdict(o) for o in tm_outcomes_all],
            "Tm_estimate": {
                "Tm": float(tm_est.Tm),
                "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                "D_liquid_target": float(getattr(tm_est, "D_liquid_target", float("nan"))),
                "method": str(tm_est.method),
                "score": float(tm_est.score),
                "idx": int(tm_est.idx),
            },
        },
        "highT": {
            "T_high": float(T_high),
            "total_steps": int(high_total_steps),
            "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
            "replicates": int(getattr(high_cfg, 'replicates', 1)),
            "stationarity": high_stationarity_summary,
            "rep_summaries": list(high_rep_summaries),
            "outcomes": [asdict(o) for o in high_outcomes],
            "melt_pool": [
                str(Path(p).relative_to(outdir)) if Path(p).is_relative_to(outdir) else str(p)
                for p in melt_pool
            ],
            "melt_data": str(Path(melt_data).relative_to(outdir)) if Path(melt_data).is_relative_to(outdir) else str(melt_data),
        },
        "rate_scan": {
            "rates": [dict(rr) for rr in rate_results],
            "decision_density": None,
            "decision_multi": None,
            "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_rate.items())],
            "accepted_rates": [float(rr.get("rate", float("nan"))) for rr in rate_results if bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))],
            "rejected_rates": [float(rr.get("rate", float("nan"))) for rr in rate_results if not bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))],
        },
        "size_scan": {
            "skipped": True,
            "skip_reason": "rate_scan_failed",
            "base_data": None,
            "base_natoms": None,
            "initial_repeat": None,
            "sizes": [],
            "decision_density": None,
            "decision_multi": None,
            "cutoffs": [],
        },
        "production": {
            "enabled": bool(getattr(getattr(config.autotune, "production", None), "enabled", False)),
            "status": "not_started",
            "converged": False,
            "n_boxes": 0,
            "n_boxes_total": 0,
            "convergence": None,
            "rejected_boxes": [],
        },
        "production_plan": None,
        "recommendation": None,
        "metric_warnings": list(metric_warnings),
        "effective_metrics": dict(metrics_summary),
        "paths": {
            "autotune_results": "autotune_results.json",
            "autotune": "autotune.json",
            "condensed_log": "condensed.log",
        },
    }
    write_autotune_outputs(outdir, results)
    if progress is not None:
        progress.error("rate_scan", f"{failure_message} Diagnostics written to autotune_results.json")
    return results


def _get_type_to_species(config: RunConfig) -> Optional[list[str]]:
    m = config.autotune.metrics
    if m.type_to_species is not None:
        return list(m.type_to_species)
    if config.kim is not None and config.kim.interactions != "fixed_types":
        return list(config.kim.interactions)
    if getattr(config, "engine", "lammps") == "cp2k":
        raise ValueError("engine='cp2k' requires autotune.metrics.type_to_species")
    return None


def _aggregate_scalar_metrics(reps: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate scalar metrics."""

    keys: set[str] = set()
    for r in reps:
        keys.update(r.keys())

    mu: dict[str, float] = {}
    se: dict[str, float] = {}

    for k in sorted(keys):
        arr = np.asarray([float(r.get(k, float("nan"))) for r in reps], dtype=float)
        m = np.isfinite(arr)
        if int(np.sum(m)) == 0:
            mu[k] = float("nan")
            se[k] = float("nan")
            continue
        vals = arr[m]
        mu[k] = float(np.mean(vals))
        if vals.size > 1:
            se[k] = float(np.std(vals, ddof=1) / math.sqrt(vals.size))
        else:
            se[k] = 0.0
    return mu, se


def _resolve_replicate_traj_path(*, outdir: Path, rep_entry: Mapping[str, Any]) -> Path:
    p = Path(outdir) / Path(rep_entry["dump"])
    cand = Path(p).parent / "traj.extxyz"
    return cand if cand.exists() else p


def _collect_scan_tail_frames(
    scan_results: Sequence[Mapping[str, Any]],
    *,
    outdir: Path,
    metrics_cfg,
    type_to_species: Optional[Sequence[str]],
) -> dict[str, list[Any]]:
    """Scan tail frames."""

    n_frames = max(1, int(getattr(metrics_cfg, "time_average_frames", 1) or 1))
    frames_by_path: dict[str, list[Any]] = {}
    for row in list(scan_results or []):
        if not isinstance(row, Mapping):
            continue
        for rep_entry in list(row.get("replicates", []) or []):
            if not isinstance(rep_entry, Mapping) or "dump" not in rep_entry:
                continue
            traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
            key = str(traj_path)
            if key in frames_by_path:
                continue
            frames_by_path[key] = read_last_frames_auto(
                traj_path,
                n_frames,
                type_to_species=type_to_species,
            )
    return frames_by_path


def _collect_rate_scan_cutoff_reference_frames(
    *,
    rate_results: Sequence[Mapping[str, Any]],
    outdir: Path,
    metrics_cfg,
    type_to_species: Optional[Sequence[str]],
) -> list[Any]:
    """Rate scan cutoff."""

    frames_by_path = _collect_scan_tail_frames(
        rate_results,
        outdir=outdir,
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
    )
    out: list[Any] = []
    for frames in frames_by_path.values():
        out.extend(list(frames or []))
    return out


def _estimate_pooled_scan_cutoffs(
    scan_results: Sequence[Mapping[str, Any]],
    *,
    outdir: Path,
    metrics_cfg,
    required_pairs: Sequence[Tuple[int, int]],
    fixed_cutoffs: Mapping[Tuple[int, int], float],
    type_to_species: Optional[Sequence[str]],
) -> tuple[dict[tuple[int, int], float], dict[str, list[Any]]]:
    """Pooled scan cutoffs."""

    frames_by_path = _collect_scan_tail_frames(
        scan_results,
        outdir=outdir,
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
    )
    pooled_frames: list[Any] = []
    for frames in frames_by_path.values():
        pooled_frames.extend(list(frames or []))

    missing_pairs = []
    for pair in list(required_pairs or []):
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        key = (a, b) if a <= b else (b, a)
        if key not in fixed_cutoffs:
            missing_pairs.append(key)
    if len(missing_pairs) > 0 and len(pooled_frames) == 0:
        raise ValueError("No trajectory frames available to estimate pooled scan cutoffs.")

    cutoffs = estimate_pair_cutoffs(
        pooled_frames,
        required_pairs,
        auto=metrics_cfg.auto_cutoff,
        fixed_cutoffs=fixed_cutoffs,
    )
    return cutoffs, frames_by_path


def _cutoffs_any_to_dict(obj: Any) -> dict[tuple[int, int], float]:
    """Cutoffs any to."""

    if obj is None:
        return {}

    if isinstance(obj, dict):
        out: dict[tuple[int, int], float] = {}
        for k, v in obj.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                a, b = int(k[0]), int(k[1])
            else:
                s = str(k).strip().lstrip("(").rstrip(")")
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) != 2:
                    continue
                a, b = int(parts[0]), int(parts[1])
            out[(min(a, b), max(a, b))] = float(v)
        return out

    if isinstance(obj, list):
        out: dict[tuple[int, int], float] = {}
        for ent in obj:
            if not isinstance(ent, dict):
                continue
            pair = ent.get("pair", None)
            cutoff = ent.get("cutoff", None)
            if pair is None or cutoff is None:
                continue
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = int(pair[0]), int(pair[1])
                out[(min(a, b), max(a, b))] = float(cutoff)
        return out

    return {}


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
    # absolute tolerance uncertainty
    if name.startswith("gr_") and name.endswith("_peak_r"):
        return conv.gr_peak_r_rel_tol, conv.gr_peak_r_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_height"):
        return conv.gr_peak_height_rel_tol, conv.gr_peak_height_abs_tol
    if name.startswith("gr_") and name.endswith("_peak_fwhm"):
        return conv.gr_peak_fwhm_rel_tol, conv.gr_peak_fwhm_abs_tol
    return 0.0, 0.0

def _multimetric_decision(
    x: list[float],
    mu_maps: list[dict[str, float]],
    se_maps: list[dict[str, float]],
    *,
    conv,
    kind: str,
) -> dict[str, Any]:
    """Multimetric decision."""

    if len(x) < 2:
        raise ValueError("Need >= 2 points for convergence decision")
    if not (len(x) == len(mu_maps) == len(se_maps)):
        raise ValueError("x, mu_maps, se_maps must have same length")

    ref = len(x) - 1
    # metrics reference finite
    metrics = sorted({k for k, v in mu_maps[ref].items() if np.isfinite(float(v))})

    per_metric: dict[str, dict[str, list[float] | list[bool]]] = {}
    for m in metrics:
        per_metric[m] = {"deltas": [], "allowed": [], "passed": []}

    combined_pass: list[bool] = []
    chosen: Optional[int] = None

    for i in range(len(x)):
        ok_all = True
        for m in metrics:
            mu_i = float(mu_maps[i].get(m, float("nan")))
            se_i = float(se_maps[i].get(m, float("nan")))
            mu_r = float(mu_maps[ref].get(m, float("nan")))
            se_r = float(se_maps[ref].get(m, float("nan")))

            if not (np.isfinite(mu_i) and np.isfinite(se_i) and np.isfinite(mu_r) and np.isfinite(se_r)):
                d = float("nan")
                a = float("nan")
                ok = False
            else:
                rel_tol, abs_tol = _tol_for_metric(m, conv)
                d = abs(mu_i - mu_r)
                a = allowed_delta(mu_r, se_i, se_r, rel_tol, abs_tol, float(conv.zscore))
                ok = bool(d <= a)

            per_metric[m]["deltas"].append(float(d))
            per_metric[m]["allowed"].append(float(a))
            per_metric[m]["passed"].append(bool(ok))
            if not ok:
                ok_all = False

        combined_pass.append(bool(ok_all))
        if chosen is None and ok_all:
            chosen = i

    if chosen is None:
        chosen = ref

    return {
        "kind": kind,
        "chosen_index": int(chosen),
        "chosen_value": float(x[chosen]),
        "reference_index": int(ref),
        "metrics": per_metric,
        "combined_passed": combined_pass,
    }


def _stage_run(
    runner: Union[LammpsRunner, Cp2kRunner],
    pot_cfg,
    md_cfg,
    stage: StageSpec,
    stage_dir: Path,
    *,
    potential_lines: Optional[list[str]] = None,
    type_to_species: Optional[list[str]] = None,
) -> StageOutcome:
    """Stage run."""

    art = run_stage_local(
        runner,
        pot_cfg,
        md_cfg,
        stage,
        stage_dir,
        potential_lines=potential_lines,
        log_name="log.lammps",
        type_to_species=type_to_species,
    )
    return stage_outcome_from_artifacts(art, md_cfg=md_cfg, stage=stage)






class _ProductionEnsembleRunner:
    """Production ensemble runner."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> dict[str, Any]:
        config = self.config
        outdir = self.outdir
        runner = self.runner
        pot_cfg = self.pot_cfg
        md_use = self.md_use
        potential_lines = self.potential_lines
        type_to_species = self.type_to_species
        metrics_cfg = self.metrics_cfg
        tm_cfg = self.tm_cfg
        q_cfg = self.q_cfg
        size_base_data = self.size_base_data
        chosen_replicate = self.chosen_replicate
        chosen_rate = self.chosen_rate
        dt_ref = self.dt_ref
        dt_mq = self.dt_mq
        cooling_rate_ps = self.cooling_rate_ps
        cutoffs_rate = self.cutoffs_rate
        cutoffs_size = self.cutoffs_size
        T_high = self.T_high
        high_total_steps = self.high_total_steps
        resume_state = self.resume_state
        sampling_hint = self.sampling_hint
        progress = self.progress
        checkpoint_cb = self.checkpoint_cb
        pressure_override = self.pressure_override
        seed_base = self.seed_base
        prod_cfg_override = self.prod_cfg_override
        conv_cfg_override = self.conv_cfg_override
        quench_steps_override = self.quench_steps_override
        relax_steps_override = self.relax_steps_override
        time_unit_ps_override = getattr(self, "time_unit_ps_override", None)
        if progress is None:
            progress = CondensedProgressLog(outdir / "condensed.log")
        progress.info("production", "initialising production ensemble")
        # production ensemble
        production: dict[str, Any] = {"enabled": False}
        prod_cfg = (prod_cfg_override if prod_cfg_override is not None else getattr(config.autotune, "production", None))
        if prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False)):
            warmup_start_temperature = resolve_production_warmup_start_temperature(
                prod_cfg=prod_cfg,
                T_high=float(T_high),
            )
            warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
            warmup_time_unit_ps = resolve_production_time_unit_ps(
                config=config,
                engine=str(getattr(config, "engine", "lammps") or "lammps"),
                pot_cfg=pot_cfg,
                time_unit_ps=time_unit_ps_override,
            )
            warmup_steps = resolve_production_warmup_steps(
                prod_cfg=prod_cfg,
                md_timestep=float(dt_mq),
                time_unit_ps=warmup_time_unit_ps,
            )
            prod_dir = outdir / "production"
            ensure_dir(prod_dir)

            # selected rate size
            prod_rate = float(chosen_rate)  # k lammps time
            nx, ny, nz = (int(chosen_replicate[0]), int(chosen_replicate[1]), int(chosen_replicate[2]))
            if nx < 1 or ny < 1 or nz < 1:
                raise ValueError(f"Invalid production replicate factors: {chosen_replicate}")

            # quench selected rate
            # ceil high rate
            dT = float(T_high - q_cfg.t_final)
            n_quench_prod = (
                int(quench_steps_override) if quench_steps_override is not None else quench_steps_for_rate(float(dT), float(prod_rate), float(dt_mq), min_steps=1)
            )

            # relax temperature timestep
            relax_steps_prod = (
                int(relax_steps_override) if relax_steps_override is not None else scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
            )

            # resuming existing production
            # guarantee consistency yaml
            if resume_state is not None:
                rs_q = resume_state.get("quench_steps", None)
                rs_r = resume_state.get("relax_steps", None)
                if isinstance(rs_q, int) and rs_q > 0:
                    n_quench_prod = int(rs_q)
                if isinstance(rs_r, int) and rs_r > 0:
                    relax_steps_prod = int(rs_r)

            # seed
            rng_prod = random.Random(int(seed_base) if seed_base is not None else (int(config.random_seed) + 13579))

            # production warmup initializes
            # downstream preserve recreate
            cont = str(getattr(md_use, "stage_continuity", "discontinuous")).strip().lower()
            vel_next = "preserve" if cont == "continuous" else "create"
            melt_force_iso = bool(getattr(md_use, "force_isotropic", False))

            md_pressure = float(resolve_md_pressure(config, md_use=md_use, override=pressure_override, default=0.0))

            def _maybe_elastic(
                stage_role: str,
                *,
                stage_dir: Path,
                structure_data: Path,
                input_data: Path,
                force_iso_context: bool,
            ) -> Optional[dict[str, Any]]:
                run_screen, strict, _cfg = should_run_elastic_screen(
                    metrics_cfg,
                    runner=runner,
                    stage_role=stage_role,
                    force_isotropic=bool(force_iso_context),
                )
                if not run_screen:
                    return None
                try:
                    return run_elastic_screen_lammps(
                        runner,
                        pot_cfg,
                        md_use,
                        structure_data=structure_data,
                        stage_dir=stage_dir,
                        potential_lines=potential_lines,
                        metrics_cfg=metrics_cfg,
                        force_isotropic=bool(force_iso_context),
                        input_data_for_affine_strain=input_data if bool(force_iso_context) else None,
                        outdir=outdir,
                    )
                except Exception:
                    if strict:
                        raise
                    return None

            def _maybe_elastic_series(
                stage_role: str,
                *,
                stage_dir: Path,
                stage_output_data: Path,
                force_iso_context: bool,
                sampling_hint: Optional[dict[str, float]] = None,
            ) -> Optional[dict[str, Any]]:
                run_series, strict, _cfg = should_collect_elastic_stage_timeseries(
                    metrics_cfg,
                    runner=runner,
                    stage_role=stage_role,
                    force_isotropic=bool(force_iso_context),
                )
                if not run_series:
                    return None
                try:
                    return run_elastic_screen_timeseries_lammps(
                        runner,
                        pot_cfg,
                        md_use,
                        stage_dir=stage_dir,
                        stage_output_data=stage_output_data,
                        stage_role=stage_role,
                        potential_lines=potential_lines,
                        metrics_cfg=metrics_cfg,
                        force_isotropic=bool(force_iso_context),
                        outdir=outdir,
                        sampling_hint=sampling_hint,
                    )
                except Exception:
                    if strict:
                        raise
                    return None

            # structural metrics convergence
            if not bool(metrics_cfg.enabled):
                raise RuntimeError(
                    "Production ensemble generation requires autotune.metrics.enabled=true "
                    "(needed for bond/angle/coordination/ring/g(r) convergence)."
                )

            # scan cutoffs rate
            # possibly downstream analysis
            prod_cutoffs: dict[tuple[int, int], float] = {}
            if isinstance(cutoffs_size, dict) and len(cutoffs_size) > 0:
                prod_cutoffs = dict(cutoffs_size)
            elif isinstance(cutoffs_rate, dict) and len(cutoffs_rate) > 0:
                prod_cutoffs = dict(cutoffs_rate)

            # cutoffs production analysis
            if resume_state is not None:
                rs_cut = _cutoffs_any_to_dict(resume_state.get("cutoffs", None))
                if len(rs_cut) > 0:
                    prod_cutoffs = dict(rs_cut)

            required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
            fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

            # convergence helpers distribution
            conv_cfg = (conv_cfg_override if conv_cfg_override is not None else config.autotune.convergence)

            def _alpha_from_z(z: float) -> float:
                # sided alpha normal
                p = float(NormalDist().cdf(abs(float(z))))
                a = 2.0 * max(0.0, 1.0 - float(p))
                return float(min(1.0, max(0.0, a)))

            def _critical_value(n: int, alpha: float) -> tuple[float, str]:
                """Critical value."""
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
                # fallback
                crit = float(NormalDist().inv_cdf(1.0 - a / 2.0))
                return crit, "z"

            def _tol_scalar(name: str) -> tuple[float, float]:
                if name == "density":
                    return float(conv_cfg.density_rel_tol), float(conv_cfg.density_abs_tol)
                return _tol_for_metric(name, conv_cfg)

            def _tol_curve(kind: str) -> tuple[float, float]:
                if kind == "bondlen_cdf":
                    return float(conv_cfg.bondlen_cdf_rel_tol), float(conv_cfg.bondlen_cdf_abs_tol)
                if kind == "angle_cdf":
                    return float(conv_cfg.angle_cdf_rel_tol), float(conv_cfg.angle_cdf_abs_tol)
                if kind == "coord_cdf":
                    return float(conv_cfg.coord_cdf_rel_tol), float(conv_cfg.coord_cdf_abs_tol)
                if kind == "gr_curve":
                    return float(conv_cfg.gr_curve_rel_tol), float(conv_cfg.gr_curve_abs_tol)
                if kind == "sq_curve":
                    return float(conv_cfg.sq_curve_rel_tol), float(conv_cfg.sq_curve_abs_tol)
                if kind == "void_cdf":
                    return float(conv_cfg.void_cdf_rel_tol), float(conv_cfg.void_cdf_abs_tol)
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
                """Vector stats."""
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

            def _check_convergence(boxes: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
                return check_production_convergence(boxes, spec, conv_cfg)

            # adaptive box converged
            target = int(getattr(prod_cfg, "min_boxes", 10))
            if target < 1:
                target = 1

            # boxes hard unset
            # generating convergence achieved
            max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
            max_boxes: int | None
            if max_boxes_raw is None:
                max_boxes = None
            else:
                try:
                    max_boxes = int(max_boxes_raw)
                except Exception:
                    max_boxes = None
            if max_boxes is not None and int(max_boxes) <= 0:
                max_boxes = None
            if max_boxes is not None and int(max_boxes) < target:
                max_boxes = target

            # loop
            # active boxes unset
            HARD_MAX_BOXES = 10000
            batch = int(getattr(prod_cfg, "batch_boxes", 5))
            if batch < 1:
                batch = 1
            do_converge = bool(getattr(prod_cfg, "check_convergence", True))
            stage_diag = plan_production_stage_diagnostics(
                prod_cfg=prod_cfg,
                metrics_cfg=metrics_cfg,
                runner=runner,
                force_isotropic=bool(melt_force_iso),
                total_quench_steps=int(n_quench_prod),
                temperature_start=float(T_high),
                temperature_stop=float(q_cfg.t_final),
                sampling_hint=sampling_hint,
            )
            dump_traj = bool(stage_diag["dump_traj"])
            dump_every = int(stage_diag["dump_every"])
            collect_stage_metric_series = bool(stage_diag["collect_stage_metric_series"])
            collect_elastic_series = dict(stage_diag["collect_elastic_series"])
            need_stage_dump = dict(stage_diag["need_stage_dump"])
            quench_dump_every = int(stage_diag["quench_dump_every"])
            quench_window_steps_range = stage_diag["quench_window_steps_range"]
            relax_dump_settings = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)
            if sampling_hint is not None:
                progress.info(
                    "production",
                    f"quench analysis focus: Tm={(sampling_hint or {}).get('Tm')} -> freeze={(sampling_hint or {}).get('freeze_temperature')} ; dump_every={quench_dump_every}",
                )

            bondlen_cdf_points = int(getattr(prod_cfg, "bondlen_cdf_points", 200))
            angle_cdf_points = int(getattr(prod_cfg, "angle_cdf_points", 180))
            store_distributions = bool(getattr(prod_cfg, "store_distributions", True))

            # refinement cell optimisation
            dft_cfg = getattr(prod_cfg, "dft_opt", None)
            dft_enabled = bool(getattr(dft_cfg, "enabled", False)) if dft_cfg is not None else False
            cp2k_opt_runner: Cp2kRunner | None = None
            if dft_enabled:
                # runner backend otherwise
                # dedicated runner refinement
                if isinstance(runner, Cp2kRunner):
                    cp2k_opt_runner = runner
                else:
                    if getattr(config, "cp2k", None) is None:
                        raise RuntimeError(
                            "autotune.production.dft_opt.enabled=true but no cp2k configuration was provided"
                        )
                    cp2k_opt_runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]

            exclude_defects = bool(getattr(prod_cfg, "exclude_coordination_defects", False))
            rejects_subdir = str(getattr(prod_cfg, "rejects_subdir", "rejects") or "rejects")
            rejects_dir = prod_dir / rejects_subdir
            if exclude_defects:
                ensure_dir(rejects_dir)

            # rejected failure coordination
            rejected_boxes_dft: list[dict[str, Any]] = []
            if resume_state is not None:
                prev_rej_dft = resume_state.get("rejected_boxes_dft", None)
                if isinstance(prev_rej_dft, list):
                    rejected_boxes_dft = deepcopy(prev_rej_dft)

            # density conversion lammps
            lammps_units_style = resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal")

            def _density_from_cp2k_atoms(atoms) -> float:
                # cp2 ase angstrom
                rho_g_cm3 = float(density_g_cm3_from_atoms(atoms))
                if lammps_units_style in {"metal", "real", "cgs"}:
                    return float(rho_g_cm3)
                if lammps_units_style == "si":
                    # g cm kg
                    return float(rho_g_cm3) * 1000.0
                raise ValueError(
                    "DFT refinement requires a density-unit conversion, but user_units="
                    f"{lammps_units_style!r} is not supported. Use metal/real/si/cgs, or disable dft_opt."
                )

            def _pressure_to_bar(P: float) -> float:
                """Pressure to bar."""
                p = float(P)
                if lammps_units_style == "metal":
                    # bar
                    return p
                if lammps_units_style == "real":
                    # atm bar
                    return p * 1.01325
                if lammps_units_style == "si":
                    # pa bar
                    return p / 1.0e5
                if lammps_units_style == "cgs":
                    # dyne barye bar
                    return p / 1.0e6
                raise ValueError(
                    "DFT refinement requires a pressure-unit conversion, but user_units="
                    f"{lammps_units_style!r} is not supported. Use metal/real/si/cgs, or set dft_opt.external_pressure_bar explicitly."
                )

            def _slug(s: str) -> str:
                return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()

            def _atoms_to_dumpframe(atoms, *, type_to_species: Optional[list[str]] = None) -> "DumpFrame":
                """Atoms to dumpframe."""
                from ..analysis.dump import DumpFrame

                syms = list(atoms.get_chemical_symbols())
                n = len(syms)
                if int(n) < 1:
                    raise ValueError("DFT optimisation produced an empty structure")

                # symbols lammps integer
                if type_to_species is None:
                    # unique symbols order
                    uniq = sorted(set(syms))
                    sym_to_type = {s: i + 1 for i, s in enumerate(uniq)}
                else:
                    sym_to_type = {str(s): i + 1 for i, s in enumerate(list(type_to_species))}
                try:
                    types = np.asarray([int(sym_to_type[str(s)]) for s in syms], dtype=int)
                except KeyError as e:
                    raise ValueError(f"DFT structure contains symbol not present in type_to_species: {e}") from e

                pos = np.asarray(atoms.get_positions(), dtype=float)
                cell = np.asarray(atoms.get_cell(), dtype=float)
                if cell.shape != (3, 3):
                    raise ValueError("DFT structure has invalid cell shape")
                origin = np.zeros(3, dtype=float)
                ids = np.arange(1, int(n) + 1, dtype=int)
                return DumpFrame(timestep=0, cell=cell, origin=origin, types=types, positions=pos, ids=ids)

            def _run_dft_cell_opt_for_box(entry: dict[str, Any]) -> dict[str, Any]:
                """Dft cell opt."""
                assert cp2k_opt_runner is not None
                assert conv_spec is not None
                assert prod_cutoffs is not None

                from ase.io import read as ase_read
                from ase.io.cp2k import read_cp2k_dcd

                box_id = int(entry.get("box", 0) or 0)
                box_dir = prod_dir / f"box_{box_id:03d}"
                dft_dir = box_dir / "dft_opt"
                ensure_dir(dft_dir)

                def _read_lammps_data_to_atoms(p: Path):
                    atoms = None
                    try:
                        atoms = ase_read(
                            str(p),
                            format="lammps-data",
                            style=str(md_use.atom_style),
                            specorder=type_to_species,
                        )
                    except Exception:
                        try:
                            from ..io.lammps_data_minimal import read_lammps_data_minimal

                            atoms = read_lammps_data_minimal(p, specorder=type_to_species)
                        except Exception as e:
                            raise RuntimeError(f"Failed to read LAMMPS data: {p}") from e
                    assert atoms is not None
                    return atoms

                def _analyse_atoms_final(atoms_final, paths: dict[str, str]) -> dict[str, Any]:
                    # cell consistent analysis
                    try:
                        sp = atoms_final.get_scaled_positions(wrap=True)
                        atoms_final.set_scaled_positions(sp)
                    except Exception:
                        pass

                    # dump metrics cutoffs
                    fr = _atoms_to_dumpframe(atoms_final, type_to_species=type_to_species)
                    frames_dft = [fr]

                    sm_dft = compute_structure_metrics_timeavg(
                        frames_dft,
                        metrics_cfg,
                        cutoffs=prod_cutoffs,
                        type_to_species=type_to_species,
                    )
                    struct_vals_dft = dict(sm_dft.values)

                    dist_dft = compute_structure_distributions_timeavg(
                        frames_dft,
                        metrics_cfg,
                        cutoffs=prod_cutoffs,
                        type_to_species=type_to_species,
                        bondlen_cdf_points=bondlen_cdf_points,
                        angle_cdf_points=angle_cdf_points,
                    )

                    gr_curves_dft: dict[str, Any] = {}
                    for gm in list(metrics_cfg.gr):
                        label = "all" if gm.pair is None else f"{gm.pair[0]}-{gm.pair[1]}"
                        key = f"gr_{_slug(label)}"
                        r, g, _l = compute_gr(
                            frames_dft,
                            r_max=float(gm.r_max),
                            nbins=int(gm.nbins),
                            pair=gm.pair,
                            type_to_species=type_to_species,
                        )
                        gr_curves_dft[key] = {
                            "label": str(label),
                            "r": [float(v) for v in r.tolist()],
                            "g": [float(v) for v in g.tolist()],
                        }

                    sq_curves_dft: dict[str, Any] = {}
                    if hasattr(metrics_cfg, "sq"):
                        from ..analysis.sq import compute_sq
                        for sm in list(getattr(metrics_cfg, "sq", [])):
                            label = "all" if sm.pair is None else f"{sm.pair[0]}-{sm.pair[1]}"
                            key = f"sq_{_slug(label)}"
                            q, s = compute_sq(
                                frames_dft,
                                q_max=float(sm.q_max),
                                nq=int(sm.nq),
                                r_max=float(sm.r_max),
                                nbins=int(sm.nbins),
                                pair=sm.pair,
                                type_to_species=type_to_species,
                                window=str(getattr(sm, "window", "lorch")),
                            )
                            sq_curves_dft[key] = {
                                "label": str(label),
                                "q": [float(v) for v in q.tolist()],
                                "s": [float(v) for v in s.tolist()],
                            }

                    dist_all_dft = {
                        "bondlen": dist_dft.get("bondlen", {}),
                        "angle": dist_dft.get("angle", {}),
                        "coord": dist_dft.get("coord", {}),
                        "void": dist_dft.get("void", {}),
                        "gr": gr_curves_dft,
                        "sq": sq_curves_dft,
                    }

                    # guard
                    for nm in conv_spec.get("bondlen_names", []):
                        if nm not in dist_all_dft["bondlen"]:
                            raise RuntimeError(f"DFT box {box_id} missing bond-length distribution '{nm}'")
                    for nm in conv_spec.get("angle_names", []):
                        if nm not in dist_all_dft["angle"]:
                            raise RuntimeError(f"DFT box {box_id} missing angle distribution '{nm}'")
                    for nm in conv_spec.get("coord_names", []):
                        if nm not in dist_all_dft["coord"]:
                            raise RuntimeError(f"DFT box {box_id} missing coordination distribution '{nm}'")
                    for lab in conv_spec.get("gr_labels", []):
                        if lab not in dist_all_dft["gr"]:
                            raise RuntimeError(f"DFT box {box_id} missing g(r) curve '{lab}'")

                    for nm in conv_spec.get("void_names", []):
                        if nm not in dist_all_dft.get("void", {}):
                            raise RuntimeError(f"DFT box {box_id} missing void distribution '{nm}'")

                    for lab in conv_spec.get("sq_labels", []):
                        if lab not in dist_all_dft.get("sq", {}):
                            raise RuntimeError(f"DFT box {box_id} missing S(q) curve '{lab}'")

                    # coordination defects dft
                    coord_defects_dft = compute_coordination_defects(
                        fr,
                        metrics_cfg,
                        cutoffs=prod_cutoffs,
                        type_to_species=type_to_species,
                    )
                    has_coord_defects_dft = any(
                        bool((v or {}).get("has_defect", False)) for v in (coord_defects_dft or {}).values()
                    )

                    return {
                        "status": "ok",
                        "density": float(_density_from_cp2k_atoms(atoms_final)),
                        "density_stderr": 0.0,
                        "metrics": struct_vals_dft,
                        "distributions": dist_all_dft,
                        "has_coordination_defects": bool(has_coord_defects_dft),
                        "coordination_defects": coord_defects_dft,
                        "paths": paths,
                    }

                # optimisation already produced
                # reuse re cp2
                existing_data = dft_dir / "dft_opt.data"
                if existing_data.exists():
                    atoms_final = _read_lammps_data_to_atoms(existing_data)
                    paths: dict[str, str] = {
                        "dft_dir": str(dft_dir.relative_to(outdir)),
                        "dft_data": str(existing_data.relative_to(outdir)),
                    }
                    inp_path0 = dft_dir / "cell_opt.inp"
                    if inp_path0.exists():
                        paths["dft_input"] = str(inp_path0.relative_to(outdir))
                    out_path0 = dft_dir / "cp2k.out"
                    if out_path0.exists():
                        paths["dft_output"] = str(out_path0.relative_to(outdir))
                    traj_path0 = dft_dir / "traj.dcd"
                    if traj_path0.exists():
                        paths["dft_traj"] = str(traj_path0.relative_to(outdir))

                    return _analyse_atoms_final(atoms_final, paths)

                # structure classical relax
                relax_data_rel = Path(entry["paths"]["relax_data"])
                relax_data = (outdir / relax_data_rel) if not relax_data_rel.is_absolute() else relax_data_rel
                if not relax_data.exists():
                    raise FileNotFoundError(str(relax_data))

                atoms0 = _read_lammps_data_to_atoms(relax_data)

                # locally basis pseudopotentials
                # reference explicitly input
                cp2k_opt_runner._ensure_data_files_present(dft_dir)
                import os
                basis_cfg = str(cp2k_opt_runner.cfg.basis_set_file_name)
                pot_cfg = str(cp2k_opt_runner.cfg.potential_file_name)

                if os.path.isabs(basis_cfg) or ("/" in basis_cfg) or ("\\" in basis_cfg):
                    basis_file = basis_cfg
                    basis_path = Path(basis_cfg)
                else:
                    basis_path = dft_dir / Path(basis_cfg).name
                    basis_file = basis_path.name

                if os.path.isabs(pot_cfg) or ("/" in pot_cfg) or ("\\" in pot_cfg):
                    pot_file = pot_cfg
                    pot_path = Path(pot_cfg)
                else:
                    pot_path = dft_dir / Path(pot_cfg).name
                    pot_file = pot_path.name

                if not basis_path.exists():
                    raise FileNotFoundError(
                        "CP2K basis set file was not staged into the working directory. "
                        f"Expected: {basis_path}. "
                        "Either set cp2k.basis_set_file_name to an absolute path, or ensure the file is available via vitriflow/data/cp2k or CP2K_DATA_DIR."
                    )
                if not pot_path.exists():
                    raise FileNotFoundError(
                        "CP2K potential file was not staged into the working directory. "
                        f"Expected: {pot_path}. "
                        "Either set cp2k.potential_file_name to an absolute path, or ensure the file is available via vitriflow/data/cp2k or CP2K_DATA_DIR."
                    )

                # determine external pressure
                if getattr(dft_cfg, "external_pressure_bar", None) is not None:
                    extP = float(getattr(dft_cfg, "external_pressure_bar"))
                else:
                    extP = float(_pressure_to_bar(float(md_pressure)))

                # cp2 k input
                restart_file = None
                for rp in sorted(dft_dir.glob("dft_opt-*.restart")):
                    restart_file = rp.name
                    break
                inp_txt = render_cp2k_cell_opt_input(
                    atoms=atoms0,
                    cfg=cp2k_opt_runner.cfg,
                    basis_set_file_name=basis_file,
                    potential_file_name=pot_file,
                    project="dft_opt",
                    optimizer=str(getattr(dft_cfg, "optimizer", "LBFGS")),
                    max_iter=int(getattr(dft_cfg, "max_iter", 200)),
                    keep_angles=bool(getattr(dft_cfg, "keep_angles", True)),
                    external_pressure_bar=float(extP),
                    traj_every=int(getattr(dft_cfg, "traj_every", 1)),
                    traj_file="traj.dcd",
                    print_level=str(getattr(dft_cfg, "print_level", "LOW")),
                    restart_file=restart_file,
                )
                inp_path = dft_dir / "cell_opt.inp"
                inp_path.write_text(inp_txt)

                # execute cp2 k
                rr = cp2k_opt_runner.run(inp_path, dft_dir, output_name="cp2k.out")
                if int(rr.returncode) != 0:
                    raise RuntimeError(f"CP2K CELL_OPT failed (returncode={rr.returncode})")

                # recover frame trajectory
                traj_path = dft_dir / "traj.dcd"
                if not traj_path.exists():
                    raise RuntimeError("CP2K CELL_OPT did not produce trajectory file 'traj.dcd'")

                atoms_final = None
                try:
                    atoms_final = read_cp2k_dcd(str(traj_path), ref_atoms=atoms0, index=-1)
                except Exception as e:
                    raise RuntimeError(f"Failed to read CP2K trajectory: {traj_path}") from e
                if atoms_final is None:
                    raise RuntimeError("Failed to load final CP2K frame")

                # cell consistent analysis
                try:
                    sp = atoms_final.get_scaled_positions(wrap=True)
                    atoms_final.set_scaled_positions(sp)
                except Exception:
                    pass

                # persist lammps auditability
                try:
                    from ..structuregen import write_lammps_data

                    out_data = dft_dir / "dft_opt.data"
                    write_lammps_data(out_data, atoms_final, atom_style=str(md_use.atom_style), type_to_species=type_to_species)
                except Exception as e:
                    raise RuntimeError("Failed to write DFT-optimised LAMMPS data") from e

                return _analyse_atoms_final(
                    atoms_final,
                    {
                        "dft_dir": str(dft_dir.relative_to(outdir)),
                        "dft_input": str(inp_path.relative_to(outdir)),
                        "dft_output": str((dft_dir / "cp2k.out").relative_to(outdir)),
                        "dft_traj": str(traj_path.relative_to(outdir)),
                        "dft_data": str((dft_dir / "dft_opt.data").relative_to(outdir)),
                    },
                )

            def _ensure_dft_results() -> None:
                """Dft results."""
                if not dft_enabled:
                    return
                assert cp2k_opt_runner is not None
                for ent in boxes:
                    if ent.get("dft_opt", {}).get("status") in {"ok", "failed"}:
                        continue
                    try:
                        ent["dft_opt"] = _run_dft_cell_opt_for_box(ent)
                    except Exception as e:
                        ent["dft_opt"] = {
                            "status": "failed",
                            "error": str(e),
                            "paths": {"dft_dir": str((prod_dir / f"box_{int(ent.get('box',0)):03d}" / 'dft_opt').relative_to(outdir))},
                        }
                        rejected_boxes_dft.append({"box": int(ent.get("box", 0) or 0), "reason": "cp2k_failed", "error": str(e)})

            def _dft_accepted_boxes_view() -> list[dict[str, Any]]:
                """Dft accepted boxes."""
                out: list[dict[str, Any]] = []
                for ent in boxes:
                    d = ent.get("dft_opt", {})
                    if str(d.get("status", "")) != "ok":
                        continue
                    if exclude_defects and bool(d.get("has_coordination_defects", False)):
                        continue
                    out.append({"density": float(d["density"]), "metrics": d["metrics"], "distributions": d["distributions"]})
                return out

            boxes: list[dict[str, Any]] = []
            rejected_boxes: list[dict[str, Any]] = []
            conv_spec: dict[str, Any] | None = None

            # resume production regenerating
            next_box_id = 1
            if resume_state is not None:
                prev_boxes = resume_state.get("boxes", [])
                prev_rejected = resume_state.get("rejected_boxes", [])
                if isinstance(prev_boxes, list):
                    boxes = deepcopy(prev_boxes)
                if isinstance(prev_rejected, list):
                    rejected_boxes = deepcopy(prev_rejected)

                def _box_id(ent: dict[str, Any]) -> int:
                    try:
                        return int(ent.get("box", 0) or 0)
                    except Exception:
                        return 0

                boxes.sort(key=_box_id)
                rejected_boxes.sort(key=_box_id)

                rs_spec = resume_state.get("convergence_spec", None)
                if isinstance(rs_spec, dict) and len(rs_spec) > 0:
                    conv_spec = deepcopy(rs_spec)

                # coordination exclusion retroactively
                if exclude_defects and len(boxes) > 0:
                    kept: list[dict[str, Any]] = []
                    for ent in boxes:
                        if bool(ent.get("has_coordination_defects", False)):
                            ent_rej = deepcopy(ent)
                            ent_rej["reason"] = "coordination_defects"
                            rejected_boxes.append(ent_rej)
                        else:
                            kept.append(ent)
                    boxes = kept

                ids: list[int] = []
                for ent in list(boxes) + list(rejected_boxes):
                    try:
                        ids.append(int(ent.get("box", 0) or 0))
                    except Exception:
                        pass
                if len(ids) > 0:
                    next_box_id = max(ids) + 1

                # seed
                prev_entries = []
                if isinstance(prev_boxes, list):
                    prev_entries.extend(prev_boxes)
                if isinstance(prev_rejected, list):
                    prev_entries.extend(prev_rejected)
                default_seed_draws = 4 if int(warmup_steps) > 0 else 3
                consumed_draws = _count_production_resume_seed_draws(
                    prev_entries,
                    default_draws=default_seed_draws,
                )
                for _ in range(int(consumed_draws)):
                    rng_prod.randrange(1, 2**31 - 1)

            # resuming reconstruct convergence
            # accepted box
            if conv_spec is None and len(boxes) > 0:
                b0 = boxes[0]
                dist0 = b0.get("distributions", {})
                met0 = b0.get("metrics", {})
                if not isinstance(dist0, dict) or not isinstance(met0, dict):
                    raise RuntimeError("Cannot resume: production boxes lack stored metrics/distributions")

                bond_names = sorted(list((dist0.get("bondlen", {}) or {}).keys()))
                angle_names = sorted(list((dist0.get("angle", {}) or {}).keys()))
                coord_names = sorted(list((dist0.get("coord", {}) or {}).keys()))
                ring_keys = sorted([k for k in met0.keys() if str(k).startswith("ring_frac_")])
                ring_has_mean_size = bool("ring_mean_size" in met0)
                gr_labels = sorted(list((dist0.get("gr", {}) or {}).keys()))
                sq_labels = sorted(list((dist0.get("sq", {}) or {}).keys()))
                void_names = sorted(list((dist0.get("void", {}) or {}).keys()))

                # empty malformed distributions
                for nm in bond_names:
                    if not (dist0.get("bondlen", {}).get(nm, {}) or {}).get("cdf"):
                        raise RuntimeError(f"Resume: empty bond-length CDF for '{nm}'")
                for nm in angle_names:
                    if not (dist0.get("angle", {}).get(nm, {}) or {}).get("cdf"):
                        raise RuntimeError(f"Resume: empty angle CDF for '{nm}'")
                for nm in coord_names:
                    if not (dist0.get("coord", {}).get(nm, {}) or {}).get("cdf"):
                        raise RuntimeError(f"Resume: empty coordination CDF for '{nm}'")
                for lab in gr_labels:
                    if not (dist0.get("gr", {}).get(lab, {}) or {}).get("g"):
                        raise RuntimeError(f"Resume: empty g(r) for '{lab}'")
                for nm in void_names:
                    if not (dist0.get("void", {}).get(nm, {}) or {}).get("cdf"):
                        raise RuntimeError(f"Resume: empty void CDF for '{nm}'")
                for lab in sq_labels:
                    if not (dist0.get("sq", {}).get(lab, {}) or {}).get("s"):
                        raise RuntimeError(f"Resume: empty S(q) for '{lab}'")

                conv_spec = {
                    "bondlen_names": bond_names,
                    "angle_names": angle_names,
                    "coord_names": coord_names,
                    "ring_keys": ring_keys,
                    "ring_has_mean_size": ring_has_mean_size,
                    "gr_labels": gr_labels,
                    "sq_labels": sq_labels,
                    "void_names": void_names,
                }
            conv_report_md: dict[str, Any] = {}
            conv_report_dft: dict[str, Any] = {}
            converged_md = False
            converged_dft: bool | None = None
            converged = False

            required_streak = max(1, int(getattr(prod_cfg, "consecutive_converged_checks", 1)))
            converged_streak = 0

            # resuming finished production
            # seed
            # resume immediately trigger
            # forcing unnecessary extra
            if resume_state is not None and bool(resume_state.get("converged_md", False)):
                converged_streak = max(converged_streak, required_streak - 1)

            dft_summary: dict[str, Any] | None = None
            boxes_dft_final: list[int] | None = None

            def _production_state(*, status: str, error: Optional[str] = None) -> dict[str, Any]:
                metrics_checked = metrics_checked_from_conv_spec(conv_spec)
                final_conv_report: dict[str, Any] = conv_report_md
                if dft_enabled and converged_dft is not None:
                    final_conv_report = conv_report_dft
                motif_summary = summarize_production_crystal_motifs(boxes, rejected_boxes=rejected_boxes)
                return {
                    "enabled": True,
                    "status": str(status),
                    "error": (str(error) if error is not None else None),
                    "converged": bool(converged),
                    "n_boxes": int(len(boxes)),
                    "n_boxes_accepted": int(len(boxes)),
                    "n_boxes_rejected": int(len(rejected_boxes)),
                    "n_boxes_total": int(len(boxes) + len(rejected_boxes)),
                    "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)),
                    "max_boxes": int(max_boxes) if max_boxes is not None else None,
                    "batch_boxes": int(batch),
                    "check_convergence": bool(do_converge),
                    "dump_trajectory": bool(dump_traj),
                    "dump_every_steps": int(dump_every),
                    "rate_K_per_time": float(prod_rate),
                    "rate_K_per_ps": float(cooling_rate_ps) if cooling_rate_ps is not None else None,
                    "replicate": [int(nx), int(ny), int(nz)],
                    "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "exclude_coordination_defects": bool(exclude_defects),
                    "rejects_subdir": str(rejects_subdir) if bool(exclude_defects) else None,
                    "rejects_dir": str(rejects_dir.relative_to(outdir)) if bool(exclude_defects) and rejects_dir.exists() else None,
                    "warmup_start_temperature": float(warmup_start_temperature),
                    "warmup_duration_ps": float(warmup_duration_ps),
                    "warmup_steps": int(warmup_steps),
                    "T_high": float(T_high),
                    "t_final": float(q_cfg.t_final),
                    "quench_steps": int(n_quench_prod),
                    "highT_steps": int(high_total_steps),
                    "relax_steps": int(relax_steps_prod),
                    "cutoffs": ([{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted((prod_cutoffs or {}).items())] if isinstance(prod_cutoffs, dict) else None),
                    "metrics_checked": metrics_checked,
                    "convergence_spec": conv_spec,
                    "converged_md": bool(converged_md),
                    "convergence_md": conv_report_md,
                    "converged_dft": (bool(converged_dft) if dft_enabled and converged_dft is not None else None),
                    "convergence_dft": (conv_report_dft if dft_enabled and converged_dft is not None else None),
                    "convergence": final_conv_report,
                    "crystal_motifs": motif_summary,
                    "dft_opt": dft_summary,
                    "boxes_dft_final": boxes_dft_final,
                    "n_boxes_dft_accepted": (int(len(boxes_dft_final)) if isinstance(boxes_dft_final, list) else None),
                    "rejected_boxes_dft": rejected_boxes_dft if dft_enabled else None,
                    "boxes": boxes,
                    "rejected_boxes": rejected_boxes,
                    "ensemble_dir": str(prod_dir.relative_to(outdir)),
                }

            def _checkpoint(*, status: str, error: Optional[str] = None) -> None:
                if checkpoint_cb is None:
                    return
                checkpoint_cb(_production_state(status=status, error=error))

            _checkpoint(status="starting")
            while True:
                while len(boxes) < target:
                    total_boxes = len(boxes) + len(rejected_boxes)
                    if max_boxes is not None and total_boxes >= int(max_boxes):
                        break
                    if max_boxes is None and total_boxes >= int(HARD_MAX_BOXES):
                        raise RuntimeError(
                            f"Production ensemble failed to converge after {HARD_MAX_BOXES} boxes. "
                            "This indicates tolerances are likely too strict or metrics are ill-posed."
                        )
                    b = int(next_box_id)
                    next_box_id += 1
                    bdir = prod_dir / f"box_{b:03d}"
                    ensure_dir(bdir)
                    progress.info("production", f"box {b}: starting ({len(boxes)+len(rejected_boxes)+1} total attempted)")

                    # distinct warm followed
                    # independent melt preserves
                    # high duration physics
                    seed_warmup = int(rng_prod.randrange(1, 2**31 - 1))
                    warmup_stage = StageSpec(
                        name="warmup",
                        input_data=size_base_data,
                        output_data=bdir / "warmup.data",
                        temperature_start=float(warmup_start_temperature),
                        temperature_stop=float(T_high),
                        pressure=float(md_pressure),
                        equil_steps=0,
                        run_steps=int(warmup_steps),
                        seed=seed_warmup,
                        velocity_mode="create",
                        force_isotropic=melt_force_iso,
                        replicate=(nx, ny, nz),
                        write_dump=bool(need_stage_dump["melt"]),
                        dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                        msd_every=int(tm_cfg.msd_every),
                    )

                    seed_melt = int(rng_prod.randrange(1, 2**31 - 1))
                    melt_stage = StageSpec(
                        name="melt",
                        input_data=bdir / "warmup.data",
                        output_data=bdir / "melt.data",
                        temperature_start=float(T_high),
                        temperature_stop=float(T_high),
                        pressure=float(md_pressure),
                        equil_steps=0,
                        run_steps=int(high_total_steps),
                        seed=seed_melt,
                        velocity_mode="preserve",
                        force_isotropic=False,
                        replicate=None,
                        write_dump=bool(need_stage_dump["melt"]),
                        dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    if cont == "continuous" and isinstance(runner, LammpsRunner):
                        progress.info("production", f"box {b}: continuous warmup→melt→quench→relax")
                        # lammps warmup quench
                        # directories populated analysis
                        seed_quench = int(rng_prod.randrange(1, 2**31 - 1))
                        quench_stage = StageSpec(
                            name="quench",
                            input_data=bdir / "melt.data",  # ignored continuous rendering
                            output_data=bdir / "quench.data",
                            temperature_start=T_high,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(n_quench_prod),
                            seed=seed_quench,
                            velocity_mode="preserve",
                            replicate=None,
                            write_dump=bool(need_stage_dump["quench"]),
                            dump_every=int(quench_dump_every) if need_stage_dump["quench"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )

                        seed_relax = int(rng_prod.randrange(1, 2**31 - 1))
                        relax_stage = StageSpec(
                            name="relax",
                            input_data=bdir / "quench.data",  # ignored continuous rendering
                            output_data=bdir / "relax.data",
                            temperature_start=q_cfg.t_final,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(relax_steps_prod),
                            seed=seed_relax,
                            velocity_mode="preserve",
                            replicate=None,
                            write_dump=bool(relax_dump_settings["write_dump"]),
                            dump_every=relax_dump_settings["dump_every"],
                            tail_dump_frames=relax_dump_settings["tail_dump_frames"],
                            tail_dump_stride=relax_dump_settings["tail_dump_stride"],
                            msd_every=int(tm_cfg.msd_every),
                        )
                        relax_dir = bdir / "relax"

                        arts = run_stages_continuous_lammps(
                            runner,
                            pot_cfg,
                            md_use,
                            [warmup_stage, melt_stage, quench_stage, relax_stage],
                            [bdir / "warmup", bdir / "melt", bdir / "quench", relax_dir],
                            bdir / "continuous",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )
                        warmup_out = stage_outcome_from_artifacts(arts[0], md_cfg=md_use, stage=warmup_stage)
                        melt_out = stage_outcome_from_artifacts(arts[1], md_cfg=md_use, stage=melt_stage)
                        quench_out = stage_outcome_from_artifacts(arts[2], md_cfg=md_use, stage=quench_stage)
                        relax_out = stage_outcome_from_artifacts(arts[3], md_cfg=md_use, stage=relax_stage)
                    else:
                        progress.info("production", f"box {b}: warmup")
                        warmup_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            warmup_stage,
                            bdir / "warmup",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        melt_stage = StageSpec(
                            name="melt",
                            input_data=(bdir / "warmup" / warmup_out.output_data),
                            output_data=bdir / "melt.data",
                            temperature_start=float(T_high),
                            temperature_stop=float(T_high),
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(high_total_steps),
                            seed=seed_melt,
                            velocity_mode="preserve",
                            force_isotropic=False,
                            replicate=None,
                            write_dump=bool(need_stage_dump["melt"]),
                            dump_every=int(dump_every) if need_stage_dump["melt"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )

                        progress.info("production", f"box {b}: melt")
                        melt_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            melt_stage,
                            bdir / "melt",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        # quench selected rate
                        progress.info("production", f"box {b}: quench")
                        seed_quench = int(rng_prod.randrange(1, 2**31 - 1))
                        quench_stage = StageSpec(
                            name="quench",
                            input_data=(bdir / "melt" / melt_out.output_data),
                            output_data=bdir / "quench.data",
                            temperature_start=T_high,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(n_quench_prod),
                            seed=seed_quench,
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(need_stage_dump["quench"]),
                            dump_every=int(quench_dump_every) if need_stage_dump["quench"] else None,
                            msd_every=int(tm_cfg.msd_every),
                        )
                        quench_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            quench_stage,
                            bdir / "quench",
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                        # relax dump metrics
                        progress.info("production", f"box {b}: relax")
                        seed_relax = int(rng_prod.randrange(1, 2**31 - 1))
                        relax_stage = StageSpec(
                            name="relax",
                            input_data=(bdir / "quench" / quench_out.output_data),
                            output_data=bdir / "relax.data",
                            temperature_start=q_cfg.t_final,
                            temperature_stop=q_cfg.t_final,
                            pressure=float(md_pressure),
                            equil_steps=0,
                            run_steps=int(relax_steps_prod),
                            seed=seed_relax,
                            velocity_mode=vel_next,
                            replicate=None,
                            write_dump=bool(relax_dump_settings["write_dump"]),
                            dump_every=relax_dump_settings["dump_every"],
                            tail_dump_frames=relax_dump_settings["tail_dump_frames"],
                            tail_dump_stride=relax_dump_settings["tail_dump_stride"],
                            msd_every=int(tm_cfg.msd_every),
                        )
                        relax_dir = bdir / "relax"
                        relax_out = _stage_run(
                            runner,
                            pot_cfg,
                            md_use,
                            relax_stage,
                            relax_dir,
                            potential_lines=potential_lines,
                            type_to_species=type_to_species,
                        )

                    progress.info("production", f"box {b}: stage execution complete")
                    melt_stage_dir = bdir / "melt"
                    melt_elastic = _maybe_elastic(
                        "melt",
                        stage_dir=melt_stage_dir,
                        structure_data=Path(melt_out.output_data) if isinstance(melt_out.output_data, Path) and Path(melt_out.output_data).is_absolute() else (melt_stage_dir / melt_out.output_data),
                        input_data=(bdir / "warmup" / warmup_out.output_data),
                        force_iso_context=bool(melt_force_iso),
                    )
                    quench_stage_dir = bdir / "quench"
                    relax_elastic = _maybe_elastic(
                        "relax",
                        stage_dir=relax_dir,
                        structure_data=(relax_dir / relax_out.output_data),
                        input_data=(quench_stage_dir / quench_out.output_data),
                        force_iso_context=bool(melt_force_iso),
                    )
                    elastic_timeseries = {
                        "melt": _maybe_elastic_series(
                            "melt",
                            stage_dir=melt_stage_dir,
                            stage_output_data=(
                                Path(melt_out.output_data)
                                if isinstance(melt_out.output_data, Path) and Path(melt_out.output_data).is_absolute()
                                else (melt_stage_dir / melt_out.output_data)
                            ),
                            force_iso_context=bool(melt_force_iso),
                        ),
                        "quench": _maybe_elastic_series(
                            "quench",
                            stage_dir=quench_stage_dir,
                            stage_output_data=(quench_stage_dir / quench_out.output_data),
                            force_iso_context=bool(melt_force_iso),
                            sampling_hint=sampling_hint,
                        ),
                        "relax": _maybe_elastic_series(
                            "relax",
                            stage_dir=relax_dir,
                            stage_output_data=(relax_dir / relax_out.output_data),
                            force_iso_context=bool(melt_force_iso),
                        ),
                    }

                    dump_path = relax_dir / f"{relax_stage.name}.lammpstrj"
                    cand = relax_dir / "traj.extxyz"
                    traj_path = cand if cand.exists() else dump_path
                    entry, prod_cutoffs = analyse_production_box(
                        box_id=int(b),
                        outdir=outdir,
                        melt_stage_dir=melt_stage_dir,
                        quench_stage_dir=quench_stage_dir,
                        relax_stage_dir=relax_dir,
                        relax_data_path=(relax_dir / relax_out.output_data),
                        density_mean=float(relax_out.density_mean),
                        density_stderr=float(relax_out.density_stderr),
                        metrics_cfg=metrics_cfg,
                        cutoffs=prod_cutoffs,
                        required_pairs=required_pairs,
                        fixed_cutoffs=fixed_cut,
                        type_to_species=type_to_species,
                        md_timestep=float(md_use.timestep),
                        quench_window_steps_range=quench_window_steps_range,
                        sampling_hint=sampling_hint,
                        bondlen_cdf_points=int(bondlen_cdf_points),
                        angle_cdf_points=int(angle_cdf_points),
                        seeds={"warmup": int(seed_warmup), "melt": int(seed_melt), "quench": int(seed_quench), "relax": int(seed_relax)},
                        melt_elastic=melt_elastic,
                        relax_elastic=relax_elastic,
                        elastic_timeseries=elastic_timeseries,
                        exclude_coordination_defects=bool(exclude_defects),
                        rejects_dir=(rejects_dir if bool(exclude_defects) else None),
                        relax_dump_path=dump_path,
                        relax_traj_path=traj_path,
                    )

                    if conv_spec is None:
                        conv_spec = build_production_convergence_spec(entry)
                    else:
                        validate_production_entry_against_spec(entry, conv_spec, box_label=b)

                    if bool(entry.get("reject")):
                        rejected_boxes.append(entry)
                        rej = dict(entry.get("reject", {}) or {})
                        reason = str(rej.get("reason", "rejected"))
                        progress.warn("production", f"box {b}: rejected ({reason})")
                    else:
                        boxes.append(entry)
                        progress.info("production", f"box {b}: accepted ({len(boxes)} accepted total)")
                    _checkpoint(status="running")

                if not do_converge:
                    # convergence checking requested
                    # finished generating requested
                    converged = True
                    conv_report_md = {}
                    break

                # convergence configured metric
                if conv_spec is None:
                    # distributions metrics converge
                    converged = False
                    conv_report_md = {"error": "no distributions available for convergence"}
                    break

                if len(boxes) < 1:
                    converged = False
                    conv_report_md = {"error": "no accepted boxes (all rejected)"}
                    break

                # convergence classical boxes
                converged_md, conv_report_md = _check_convergence(boxes, conv_spec)

                if converged_md:
                    converged_streak += 1
                else:
                    converged_streak = 0
                progress.convergence("production", conv_report_md)
                progress.info("production", f"MD convergence streak: {converged_streak}/{required_streak}")
                _checkpoint(status="running")


                # refinement disabled ensemble
                # remained converged requested
                if (not dft_enabled) and converged_md and converged_streak >= required_streak:
                    converged = True
                    break

                # refinement enabled refined
                # ensemble converged expensive
                # ensemble requested convergence
                if dft_enabled and converged_md and converged_streak >= required_streak:
                    # box dft optimisation
                    _ensure_dft_results()
                    dft_view = _dft_accepted_boxes_view()
                    if len(dft_view) < 1:
                        converged_dft = False
                        conv_report_dft = {"error": "no DFT-accepted boxes (all failed or rejected)"}
                        converged = False
                    else:
                        converged_dft, conv_report_dft = _check_convergence(dft_view, conv_spec)
                        converged = bool(converged_dft)

                    if converged:
                        break

                    # loop
                    # add structures
                    converged_streak = 0
                if max_boxes is not None and (len(boxes) + len(rejected_boxes)) >= int(max_boxes):
                    break
                if max_boxes is None and (len(boxes) + len(rejected_boxes)) >= int(HARD_MAX_BOXES):
                    raise RuntimeError(
                        f"Production ensemble failed to converge after {HARD_MAX_BOXES} boxes. "
                        "Relax convergence tolerances or set production.max_boxes to impose a cap."
                    )

                if max_boxes is None:
                    target = len(boxes) + batch
                else:
                    target = min(int(max_boxes), len(boxes) + batch)

            # box distributions smaller
            if not store_distributions:
                for b in boxes:
                    if "distributions" in b:
                        b.pop("distributions")
                    if isinstance(b.get("dft_opt"), dict) and "distributions" in b["dft_opt"]:
                        b["dft_opt"].pop("distributions")
                for b in rejected_boxes:
                    if "distributions" in b:
                        b.pop("distributions")

            metrics_checked = metrics_checked_from_conv_spec(conv_spec)

            # summarise refinement outcomes
            dft_summary: dict[str, Any] | None = None
            boxes_dft_final: list[int] | None = None
            if dft_enabled:
                n_ok = 0
                n_failed = 0
                n_defects = 0
                n_not_run = 0
                final_ids: list[int] = []
                for ent in boxes:
                    d = ent.get("dft_opt", {}) if isinstance(ent.get("dft_opt"), dict) else {}
                    st = str(d.get("status", ""))
                    if st == "ok":
                        n_ok += 1
                        if exclude_defects and bool(d.get("has_coordination_defects", False)):
                            n_defects += 1
                            rejected_boxes_dft.append({"box": int(ent.get("box", 0) or 0), "reason": "coordination_defects_dft"})
                        else:
                            final_ids.append(int(ent.get("box", 0) or 0))
                    elif st == "failed":
                        n_failed += 1
                    else:
                        n_not_run += 1

                boxes_dft_final = sorted(final_ids)
                extP_used = (
                    float(getattr(dft_cfg, "external_pressure_bar", 0.0))
                    if getattr(dft_cfg, "external_pressure_bar", None) is not None
                    else float(_pressure_to_bar(float(md_pressure)))
                )
                dft_summary = {
                    "enabled": True,
                    "optimizer": str(getattr(dft_cfg, "optimizer", "LBFGS")),
                    "max_iter": int(getattr(dft_cfg, "max_iter", 200)),
                    "keep_angles": True,
                    "external_pressure_bar": float(extP_used),
                    "traj_every": int(getattr(dft_cfg, "traj_every", 1)),
                    "print_level": str(getattr(dft_cfg, "print_level", "LOW")),
                    "n_boxes_ok": int(n_ok),
                    "n_boxes_failed": int(n_failed),
                    "n_boxes_rejected_coordination_defects": int(n_defects),
                    "n_boxes_not_run": int(n_not_run),
                    "n_boxes_accepted": int(len(final_ids)),
                }

            production = _production_state(status="ok")

        return production

def _run_production_ensemble(
    *,
    config: RunConfig,
    outdir: Path,
    runner: Union[LammpsRunner, Cp2kRunner],
    pot_cfg: KimConfig,
    md_use: MDConfig,
    potential_lines: Optional[list[str]],
    type_to_species: Optional[list[str]],
    metrics_cfg,
    tm_cfg,
    q_cfg,
    size_base_data: Path,
    chosen_replicate: list[int],
    chosen_rate: float,
    dt_ref: float,
    dt_mq: float,
    cooling_rate_ps: Optional[float],
    cutoffs_rate: dict[tuple[int, int], float],
    cutoffs_size: dict[tuple[int, int], float],
    T_high: float,
    high_total_steps: int,
    resume_state: Optional[dict[str, Any]] = None,
    sampling_hint: Optional[dict[str, float]] = None,
    progress: Optional[CondensedProgressLog] = None,
    checkpoint_cb=None,
    pressure_override: Optional[float] = None,
    seed_base: Optional[int] = None,
    time_unit_ps_override: Optional[float] = None,
    prod_cfg_override=None,
    conv_cfg_override=None,
    quench_steps_override: Optional[int] = None,
    relax_steps_override: Optional[int] = None,
) -> dict[str, Any]:
    """Production ensemble."""
    return _ProductionEnsembleRunner(
        config=config,
        outdir=outdir,
        runner=runner,
        pot_cfg=pot_cfg,
        md_use=md_use,
        potential_lines=potential_lines,
        type_to_species=type_to_species,
        metrics_cfg=metrics_cfg,
        tm_cfg=tm_cfg,
        q_cfg=q_cfg,
        size_base_data=size_base_data,
        chosen_replicate=chosen_replicate,
        chosen_rate=chosen_rate,
        dt_ref=dt_ref,
        dt_mq=dt_mq,
        cooling_rate_ps=cooling_rate_ps,
        cutoffs_rate=cutoffs_rate,
        cutoffs_size=cutoffs_size,
        T_high=T_high,
        high_total_steps=high_total_steps,
        resume_state=resume_state,
        sampling_hint=sampling_hint,
        progress=progress,
        checkpoint_cb=checkpoint_cb,
        pressure_override=pressure_override,
        seed_base=seed_base,
        time_unit_ps_override=time_unit_ps_override,
        prod_cfg_override=prod_cfg_override,
        conv_cfg_override=conv_cfg_override,
        quench_steps_override=quench_steps_override,
        relax_steps_override=relax_steps_override,
    ).run()


def _autotune_resume_from_results(*, config: RunConfig, outdir: Path, prev: dict[str, Any]) -> dict[str, Any]:
    """Autotune resume from."""

    # sanity resuming production
    prev_prod = prev.get("production", None)
    prev_rec = prev.get("recommendation", {}) if isinstance(prev.get("recommendation", {}), dict) else {}
    if not isinstance(prev_prod, dict):
        raise RuntimeError("Cannot resume: existing autotune_results.json has no 'production' state")

    # runner selection
    engine = str(getattr(config, "engine", "lammps") or "lammps").strip().lower()
    if engine not in {"lammps", "cp2k"}:
        raise ValueError(f"Unsupported engine '{engine}'")

    if engine == "lammps":
        runner: Union[LammpsRunner, Cp2kRunner] = LammpsRunner(config.lammps)
    else:
        runner = Cp2kRunner(config.cp2k)

    # kim already installed
    pot_cfg = config.kim
    kim_install = ensure_model_installed(pot_cfg, outdir)

    # species mapping
    type_to_species = _get_type_to_species(config)

    progress = CondensedProgressLog(outdir / "condensed.log")
    progress.info("autotune", "resuming from existing autotune_results.json")

    metric_warnings: list[str] = list(prev.get("metric_warnings", []) or [])

    def _warn_metric(msg: str) -> None:
        if str(msg) not in metric_warnings:
            metric_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("metrics", str(msg))

    # preflight integrator consistent
    pf = prev.get("preflight", {}) if isinstance(prev.get("preflight", {}), dict) else {}
    md_rec = prev_rec.get("md", {}) if isinstance(prev_rec.get("md", {}), dict) else {}

    timestep = md_rec.get("timestep", pf.get("selected_timestep", None))
    ensemble = md_rec.get("ensemble", pf.get("selected_ensemble", None))
    tdamp = (md_rec.get("thermostat", {}) or {}).get("tdamp", pf.get("selected_tdamp", None))
    pdamp = (md_rec.get("barostat", {}) or {}).get("pdamp", pf.get("selected_pdamp", None))

    md_use = config.md
    md_update: dict[str, Any] = {}
    if timestep is not None:
        md_update["timestep"] = float(timestep)
    if ensemble is not None:
        md_update["ensemble"] = str(ensemble)
    if tdamp is not None:
        md_update["thermostat"] = md_use.thermostat.model_copy(update={"tdamp": float(tdamp)})
    if pdamp is not None and md_use.barostat is not None:
        md_update["barostat"] = md_use.barostat.model_copy(update={"pdamp": float(pdamp)})
    if len(md_update) > 0:
        md_use = md_use.model_copy(update=md_update)

    # potential preflight overlay
    potential_lines = pf.get("potential_lines", None)

    # production parameters
    chosen_rate = prev_prod.get("rate_K_per_time", prev_rec.get("cooling_rate_K_per_time", None))
    if chosen_rate is None:
        raise RuntimeError("Cannot resume: missing cooling rate in existing results")
    chosen_rate = float(chosen_rate)

    chosen_replicate = prev_prod.get("replicate", prev_rec.get("replicate", None))
    if not isinstance(chosen_replicate, list) or len(chosen_replicate) != 3:
        raise RuntimeError("Cannot resume: missing/invalid replicate in existing results")
    chosen_replicate = [int(x) for x in chosen_replicate]

    T_high = prev_prod.get("T_high", prev_rec.get("T_high", None))
    high_total_steps = prev_prod.get("highT_steps", prev_rec.get("highT_steps", None))
    if T_high is None or high_total_steps is None:
        raise RuntimeError("Cannot resume: missing high-T parameters in existing results")
    T_high = float(T_high)
    high_total_steps = int(high_total_steps)
    if bool(getattr(config.md, "force_isotropic", False)):
        prev_high = prev.get("highT", {}) if isinstance(prev.get("highT", {}), dict) else {}
        try:
            prev_factor = float(prev_high.get("force_isotropic_extension_factor", 1.0))
        except Exception:
            prev_factor = 1.0
        if prev_factor <= 1.0 + 1.0e-12:
            high_total_steps = extend_highT_steps_for_force_isotropic(
                int(high_total_steps),
                force_isotropic=True,
            )

    # cooling rate ps
    units = prev.get("units", {}) if isinstance(prev.get("units", {}), dict) else {}
    time_unit_ps = units.get("time_unit_ps", None)
    cooling_rate_ps = prev_rec.get("cooling_rate_K_per_ps", None)
    if cooling_rate_ps is None:
        if time_unit_ps is not None:
            try:
                cooling_rate_ps = float(chosen_rate) / float(time_unit_ps)
            except Exception:
                cooling_rate_ps = None
    else:
        cooling_rate_ps = float(cooling_rate_ps)

    # structure further ensemble
    size_base_rel = None
    ss = prev.get("size_scan", {}) if isinstance(prev.get("size_scan", {}), dict) else {}
    if isinstance(ss.get("base_data", None), str):
        size_base_rel = ss.get("base_data")
    if size_base_rel is None and isinstance(prev_rec.get("structure_data", None), str):
        size_base_rel = prev_rec.get("structure_data")
    if size_base_rel is None:
        raise RuntimeError("Cannot resume: missing base_data path in existing results")
    size_base_data = Path(size_base_rel)
    if not size_base_data.is_absolute():
        size_base_data = outdir / size_base_data

    metrics_cfg, metric_auto_defaults, metrics_summary = resolve_effective_metrics_config(
        config.autotune.metrics,
        structure_data=Path(size_base_data),
        type_to_species=type_to_species,
        warn_fn=_warn_metric,
        context="autotune production",
    )
    progress.info("metrics", f"effective metrics summary: {metrics_summary}")

    # cutoffs production override
    rs_rate = prev.get("rate_scan", {})
    rs_size = prev.get("size_scan", {})
    cutoffs_rate = _cutoffs_any_to_dict(rs_rate.get("cutoffs", None) if isinstance(rs_rate, dict) else None)
    cutoffs_size = _cutoffs_any_to_dict(rs_size.get("cutoffs", None) if isinstance(rs_size, dict) else None)

    # quench temperature relax
    q_cfg = config.autotune.quench
    if "t_final" in prev_prod:
        q_cfg = q_cfg.model_copy(update={"t_final": float(prev_prod["t_final"])})
    if "relax_steps" in prev_prod:
        q_cfg = q_cfg.model_copy(update={"relax_steps": int(prev_prod["relax_steps"])})

    tm_cfg = config.autotune.tm_scan

    dt_ref = float(getattr(config.md, "timestep", md_use.timestep))
    dt_mq = float(md_use.timestep)

    prev["kim_install"] = kim_install
    prev["metric_warnings"] = list(metric_warnings)
    prev["effective_metrics"] = dict(metrics_summary)
    prev["paths"] = {
        "autotune_results": "autotune_results.json",
        "autotune": "autotune.json",
        "condensed_log": "condensed.log",
    }

    def _checkpoint_production(prod_state: dict[str, Any]) -> None:
        prev["status"] = "running"
        prev["production"] = dict(prod_state)
        prev["metric_warnings"] = list(metric_warnings)
        prev["effective_metrics"] = dict(metrics_summary)
        write_autotune_outputs(outdir, prev)

    write_autotune_outputs(outdir, prev)
    progress.info("production", "resuming production ensemble")
    production = _run_production_ensemble(
        config=config,
        outdir=outdir,
        runner=runner,
        pot_cfg=pot_cfg,
        md_use=md_use,
        potential_lines=potential_lines,
        type_to_species=type_to_species,
        metrics_cfg=metrics_cfg,
        tm_cfg=tm_cfg,
        q_cfg=q_cfg,
        size_base_data=size_base_data,
        chosen_replicate=chosen_replicate,
        chosen_rate=chosen_rate,
        dt_ref=dt_ref,
        dt_mq=dt_mq,
        cooling_rate_ps=cooling_rate_ps,
        cutoffs_rate=cutoffs_rate,
        cutoffs_size=cutoffs_size,
        T_high=T_high,
        high_total_steps=high_total_steps,
        resume_state=prev_prod,
        progress=progress,
        checkpoint_cb=_checkpoint_production,
        time_unit_ps_override=(None if time_unit_ps is None else float(time_unit_ps)),
    )

    prev["status"] = "ok"
    prev["production"] = production
    prev["metric_warnings"] = list(metric_warnings)
    prev["effective_metrics"] = dict(metrics_summary)
    write_autotune_outputs(outdir, prev)
    return prev




class _AutotuneWorkflow:
    """Autotune workflow."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self) -> dict[str, Any]:
        config = self.config
        outdir = self.outdir
        resume = self.resume
        ensure_dir(outdir)

        results_path = outdir / "autotune_results.json"
        do_resume = False
        if resume is None:
            do_resume = results_path.exists()
        else:
            do_resume = bool(resume) and results_path.exists()
        if do_resume:
            prev = json.loads(results_path.read_text())
            return _autotune_resume_from_results(config=config, outdir=outdir, prev=prev)

        # starting structure benchmarks
        initial_data = prepare_initial_structure(config, outdir)

        type_to_species = _get_type_to_species(config)

        if getattr(config, "engine", "lammps") == "cp2k":
            kim_install = None
            runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
        else:
            # kim installation potentials
            kim_install = ensure_model_installed(getattr(config.kim, "model", ""))
            runner = LammpsRunner(config.lammps)

        if bool(getattr(config.md, "force_isotropic", False)) and (not isinstance(runner, LammpsRunner)):
            raise ValueError(
                "md.force_isotropic is currently supported only for engine='lammps'"
            )

        progress = CondensedProgressLog(outdir / "condensed.log")
        progress.info("autotune", "initialising workflow")

        metric_warnings: list[str] = []

        def _warn_metric(msg: str) -> None:
            metric_warnings.append(str(msg))
            warnings.warn(str(msg), stacklevel=2)
            progress.warn("metrics", str(msg))

        metrics_cfg, metric_auto_defaults, metrics_summary = resolve_effective_metrics_config(
            config.autotune.metrics,
            structure_data=Path(initial_data),
            type_to_species=type_to_species,
            warn_fn=_warn_metric,
            context="autotune production",
        )
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")

        pot_cfg = config.kim

        rng = random.Random(config.random_seed)

        # preflight
        # preflight
        # buckingham stabilization overlay
        # thermostat parameter scan
        # progress info preflight
        progress.info("preflight", "starting preflight checks")
        preflight = run_preflight(runner, config, initial_data, outdir)

        # thermo dump frequencies
        # applying preflight selected
        md_use = config.md.model_copy(
            deep=True,
            update={
                "timestep": float(preflight.selected_timestep),
                "ensemble": str(preflight.selected_ensemble),
                "thermostat": ThermostatConfig(style=config.md.thermostat.style, tdamp=float(preflight.selected_tdamp)),
                "barostat": (
                    BarostatConfig(style=config.md.barostat.style, pdamp=float(preflight.selected_pdamp))
                    if str(preflight.selected_ensemble) == "npt" and preflight.selected_pdamp is not None
                    else config.md.barostat
                ),
            },
        )

        potential_lines = preflight.potential_lines

        def _maybe_elastic(
            stage_role: str,
            *,
            stage_dir: Path,
            structure_data: Path,
            input_data: Path,
            force_iso_context: bool,
        ) -> Optional[dict[str, Any]]:
            run_screen, strict, _cfg = should_run_elastic_screen(
                metrics_cfg,
                runner=runner,
                stage_role=stage_role,
                force_isotropic=bool(force_iso_context),
            )
            if not run_screen:
                return None
            try:
                return run_elastic_screen_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    structure_data=structure_data,
                    stage_dir=stage_dir,
                    potential_lines=potential_lines,
                    metrics_cfg=metrics_cfg,
                    force_isotropic=bool(force_iso_context),
                    input_data_for_affine_strain=input_data if bool(force_iso_context) else None,
                    outdir=outdir,
                )
            except Exception:
                if strict:
                    raise
                return None

        # progress info tm
        progress.info("tm_scan", "starting operational melting scan")
        # melting scan
        # tm cfg
        tm_cfg = config.autotune.tm_scan
        scan_dir = outdir / "tm_scan"
        ensure_dir(scan_dir)

        T_values = np.arange(tm_cfg.t_min, tm_cfg.t_max + 0.5 * tm_cfg.dT, tm_cfg.dT, dtype=float)

        # melt scan statistically
        # reliable isolated outliers
        tm_nrep = int(getattr(tm_cfg, 'replicates_per_temp', 1))
        if tm_nrep < 1:
            tm_nrep = 1

        tm_outcomes_all: list[StageOutcome] = []
        tm_by_T: dict[float, list[StageOutcome]] = {float(t): [] for t in T_values.tolist()}
        tm_end_data: list[Path] = []

        # scan ensemble overridden
        md_scan = md_use
        ens_override = getattr(tm_cfg, 'ensemble', None)
        if ens_override is not None:
            ens_override = str(ens_override).strip().lower()
            if ens_override in ('nvt','npt') and ens_override != str(md_use.ensemble):
                md_scan = md_use.model_copy(deep=True, update={"ensemble": ens_override})


        gr_enabled = bool(getattr(tm_cfg, "gr", None) and tm_cfg.gr.enabled)

        # physical equilibration sampling
        # timestep adjusted preflight
        dt_ref = float(config.md.timestep)
        dt_scan = float(md_scan.timestep)
        equil_steps = scale_steps_for_timestep(int(tm_cfg.equil_steps), dt_ref, dt_scan, min_steps=0)
        sample_steps = scale_steps_for_timestep(int(tm_cfg.sample_steps), dt_ref, dt_scan, min_steps=1)
        stride_steps = None
        if gr_enabled:
            stride_steps = scale_steps_for_timestep(int(tm_cfg.gr.stride), dt_ref, dt_scan, min_steps=1)

        # independent heating replicas
        for rep_id in range(1, tm_nrep + 1):
            prev_data = initial_data
            for k, T in enumerate(T_values):
                if tm_nrep == 1:
                    stage_name = f"tm_{int(round(T))}K"
                else:
                    stage_name = f"tm_{int(round(T))}K_rep{rep_id}"
                stage_dir = scan_dir / stage_name

                seed = rng.randrange(1, 2**31 - 1)

                stage = StageSpec(
                    name=stage_name,
                    input_data=prev_data,
                    output_data=stage_dir / "output.data",
                    temperature_start=float(T),
                    temperature_stop=float(T),
                    pressure=float(md_use.pressure),
                    equil_steps=int(equil_steps),
                    run_steps=int(sample_steps),
                    seed=int(seed),
                    replicate=None,
                    write_dump=bool(gr_enabled),
                    tail_dump_frames=int(tm_cfg.gr.frames) if gr_enabled else None,
                    tail_dump_stride=int(stride_steps) if gr_enabled else None,
                    msd_every=int(tm_cfg.msd_every),
                    sample_ensemble=(
                        "nvt"
                        if (getattr(tm_cfg, "sample_in_nvt", False) and md_scan.ensemble == "npt")
                        else None
                    ),
                )
                out = _stage_run(
                    runner,
                    pot_cfg,
                    md_scan,
                    stage,
                    stage_dir,
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                # g melt indicator
                if gr_enabled:
                    dump_path = stage_dir / f"{stage_name}.lammpstrj"
                    try:
                        cand = stage_dir / "traj.extxyz"
                        traj_path = cand if cand.exists() else dump_path
                        frames = read_last_frames_auto(traj_path, int(tm_cfg.gr.frames))
                        gr = compute_first_peak_gr(
                            frames,
                            r_max=float(tm_cfg.gr.r_max),
                            nbins=int(tm_cfg.gr.nbins),
                            smooth=int(tm_cfg.gr.smooth),
                            pair=tm_cfg.gr.pair,
                            type_to_species=type_to_species,
                            r_ignore_factor=float(tm_cfg.gr.r_ignore_factor),
                            r_search_factor=float(tm_cfg.gr.r_search_factor),
                        )
                        out = StageOutcome(
                            **{
                                **asdict(out),
                                "gr_peak_r": float(gr.peak_r),
                                "gr_peak_height": float(gr.peak_height),
                                "gr_peak_fwhm": float(gr.peak_fwhm),
                            }
                        )
                    except Exception:
                        # scan analysis fails
                        pass

                # downstream aggregation reporting
                out = StageOutcome(**{**asdict(out), "rep_id": int(rep_id)})

                tm_outcomes_all.append(out)
                tm_by_T[float(T)].append(out)
                prev_data = stage_dir / out.output_data  # output inside dir

            tm_end_data.append(Path(prev_data))

        # aggregate temperature statistics
        def _agg(x: list[float], *, clamp_nonneg: bool = False) -> tuple[float, float, float]:
            arr = np.asarray([float(v) for v in x], dtype=float)
            m = np.isfinite(arr)
            if int(np.sum(m)) == 0:
                return float('nan'), float('nan'), float('nan')
            vals = arr[m]
            if clamp_nonneg:
                vals = np.where(vals > 0.0, vals, 0.0)
            mu = float(np.mean(vals))
            med = float(np.median(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            return mu, se, med

        T = np.asarray([float(t) for t in T_values.tolist()], dtype=float)
        tm_summary: list[dict[str, Any]] = []
        D_mu: list[float] = []
        D_se: list[float] = []
        D_med: list[float] = []
        H_mu: list[float] = []
        H_se: list[float] = []
        H_med: list[float] = []
        W_mu: list[float] = []
        W_se: list[float] = []
        W_med: list[float] = []
        msd_mu: list[float] = []
        msd_se: list[float] = []
        msd_med: list[float] = []
        vol_mu: list[float] = []
        vol_se: list[float] = []
        vol_med: list[float] = []

        for t in T.tolist():
            reps = tm_by_T.get(float(t), [])
            Dm, Ds, Dmd = _agg([o.D for o in reps], clamp_nonneg=True)
            D_mu.append(Dm)
            D_se.append(Ds)
            D_med.append(Dmd)
            Hm, Hs, Hmd = _agg([o.gr_peak_height for o in reps])
            Wm, Ws, Wmd = _agg([o.gr_peak_fwhm for o in reps])
            H_mu.append(Hm)
            H_se.append(Hs)
            H_med.append(Hmd)
            W_mu.append(Wm)
            W_se.append(Ws)
            W_med.append(Wmd)
            mm, ms, mmd = _agg([o.msd_rms_last for o in reps], clamp_nonneg=True)
            vm, vs, vmd = _agg([o.vol_last for o in reps], clamp_nonneg=True)
            msd_mu.append(mm)
            msd_se.append(ms)
            msd_med.append(mmd)
            vol_mu.append(vm)
            vol_se.append(vs)
            vol_med.append(vmd)
            tm_summary.append(
                {
                    "T": float(t),
                    "nrep": int(len(reps)),
                    "D_mean": float(Dm),
                    "D_stderr": float(Ds),
                    "D_median": float(Dmd),
                    "gr_peak_height_mean": float(Hm),
                    "gr_peak_height_stderr": float(Hs),
                    "gr_peak_height_median": float(Hmd),
                    "gr_peak_fwhm_mean": float(Wm),
                    "gr_peak_fwhm_stderr": float(Ws),
                    "gr_peak_fwhm_median": float(Wmd),
                    "msd_rms_last_mean": float(mm),
                    "msd_rms_last_stderr": float(ms),
                    "msd_rms_last_median": float(mmd),
                    "vol_last_mean": float(vm),
                    "vol_last_stderr": float(vs),
                    "vol_last_median": float(vmd),
                }
            )

        # replicas representative selection
        D = np.asarray(D_med, dtype=float)

        # estimate diffusion combined
        if bool(getattr(tm_cfg, "gr", None) and tm_cfg.gr.enabled):
            H = np.asarray(H_med, dtype=float)
            W = np.asarray(W_med, dtype=float)
            msd_rms = np.asarray(msd_med, dtype=float)
            vol_last = np.asarray(vol_med, dtype=float)
            natoms = int(count_atoms_in_datafile(initial_data))
            tm_est = estimate_tm(
                T,
                D,
                gr_peak_height=H,
                gr_peak_fwhm=W,
                msd_rms_last=msd_rms,
                vol_last=vol_last,
                natoms=natoms,
                w_diffusion=float(tm_cfg.gr.w_diffusion),
                w_peak_height=float(tm_cfg.gr.w_peak_height),
                w_peak_fwhm=float(tm_cfg.gr.w_peak_fwhm),
                liquid_D_frac=float(getattr(tm_cfg, 'liquid_D_frac', 0.2)),
                liquid_top_k=int(getattr(tm_cfg, 'liquid_top_k', 3)),
                liquid_min_consecutive=int(getattr(tm_cfg, 'liquid_min_consecutive', 2)),
            )
        else:
            tm_est = estimate_tm_from_diffusion(T, D)

        # temperature melt operational
        T_high_base = float(tm_est.Tm)
        if getattr(tm_cfg, 'highT_mode', 'onset') == 'liquid' and hasattr(tm_est, 'T_liquid') and np.isfinite(tm_est.T_liquid):
            T_high_base = float(tm_est.T_liquid)
        T_high = float(T_high_base + config.autotune.highT.margin)

        # warn
        # inherently stability diagnostics
        # evaluated higher temperature
        try:
            t_scan_max = float(getattr(tm_cfg, 't_max', float('nan')))
            if math.isfinite(t_scan_max) and T_high > t_scan_max:
                warnings.warn(
                    f"Selected T_high={T_high:g} K exceeds tm_scan.t_max={t_scan_max:g} K. "
                    "Consider extending the scan range if unexpected instability occurs.",
                    stacklevel=2,
                )
        except Exception:
            pass

        # progress info high
        progress.info("highT", "starting high-temperature disordering")
        # high equilibration time
        # high cfg
        high_cfg = config.autotune.highT
        high_dir = outdir / "highT"
        ensure_dir(high_dir)

        # equilibration production ensemble
        # volume transient quench
        dt_mq = float(md_use.timestep)
        force_iso_active = bool(getattr(md_use, "force_isotropic", False))
        chunk_steps = scale_steps_for_timestep(int(high_cfg.chunk_steps), dt_ref, dt_mq, min_steps=1)
        min_total_steps = scale_steps_for_timestep(int(high_cfg.min_total_steps), dt_ref, dt_mq, min_steps=1)
        min_total_steps = extend_highT_steps_for_force_isotropic(int(min_total_steps), force_isotropic=force_iso_active)

        # continuous segment replica
        # chunk sizing equivalent
        # loop
        max_chunks = int(getattr(high_cfg, "max_chunks", 20))
        if max_chunks < 1:
            max_chunks = 1
        max_total_steps = int(max(int(min_total_steps), int(chunk_steps) * int(max_chunks)))
        max_total_steps = extend_highT_steps_for_force_isotropic(int(max_total_steps), force_isotropic=force_iso_active)

        # starting disordering scan
        # scan temperature structure
        start_pool: list[Path] = [Path(p) for p in tm_end_data if Path(p).exists()]
        if len(start_pool) == 0:
            start_pool = [Path(initial_data)]

        high_outcomes: list[StageOutcome] = []
        high_rep_summaries: list[dict[str, Any]] = []
        melt_pool: list[Path] = []
        high_steps: list[int] = []

        n_high_rep = int(getattr(high_cfg, 'replicates', 1))
        if n_high_rep < 1:
            n_high_rep = 1

        for rep_id in range(1, n_high_rep + 1):
            start_data = start_pool[(rep_id - 1) % len(start_pool)]
            if n_high_rep == 1:
                stage_name = "highT"
            else:
                stage_name = f"highT_rep{rep_id}"
            stage_dir = high_dir / stage_name
            seed = rng.randrange(1, 2**31 - 1)

            # coordinate msd equilibration
            # followed equilibrated volume
            msd_every_high = max(1, int(tm_cfg.msd_every))

            equil_steps_high = 0
            run_steps_high = int(max_total_steps)
            sample_ensemble_high = None
            if str(md_use.ensemble).lower() == "npt":
                sample_ensemble_high = "nvt"
                equil_steps_high = int(min_total_steps)
                run_steps_high = int(max_total_steps) - int(equil_steps_high)

            # sampling segment msd
            if int(run_steps_high) < 3:
                shift = 3 - int(run_steps_high)
                run_steps_high = 3
                equil_steps_high = max(0, int(equil_steps_high) - int(shift))

            if int(run_steps_high) < 3 * int(msd_every_high):
                msd_every_high = max(1, int(run_steps_high) // 3)

            total_steps_high = int(equil_steps_high) + int(run_steps_high)

            stage = StageSpec(
                name=stage_name,
                input_data=start_data,
                output_data=stage_dir / "output.data",
                temperature_start=T_high,
                temperature_stop=T_high,
                pressure=float(md_use.pressure),
                equil_steps=int(equil_steps_high),
                run_steps=int(run_steps_high),
                seed=int(seed),
                force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                replicate=None,
                write_dump=False,
                msd_every=int(msd_every_high),
                sample_ensemble=sample_ensemble_high,
            )
            out = _stage_run(
                runner,
                pot_cfg,
                md_use,
                stage,
                stage_dir,
                potential_lines=potential_lines,
                type_to_species=type_to_species,
            )
            out = StageOutcome(**{**asdict(out), "rep_id": int(rep_id)})
            high_outcomes.append(out)

            elastic_high = _maybe_elastic(
                "highT",
                stage_dir=stage_dir,
                structure_data=(stage_dir / out.output_data),
                input_data=Path(start_data),
                force_iso_context=bool(force_iso_active),
            )

            melt_data_rep = stage_dir / out.output_data
            melt_pool.append(Path(melt_data_rep))

            # spacing estimate volume
            if out.n_atoms > 0 and out.vol_last > 0:
                l = float((out.vol_last / out.n_atoms) ** (1.0 / 3.0))
            else:
                l = float("nan")
            thresh = float(high_cfg.rms_multiple) * float(l) if math.isfinite(l) else float("nan")

            # determine continuous msd
            disorder_step = int(total_steps_high)
            disorder_rms = float("nan")
            reached_rms = False
            try:
                msd = parse_msd_file(stage_dir / f"{stage_name}.msd.dat")
                steps = np.asarray(msd.step, dtype=int)
                total_steps = steps + int(equil_steps_high)
                rms = np.sqrt(np.maximum(0.0, np.asarray(msd.msd, dtype=float)))
                if math.isfinite(thresh) and thresh > 0:
                    m = (total_steps >= int(min_total_steps)) & (rms >= float(thresh))
                    if np.any(m):
                        j = int(np.where(m)[0][0])
                        disorder_step = int(total_steps[j])
                        disorder_rms = float(rms[j])
                        reached_rms = True
            except Exception:
                pass
            # stationarity diagnostic density
            dens_early = float('nan')
            dens_late = float('nan')
            dens_rel = float('nan')
            pe_early = float('nan')
            pe_late = float('nan')
            pe_rel = float('nan')
            stationarity_ok = False
            stationarity_density_segment = "sample"
            stationarity_pe_segment = "sample"
            try:
                # prefer parsing handle
                # particular equilibration followed
                # thermo table density
                # stationarity diagnostics density
                # segment pot sampling

                n_tbl = 1
                try:
                    tbls = parse_all_thermo_tables(stage_dir / 'log.lammps')
                    n_tbl = int(len(tbls))
                    thermo_equil = tbls[0].as_dict()
                    thermo_sample = tbls[-1].as_dict()
                except Exception:
                    thermo_sample = parse_thermo_csv(stage_dir / 'thermo.csv').as_dict()
                    thermo_equil = thermo_sample

                use_split = (
                    int(n_tbl) >= 2
                    and sample_ensemble_high is not None
                    and str(md_use.ensemble).strip().lower() == 'npt'
                    and str(sample_ensemble_high).strip().lower() == 'nvt'
                    and int(equil_steps_high) > 0
                )

                dens_series = (thermo_equil.get('Density', []) if use_split else thermo_sample.get('Density', []))
                pe_series = thermo_sample.get('PotEng', [])

                stationarity_density_segment = 'equil' if use_split else 'sample'
                stationarity_pe_segment = 'sample'

                dens_chg = early_late_change(dens_series, split_fraction=0.5, denom='late')
                pe_chg = early_late_change(pe_series, split_fraction=0.5, denom='late')
                dens_early = float(dens_chg.early_mean)
                dens_late = float(dens_chg.late_mean)
                dens_rel = float(dens_chg.rel_change)
                pe_early = float(pe_chg.early_mean)
                pe_late = float(pe_chg.late_mean)
                pe_rel = float(pe_chg.rel_change)
                tol = float(high_cfg.stationarity_tol)
                stationarity_ok = (
                    math.isfinite(dens_rel)
                    and math.isfinite(pe_rel)
                    and (dens_rel <= tol)
                    and (pe_rel <= tol)
                )
            except Exception:
                stationarity_ok = False

            # criterion triggers immediately
            disorder_step = int(max(int(min_total_steps), int(disorder_step)))
            high_steps.append(int(disorder_step))

            high_rep_summaries.append(
                {
                    "rep_id": int(rep_id),
                    "seed": int(seed),
                    "start_data": str(Path(start_data).relative_to(outdir)) if Path(start_data).is_relative_to(outdir) else str(start_data),
                    "equil_steps": int(equil_steps_high),
                    "sample_steps": int(run_steps_high),
                    "run_steps": int(total_steps_high),
                    "max_total_steps": int(max_total_steps),
                    "sample_ensemble": str(sample_ensemble_high) if sample_ensemble_high is not None else str(md_use.ensemble),
                    "min_total_steps": int(min_total_steps),
                    "disorder_step": int(disorder_step),
                    "reached_rms": bool(reached_rms),
                    "rms_threshold": float(thresh),
                    "rms_at_disorder": float(disorder_rms),
                    "spacing_l": float(l),
                    "density_mean": float(out.density_mean),
                    "density_stderr": float(out.density_stderr),
                    "density_early_mean": float(dens_early),
                    "density_late_mean": float(dens_late),
                    "density_rel_change": float(dens_rel),
                    "pe_mean": float(out.pe_mean),
                    "pe_stderr": float(out.pe_stderr),
                    "pe_early_mean": float(pe_early),
                    "pe_late_mean": float(pe_late),
                    "pe_rel_change": float(pe_rel),
                    "stationarity_ok": bool(stationarity_ok),
                    "stationarity_density_segment": str(stationarity_density_segment),
                    "stationarity_pe_segment": str(stationarity_pe_segment),
                    "melt_data": str(Path(melt_data_rep).relative_to(outdir)) if Path(melt_data_rep).is_relative_to(outdir) else str(melt_data_rep),
                    "elastic": elastic_high,
                }
            )

        # recommended disordering observed
        high_total_steps = int(max(high_steps) if len(high_steps) > 0 else int(min_total_steps))

        # conservative melt subsequent
        # longest time disorder
        idx_max: Optional[int] = None
        if len(melt_pool) > 0 and len(high_steps) > 0:
            idx_max = int(np.argmax(np.asarray(high_steps, dtype=int)))
            melt_data = melt_pool[idx_max]
        else:
            melt_data = Path(start_pool[-1])

        # stationarity summary enforcement
        stationarity_flags = [bool(r.get("stationarity_ok", False)) for r in high_rep_summaries]
        n_stat = int(len(stationarity_flags))
        n_ok = int(sum(1 for x in stationarity_flags if x))
        ok_fraction = float(n_ok) / float(n_stat) if n_stat > 0 else float("nan")

        dens_rel_arr = np.asarray([float(r.get("density_rel_change", float("nan"))) for r in high_rep_summaries], dtype=float)
        pe_rel_arr = np.asarray([float(r.get("pe_rel_change", float("nan"))) for r in high_rep_summaries], dtype=float)
        dens_rel_max = float(np.nanmax(dens_rel_arr)) if np.any(np.isfinite(dens_rel_arr)) else float("nan")
        pe_rel_max = float(np.nanmax(pe_rel_arr)) if np.any(np.isfinite(pe_rel_arr)) else float("nan")

        high_stationarity_summary = {
            "tol": float(high_cfg.stationarity_tol),
            "enforced": bool(getattr(high_cfg, "enforce_stationarity", False)),
            "ok_fraction": float(ok_fraction),
            "ok_count": int(n_ok),
            "n": int(n_stat),
            "density_rel_change_max": float(dens_rel_max),
            "pe_rel_change_max": float(pe_rel_max),
            "rep_id_max_disorder": int(high_rep_summaries[idx_max]["rep_id"]) if idx_max is not None and idx_max < len(high_rep_summaries) else None,
            "stationarity_ok_max_disorder": bool(high_rep_summaries[idx_max]["stationarity_ok"]) if idx_max is not None and idx_max < len(high_rep_summaries) else None,
        }

        if bool(getattr(high_cfg, "enforce_stationarity", False)):
            if idx_max is None or idx_max >= len(high_rep_summaries):
                raise ValueError("High-T stationarity enforcement requested, but no replicate summary is available")
            if not bool(high_rep_summaries[idx_max].get("stationarity_ok", False)):
                tol = float(high_cfg.stationarity_tol)
                dens_rel = float(high_rep_summaries[idx_max].get("density_rel_change", float("nan")))
                pe_rel = float(high_rep_summaries[idx_max].get("pe_rel_change", float("nan")))
                raise ValueError(
                    f"High-T stationarity check failed for the max-disorder replica (rep_id={high_rep_summaries[idx_max].get('rep_id')}). "
                    f"density_rel_change={dens_rel:g}, pe_rel_change={pe_rel:g}, tol={tol:g}. "
                    "Increase highT.min_total_steps/max_chunks or relax highT.stationarity_tol."
                )
        else:
            # warn
            if idx_max is not None and idx_max < len(high_rep_summaries):
                if not bool(high_rep_summaries[idx_max].get("stationarity_ok", False)):
                    tol = float(high_cfg.stationarity_tol)
                    dens_rel = float(high_rep_summaries[idx_max].get("density_rel_change", float("nan")))
                    pe_rel = float(high_rep_summaries[idx_max].get("pe_rel_change", float("nan")))
                    warnings.warn(
                        f"High-T stationarity check failed for the max-disorder replica (rep_id={high_rep_summaries[idx_max].get('rep_id')}). "
                        f"density_rel_change={dens_rel:g}, pe_rel_change={pe_rel:g}, tol={tol:g}. "
                        "Proceeding because highT.enforce_stationarity=false.",
                        stacklevel=2,
                    )

        # progress info rate
        progress.info("rate_scan", "starting quench-rate scan")
        # rate scan density
        # q cfg
        q_cfg = config.autotune.quench
        rates, time_unit_ps, rates_ps_sorted = resolve_quench_rates_K_per_time(config)
        rate_dir = outdir / "rates"
        ensure_dir(rate_dir)

        n_rep_rate = int(config.autotune.quench.replicates_per_rate)
        rate_results: list[dict[str, Any]] = []
        for idx_r, r in enumerate(rates):
            dens_reps: list[float] = []
            rep_entries: list[dict[str, Any]] = []
            r_ps = float("nan")
            if time_unit_ps is not None:
                r_ps = float(r) / float(time_unit_ps)
            elif rates_ps_sorted is not None and idx_r < len(rates_ps_sorted):
                r_ps = float(rates_ps_sorted[idx_r])
            # quench steps rate
            # important timestep yaml
            # timestep requested rate
            dT = T_high - q_cfg.t_final
            n_quench = quench_steps_for_rate(float(dT), float(r), float(dt_mq), min_steps=1)

            for rep in range(n_rep_rate):
                rtag = f"r{r:g}_rep{rep+1}"
                rdir = rate_dir / rtag
                ensure_dir(rdir)

                # quench once rate

                seed_melt = rng.randrange(1, 2**31 - 1)
                # diverse starting snapshots
                # correlation between replicates
                melt_seed_data = melt_data
                if len(melt_pool) > 0:
                    melt_seed_data = melt_pool[rng.randrange(0, len(melt_pool))]
                melt_stage = StageSpec(
                    name=f"melt_r{r:g}",
                    input_data=Path(melt_seed_data),
                    output_data=rdir / "melt.data",
                    temperature_start=T_high,
                    temperature_stop=T_high,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=int(high_total_steps),
                    seed=int(seed_melt),
                    force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                    replicate=None,
                    write_dump=False,
                    msd_every=int(tm_cfg.msd_every),
                )
                melt_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    melt_stage,
                    rdir / "melt",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                seed = rng.randrange(1, 2**31 - 1)
                quench_stage = StageSpec(
                    name=f"quench_r{r:g}",
                    input_data=(rdir / "melt" / melt_out.output_data),
                    output_data=rdir / "quench.data",
                    temperature_start=T_high,
                    temperature_stop=q_cfg.t_final,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=n_quench,
                    seed=int(seed),
                    replicate=None,
                    write_dump=False,
                    msd_every=int(tm_cfg.msd_every),
                )
                quench_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    quench_stage,
                    rdir / "quench",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )

                seed2 = rng.randrange(1, 2**31 - 1)
                mcfg = config.autotune.metrics
                relax_steps = scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
                relax_stage = StageSpec(
                    name=f"relax_r{r:g}",
                    input_data=(rdir / "quench" / quench_out.output_data),
                    output_data=rdir / "relax.data",
                    temperature_start=q_cfg.t_final,
                    temperature_stop=q_cfg.t_final,
                    pressure=float(md_use.pressure),
                    equil_steps=0,
                    run_steps=int(relax_steps),
                    seed=int(seed2),
                    replicate=None,
                    write_dump=True,
                    dump_every=int(relax_steps) if not mcfg.enabled else None,
                    tail_dump_frames=int(mcfg.time_average_frames) if mcfg.enabled else None,
                    tail_dump_stride=int(mcfg.time_average_stride) if mcfg.enabled else None,
                    msd_every=int(tm_cfg.msd_every),
                )
                relax_dir = rdir / "relax"
                relax_out = _stage_run(
                    runner,
                    pot_cfg,
                    md_use,
                    relax_stage,
                    relax_dir,
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )
                melt_elastic = _maybe_elastic(
                    "melt",
                    stage_dir=rdir / "melt",
                    structure_data=(rdir / "melt" / melt_out.output_data),
                    input_data=Path(melt_seed_data),
                    force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                )
                relax_elastic = _maybe_elastic(
                    "relax",
                    stage_dir=relax_dir,
                    structure_data=(relax_dir / relax_out.output_data),
                    input_data=(rdir / "quench" / quench_out.output_data),
                    force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                )
                dens_reps.append(relax_out.density_mean)
                rep_entries.append(
                    {
                        "density": float(relax_out.density_mean),
                        "n_quench_steps": int(n_quench),
                        "cooling_rate_K_per_time": float(r),
                        "cooling_rate_K_per_ps": float(r_ps) if math.isfinite(r_ps) else None,
                        "final_data": str((relax_dir / relax_out.output_data).relative_to(outdir))
                        if (relax_dir / relax_out.output_data).is_relative_to(outdir)
                        else str(relax_dir / relax_out.output_data),
                        "dump": str((relax_dir / f"{relax_stage.name}.lammpstrj").relative_to(outdir))
                        if (relax_dir / f"{relax_stage.name}.lammpstrj").is_relative_to(outdir)
                        else str(relax_dir / f"{relax_stage.name}.lammpstrj"),
                        "elastic_melt": melt_elastic,
                        "elastic_relax": relax_elastic,
                    }
                )

            dens_arr = np.array(dens_reps, dtype=float)
            mu = float(np.mean(dens_arr))
            se = float(np.std(dens_arr, ddof=1) / math.sqrt(len(dens_arr))) if len(dens_arr) > 1 else 0.0
            rate_results.append(
                {
                    "rate": float(r),
                    "rate_K_per_ps": float(r_ps) if math.isfinite(r_ps) else None,
                    "n_quench_steps": int(n_quench),
                    "density_mean": mu,
                    "density_stderr": se,
                    "nrep": len(dens_reps),
                    "replicates": rep_entries,
                }
            )

        # density structure metrics
        # effective metrics start
        type_to_species = _get_type_to_species(config)
        cutoffs_rate: dict[tuple[int, int], float] = {}
        rate_results_for_selection = list(rate_results)

        if metrics_cfg.enabled:
            required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
            fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

            cutoffs_rate, frames_by_path = _estimate_pooled_scan_cutoffs(
                rate_results,
                outdir=outdir,
                metrics_cfg=metrics_cfg,
                required_pairs=required_pairs,
                fixed_cutoffs=fixed_cut,
                type_to_species=type_to_species,
            )

            # replicate metrics rate
            for rr in rate_results:
                for rep_entry in rr["replicates"]:
                    traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
                    frames = frames_by_path.get(str(traj_path))
                    if frames is None:
                        frames = read_last_frames_auto(
                            traj_path,
                            int(metrics_cfg.time_average_frames),
                            type_to_species=type_to_species,
                        )
                        frames_by_path[str(traj_path)] = frames
                    sm = compute_structure_metrics_timeavg(frames, metrics_cfg, cutoffs=cutoffs_rate, type_to_species=type_to_species)
                    rep_entry["metrics"] = dict(sm.values)
                    if bool(getattr(getattr(metrics_cfg, "amorphous", None), "enabled", False)):
                        rep_entry["amorphous"] = analyse_amorphous_state(
                            frames,
                            metrics_cfg=metrics_cfg,
                            cutoffs=cutoffs_rate,
                            type_to_species=type_to_species,
                            cache_dir=(outdir / "amorphous_references"),
                            progress=progress,
                        )

            # aggregate rate
            for rr in rate_results:
                rep_metrics = [re.get("metrics", {}) for re in rr["replicates"]]
                mu_m, se_m = _aggregate_scalar_metrics(rep_metrics)
                rr["metrics_mean"] = mu_m
                rr["metrics_stderr"] = se_m

            amorph_cfg = getattr(metrics_cfg, "amorphous", None)
            if bool(getattr(amorph_cfg, "enabled", False)):
                rate_amorphous = summarize_rate_amorphous_acceptance(rate_results, amorph_cfg=amorph_cfg)
                for rr, summary in zip(rate_results, rate_amorphous):
                    rr["amorphous_summary"] = dict(summary)
                if bool(getattr(amorph_cfg, "enforce_during_rate_scan", False)):
                    rate_results_for_selection = [
                        rr for rr in rate_results if bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))
                    ]
                    rejected_rates = [
                        rr for rr in rate_results if not bool((rr.get("amorphous_summary", {}) or {}).get("accepted", False))
                    ]
                    for rr in rejected_rates:
                        summ = dict(rr.get("amorphous_summary", {}) or {})
                        crit_txt = _format_rate_amorphous_criteria_summary(summ)
                        msg = (
                            f"rate {float(rr['rate']):g}: rejected by amorphous gate "
                            f"(pass_fraction={float(summ.get('pass_fraction', float('nan'))):.3g}, required={float(summ.get('required_pass_fraction', float('nan'))):.3g})"
                        )
                        if crit_txt:
                            msg += f"; {crit_txt}"
                        progress.warn("rate_scan", msg)
                    if len(rate_results_for_selection) == 0:
                        failure_message = (
                            "No cooling rates satisfied the amorphous acceptance gate. "
                            "Relax amorphous thresholds or increase quench rates."
                        )
                        _write_rate_scan_failure_snapshot(
                            outdir=outdir,
                            config=config,
                            pot_cfg=pot_cfg,
                            kim_install=kim_install,
                            preflight=preflight,
                            T=T,
                            D=D,
                            D_mu=D_mu,
                            D_se=D_se,
                            D_med=D_med,
                            tm_cfg=tm_cfg,
                            tm_summary=tm_summary,
                            tm_outcomes_all=tm_outcomes_all,
                            tm_est=tm_est,
                            time_unit_ps=time_unit_ps,
                            T_high=T_high,
                            high_total_steps=high_total_steps,
                            force_iso_active=force_iso_active,
                            high_cfg=high_cfg,
                            high_stationarity_summary=high_stationarity_summary,
                            high_rep_summaries=high_rep_summaries,
                            high_outcomes=high_outcomes,
                            melt_pool=melt_pool,
                            melt_data=melt_data,
                            rate_results=rate_results,
                            cutoffs_rate=cutoffs_rate,
                            metric_warnings=metric_warnings,
                            metrics_summary=metrics_summary,
                            failure_message=failure_message,
                            progress=progress,
                        )
                        raise ValueError(failure_message)

        # density convergence rate
        if len(rate_results_for_selection) >= 2:
            decision_rate_density = choose_fastest_converged(
                [rr["rate"] for rr in rate_results_for_selection],
                [rr["density_mean"] for rr in rate_results_for_selection],
                [rr["density_stderr"] for rr in rate_results_for_selection],
                rel_tol=config.autotune.convergence.density_rel_tol,
                abs_tol=config.autotune.convergence.density_abs_tol,
                z=config.autotune.convergence.zscore,
                kind="rate",
            )
        else:
            rr0 = rate_results_for_selection[0]
            decision_rate_density = {
                "kind": "rate",
                "chosen_index": 0,
                "chosen_value": float(rr0["rate"]),
                "reference_value": float(rr0["density_mean"]),
                "deltas": [0.0],
                "allowed": [float("inf")],
                "passed": [True],
                "accepted_subset": True,
            }

        if metrics_cfg.enabled:
            # metric decision accepted
            if len(rate_results_for_selection) >= 2:
                x = [float(rr["rate"]) for rr in rate_results_for_selection]
                mu_maps: list[dict[str, float]] = []
                se_maps: list[dict[str, float]] = []
                for rr in rate_results_for_selection:
                    mu_map = {"density": float(rr["density_mean"])}
                    se_map = {"density": float(rr["density_stderr"])}
                    for k, v in rr.get("metrics_mean", {}).items():
                        mu_map[str(k)] = float(v)
                    for k, v in rr.get("metrics_stderr", {}).items():
                        se_map[str(k)] = float(v)
                    mu_maps.append(mu_map)
                    se_maps.append(se_map)

                decision_rate_multi = _multimetric_decision(x, mu_maps, se_maps, conv=config.autotune.convergence, kind="rate")
                chosen_rate = float(decision_rate_multi["chosen_value"])
            else:
                rr0 = rate_results_for_selection[0]
                decision_rate_multi = {
                    "kind": "rate",
                    "chosen_index": 0,
                    "chosen_value": float(rr0["rate"]),
                    "reference_index": 0,
                    "metrics": {},
                    "combined_passed": [True],
                    "accepted_subset": True,
                }
                chosen_rate = float(rr0["rate"])
        else:
            decision_rate_multi = None
            chosen_rate = float(decision_rate_density["chosen_value"] if isinstance(decision_rate_density, dict) else decision_rate_density.chosen_value)

        # progress info size
        progress.info("size_scan", "starting size scan")
        # box scan density
        # size cfg
        size_cfg = config.autotune.size
        size_dir = outdir / "sizes"
        ensure_dir(size_dir)

        # construct cell production
        # anisotropic expansions misleading
        size_base_data, initial_repeat = prepare_size_scan_base_structure(config, outdir, initial_data)
        base_natoms = count_atoms_in_datafile(size_base_data)

        engine_name = str(getattr(config, "engine", "lammps")).strip().lower()
        size_enabled = bool(getattr(size_cfg, "enabled", False))
        if engine_name == "cp2k":
            size_scan_skipped = True
            size_scan_reason = "cp2k engine: size scan disabled"
        elif not size_enabled:
            size_scan_skipped = True
            size_scan_reason = "autotune.size.enabled=false"
        else:
            size_scan_skipped = False
            size_scan_reason = None

        # containers skipping
        size_results: list[dict[str, Any]] = []
        decision_size_density = None
        decision_size_multi = None
        cutoffs_size: dict[tuple[int, int], float] = {}

        if size_scan_skipped:
            # scans disabled expensive
            # structure size downstream
            chosen_replicate = [int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2])]
        else:
            rx0, ry0, rz0 = (int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2]))
            n0 = 1
            if rx0 == ry0 == rz0 and rx0 >= 1:
                n0 = int(rx0)
            else:
                # approximation scan isotropic
                prod_rep = max(1, int(rx0 * ry0 * rz0))
                n0 = int(max(1, round(float(prod_rep) ** (1.0 / 3.0))))

            # autotune configured isotropic
            # additional isotropic increasing
            max_atoms = int(getattr(size_cfg, "max_atoms", 0) or 0)

            def _natoms_for_n(n: int) -> int:
                return int(base_natoms) * int(n) * int(n) * int(n)

            # interpret yaml multipliers
            # enforce isotropic replication
            n_candidates: list[int] = []
            for r in list(size_cfg.replicas):
                try:
                    mx, my, mz = (int(r[0]), int(r[1]), int(r[2]))
                except Exception:
                    continue
                if mx < 1 or my < 1 or mz < 1:
                    continue
                if mx != my or mx != mz:
                    # isotropic replicas alternatives
                    continue
                n_candidates.append(int(n0 * mx))

            if int(n0) not in n_candidates:
                n_candidates.insert(0, int(n0))

            # filter count always
            if max_atoms > 0 and base_natoms > 0:
                max_n = int(math.floor((float(max_atoms) / float(base_natoms)) ** (1.0 / 3.0)))
                max_n = max(1, int(max_n))
                n_filtered: list[int] = []
                for n in n_candidates:
                    n = int(n)
                    if n == int(n0):
                        n_filtered.append(n)
                        continue
                    if n <= max_n and _natoms_for_n(n) <= int(max_atoms):
                        n_filtered.append(n)
                n_candidates = n_filtered

                # isotropic sizes total
                if len(set(n_candidates)) < 2 and max_n > int(n0):
                    for n in range(int(n0) + 1, int(max_n) + 1):
                        n_candidates.append(int(n))
                        if len(set(n_candidates)) >= 3:
                            break

            # isotropic scan sorted
            n_list = sorted(set(int(n) for n in n_candidates if int(n) >= 1))
            replicas = [(int(n), int(n), int(n)) for n in n_list]

            size_results = []
            for repfac in replicas:
                nx, ny, nz = repfac
                dens_reps: list[float] = []
                rep_entries: list[dict[str, Any]] = []
                n_rep_size = int(getattr(size_cfg, "replicates_per_size", 1))
                if n_rep_size < 1:
                    n_rep_size = 1
                for rep in range(n_rep_size):
                    stag = f"size_{nx}x{ny}x{nz}_rep{rep+1}"
                    sdir = size_dir / stag
                    ensure_dir(sdir)

                    # melt total replicate
                    seed = rng.randrange(1, 2**31 - 1)
                    melt_stage = StageSpec(
                        name="melt",
                        input_data=size_base_data,
                        output_data=sdir / "melt.data",
                        temperature_start=T_high,
                        temperature_stop=T_high,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=int(high_total_steps),
                        seed=int(seed),
                        force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
                        replicate=(nx, ny, nz),
                        write_dump=False,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    melt_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        melt_stage,
                        sdir / "melt",
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )

                    # quench rate selected
                    dT = T_high - q_cfg.t_final
                    n_quench = quench_steps_for_rate(float(dT), float(chosen_rate), float(dt_mq), min_steps=1)

                    seed2 = rng.randrange(1, 2**31 - 1)
                    quench_stage = StageSpec(
                        name="quench",
                        input_data=(sdir / "melt" / melt_out.output_data),
                        output_data=sdir / "quench.data",
                        temperature_start=T_high,
                        temperature_stop=q_cfg.t_final,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=n_quench,
                        seed=int(seed2),
                        replicate=None,
                        write_dump=False,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    quench_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        quench_stage,
                        sdir / "quench",
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )

                    seed3 = rng.randrange(1, 2**31 - 1)
                    mcfg = config.autotune.metrics
                    relax_steps = scale_steps_for_timestep(int(q_cfg.relax_steps), dt_ref, dt_mq, min_steps=1)
                    relax_stage = StageSpec(
                        name="relax",
                        input_data=(sdir / "quench" / quench_out.output_data),
                        output_data=sdir / "final.data",
                        temperature_start=q_cfg.t_final,
                        temperature_stop=q_cfg.t_final,
                        pressure=float(md_use.pressure),
                        equil_steps=0,
                        run_steps=int(relax_steps),
                        seed=int(seed3),
                        replicate=None,
                        write_dump=True,
                        dump_every=int(relax_steps) if not mcfg.enabled else None,
                        tail_dump_frames=int(mcfg.time_average_frames) if mcfg.enabled else None,
                        tail_dump_stride=int(mcfg.time_average_stride) if mcfg.enabled else None,
                        msd_every=int(tm_cfg.msd_every),
                    )
                    relax_dir = sdir / "relax"
                    relax_out = _stage_run(
                        runner,
                        pot_cfg,
                        md_use,
                        relax_stage,
                        relax_dir,
                        potential_lines=potential_lines,
                        type_to_species=type_to_species,
                    )
                    melt_elastic = _maybe_elastic(
                        "melt",
                        stage_dir=sdir / "melt",
                        structure_data=(sdir / "melt" / melt_out.output_data),
                        input_data=Path(size_base_data),
                        force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                    )
                    relax_elastic = _maybe_elastic(
                        "relax",
                        stage_dir=relax_dir,
                        structure_data=(relax_dir / relax_out.output_data),
                        input_data=(sdir / "quench" / quench_out.output_data),
                        force_iso_context=bool(getattr(md_use, "force_isotropic", False)),
                    )
                    dens_reps.append(relax_out.density_mean)
                    rep_entries.append(
                        {
                            "density": float(relax_out.density_mean),
                            "n_atoms": int(_natoms_for_n(int(nx))) if base_natoms > 0 else None,
                            "final_data": str((relax_dir / relax_out.output_data).relative_to(outdir))
                            if (relax_dir / relax_out.output_data).is_relative_to(outdir)
                            else str(relax_dir / relax_out.output_data),
                            "dump": str((relax_dir / f"{relax_stage.name}.lammpstrj").relative_to(outdir))
                            if (relax_dir / f"{relax_stage.name}.lammpstrj").is_relative_to(outdir)
                            else str(relax_dir / f"{relax_stage.name}.lammpstrj"),
                            "elastic_melt": melt_elastic,
                            "elastic_relax": relax_elastic,
                        }
                    )

                dens_arr = np.array(dens_reps, dtype=float)
                mu = float(np.mean(dens_arr))
                se = float(np.std(dens_arr, ddof=1) / math.sqrt(len(dens_arr))) if len(dens_arr) > 1 else 0.0
                size_results.append(
                    {
                        "replicate": [nx, ny, nz],
                        "multiplier": int(nx * ny * nz),
                        "n_atoms": int(_natoms_for_n(int(nx))) if base_natoms > 0 else None,
                        "density_mean": mu,
                        "density_stderr": se,
                        "nrep": len(dens_reps),
                        "replicates": rep_entries,
                    }
                )

            # decide converged point
            # cell already atoms
            decision_size_density = None
            decision_size_multi = None
            cutoffs_size = {}

            if not size_results:
                chosen_multiplier = 1.0
                chosen_replicate = [int(n0), int(n0), int(n0)]
            elif len(size_results) < 2:
                chosen_multiplier = float(size_results[0].get("n_atoms") or size_results[0]["multiplier"])
                chosen_replicate = size_results[0]["replicate"]
            else:
                x_size = [float(sr.get("n_atoms") or sr["multiplier"]) for sr in size_results]
                decision_size_density = choose_fastest_converged(
                    x_size,
                    [sr["density_mean"] for sr in size_results],
                    [sr["density_stderr"] for sr in size_results],
                    rel_tol=config.autotune.convergence.density_rel_tol,
                    abs_tol=config.autotune.convergence.density_abs_tol,
                    z=config.autotune.convergence.zscore,
                    kind="size",
                )

                if metrics_cfg.enabled:
                    required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
                    fixed_cut = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)

                    cutoffs_size, frames_by_path = _estimate_pooled_scan_cutoffs(
                        size_results,
                        outdir=outdir,
                        metrics_cfg=metrics_cfg,
                        required_pairs=required_pairs,
                        fixed_cutoffs=fixed_cut,
                        type_to_species=type_to_species,
                    )

                    for sr in size_results:
                        for rep_entry in sr["replicates"]:
                            traj_path = _resolve_replicate_traj_path(outdir=outdir, rep_entry=rep_entry)
                            frames = frames_by_path.get(str(traj_path))
                            if frames is None:
                                frames = read_last_frames_auto(
                                    traj_path,
                                    int(metrics_cfg.time_average_frames),
                                    type_to_species=type_to_species,
                                )
                                frames_by_path[str(traj_path)] = frames
                            sm = compute_structure_metrics_timeavg(
                                frames,
                                metrics_cfg,
                                cutoffs=cutoffs_size,
                                type_to_species=type_to_species,
                            )
                            rep_entry["metrics"] = dict(sm.values)

                    for sr in size_results:
                        rep_metrics = [re.get("metrics", {}) for re in sr["replicates"]]
                        mu_m, se_m = _aggregate_scalar_metrics(rep_metrics)
                        sr["metrics_mean"] = mu_m
                        sr["metrics_stderr"] = se_m

                    x = [float(sr.get("n_atoms") or sr["multiplier"]) for sr in size_results]
                    mu_maps = []
                    se_maps = []
                    for sr in size_results:
                        mu_map = {"density": float(sr["density_mean"])}
                        se_map = {"density": float(sr["density_stderr"])}
                        for k, v in sr.get("metrics_mean", {}).items():
                            mu_map[str(k)] = float(v)
                        for k, v in sr.get("metrics_stderr", {}).items():
                            se_map[str(k)] = float(v)
                        mu_maps.append(mu_map)
                        se_maps.append(se_map)

                    decision_size_multi = _multimetric_decision(x, mu_maps, se_maps, conv=config.autotune.convergence, kind="size")
                    chosen_multiplier = float(decision_size_multi["chosen_value"])
                else:
                    chosen_multiplier = float(decision_size_density.chosen_value)

                chosen_replicate = None
                for sr in size_results:
                    xval = float(sr.get("n_atoms") or sr["multiplier"])
                    if float(xval) == float(chosen_multiplier):
                        chosen_replicate = sr["replicate"]
                        break
                if chosen_replicate is None:
                    chosen_replicate = size_results[-1]["replicate"]
        # rate conversion lammps
        cooling_rate_ps = None
        if time_unit_ps is not None:
            cooling_rate_ps = float(chosen_rate) / float(time_unit_ps)

        def _build_results(*, production_state: dict[str, Any], status: str) -> dict[str, Any]:
            return {
                "status": str(status),
                "units": {
                    "engine": str(getattr(config, "engine", "lammps")),
                    "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
                    "time_unit_ps": float(time_unit_ps) if time_unit_ps is not None else None,
                },
                "kim_install": asdict(kim_install) if kim_install is not None else None,
                "preflight": asdict(preflight),
                "tm_scan": {
                    "temps": [float(t) for t in T],
                    "replicates_per_temp": int(getattr(tm_cfg, 'replicates_per_temp', 1)),
                    "D": [float(x) for x in D],
                    "D_mean": [float(x) for x in D_mu],
                    "D_stderr": [float(x) for x in D_se],
                    "D_median": [float(x) for x in D_med],
                    "summary": tm_summary,
                    "outcomes": [asdict(o) for o in tm_outcomes_all],
                    "Tm_estimate": {
                        "Tm": float(tm_est.Tm),
                        "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                        "D_liquid_target": float(getattr(tm_est, "D_liquid_target", float("nan"))),
                        "method": str(tm_est.method),
                        "score": float(tm_est.score),
                        "idx": int(tm_est.idx),
                    },
                },
                "highT": {
                    "T_high": float(T_high),
                    "total_steps": int(high_total_steps),
                    "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
                    "replicates": int(getattr(high_cfg, 'replicates', 1)),
                    "stationarity": high_stationarity_summary,
                    "rep_summaries": high_rep_summaries,
                    "outcomes": [asdict(o) for o in high_outcomes],
                    "melt_pool": [
                        str(Path(p).relative_to(outdir)) if Path(p).is_relative_to(outdir) else str(p)
                        for p in melt_pool
                    ],
                    "melt_data": str(Path(melt_data).relative_to(outdir)) if Path(melt_data).is_relative_to(outdir) else str(melt_data),
                },
                "rate_scan": {
                    "rates": rate_results,
                    "decision_density": (dict(decision_rate_density) if isinstance(decision_rate_density, dict) else asdict(decision_rate_density)),
                    "decision_multi": decision_rate_multi,
                    "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_rate.items())],
                },
                "size_scan": {
                    "skipped": bool(size_scan_skipped),
                    "skip_reason": str(size_scan_reason) if size_scan_reason is not None else None,
                    "base_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "base_natoms": int(base_natoms),
                    "initial_repeat": [int(initial_repeat[0]), int(initial_repeat[1]), int(initial_repeat[2])],
                    "sizes": size_results,
                    "decision_density": asdict(decision_size_density) if decision_size_density is not None else None,
                    "decision_multi": decision_size_multi,
                    "cutoffs": [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(cutoffs_size.items())],
                },
                "production": production_state,
                "production_plan": production_plan,
                "recommendation": {
                    "T_high": T_high,
                    "Tm_operational": tm_est.Tm,
                    "T_liquid": float(getattr(tm_est, "T_liquid", float("nan"))),
                    "highT_steps": high_total_steps,
                    "force_isotropic_extension_factor": 1.5 if bool(force_iso_active) else 1.0,
                    "cooling_rate_K_per_time": chosen_rate,
                    "cooling_rate_K_per_ps": cooling_rate_ps,
                    "replicate": chosen_replicate,
                    "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
                    "t_final": q_cfg.t_final,
                    "pressure": config.md.pressure,
                    "md": {
                        "ensemble": str(md_use.ensemble),
                        "timestep": float(md_use.timestep),
                        "atom_style": str(md_use.atom_style),
                        "force_isotropic": bool(getattr(md_use, "force_isotropic", False)),
                        "thermostat": {"style": str(md_use.thermostat.style), "tdamp": float(md_use.thermostat.tdamp)},
                        "barostat": {"style": str(md_use.barostat.style), "pdamp": float(md_use.barostat.pdamp)},
                    },
                    "core_repulsion": asdict(preflight.core_repulsion),
                },
                "metric_warnings": list(metric_warnings),
                "effective_metrics": dict(metrics_summary),
                "paths": {
                    "autotune_results": "autotune_results.json",
                    "autotune": "autotune.json",
                    "condensed_log": "condensed.log",
                },
            }

        prod_cfg = getattr(config.autotune, "production", None)
        pre_n_quench_prod = quench_steps_for_rate(float(T_high - q_cfg.t_final), float(chosen_rate), float(dt_mq), min_steps=1)
        pre_relax_steps_prod = scale_steps_for_timestep(int(q_cfg.relax_steps), float(dt_ref), float(dt_mq), min_steps=1)
        pre_prod_cutoffs = dict(cutoffs_size) if len(cutoffs_size) > 0 else dict(cutoffs_rate)
        production_plan = production_plan_to_dict(
            make_production_plan(
                engine=str(getattr(config, "engine", "lammps")),
                structure_data=Path(size_base_data),
                T_high=float(T_high),
                high_total_steps=int(high_total_steps),
                t_final=float(q_cfg.t_final),
                chosen_rate=float(chosen_rate),
                cooling_rate_ps=(None if cooling_rate_ps is None else float(cooling_rate_ps)),
                replicate=chosen_replicate,
                pressure=float(md_use.pressure),
                md_use=md_use.model_dump(mode="json"),
                potential_config=(pot_cfg.model_dump(mode="json") if hasattr(pot_cfg, "model_dump") else None),
                potential_lines=potential_lines,
                core_repulsion=asdict(preflight.core_repulsion),
                type_to_species=type_to_species,
                metrics_cfg=metrics_cfg.model_dump(mode="json"),
                effective_metrics=dict(metrics_summary),
                production_cfg=(prod_cfg.model_dump(mode="json") if prod_cfg is not None else {}),
                convergence_cfg=config.autotune.convergence.model_dump(mode="json"),
                cutoffs_rate=cutoffs_rate,
                cutoffs_size=cutoffs_size,
                preferred_cutoffs=pre_prod_cutoffs,
                quench_steps=int(pre_n_quench_prod),
                relax_steps=int(pre_relax_steps_prod),
                msd_every=int(tm_cfg.msd_every),
                seed_base=int(config.random_seed) + 13579,
                time_unit_ps=(None if time_unit_ps is None else float(time_unit_ps)),
                sampling_hint=None,
                execution_mode=("adaptive" if bool(prod_cfg is not None and getattr(prod_cfg, "check_convergence", True)) else "fixed"),
                source_kind="autotune",
            ),
            relative_to=outdir,
        )
        preprod_warmup_start_temperature = 300.0
        preprod_warmup_duration_ps = 5.0
        preprod_warmup_steps = None
        if prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False)):
            preprod_warmup_start_temperature = resolve_production_warmup_start_temperature(
                prod_cfg=prod_cfg,
                T_high=float(T_high),
            )
            preprod_warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
            preprod_warmup_steps = resolve_production_warmup_steps(
                prod_cfg=prod_cfg,
                md_timestep=float(dt_mq),
                time_unit_ps=resolve_production_time_unit_ps(
                    config=config,
                    engine=str(getattr(config, "engine", "lammps") or "lammps"),
                    pot_cfg=pot_cfg,
                    time_unit_ps=time_unit_ps,
                ),
            )

        preprod_state = {
            "enabled": bool(prod_cfg is not None and bool(getattr(prod_cfg, "enabled", False))),
            "status": "starting",
            "error": None,
            "converged": False,
            "n_boxes": 0,
            "n_boxes_accepted": 0,
            "n_boxes_rejected": 0,
            "n_boxes_total": 0,
            "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)) if prod_cfg is not None else 0,
            "max_boxes": (int(getattr(prod_cfg, "max_boxes", 0)) if (prod_cfg is not None and getattr(prod_cfg, "max_boxes", None) not in (None, 0)) else None),
            "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)) if prod_cfg is not None else 1,
            "check_convergence": bool(getattr(prod_cfg, "check_convergence", True)) if prod_cfg is not None else False,
            "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)) if prod_cfg is not None else False,
            "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000) or 5000) if prod_cfg is not None else 5000,
            "rate_K_per_time": float(chosen_rate),
            "rate_K_per_ps": float(cooling_rate_ps) if cooling_rate_ps is not None else None,
            "replicate": [int(x) for x in chosen_replicate],
            "structure_data": str(Path(size_base_data).relative_to(outdir)) if Path(size_base_data).is_relative_to(outdir) else str(size_base_data),
            "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)) if prod_cfg is not None else False,
            "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")) if prod_cfg is not None else None,
            "rejects_dir": None,
            "warmup_start_temperature": float(preprod_warmup_start_temperature),
            "warmup_duration_ps": float(preprod_warmup_duration_ps),
            "warmup_steps": (int(preprod_warmup_steps) if preprod_warmup_steps is not None else None),
            "T_high": float(T_high),
            "t_final": float(q_cfg.t_final),
            "quench_steps": int(pre_n_quench_prod),
            "highT_steps": int(high_total_steps),
            "relax_steps": int(pre_relax_steps_prod),
            "cutoffs": ([{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(pre_prod_cutoffs.items())] if len(pre_prod_cutoffs) > 0 else None),
            "metrics_checked": None,
            "convergence_spec": None,
            "converged_md": False,
            "convergence_md": {},
            "converged_dft": None,
            "convergence_dft": None,
            "convergence": {},
            "dft_opt": None,
            "boxes_dft_final": None,
            "n_boxes_dft_accepted": None,
            "rejected_boxes_dft": None,
            "boxes": [],
            "rejected_boxes": [],
            "ensemble_dir": "production",
        }

        write_autotune_outputs(outdir, _build_results(production_state=preprod_state, status="running"))

        def _checkpoint_production(prod_state: dict[str, Any]) -> None:
            write_autotune_outputs(outdir, _build_results(production_state=dict(prod_state), status="running"))

        progress.info("production", "starting production ensemble")
        production = _run_production_ensemble(
            config=config,
            outdir=outdir,
            runner=runner,
            pot_cfg=pot_cfg,
            md_use=md_use,
            potential_lines=potential_lines,
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg,
            tm_cfg=tm_cfg,
            q_cfg=q_cfg,
            size_base_data=Path(size_base_data),
            chosen_replicate=[int(x) for x in chosen_replicate],
            chosen_rate=float(chosen_rate),
            dt_ref=float(dt_ref),
            dt_mq=float(dt_mq),
            cooling_rate_ps=(float(cooling_rate_ps) if cooling_rate_ps is not None else None),
            cutoffs_rate=dict(cutoffs_rate),
            cutoffs_size=dict(cutoffs_size),
            T_high=float(T_high),
            high_total_steps=int(high_total_steps),
            resume_state=None,
            progress=progress,
            checkpoint_cb=_checkpoint_production,
            time_unit_ps_override=(None if time_unit_ps is None else float(time_unit_ps)),
        )

        results = _build_results(production_state=production, status="ok")
        write_autotune_outputs(outdir, results)
        return results

def autotune(config: RunConfig, outdir: Path, *, resume: bool | None = None) -> dict[str, Any]:
    """Autotune."""
    return _AutotuneWorkflow(
        config=config,
        outdir=outdir,
        resume=resume,
    ).run()
