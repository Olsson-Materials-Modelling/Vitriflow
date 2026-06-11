from __future__ import annotations

import warnings
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import TypeAdapter

from ..config import (
    RunConfig,
    MDConfig,
    ThermostatConfig,
    BarostatConfig,
    PotentialConfig,
    StructureMetricsConfig,
    ProductionEnsembleConfig,
    ConvergenceConfig,
)
from ..kim import ensure_model_installed
from ..runner import Cp2kRunner, LammpsRunner
from ..structuregen import prepare_initial_structure
from ..utils import ensure_dir
from .metrics_policy import resolve_effective_metrics_config
from .preflight import run_preflight
from .progress import CondensedProgressLog, atomic_write_json
import importlib
from .quench_rates import quench_steps_for_rate, resolve_quench_rates_K_per_time
from .step_counts import extend_highT_steps_for_force_isotropic, resolve_lammps_units_style, resolve_md_pressure
from .elastic_screen import build_elastic_sampling_hint


def _get_type_to_species(config: RunConfig) -> Optional[list[str]]:
    m = config.autotune.metrics
    if m.type_to_species is not None:
        return list(m.type_to_species)
    if config.kim is not None and config.kim.interactions != "fixed_types":
        return list(config.kim.interactions)
    if getattr(config, "engine", "lammps") == "cp2k":
        raise ValueError("engine='cp2k' requires autotune.metrics.type_to_species")
    return None


def _model_dump_jsonlike(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _run_production_executor(**kwargs):
    from .autotune import _run_production_ensemble

    return _run_production_ensemble(**kwargs)


def _production_common_module():
    return importlib.import_module("vitriflow.workflows.production_common")


def _strip_distributions(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ent in entries:
        d = dict(ent)
        d.pop("distributions", None)
        out.append(d)
    return out


def _resolve_path_from_source(value: Any, *, base_dir: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute() and base_dir is not None:
        p = (Path(base_dir) / p).expanduser()
    return p.resolve(strict=False)


_POTENTIAL_ADAPTER = TypeAdapter(PotentialConfig)


def _potential_from_dict(data: Optional[Mapping[str, Any]], fallback: Any) -> Any:
    if not isinstance(data, Mapping) or len(data) == 0:
        return fallback
    return _POTENTIAL_ADAPTER.validate_python(dict(data))


def _fixed_production_cfg(base_cfg: ProductionEnsembleConfig, n_replicates: int) -> ProductionEnsembleConfig:
    n = max(1, int(n_replicates))
    data = _model_dump_jsonlike(base_cfg)
    data.update(
        {
            "enabled": True,
            "min_boxes": n,
            "max_boxes": n,
            "batch_boxes": n,
            "consecutive_converged_checks": min(max(1, int(data.get("consecutive_converged_checks", 1) or 1)), n),
        }
    )
    return ProductionEnsembleConfig.model_validate(data)


def _production_cfg_from_summary(
    base_cfg: ProductionEnsembleConfig,
    summary: Mapping[str, Any],
    *,
    fallback_n_replicates: Optional[int] = None,
) -> ProductionEnsembleConfig:
    if fallback_n_replicates is not None:
        return _fixed_production_cfg(base_cfg, fallback_n_replicates)

    data = _model_dump_jsonlike(base_cfg)
    data["enabled"] = bool(summary.get("enabled", True))
    for key in (
        "min_boxes",
        "max_boxes",
        "batch_boxes",
        "dump_trajectory",
        "dump_every_steps",
        "check_convergence",
        "consecutive_converged_checks",
        "bondlen_cdf_points",
        "angle_cdf_points",
        "store_distributions",
        "warmup_start_temperature",
        "warmup_duration_ps",
        "exclude_coordination_defects",
        "rejects_subdir",
    ):
        if key in summary and summary.get(key) is not None:
            data[key] = summary.get(key)
    dft_summary = summary.get("dft_opt", None)
    if isinstance(dft_summary, Mapping):
        dft_cfg = dict(_model_dump_jsonlike(base_cfg.dft_opt))
        dft_cfg["enabled"] = bool(dft_summary.get("enabled", False))
        for key in ("optimizer", "max_iter", "keep_angles", "external_pressure_bar", "traj_every", "print_level"):
            if key in dft_summary and dft_summary.get(key) is not None:
                dft_cfg[key] = dft_summary.get(key)
        data["dft_opt"] = dft_cfg
    return ProductionEnsembleConfig.model_validate(data)


def _legacy_md_use_and_potential_lines(config: RunConfig, source: Mapping[str, Any]) -> tuple[Optional[MDConfig], Optional[list[str]]]:
    rec = source.get("recommendation", source) if isinstance(source, Mapping) else {}
    if not isinstance(rec, Mapping):
        rec = {}
    pf = source.get("preflight", {}) if isinstance(source.get("preflight", {}), Mapping) else {}
    md_rec = rec.get("md", {}) if isinstance(rec.get("md", {}), Mapping) else {}

    timestep = md_rec.get("timestep", pf.get("selected_timestep", None))
    ensemble = md_rec.get("ensemble", pf.get("selected_ensemble", None))
    tdamp = (md_rec.get("thermostat", {}) or {}).get("tdamp", pf.get("selected_tdamp", None))
    pdamp = (md_rec.get("barostat", {}) or {}).get("pdamp", pf.get("selected_pdamp", None))
    force_isotropic = md_rec.get("force_isotropic", None)

    if timestep is None and ensemble is None and tdamp is None and pdamp is None and force_isotropic is None:
        return None, None

    updates: dict[str, Any] = {}
    if timestep is not None:
        updates["timestep"] = float(timestep)
    if ensemble is not None:
        updates["ensemble"] = str(ensemble)
    if tdamp is not None:
        updates["thermostat"] = config.md.thermostat.model_copy(update={"tdamp": float(tdamp)})
    if pdamp is not None and config.md.barostat is not None:
        updates["barostat"] = config.md.barostat.model_copy(update={"pdamp": float(pdamp)})
    if force_isotropic is not None:
        updates["force_isotropic"] = bool(force_isotropic)
    md_use = config.md.model_copy(update=updates) if updates else config.md

    potential_lines = pf.get("potential_lines", None)
    if potential_lines is not None:
        potential_lines = [str(x) for x in potential_lines]
    return md_use, potential_lines


def _legacy_rate_and_structure(config: RunConfig, source: Mapping[str, Any], base_dir: Optional[Path]) -> tuple[Optional[Path], dict[str, Any], dict[str, Any], Optional[Path]]:
    rec = source.get("recommendation", source) if isinstance(source, Mapping) else {}
    if not isinstance(rec, Mapping):
        rec = {}
    size_scan = source.get("size_scan", {}) if isinstance(source.get("size_scan", {}), Mapping) else {}
    production = source.get("production", {}) if isinstance(source.get("production", {}), Mapping) else {}

    structure_override = rec.get("structure_data", None)
    if structure_override is None and isinstance(size_scan.get("base_data", None), str):
        structure_override = size_scan.get("base_data")
    structure_path = _resolve_path_from_source(structure_override, base_dir=base_dir)
    return structure_path, dict(rec), dict(production), _resolve_path_from_source(size_scan.get("base_data", None), base_dir=base_dir)


def run_meltquench(
    config: RunConfig,
    outdir: Path,
    *,
    production_source: Optional[dict[str, Any]] = None,
    recommendation: Optional[dict[str, Any]] = None,
    recommendation_base_dir: Optional[Path] = None,
    n_replicates: int = 1,
    external_mode: str = "local",
    job_template: Optional[Path] = None,
    max_parallel_boxes: int = 1,
    resume: bool | None = None,
) -> dict[str, Any]:
    ensure_dir(outdir)
    progress = CondensedProgressLog(outdir / "condensed.log")
    progress.info("run", "initialising melt-quench workflow")

    metric_warnings: list[str] = []
    run_warnings: list[str] = []

    def _warn_metric(msg: str) -> None:
        metric_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("metrics", str(msg))

    def _warn_run(msg: str) -> None:
        run_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)
        progress.warn("run", str(msg))

    external_mode_norm = str(external_mode).strip().lower()
    if external_mode_norm not in {"local", "dry-run", "full-run"}:
        raise ValueError(f"Unsupported external_mode={external_mode!r}; expected one of local, dry-run, full-run")

    source = production_source if production_source is not None else recommendation
    production_common = _production_common_module()
    resume_state: Optional[dict[str, Any]] = None
    results_path = outdir / "run_results.json"
    do_resume = results_path.exists() if resume is None else (bool(resume) and results_path.exists())
    if do_resume:
        prev = json.loads(results_path.read_text())
        prev_prod = prev.get("production", None)
        if not isinstance(prev_prod, Mapping):
            raise RuntimeError("Cannot resume: existing run_results.json has no 'production' state")
        prev_status = str(prev_prod.get("status", prev.get("status", "")) or "").strip().lower()
        if prev_status == "ok":
            progress.info("run", "existing run_results.json is already complete; returning cached summary")
            return dict(prev)
        resume_state = dict(prev_prod)
        if isinstance(prev.get("metric_warnings", None), list):
            metric_warnings.extend(str(x) for x in prev.get("metric_warnings", []) or [])
        if isinstance(prev.get("run_warnings", None), list):
            run_warnings.extend(str(x) for x in prev.get("run_warnings", []) or [])
        prev_plan = production_common.production_plan_from_source(prev, base_dir=outdir)
        if prev_plan is not None:
            source = prev
            recommendation_base_dir = outdir
            progress.info(
                "run",
                "resuming production from existing run_results.json "
                f"({int(prev_prod.get('n_boxes_total', 0) or 0)} boxes already attempted)",
            )
        else:
            progress.warn(
                "run",
                "existing run_results.json has no stored production_plan; "
                "resuming with the current config/source",
            )

    plan = (
        None
        if source is None
        else production_common.production_plan_from_source(source, base_dir=recommendation_base_dir)
    )
    source_kind = "none"

    if plan is not None:
        source_kind = plan.source_kind
        type_to_species = list(plan.type_to_species) if plan.type_to_species is not None else _get_type_to_species(config)
        metrics_cfg = StructureMetricsConfig.model_validate(plan.metrics_cfg or _model_dump_jsonlike(config.autotune.metrics))
        metrics_summary = dict(plan.effective_metrics or {})
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")
        prod_cfg_override = ProductionEnsembleConfig.model_validate(plan.production_cfg)
        conv_cfg_override = ConvergenceConfig.model_validate(plan.convergence_cfg)
        md_use = MDConfig.model_validate(plan.md_use)
        q_cfg = config.autotune.quench.model_copy(update={"t_final": float(plan.t_final), "relax_steps": int(plan.relax_steps)})
        tm_cfg = config.autotune.tm_scan.model_copy(update={"msd_every": int(plan.msd_every)})
        pot_cfg = _potential_from_dict(plan.potential_config, config.kim)
        potential_lines = (None if plan.potential_lines is None else [str(x) for x in plan.potential_lines])
        cutoffs_rate = production_common.cutoffs_dict_from_any(plan.cutoffs_rate)
        cutoffs_size = production_common.cutoffs_dict_from_any(plan.preferred_cutoffs or plan.cutoffs_size or plan.cutoffs_rate)
        if int(n_replicates) != 1:
            _warn_run("run --n-replicates is ignored when an explicit production plan is supplied")
        engine = str(plan.engine).strip().lower() or str(getattr(config, "engine", "lammps"))
        if engine != str(getattr(config, "engine", "lammps")).strip().lower():
            _warn_run(f"production plan engine='{engine}' overrides config.engine='{config.engine}'")
    else:
        engine = str(getattr(config, "engine", "lammps")).strip().lower()
        type_to_species = _get_type_to_species(config)

        structure_data: Optional[Path] = None
        legacy_rec: dict[str, Any] = {}
        legacy_prod: dict[str, Any] = {}
        if isinstance(source, Mapping):
            structure_data, legacy_rec, legacy_prod, _legacy_size_base = _legacy_rate_and_structure(config, source, recommendation_base_dir)
        if structure_data is None:
            structure_data = Path(prepare_initial_structure(config, outdir))

        metrics_cfg, _auto_defaults, metrics_summary = resolve_effective_metrics_config(
            config.autotune.metrics,
            structure_data=Path(structure_data),
            type_to_species=type_to_species,
            warn_fn=_warn_metric,
            context="run production",
        )
        progress.info("metrics", f"effective metrics summary: {metrics_summary}")

        legacy_md_use, legacy_potential_lines = (
            _legacy_md_use_and_potential_lines(config, source) if isinstance(source, Mapping) else (None, None)
        )
        if legacy_md_use is not None:
            md_use = legacy_md_use
            potential_lines = legacy_potential_lines
            _warn_run("legacy autotune results do not carry a full production plan; replay uses stored recommendation/preflight state plus current metrics/convergence settings")
        else:
            runner_for_preflight = Cp2kRunner(config.cp2k) if engine == "cp2k" else LammpsRunner(config.lammps)
            preflight = run_preflight(runner_for_preflight, config, str(structure_data), outdir)
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

        rates_time, time_unit_ps, _rates_ps = resolve_quench_rates_K_per_time(config)
        default_rate_time = float(rates_time[-1])
        T_high = float(legacy_rec.get("T_high", config.autotune.tm_scan.t_max))
        high_steps = int(legacy_rec.get("highT_steps", config.autotune.highT.min_total_steps))
        t_final = float(legacy_prod.get("t_final", legacy_rec.get("t_final", config.autotune.quench.t_final)))
        if "cooling_rate_K_per_time" in legacy_rec:
            chosen_rate = float(legacy_rec.get("cooling_rate_K_per_time"))
        elif "rate_K_per_time" in legacy_prod:
            chosen_rate = float(legacy_prod.get("rate_K_per_time"))
        elif "cooling_rate_K_per_ps" in legacy_rec and time_unit_ps is not None:
            chosen_rate = float(legacy_rec.get("cooling_rate_K_per_ps")) * float(time_unit_ps)
        else:
            chosen_rate = float(default_rate_time)
        cooling_rate_ps = None if time_unit_ps is None else float(chosen_rate) / float(time_unit_ps)
        replicate = legacy_prod.get("replicate", legacy_rec.get("replicate", [1, 1, 1])) or [1, 1, 1]
        high_steps = extend_highT_steps_for_force_isotropic(
            int(high_steps),
            force_isotropic=bool(getattr(md_use, "force_isotropic", False)),
        )
        quench_sampling_hint = build_elastic_sampling_hint(
            Tm=legacy_rec.get("Tm_operational", legacy_rec.get("Tm", None)),
            freeze_temperature=legacy_rec.get("T_diffusion_freeze", legacy_rec.get("diffusion_freeze_temperature", None)),
            threshold_A2_per_ps=float(getattr(metrics_cfg.elastic, "diffusion_freeze_threshold_A2_per_ps", 0.1)),
        )
        quench_steps = int(legacy_prod.get("quench_steps", quench_steps_for_rate(float(T_high - t_final), float(chosen_rate), float(md_use.timestep), min_steps=1)))
        relax_steps = int(legacy_prod.get("relax_steps", config.autotune.quench.relax_steps))
        pot_cfg = config.kim
        cutoffs_rate = production_common.cutoffs_dict_from_any((source or {}).get("rate_scan", {}).get("cutoffs", None) if isinstance(source, Mapping) else None)
        cutoffs_size = production_common.cutoffs_dict_from_any((source or {}).get("size_scan", {}).get("cutoffs", None) if isinstance(source, Mapping) else None)
        preferred_cutoffs = production_common.cutoffs_dict_from_any(legacy_prod.get("cutoffs", None))
        if len(preferred_cutoffs) > 0:
            cutoffs_size = dict(preferred_cutoffs)
        if isinstance(source, Mapping) and ("production" in source):
            prod_cfg_override = _production_cfg_from_summary(config.autotune.production, legacy_prod)
            source_kind = "autotune_legacy"
        else:
            prod_cfg_override = _fixed_production_cfg(config.autotune.production, n_replicates)
            source_kind = "live_config"
        conv_cfg_override = ConvergenceConfig.model_validate(_model_dump_jsonlike(config.autotune.convergence))
        plan = production_common.make_production_plan(
            engine=engine,
            structure_data=Path(structure_data),
            T_high=float(T_high),
            high_total_steps=int(high_steps),
            t_final=float(t_final),
            chosen_rate=float(chosen_rate),
            cooling_rate_ps=cooling_rate_ps,
            replicate=replicate,
            pressure=float(md_use.pressure),
            md_use=md_use.model_dump(mode="json"),
            potential_config=(_model_dump_jsonlike(pot_cfg) if pot_cfg is not None else None),
            potential_lines=potential_lines,
            core_repulsion=_model_dump_jsonlike(getattr(pot_cfg, "core_repulsion", None)),
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg.model_dump(mode="json"),
            effective_metrics=dict(metrics_summary),
            production_cfg=prod_cfg_override.model_dump(mode="json"),
            convergence_cfg=conv_cfg_override.model_dump(mode="json"),
            cutoffs_rate=cutoffs_rate,
            cutoffs_size=cutoffs_size,
            preferred_cutoffs=(cutoffs_size if len(cutoffs_size) > 0 else cutoffs_rate),
            quench_steps=int(quench_steps),
            relax_steps=int(relax_steps),
            msd_every=int(config.autotune.tm_scan.msd_every),
            seed_base=int(config.random_seed) + 13579,
            time_unit_ps=time_unit_ps,
            sampling_hint=quench_sampling_hint,
            execution_mode=("adaptive" if isinstance(source, Mapping) and ("production" in source) else "fixed"),
            source_kind=source_kind,
        )

    if plan is None:
        raise RuntimeError("failed to construct a production plan")

    engine = str(plan.engine).strip().lower()
    pot_cfg = config.kim
    runner = None
    if engine == "cp2k":
        if external_mode_norm == "local":
            runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
    else:
        if plan.potential_config is not None:
            pot_cfg = _potential_from_dict(plan.potential_config, config.kim)
        if external_mode_norm == "local":
            if pot_cfg is not None:
                ensure_model_installed(getattr(pot_cfg, "model", ""))
            runner = LammpsRunner(config.lammps)

    if bool(getattr(md_use, "force_isotropic", False)) and engine != "lammps":
        raise ValueError("md.force_isotropic is currently supported only for engine='lammps'")

    progress.info("run", f"executing shared production plan ({plan.execution_mode}; source={source_kind})")
    if isinstance(source, Mapping) and isinstance(source.get("production", None), Mapping):
        rec_conv = source.get("production", {}).get("convergence", None)
        if isinstance(rec_conv, Mapping):
            progress.convergence("recommendation", dict(rec_conv))

    q_cfg_exec = config.autotune.quench.model_copy(update={"t_final": float(plan.t_final), "relax_steps": int(plan.relax_steps)})
    tm_cfg_exec = config.autotune.tm_scan.model_copy(update={"msd_every": int(plan.msd_every)})
    store_distributions = bool(getattr(prod_cfg_override, "store_distributions", True))

    def _build_summary(prod_state: dict[str, Any]) -> dict[str, Any]:
        all_entries = list(prod_state.get("boxes", [])) + list(prod_state.get("rejected_boxes", []))
        rep_entries = all_entries if store_distributions else _strip_distributions(all_entries)
        return {
            "status": str(prod_state.get("status", "ok")),
            "parameters": {
                "engine": str(engine),
                "lammps_units": resolve_lammps_units_style(config, pot_cfg=pot_cfg, default="metal"),
                "time_unit_ps": (None if plan.time_unit_ps is None else float(plan.time_unit_ps)),
                "warmup_start_temperature": float(getattr(prod_cfg_override, "warmup_start_temperature", 300.0)),
                "warmup_duration_ps": float(getattr(prod_cfg_override, "warmup_duration_ps", 5.0)),
                "warmup_steps": (int(prod_state.get("warmup_steps")) if prod_state.get("warmup_steps", None) is not None else None),
                "T_high": float(plan.T_high),
                "highT_steps": int(plan.high_total_steps),
                "t_final": float(plan.t_final),
                "cooling_rate_K_per_time": float(plan.chosen_rate),
                "cooling_rate_K_per_ps": (None if plan.cooling_rate_ps is None else float(plan.cooling_rate_ps)),
                "n_quench_steps": int(plan.quench_steps),
                "replicate": [int(plan.replicate[0]), int(plan.replicate[1]), int(plan.replicate[2])],
                "pressure": float(plan.pressure),
                "md": dict(plan.md_use),
                "seed_base": int(plan.seed_base),
                "execution_mode": str(plan.execution_mode),
            },
            "replicates": rep_entries,
            "production": prod_state,
            "cutoffs": list(plan.preferred_cutoffs),
            "metric_warnings": list(metric_warnings),
            "run_warnings": list(run_warnings),
            "effective_metrics": dict(plan.effective_metrics or {}),
            "production_plan": production_common.production_plan_to_dict(plan, relative_to=outdir),
            "paths": {"condensed_log": "condensed.log", "run_results": "run_results.json"},
        }

    def _checkpoint(prod_state: dict[str, Any]) -> None:
        atomic_write_json(outdir / "run_results.json", _build_summary(dict(prod_state)))

    if external_mode_norm == "local":
        production = _run_production_executor(
            config=config,
            outdir=outdir,
            runner=runner,
            pot_cfg=pot_cfg,
            md_use=md_use,
            potential_lines=(None if plan.potential_lines is None else [str(x) for x in plan.potential_lines]),
            type_to_species=type_to_species,
            metrics_cfg=metrics_cfg,
            tm_cfg=tm_cfg_exec,
            q_cfg=q_cfg_exec,
            size_base_data=Path(plan.structure_data),
            chosen_replicate=[int(plan.replicate[0]), int(plan.replicate[1]), int(plan.replicate[2])],
            chosen_rate=float(plan.chosen_rate),
            dt_ref=float(md_use.timestep),
            dt_mq=float(md_use.timestep),
            cooling_rate_ps=plan.cooling_rate_ps,
            cutoffs_rate=production_common.cutoffs_dict_from_any(plan.cutoffs_rate),
            cutoffs_size=production_common.cutoffs_dict_from_any(plan.preferred_cutoffs or plan.cutoffs_size or plan.cutoffs_rate),
            T_high=float(plan.T_high),
            high_total_steps=int(plan.high_total_steps),
            resume_state=resume_state,
            sampling_hint=(None if plan.sampling_hint is None else dict(plan.sampling_hint)),
            progress=progress,
            checkpoint_cb=_checkpoint,
            pressure_override=float(plan.pressure),
            seed_base=int(plan.seed_base),
            time_unit_ps_override=(None if plan.time_unit_ps is None else float(plan.time_unit_ps)),
            prod_cfg_override=prod_cfg_override,
            conv_cfg_override=conv_cfg_override,
            quench_steps_override=int(plan.quench_steps),
            relax_steps_override=int(plan.relax_steps),
        )
    else:
        from .hpc import dry_run_external_production, full_run_external_production

        plan_dict = production_common.production_plan_to_dict(plan, relative_to=outdir)
        progress.info("external", f"dispatching production via {external_mode_norm}")
        if external_mode_norm == "dry-run":
            production = dry_run_external_production(
                config=config,
                outdir=outdir,
                plan=plan_dict,
                job_template=job_template,
                progress=progress,
            )
        else:
            production = full_run_external_production(
                config=config,
                outdir=outdir,
                plan=plan_dict,
                job_template=job_template,
                max_parallel_boxes=max(1, int(max_parallel_boxes)),
                progress=progress,
            )

    summary = _build_summary(dict(production))
    atomic_write_json(outdir / "run_results.json", summary)
    progress.info("run", "workflow complete; wrote run_results.json")
    return summary
