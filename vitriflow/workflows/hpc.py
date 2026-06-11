from __future__ import annotations

"""External/HPC-compatible production task orchestration.

The external layer keeps the *planning* and *analysis* steps in vitriflow while
allowing production boxes to be executed independently from task manifests.

Modes
-----
- dry-run: create box directories, input snapshots, task manifests, preview
  inputs, and optional Slurm submission scripts without executing MD.
- full-run: create task manifests and execute them in batched waves, then run
  the generic output-analysis pipeline to compute metrics and convergence.
"""

import hashlib
import json
import random
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import TypeAdapter

from ..config import MDConfig, PotentialConfig, ProductionEnsembleConfig, RunConfig, StructureMetricsConfig
from ..kim import ensure_model_installed
from ..lammps_input import StageSpec, render_continuous_stages, render_stage
from ..potential import prepare_potential_files
from ..runner import Cp2kRunner, LammpsRunner
from ..utils import ensure_dir
from .metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
from .output_analysis import analyze_output_data
from .production_common import (
    plan_production_stage_diagnostics,
    production_plan_to_dict,
    resolve_production_relax_dump_settings,
    resolve_production_time_unit_ps,
    resolve_production_warmup_duration_ps,
    resolve_production_warmup_start_temperature,
    resolve_production_warmup_steps,
)
from .progress import CondensedProgressLog, atomic_write_json
from .stage_runner import run_stage_local, run_stages_continuous_lammps, stage_outcome_from_artifacts


_POTENTIAL_ADAPTER = TypeAdapter(PotentialConfig)


def _potential_from_dict(data: Optional[Mapping[str, Any]], fallback: Any) -> Any:
    if not isinstance(data, Mapping) or len(data) == 0:
        return fallback
    return _POTENTIAL_ADAPTER.validate_python(dict(data))


def _relpath_or_str(path: Path | None, base: Path) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def _task_manifest_digest(task_data: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(task_data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _copy_input_snapshot(src: Path, dst: Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return dst
            except Exception:
                pass
            try:
                if src.stat().st_size == dst.stat().st_size and src.read_bytes() == dst.read_bytes():
                    return dst
            except Exception:
                pass
    except Exception:
        pass

    tmp = dst.with_name(dst.name + '.tmp')
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    except Exception:
        pass
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return dst


def _box_seed_map(seed_base: int, box_id: int) -> dict[str, int]:
    if int(box_id) < 1:
        raise ValueError("box_id must be >= 1")
    rng = random.Random(int(seed_base))
    for _ in range((int(box_id) - 1) * 4):
        rng.randrange(1, 2**31 - 1)
    return {
        "warmup": int(rng.randrange(1, 2**31 - 1)),
        "melt": int(rng.randrange(1, 2**31 - 1)),
        "quench": int(rng.randrange(1, 2**31 - 1)),
        "relax": int(rng.randrange(1, 2**31 - 1)),
    }


def _planned_box_count(prod_cfg: ProductionEnsembleConfig) -> int:
    target = max(1, int(getattr(prod_cfg, "min_boxes", 1) or 1))
    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    if max_boxes_raw is None:
        return int(target)
    try:
        max_boxes = int(max_boxes_raw)
    except Exception:
        return int(target)
    if max_boxes <= 0:
        return int(target)
    return int(max(target, max_boxes))


def _analysis_metrics_cfg(plan: Mapping[str, Any]) -> StructureMetricsConfig:
    metrics_cfg = StructureMetricsConfig.model_validate(plan.get("metrics_cfg", {}))
    elastic_cfg = getattr(metrics_cfg, "elastic", None)
    if elastic_cfg is not None and hasattr(elastic_cfg, "model_copy"):
        elastic_cfg = elastic_cfg.model_copy(update={"enabled": False})
    return metrics_cfg.model_copy(
        deep=True,
        update={
            "collect_during_production_stages": False,
            "stage_timeseries_make_plot": False,
            "elastic": elastic_cfg,
        },
    )


def _type_to_species_from_plan(config: RunConfig, plan: Mapping[str, Any]) -> Optional[list[str]]:
    val = plan.get("type_to_species", None)
    if val is not None:
        return [str(x) for x in val]
    metrics = config.autotune.metrics
    if metrics.type_to_species is not None:
        return [str(x) for x in metrics.type_to_species]
    pot = getattr(config, "kim", None)
    interactions = getattr(pot, "interactions", None)
    if interactions is not None and interactions != "fixed_types":
        return [str(x) for x in interactions]
    return None


def _stage_specs_for_box(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    box_dir: Path,
    input_snapshot: Path,
) -> tuple[list[StageSpec], dict[str, Any], Any, Any, Optional[list[str]], Optional[list[str]], bool, str]:
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    warmup_start_temperature = resolve_production_warmup_start_temperature(
        prod_cfg=prod_cfg,
        T_high=float(plan.get("T_high")),
    )
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    metrics_cfg = _analysis_metrics_cfg(plan)
    potential_lines = plan.get("potential_lines", None)
    if potential_lines is not None:
        potential_lines = [str(x) for x in potential_lines]
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    type_to_species = _type_to_species_from_plan(config, plan)

    if engine == "cp2k":
        runner = Cp2kRunner(config.cp2k)  # type: ignore[arg-type]
    else:
        runner = LammpsRunner(config.lammps)

    T_high = float(plan.get("T_high"))
    warmup_time_unit_ps = resolve_production_time_unit_ps(
        config=config,
        engine=engine,
        pot_cfg=pot_cfg,
        time_unit_ps=plan.get("time_unit_ps", None),
    )
    t_final = float(plan.get("t_final"))
    quench_steps = int(plan.get("quench_steps"))
    relax_steps = int(plan.get("relax_steps"))
    high_total_steps = int(plan.get("high_total_steps"))
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=warmup_time_unit_ps,
    )
    replicate = [int(x) for x in plan.get("replicate", [1, 1, 1])]
    if len(replicate) != 3:
        raise ValueError("production plan replicate must have length 3")
    md_pressure = float(plan.get("pressure"))
    melt_force_iso = bool(getattr(md_use, "force_isotropic", False))
    cont = str(getattr(md_use, "stage_continuity", "discontinuous")).strip().lower()
    vel_next = "preserve" if cont == "continuous" else "create"
    stage_diag = plan_production_stage_diagnostics(
        prod_cfg=prod_cfg,
        metrics_cfg=metrics_cfg,
        runner=runner,
        force_isotropic=bool(melt_force_iso),
        total_quench_steps=int(quench_steps),
        temperature_start=float(T_high),
        temperature_stop=float(t_final),
        sampling_hint=(plan.get("sampling_hint", None) if isinstance(plan.get("sampling_hint", None), Mapping) else None),
    )
    need_stage_dump = dict(stage_diag.get("need_stage_dump", {}))
    dump_every = int(stage_diag.get("dump_every", getattr(prod_cfg, "dump_every_steps", 5000) or 5000))
    quench_dump_every = int(stage_diag.get("quench_dump_every", dump_every))
    msd_every = int(plan.get("msd_every", config.autotune.tm_scan.msd_every))
    relax_dump_settings = resolve_production_relax_dump_settings(stage_diag=stage_diag, metrics_cfg=metrics_cfg)

    seeds = _box_seed_map(int(plan.get("seed_base", config.random_seed + 13579)), int(box_dir.name.split("_")[-1]))

    warmup_stage = StageSpec(
        name="warmup",
        input_data=Path(input_snapshot),
        output_data=box_dir / "warmup.data",
        temperature_start=float(warmup_start_temperature),
        temperature_stop=float(T_high),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(warmup_steps),
        seed=int(seeds["warmup"]),
        velocity_mode="create",
        force_isotropic=bool(melt_force_iso),
        replicate=(int(replicate[0]), int(replicate[1]), int(replicate[2])),
        write_dump=bool(need_stage_dump.get("melt", True)),
        dump_every=int(dump_every) if bool(need_stage_dump.get("melt", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    melt_stage = StageSpec(
        name="melt",
        input_data=box_dir / "warmup" / "warmup.data",
        output_data=box_dir / "melt.data",
        temperature_start=float(T_high),
        temperature_stop=float(T_high),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(high_total_steps),
        seed=int(seeds["melt"]),
        velocity_mode="preserve",
        force_isotropic=False,
        replicate=None,
        write_dump=bool(need_stage_dump.get("melt", True)),
        dump_every=int(dump_every) if bool(need_stage_dump.get("melt", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    quench_stage = StageSpec(
        name="quench",
        input_data=box_dir / "melt" / "melt.data",
        output_data=box_dir / "quench.data",
        temperature_start=float(T_high),
        temperature_stop=float(t_final),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(quench_steps),
        seed=int(seeds["quench"]),
        velocity_mode=("preserve" if cont == "continuous" else vel_next),
        replicate=None,
        write_dump=bool(need_stage_dump.get("quench", True)),
        dump_every=int(quench_dump_every) if bool(need_stage_dump.get("quench", True)) else None,
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    relax_stage = StageSpec(
        name="relax",
        input_data=box_dir / "quench" / "quench.data",
        output_data=box_dir / "relax.data",
        temperature_start=float(t_final),
        temperature_stop=float(t_final),
        pressure=float(md_pressure),
        equil_steps=0,
        run_steps=int(relax_steps),
        seed=int(seeds["relax"]),
        velocity_mode=("preserve" if cont == "continuous" else vel_next),
        replicate=None,
        write_dump=bool(relax_dump_settings["write_dump"]),
        dump_every=relax_dump_settings["dump_every"],
        tail_dump_frames=relax_dump_settings["tail_dump_frames"],
        tail_dump_stride=relax_dump_settings["tail_dump_stride"],
        msd_every=int(msd_every),
        potential_lines=potential_lines,
    )
    return [warmup_stage, melt_stage, quench_stage, relax_stage], stage_diag, runner, pot_cfg, type_to_species, potential_lines, bool(melt_force_iso), cont


def _render_preview_inputs(
    *,
    config: RunConfig,
    plan: Mapping[str, Any],
    box_dir: Path,
    stages: Sequence[StageSpec],
    potential_cfg: Any,
    stage_continuity: str,
) -> list[str]:
    preview_dir = Path(box_dir) / "preview"
    ensure_dir(preview_dir)
    written: list[str] = []
    engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()

    if engine == "lammps":
        if potential_cfg is not None:
            prepare_potential_files(potential_cfg, preview_dir, plan.get("potential_lines", None))
        md_use = MDConfig.model_validate(plan.get("md_use", {}))
        if stage_continuity == "continuous":
            script = render_continuous_stages(
                potential_cfg,
                md_use,
                stages,
                stage_dir_prefixes={s.name: s.name for s in stages},
                log_name="log.lammps",
            )
            path = preview_dir / "continuous.in.lammps"
            path.write_text(script)
            written.append(str(path))
        else:
            for stage in stages:
                path = preview_dir / f"{stage.name}.in.lammps"
                path.write_text(render_stage(potential_cfg, md_use, stage))
                written.append(str(path))
    else:
        # preview limited execution
        # segmentation depends backend

        def _jsonable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        preview = {
            "schema": "vitriflow.preview_stage_specs.v1",
            "engine": engine,
            "stages": [_jsonable(asdict(stage)) for stage in stages],
            "note": "Exact CP2K native input files are materialised during task execution.",
        }
        path = preview_dir / "stage_specs.json"
        path.write_text(json.dumps(preview, indent=2))
        written.append(str(path))
    return written


def _render_template(template_text: str, mapping: Mapping[str, Any]) -> str:
    out = str(template_text)
    for key, value in mapping.items():
        sval = str(value)
        out = out.replace(f"{{{{{key}}}}}", sval)
        out = out.replace(f"${{{key}}}", sval)
    return out


def _write_submission_script(task_json: Path, template_path: Optional[Path]) -> Optional[Path]:
    if template_path is None:
        return None
    template = Path(template_path)
    text = template.read_text()
    box_dir = Path(task_json).parent
    box_id = int(box_dir.name.split("_")[-1])
    mapping = {
        "TASK_JSON": str(task_json),
        "TASK_DIR": str(box_dir),
        "BOX_ID": str(box_id),
        "JOB_NAME": f"vitriflow_box_{box_id:03d}",
        "EXECUTE_CMD": f"vitriflow execute-task --task {task_json}",
    }
    rendered = _render_template(text, mapping)
    submit_path = box_dir / "submit.slurm"
    submit_path.write_text(rendered)
    return submit_path


def _task_record(base_dir: Path, task_json: Path, task_result: Path, submit_script: Optional[Path]) -> dict[str, Any]:
    box_dir = task_json.parent
    return {
        "box": int(box_dir.name.split("_")[-1]),
        "box_dir": _relpath_or_str(box_dir, base_dir),
        "task_json": _relpath_or_str(task_json, base_dir),
        "task_result": _relpath_or_str(task_result, base_dir),
        "submit_script": _relpath_or_str(submit_script, base_dir),
    }


def materialize_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    n_boxes: Optional[int] = None,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    """Materialize external production."""

    outdir = Path(outdir)
    prod_dir = outdir / "production"
    ensure_dir(prod_dir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    dft_enabled = bool(getattr(getattr(prod_cfg, "dft_opt", None), "enabled", False))
    if dft_enabled:
        raise ValueError("external dry/full-run does not support production.dft_opt refinement")

    planned = int(n_boxes if n_boxes is not None else _planned_box_count(prod_cfg))
    if planned < 1:
        planned = 1

    structure_src = Path(str(plan.get("structure_data", ""))).expanduser()
    if not structure_src.is_absolute():
        structure_src = (outdir / structure_src).resolve(strict=False)
    if not structure_src.exists():
        raise FileNotFoundError(f"Production-plan structure_data not found: {structure_src}")

    task_records: list[dict[str, Any]] = []
    for box_id in range(1, int(planned) + 1):
        box_dir = prod_dir / f"box_{box_id:03d}"
        ensure_dir(box_dir)
        input_snapshot = _copy_input_snapshot(structure_src, box_dir / "input" / structure_src.name)
        stages, _stage_diag, _runner, pot_cfg, _type_to_species, _potential_lines, _force_iso, stage_cont = _stage_specs_for_box(
            config=config,
            plan=plan,
            box_dir=box_dir,
            input_snapshot=input_snapshot,
        )
        preview_paths = _render_preview_inputs(
            config=config,
            plan=plan,
            box_dir=box_dir,
            stages=stages,
            potential_cfg=pot_cfg,
            stage_continuity=stage_cont,
        )
        task_json = box_dir / "task.json"
        task_result = box_dir / "task_result.json"
        task = {
            "schema": "vitriflow.box_task.v1",
            "config": config.model_dump(mode="json"),
            "production_plan": dict(plan),
            "task": {
                "box": int(box_id),
                "box_dir": str(box_dir),
                "input_snapshot": str(input_snapshot),
                "task_json": str(task_json),
                "task_result": str(task_result),
                "preview_inputs": list(preview_paths),
            },
        }
        atomic_write_json(task_json, task)
        submit_script = _write_submission_script(task_json, job_template)
        task_records.append(_task_record(outdir, task_json, task_result, submit_script))
        progress.info("external", f"materialised task for box {box_id}")

    submit_all = prod_dir / "submit_all.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for rec in task_records:
        if rec.get("submit_script"):
            lines.append(f"sbatch {rec['submit_script']}")
    submit_all.write_text("\n".join(lines) + "\n")
    submit_all.chmod(0o755)

    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(prod_dir),
        "boxes": [
            {
                "box": int(rec["box"]),
                "box_dir": rec["box_dir"],
                "task_result": rec["task_result"],
            }
            for rec in task_records
        ],
        "planned_boxes": int(planned),
        "job_template": (None if job_template is None else str(job_template)),
    }
    atomic_write_json(prod_dir / "output_dataset.json", dataset)
    atomic_write_json(prod_dir / "tasks.json", {"schema": "vitriflow.task_index.v1", "tasks": task_records})
    progress.info("external", f"wrote {len(task_records)} task manifests")

    return {
        "planned_boxes": int(planned),
        "tasks": task_records,
        "output_dataset": _relpath_or_str(prod_dir / "output_dataset.json", outdir),
        "task_index": _relpath_or_str(prod_dir / "tasks.json", outdir),
        "submit_all": _relpath_or_str(submit_all, outdir),
    }


def execute_production_box_task(task: Path | Mapping[str, Any]) -> dict[str, Any]:
    """Production box task."""

    task_data = dict(task) if isinstance(task, Mapping) else json.loads(Path(task).read_text())
    if str(task_data.get("schema", "")).strip().lower() != "vitriflow.box_task.v1":
        raise ValueError("Unsupported task manifest schema")

    task_manifest_sha256 = _task_manifest_digest(task_data)

    task_meta = dict(task_data.get("task", {}) or {})
    box_id = int(task_meta.get("box"))
    box_dir = Path(str(task_meta.get("box_dir")))
    input_snapshot = Path(str(task_meta.get("input_snapshot")))
    task_result = Path(str(task_meta.get("task_result", box_dir / "task_result.json")))

    if task_result.exists():
        try:
            cached = json.loads(task_result.read_text())
            status = str(cached.get("status", "")).strip().lower()
            cached_manifest = str(cached.get("task_manifest_sha256", "") or "").strip().lower()
            if status in {"ok", "success"} and cached_manifest == task_manifest_sha256:
                return cached
        except Exception:
            pass

    try:
        config = RunConfig.model_validate(task_data.get("config", {}))
        plan = dict(task_data.get("production_plan", {}) or {})
        stages, _stage_diag, runner, pot_cfg, type_to_species, potential_lines, _force_iso, stage_cont = _stage_specs_for_box(
            config=config,
            plan=plan,
            box_dir=box_dir,
            input_snapshot=input_snapshot,
        )
        engine = str(plan.get("engine", getattr(config, "engine", "lammps"))).strip().lower()
        md_use = MDConfig.model_validate(plan.get("md_use", {}))

        if engine == "lammps":
            if pot_cfg is not None:
                ensure_model_installed(getattr(pot_cfg, "model", ""))
            if stage_cont == "continuous" and isinstance(runner, LammpsRunner):
                arts = run_stages_continuous_lammps(
                    runner,
                    pot_cfg,
                    md_use,
                    list(stages),
                    [box_dir / "warmup", box_dir / "melt", box_dir / "quench", box_dir / "relax"],
                    box_dir / "continuous",
                    potential_lines=potential_lines,
                    type_to_species=type_to_species,
                )
                outcomes = [stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage) for art, stage in zip(arts, stages)]
            else:
                outcomes = []
                for stage in stages:
                    stage_dir = box_dir / str(stage.name)
                    art = run_stage_local(
                        runner,
                        pot_cfg,
                        md_use,
                        stage,
                        stage_dir,
                        potential_lines=potential_lines,
                        log_name="log.lammps",
                        type_to_species=type_to_species,
                    )
                    outcomes.append(stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage))
        else:
            outcomes = []
            for stage in stages:
                stage_dir = box_dir / str(stage.name)
                art = run_stage_local(
                    runner,
                    pot_cfg,
                    md_use,
                    stage,
                    stage_dir,
                    potential_lines=None,
                    log_name="cp2k.out",
                    type_to_species=type_to_species,
                )
                outcomes.append(stage_outcome_from_artifacts(art, md_cfg=md_use, stage=stage))

        outcome_map = {str(stage.name): asdict(out) for stage, out in zip(stages, outcomes)}
        relax_out = outcomes[-1]
        result = {
            "schema": "vitriflow.box_task_result.v1",
            "status": "ok",
            "box": int(box_id),
            "engine": engine,
            "task": task_meta,
            "task_manifest_sha256": task_manifest_sha256,
            "seeds": _box_seed_map(int(plan.get("seed_base", config.random_seed + 13579)), int(box_id)),
            "outcomes": outcome_map,
            "density": float(relax_out.density_mean),
            "density_stderr": float(relax_out.density_stderr),
            "paths": {
                "box_dir": str(box_dir),
                "warmup_dir": str(box_dir / "warmup"),
                "melt_dir": str(box_dir / "melt"),
                "quench_dir": str(box_dir / "quench"),
                "relax_dir": str(box_dir / "relax"),
                "task_json": str(task_meta.get("task_json", "")),
                "input_snapshot": str(input_snapshot),
            },
        }
        atomic_write_json(task_result, result)
        return result
    except Exception as exc:
        failed = {
            "schema": "vitriflow.box_task_result.v1",
            "status": "failed",
            "box": int(box_id),
            "task": task_meta,
            "task_manifest_sha256": task_manifest_sha256,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(task_result, failed)
        raise RuntimeError(f"production box task {box_id} failed: {exc}") from exc


def _execute_task_path(task_json: Path) -> dict[str, Any]:
    return execute_production_box_task(task_json)


def _summarise_analysis_as_production_state(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    prod_cfg: ProductionEnsembleConfig,
    analysis: Mapping[str, Any],
    mode: str,
    max_parallel_boxes: int,
    job_template: Optional[Path],
    planned_boxes: int,
) -> dict[str, Any]:
    outdir = Path(outdir)
    prod_dir = outdir / "production"
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=resolve_production_time_unit_ps(
            config=config,
            engine=str(plan.get("engine", getattr(config, "engine", "lammps"))),
            pot_cfg=pot_cfg,
            time_unit_ps=plan.get("time_unit_ps", None),
        ),
    )
    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    max_boxes: int | None
    if max_boxes_raw is None:
        max_boxes = None
    else:
        try:
            max_boxes = int(max_boxes_raw)
        except Exception:
            max_boxes = None
        if max_boxes is not None and max_boxes <= 0:
            max_boxes = None
    return {
        "enabled": True,
        "status": str(analysis.get("status", "ok")),
        "error": analysis.get("error", None),
        "converged": bool(analysis.get("converged", False)),
        "n_boxes": int(analysis.get("n_boxes", 0)),
        "n_boxes_accepted": int(analysis.get("n_boxes_accepted", analysis.get("n_boxes", 0))),
        "n_boxes_rejected": int(analysis.get("n_boxes_rejected", 0)),
        "n_boxes_total": int(analysis.get("n_boxes_total", 0)),
        "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)),
        "max_boxes": max_boxes,
        "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)),
        "check_convergence": bool(getattr(prod_cfg, "check_convergence", True)),
        "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)),
        "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000)),
        "rate_K_per_time": float(plan.get("chosen_rate")),
        "rate_K_per_ps": (None if plan.get("cooling_rate_ps", None) is None else float(plan.get("cooling_rate_ps"))),
        "replicate": [int(x) for x in plan.get("replicate", [1, 1, 1])],
        "structure_data": _relpath_or_str(Path(str(plan.get("structure_data"))), outdir),
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "rejects_dir": (_relpath_or_str(prod_dir / str(getattr(prod_cfg, "rejects_subdir", "rejects")), outdir) if (prod_dir / str(getattr(prod_cfg, "rejects_subdir", "rejects"))).exists() else None),
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "T_high": float(plan.get("T_high")),
        "t_final": float(plan.get("t_final")),
        "quench_steps": int(plan.get("quench_steps")),
        "highT_steps": int(plan.get("high_total_steps")),
        "relax_steps": int(plan.get("relax_steps")),
        "cutoffs": list(analysis.get("cutoffs", [])),
        "metrics_checked": analysis.get("metrics_checked", None),
        "convergence_spec": analysis.get("convergence_spec", None),
        "converged_md": bool(analysis.get("converged", False)),
        "convergence_md": analysis.get("convergence", {}),
        "converged_dft": None,
        "convergence_dft": None,
        "convergence": analysis.get("convergence", {}),
        "dft_opt": None,
        "boxes_dft_final": None,
        "n_boxes_dft_accepted": None,
        "rejected_boxes_dft": None,
        "boxes": list(analysis.get("boxes", [])),
        "rejected_boxes": list(analysis.get("rejected_boxes", [])),
        "ensemble_dir": "production",
        "execution": {
            "mode": str(mode),
            "planned_boxes": int(planned_boxes),
            "max_parallel_boxes": int(max_parallel_boxes),
            "job_template": (None if job_template is None else str(job_template)),
            "output_dataset": "production/output_dataset.json",
            "analysis_results": "production/analysis_results.json",
        },
    }


def dry_run_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    md_use = MDConfig.model_validate(plan.get("md_use", {}))
    pot_cfg = _potential_from_dict(plan.get("potential_config", None), config.kim)
    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(md_use.timestep),
        time_unit_ps=resolve_production_time_unit_ps(
            config=config,
            engine=str(plan.get("engine", getattr(config, "engine", "lammps"))),
            pot_cfg=pot_cfg,
            time_unit_ps=plan.get("time_unit_ps", None),
        ),
    )
    materialized = materialize_external_production(
        config=config,
        outdir=outdir,
        plan=plan,
        job_template=job_template,
        progress=progress,
    )
    return {
        "enabled": True,
        "status": "planned",
        "error": None,
        "converged": False,
        "n_boxes": 0,
        "n_boxes_accepted": 0,
        "n_boxes_rejected": 0,
        "n_boxes_total": int(materialized.get("planned_boxes", 0)),
        "min_boxes": int(getattr(prod_cfg, "min_boxes", 0)),
        "max_boxes": getattr(prod_cfg, "max_boxes", None),
        "batch_boxes": int(getattr(prod_cfg, "batch_boxes", 1)),
        "check_convergence": bool(getattr(prod_cfg, "check_convergence", True)),
        "dump_trajectory": bool(getattr(prod_cfg, "dump_trajectory", True)),
        "dump_every_steps": int(getattr(prod_cfg, "dump_every_steps", 5000)),
        "rate_K_per_time": float(plan.get("chosen_rate")),
        "rate_K_per_ps": (None if plan.get("cooling_rate_ps", None) is None else float(plan.get("cooling_rate_ps"))),
        "replicate": [int(x) for x in plan.get("replicate", [1, 1, 1])],
        "structure_data": _relpath_or_str(Path(str(plan.get("structure_data"))), Path(outdir)),
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "rejects_dir": None,
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "T_high": float(plan.get("T_high")),
        "t_final": float(plan.get("t_final")),
        "quench_steps": int(plan.get("quench_steps")),
        "highT_steps": int(plan.get("high_total_steps")),
        "relax_steps": int(plan.get("relax_steps")),
        "cutoffs": list(plan.get("preferred_cutoffs", []) or plan.get("cutoffs_size", []) or plan.get("cutoffs_rate", [])),
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
        "execution": {
            "mode": "dry-run",
            **materialized,
        },
    }


def full_run_external_production(
    *,
    config: RunConfig,
    outdir: Path,
    plan: Mapping[str, Any],
    job_template: Optional[Path] = None,
    max_parallel_boxes: int = 1,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    prod_dir = outdir / "production"
    ensure_dir(prod_dir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    dft_enabled = bool(getattr(getattr(prod_cfg, "dft_opt", None), "enabled", False))
    if dft_enabled:
        raise ValueError("external dry/full-run does not support production.dft_opt refinement")

    target = max(1, int(getattr(prod_cfg, "min_boxes", 1) or 1))
    batch = max(1, int(getattr(prod_cfg, "batch_boxes", 1) or 1))
    do_converge = bool(getattr(prod_cfg, "check_convergence", True))

    max_boxes_raw = getattr(prod_cfg, "max_boxes", None)
    max_boxes: Optional[int]
    if max_boxes_raw is None:
        max_boxes = None
    else:
        try:
            max_boxes = int(max_boxes_raw)
        except Exception:
            max_boxes = None
        if max_boxes is not None and max_boxes <= 0:
            max_boxes = None
        if max_boxes is not None and max_boxes < target:
            max_boxes = target

    planned_initial = int(target)
    materialize_external_production(
        config=config,
        outdir=outdir,
        plan=plan,
        job_template=job_template,
        n_boxes=planned_initial,
        progress=progress,
    )

    analysis: dict[str, Any] = {
        "status": "ok",
        "converged": False,
        "n_boxes": 0,
        "n_boxes_total": 0,
        "boxes": [],
        "rejected_boxes": [],
        "convergence": {},
        "cutoffs": list(plan.get("preferred_cutoffs", []) or plan.get("cutoffs_size", []) or plan.get("cutoffs_rate", [])),
    }

    while True:
        task_paths: list[Path] = []
        for box_id in range(1, int(target) + 1):
            task_json = prod_dir / f"box_{box_id:03d}" / "task.json"
            if not task_json.exists():
                raise FileNotFoundError(f"Missing task manifest: {task_json}")
            task_result = prod_dir / f"box_{box_id:03d}" / "task_result.json"
            if not task_result.exists():
                task_paths.append(task_json)
                continue
            try:
                cached = json.loads(task_result.read_text())
                if str(cached.get("status", "")).strip().lower() not in {"ok", "success"}:
                    task_paths.append(task_json)
            except Exception:
                task_paths.append(task_json)

        if task_paths:
            workers = max(1, int(max_parallel_boxes))
            progress.info("external", f"executing {len(task_paths)} task(s) with max_parallel_boxes={workers}")
            if workers == 1:
                for task_json in task_paths:
                    execute_production_box_task(task_json)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(_execute_task_path, task_json): task_json for task_json in task_paths}
                    for fut in as_completed(futs):
                        fut.result()

        analysis = analyze_output_data(
            config=config,
            input_path=prod_dir,
            outdir=prod_dir,
            plan=plan,
            progress=progress,
        )

        converged = bool(analysis.get("converged", False))
        total_boxes = int(analysis.get("n_boxes_total", 0))
        if not do_converge:
            break
        if converged and int(analysis.get("n_boxes", 0)) >= target:
            break
        if max_boxes is not None and total_boxes >= int(max_boxes):
            break
        if max_boxes is None and total_boxes >= target:
            target = total_boxes + batch
        elif max_boxes is not None and total_boxes >= target:
            target = min(int(max_boxes), total_boxes + batch)
        if max_boxes is not None and target > int(max_boxes):
            target = int(max_boxes)
        if total_boxes >= target and task_paths == []:
            break
        if target > planned_initial:
            materialize_external_production(
                config=config,
                outdir=outdir,
                plan=plan,
                job_template=job_template,
                n_boxes=target,
                progress=progress,
            )
            planned_initial = max(planned_initial, target)

    return _summarise_analysis_as_production_state(
        config=config,
        outdir=outdir,
        plan=plan,
        prod_cfg=prod_cfg,
        analysis=analysis,
        mode="full-run",
        max_parallel_boxes=max(1, int(max_parallel_boxes)),
        job_template=job_template,
        planned_boxes=planned_initial,
    )
