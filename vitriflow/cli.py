from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import MDConfig, RunConfig, StructureMetricsConfig
from .workflows.autotune import autotune
from .workflows.run import run_meltquench
from .workflows.output_analysis import analyze_output_data, analysis_context_from_standalone_config
from .workflows.hpc import execute_production_box_task
from .plotting import (
    plot_autotune_results,
    plot_production_results,
    plot_production_comparison_results,
    plot_scan_metric,
    plot_stage_timeseries,
    plot_metrics_timeseries,
    plot_elastic_screen,
    plot_voids_map,
)
from .analysis.timeseries import compute_metrics_timeseries
from .workflows.metric_requirements import (
    fixed_cutoffs_from_metrics,
    required_pairs_from_metrics,
)
from .workflows.metrics_policy import resolve_effective_metrics_config


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _looks_like_standalone_analysis_config(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if isinstance(data.get("analysis", None), dict):
        return True
    standalone_keys = {
        "metrics",
        "production",
        "convergence",
        "cutoffs",
        "preferred_cutoffs",
        "type_to_species",
        "species",
        "types",
    }
    if any(k in data for k in standalone_keys):
        return True
    if isinstance(data.get("autotune", None), dict) and not any(k in data for k in ("structure", "potential", "kim")):
        return True
    return False


def _load_output_analysis_config(path: Path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    if _looks_like_standalone_analysis_config(data):
        return None, analysis_context_from_standalone_config(data)
    return RunConfig.from_yaml(path), None


def _load_metrics_timeseries_context(path: Path) -> tuple[StructureMetricsConfig, float, Optional[list[str]]]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        data = {}

    md_raw = data.get("md", {}) if isinstance(data.get("md", {}), dict) else {}
    md_cfg = MDConfig.model_validate(md_raw)

    autotune_raw = data.get("autotune", {}) if isinstance(data.get("autotune", {}), dict) else {}
    metrics_raw = autotune_raw.get("metrics", {}) if isinstance(autotune_raw.get("metrics", {}), dict) else {}
    metrics_cfg = StructureMetricsConfig.model_validate(metrics_raw or {})

    type_to_species = metrics_cfg.type_to_species
    if type_to_species is None:
        pot = None
        if isinstance(data.get("potential", None), dict):
            pot = data.get("potential", None)
        elif isinstance(data.get("kim", None), dict):
            pot = data.get("kim", None)
        if isinstance(pot, dict):
            interactions = pot.get("interactions", None)
            if isinstance(interactions, list) and len(interactions) > 0:
                type_to_species = [str(x) for x in interactions]

    def _warn(msg: str) -> None:
        warnings.warn(str(msg), stacklevel=2)

    metrics_cfg, _auto_defaults, _summary = resolve_effective_metrics_config(
        metrics_cfg,
        structure_data=None,
        type_to_species=type_to_species,
        warn_fn=_warn,
        context="metrics-timeseries CLI",
    )

    if metrics_cfg.type_to_species is not None:
        type_to_species = list(metrics_cfg.type_to_species)
    elif type_to_species is not None:
        type_to_species = [str(x) for x in type_to_species]

    return metrics_cfg, float(md_cfg.timestep), type_to_species


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="vitriflow",
        description="Melt-quench MD workflows in LAMMPS with OpenKIM models or explicit LAMMPS potentials.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sched = sub.add_parser(
        "run-schedule",
        aliases=["run-custom", "run-custom-schedule", "run-custom-stages", "run-cs", "run-hardcarbon", "run-hc"],
        help="Run a user-defined fixed stage schedule without changing run/autotune.",
    )
    p_sched.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file with a custom_schedule block.")
    p_sched.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory.")
    p_sched.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from an existing run_results.json in the output directory (default: auto-detect).",
    )

    p_auto = sub.add_parser("autotune", help="Auto-tune melting temperature, high-T time, cooling rate, and box size.")
    p_auto.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file.")
    p_auto.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory.")
    p_auto.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from an existing autotune_results.json in the output directory (default: auto-detect).",
    )

    p_run = sub.add_parser("run", help="Run a melt-quench workflow (optionally using autotune recommendations).")
    p_run.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file.")
    p_run.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory.")
    p_run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from an existing run_results.json in the output directory (default: auto-detect).",
    )
    p_run.add_argument("--use-autotune", type=Path, default=None, help="Path to autotune_results.json or a standalone production_plan JSON for exact production replay.")
    p_run.add_argument("--n-replicates", type=int, default=1, help="Number of independent melt-quench replicates.")
    p_run.add_argument(
        "--external-mode",
        type=str,
        default="local",
        choices=["local", "dry-run", "full-run"],
        help="Execution mode for production: local (default), dry-run task materialisation, or full-run batched external execution.",
    )
    p_run.add_argument(
        "--job-template",
        type=Path,
        default=None,
        help="Optional Slurm job-script template used for dry-run/full-run task materialisation.",
    )
    p_run.add_argument(
        "--max-parallel-boxes",
        type=int,
        default=1,
        help="Maximum number of production boxes to execute concurrently in external full-run mode.",
    )

    p_plot = sub.add_parser("plot", help="Generate a multi-panel plot from autotune_results.json.")
    p_plot.add_argument("-i", "--input", type=Path, required=True, help="Path to autotune_results.json.")
    p_plot.add_argument("-o", "--output", type=Path, required=True, help="Output plot file (png/pdf).")
    p_plot.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    p_plot.add_argument(
        "--spread",
        type=str,
        default="sd",
        choices=["sd", "se", "p16-84"],
        help="Between-replica spread to display: sd (default), se, or p16-84.",
    )
    p_plot.add_argument(
        "--show-replicates",
        action="store_true",
        help="Overlay individual replica points where available.",
    )

    p_plotp = sub.add_parser(
        "plot-production",
        help="Plot production-ensemble convergence (trend vs n_boxes) and converged descriptor distributions.",
    )
    p_plotp.add_argument("-i", "--input", type=Path, required=True, help="Path to autotune_results.json or analysis_results.json.")
    p_plotp.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file (.pdf for multi-page) or output directory for per-page PNGs.",
    )
    p_plotp.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_plotp.add_argument("--dpi", type=int, default=600, help="Output DPI (for raster outputs).")
    p_plotp.add_argument(
        "--show-boxes",
        action="store_true",
        help="Overlay per-box curves (may be visually cluttered for large n).",
    )
    p_plotp.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to emit after the convergence page.",
    )

    p_plotpc = sub.add_parser(
        "plot-production-compare",
        help="Compare production/analysis ensemble results across multiple datasets (e.g. MD, PBE, HSE06).",
    )
    p_plotpc.add_argument(
        "-i",
        "--input",
        dest="inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more autotune_results.json or analysis_results.json files.",
    )
    p_plotpc.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional dataset labels matching the input order.",
    )
    p_plotpc.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file (.pdf for multi-page) or output directory for per-page PNGs.",
    )
    p_plotpc.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_plotpc.add_argument("--dpi", type=int, default=600, help="Output DPI (for raster outputs).")
    p_plotpc.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to emit after the convergence page.",
    )

    p_plotm = sub.add_parser(
        "plot-metric",
        help="Plot a single scalar metric from autotune_results.json (tm_scan/rate_scan/size_scan/production).",
    )
    p_plotm.add_argument("-i", "--input", type=Path, required=True, help="Path to autotune_results.json.")
    p_plotm.add_argument("-o", "--output", type=Path, required=True, help="Output plot file (png/pdf).")
    p_plotm.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["tm_scan", "rate_scan", "size_scan", "production"],
        help="Which stage family to plot.",
    )
    p_plotm.add_argument("--metric", type=str, required=True, help="Metric key to plot.")
    p_plotm.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_plotm.add_argument(
        "--spread",
        type=str,
        default="sd",
        choices=["sd", "se", "p16-84"],
        help="Between-replica spread definition (tm_scan and replicate-based scans).",
    )
    p_plotm.add_argument(
        "--show-replicates",
        action="store_true",
        help="Overlay individual replicate points where available.",
    )
    p_plotm.add_argument("--dpi", type=int, default=600, help="Output DPI for raster outputs.")

    p_plots = sub.add_parser(
        "plot-stage",
        help="Plot thermo/MSD time series for a single stage directory containing thermo.csv (and optionally msd.csv).",
    )
    p_plots.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Stage directory (absolute, or relative to the results JSON if --results is given).",
    )
    p_plots.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output plot file (png/pdf).",
    )
    p_plots.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional autotune_results.json to infer timestep/time units for the x-axis.",
    )
    p_plots.add_argument(
        "--series",
        nargs="*",
        default=None,
        help="Thermo columns to plot (default: Temp Press Density PotEng Volume).",
    )
    p_plots.add_argument(
        "--all-thermo",
        action="store_true",
        help="Plot all thermo columns present in thermo.csv (except Step).",
    )
    p_plots.add_argument(
        "--no-msd",
        action="store_true",
        help="Disable MSD panel even if msd.csv is present.",
    )
    p_plots.add_argument(
        "--x",
        type=str,
        default="time",
        choices=["time", "step"],
        help="x-axis: physical time (if available) or MD step.",
    )
    p_plots.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_plots.add_argument("--dpi", type=int, default=600, help="Output DPI for raster outputs.")

    p_mts = sub.add_parser(
        "metrics-timeseries",
        help="Compute time-resolved structural metrics for a stage directory containing traj.* (and optionally thermo.csv).",
    )
    p_mts.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file (for md.timestep and metrics definitions).")
    p_mts.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Stage directory (absolute, or relative to --results if provided).",
    )
    p_mts.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path (e.g. metrics_timeseries.csv).")
    p_mts.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional autotune_results.json (used to reuse the same pair cutoffs as production/recommendation).",
    )
    p_mts.add_argument("--stride", type=int, default=1, help="Keep every k-th trajectory frame (k>=1).")
    p_mts.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of analysed frames (after striding).")
    p_mts.add_argument(
        "--include-gr-curves",
        action="store_true",
        help="Also compute per-frame g(r) curve summaries (expensive).",
    )
    p_mts.add_argument(
        "--no-coord-defects",
        action="store_true",
        help="Disable per-frame coordination defect fractions (if coordination expectations are configured).",
    )

    p_pmt = sub.add_parser(
        "plot-metrics",
        help="Plot a metrics_timeseries.csv file (multi-page PDF by default: one metric per page).",
    )
    p_pmt.add_argument("-i", "--input", type=Path, required=True, help="Input metrics_timeseries.csv")
    p_pmt.add_argument("-o", "--output", type=Path, required=True, help="Output (.pdf or directory for PNGs)")
    p_pmt.add_argument(
        "--x",
        type=str,
        default="time",
        choices=["time", "step"],
        help="x-axis: physical time (if present) or MD step.",
    )
    p_pmt.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional list of metric column names to plot (default: all columns except Step/time).",
    )
    p_pmt.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_pmt.add_argument("--dpi", type=int, default=600, help="Output DPI.")
    p_pmt.add_argument("--max-pages", type=int, default=None, help="Optional maximum number of pages/PNGs to emit.")

    p_pv = sub.add_parser(
        "plot-voids",
        help="Visualise void clearance samples for the last frame in a box/stage directory.",
    )
    p_pv.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file.")
    p_pv.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Stage/box directory containing traj.extxyz or *.lammpstrj.",
    )
    p_pv.add_argument("-o", "--output", type=Path, required=True, help="Output plot file (png/pdf).")
    p_pv.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional autotune_results.json to infer units for axis labels.",
    )
    p_pv.add_argument("--n-samples", type=int, default=None, help="Override voids.n_samples.")
    p_pv.add_argument("--min-clearance", type=float, default=None, help="Only show points with clearance >= value.")
    p_pv.add_argument("--top-n", type=int, default=2000, help="Max number of points to show (largest clearance).")
    p_pv.add_argument("--no-atoms", action="store_true", help="Do not overlay atom positions.")
    p_pv.add_argument("--write-void-extxyz", type=Path, default=None, help="Optional EXTXYZ point cloud output.")
    p_pv.add_argument(
        "--write-combined-extxyz",
        type=Path,
        default=None,
        help="Optional EXTXYZ atoms+voids output.",
    )
    p_pv.add_argument("--dpi", type=int, default=600, help="Output DPI for raster outputs.")
    p_pv.add_argument("--title", type=str, default=None, help="Optional title override.")

    p_pes = sub.add_parser(
        "plot-elastic",
        help="Plot a Born-matrix / local-stress diagnostic from a stage or elastic directory.",
    )
    p_pes.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Stage directory containing elastic/ or the elastic directory itself.",
    )
    p_pes.add_argument("-o", "--output", type=Path, required=True, help="Output plot file (png/pdf).")
    p_pes.add_argument("--title", type=str, default=None, help="Optional title override.")
    p_pes.add_argument("--dpi", type=int, default=600, help="Output DPI for raster outputs.")

    p_an = sub.add_parser(
        "analyze-output",
        help="Analyse an existing production/output directory and compute structural metrics and convergence.",
    )
    p_an.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file: either a full VitriFlow run config or a standalone analysis-only config.")
    p_an.add_argument("-i", "--input", type=Path, required=True, help="Input directory, flat directory of final structure files in any ASE-readable format, ASE database, generic ensemble of boxes, output_dataset.json, task_result.json, or a single ASE-readable structure file to analyse.")
    p_an.add_argument("-o", "--outdir", type=Path, required=True, help="Directory where analysis_results.json and output_dataset.json will be written.")
    p_an.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Optional production-plan JSON (or results JSON containing production_plan) used to reuse production metrics/settings. Auto cutoffs are re-estimated from the current ensemble unless the metrics config fixes them explicitly.",
    )

    p_exec = sub.add_parser(
        "execute-task",
        help="Execute a single external production-box task created by vitriflow dry-run/full-run planning.",
    )
    p_exec.add_argument("--task", type=Path, required=True, help="Path to task.json")

    args = p.parse_args(argv)

    if args.cmd == "autotune":
        cfg = RunConfig.from_yaml(args.config)
        res = autotune(cfg, args.outdir, resume=args.resume)
        print(json.dumps(res.get("recommendation", {}), indent=2))
        return

    if args.cmd in {"run-schedule", "run-custom", "run-custom-schedule", "run-custom-stages", "run-cs", "run-hardcarbon", "run-hc"}:
        # Import lazily so custom schedules cannot affect standard run/autotune startup.
        from .workflows.custom_schedule import run_custom_schedule

        cfg = RunConfig.from_yaml(args.config)
        res = run_custom_schedule(cfg, args.outdir, config_path=args.config, resume=args.resume)
        print(json.dumps(res, indent=2))
        return

    if args.cmd == "run":
        cfg = RunConfig.from_yaml(args.config)
        source = None
        if args.use_autotune is not None:
            source = _load_json(args.use_autotune)
        rec_base = Path(args.use_autotune).resolve().parent if args.use_autotune else None
        res = run_meltquench(
            cfg,
            args.outdir,
            production_source=source,
            recommendation_base_dir=rec_base,
            n_replicates=args.n_replicates,
            external_mode=str(args.external_mode),
            job_template=args.job_template,
            max_parallel_boxes=int(args.max_parallel_boxes),
            resume=args.resume,
        )
        print(json.dumps(res, indent=2))
        return

    if args.cmd == "analyze-output":
        cfg, standalone_ctx = _load_output_analysis_config(args.config)
        plan_data = None
        if args.plan is not None:
            plan_raw = _load_json(args.plan)
            if isinstance(plan_raw.get("production_plan", None), dict):
                plan_data = dict(plan_raw.get("production_plan", {}))
            else:
                plan_data = dict(plan_raw)
        res = analyze_output_data(
            config=cfg,
            analysis_context=standalone_ctx,
            input_path=args.input,
            outdir=args.outdir,
            plan=plan_data,
        )
        print(json.dumps(res, indent=2))
        return

    if args.cmd == "execute-task":
        res = execute_production_box_task(args.task)
        print(json.dumps(res, indent=2))
        return

    if args.cmd == "plot":
        plot_autotune_results(
            args.input,
            args.output,
            title=args.title,
            spread=args.spread,
            show_replicates=bool(args.show_replicates),
        )
        print(str(args.output))
        return

    if args.cmd == "plot-production":
        plot_production_results(
            args.input,
            args.output,
            title=args.title,
            dpi=int(args.dpi),
            show_boxes=bool(args.show_boxes),
            max_pages=args.max_pages,
        )
        print(str(args.output))
        return

    if args.cmd == "plot-production-compare":
        plot_production_comparison_results(
            args.inputs,
            args.output,
            labels=args.labels,
            title=args.title,
            dpi=int(args.dpi),
            max_pages=args.max_pages,
        )
        print(str(args.output))
        return

    if args.cmd == "plot-metric":
        plot_scan_metric(
            args.input,
            args.output,
            stage=args.stage,
            metric=args.metric,
            title=args.title,
            spread=args.spread,
            show_replicates=bool(args.show_replicates),
            dpi=int(args.dpi),
        )
        print(str(args.output))
        return

    if args.cmd == "plot-stage":
        stage_dir_in = Path(args.dir).expanduser()
        if stage_dir_in.is_absolute():
            stage_dir = stage_dir_in
        else:
            cands: list[Path] = []
            cands.append((Path.cwd() / stage_dir_in).resolve())
            if args.results is not None:
                base = Path(args.results).resolve().parent
                cands.append((base / stage_dir_in).resolve())
                try:
                    if stage_dir_in.parts and base.name == stage_dir_in.parts[0]:
                        cands.append((base / Path(*stage_dir_in.parts[1:])).resolve())
                except Exception:
                    pass
            stage_dir = None
            for c in cands:
                if c.exists():
                    stage_dir = c
                    break
            if stage_dir is None:
                tried = ", ".join(str(c) for c in cands)
                raise FileNotFoundError(f"Stage directory not found. Tried: {tried}")
        plot_stage_timeseries(
            stage_dir,
            args.output,
            results_json=args.results,
            title=args.title,
            thermo_series=list(args.series) if args.series is not None else None,
            plot_all_thermo=bool(args.all_thermo),
            include_msd=not bool(args.no_msd),
            xaxis=args.x,
            dpi=int(args.dpi),
        )
        print(str(args.output))
        return

    if args.cmd == "metrics-timeseries":
        metrics_cfg, md_timestep, t2s = _load_metrics_timeseries_context(args.config)

        stage_dir_in = Path(args.dir).expanduser()
        res_json = args.results

        # dir robustly interpreting
        # relative directory blindly
        # already containing outdir
        if stage_dir_in.is_absolute():
            stage_dir = stage_dir_in
        else:
            cands: list[Path] = []
            # relative directory cli
            cands.append((Path.cwd() / stage_dir_in).resolve())
            # relative directory
            if res_json is not None:
                base = Path(res_json).resolve().parent
                cands.append((base / stage_dir_in).resolve())
                # already included production
                # strip matches directory
                try:
                    if stage_dir_in.parts and base.name == stage_dir_in.parts[0]:
                        cands.append((base / Path(*stage_dir_in.parts[1:])).resolve())
                except Exception:
                    pass

            stage_dir = None
            for c in cands:
                if c.exists():
                    stage_dir = c
                    break
            if stage_dir is None:
                tried = ", ".join(str(c) for c in cands)
                raise FileNotFoundError(f"Stage directory not found. Tried: {tried}")

        # cutoffs production trajectory
        cut_map: dict[tuple[int, int], float] = {}
        if res_json is not None:
            res = _load_json(res_json)
            cut_list = None
            prod = res.get("production", {}) if isinstance(res, dict) else {}
            rec = res.get("recommendation", {}) if isinstance(res, dict) else {}
            if isinstance(prod, dict) and isinstance(prod.get("cutoffs", None), list):
                cut_list = prod.get("cutoffs")
            if (not cut_list) and isinstance(rec, dict) and isinstance(rec.get("cutoffs", None), list):
                cut_list = rec.get("cutoffs")
            if isinstance(cut_list, list):
                for c in cut_list:
                    try:
                        a, b = int(c["pair"][0]), int(c["pair"][1])
                        r = float(c["cutoff"])
                        cut_map[(min(a, b), max(a, b))] = r
                    except Exception:
                        continue

        if not cut_map:
            # estimate window frames
            from .analysis.trajectory import stage_trajectory_path, read_frames_auto

            traj = stage_trajectory_path(stage_dir)
            if traj is None:
                raise FileNotFoundError(f"No trajectory found in stage directory: {stage_dir}")
            last_n = int(getattr(metrics_cfg, "time_average_frames", 5))
            frames_tail = list(read_frames_auto(Path(traj), last_n=last_n))

            req_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=t2s)
            fixed = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=t2s)
            if req_pairs:
                from .analysis.structure import estimate_pair_cutoffs

                cut_map = estimate_pair_cutoffs(
                    frames_tail,
                    req_pairs,
                    auto=metrics_cfg.auto_cutoff,
                    fixed_cutoffs=fixed,
                )
            else:
                # neighbour metrics cutoffs
                cut_map = dict(fixed)

        mts = compute_metrics_timeseries(
            stage_dir=stage_dir,
            metrics=metrics_cfg,
            cutoffs=cut_map,
            md_timestep=float(md_timestep),
            type_to_species=t2s,
            frame_stride=int(args.stride),
            max_frames=int(args.max_frames) if args.max_frames is not None else None,
            include_gr_curves=bool(args.include_gr_curves),
            include_coord_defects=not bool(args.no_coord_defects),
        )
        mts.to_csv(args.output)
        print(str(args.output))
        return

    if args.cmd == "plot-metrics":
        plot_metrics_timeseries(
            args.input,
            args.output,
            xaxis=args.x,
            metrics=list(args.metrics) if args.metrics is not None else None,
            title=args.title,
            dpi=int(args.dpi),
            max_pages=args.max_pages,
        )
        print(str(args.output))
        return

    if args.cmd == "plot-voids":
        cfg = RunConfig.from_yaml(args.config)

        stage_dir_in = Path(args.dir).expanduser()
        if stage_dir_in.is_absolute():
            stage_dir = stage_dir_in
        else:
            cands: list[Path] = []
            cands.append((Path.cwd() / stage_dir_in).resolve())
            if args.results is not None:
                base = Path(args.results).resolve().parent
                cands.append((base / stage_dir_in).resolve())
                try:
                    if stage_dir_in.parts and base.name == stage_dir_in.parts[0]:
                        cands.append((base / Path(*stage_dir_in.parts[1:])).resolve())
                except Exception:
                    pass
            stage_dir = None
            for c in cands:
                if c.exists():
                    stage_dir = c
                    break
            if stage_dir is None:
                tried = ", ".join(str(c) for c in cands)
                raise FileNotFoundError(f"Stage directory not found. Tried: {tried}")

        if cfg.autotune is None or cfg.autotune.metrics is None:
            raise ValueError("plot-voids requires autotune.metrics to be defined in the YAML (for void parameters).")
        void_cfg = cfg.autotune.metrics.voids

        n_samples = int(args.n_samples) if args.n_samples is not None else int(getattr(void_cfg, "n_samples", 8192))

        # label
        units_style = ""
        if args.results is not None:
            try:
                data = _load_json(Path(args.results))
                u = data.get("units", {}) or {}
                units_style = str(u.get("lammps_units", "") or "")
            except Exception:
                units_style = ""

        plot_voids_map(
            stage_dir,
            args.output,
            n_samples=n_samples,
            sampler=str(getattr(void_cfg, "sampler", "sobol")),
            seed=int(getattr(void_cfg, "seed", 0) or 0),
            k_nearest=int(getattr(void_cfg, "k_nearest", 16) or 16),
            type_to_species=t2s,
            radii_by_species=dict(getattr(void_cfg, "radii", {}) or {}),
            default_radius=float(getattr(void_cfg, "default_radius", 0.0) or 0.0),
            min_clearance=float(args.min_clearance) if args.min_clearance is not None else None,
            top_n=int(args.top_n),
            show_atoms=not bool(args.no_atoms),
            units_style=units_style,
            title=args.title,
            write_void_extxyz=args.write_void_extxyz,
            write_combined_extxyz=args.write_combined_extxyz,
            dpi=int(args.dpi),
        )
        print(str(args.output))
        return

    if args.cmd == "plot-elastic":
        plot_elastic_screen(
            args.dir,
            args.output,
            title=args.title,
            dpi=int(args.dpi),
        )
        print(str(args.output))
        return
