from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from . import __version__
from .config import MDConfig, RunConfig, StructureMetricsConfig
from .workflows.metric_requirements import (
    fixed_cutoffs_from_metrics,
    required_pairs_from_metrics,
)
from .workflows.metrics_policy import resolve_effective_metrics_config


def _print_json(value: Any) -> None:
    """Emit standards-compliant JSON for machine-readable CLI results."""

    from .analysis.provenance import json_dumps_strict

    print(json_dumps_strict(value, indent=2, sort_keys=False))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _positive_int(text: str) -> int:
    """Argparse converter for public count/parallelism overrides."""

    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer (> 0)")
    return value


def _result_exit_code(result: Any) -> int:
    """Map terminal workflow status to a stable shell exit contract.

    ``planned`` is the successful terminal state of a dry-run.  Scientific
    incompleteness and a failed convergence target are valid result bundles,
    but they are not successful application executions and therefore return a
    distinct non-zero code.  Analysis-only descriptor non-convergence remains
    advisory because that workflow reports it under a top-level ``status=ok``.
    """

    if not isinstance(result, Mapping):
        return 1
    status = str(result.get("status", "") or "").strip().lower()
    if status in {"ok", "planned", "completed", "not_requested", "advisory"}:
        return 0
    if status in {"incomplete", "not_converged"}:
        return 2
    # A synchronous public workflow must never silently succeed with an
    # unknown, active, or explicit failure status.
    return 1


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
        "graph_rules",
        "type_to_species",
        "species",
        "types",
        "embed_structures",
    }
    if any(k in data for k in standalone_keys):
        return True
    if isinstance(data.get("autotune", None), dict) and not any(k in data for k in ("structure", "potential", "kim")):
        return True
    return False


def _load_output_analysis_config(path: Path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    if _looks_like_standalone_analysis_config(data):
        from .workflows.output_analysis import analysis_context_from_standalone_config

        return None, analysis_context_from_standalone_config(data)
    return RunConfig.from_yaml(path), None


def _parse_float_csv(text: Optional[str]) -> list[float]:
    if text is None:
        return []
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def _graph_rules_from_cli_args(args: Any) -> Optional[list[dict[str, Any]]]:
    rules: list[dict[str, Any]] = []
    for idx, cutoff in enumerate(list(getattr(args, "graph_cutoff", None) or []), start=1):
        rules.append(
            {
                "name": f"cli_hard_cutoff_{idx}",
                "kind": "hard_cutoff",
                "parameters": {"cutoff": float(cutoff)},
                "provenance": "cli:--graph-cutoff",
            }
        )
    sweep = _parse_float_csv(getattr(args, "graph_cutoff_sweep", None))
    if sweep:
        rules.append(
            {
                "name": "cli_hard_cutoff_sweep",
                "kind": "hard_cutoff_sweep",
                "parameters": {"cutoffs": [float(x) for x in sweep]},
                "provenance": "cli:--graph-cutoff-sweep",
            }
        )
    interval = getattr(args, "graph_cutoff_interval", None)
    if interval is not None:
        r_min, r_max = [float(x) for x in interval]
        rules.append(
            {
                "name": "cli_hard_cutoff_interval",
                "kind": "hard_cutoff_interval",
                "parameters": {
                    "r_min": r_min,
                    "r_max": r_max,
                    "points": int(getattr(args, "graph_interval_points", 9) or 9),
                },
                "provenance": "cli:--graph-cutoff-interval",
            }
        )
    soft = getattr(args, "soft_logistic", None)
    if soft is not None:
        r0, sigma = [float(x) for x in soft]
        rules.append(
            {
                "name": "cli_soft_logistic",
                "kind": "soft_logistic",
                "parameters": {"r0": r0, "sigma": sigma},
                "provenance": "cli:--soft-logistic",
            }
        )
    return rules or None


def _load_metrics_timeseries_context(
    path: Path,
) -> tuple[StructureMetricsConfig, float, Optional[list[str]], str, str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        data = {}

    md_raw = data.get("md", {}) if isinstance(data.get("md", {}), dict) else {}
    md_cfg = MDConfig.model_validate(md_raw)

    autotune_raw = data.get("autotune", {}) if isinstance(data.get("autotune", {}), dict) else {}
    metrics_raw = autotune_raw.get("metrics", {}) if isinstance(autotune_raw.get("metrics", {}), dict) else {}
    metrics_cfg = StructureMetricsConfig.model_validate(metrics_raw or {})

    type_to_species = metrics_cfg.type_to_species
    potential_raw = None
    if type_to_species is None:
        if isinstance(data.get("potential", None), dict):
            potential_raw = data.get("potential", None)
        elif isinstance(data.get("kim", None), dict):
            potential_raw = data.get("kim", None)
        if isinstance(potential_raw, dict):
            interactions = potential_raw.get("interactions", None)
            if isinstance(interactions, list) and len(interactions) > 0:
                type_to_species = [str(x) for x in interactions]

    def _warn(msg: str) -> None:
        warnings.warn(str(msg), stacklevel=2)

    if potential_raw is None:
        if isinstance(data.get("potential", None), dict):
            potential_raw = data.get("potential")
        elif isinstance(data.get("kim", None), dict):
            potential_raw = data.get("kim")
    engine = str(data.get("engine", "lammps") or "lammps").strip().lower()
    units_style = (
        str(potential_raw.get("user_units", "metal") or "metal")
        if engine == "lammps" and isinstance(potential_raw, dict)
        else None
    )
    if engine == "lammps" and units_style is None:
        raise ValueError(
            "metrics-timeseries requires potential.user_units/kim.user_units to resolve "
            "LAMMPS trajectory and timestep units"
        )
    metrics_cfg, _auto_defaults, _summary = resolve_effective_metrics_config(
        metrics_cfg,
        structure_data=None,
        type_to_species=type_to_species,
        lammps_units_style=units_style,
        warn_fn=_warn,
        context="metrics-timeseries CLI",
    )

    if metrics_cfg.type_to_species is not None:
        type_to_species = list(metrics_cfg.type_to_species)
    elif type_to_species is not None:
        type_to_species = [str(x) for x in type_to_species]

    if engine == "cp2k":
        timestep_ps = float(md_cfg.timestep) * 1.0e-3
    else:
        from .lammps_units import time_to_ps_factor

        timestep_ps = float(md_cfg.timestep) * float(time_to_ps_factor(units_style))
    return metrics_cfg, timestep_ps, type_to_species, (units_style or ""), engine


def _normalised_json_value(value: Any) -> Any:
    """Return a deterministic JSON-shaped value for identity comparisons."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _strict_cutoff_map(
    value: Any,
    *,
    context: str,
    n_types: Optional[int],
) -> dict[tuple[int, int], float]:
    """Parse a public cutoff list without dropping malformed entries."""

    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError(f"{context} cutoffs must be a list")
    out: dict[tuple[int, int], float] = {}
    for idx, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} cutoff entry {idx} must be an object")
        pair = entry.get("pair")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{context} cutoff entry {idx} has an invalid pair")
        try:
            a, b = int(pair[0]), int(pair[1])
            cutoff = float(entry.get("cutoff"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} cutoff entry {idx} has non-numeric fields"
            ) from exc
        if a <= 0 or b <= 0:
            raise ValueError(f"{context} cutoff entry {idx} uses non-positive atom types")
        if n_types is not None and (a > n_types or b > n_types):
            raise ValueError(
                f"{context} cutoff entry {idx} references type outside 1..{n_types}"
            )
        if not (math.isfinite(cutoff) and cutoff > 0.0):
            raise ValueError(f"{context} cutoff entry {idx} must be finite and > 0")
        key = (min(a, b), max(a, b))
        if key in out:
            raise ValueError(f"{context} contains duplicate cutoff pair {key}")
        out[key] = cutoff
    return out


def _validate_metrics_timeseries_stage_contract(
    stage_dir: Path,
    *,
    engine: str,
    timestep_ps: float,
    units_style: str,
) -> None:
    """Check a stage-local unit/time manifest when the current format has one."""

    from .io.stage_manifest import STAGE_ARTIFACT_MANIFEST_NAME, load_stage_artifact_manifest

    manifest_path = Path(stage_dir) / STAGE_ARTIFACT_MANIFEST_NAME
    if not manifest_path.exists():
        return
    manifest = load_stage_artifact_manifest(manifest_path)
    manifest_engine = str(manifest.get("engine", "") or "").strip().lower()
    if manifest_engine != str(engine).strip().lower():
        raise ValueError(
            "metrics-timeseries configuration/stage engine mismatch: "
            f"config={engine!r} stage={manifest_engine!r}"
        )
    manifest_dt = float(manifest.get("timestep_ps"))
    if not math.isclose(
        manifest_dt,
        float(timestep_ps),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            "metrics-timeseries configuration/stage timestep mismatch: "
            f"config={float(timestep_ps):.17g} ps stage={manifest_dt:.17g} ps"
        )
    if manifest_engine == "lammps":
        native = manifest.get("native_source_units", {})
        stage_units = (
            str(native.get("lammps_units_style", "") or "").strip().lower()
            if isinstance(native, Mapping)
            else ""
        )
        if stage_units != str(units_style).strip().lower():
            raise ValueError(
                "metrics-timeseries configuration/stage LAMMPS units mismatch: "
                f"config={units_style!r} stage={stage_units!r}"
            )


def _metrics_timeseries_cutoffs_from_results(
    results: Mapping[str, Any],
    *,
    results_path: Path,
    stage_dir: Path,
    metrics_cfg: StructureMetricsConfig,
    timestep_ps: float,
    type_to_species: Optional[Sequence[str]],
    units_style: str,
    engine: str,
) -> tuple[dict[tuple[int, int], float], float]:
    """Validate results/config identity and return a complete cutoff mapping.

    Current results carry a production plan and are checked exactly.  Legacy
    files without that plan remain readable, but all metadata they do provide
    is still consistency-checked and malformed cutoff rows are never ignored.
    """

    root = Path(results_path).resolve().parent
    stage_resolved = Path(stage_dir).resolve()
    if not stage_resolved.is_relative_to(root):
        raise ValueError(
            "metrics-timeseries --dir must be inside the output directory bound "
            f"to --results ({root})"
        )

    status = str(results.get("status", "") or "").strip().lower()
    if status in {"error", "failed", "failure", "starting", "running"}:
        raise ValueError(
            f"metrics-timeseries cannot reuse non-terminal/failed results status {status!r}"
        )

    production = results.get("production", {})
    production = production if isinstance(production, Mapping) else {}
    recommendation = results.get("recommendation", {})
    recommendation = recommendation if isinstance(recommendation, Mapping) else {}
    parameters = results.get("parameters", {})
    parameters = parameters if isinstance(parameters, Mapping) else {}
    units = results.get("units", {})
    units = units if isinstance(units, Mapping) else {}
    plan = results.get("production_plan", {})
    plan = plan if isinstance(plan, Mapping) else {}

    engine_claims = {
        str(value).strip().lower()
        for value in (
            units.get("engine"),
            parameters.get("engine"),
            plan.get("engine"),
        )
        if value is not None and str(value).strip()
    }
    if len(engine_claims) > 1:
        raise ValueError(
            f"metrics-timeseries results contain conflicting engine metadata: {sorted(engine_claims)}"
        )
    if engine_claims and str(engine).strip().lower() not in engine_claims:
        raise ValueError(
            "metrics-timeseries configuration/results engine mismatch: "
            f"config={engine!r} results={sorted(engine_claims)!r}"
        )

    if str(engine).strip().lower() == "lammps":
        unit_claims = {
            str(value).strip().lower()
            for value in (
                units.get("lammps_units"),
                parameters.get("lammps_units"),
            )
            if value is not None and str(value).strip()
        }
        potential_plan = plan.get("potential_config", {})
        if isinstance(potential_plan, Mapping):
            claim = str(potential_plan.get("user_units", "") or "").strip().lower()
            if claim:
                unit_claims.add(claim)
        if len(unit_claims) > 1:
            raise ValueError(
                "metrics-timeseries results contain conflicting LAMMPS unit metadata: "
                f"{sorted(unit_claims)}"
            )
        if unit_claims and str(units_style).strip().lower() not in unit_claims:
            raise ValueError(
                "metrics-timeseries configuration/results LAMMPS units mismatch: "
                f"config={units_style!r} results={sorted(unit_claims)!r}"
            )

    effective_timestep_ps = float(timestep_ps)
    if plan:
        plan_metrics = plan.get("metrics_cfg")
        if not isinstance(plan_metrics, Mapping):
            raise ValueError(
                "metrics-timeseries current production_plan is missing metrics_cfg identity"
            )
        current_metrics = _normalised_json_value(metrics_cfg)
        if _normalised_json_value(dict(plan_metrics)) != current_metrics:
            raise ValueError(
                "metrics-timeseries configuration/results metrics_cfg identity mismatch"
            )

        plan_t2s = plan.get("type_to_species")
        current_t2s = None if type_to_species is None else [str(x) for x in type_to_species]
        if plan_t2s is not None:
            if not isinstance(plan_t2s, list) or [str(x) for x in plan_t2s] != current_t2s:
                raise ValueError(
                    "metrics-timeseries configuration/results type_to_species identity mismatch"
                )

        md_plan = plan.get("md_use", {})
        if isinstance(md_plan, Mapping) and md_plan.get("timestep") is not None:
            try:
                native_dt = float(md_plan.get("timestep"))
                unit_ps = float(
                    plan.get(
                        "time_unit_ps",
                        1.0e-3 if str(engine).strip().lower() == "cp2k" else float("nan"),
                    )
                )
                plan_dt_ps = native_dt * unit_ps
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "metrics-timeseries production_plan has invalid timestep metadata"
                ) from exc
            if not (
                math.isfinite(plan_dt_ps) and plan_dt_ps > 0.0
            ):
                raise ValueError(
                    "metrics-timeseries production_plan has a non-positive/non-finite timestep"
                )
            # The production plan records the preflight-selected timestep.
            # It may legitimately differ from md.timestep in the source YAML,
            # so it is authoritative for the trajectory time axis.  The stage
            # manifest is checked against this selected value below.
            effective_timestep_ps = float(plan_dt_ps)

    n_types = len(type_to_species) if type_to_species is not None else None
    cutoff_candidates: list[tuple[str, Any]] = []
    for label, raw in (
        ("production", production.get("cutoffs")),
        ("results", results.get("cutoffs")),
        ("recommendation", recommendation.get("cutoffs")),
        ("production_plan", plan.get("preferred_cutoffs")),
    ):
        if raw is not None:
            cutoff_candidates.append((label, raw))

    parsed: list[tuple[str, dict[tuple[int, int], float]]] = [
        (
            label,
            _strict_cutoff_map(raw, context=f"metrics-timeseries {label}", n_types=n_types),
        )
        for label, raw in cutoff_candidates
    ]
    nonempty = [(label, mapping) for label, mapping in parsed if mapping]
    if nonempty:
        ref_label, cut_map = nonempty[0]
        for label, mapping in nonempty[1:]:
            if set(mapping) != set(cut_map) or any(
                not math.isclose(mapping[key], cut_map[key], rel_tol=1.0e-12, abs_tol=1.0e-12)
                for key in cut_map
            ):
                raise ValueError(
                    "metrics-timeseries results contain conflicting cutoff identities: "
                    f"{ref_label!r} versus {label!r}"
                )
        cut_map = dict(cut_map)
    else:
        cut_map = {}

    required = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in required_pairs_from_metrics(
            metrics_cfg,
            type_to_species=(None if type_to_species is None else list(type_to_species)),
        )
    }
    missing = sorted(required - set(cut_map))
    if missing:
        raise ValueError(
            "metrics-timeseries results cutoffs do not cover required metric pairs: "
            f"{missing}"
        )
    fixed = fixed_cutoffs_from_metrics(
        metrics_cfg,
        type_to_species=(None if type_to_species is None else list(type_to_species)),
    )
    for pair, value in fixed.items():
        key = (min(int(pair[0]), int(pair[1])), max(int(pair[0]), int(pair[1])))
        if key in cut_map and not math.isclose(
            float(cut_map[key]),
            float(value),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"metrics-timeseries configured fixed cutoff {key} disagrees with results"
            )
    return cut_map, effective_timestep_ps


def _type_to_species_from_run_config(config: RunConfig) -> Optional[list[str]]:
    """Resolve the public analysis type map using the workflow contract.

    Plotting commands must not depend on a local variable initialised by a
    different CLI branch.  Keep this resolution aligned with the run,
    autotune, and output-analysis workflows: an explicit metrics map wins,
    otherwise a named potential interaction order is used.  CP2K has no
    implicit atom-type ordering, so it must remain explicit.
    """

    metrics = config.autotune.metrics
    if metrics.type_to_species is not None:
        return [str(x) for x in metrics.type_to_species]
    potential = getattr(config, "kim", None)
    interactions = getattr(potential, "interactions", None)
    if interactions is not None and interactions != "fixed_types":
        return [str(x) for x in interactions]
    if str(getattr(config, "engine", "lammps")).strip().lower() == "cp2k":
        raise ValueError(
            "engine='cp2k' plotting requires autotune.metrics.type_to_species"
        )
    return None


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="vitriflow",
        description="Melt-quench MD workflows in LAMMPS with OpenKIM models or explicit LAMMPS potentials.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
        help=(
            "Resume from an existing run_results.json. In default auto mode, "
            "resume when that checkpoint exists or start only in an empty output directory. "
            "--resume requires the checkpoint; --no-resume requires an empty output directory."
        ),
    )

    p_auto = sub.add_parser("autotune", help="Auto-tune melting temperature, high-T time, cooling rate, and box size.")
    p_auto.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file.")
    p_auto.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory.")
    p_auto.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Resume from an existing autotune_results.json. In default auto mode, "
            "resume when that checkpoint exists or start only in an empty output directory. "
            "--resume requires that checkpoint; --no-resume requires an empty output directory."
        ),
    )

    p_run = sub.add_parser("run", help="Run a melt-quench workflow (optionally using autotune recommendations).")
    p_run.add_argument("-c", "--config", type=Path, required=True, help="YAML configuration file.")
    p_run.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory.")
    p_run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Resume from an existing run_results.json. In default auto mode, "
            "resume when that checkpoint exists or start only in an empty output directory. "
            "--resume requires that checkpoint; --no-resume requires an empty output directory."
        ),
    )
    p_run.add_argument("--use-autotune", type=Path, default=None, help="Path to autotune_results.json or a standalone production_plan JSON for exact production replay.")
    p_run.add_argument("--n-replicates", type=_positive_int, default=1, help="Number of independent melt-quench replicates.")
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
        type=_positive_int,
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
    p_plotp.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI (for raster outputs).")
    p_plotp.add_argument(
        "--show-boxes",
        action="store_true",
        help="Overlay per-box curves (may be visually cluttered for large n).",
    )
    p_plotp.add_argument(
        "--max-pages",
        type=_positive_int,
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
    p_plotpc.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI (for raster outputs).")
    p_plotpc.add_argument(
        "--max-pages",
        type=_positive_int,
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
    p_plotm.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI for raster outputs.")

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
        help=(
            "Optional autotune_results.json fallback for legacy stage directories without "
            "stage_artifacts.json; a stage-local manifest is authoritative when present."
        ),
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
    p_plots.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI for raster outputs.")

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
    p_mts.add_argument("--stride", type=_positive_int, default=1, help="Keep every k-th trajectory frame (k>=1).")
    p_mts.add_argument("--max-frames", type=_positive_int, default=None, help="Optional cap on number of analysed frames (after striding).")
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
    p_pmt.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI.")
    p_pmt.add_argument("--max-pages", type=_positive_int, default=None, help="Optional maximum number of pages/PNGs to emit.")

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
    p_pv.add_argument("--n-samples", type=_positive_int, default=None, help="Override voids.n_samples.")
    p_pv.add_argument("--min-clearance", type=float, default=None, help="Only show points with clearance >= value.")
    p_pv.add_argument("--top-n", type=_positive_int, default=2000, help="Max number of points to show (largest clearance).")
    p_pv.add_argument("--no-atoms", action="store_true", help="Do not overlay atom positions.")
    p_pv.add_argument("--write-void-extxyz", type=Path, default=None, help="Optional EXTXYZ point cloud output.")
    p_pv.add_argument(
        "--write-combined-extxyz",
        type=Path,
        default=None,
        help="Optional EXTXYZ atoms+voids output.",
    )
    p_pv.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI for raster outputs.")
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
    p_pes.add_argument("--dpi", type=_positive_int, default=600, help="Output DPI for raster outputs.")

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
    p_an.add_argument(
        "--graph-cutoff",
        type=float,
        action="append",
        default=None,
        help="Add an explicit hard-cutoff graph rule for graph-derived descriptors. May be supplied more than once.",
    )
    p_an.add_argument(
        "--graph-cutoff-sweep",
        type=str,
        default=None,
        help="Comma-separated hard-cutoff sweep values, e.g. 1.7,1.8,1.9.",
    )
    p_an.add_argument(
        "--graph-cutoff-interval",
        nargs=2,
        type=float,
        metavar=("R_MIN", "R_MAX"),
        default=None,
        help="Hard-cutoff interval for robustness analysis.",
    )
    p_an.add_argument(
        "--graph-interval-points",
        type=_positive_int,
        default=9,
        help="Number of hard-cutoff samples used to expand --graph-cutoff-interval.",
    )
    p_an.add_argument(
        "--soft-logistic",
        nargs=2,
        type=float,
        metavar=("R0", "SIGMA"),
        default=None,
        help="Add a soft logistic graph rule w=1/(1+exp((d-r0)/sigma)).",
    )
    embed_group = p_an.add_mutually_exclusive_group()
    embed_group.add_argument(
        "--embed-structures",
        dest="embed_structures",
        action="store_true",
        default=None,
        help="Embed full final-frame structures in analysis_results.json, overriding the YAML setting.",
    )
    embed_group.add_argument(
        "--no-embed-structures",
        dest="embed_structures",
        action="store_false",
        help="Do not embed full final-frame coordinates in analysis_results.json; retain manifest hashes and source paths.",
    )
    p_an.add_argument(
        "--analysis-workers",
        type=_positive_int,
        default=None,
        help="Number of worker processes for analyze-output box analysis. YAML: analysis.analysis_workers.",
    )
    p_an.add_argument(
        "--analysis-max-in-flight",
        type=_positive_int,
        default=None,
        help="Maximum submitted-but-uncollected analysis tasks; defaults to roughly 2x workers.",
    )
    p_an.add_argument(
        "--no-analysis-streaming",
        dest="analysis_streaming",
        action="store_false",
        default=None,
        help="Disable streamed graph sidecar chunks and use the legacy in-memory graph sidecar writer.",
    )

    p_exec = sub.add_parser(
        "execute-task",
        help="Execute a single external production-box task created by vitriflow dry-run/full-run planning.",
    )
    p_exec.add_argument("--task", type=Path, required=True, help="Path to task.json")

    args = p.parse_args(argv)

    if args.cmd == "autotune":
        from .workflows.autotune import autotune

        cfg = RunConfig.from_yaml(args.config)
        res = autotune(cfg, args.outdir, resume=args.resume)
        _print_json(res.get("recommendation", {}))
        return _result_exit_code(res)

    if args.cmd in {"run-schedule", "run-custom", "run-custom-schedule", "run-custom-stages", "run-cs", "run-hardcarbon", "run-hc"}:
        # Import lazily so custom schedules cannot affect standard run/autotune startup.
        from .workflows.custom_schedule import run_custom_schedule

        cfg = RunConfig.from_yaml(args.config)
        res = run_custom_schedule(cfg, args.outdir, config_path=args.config, resume=args.resume)
        _print_json(res)
        return _result_exit_code(res)

    if args.cmd == "run":
        from .workflows.run import run_meltquench

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
        _print_json(res)
        return _result_exit_code(res)

    if args.cmd == "analyze-output":
        from .workflows.output_analysis import analyze_output_data

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
            graph_rules_override=_graph_rules_from_cli_args(args),
            embed_structures=getattr(args, "embed_structures", None),
            analysis_workers_override=getattr(args, "analysis_workers", None),
            analysis_streaming_override=getattr(args, "analysis_streaming", None),
            analysis_max_in_flight_override=getattr(args, "analysis_max_in_flight", None),
        )
        _print_json(res)
        return _result_exit_code(res)

    if args.cmd == "execute-task":
        from .workflows.hpc import execute_production_box_task

        res = execute_production_box_task(args.task)
        _print_json(res)
        return _result_exit_code(res)

    if args.cmd == "plot":
        from .plotting import plot_autotune_results

        plot_autotune_results(
            args.input,
            args.output,
            title=args.title,
            spread=args.spread,
            show_replicates=bool(args.show_replicates),
        )
        print(str(args.output))
        return 0

    if args.cmd == "plot-production":
        from .plotting import plot_production_results

        plot_production_results(
            args.input,
            args.output,
            title=args.title,
            dpi=int(args.dpi),
            show_boxes=bool(args.show_boxes),
            max_pages=args.max_pages,
        )
        print(str(args.output))
        return 0

    if args.cmd == "plot-production-compare":
        from .plotting import plot_production_comparison_results

        plot_production_comparison_results(
            args.inputs,
            args.output,
            labels=args.labels,
            title=args.title,
            dpi=int(args.dpi),
            max_pages=args.max_pages,
        )
        print(str(args.output))
        return 0

    if args.cmd == "plot-metric":
        from .plotting import plot_scan_metric

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
        return 0

    if args.cmd == "plot-stage":
        from .plotting import plot_stage_timeseries

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
        return 0

    if args.cmd == "metrics-timeseries":
        from .analysis.timeseries import compute_metrics_timeseries

        metrics_cfg, md_timestep, t2s, units_style, engine = _load_metrics_timeseries_context(args.config)

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
            cut_map, md_timestep = _metrics_timeseries_cutoffs_from_results(
                res,
                results_path=Path(res_json),
                stage_dir=stage_dir,
                metrics_cfg=metrics_cfg,
                timestep_ps=float(md_timestep),
                type_to_species=t2s,
                units_style=units_style,
                engine=engine,
            )
        _validate_metrics_timeseries_stage_contract(
            stage_dir,
            engine=engine,
            timestep_ps=float(md_timestep),
            units_style=units_style,
        )

        if not cut_map:
            # estimate window frames
            from .analysis.trajectory import stage_trajectory_path, read_frames_auto

            traj = stage_trajectory_path(stage_dir)
            if traj is None:
                raise FileNotFoundError(f"No trajectory found in stage directory: {stage_dir}")
            last_n = int(getattr(metrics_cfg, "time_average_frames", 5))
            frames_tail = list(
                read_frames_auto(Path(traj), last_n=last_n, units_style=units_style)
            )

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
            trajectory_lammps_units_style=units_style,
        )
        mts.to_csv(args.output)
        print(str(args.output))
        return 0

    if args.cmd == "plot-metrics":
        from .plotting import plot_metrics_timeseries

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
        return 0

    if args.cmd == "plot-voids":
        from .plotting import plot_voids_map

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
        t2s = _type_to_species_from_run_config(cfg)

        n_samples = int(args.n_samples) if args.n_samples is not None else int(getattr(void_cfg, "n_samples", 8192))

        # Raw LAMMPS trajectories must use the dimensional style declared by
        # the run configuration.  Results metadata is an independent
        # consistency check, not a source for a silent metal fallback.
        engine_name = str(getattr(cfg, "engine", "lammps") or "lammps").strip().lower()
        units_style = ""
        if engine_name == "lammps":
            from .lammps_units import normalize_lammps_units_style

            units_style = normalize_lammps_units_style(
                str(getattr(cfg.kim, "user_units", "") or "")
            )
        if args.results is not None:
            try:
                data = _load_json(Path(args.results))
                u = data.get("units", {}) or {}
                reported_units = str(u.get("lammps_units", "") or "").strip().lower()
                if reported_units and units_style and reported_units != units_style:
                    raise ValueError(
                        "plot-voids configuration/results LAMMPS units mismatch: "
                        f"config={units_style!r} results={reported_units!r}"
                    )
            except Exception:
                raise

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
        return 0

    if args.cmd == "plot-elastic":
        from .plotting import plot_elastic_screen

        plot_elastic_screen(
            args.dir,
            args.output,
            title=args.title,
            dpi=int(args.dpi),
        )
        print(str(args.output))
        return 0

    # ``argparse`` requires one of the registered subcommands, so reaching
    # this point would indicate an internal dispatch bug.
    raise RuntimeError(f"Unhandled command: {args.cmd!r}")
