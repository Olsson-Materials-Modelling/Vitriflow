from __future__ import annotations

"""Generic analysis for existing melt/quench production outputs.

This module intentionally operates on *existing* output trees. It does not run
elastic screens or stage-timeseries diagnostics; the goal is to compute the same
production-style structural metrics and convergence statistics from arbitrary
output data layouts (local runs, externally executed task batches, manually
assembled box directories, or direct ensembles of amorphous box snapshots from
LAMMPS/CP2K/VASP-style files).
"""

import fnmatch
import hashlib
import inspect
import json
import math
import os
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from decimal import Decimal, InvalidOperation
from numbers import Integral

import numpy as np
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import re
import shutil

from ..config import ConvergenceConfig, MDConfig, ProductionEnsembleConfig, RunConfig, StructureMetricsConfig
from ..io.thermo import parse_thermo_csv
from ..io.ase_compat import ase_read_lammps_data
from ..analysis.stats import window_mean_stderr
from ..analysis.trajectory import quench_window_steps, read_last_frames_auto
from ..utils import ensure_dir, stable_file_identity
from .metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
from .metrics_policy import resolve_effective_metrics_config
from .production_common import build_ensemble_cdf_sidecar, graph_analysis_requested
from .progress import CondensedProgressLog, atomic_write_json
from .workflow_lock import WORKFLOW_LOCK_FILENAME, locked_output_workflow
from .resume_integrity import (
    TASK_RESULT_SCHEMA,
    canonical_json_sha256,
    validate_production_resume_state,
    validate_task_result_integrity,
    validate_task_result_replay_artifacts,
)
from ..analysis.provenance import json_sanitize
from ..engine_identity import (
    homogeneous_successful_task_engine_identity,
    validate_engine_build_identity,
    validate_engine_build_identity_bundle,
)


@dataclass(frozen=True)
class DiscoveredBox:
    box: int
    box_dir: Path
    melt_dir: Path
    quench_dir: Path
    relax_dir: Path
    input_structure: Optional[Path]
    final_structure: Optional[Path]
    relax_data: Optional[Path]
    relax_dump: Optional[Path]
    relax_traj: Optional[Path]
    analysis_source: Optional[Path]
    analysis_source_role: Optional[str]
    density: Optional[float]
    density_stderr: Optional[float]
    task_result: Optional[Path]
    source_layout: Optional[str] = None
    source_record: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class AnalysisContext:
    metrics_cfg: StructureMetricsConfig
    type_to_species: Optional[list[str]]
    prod_cfg: ProductionEnsembleConfig
    conv_cfg: ConvergenceConfig
    md_timestep: float
    atom_style: str
    cutoffs: dict[Tuple[int, int], float]
    metric_warnings: list[str]
    effective_metrics: dict[str, Any]
    quench_window_steps_range: Optional[Tuple[float, float]]
    sampling_hint: Optional[dict[str, float]]
    source_selection: Optional[dict[str, Any]] = None
    embed_structures: bool = True
    filter_decision_mode: str = "enforcing"
    analysis_workers: int = 1
    analysis_streaming: bool = True
    analysis_max_in_flight: Optional[int] = None
    # ``None`` is valid for canonical ASE/EXTXYZ/CP2K sources.  Raw LAMMPS
    # dump/data readers require a resolved dimensional style and fail closed.
    lammps_units_style: Optional[str] = None
    engine: str = "lammps"
    # Only a production plan makes persisted cutoffs criterion-defining replay
    # inputs.  Generic config/standalone cutoffs retain their historical
    # fallback semantics.
    exact_plan_cutoffs: bool = False


@dataclass(frozen=True)
class ResolvedBoxSources:
    candidates: tuple[Path, ...]
    input_structure: Optional[Path]
    final_structure: Optional[Path]
    relax_trajectory: Optional[Path]
    analysis_source: Optional[Path]
    analysis_source_role: Optional[str]


def _model_dump_jsonlike(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return dict(obj)
    return {}


def _relpath_or_str(path: Path | None, base: Path) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    b = Path(base)
    try:
        return str(p.relative_to(b))
    except Exception:
        return str(p)


def _path_from_record(value: Any, *, base_dir: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve(strict=False)
    return p



_STRICT_FINAL_RESTART_RE = re.compile(r"^.*-1\.restart$", re.IGNORECASE)


def _has_delimited_token(label: str, token: str) -> bool:
    pattern = rf"(?:^|[^A-Za-z0-9]){re.escape(str(token).lower())}(?=$|[^A-Za-z0-9])"
    return re.search(pattern, str(label).lower()) is not None


def _is_strict_final_restart_name(path_or_name: Path | str) -> bool:
    """Return True only for restart names that unambiguously denote a final frame.

    External Si3N4 datasets commonly use a terminal ``-1.restart`` marker for
    the final/converged snapshot.  This must not be relaxed to ``1.restart`` or
    ``*_001.restart``: those names can denote non-final intermediates and used
    to pollute standalone analyses when ASE-readable files were discovered too
    broadly.
    """

    name = Path(str(path_or_name)).name.lower()
    return bool(name == "restart" or _STRICT_FINAL_RESTART_RE.match(name))


def _strip_strict_final_restart_marker(label: str) -> str:
    """Remove only the terminal final-frame marker before box-id parsing."""

    text = str(label)
    return text[:-2] if text.lower().endswith("-1") else text


_BOX_ID_RE = re.compile(r"(?:^|[^A-Za-z0-9])box[_-]*0*([0-9]+)(?=$|[^A-Za-z0-9])", re.IGNORECASE)
_TRAILING_NUMERIC_TOKEN_RE = re.compile(r"(?:^|[^A-Za-z0-9])0*([0-9]+)$")
_NATURAL_TOKEN_RE = re.compile(r"(\d+)")


def _label_for_box_id(name: str) -> str:
    """Return the filename/dirname label used for conservative box-id parsing.

    We intentionally avoid concatenating every digit in a name because chemical
    formulae such as ``Si3N4_001.data`` are common in flat final-structure
    ensembles. Only explicit ``box_###`` labels or a separated trailing numeric
    token are treated as box identifiers.  For strict final-restart files such
    as ``Si3N4_010-1.restart``, the terminal ``-1`` is a final-frame marker and
    is removed before looking for the box/sample id.
    """

    raw = str(name)
    p = Path(raw)
    suffix = p.suffix
    if suffix and any(ch.isalpha() for ch in suffix):
        label = p.stem
    else:
        label = p.name
    if _is_strict_final_restart_name(p.name):
        label = _strip_strict_final_restart_marker(label)
    return label


def _box_id_from_label(name: str) -> int:
    label = _label_for_box_id(str(name))
    m = _BOX_ID_RE.search(label)
    if m is not None:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    m = _TRAILING_NUMERIC_TOKEN_RE.search(label)
    if m is not None:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _slug_box_id(path: Path) -> int:
    return _box_id_from_label(Path(path).name)


def _resolved_box_id(path: Path, *, fallback: Optional[int] = None) -> int:
    box = _slug_box_id(Path(path))
    if box > 0:
        return int(box)
    return int(0 if fallback is None else fallback)


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = _NATURAL_TOKEN_RE.split(Path(path).name.lower())
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def _next_unused_positive_id(preferred: int, used: set[int]) -> int:
    cand = int(preferred) if int(preferred) > 0 else 1
    while cand in used:
        cand += 1
    return cand


def _cutoffs_dict_from_any(obj: Any) -> dict[Tuple[int, int], float]:
    if obj in (None, ""):
        return {}
    out: dict[Tuple[int, int], float] = {}

    def _add(pair: Any, cutoff: Any, *, label: str) -> None:
        if isinstance(pair, str):
            parts = [part for part in re.split(r"[-_,:\s]+", pair.strip()) if part]
        elif isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)):
            parts = list(pair)
        else:
            raise ValueError(f"{label} pair must contain two type ids")
        if len(parts) != 2:
            raise ValueError(f"{label} pair must contain exactly two type ids")
        try:
            numeric_ids = [float(parts[0]), float(parts[1])]
            a, b = int(numeric_ids[0]), int(numeric_ids[1])
            value = float(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains non-numeric pair/cutoff data") from exc
        if any(not math.isfinite(x) or x != float(int(x)) or int(x) < 1 for x in numeric_ids):
            raise ValueError(f"{label} type ids must be integers >= 1")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} cutoff must be finite and > 0")
        key = (min(a, b), max(a, b))
        if key in out and not math.isclose(out[key], value, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{label} contains conflicting values for pair {key}")
        out[key] = value

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _add(k, v, label=f"cutoffs[{k!r}]")
        return out
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for index, ent in enumerate(obj):
            if not isinstance(ent, Mapping):
                raise ValueError(f"cutoffs[{index}] must be a mapping")
            pair = ent.get("pair", None)
            cutoff = ent.get("cutoff", None)
            if pair is None or cutoff is None:
                raise ValueError(f"cutoffs[{index}] requires pair and cutoff")
            _add(pair, cutoff, label=f"cutoffs[{index}]")
        return out
    raise ValueError("cutoffs must be a mapping or a sequence of {pair, cutoff} records")


def _cutoffs_list_from_dict(obj: Mapping[Tuple[int, int], float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (a, b), c in sorted(dict(obj).items()):
        out.append({"pair": [int(a), int(b)], "cutoff": float(c)})
    return out


def _get_type_to_species(config: RunConfig) -> Optional[list[str]]:
    metrics = config.autotune.metrics
    if metrics.type_to_species is not None:
        return [str(x) for x in metrics.type_to_species]
    pot = getattr(config, "kim", None)
    interactions = getattr(pot, "interactions", None)
    if interactions is not None and interactions != "fixed_types":
        return [str(x) for x in interactions]
    if str(getattr(config, "engine", "lammps")).strip().lower() == "cp2k":
        raise ValueError("engine='cp2k' analysis requires autotune.metrics.type_to_species")
    return None


def _analysis_metrics_config(metrics_cfg: StructureMetricsConfig) -> StructureMetricsConfig:
    elastic_cfg = getattr(metrics_cfg, "elastic", None)
    elastic_update = {"enabled": False}
    if elastic_cfg is not None and hasattr(elastic_cfg, "model_copy"):
        elastic_cfg = elastic_cfg.model_copy(update=elastic_update)
    cfg = metrics_cfg.model_copy(
        deep=True,
        update={
            "collect_during_production_stages": False,
            "stage_timeseries_make_plot": False,
            "elastic": elastic_cfg,
        },
    )
    return StructureMetricsConfig.model_validate(cfg)


def _nested_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _analysis_root_from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the standalone-analysis section from a YAML-like mapping.

    ``analyze-output`` accepts either a full VitriFlow ``RunConfig`` or a
    small analysis-only YAML file for structures produced elsewhere. The latter
    may use an explicit top-level ``analysis:`` block, or may put ``metrics:``,
    ``production:``, and ``convergence:`` at the document root.
    """

    if isinstance(data.get("analysis", None), Mapping):
        return dict(data.get("analysis", {}))
    return dict(data)


_STANDALONE_ANALYSIS_ROOT_KEYS = {
    "metrics",
    "production",
    "convergence",
    "autotune",
    "md",
    "output",
    "graph_rules",
    "type_to_species",
    "species",
    "types",
    "cutoffs",
    "preferred_cutoffs",
    "timestep",
    "atom_style",
    "units_style",
    "lammps_units_style",
    "check_convergence",
    "store_distributions",
    "embed_structures",
    "exclude_coordination_defects",
    "rejects_subdir",
    "analysis_workers",
    "analysis_streaming",
    "analysis_max_in_flight",
    "sources",
    "source_selection",
    "source_discovery",
    "source_globs",
    "source_include_globs",
    "source_exclude_globs",
}
_STANDALONE_SOURCE_KEYS = {
    "include_globs",
    "include",
    "globs",
    "patterns",
    "exclude_globs",
    "exclude",
}


def _validate_standalone_analysis_config(data: Mapping[str, Any], root: Mapping[str, Any]) -> None:
    """Fail closed on ignored standalone-analysis keys and malformed aliases."""

    if "analysis" in data:
        if not isinstance(data.get("analysis"), Mapping):
            raise ValueError("Standalone analysis config field 'analysis' must be a mapping")
        outer_unknown = sorted(str(key) for key in data if str(key) != "analysis")
        if outer_unknown:
            raise ValueError(
                "Unknown standalone analysis top-level key(s) outside 'analysis': "
                + ", ".join(outer_unknown)
            )

    unknown = sorted(str(key) for key in root if str(key) not in _STANDALONE_ANALYSIS_ROOT_KEYS)
    if unknown:
        raise ValueError("Unknown standalone analysis key(s): " + ", ".join(unknown))

    for key in ("metrics", "production", "convergence", "autotune", "md", "output"):
        if key in root and not isinstance(root.get(key), Mapping):
            raise ValueError(f"Standalone analysis field '{key}' must be a mapping")

    autotune = root.get("autotune", {})
    if isinstance(autotune, Mapping):
        autotune_unknown = sorted(
            str(key) for key in autotune if str(key) not in {"metrics", "production", "convergence"}
        )
        if autotune_unknown:
            raise ValueError(
                "Unknown standalone analysis autotune key(s): " + ", ".join(autotune_unknown)
            )
        for key in ("metrics", "production", "convergence"):
            if key in autotune and not isinstance(autotune.get(key), Mapping):
                raise ValueError(f"Standalone analysis field 'autotune.{key}' must be a mapping")
            if key in root and key in autotune and dict(root.get(key, {})) != dict(autotune.get(key, {})):
                raise ValueError(
                    f"Conflicting standalone analysis '{key}' and 'autotune.{key}' blocks"
                )

    species_aliases: list[tuple[str, list[str]]] = []
    for key in ("type_to_species", "species", "types"):
        if key not in root:
            continue
        value = root.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"Standalone analysis field '{key}' must be a sequence of species names")
        species_aliases.append((key, [str(item) for item in value]))
    if species_aliases and len({tuple(value) for _key, value in species_aliases}) > 1:
        raise ValueError(
            "Conflicting standalone species aliases: "
            + ", ".join(f"{key}={value}" for key, value in species_aliases)
        )
    effective_metrics = root.get("metrics") or (autotune.get("metrics") if isinstance(autotune, Mapping) else None)
    if isinstance(effective_metrics, Mapping) and effective_metrics.get("type_to_species") is not None:
        metric_species_raw = effective_metrics.get("type_to_species")
        if not isinstance(metric_species_raw, Sequence) or isinstance(
            metric_species_raw, (str, bytes, bytearray)
        ):
            raise ValueError("Standalone analysis metrics.type_to_species must be a sequence")
        metric_species = [str(item) for item in metric_species_raw]
        if species_aliases and any(value != metric_species for _key, value in species_aliases):
            raise ValueError(
                "Conflicting standalone root species mapping and metrics.type_to_species"
            )

    graph_rules = root.get("graph_rules", None)
    if graph_rules is not None and (
        not isinstance(graph_rules, Sequence)
        or isinstance(graph_rules, (str, bytes, bytearray))
        or not all(isinstance(rule, Mapping) for rule in graph_rules)
    ):
        raise ValueError("Standalone analysis field 'graph_rules' must be a sequence of mappings")
    if (
        graph_rules is not None
        and isinstance(effective_metrics, Mapping)
        and effective_metrics.get("graph_rules") is not None
        and list(graph_rules) != list(effective_metrics.get("graph_rules") or [])
    ):
        raise ValueError("Conflicting standalone root graph_rules and metrics.graph_rules values")

    if "cutoffs" in root and "preferred_cutoffs" in root and root.get("cutoffs") != root.get("preferred_cutoffs"):
        raise ValueError("Conflicting standalone analysis 'cutoffs' and 'preferred_cutoffs' values")

    md = root.get("md", {})
    if isinstance(md, Mapping):
        md_unknown = sorted(str(key) for key in md if str(key) not in {"timestep", "atom_style"})
        if md_unknown:
            raise ValueError("Unknown standalone analysis md key(s): " + ", ".join(md_unknown))
        for key in ("timestep", "atom_style"):
            if key in root and key in md and root.get(key) != md.get(key):
                raise ValueError(f"Conflicting standalone analysis '{key}' and 'md.{key}' values")

    output = root.get("output", {})
    if isinstance(output, Mapping):
        output_unknown = sorted(str(key) for key in output if str(key) != "embed_structures")
        if output_unknown:
            raise ValueError("Unknown standalone analysis output key(s): " + ", ".join(output_unknown))

    source_sections = [key for key in ("sources", "source_selection", "source_discovery") if key in root]
    for key in source_sections:
        value = root.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"Standalone analysis field '{key}' must be a mapping")
        source_unknown = sorted(str(name) for name in value if str(name) not in _STANDALONE_SOURCE_KEYS)
        if source_unknown:
            raise ValueError(
                f"Unknown standalone analysis {key} key(s): " + ", ".join(source_unknown)
            )
        for name, glob_value in value.items():
            valid_glob = isinstance(glob_value, str) or (
                isinstance(glob_value, Sequence)
                and not isinstance(glob_value, (str, bytes, bytearray))
                and all(isinstance(item, str) for item in glob_value)
            )
            if not valid_glob:
                raise ValueError(
                    f"Standalone analysis field '{key}.{name}' must be a string or sequence of strings"
                )
    if len(source_sections) > 1:
        raise ValueError(
            "Standalone analysis source selection must use exactly one of: "
            "sources, source_selection, source_discovery"
        )

    for key in ("source_globs", "source_include_globs", "source_exclude_globs"):
        if key not in root:
            continue
        value = root.get(key)
        valid = isinstance(value, str) or (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and all(isinstance(item, str) for item in value)
        )
        if not valid:
            raise ValueError(f"Standalone analysis field '{key}' must be a string or sequence of strings")


def _standalone_analysis_metric_data(root: Mapping[str, Any]) -> dict[str, Any]:
    autotune = _nested_mapping(root, "autotune")
    metrics = _nested_mapping(root, "metrics") or _nested_mapping(autotune, "metrics")
    if not metrics:
        raise ValueError(
            "Standalone output analysis requires a metrics block. Use either "
            "analysis.metrics: or top-level metrics:."
        )
    metrics = dict(metrics)
    if "graph_rules" not in metrics and root.get("graph_rules", None) is not None:
        metrics["graph_rules"] = list(root.get("graph_rules") or [])
    if "type_to_species" not in metrics:
        for key in ("type_to_species", "species", "types"):
            if key in root and root.get(key) is not None:
                metrics["type_to_species"] = list(root.get(key) or [])
                break
    metrics.setdefault("enabled", True)
    # Standalone ensembles are commonly final-structure snapshots. Using one
    # frame is the least surprising default; trajectory users can opt into a
    # longer tail average explicitly.
    metrics.setdefault("time_average_frames", 1)
    return metrics


def _standalone_analysis_production_data(root: Mapping[str, Any]) -> dict[str, Any]:
    autotune = _nested_mapping(root, "autotune")
    prod = _nested_mapping(root, "production") or _nested_mapping(autotune, "production")
    prod = dict(prod)
    # Analysis-only YAMLs can expose these knobs directly under `analysis:`
    # without introducing a production workflow section.
    for key in ("check_convergence", "store_distributions", "embed_structures", "exclude_coordination_defects", "rejects_subdir", "analysis_workers", "analysis_streaming", "analysis_max_in_flight"):
        if key in root and root.get(key) is not None:
            prod[key] = root.get(key)
    output = _nested_mapping(root, "output")
    if "embed_structures" in output and output.get("embed_structures") is not None:
        prod["embed_structures"] = output.get("embed_structures")
    prod.setdefault("enabled", True)
    prod.setdefault("min_boxes", 1)
    prod.setdefault("batch_boxes", 1)
    prod.setdefault("check_convergence", True)
    prod.setdefault("store_distributions", True)
    # Analysis-only JSON defaults to manifest+sidecar structure retention;
    # production/run-production keep embedded structures through their own config default.
    prod.setdefault("embed_structures", False)
    prod.setdefault("analysis_workers", 1)
    prod.setdefault("analysis_streaming", True)
    prod.setdefault("analysis_max_in_flight", None)
    # External final structures should not be silently discarded unless the
    # user asks for production-style defect rejection.
    prod.setdefault("exclude_coordination_defects", False)
    return prod


def _standalone_analysis_convergence_data(root: Mapping[str, Any]) -> dict[str, Any]:
    autotune = _nested_mapping(root, "autotune")
    return _nested_mapping(root, "convergence") or _nested_mapping(autotune, "convergence")


def _standalone_analysis_md_data(root: Mapping[str, Any]) -> dict[str, Any]:
    md = _nested_mapping(root, "md")
    if "timestep" not in md and root.get("timestep", None) is not None:
        md["timestep"] = root.get("timestep")
    if "atom_style" not in md and root.get("atom_style", None) is not None:
        md["atom_style"] = root.get("atom_style")
    md.setdefault("timestep", 1.0)
    md.setdefault("atom_style", "atomic")
    return md


def _bool_from_analysis_value(value: Any, *, default: bool, field: str) -> bool:
    if value is None:
        raise ValueError(f"{field} must be a boolean; explicit null is not allowed")
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
        raise ValueError(f"{field} must be a boolean (or 0/1); got {value!r}")
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field} must be a boolean; got {value!r}")


def _standalone_analysis_embed_structures(root: Mapping[str, Any]) -> bool:
    """Return whether per-box structure coordinates should be embedded in JSON.

    Standalone ``analyze-output`` can produce very large JSON files when full
    final-frame coordinates are embedded per box.  Its default is therefore
    manifest+sidecar structure retention, while production/run-production keep
    their historical embedded default through ``ProductionEnsembleConfig``.
    Users can set ``embed_structures: true`` either directly under
    ``analysis:`` or under ``analysis.output:`` when fully embedded JSON is
    desired.
    """

    candidates: list[tuple[str, Any]] = []
    if "embed_structures" in root:
        candidates.append(("analysis.embed_structures", root.get("embed_structures")))
    output = _nested_mapping(root, "output")
    if "embed_structures" in output:
        candidates.append(("analysis.output.embed_structures", output.get("embed_structures")))
    production = _nested_mapping(root, "production")
    if "embed_structures" in production:
        candidates.append(("analysis.production.embed_structures", production.get("embed_structures")))
    autotune_production = _nested_mapping(_nested_mapping(root, "autotune"), "production")
    if "embed_structures" in autotune_production:
        candidates.append(
            ("analysis.autotune.production.embed_structures", autotune_production.get("embed_structures"))
        )
    if not candidates:
        return False
    parsed = [
        (
            field,
            _bool_from_analysis_value(value, default=False, field=field),
        )
        for field, value in candidates
    ]
    values = {value for _field, value in parsed}
    if len(values) != 1:
        details = ", ".join(f"{field}={value}" for field, value in parsed)
        raise ValueError(f"Conflicting standalone embed_structures settings: {details}")
    return bool(parsed[0][1])


def analysis_context_from_standalone_config(data: Mapping[str, Any]) -> AnalysisContext:
    """Build an output-analysis context from an analysis-only config mapping.

    This is intended for final structures generated outside VitriFlow/MQFlow,
    where there is no original simulation ``config.yaml``. The mapping still
    needs to define the analysis choices (species/type mapping, metrics, and
    optionally convergence tolerances), but it does not require a potential,
    structure-generation recipe, or MD engine configuration.
    """

    if not isinstance(data, Mapping):
        raise ValueError("Standalone analysis config must be a YAML mapping")
    root = _analysis_root_from_mapping(data)
    _validate_standalone_analysis_config(data, root)
    metrics_cfg = StructureMetricsConfig.model_validate(_standalone_analysis_metric_data(root))
    metrics_cfg = _analysis_metrics_config(metrics_cfg)
    embed_structures = _standalone_analysis_embed_structures(root)
    prod_cfg = ProductionEnsembleConfig.model_validate(_standalone_analysis_production_data(root)).model_copy(
        update={"embed_structures": bool(embed_structures)}
    )
    conv_cfg = ConvergenceConfig.model_validate(_standalone_analysis_convergence_data(root))
    md_cfg = MDConfig.model_validate(_standalone_analysis_md_data(root))
    cutoffs = _cutoffs_dict_from_any(root.get("cutoffs", None) or root.get("preferred_cutoffs", None))
    type_to_species = (
        [str(x) for x in metrics_cfg.type_to_species]
        if metrics_cfg.type_to_species is not None
        else None
    )
    units_candidates = [
        str(root[key]).strip().lower()
        for key in ("units_style", "lammps_units_style")
        if root.get(key, None) not in (None, "")
    ]
    if len(set(units_candidates)) > 1:
        raise ValueError("Conflicting standalone units_style and lammps_units_style values")
    units_style = units_candidates[0] if units_candidates else None
    from ..lammps_units import normalize_lammps_units_style

    if units_style is not None:
        units_style = normalize_lammps_units_style(units_style)
    return AnalysisContext(
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
        prod_cfg=prod_cfg,
        conv_cfg=conv_cfg,
        md_timestep=float(md_cfg.timestep),
        atom_style=str(md_cfg.atom_style),
        cutoffs=dict(cutoffs),
        metric_warnings=[],
        effective_metrics={"source": "standalone_analysis_config"},
        quench_window_steps_range=None,
        sampling_hint=None,
        source_selection=_standalone_analysis_source_selection_data(root),
        embed_structures=bool(embed_structures),
        filter_decision_mode="advisory",
        analysis_workers=int(getattr(prod_cfg, "analysis_workers", 1) or 1),
        analysis_streaming=bool(getattr(prod_cfg, "analysis_streaming", True)),
        analysis_max_in_flight=(None if getattr(prod_cfg, "analysis_max_in_flight", None) is None else int(getattr(prod_cfg, "analysis_max_in_flight"))),
        lammps_units_style=units_style,
        engine=str(root.get("engine", "lammps") or "lammps"),
    )


def _collect_density_stats(relax_dir: Path) -> tuple[Optional[float], Optional[float]]:
    thermo_csv = Path(relax_dir) / "thermo.csv"
    if not thermo_csv.exists():
        return None, None
    try:
        tab = parse_thermo_csv(thermo_csv).as_dict()
        if "Density" not in tab:
            return None, None
        win = window_mean_stderr(tab.get("Density", []), start_fraction=0.5)
        return float(win.mean), float(win.stderr)
    except Exception:
        return None, None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = Decimal(
            str(value).strip().replace("D", "E").replace("d", "e")
        )
    except (InvalidOperation, ValueError):
        return None
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        return None
    return int(numeric)


_ANALYSIS_SOURCE_SUFFIXES = {
    ".extxyz",
    ".xyz",
    ".restart",
    ".lammpstrj",
    ".dump",
    ".trj",
    ".data",
    ".lmp",
    ".lammps",
    ".vasp",
    ".poscar",
    ".contcar",
    ".cif",
    ".pdb",
}

_ANALYSIS_PRIORITY_NAMES = (
    "-1.restart",
    "restart",
    "traj.extxyz",
    "final.extxyz",
    "traj.xyz",
    "final.xyz",
    "relax.lammpstrj",
    "traj.lammpstrj",
    "XDATCAR",
    "CONTCAR",
    "POSCAR",
    "relax.data",
    "output.data",
    "structure.data",
)

_ANALYSIS_GLOB_PATTERNS = (
    "*.restart",
    "*.extxyz",
    "*.xyz",
    "*.lammpstrj",
    "*.dump",
    "*.trj",
    "*.data",
    "*.lmp",
    "*.lammps",
    "*.vasp",
    "*.poscar",
    "*.contcar",
    "*.cif",
    "*.pdb",
)

_ANALYSIS_SKIP_NAMES = {
    WORKFLOW_LOCK_FILENAME,
    "analysis_results.json",
    "autotune_results.json",
    "condensed.log",
    "output_dataset.json",
    "run_results.json",
    "task_result.json",
    "thermo.csv",
}

_ANALYSIS_SKIP_SUFFIXES = {
    ".csv",
    ".json",
    ".jpeg",
    ".jpg",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".zip",
}


def _string_list_from_any(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(x) for x in value if x not in (None, "")]
    return []


def _normalise_source_selection(value: Any) -> Optional[dict[str, Any]]:
    if value in (None, ""):
        return None
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    include = (
        _string_list_from_any(data.get("include_globs", None))
        or _string_list_from_any(data.get("include", None))
        or _string_list_from_any(data.get("globs", None))
        or _string_list_from_any(data.get("patterns", None))
    )
    exclude = _string_list_from_any(data.get("exclude_globs", None)) or _string_list_from_any(data.get("exclude", None))
    if not include and not exclude:
        return None
    return {
        "include_globs": tuple(str(x) for x in include),
        "exclude_globs": tuple(str(x) for x in exclude),
    }


def _standalone_analysis_source_selection_data(root: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    source_sel = None
    for key in ("sources", "source_selection", "source_discovery"):
        if isinstance(root.get(key, None), Mapping):
            source_sel = dict(root.get(key, {}))
            break
    if source_sel is None:
        source_sel = {}
    if "source_globs" in root and "include_globs" not in source_sel:
        source_sel["include_globs"] = root.get("source_globs")
    if "source_include_globs" in root and "include_globs" not in source_sel:
        source_sel["include_globs"] = root.get("source_include_globs")
    if "source_exclude_globs" in root and "exclude_globs" not in source_sel:
        source_sel["exclude_globs"] = root.get("source_exclude_globs")
    return _normalise_source_selection(source_sel)


def _source_selection_for_metadata(source_selection: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(source_selection, Mapping):
        return None
    include = [str(x) for x in _string_list_from_any(source_selection.get("include_globs", None))]
    exclude = [str(x) for x in _string_list_from_any(source_selection.get("exclude_globs", None))]
    if not include and not exclude:
        return None
    return {"include_globs": include, "exclude_globs": exclude}


def _path_matches_any_glob(path: Path, patterns: Sequence[str]) -> bool:
    p = Path(path)
    candidates = [p.name, str(p)]
    try:
        candidates.append(p.as_posix())
    except Exception:
        pass
    for pat in [str(x) for x in patterns if str(x)]:
        for cand in candidates:
            if fnmatch.fnmatch(cand, pat):
                return True
    return False


def _source_selection_allows_path(path: Path, source_selection: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(source_selection, Mapping):
        return True
    include = _string_list_from_any(source_selection.get("include_globs", None))
    exclude = _string_list_from_any(source_selection.get("exclude_globs", None))
    p = Path(path)
    if exclude and _path_matches_any_glob(p, exclude):
        return False
    if include and not _path_matches_any_glob(p, include):
        return False
    return True


def _read_text_head(path: Path, *, max_lines: int = 80) -> list[str]:
    try:
        return Path(path).read_text(errors="replace").splitlines()[: int(max_lines)]
    except Exception:
        return []


def _looks_like_lammps_dump_file(path: Path) -> bool:
    p = Path(path)
    if p.suffix.lower() in {".lammpstrj", ".dump", ".trj"}:
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    up = [str(ln).strip().upper() for ln in head]
    return bool(any(ln.startswith("ITEM: TIMESTEP") for ln in up) and any(ln.startswith("ITEM: ATOMS") for ln in up))


def _looks_like_lammps_data_file(path: Path) -> bool:
    p = Path(path)
    name = p.name.lower()
    if name in {"relax.data", "output.data", "input.data", "structure.data"}:
        return True
    if p.suffix.lower() in {".data", ".lmp", ".dat"}:
        # Conventional LAMMPS data extensions are accepted immediately for
        # backwards compatibility. Reading is still validated later.
        return True
    head = _read_text_head(p, max_lines=80)
    if not head:
        return False
    low = [str(ln).lower() for ln in head]
    atoms_hdr = any(" atoms" in ln for ln in low)
    types_hdr = any(" atom types" in ln for ln in low)
    bounds_hdr = any("xlo xhi" in ln for ln in low)
    atoms_section = any(str(ln).strip().lower().startswith("atoms") for ln in low)
    return bool(atoms_hdr and bounds_hdr and (types_hdr or atoms_section))


def _looks_like_ase_structure_file(path: Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        if int(p.stat().st_size) <= 0:
            return False
    except Exception:
        return False

    try:
        from ase.io import read as ase_read
    except Exception:
        return False

    images = None
    try:
        images = ase_read(str(p), index=-1)
    except Exception:
        try:
            images = ase_read(str(p))
        except Exception:
            return False

    atoms = None
    if isinstance(images, (list, tuple)):
        if not images:
            return False
        atoms = images[-1]
    else:
        atoms = images
    if atoms is None:
        return False

    try:
        n_atoms = int(len(atoms))
    except Exception:
        try:
            n_atoms = int(atoms.get_global_number_of_atoms())
        except Exception:
            return False
    if n_atoms <= 0:
        return False

    try:
        cell = np.asarray(atoms.get_cell(), dtype=float)
    except Exception:
        return False
    return bool(cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1.0e-12)


def _atoms_has_valid_periodic_cell(atoms: Any) -> bool:
    if atoms is None:
        return False
    try:
        n_atoms = int(len(atoms))
    except Exception:
        try:
            n_atoms = int(atoms.get_global_number_of_atoms())
        except Exception:
            return False
    if n_atoms <= 0:
        return False
    try:
        cell = np.asarray(atoms.get_cell(), dtype=float)
    except Exception:
        return False
    return bool(cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1.0e-12)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        if hasattr(row, key):
            value = getattr(row, key)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        if isinstance(row, Mapping) and key in row:
            return row.get(key, default)
    except Exception:
        pass
    try:
        value = row.get(key, default)
        if value is not None:
            return value
    except Exception:
        pass
    kvp = None
    try:
        kvp = getattr(row, "key_value_pairs", None)
    except Exception:
        kvp = None
    if isinstance(kvp, Mapping) and key in kvp:
        return kvp.get(key, default)
    data = None
    try:
        data = getattr(row, "data", None)
    except Exception:
        data = None
    if isinstance(data, Mapping) and key in data:
        return data.get(key, default)
    return default


def _looks_like_ase_database_file(path: Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        return False
    # ASE database is the intended interpretation of these suffixes.  A later
    # read step will provide a precise error if the file is not a valid ASE DB.
    return True


def _open_ase_database(path: Path):
    try:
        from ase.db import connect
    except Exception as exc:
        raise RuntimeError("ASE database input requires ase.db") from exc
    return connect(str(path))


def _ase_database_rows(path: Path) -> list[Any]:
    db = _open_ase_database(path)
    try:
        return list(db.select())
    except TypeError:
        return list(db.select(None))


def _ase_database_row_atoms(path: Path, source_record: Mapping[str, Any]):
    db = _open_ase_database(path)
    row = None
    row_id = source_record.get("row_id", None) or source_record.get("id", None)
    if row_id not in (None, ""):
        try:
            row = db.get(id=int(row_id))
        except TypeError:
            row = db.get(int(row_id))
    if row is None:
        rows = _ase_database_rows(path)
        row_index = int(source_record.get("row_index", 1) or 1)
        if row_index < 1 or row_index > len(rows):
            raise IndexError(f"ASE database row_index out of range: {row_index}")
        row = rows[row_index - 1]
    try:
        return row.toatoms()
    except Exception as exc:
        raise RuntimeError(f"ASE database row could not be converted to Atoms: {path}") from exc


def _ase_database_record_for_row(path: Path, row: Any, row_index: int) -> dict[str, Any]:
    row_id = _row_value(row, "id", None)
    name = _row_value(row, "name", None)
    label = None
    for key in ("box", "box_id", "label", "structure_id", "name", "uid", "unique_id"):
        value = _row_value(row, key, None)
        if value not in (None, ""):
            label = str(value)
            break
    if label in (None, "") and name not in (None, ""):
        label = str(name)
    if label in (None, "") and row_id not in (None, ""):
        label = str(row_id)
    rec: dict[str, Any] = {
        "kind": "ase_database_row",
        "database": str(Path(path).resolve(strict=False)),
        "row_index": int(row_index),
    }
    if row_id not in (None, ""):
        try:
            rec["row_id"] = int(row_id)
        except Exception:
            rec["row_id"] = str(row_id)
    if name not in (None, ""):
        rec["row_name"] = str(name)
    if label not in (None, ""):
        rec["row_label"] = str(label)
    return rec


def _box_id_from_ase_database_record(rec: Mapping[str, Any], *, fallback: int) -> int:
    for key in ("row_label", "row_name"):
        value = rec.get(key, None)
        if value not in (None, ""):
            box = _box_id_from_label(str(value))
            if box > 0:
                return int(box)
    row_id = rec.get("row_id", None)
    try:
        row_id_i = int(row_id)
        if row_id_i > 0:
            return row_id_i
    except Exception:
        pass
    return int(fallback)


def _box_from_ase_database_record(
    database_path: Path,
    *,
    source_record: Mapping[str, Any],
    box: int,
) -> DiscoveredBox:
    db_path = Path(database_path)
    return _build_discovered_box(
        box=int(box),
        box_dir=db_path.parent,
        melt_dir=db_path.parent / "melt",
        quench_dir=db_path.parent / "quench",
        relax_dir=db_path.parent,
        input_structure=None,
        final_structure=db_path,
        relax_data=db_path,
        relax_dump=None,
        relax_traj=db_path,
        analysis_source=db_path,
        analysis_source_role="final_structure",
        density=None,
        density_stderr=None,
        task_result=None,
        source_layout="ase_database",
        source_record=source_record,
    )


def _discover_from_ase_database_file(path: Path) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    p = Path(path).resolve()
    rows = _ase_database_rows(p)
    raw_boxes: list[DiscoveredBox] = []
    rejected: list[dict[str, Any]] = []
    used: set[int] = set()
    for idx, row in enumerate(rows, start=1):
        rec = _ase_database_record_for_row(p, row, idx)
        try:
            atoms = row.toatoms()
            if not _atoms_has_valid_periodic_cell(atoms):
                raise ValueError("row is missing a valid periodic cell")
        except Exception as exc:
            rejected.append(
                {
                    "box": int(idx),
                    "reason": "ase_database_row_unreadable",
                    "error": str(exc),
                    "source_record": dict(rec),
                    "paths": {"database": str(p)},
                }
            )
            continue
        preferred = _box_id_from_ase_database_record(rec, fallback=idx)
        box_id = _next_unused_positive_id(preferred, used)
        used.add(int(box_id))
        raw_boxes.append(_box_from_ase_database_record(p, source_record=rec, box=box_id))
    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(p),
        "layout": "ase_database",
        "n_database_rows": int(len(rows)),
        "n_database_rows_accepted": int(len(raw_boxes)),
        "n_database_rows_rejected": int(len(rejected)),
    }
    return raw_boxes, [], rejected, dataset


def _frames_from_ase_database_box(
    box: DiscoveredBox,
    *,
    type_to_species: Optional[Sequence[str]],
) -> list[Any]:
    source = box.analysis_source or box.relax_data or box.relax_traj
    if source is None or box.source_record is None:
        raise ValueError("ASE database box is missing source_record metadata")
    atoms = _ase_database_row_atoms(Path(source), box.source_record)
    from ..analysis.trajectory import _atoms_to_dumpframe

    return [_atoms_to_dumpframe(atoms, type_to_species=type_to_species, timestep=int(box.box))]


def _materialize_ase_database_box_source(box: DiscoveredBox, *, outdir: Path) -> Optional[Path]:
    if str(box.source_layout or "") != "ase_database":
        return None
    source = box.analysis_source or box.relax_data or box.relax_traj
    if source is None or box.source_record is None:
        raise ValueError("ASE database box is missing source_record metadata")
    atoms = _ase_database_row_atoms(Path(source), box.source_record)
    dest_dir = Path(outdir) / "ase_database_sources" / Path(source).stem
    ensure_dir(dest_dir)
    dest = dest_dir / f"box_{int(box.box):03d}.extxyz"
    try:
        from ase.io import write as ase_write
    except Exception as exc:
        raise RuntimeError("Materialising ASE database rows requires ase.io.write") from exc
    ase_write(str(dest), atoms, format="extxyz")
    return dest


def _is_analysis_source_candidate(path: Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False

    name_up = p.name.upper()
    if name_up in {"CONTCAR", "POSCAR", "XDATCAR"}:
        return True
    if name_up.startswith("CONTCAR.") or name_up.startswith("POSCAR.") or name_up.startswith("XDATCAR."):
        return True

    name_low = p.name.lower()
    if name_low in _ANALYSIS_SKIP_NAMES:
        return False
    if name_low.startswith("log.") or name_low.endswith(".in.lammps") or name_low in {"in.lammps", "input.in"}:
        return False

    suffix = p.suffix.lower()
    if suffix == ".restart" or name_low == "restart":
        return _is_strict_final_restart_name(p.name)
    if _looks_like_ase_database_file(p):
        return True
    if suffix == ".lammps":
        return _looks_like_lammps_dump_file(p) or _looks_like_lammps_data_file(p)
    if suffix in _ANALYSIS_SOURCE_SUFFIXES:
        return True
    if _looks_like_lammps_dump_file(p) or _looks_like_lammps_data_file(p):
        return True
    # Most obvious non-structure artefacts are rejected by explicit filename
    # checks above.  For all other files, including suffixes such as .cfg,
    # .gen, .traj, .res, and even ASE JSON, probe ASE directly so a flat
    # directory can contain any periodic ASE-readable final-structure format.
    if suffix in _ANALYSIS_SKIP_SUFFIXES:
        return _looks_like_ase_structure_file(p)
    return _looks_like_ase_structure_file(p)


def _iter_analysis_source_candidates(directory: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> list[Path]:
    d = Path(directory)
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        p = Path(path)
        if p in seen:
            return
        if _is_analysis_source_candidate(p) and _source_selection_allows_path(p, source_selection):
            out.append(p)
            seen.add(p)

    for name in _ANALYSIS_PRIORITY_NAMES:
        _add(d / name)
    for stem in ("XDATCAR", "CONTCAR", "POSCAR"):
        for cand in sorted(d.glob(f"{stem}*")):
            _add(cand)
    for pattern in _ANALYSIS_GLOB_PATTERNS:
        for cand in sorted(d.glob(pattern)):
            _add(cand)
    try:
        for cand in sorted(d.iterdir(), key=lambda p: p.name):
            _add(cand)
    except Exception:
        pass
    return out


def _is_dump_like(path: Path) -> bool:
    return _looks_like_lammps_dump_file(Path(path))


def _sort_flat_sources(sources: Sequence[Path]) -> list[Path]:
    def _key(path: Path) -> tuple[int, int, tuple[Any, ...]]:
        box = _slug_box_id(Path(path))
        return (0 if box > 0 else 1, int(box if box > 0 else 0), _natural_sort_key(Path(path)))

    return sorted([Path(p) for p in sources], key=_key)


def _flat_source_box_assignments(sources: Sequence[Path], *, used: Optional[set[int]] = None) -> list[tuple[Path, int]]:
    assignments: list[tuple[Path, int]] = []
    used_ids: set[int] = set(int(x) for x in (used or set()))
    for idx, source in enumerate(_sort_flat_sources(sources), start=1):
        candidate = _slug_box_id(Path(source))
        if candidate <= 0 or candidate in used_ids:
            candidate = _next_unused_positive_id(idx, used_ids)
        used_ids.add(int(candidate))
        assignments.append((Path(source), int(candidate)))
    if used is not None:
        used.update(used_ids)
    return assignments


# Strict restart-final matching is provided by _is_strict_final_restart_name.


def _source_role_score(path: Path, *, role: str) -> int:
    p = Path(path)
    name_low = p.name.lower()
    stem_low = p.stem.lower()
    suffix = p.suffix.lower()
    is_dump = _is_dump_like(p)
    is_restart = _is_strict_final_restart_name(p.name)
    is_contcar = bool(p.name.upper() == "CONTCAR" or p.name.upper().startswith("CONTCAR."))
    is_poscar = bool(p.name.upper() == "POSCAR" or p.name.upper().startswith("POSCAR."))
    is_xdatcar = bool(p.name.upper() == "XDATCAR" or p.name.upper().startswith("XDATCAR."))

    final_keywords = ("final", "relaxed", "optimized", "optimised", "converged", "last", "endpoint")
    traj_keywords = ("traj", "trajectory", "history", "movie", "path", "dump", "xdatcar")
    input_keywords = ("input", "initial", "seed", "start", "origin", "orig", "source")

    has_final_kw = any(tok in name_low for tok in final_keywords)
    has_traj_kw = any(tok in name_low for tok in traj_keywords)
    # Use token-aware matching for input/start labels.  A strict final restart
    # filename contains the substring "start" inside "restart", which must not
    # demote it below POSCAR/CONTCAR during final-source resolution.
    has_input_kw = any(_has_delimited_token(stem_low, tok) for tok in input_keywords)

    if role == "final_structure":
        score = 0
        if is_restart and name_low.endswith("-1.restart"):
            score = max(score, 1400)
        if is_restart:
            score = max(score, 1300)
        if is_contcar:
            score = max(score, 1250)
        if is_poscar:
            score = max(score, 1200)
        if has_final_kw:
            score = max(score, 1100)
        if suffix in {".data", ".lmp", ".lammps", ".dat", ".cif", ".pdb", ".vasp", ".poscar", ".contcar"}:
            score = max(score, 850)
        if name_low in {"relax.data", "output.data", "structure.data"}:
            score = max(score, 900)
        if is_dump or has_traj_kw or is_xdatcar:
            score -= 500
        if has_input_kw:
            score -= 700
        if suffix in {".extxyz", ".xyz"} and not has_final_kw and not is_restart:
            score -= 150
        return score

    if role == "relax_trajectory":
        score = 0
        if is_dump:
            score = max(score, 1300)
        if name_low in {"traj.extxyz", "traj.xyz", "trajectory.extxyz", "trajectory.xyz", "relax.lammpstrj", "traj.lammpstrj"}:
            score = max(score, 1250)
        if is_xdatcar:
            score = max(score, 1200)
        if has_traj_kw:
            score = max(score, 1100)
        if suffix == ".extxyz":
            score = max(score, 900)
        if suffix == ".xyz":
            score = max(score, 750)
        if is_restart or is_contcar or is_poscar or has_final_kw:
            score -= 600
        if has_input_kw:
            score -= 400
        return score

    if role == "input_structure":
        score = 0
        if has_input_kw:
            score = max(score, 1300)
        if stem_low in {"input", "initial", "seed", "start"}:
            score = max(score, 1350)
        if is_restart or is_contcar or is_poscar or has_final_kw:
            score -= 800
        if is_dump or has_traj_kw or is_xdatcar:
            score -= 500
        return score

    raise ValueError(f"Unknown source role: {role}")


def _candidate_tiebreak_key(path: Path) -> tuple[int, str]:
    p = Path(path)
    suffix = p.suffix.lower()
    priority = 0
    if _is_strict_final_restart_name(p.name):
        priority = 30
    elif p.name.upper() == "CONTCAR":
        priority = 25
    elif p.name.upper() == "POSCAR":
        priority = 24
    elif suffix in {".data", ".lmp", ".lammps", ".dat"}:
        priority = 20
    elif suffix == ".extxyz":
        priority = 15
    elif suffix == ".xyz":
        priority = 10
    return (priority, p.name)


def _pick_best_candidate(candidates: Sequence[Path], *, role: str, exclude: Optional[set[Path]] = None) -> Optional[Path]:
    excluded = set(exclude or set())
    best: Optional[Path] = None
    best_key: Optional[tuple[int, tuple[int, str]]] = None
    for cand in candidates:
        p = Path(cand)
        if p in excluded:
            continue
        score = int(_source_role_score(p, role=role))
        if score <= 0:
            continue
        key = (score, _candidate_tiebreak_key(p))
        if best_key is None or key > best_key:
            best = p
            best_key = key
    return best


def _is_low_confidence_relax_data_final(path: Path) -> bool:
    p = Path(path)
    name_low = p.name.lower()
    suffix = p.suffix.lower()
    if _is_strict_final_restart_name(p.name):
        return False
    if p.name.upper().startswith(("CONTCAR", "POSCAR")):
        return False
    explicit_final_tokens = ("final", "relaxed", "optimized", "optimised", "converged", "last", "endpoint")
    if any(tok in name_low for tok in explicit_final_tokens):
        return False
    return name_low in {"relax.data", "output.data", "structure.data"}


def _is_canonical_relax_trajectory(path: Path) -> bool:
    p = Path(path)
    name_low = p.name.lower()
    if name_low in {"traj.extxyz", "traj.xyz", "trajectory.extxyz", "trajectory.xyz", "relax.lammpstrj", "traj.lammpstrj"}:
        return True
    return bool(_is_dump_like(p))


def _prefer_legacy_relax_trajectory_source(
    relax_dir: Path,
    *,
    final_structure: Optional[Path],
    relax_trajectory: Optional[Path],
) -> bool:
    """Keep VitriFlow-generated box directories on their original analysis path.

    Historical VitriFlow production analysis reads the relaxation trajectory
    (usually ``relax/traj.extxyz``) for time-averaged structural metrics.  The
    generic final-structure discovery added for external ensembles should not
    redirect those canonical MD directories to ``relax.data`` just because that
    file is also ASE-readable.  High-confidence final structures such as
    ``*-1.restart``, ``CONTCAR`` or explicitly named ``final.*`` still win.
    """

    if final_structure is None or relax_trajectory is None:
        return False
    if Path(relax_dir).name.lower() != "relax":
        return False
    return _is_low_confidence_relax_data_final(Path(final_structure)) and _is_canonical_relax_trajectory(Path(relax_trajectory))


def _resolve_box_sources(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> ResolvedBoxSources:
    d = Path(relax_dir)
    cands = _iter_analysis_source_candidates(d, source_selection=source_selection)
    input_structure = _pick_best_candidate(cands, role="input_structure")
    final_structure = _pick_best_candidate(cands, role="final_structure")
    relax_trajectory = _pick_best_candidate(
        cands,
        role="relax_trajectory",
        exclude=({final_structure} if final_structure is not None else None),
    )

    analysis_source: Optional[Path] = None
    analysis_role: Optional[str] = None
    if _prefer_legacy_relax_trajectory_source(
        d,
        final_structure=final_structure,
        relax_trajectory=relax_trajectory,
    ):
        analysis_source = relax_trajectory
        analysis_role = "relax_trajectory"
    elif final_structure is not None:
        analysis_source = final_structure
        analysis_role = "final_structure"
    elif relax_trajectory is not None:
        analysis_source = relax_trajectory
        analysis_role = "relax_trajectory"
    else:
        fallback = next((Path(c) for c in cands if Path(c) != input_structure), None)
        if fallback is None and input_structure is not None:
            fallback = Path(input_structure)
        analysis_source = fallback
        if analysis_source is not None:
            analysis_role = "single_structure"

    return ResolvedBoxSources(
        candidates=tuple(Path(c) for c in cands),
        input_structure=input_structure,
        final_structure=final_structure,
        relax_trajectory=relax_trajectory,
        analysis_source=analysis_source,
        analysis_source_role=analysis_role,
    )


def _guess_analysis_source(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    return _resolve_box_sources(relax_dir, source_selection=source_selection).analysis_source


def _guess_analysis_source_role(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    return _resolve_box_sources(relax_dir, source_selection=source_selection).analysis_source_role


def _guess_input_structure(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    return _resolve_box_sources(relax_dir, source_selection=source_selection).input_structure


def _guess_final_structure(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    return _resolve_box_sources(relax_dir, source_selection=source_selection).final_structure


def _guess_relax_data(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir, source_selection=source_selection)
    if resolved.final_structure is not None:
        return resolved.final_structure
    if resolved.analysis_source is not None and not _is_dump_like(resolved.analysis_source):
        return resolved.analysis_source
    d = Path(relax_dir)
    for cand in (d / "relax.data", d / "output.data"):
        if cand.exists() and _source_selection_allows_path(cand, source_selection):
            return cand
    return next((Path(c) for c in resolved.candidates if not _is_dump_like(c)), None)


def _guess_relax_dump(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir, source_selection=source_selection)
    if resolved.relax_trajectory is not None and _is_dump_like(resolved.relax_trajectory):
        return resolved.relax_trajectory
    d = Path(relax_dir)
    for cand in (d / "relax.lammpstrj", d / "traj.lammpstrj"):
        if cand.exists() and _source_selection_allows_path(cand, source_selection):
            return cand
    dumps = sorted(
        p
        for p in (list(d.glob("*.lammpstrj")) + list(d.glob("*.dump")) + list(d.glob("*.trj")))
        if _source_selection_allows_path(p, source_selection)
    )
    return dumps[0] if dumps else None


def _guess_relax_traj(relax_dir: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir, source_selection=source_selection)
    if resolved.relax_trajectory is not None:
        return resolved.relax_trajectory
    return resolved.analysis_source


def _build_discovered_box(
    *,
    box: int,
    box_dir: Path,
    melt_dir: Path,
    quench_dir: Path,
    relax_dir: Path,
    input_structure: Optional[Path],
    final_structure: Optional[Path],
    relax_data: Optional[Path],
    relax_dump: Optional[Path],
    relax_traj: Optional[Path],
    density: Optional[float],
    density_stderr: Optional[float],
    task_result: Optional[Path],
    analysis_source: Optional[Path] = None,
    analysis_source_role: Optional[str] = None,
    source_layout: Optional[str] = None,
    source_record: Optional[Mapping[str, Any]] = None,
) -> DiscoveredBox:
    source = Path(analysis_source) if analysis_source is not None else (relax_traj or relax_data)
    rdata = Path(relax_data) if relax_data is not None else source
    rtraj = Path(relax_traj) if relax_traj is not None else source
    rdump = Path(relax_dump) if relax_dump is not None else (rtraj if rtraj is not None and _is_dump_like(rtraj) else None)
    return DiscoveredBox(
        box=int(box),
        box_dir=Path(box_dir),
        melt_dir=Path(melt_dir),
        quench_dir=Path(quench_dir),
        relax_dir=Path(relax_dir),
        input_structure=(Path(input_structure) if input_structure is not None else None),
        final_structure=(Path(final_structure) if final_structure is not None else None),
        relax_data=rdata,
        relax_dump=rdump,
        relax_traj=rtraj,
        analysis_source=source,
        analysis_source_role=(None if analysis_source_role in (None, "") else str(analysis_source_role)),
        density=density,
        density_stderr=density_stderr,
        task_result=task_result,
        source_layout=(None if source_layout in (None, "") else str(source_layout)),
        source_record=(None if source_record is None else dict(source_record)),
    )


def _authoritative_task_density(
    task_result: Optional[Path],
) -> tuple[Optional[float], Optional[float]]:
    """Read the StageOutcome density contract persisted by an HPC task.

    Re-estimating density from the final structure is not equivalent to the
    relaxation-window thermo mean used by local production.  Successful task
    results therefore remain authoritative when those fields are present.
    Legacy/minimal task results without them fall back to relax/thermo.csv.
    """

    if task_result is None or not Path(task_result).is_file():
        return None, None
    payload = _read_task_result_payload(Path(task_result))
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"ok", "success"}:
        return None, None
    has_density = "density" in payload
    has_stderr = "density_stderr" in payload
    if not has_density and not has_stderr:
        return None, None
    if not (has_density and has_stderr):
        raise ValueError(
            "Successful task result must persist density and density_stderr together: "
            f"{task_result}"
        )
    try:
        density = float(payload.get("density"))
        stderr = float(payload.get("density_stderr"))
    except Exception as exc:
        raise ValueError(f"Task density metadata is not numeric: {task_result}") from exc
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError(
            f"Task density must be finite and > 0: {task_result}"
        )
    if not math.isfinite(stderr) or stderr < 0.0:
        raise ValueError(
            f"Task density_stderr must be finite and >= 0: {task_result}"
        )
    return float(density), float(stderr)


def _validated_task_diagnostics(
    task_result: Optional[Path],
    *,
    result_base: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Load the exact local-equivalent diagnostics persisted by an HPC task.

    ``analyze-output`` is a read-only replay.  Re-running stage collectors in
    the source melt/quench/relax directories would both mutate its input and
    make the external result differ from local production.  A protected task
    diagnostic plan is therefore validated and its outputs are reused exactly.
    """

    if task_result is None or not Path(task_result).is_file():
        return None
    result_path = Path(task_result)
    result = _read_task_result_payload(result_path)
    if str(result.get("status", "")).strip().lower() not in {"ok", "success"}:
        return None
    # Legacy task results remain useful as generic structural inputs, but
    # their adjacent task/diagnostic artifacts were not authenticated by the
    # current v3 manifest contract.  Reusing those diagnostics would turn an
    # unverified CSV/JSON into a false exact-parity claim.
    if str(result.get("schema", "")) != TASK_RESULT_SCHEMA:
        return None
    diagnostics = result.get("diagnostics", None)
    if diagnostics is None:
        # Backward-compatible task results predate the diagnostic contract.
        # They remain analysable, but cannot claim external/local diagnostic
        # parity and are labelled explicitly by the caller.
        raise ValueError(
            f"Current task result lacks its protected diagnostics: {result_path}"
        )
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"Task diagnostics must be a mapping: {result_path}")
    if str(diagnostics.get("schema", "")) != "vitriflow.production_task_diagnostics.v1":
        raise ValueError(f"Unsupported task diagnostics schema: {result_path}")
    if str(diagnostics.get("path_base", "")) != "task_box":
        raise ValueError(
            f"Task diagnostic paths must declare path_base='task_box': {result_path}"
        )
    if str(diagnostics.get("status", "")).strip().lower() != "ok":
        raise ValueError(
            f"Task diagnostics did not complete successfully: {result_path}"
        )
    plan = diagnostics.get("plan", None)
    if not isinstance(plan, Mapping) or str(plan.get("schema", "")) != (
        "vitriflow.production_task_diagnostic_plan.v1"
    ):
        raise ValueError(f"Task diagnostics lack a valid protected plan: {result_path}")

    task_json = result_path.parent / "task.json"
    if not task_json.is_file():
        raise ValueError(
            f"Task diagnostics cannot be verified without their task manifest: {task_json}"
        )
    try:
        declared_task = json.loads(task_json.read_text())
    except Exception as exc:
        raise ValueError(f"Could not parse declared task manifest: {task_json}") from exc
    if not isinstance(declared_task, Mapping):
        raise ValueError(f"Task manifest must be a mapping: {task_json}")
    if result_path.is_symlink() or task_json.is_symlink():
        raise ValueError(
            f"Current task replay metadata must be regular files: {result_path}"
        )
    try:
        validate_task_result_replay_artifacts(
            result=result,
            task_data=declared_task,
            box_dir=result_path.parent,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Current task replay authentication failed: {result_path}: {exc}"
        ) from exc
    declared_plan = declared_task.get("diagnostic_plan", None)
    if not isinstance(declared_plan, Mapping) or dict(declared_plan) != dict(plan):
        raise ValueError(
            f"Task diagnostics disagree with the declared task plan: {result_path}"
        )

    def _require_role_payloads(
        family: str,
        payload: Any,
        role_plan: Mapping[str, Any],
    ) -> Any:
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError(f"Task diagnostic family {family!r} must be a role mapping")
        role_payloads = dict(payload or {})
        extra_roles = sorted(set(str(role) for role in role_payloads) - set(str(role) for role in role_plan))
        if extra_roles:
            raise ValueError(
                f"Task diagnostic family {family!r} has undeclared roles: "
                + ", ".join(extra_roles)
            )
        for role, controls_raw in role_plan.items():
            controls = dict(controls_raw or {}) if isinstance(controls_raw, Mapping) else {}
            enabled = bool(controls.get("enabled", False))
            value = role_payloads.get(str(role), None)
            if enabled:
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Task diagnostic {family}.{role} was planned but is absent"
                    )
                if str(value.get("status", "")).strip().lower() != "ok":
                    raise ValueError(
                        f"Task diagnostic {family}.{role} was planned but is not ok"
                    )
            elif value is not None:
                raise ValueError(
                    f"Task diagnostic {family}.{role} was not planned but has a payload"
                )
        return role_payloads if payload is not None else None

    stage_plan = dict(plan.get("stage_metrics", {}) or {})
    stage_metrics = diagnostics.get("stage_metrics", None)
    if bool(stage_plan.get("enabled", False)):
        roles = {str(role): {"enabled": True} for role in list(stage_plan.get("roles", []) or [])}
        if set(roles) != {"melt", "quench", "relax"}:
            raise ValueError(
                f"Enabled task stage-metric plan must cover melt/quench/relax: {result_path}"
            )
        stage_metrics = _require_role_payloads("stage_metrics", stage_metrics, roles)
    elif stage_metrics is not None:
        raise ValueError(
            f"Task stage metrics were produced despite a disabled plan: {result_path}"
        )

    elastic_screens_plan = dict(plan.get("elastic_screens", {}) or {})
    elastic_screens = _require_role_payloads(
        "elastic_screens",
        diagnostics.get("elastic_screens", None),
        dict(elastic_screens_plan.get("roles", {}) or {}),
    )
    elastic_timeseries_plan = dict(plan.get("elastic_timeseries", {}) or {})
    elastic_timeseries = _require_role_payloads(
        "elastic_timeseries",
        diagnostics.get("elastic_timeseries", None),
        dict(elastic_timeseries_plan.get("roles", {}) or {}),
    )

    def _rebase_artifact_paths(value: Any, *, key: Optional[str] = None) -> Any:
        if isinstance(value, Mapping):
            return {
                str(child_key): _rebase_artifact_paths(
                    child_value, key=str(child_key)
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [_rebase_artifact_paths(child, key=key) for child in value]
        if key in {"csv", "summary", "plot"} and isinstance(value, str) and value.strip():
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Task diagnostic artifact path is not task-box-relative: {value!r}"
                )
            resolved = result_path.parent / relative
            if not resolved.is_file():
                raise ValueError(
                    f"Task diagnostic artifact is missing: {resolved}"
                )
            if result_base is None:
                return str(relative)
            return str(os.path.relpath(resolved, Path(result_base)))
        return value

    stage_metrics = _rebase_artifact_paths(stage_metrics)
    elastic_screens = _rebase_artifact_paths(elastic_screens)
    elastic_timeseries = _rebase_artifact_paths(elastic_timeseries)

    return {
        "stage_metrics": stage_metrics,
        "elastic_melt": (elastic_screens or {}).get("melt"),
        "elastic_relax": (elastic_screens or {}).get("relax"),
        "elastic_timeseries": elastic_timeseries,
        "provenance": {
            "schema": "vitriflow.reused_task_diagnostics.v1",
            "mode": "validated_read_only_reuse",
            "task_result": str(result_path),
            "task_manifest": str(task_json) if task_json.is_file() else None,
            "source_path_base": "task_box",
            "result_path_base": (
                None if result_base is None else str(Path(result_base))
            ),
            "diagnostic_plan": dict(plan),
        },
    }


def _box_from_source_file(
    source_path: Path,
    *,
    task_result: Optional[Path] = None,
    box: Optional[int] = None,
    explicit_box: bool = False,
    analysis_source_role: str = "single_structure",
    source_layout: Optional[str] = None,
    source_record: Optional[Mapping[str, Any]] = None,
) -> DiscoveredBox:
    src = Path(source_path)
    if src.parent.name == "relax" and src.parent.parent.exists():
        box_dir = src.parent.parent
        relax_dir = src.parent
    else:
        box_dir = src.parent
        relax_dir = src.parent
    density, density_stderr = _collect_density_stats(relax_dir)
    if explicit_box and box is not None and int(box) > 0:
        box_id = int(box)
    else:
        box_id = _resolved_box_id(src, fallback=box)
    if box_dir.name.lower().startswith("box"):
        box_id = _resolved_box_id(box_dir, fallback=box_id)
    return _build_discovered_box(
        box=box_id,
        box_dir=box_dir,
        melt_dir=box_dir / "melt",
        quench_dir=box_dir / "quench",
        relax_dir=relax_dir,
        input_structure=None,
        final_structure=src,
        relax_data=src,
        relax_dump=(src if _is_dump_like(src) else None),
        relax_traj=src,
        analysis_source=src,
        analysis_source_role=str(analysis_source_role or "single_structure"),
        density=density,
        density_stderr=density_stderr,
        task_result=task_result,
        source_layout=source_layout,
        source_record=source_record,
    )


def _discovered_box_matches_source_selection(
    box: DiscoveredBox,
    source_selection: Optional[Mapping[str, Any]],
) -> bool:
    if not isinstance(source_selection, Mapping):
        return True
    candidate_paths = (
        box.analysis_source,
        box.final_structure,
        box.relax_traj,
        box.relax_data,
        box.relax_dump,
    )
    return any(
        p is not None and _source_selection_allows_path(Path(p), source_selection)
        for p in candidate_paths
    )


def _box_from_dirs(
    box_dir: Path,
    *,
    task_result: Optional[Path] = None,
    box: Optional[int] = None,
    melt_dir: Optional[Path] = None,
    quench_dir: Optional[Path] = None,
    relax_dir: Optional[Path] = None,
    relax_data: Optional[Path] = None,
    relax_dump: Optional[Path] = None,
    relax_traj: Optional[Path] = None,
    density: Optional[float] = None,
    density_stderr: Optional[float] = None,
    analysis_source: Optional[Path] = None,
    analysis_source_role: Optional[str] = None,
    input_structure: Optional[Path] = None,
    final_structure: Optional[Path] = None,
    source_layout: Optional[str] = None,
    source_record: Optional[Mapping[str, Any]] = None,
    source_selection: Optional[Mapping[str, Any]] = None,
) -> DiscoveredBox:
    bdir = Path(box_dir)
    melt_use = Path(melt_dir) if melt_dir is not None else (bdir / "melt")
    quench_use = Path(quench_dir) if quench_dir is not None else (bdir / "quench")
    relax_use = Path(relax_dir) if relax_dir is not None else (bdir / "relax")
    if relax_dir is None and not relax_use.exists():
        relax_use = bdir

    def _selected_explicit_path(value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        p = Path(value)
        return p if _source_selection_allows_path(p, source_selection) else None

    resolved = _resolve_box_sources(relax_use, source_selection=source_selection)
    explicit_analysis_source = _selected_explicit_path(analysis_source)
    input_structure_use = _selected_explicit_path(input_structure) or resolved.input_structure
    final_structure_use = _selected_explicit_path(final_structure) or resolved.final_structure
    relax_data_use = (
        _selected_explicit_path(relax_data)
        or resolved.final_structure
        or _guess_relax_data(relax_use, source_selection=source_selection)
    )
    relax_dump_use = _selected_explicit_path(relax_dump) or _guess_relax_dump(
        relax_use,
        source_selection=source_selection,
    )
    relax_traj_use = _selected_explicit_path(relax_traj) or _guess_relax_traj(
        relax_use,
        source_selection=source_selection,
    )
    analysis_source_use = explicit_analysis_source or resolved.analysis_source or relax_data_use or relax_traj_use
    analysis_source_role_use = (
        str(analysis_source_role or "single_structure")
        if explicit_analysis_source is not None
        else resolved.analysis_source_role
    )

    # Existing output_dataset.json files are allowed to carry explicit paths, but
    # older HSE/PBE discovery runs sometimes recorded POSCAR as the analysis
    # source even though a strict CP2K final restart was present in the same
    # directory.  For post-production analysis, the strict final restart is the
    # safer source because it preserves the final-frame naming contract
    # (``*-1.restart`` is final; ``*1.restart`` is not).  Do not override an
    # explicitly requested relaxation trajectory, ASE database row, or a source
    # excluded by the caller's source-selection filter.
    strict_resolved_final = (
        Path(resolved.final_structure)
        if resolved.final_structure is not None and _is_strict_final_restart_name(Path(resolved.final_structure).name)
        else None
    )
    if strict_resolved_final is not None and _source_selection_allows_path(strict_resolved_final, source_selection):
        if final_structure_use is None or not _is_strict_final_restart_name(Path(final_structure_use).name):
            final_structure_use = strict_resolved_final
        if relax_data_use is None or not _is_strict_final_restart_name(Path(relax_data_use).name):
            relax_data_use = strict_resolved_final
        role_low = str(analysis_source_role_use or "").strip().lower()
        source_layout_low = str(source_layout or "").strip().lower()
        may_override_analysis_source = (
            source_layout_low != "ase_database"
            and role_low in {"", "single_structure", "final_structure"}
            and (analysis_source_use is None or not _is_strict_final_restart_name(Path(analysis_source_use).name))
        )
        if may_override_analysis_source:
            analysis_source_use = strict_resolved_final
            analysis_source_role_use = "final_structure"

    dens = _optional_float(density)
    dens_se = _optional_float(density_stderr)
    task_density, task_density_stderr = _authoritative_task_density(task_result)
    if task_density is not None:
        if dens is not None and not math.isclose(
            float(dens), float(task_density), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "Dataset density conflicts with authoritative task-result thermo density "
                f"for box {_resolved_box_id(bdir, fallback=box)}"
            )
        if dens_se is not None and not math.isclose(
            float(dens_se), float(task_density_stderr), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "Dataset density_stderr conflicts with authoritative task-result value "
                f"for box {_resolved_box_id(bdir, fallback=box)}"
            )
        dens = float(task_density)
        dens_se = float(task_density_stderr)
    if dens is None and dens_se is None:
        dens, dens_se = _collect_density_stats(relax_use)

    return _build_discovered_box(
        box=_resolved_box_id(bdir, fallback=box),
        box_dir=bdir,
        melt_dir=melt_use,
        quench_dir=quench_use,
        relax_dir=relax_use,
        input_structure=input_structure_use,
        final_structure=final_structure_use,
        relax_data=relax_data_use,
        relax_dump=relax_dump_use,
        relax_traj=relax_traj_use,
        analysis_source=analysis_source_use,
        analysis_source_role=analysis_source_role_use,
        density=dens,
        density_stderr=dens_se,
        task_result=task_result,
        source_layout=source_layout,
        source_record=source_record,
    )


def _read_atoms_snapshot(
    source_path: Path,
    *,
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
    lammps_units_style: Optional[str] = None,
):
    src = Path(source_path)
    if _is_dump_like(src):
        return None
    if _looks_like_lammps_data_file(src) and lammps_units_style in (None, ""):
        raise ValueError(
            "lammps_units_style is required to canonicalize raw LAMMPS data "
            f"source {src}"
        )
    try:
        if _looks_like_lammps_data_file(src):
            try:
                # The serialized Masses section is authoritative for density.
                # ASE may substitute its elemental reference mass, producing a
                # small but real live/replay convergence mismatch.
                from ..io.lammps_data_minimal import read_lammps_data_minimal

                return read_lammps_data_minimal(
                    src,
                    atom_style=str(atom_style),
                    specorder=(None if type_to_species is None else list(type_to_species)),
                    units_style=str(lammps_units_style),
                )
            except Exception:
                return ase_read_lammps_data(
                    src,
                    atom_style=str(atom_style),
                    specorder=(None if type_to_species is None else list(type_to_species)),
                    units=str(lammps_units_style),
                )
        from ase.io import read as ase_read

        try:
            return ase_read(str(src), index=-1)
        except Exception:
            return ase_read(str(src))
    except Exception:
        return None


def _estimate_density_from_source(
    source_path: Path,
    *,
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
    lammps_units_style: Optional[str] = None,
) -> Optional[float]:
    atoms = _read_atoms_snapshot(
        source_path,
        type_to_species=type_to_species,
        atom_style=atom_style,
        lammps_units_style=lammps_units_style,
    )
    if atoms is not None:
        try:
            vol = float(atoms.get_volume())
            masses = [float(x) for x in atoms.get_masses()]
            if vol > 1.0e-12 and masses:
                return float(sum(masses) * 1.66053906660 / vol)
        except Exception:
            pass

    if type_to_species is None:
        return None
    try:
        from ase.data import atomic_masses, atomic_numbers

        masses_by_type = {
            i + 1: float(atomic_masses[int(atomic_numbers[str(sym)])])
            for i, sym in enumerate(list(type_to_species))
        }
        frames = read_last_frames_auto(
            source_path,
            1,
            type_to_species=type_to_species,
            atom_style=str(atom_style),
            units_style=lammps_units_style,
        )
        if not frames:
            return None
        frame = frames[-1]
        vol = abs(float(np.linalg.det(frame.cell)))
        if vol <= 1.0e-12:
            return None
        total_mass = 0.0
        for t in frame.types.tolist():
            mass = masses_by_type.get(int(t), None)
            if mass is None:
                return None
            total_mass += float(mass)
        return float(total_mass * 1.66053906660 / vol)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_task_result_payload(path: Path) -> dict[str, Any]:
    """Read task metadata and enforce integrity for the current schema."""

    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text())
    except Exception as exc:
        raise ValueError(
            f"Could not parse task result metadata: {result_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Task result must be a mapping: {result_path}")
    result = dict(payload)
    try:
        # Version-1/version-2 records remain generic analysis inputs.  A current
        # version-3 result, however, is never trusted without its exact digest.
        validate_task_result_integrity(result, require_current=False)
    except RuntimeError as exc:
        raise ValueError(
            f"Task result integrity validation failed: {result_path}: {exc}"
        ) from exc
    return result


def _validated_current_task_engine_identity(
    paths: Sequence[Path],
) -> Optional[dict[str, Any]]:
    """Require authenticated homogeneous builds for current task results."""

    current: list[Mapping[str, Any]] = []
    for raw_path in paths:
        payload = _read_task_result_payload(Path(raw_path))
        if str(payload.get("schema", "")) == TASK_RESULT_SCHEMA:
            current.append(payload)
    try:
        return homogeneous_successful_task_engine_identity(current)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Current external task results do not have one authenticated "
            f"engine build identity: {exc}"
        ) from exc


def _self_hashed_json_fingerprint_valid(
    fingerprint: Any,
    *,
    workflow: Optional[str] = None,
) -> bool:
    if not isinstance(fingerprint, Mapping):
        return False
    payload = fingerprint.get("payload", {})
    if not isinstance(payload, Mapping):
        return False
    if workflow is not None and str(payload.get("workflow", "")) != str(workflow):
        return False
    try:
        stored_sha = str(fingerprint.get("sha256", "")).strip().lower()
        canonical = json.dumps(
            json_sanitize(dict(payload)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return bool(stored_sha) and stored_sha == hashlib.sha256(canonical).hexdigest()
    except Exception:
        return False


_CURRENT_RESULTS_FINGERPRINTS: dict[str, tuple[str, bool]] = {
    "vitriflow.run.resume_fingerprint.v5": ("run_meltquench", True),
    "vitriflow.autotune.resume_fingerprint.v3": ("autotune", True),
    "vitriflow.custom_schedule.resume_fingerprint.v3": (
        "custom_stage_schedule",
        False,
    ),
}


def _stable_results_payload(path: Path) -> dict[str, Any]:
    """Read one result JSON whose configured final path remains the read inode."""

    result_path = Path(path)
    if result_path.is_symlink():
        raise ValueError(
            f"Current VitriFlow result metadata must not be a symbolic link: {result_path}"
        )
    try:
        raw = result_path.read_bytes()
        identity = stable_file_identity(result_path, reject_final_symlink=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not read stable result metadata: {result_path}") from exc
    if hashlib.sha256(raw).hexdigest() != str(identity["sha256"]):
        raise ValueError(f"Result metadata changed while it was read: {result_path}")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Could not parse result metadata: {result_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Result metadata must be a JSON object: {result_path}")
    return dict(payload)


def _validate_deferred_engine_bundle(
    value: Any,
    *,
    expected_engine: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("External result has no deferred engine identity marker")
    envelope = dict(value)
    digest = str(envelope.pop("identity_sha256", "")).strip().lower()
    algorithm = str(envelope.pop("algorithm", ""))
    if (
        envelope.get("schema") != "vitriflow.engine_build_identities.v1"
        or envelope.get("status") != "deferred_to_external_worker"
        or str(envelope.get("primary_engine", "")).strip().lower()
        != str(expected_engine).strip().lower()
        or envelope.get("engines") != {}
        or algorithm != "sha256:c14n-json:v1"
        or digest != canonical_json_sha256(envelope)
    ):
        raise ValueError("External result has an invalid deferred engine identity marker")


def _authenticate_external_task_results(
    *,
    results_path: Path,
    production: Mapping[str, Any],
    protected_plan: Mapping[str, Any],
    protected_execution_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the complete successful external-worker task prefix."""

    base = Path(results_path).resolve(strict=False).parent
    execution = production.get("execution")
    candidates: list[Any] = [production.get("ensemble_dir")]
    if isinstance(execution, Mapping):
        candidates.extend(
            [execution.get("output_dataset"), execution.get("production_dir")]
        )
    prod_dir: Optional[Path] = None
    for raw in candidates:
        if raw in (None, ""):
            continue
        candidate = _path_from_record(raw, base_dir=base)
        if candidate is None:
            continue
        try:
            candidate.resolve(strict=True).relative_to(base.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "External production task directory escapes the result root"
            ) from exc
        prod_dir = candidate
        break
    if prod_dir is None:
        fallback = base / "production"
        if fallback.is_dir():
            prod_dir = fallback
    if prod_dir is None or prod_dir.is_symlink() or not prod_dir.is_dir():
        raise ValueError("External result has no trusted production task directory")

    entries = list(production.get("boxes", []) or []) + list(
        production.get("rejected_boxes", []) or []
    )
    try:
        box_ids = sorted(int(entry.get("box")) for entry in entries)
        n_total = int(production.get("n_boxes_total"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("External result has malformed attempted box ids") from exc
    if box_ids != list(range(1, n_total + 1)):
        raise ValueError(
            "External result does not identify the complete one-based task prefix"
        )

    task_results: list[dict[str, Any]] = []
    for box_id in box_ids:
        box_dir = prod_dir / f"box_{box_id:03d}"
        task_json = box_dir / "task.json"
        task_result = box_dir / "task_result.json"
        if (
            box_dir.is_symlink()
            or task_json.is_symlink()
            or task_result.is_symlink()
            or not task_json.is_file()
            or not task_result.is_file()
        ):
            raise ValueError(
                f"External task {box_id} is missing trusted task/result metadata"
            )
        task_data = _stable_results_payload(task_json)
        result = _stable_results_payload(task_result)
        task_meta = task_data.get("task")
        task_plan = task_data.get("production_plan")
        task_config = task_data.get("config")
        engine = str(protected_plan.get("engine", "")).strip().lower()
        recorded_box = (
            Path(str(task_meta.get("box_dir", "")))
            if isinstance(task_meta, Mapping)
            else Path()
        )
        recorded_task = (
            Path(str(task_meta.get("task_json", "")))
            if isinstance(task_meta, Mapping)
            else Path()
        )
        recorded_result = (
            Path(str(task_meta.get("task_result", "")))
            if isinstance(task_meta, Mapping)
            else Path()
        )
        # External trees are intentionally portable: task manifests created on
        # a worker record absolute materialisation-time paths, so equality to
        # the current path would reject an otherwise byte-identical copied
        # result tree.  Authenticate the adjacent files we actually opened and
        # require the recorded paths to describe the same internally-contained
        # box layout.  The complete task manifest is itself authenticated by
        # task_manifest_sha256 below.
        recorded_layout_ok = bool(
            str(recorded_box)
            and recorded_box.name == box_dir.name
            and recorded_box.parent.name == prod_dir.name
            and recorded_task == recorded_box / "task.json"
            and recorded_result == recorded_box / "task_result.json"
        )
        if (
            not isinstance(task_meta, Mapping)
            or int(task_meta.get("box", -1)) != box_id
            or not recorded_layout_ok
            or int(result.get("box", -1)) != box_id
            or not isinstance(task_plan, Mapping)
            or canonical_json_sha256(dict(task_plan))
            != canonical_json_sha256(dict(protected_plan))
            or not isinstance(task_config, Mapping)
            or not isinstance(task_config.get(engine), Mapping)
            or canonical_json_sha256(dict(task_config.get(engine, {})))
            != canonical_json_sha256(dict(protected_execution_config))
        ):
            raise ValueError(
                f"External task {box_id} is inconsistent with the protected run plan, "
                "box identity, or engine execution configuration"
            )
        try:
            validate_task_result_replay_artifacts(
                result=result,
                task_data=task_data,
                box_dir=box_dir,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"External task {box_id} replay authentication failed: {exc}"
            ) from exc
        task_results.append(result)

    try:
        homogeneous = homogeneous_successful_task_engine_identity(task_results)
        protected = validate_engine_build_identity(
            production.get("engine_build_identity")  # type: ignore[arg-type]
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"External worker engine identity authentication failed: {exc}"
        ) from exc
    if (
        not isinstance(homogeneous, Mapping)
        or str(homogeneous.get("identity_sha256", ""))
        != str(protected.get("identity_sha256", ""))
        or str(production.get("engine_build_identity_status", ""))
        != "verified_homogeneous_workers"
    ):
        raise ValueError(
            "External worker identities disagree with the protected production state"
        )
    return dict(protected)


def _authenticate_current_results_replay(
    path: Path,
    payload: Mapping[str, Any],
) -> bool:
    """Authenticate every field used for exact result replay and parity.

    Returns ``False`` for a legacy result so it can still be analysed as an
    explicitly generic dataset.  Once a result advertises a current fingerprint
    schema, malformed fingerprints, plan cross-links, engine identities,
    production state, artifact bytes, or status cross-links are terminal.
    """

    fingerprint = payload.get("resume_fingerprint")
    if not isinstance(fingerprint, Mapping):
        return False
    schema = str(fingerprint.get("schema", ""))
    contract = _CURRENT_RESULTS_FINGERPRINTS.get(schema)
    if contract is None:
        return False
    expected_workflow, requires_plan = contract
    fp_payload = fingerprint.get("payload")
    if (
        str(fingerprint.get("algorithm", "")) != "sha256:c14n-json:v1"
        or not isinstance(fp_payload, Mapping)
        or str(fp_payload.get("schema", "")) != schema
        or str(fp_payload.get("workflow", "")) != expected_workflow
        or str(fingerprint.get("sha256", "")).strip().lower()
        != canonical_json_sha256(dict(fp_payload))
    ):
        raise ValueError(
            f"Current VitriFlow result has an invalid resume fingerprint: {path}"
        )

    if requires_plan:
        result_plan = payload.get("production_plan")
        protected_plan = fp_payload.get("production_plan")
        if (
            not isinstance(result_plan, Mapping)
            or not result_plan
            or not isinstance(protected_plan, Mapping)
            or canonical_json_sha256(dict(result_plan))
            != canonical_json_sha256(dict(protected_plan))
        ):
            raise ValueError(
                f"Current VitriFlow result has a modified or missing protected production plan: {path}"
            )

    production = payload.get("production")
    if not isinstance(production, Mapping):
        raise ValueError(f"Current VitriFlow result has no production state: {path}")
    try:
        validate_production_resume_state(
            production,
            outdir=Path(path).resolve(strict=False).parent,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Current VitriFlow result production authentication failed: {path}: {exc}"
        ) from exc

    external_mode = str(fp_payload.get("external_mode", "local")).strip().lower()
    external_run = expected_workflow == "run_meltquench" and external_mode in {
        "dry-run",
        "full-run",
    }
    if expected_workflow == "custom_stage_schedule":
        runner = fp_payload.get("runner")
        engine_bundle = (
            runner.get("engine_build_identities")
            if isinstance(runner, Mapping)
            else None
        )
        top_bundle = payload.get("engine_build_identities")
        if (
            not isinstance(engine_bundle, Mapping)
            or not isinstance(top_bundle, Mapping)
            or canonical_json_sha256(dict(engine_bundle))
            != canonical_json_sha256(dict(top_bundle))
        ):
            raise ValueError(
                f"Current custom-schedule result has inconsistent engine identity evidence: {path}"
            )
    elif external_run:
        protected_plan = fp_payload.get("production_plan")
        plan_engine = (
            str(protected_plan.get("engine", ""))
            if isinstance(protected_plan, Mapping)
            else ""
        )
        engine_bundle = fp_payload.get("engine_build_identities")
        _validate_deferred_engine_bundle(
            engine_bundle,
            expected_engine=plan_engine,
        )
        if external_mode == "full-run":
            _authenticate_external_task_results(
                results_path=Path(path),
                production=production,
                protected_plan=(
                    protected_plan if isinstance(protected_plan, Mapping) else {}
                ),
                protected_execution_config=(
                    fp_payload.get("execution_config")
                    if isinstance(fp_payload.get("execution_config"), Mapping)
                    else {}
                ),
            )
        elif (
            str(production.get("status", "")).strip().lower() != "planned"
            or int(production.get("n_boxes_total", -1)) != 0
            or production.get("engine_build_identity") is not None
            or str(production.get("engine_build_identity_status", ""))
            != "deferred_to_external_worker"
        ):
            raise ValueError(
                f"Current external dry-run result has inconsistent execution state: {path}"
            )
    else:
        engine_bundle = fp_payload.get("engine_build_identities")
    if not external_run:
        try:
            validate_engine_build_identity_bundle(engine_bundle)  # type: ignore[arg-type]
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Current VitriFlow result has invalid engine identity evidence: {path}: {exc}"
            ) from exc

    result_status = str(payload.get("status", "")).strip().lower()
    production_status = str(production.get("status", "")).strip().lower()
    result_execution = str(payload.get("execution_status", "")).strip().lower()
    production_execution = str(production.get("execution_status", "")).strip().lower()
    if (
        not result_status
        or result_status != production_status
        or not result_execution
        or result_execution != production_execution
    ):
        raise ValueError(
            f"Current VitriFlow result status disagrees with its protected production state: {path}"
        )
    if expected_workflow == "custom_stage_schedule" and str(
        payload.get("workflow", "")
    ) != expected_workflow:
        raise ValueError(
            f"Current custom-schedule result has an inconsistent workflow label: {path}"
        )
    return True


def _load_task_result_entry(path: Path) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    data = _read_task_result_payload(path)
    status = str(data.get("status", "ok")).strip().lower()
    if status in {"ok", "success"}:
        entry: Optional[dict[str, Any]] = None
        if isinstance(data.get("box_entry", None), Mapping):
            entry = dict(data.get("box_entry", {}))
        elif isinstance(data.get("entry", None), Mapping):
            entry = dict(data.get("entry", {}))
        elif {"box", "metrics", "distributions"}.issubset(set(data.keys())):
            entry = dict(data)
        if entry is not None:
            reused = _validated_task_diagnostics(path)
            if reused is not None:
                entry["stage_metrics"] = reused.get("stage_metrics")
                entry["elastic_melt"] = reused.get("elastic_melt")
                entry["elastic_relax"] = reused.get("elastic_relax")
                entry["elastic_timeseries"] = reused.get("elastic_timeseries")
                entry["task_diagnostics_provenance"] = dict(
                    reused.get("provenance", {}) or {}
                )
                entry["_task_result_path_for_diagnostic_rebase"] = str(path)
            return entry, None
        # task box analysis
        return None, None
    box_label = data.get("box", None)
    if box_label is None and isinstance(data.get("task", None), Mapping):
        box_label = data.get("task", {}).get("box", None)
    if box_label is None:
        raise ValueError(
            f"Failed task result does not identify its box: {path}"
        )
    reject = {
        "box": _strict_analysis_box_id(
            box_label,
            context=f"Failed task result box id in {path}",
            minimum=0,
        ),
        "reason": "task_failed",
        "error": str(data.get("error", f"task_result status={status!r}")),
        "paths": {
            "task_result": str(path),
        },
    }
    return None, reject


def _require_unique_dataset_box_ids(
    raw_boxes: Sequence[DiscoveredBox],
    entries: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    *,
    minimum_box_id: int = 1,
) -> None:
    """Reject ambiguous box identities before any keyed aggregation occurs.

    Several downstream products are intentionally keyed by integer box id.
    Accepting duplicate ids would silently overwrite worker results, streaming
    chunks, and provenance records even though both inputs were discovered.
    """

    if minimum_box_id not in {0, 1}:
        raise ValueError("minimum_box_id must be 0 or 1")

    seen: dict[int, str] = {}

    def _record(box_value: Any, location: str) -> None:
        box_id = _strict_analysis_box_id(
            box_value,
            context=f"Output dataset box id at {location}",
            minimum=minimum_box_id,
        )
        prior = seen.get(box_id)
        if prior is not None:
            raise ValueError(
                f"Output dataset contains duplicate box id {box_id}: {prior} and {location}. "
                "Box ids must be unique before analysis."
            )
        seen[box_id] = location

    for idx, box in enumerate(raw_boxes):
        _record(box.box, f"raw_boxes[{idx}]")
    for collection_name, collection in (("entries", entries), ("rejected", rejected)):
        for idx, item in enumerate(collection):
            if "box" not in item:
                raise ValueError(f"Output dataset {collection_name}[{idx}] has no box id")
            _record(item.get("box"), f"{collection_name}[{idx}]")


def _strict_analysis_box_id(
    value: Any,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    """Return one finite integral box id without treating zero as missing."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a non-boolean integer; got {value!r}")
    if isinstance(value, Integral):
        parsed = int(value)
    else:
        try:
            exact = Decimal(str(value).strip())
        except (InvalidOperation, ValueError, AttributeError) as exc:
            raise ValueError(f"{context} is invalid: {value!r}") from exc
        if not exact.is_finite() or exact != exact.to_integral_value():
            requirement = (
                "non-negative" if int(minimum) == 0 else f">= {int(minimum)}"
            )
            raise ValueError(
                f"{context} must be a finite integral value {requirement}; got {value!r}"
            )
        parsed = int(exact)
    if parsed < int(minimum):
        requirement = (
            "non-negative" if int(minimum) == 0 else f">= {int(minimum)}"
        )
        raise ValueError(
            f"{context} must be a finite integral value {requirement}; got {value!r}"
        )
    return int(parsed)


def _discover_from_results_file(path: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _stable_results_payload(path)
    authenticated_current = _authenticate_current_results_replay(path, data)

    prod = data.get("production", {}) if isinstance(data.get("production", {}), Mapping) else {}
    base_dir = Path(path).resolve().parent
    dataset_hint = None

    exec_meta = prod.get("execution", {}) if isinstance(prod.get("execution", {}), Mapping) else {}
    if isinstance(exec_meta.get("output_dataset", None), str):
        dataset_hint = _path_from_record(exec_meta.get("output_dataset"), base_dir=base_dir)

    if dataset_hint is None and isinstance(prod.get("ensemble_dir", None), str):
        dataset_hint = _path_from_record(prod.get("ensemble_dir"), base_dir=base_dir)

    if dataset_hint is None:
        default_prod_dir = base_dir / "production"
        if default_prod_dir.exists():
            dataset_hint = default_prod_dir

    if dataset_hint is None:
        raise ValueError(f"Could not locate production output directory from results file: {path}")

    raw_boxes, entries, rejected, dataset_meta = discover_output_dataset(dataset_hint, source_selection=source_selection)

    # A current VitriFlow result is more than a hint about where its production
    # directory lives: every accepted box records the exact source used by
    # ``analyse_production_box``.  Re-resolving that directory generically can
    # select ``final.extxyz`` ahead of ``traj.extxyz`` and thereby replace the
    # configured time-average with a single-frame calculation.  That is a
    # scientifically different estimator even when a very short smoke run
    # happens to yield the same scalar values.
    #
    # Therefore, when the protected production plan and convergence report are
    # present, rebuild the accepted ensemble from the recorded source paths.
    # Missing/ambiguous evidence is terminal: silently falling back to another
    # readable file would make the advertised live/replay parity unverifiable.
    plan = data.get("production_plan", {})
    convergence = prod.get("convergence", None)
    if not isinstance(convergence, Mapping):
        convergence = prod.get("convergence_dft", None)
    if not isinstance(convergence, Mapping):
        convergence = prod.get("convergence_md", None)
    accepted_rows = prod.get("boxes", None)
    resume_fingerprint = data.get("resume_fingerprint", {})
    protected_custom_schedule = _self_hashed_json_fingerprint_valid(
        resume_fingerprint,
        workflow="custom_stage_schedule",
    )
    exact_replay_contract = bool(
        authenticated_current
        and
        ((isinstance(plan, Mapping) and bool(plan)) or protected_custom_schedule)
        and isinstance(convergence, Mapping)
        and bool(convergence)
        and isinstance(accepted_rows, list)
        and bool(accepted_rows)
        and str(prod.get("status", "")).strip().lower() != "planned"
    )
    if exact_replay_contract:
        # Autotune/run production IDs are one-based, whereas the public
        # custom-schedule workflow deliberately starts at ``box_000``.  The
        # latter is proven by the workflow-bound, self-hashed fingerprint
        # above, so accepting zero here cannot weaken an autotune/run result's
        # positive-ID contract.
        minimum_box_id = 0 if protected_custom_schedule else 1
        box_id_requirement = (
            "non-negative integer" if protected_custom_schedule else "positive integer"
        )
        discovered_by_id = {int(box.box): box for box in raw_boxes}
        rebound: list[DiscoveredBox] = []
        seen_ids: set[int] = set()
        for index, row_raw in enumerate(list(accepted_rows or [])):
            if not isinstance(row_raw, Mapping):
                raise ValueError(
                    "Cannot exactly replay production: production.boxes contains "
                    f"a non-object entry at index {index}"
                )
            try:
                numeric_id = float(row_raw.get("box"))
                box_id = int(numeric_id)
            except Exception as exc:
                raise ValueError(
                    "Cannot exactly replay production: every accepted box requires "
                    f"a {box_id_requirement} box id"
                ) from exc
            if (
                not math.isfinite(numeric_id)
                or numeric_id != float(box_id)
                or box_id < minimum_box_id
                or box_id in seen_ids
            ):
                raise ValueError(
                    "Cannot exactly replay production: accepted box ids must be "
                    f"unique {box_id_requirement}s"
                )
            seen_ids.add(box_id)

            paths = row_raw.get("paths", {})
            if not isinstance(paths, Mapping):
                raise ValueError(
                    f"Cannot exactly replay production box {box_id}: paths record is missing"
                )
            analysis_source = _path_from_record(
                paths.get("analysis_source", None), base_dir=base_dir
            )
            role = str(row_raw.get("analysis_source_role", "") or "").strip()
            if analysis_source is None or not analysis_source.is_file() or not role:
                raise ValueError(
                    f"Cannot exactly replay production box {box_id}: the recorded "
                    "analysis source/role is missing or unreadable"
                )
            if not _source_selection_allows_path(analysis_source, source_selection):
                raise ValueError(
                    f"Cannot exactly replay production box {box_id}: source-selection "
                    "filters exclude the recorded analysis source"
                )

            original = discovered_by_id.get(box_id)
            relax_dir = _path_from_record(paths.get("relax_dir", None), base_dir=base_dir)
            if relax_dir is None:
                relax_dir = analysis_source.parent
            box_dir = relax_dir.parent if relax_dir.name.lower() == "relax" else relax_dir
            melt_dir = _path_from_record(paths.get("melt_dir", None), base_dir=base_dir)
            quench_dir = _path_from_record(paths.get("quench_dir", None), base_dir=base_dir)
            relax_data = _path_from_record(paths.get("relax_data", None), base_dir=base_dir)
            relax_dump = _path_from_record(paths.get("relax_dump", None), base_dir=base_dir)
            relax_traj = _path_from_record(paths.get("relax_traj", None), base_dir=base_dir)

            rebound.append(
                _box_from_dirs(
                    box_dir,
                    task_result=(None if original is None else original.task_result),
                    box=box_id,
                    melt_dir=melt_dir,
                    quench_dir=quench_dir,
                    relax_dir=relax_dir,
                    relax_data=relax_data,
                    relax_dump=relax_dump,
                    relax_traj=relax_traj,
                    density=row_raw.get("density", None),
                    density_stderr=row_raw.get("density_stderr", None),
                    analysis_source=analysis_source,
                    analysis_source_role=role,
                    input_structure=(None if original is None else original.input_structure),
                    final_structure=(None if original is None else original.final_structure),
                    source_layout="production_results_recorded_source",
                    source_record={
                        "schema": "vitriflow.production_replay_source.v1",
                        "source_results": str(Path(path).resolve()),
                        "box": int(box_id),
                        "analysis_source": str(analysis_source),
                        "analysis_source_role": role,
                    },
                    source_selection=source_selection,
                )
            )

        raw_boxes = sorted(rebound, key=lambda box: int(box.box))
        entries = []
        rejected = []
        dataset_meta = {
            **dict(dataset_meta),
            "layout": "production_results_recorded_sources",
            "exact_replay_source_contract": True,
            "n_original_rejected": int(len(list(prod.get("rejected_boxes", []) or []))),
        }
    meta = dict(dataset_meta)
    meta["source_results_json"] = str(path)
    meta["current_results_authentication"] = (
        "verified" if authenticated_current else "legacy_generic_only"
    )
    return raw_boxes, entries, rejected, meta


def _discover_from_dataset_file(path: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(path)
    base_dir = Path(path).resolve().parent
    boxes_raw = data.get("boxes", []) if isinstance(data.get("boxes", []), list) else []
    raw_boxes: list[DiscoveredBox] = []
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    task_result_paths = [
        candidate
        for record in boxes_raw
        if isinstance(record, Mapping)
        for candidate in [
            _path_from_record(record.get("task_result", None), base_dir=base_dir)
        ]
        if candidate is not None and candidate.is_file()
    ]
    engine_build_identity = _validated_current_task_engine_identity(
        task_result_paths
    )
    for idx, rec in enumerate(boxes_raw, start=1):
        if not isinstance(rec, Mapping):
            continue
        task_result = _path_from_record(rec.get("task_result", None), base_dir=base_dir)
        if task_result is not None and task_result.exists():
            entry, reject = _load_task_result_entry(task_result)
            if entry is not None:
                entries.append(entry)
                continue
            if reject is not None:
                rejected.append(reject)
                continue

        box_label = _optional_int(rec.get("box", None))
        box_fallback = (int(box_label) if box_label is not None and int(box_label) > 0 else int(idx))
        box_dir = _path_from_record(rec.get("box_dir", None), base_dir=base_dir)
        melt_dir = _path_from_record(rec.get("melt_dir", None), base_dir=base_dir)
        quench_dir = _path_from_record(rec.get("quench_dir", None), base_dir=base_dir)
        relax_dir = _path_from_record(rec.get("relax_dir", None), base_dir=base_dir)
        input_structure = _path_from_record(rec.get("input_structure", None), base_dir=base_dir)
        final_structure = _path_from_record(rec.get("final_structure", None), base_dir=base_dir)
        relax_data = _path_from_record(rec.get("relax_data", None), base_dir=base_dir)
        relax_dump = _path_from_record(rec.get("relax_dump", None), base_dir=base_dir)
        relax_traj = _path_from_record(rec.get("relax_traj", None), base_dir=base_dir)
        analysis_source = _path_from_record(rec.get("analysis_source", None), base_dir=base_dir)
        analysis_source_role = rec.get("analysis_source_role", None)
        source_layout = rec.get("source_layout", None)
        source_record = rec.get("source_record", None) if isinstance(rec.get("source_record", None), Mapping) else None

        if box_dir is None:
            source = analysis_source or relax_traj or relax_data
            if source is None:
                continue
            box_obj = _box_from_source_file(
                source,
                task_result=task_result,
                box=box_fallback,
                explicit_box=True,
                analysis_source_role=str(analysis_source_role or "single_structure"),
                source_layout=(None if source_layout in (None, "") else str(source_layout)),
                source_record=source_record,
            )
            if _discovered_box_matches_source_selection(box_obj, source_selection):
                raw_boxes.append(box_obj)
            continue

        box_obj = _box_from_dirs(
            box_dir,
            task_result=task_result,
            box=box_fallback,
            melt_dir=melt_dir,
            quench_dir=quench_dir,
            relax_dir=relax_dir,
            input_structure=input_structure,
            final_structure=final_structure,
            relax_data=relax_data,
            relax_dump=relax_dump,
            relax_traj=relax_traj,
            density=rec.get("density", None),
            density_stderr=rec.get("density_stderr", None),
            analysis_source=analysis_source,
            analysis_source_role=analysis_source_role,
            source_layout=(None if source_layout in (None, "") else str(source_layout)),
            source_record=source_record,
            source_selection=source_selection,
        )
        if _discovered_box_matches_source_selection(box_obj, source_selection):
            raw_boxes.append(box_obj)
    dataset = dict(data)
    if engine_build_identity is not None:
        dataset["engine_build_identity"] = engine_build_identity
        dataset["engine_build_identity_status"] = "verified_homogeneous_workers"
    return raw_boxes, entries, rejected, dataset


def _discover_from_directory(path: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(path).resolve()
    dataset_file = root / "output_dataset.json"
    if dataset_file.exists():
        return _discover_from_dataset_file(dataset_file, source_selection=source_selection)

    task_results = sorted(root.rglob("task_result.json"))
    engine_build_identity = _validated_current_task_engine_identity(task_results)
    raw_boxes: list[DiscoveredBox] = []
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_box_dirs: set[Path] = set()

    for task_result in task_results:
        box_dir = task_result.parent
        if box_dir.name == "task":
            box_dir = box_dir.parent
        if box_dir.name == "results":
            box_dir = box_dir.parent
        try:
            box_dir = box_dir.resolve(strict=False)
        except Exception:
            pass
        entry, reject = _load_task_result_entry(task_result)
        if entry is not None:
            entries.append(entry)
            seen_box_dirs.add(box_dir)
            continue
        if reject is not None:
            rejected.append(reject)
            seen_box_dirs.add(box_dir)
            continue
        raw_boxes.append(_box_from_dirs(box_dir, task_result=task_result, source_selection=source_selection))
        seen_box_dirs.add(box_dir)

    box_dirs = sorted([p for p in root.glob("box_*") if p.is_dir()], key=_slug_box_id)
    for box_dir in box_dirs:
        if box_dir.resolve(strict=False) in seen_box_dirs:
            continue
        raw_boxes.append(_box_from_dirs(box_dir, source_selection=source_selection))
        seen_box_dirs.add(box_dir.resolve(strict=False))

    # box tree wrapper
    if not box_dirs and (root / "relax").exists():
        raw_boxes.append(_box_from_dirs(root, source_selection=source_selection))

    if not raw_boxes and not box_dirs and not (root / "relax").exists():
        skip_names = {"melt", "quench", "relax", "results", "task", "preview", "analysis", "rejects"}
        loose_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name not in skip_names], key=lambda p: p.name)
        for idx, box_dir in enumerate(loose_dirs, start=1):
            if box_dir.resolve(strict=False) in seen_box_dirs:
                continue
            relax_dir = box_dir / "relax" if (box_dir / "relax").exists() else box_dir
            if _guess_analysis_source(relax_dir, source_selection=source_selection) is None:
                continue
            raw_boxes.append(_box_from_dirs(box_dir, box=_resolved_box_id(box_dir, fallback=idx), source_selection=source_selection))
            seen_box_dirs.add(box_dir.resolve(strict=False))

    flat_file_sources = 0
    ase_database_files = 0
    ase_database_rows = 0
    if not raw_boxes and not box_dirs and not (root / "relax").exists():
        direct_sources_all = [p for p in _iter_analysis_source_candidates(root, source_selection=source_selection) if p.parent == root]
        db_sources = [p for p in direct_sources_all if _looks_like_ase_database_file(p)]
        direct_sources = [p for p in direct_sources_all if not _looks_like_ase_database_file(p)]
        used_ids: set[int] = set()
        for db_source in sorted(db_sources, key=_natural_sort_key):
            db_boxes, _db_entries, db_rejected, db_meta = _discover_from_ase_database_file(db_source)
            ase_database_files += 1
            ase_database_rows += int(db_meta.get("n_database_rows", len(db_boxes)))
            rejected.extend(db_rejected)
            for db_box in db_boxes:
                preferred = int(db_box.box)
                box_id = _next_unused_positive_id(preferred, used_ids)
                used_ids.add(int(box_id))
                if int(box_id) == int(db_box.box):
                    raw_boxes.append(db_box)
                else:
                    raw_boxes.append(
                        _box_from_ase_database_record(
                            Path(db_source),
                            source_record=(db_box.source_record or {}),
                            box=int(box_id),
                        )
                    )
        assignments = _flat_source_box_assignments(direct_sources, used=used_ids)
        flat_file_sources = int(len(assignments))
        for source, box_id in assignments:
            raw_boxes.append(
                _box_from_source_file(
                    source,
                    box=int(box_id),
                    explicit_box=True,
                    analysis_source_role="final_structure",
                    source_layout="flat_file_ensemble",
                )
            )

    if ase_database_files and flat_file_sources:
        layout = "mixed_flat_ensemble"
    elif ase_database_files:
        layout = "ase_database"
    elif flat_file_sources:
        layout = "flat_file_ensemble"
    else:
        layout = "directory"

    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(root),
        "layout": layout,
        "n_flat_file_sources": int(flat_file_sources),
        "n_ase_database_files": int(ase_database_files),
        "n_ase_database_rows": int(ase_database_rows),
    }
    if engine_build_identity is not None:
        dataset["engine_build_identity"] = engine_build_identity
        dataset["engine_build_identity_status"] = "verified_homogeneous_workers"
    return raw_boxes, entries, rejected, dataset


def discover_output_dataset(path: Path, *, source_selection: Optional[Mapping[str, Any]] = None) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    p = Path(path).expanduser()
    if p.is_dir():
        return _discover_from_directory(p, source_selection=source_selection)
    if _looks_like_ase_database_file(p):
        if not _source_selection_allows_path(p, source_selection):
            return [], [], [], {
                "schema": "vitriflow.output_dataset.v1",
                "source_root": str(p.parent.resolve()),
                "layout": "single_file",
                "n_flat_file_sources": 0,
                "source_selection": _source_selection_for_metadata(source_selection),
            }
        return _discover_from_ase_database_file(p)
    if _is_analysis_source_candidate(p):
        dataset = {
            "schema": "vitriflow.output_dataset.v1",
            "source_root": str(p.parent.resolve()),
            "layout": "single_file",
            "n_flat_file_sources": 1,
        }
        if not _source_selection_allows_path(p, source_selection):
            dataset["n_flat_file_sources"] = 0
            dataset["source_selection"] = _source_selection_for_metadata(source_selection)
            return [], [], [], dataset
        return [_box_from_source_file(p, box=1)], [], [], dataset
    data = _load_json(p)
    schema = str(data.get("schema", "")).strip().lower() if isinstance(data, Mapping) else ""
    if schema == "vitriflow.output_dataset.v1" or p.name == "output_dataset.json":
        return _discover_from_dataset_file(p, source_selection=source_selection)
    if p.name == "task_result.json":
        engine_build_identity = _validated_current_task_engine_identity([p])
        entry, reject = _load_task_result_entry(p)
        box_dir = p.parent
        raw: list[DiscoveredBox] = []
        entries = [entry] if entry is not None else []
        rejected = [reject] if reject is not None else []
        if entry is None and reject is None:
            raw.append(_box_from_dirs(box_dir, task_result=p, source_selection=source_selection))
        dataset = {"schema": "vitriflow.output_dataset.v1", "source_root": str(p.parent.resolve())}
        if engine_build_identity is not None:
            dataset["engine_build_identity"] = engine_build_identity
            dataset["engine_build_identity_status"] = "verified_homogeneous_workers"
        return raw, entries, rejected, dataset
    if p.name in {"run_results.json", "autotune_results.json"}:
        return _discover_from_results_file(p, source_selection=source_selection)
    if isinstance(data, Mapping) and isinstance(data.get("production", None), Mapping):
        return _discover_from_results_file(p, source_selection=source_selection)
    raise ValueError(f"Unsupported output-analysis input: {p}")


def _embedded_production_replay_contract(path: Path) -> dict[str, Any]:
    """Load exact production settings/evidence from a results JSON, if present."""

    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".json":
        return {}
    try:
        payload = _stable_results_payload(p)
    except ValueError:
        # A JSON file which is not a VitriFlow result remains eligible for
        # generic discovery. A file advertising a current fingerprint is
        # re-read and rejected by discovery rather than promoted to parity.
        return {}
    if not _authenticate_current_results_replay(p, payload):
        return {}
    production = payload.get("production", {})
    if not isinstance(production, Mapping):
        return {}
    plan = payload.get("production_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    resume_fingerprint = payload.get("resume_fingerprint", {})
    if not isinstance(resume_fingerprint, Mapping):
        resume_fingerprint = {}
    report = production.get("convergence", None)
    if not isinstance(report, Mapping):
        report = production.get("convergence_dft", None)
    if not isinstance(report, Mapping):
        report = production.get("convergence_md", None)
    spec = production.get("convergence_spec", None)
    accepted_dft_ids = production.get("boxes_dft_final", None)
    if not isinstance(accepted_dft_ids, list):
        accepted_dft_ids = []
        for entry in list(production.get("boxes", []) or []):
            if not isinstance(entry, Mapping):
                continue
            dft = entry.get("dft_opt", {})
            if isinstance(dft, Mapping) and str(dft.get("status", "")) == "ok":
                accepted_dft_ids.append(int(entry.get("box", 0) or 0))
    parsed_dft_ids: list[int] = []
    for value in accepted_dft_ids:
        if isinstance(value, bool):
            raise ValueError(
                "production.boxes_dft_final must contain positive integer box ids"
            )
        try:
            numeric = float(value)
            box_id = int(numeric)
        except Exception as exc:
            raise ValueError(
                "production.boxes_dft_final must contain positive integer box ids"
            ) from exc
        if not math.isfinite(numeric) or numeric != float(box_id) or box_id <= 0:
            raise ValueError(
                "production.boxes_dft_final must contain positive integer box ids"
            )
        parsed_dft_ids.append(int(box_id))
    if len(parsed_dft_ids) != len(set(parsed_dft_ids)):
        raise ValueError("production.boxes_dft_final contains duplicate box ids")
    return {
        "source_results": str(p),
        "plan": dict(plan),
        "workflow": str(payload.get("workflow", "") or ""),
        "resume_fingerprint": dict(resume_fingerprint),
        "convergence": (dict(report) if isinstance(report, Mapping) else None),
        "convergence_spec": (dict(spec) if isinstance(spec, Mapping) else None),
        "accepted_dft_box_ids": sorted(parsed_dft_ids),
    }


def _custom_schedule_replay_config_matches(
    config: Optional[RunConfig],
    replay_contract: Mapping[str, Any],
) -> bool:
    """Prove that a custom-schedule replay uses its producer analysis config.

    Custom schedules predate ``production_plan`` but current results contain a
    self-hashed resume fingerprint with every analysis-defining input.  Exact
    convergence comparison is valid only when that hash is internally sound
    and the supplied YAML reproduces the metric, convergence, production,
    dynamics, potential and engine sections used by the producer.
    """

    if config is None or str(replay_contract.get("workflow", "")) != "custom_stage_schedule":
        return False
    fingerprint = replay_contract.get("resume_fingerprint", {})
    if not isinstance(fingerprint, Mapping):
        return False
    payload = fingerprint.get("payload", {})
    if not isinstance(payload, Mapping) or not _self_hashed_json_fingerprint_valid(
        fingerprint,
        workflow="custom_stage_schedule",
    ):
        return False
    try:
        runner = payload.get("runner", {})
        potential = payload.get("potential", {})
        if not isinstance(runner, Mapping) or not isinstance(potential, Mapping):
            return False
        expected = json_sanitize(
            {
                "engine": runner.get("engine"),
                "md": payload.get("md"),
                "potential": potential.get("config"),
                "metrics": payload.get("metrics"),
                "convergence": payload.get("convergence"),
                "production_acceptance": payload.get("production_acceptance"),
            }
        )
        current = json_sanitize(
            {
                "engine": str(config.engine),
                "md": _model_dump_jsonlike(config.md),
                "potential": _model_dump_jsonlike(config.kim),
                "metrics": _model_dump_jsonlike(config.autotune.metrics),
                "convergence": _model_dump_jsonlike(config.autotune.convergence),
                "production_acceptance": _model_dump_jsonlike(config.autotune.production),
            }
        )
        return expected == current
    except Exception:
        return False


def _select_dft_refined_sources(
    raw_boxes: Sequence[DiscoveredBox],
    *,
    required_box_ids: Optional[Sequence[int]],
) -> tuple[list[DiscoveredBox], list[dict[str, Any]]]:
    """Select only positively converged CP2K dft_opt final structures."""

    required = (
        None
        if required_box_ids is None
        else {int(value) for value in required_box_ids}
    )
    selected: list[DiscoveredBox] = []
    rejected: list[dict[str, Any]] = []
    seen: set[int] = set()
    from ..cp2k_driver import assert_cp2k_cell_opt_converged

    for box in raw_boxes:
        box_id = int(box.box)
        seen.add(box_id)
        dft_dir = Path(box.box_dir) / "dft_opt"
        dft_data = dft_dir / "dft_opt.data"
        cp2k_output = dft_dir / "cp2k.out"
        if required is not None and box_id not in required:
            rejected.append(
                {
                    "box": box_id,
                    "reason": "not_in_source_refined_accepted_ensemble",
                    "analysis_source_role": "dft_opt_final",
                }
            )
            continue
        try:
            if not dft_data.is_file():
                raise FileNotFoundError(str(dft_data))
            if not cp2k_output.is_file():
                raise FileNotFoundError(str(cp2k_output))
            assert_cp2k_cell_opt_converged(cp2k_output)
        except Exception as exc:
            if required is not None and box_id in required:
                raise RuntimeError(
                    "A source-declared accepted refined box lacks valid CP2K "
                    f"CELL_OPT evidence: box {box_id}: {exc}"
                ) from exc
            rejected.append(
                {
                    "box": box_id,
                    "reason": "dft_opt_unavailable_or_not_converged",
                    "error": str(exc),
                    "analysis_source_role": "dft_opt_final",
                }
            )
            continue
        selected.append(
            replace(
                box,
                relax_dir=dft_dir,
                final_structure=dft_data,
                relax_data=dft_data,
                relax_dump=None,
                relax_traj=dft_data,
                analysis_source=dft_data,
                analysis_source_role="dft_opt_final",
                # The refined structure defines density exactly; do not carry
                # the parent MD relaxation-window mean into DFT convergence.
                density=None,
                density_stderr=0.0,
            )
        )
    if required is not None:
        missing = sorted(required - seen)
        if missing:
            raise RuntimeError(
                "Source refined ensemble refers to undiscovered box ids: "
                + ", ".join(str(value) for value in missing)
            )
    return selected, rejected


def _analysis_context_from_config(
    config: RunConfig,
    *,
    metrics_cfg: Optional[StructureMetricsConfig] = None,
    prod_cfg: Optional[ProductionEnsembleConfig] = None,
    conv_cfg: Optional[ConvergenceConfig] = None,
    cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
) -> AnalysisContext:
    metric_warnings: list[str] = []

    def _warn(msg: str) -> None:
        metric_warnings.append(str(msg))
        warnings.warn(str(msg), stacklevel=2)

    engine_name = str(getattr(config, "engine", "lammps") or "lammps").strip().lower()
    lammps_units_style: Optional[str] = None
    if engine_name == "lammps":
        from ..lammps_units import normalize_lammps_units_style

        lammps_units_style = normalize_lammps_units_style(
            str(getattr(config.kim, "user_units", "") or "")
        )
    type_to_species = _get_type_to_species(config)
    metrics_in = metrics_cfg if metrics_cfg is not None else config.autotune.metrics
    metrics_eff, _warnings, summary = resolve_effective_metrics_config(
        metrics_in,
        structure_data=None,
        type_to_species=type_to_species,
        lammps_units_style=lammps_units_style,
        warn_fn=_warn,
        context="output analysis",
    )
    metrics_eff = _analysis_metrics_config(metrics_eff)
    prod_eff = prod_cfg if prod_cfg is not None else config.autotune.production
    conv_eff = conv_cfg if conv_cfg is not None else config.autotune.convergence
    md_use = config.md
    sampling_hint: Optional[dict[str, float]] = None
    return AnalysisContext(
        metrics_cfg=metrics_eff,
        type_to_species=type_to_species,
        prod_cfg=ProductionEnsembleConfig.model_validate(_model_dump_jsonlike(prod_eff)),
        conv_cfg=ConvergenceConfig.model_validate(_model_dump_jsonlike(conv_eff)),
        md_timestep=float(md_use.timestep),
        atom_style=str(md_use.atom_style),
        cutoffs=dict(cutoffs or {}),
        metric_warnings=list(metric_warnings),
        effective_metrics=dict(summary),
        quench_window_steps_range=None,
        sampling_hint=sampling_hint,
        source_selection=None,
        embed_structures=bool(getattr(prod_eff, "embed_structures", True)),
        filter_decision_mode="enforcing",
        analysis_workers=int(getattr(prod_eff, "analysis_workers", 1) or 1),
        analysis_streaming=bool(getattr(prod_eff, "analysis_streaming", True)),
        analysis_max_in_flight=(None if getattr(prod_eff, "analysis_max_in_flight", None) is None else int(getattr(prod_eff, "analysis_max_in_flight"))),
        lammps_units_style=lammps_units_style,
        engine=engine_name,
    )


def _analysis_context_from_plan(config: Optional[RunConfig], plan: Mapping[str, Any]) -> AnalysisContext:
    metric_warnings: list[str] = []
    metrics_cfg = StructureMetricsConfig.model_validate(plan.get("metrics_cfg", {}))
    metrics_cfg = _analysis_metrics_config(metrics_cfg)
    type_to_species = plan.get("type_to_species", None)
    if type_to_species is not None:
        type_to_species = [str(x) for x in type_to_species]
    else:
        type_to_species = _get_type_to_species(config) if config is not None else None
    prod_cfg = ProductionEnsembleConfig.model_validate(plan.get("production_cfg", {}))
    conv_cfg = ConvergenceConfig.model_validate(plan.get("convergence_cfg", {}))
    dft_enabled = bool(getattr(getattr(prod_cfg, "dft_opt", None), "enabled", False))
    sampling_hint = plan.get("sampling_hint", None)
    if isinstance(sampling_hint, Mapping):
        sampling_hint = {str(k): float(v) for k, v in sampling_hint.items() if v is not None}
    else:
        sampling_hint = None
    quench_window = quench_window_steps(
        T_start=float(plan.get("T_high")),
        T_stop=float(plan.get("t_final")),
        total_steps=int(plan.get("quench_steps")),
        T_upper=(sampling_hint or {}).get("Tm") if sampling_hint is not None else None,
        T_lower=(sampling_hint or {}).get("freeze_temperature") if sampling_hint is not None else None,
    )
    cutoffs = _cutoffs_dict_from_any(plan.get("preferred_cutoffs", None) or plan.get("cutoffs_size", None) or plan.get("cutoffs_rate", None))
    md_plan = plan.get("md_use", {}) if isinstance(plan.get("md_use", {}), Mapping) else {}
    engine_name = str(
        plan.get(
            "engine",
            getattr(config, "engine", "lammps") if config is not None else "lammps",
        )
        or "lammps"
    ).strip().lower()
    units_candidate = plan.get("lammps_units_style", plan.get("lammps_units", None))
    if units_candidate in (None, "") and config is not None and engine_name == "lammps":
        units_candidate = getattr(getattr(config, "kim", None), "user_units", None)
    lammps_units_style: Optional[str] = None
    # A plan produced by a non-LAMMPS engine must not inherit stale LAMMPS
    # metadata.  Its canonical structure sources need no conversion; an
    # unexpected raw LAMMPS source will consequently fail closed below.
    if dft_enabled and units_candidate in (None, ""):
        # dft_opt.data is written through the canonical LAMMPS-data bridge in
        # metal units even when the parent trajectory engine is CP2K.
        units_candidate = "metal"
    if (engine_name == "lammps" or dft_enabled) and units_candidate not in (None, ""):
        from ..lammps_units import normalize_lammps_units_style

        lammps_units_style = normalize_lammps_units_style(str(units_candidate))
    return AnalysisContext(
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
        prod_cfg=prod_cfg,
        conv_cfg=conv_cfg,
        md_timestep=float(md_plan.get("timestep", getattr(getattr(config, "md", None), "timestep", 1.0))),
        atom_style=str(md_plan.get("atom_style", getattr(getattr(config, "md", None), "atom_style", "atomic"))),
        cutoffs=cutoffs,
        metric_warnings=list(metric_warnings),
        effective_metrics=dict(plan.get("effective_metrics", {}) or {}),
        quench_window_steps_range=quench_window,
        sampling_hint=sampling_hint,
        source_selection=None,
        embed_structures=bool(getattr(prod_cfg, "embed_structures", True)),
        filter_decision_mode="enforcing",
        analysis_workers=int(getattr(prod_cfg, "analysis_workers", 1) or 1),
        analysis_streaming=bool(getattr(prod_cfg, "analysis_streaming", True)),
        analysis_max_in_flight=(None if getattr(prod_cfg, "analysis_max_in_flight", None) is None else int(getattr(prod_cfg, "analysis_max_in_flight"))),
        lammps_units_style=lammps_units_style,
        engine=engine_name,
        exact_plan_cutoffs=True,
    )


def _dataset_record_for_box(box: DiscoveredBox, *, base_dir: Path) -> dict[str, Any]:
    return {
        "box": int(box.box),
        "box_dir": _relpath_or_str(box.box_dir, base_dir),
        "melt_dir": _relpath_or_str(box.melt_dir, base_dir),
        "quench_dir": _relpath_or_str(box.quench_dir, base_dir),
        "relax_dir": _relpath_or_str(box.relax_dir, base_dir),
        "input_structure": _relpath_or_str(box.input_structure, base_dir),
        "final_structure": _relpath_or_str(box.final_structure, base_dir),
        "relax_data": _relpath_or_str(box.relax_data, base_dir),
        "relax_dump": _relpath_or_str(box.relax_dump, base_dir),
        "relax_traj": _relpath_or_str(box.relax_traj, base_dir),
        "analysis_source": _relpath_or_str(box.analysis_source, base_dir),
        "analysis_source_role": box.analysis_source_role,
        "source_layout": box.source_layout,
        "source_record": (None if box.source_record is None else dict(box.source_record)),
        "density": box.density,
        "density_stderr": box.density_stderr,
        "task_result": _relpath_or_str(box.task_result, base_dir),
    }


def _required_pair_keys(required_pairs: Sequence[Tuple[Any, Any]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in list(required_pairs or []):
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _analysis_source_role_counts(raw_boxes: Sequence[DiscoveredBox]) -> dict[str, int]:
    counts = Counter(str(box.analysis_source_role or "unknown") for box in raw_boxes)
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _analysis_frames_for_box(
    box: DiscoveredBox,
    *,
    metrics_cfg: StructureMetricsConfig,
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
    lammps_units_style: Optional[str],
) -> tuple[list[Any], Optional[str]]:
    source = box.analysis_source or box.relax_traj or box.relax_data
    if source is None:
        return [], "missing_analysis_source"
    p = Path(source)
    n_frames = max(1, int(getattr(metrics_cfg, "time_average_frames", 1) or 1))
    try:
        if str(box.source_layout or "") == "ase_database":
            frames = _frames_from_ase_database_box(
                box,
                type_to_species=type_to_species,
            )
        else:
            if not p.exists():
                return [], f"analysis_source_not_found: {p}"
            frames = read_last_frames_auto(
                p,
                int(n_frames),
                type_to_species=type_to_species,
                atom_style=str(atom_style),
                units_style=lammps_units_style,
            )
    except Exception as exc:
        return [], f"failed_to_read_analysis_source {p}: {exc}"
    if not frames:
        return [], f"no_frames_from_analysis_source: {p}"
    return list(frames), None




def _metrics_have_adaptive_graph_rules(metrics: StructureMetricsConfig) -> bool:
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
    for rule in list(getattr(metrics, "graph_rules", []) or []):
        kind = str(getattr(rule, "kind", ""))
        params = getattr(rule, "parameters", {}) or {}
        derive = str(params.get("derive_from", "")).strip().lower() if isinstance(params, dict) else ""
        source = str(params.get("source", params.get("cutoff_source", ""))).strip().lower() if isinstance(params, dict) else ""
        if kind in adaptive or derive in {"rdf", "rdf_minimum", "rdf_first_minimum", "pair_distribution", "pair_distribution_function", "shell_separability"} or source in {"rdf", "rdf_minimum", "rdf_first_minimum", "pair_distribution", "pair_distribution_function", "shell_separability"}:
            return True
    return False


def _merge_analysis_graph_rules(
    metrics: StructureMetricsConfig,
    additional_rules: Optional[Sequence[Mapping[str, Any]]],
) -> StructureMetricsConfig:
    """Add CLI graph rules to configured rules without ambiguous replacement.

    Rule names are output/provenance identifiers.  Reusing a name would make
    sidecar rows indistinguishable, so collisions (including duplicates already
    present in YAML) are rejected rather than silently overwritten or renamed.
    """

    if additional_rules is None:
        return metrics

    configured: list[dict[str, Any]] = []
    for rule in list(getattr(metrics, "graph_rules", []) or []):
        if hasattr(rule, "model_dump"):
            configured.append(dict(rule.model_dump(mode="python")))
        elif isinstance(rule, Mapping):
            configured.append(dict(rule))
        else:
            raise TypeError(f"Configured graph rule must be mapping-like, got {type(rule)!r}")

    additions = [dict(rule) for rule in list(additional_rules or [])]
    merged = configured + additions
    names: dict[str, str] = {}
    for idx, rule in enumerate(merged):
        name = str(rule.get("name", "graph_rule")).strip()
        if not name:
            raise ValueError(f"Graph rule at index {idx} has an empty name")
        source = "configured YAML" if idx < len(configured) else "CLI"
        if name in names:
            raise ValueError(
                f"Duplicate graph-rule name {name!r}: {names[name]} and {source}. "
                "Graph-rule names must be unique so metric provenance is unambiguous."
            )
        names[name] = source

    data = metrics.model_dump(mode="python")
    data["graph_rules"] = merged
    return StructureMetricsConfig.model_validate(data)

def _resolve_output_analysis_cutoffs(
    *,
    raw_boxes: Sequence[DiscoveredBox],
    ctx: AnalysisContext,
    required_pairs: Sequence[Tuple[int, int]],
    fixed_cutoffs: Mapping[Tuple[int, int], float],
) -> tuple[dict[Tuple[int, int], float], dict[str, Any]]:
    required_keys = _required_pair_keys(required_pairs)
    fixed = dict(fixed_cutoffs)
    plan_cutoffs = dict(ctx.cutoffs or {})
    missing_pairs = [pair for pair in required_keys if pair not in fixed]
    scope = str(getattr(ctx.metrics_cfg.auto_cutoff, "scope", "pooled_ensemble"))
    if scope not in {"pooled_ensemble", "per_box", "disabled"}:
        raise ValueError(
            "metrics.auto_cutoff.scope must be one of "
            "'pooled_ensemble', 'per_box', or 'disabled'"
        )

    provenance: dict[str, Any] = {
        "mode": "metrics_fixed_only",
        "scope": scope,
        "required_pairs": [[int(a), int(b)] for (a, b) in required_keys],
        "fixed_pairs": [[int(a), int(b)] for (a, b) in sorted(fixed)],
        "plan_cutoffs_available": bool(plan_cutoffs),
        "plan_cutoffs_reused": False,
        "n_boxes_sampled": 0,
        "n_frames_sampled": 0,
        "analysis_source_roles": _analysis_source_role_counts(raw_boxes),
        "errors": [],
        "notes": [],
    }

    if not missing_pairs:
        provenance["notes"].append("All required pairs were covered by explicit metric cutoffs.")
        return fixed, provenance

    if (
        bool(getattr(ctx, "exact_plan_cutoffs", False))
        and plan_cutoffs
        and all(pair in plan_cutoffs for pair in missing_pairs)
    ):
        # A production plan records the exact shared cutoffs used to construct
        # the source ensemble.  Re-estimating them from the finished boxes can
        # change graph membership and make analyze-output answer a different
        # question from production convergence.
        exact = dict(plan_cutoffs)
        exact.update(fixed)
        provenance["mode"] = "production_plan_exact_replay"
        provenance["plan_cutoffs_reused"] = True
        provenance["notes"].append(
            "Reused the production-plan cutoff map exactly for convergence parity."
        )
        return exact, provenance

    if _metrics_have_adaptive_graph_rules(ctx.metrics_cfg) and not fixed:
        provenance["mode"] = "adaptive_graph_rules_only"
        provenance["notes"].append(
            "No legacy single-cutoff map was constructed because adaptive RDF graph rules are configured; "
            "graph-derived descriptors are evaluated through per-structure graph_rule provenance only."
        )
        return {}, provenance

    if scope == "disabled":
        # Plan cutoffs are persisted explicit values and may fill gaps, but
        # metric-level cutoffs have precedence when both sources define a pair.
        explicit = dict(plan_cutoffs)
        explicit.update(fixed)
        still_missing = [pair for pair in missing_pairs if pair not in explicit]
        if still_missing:
            missing_text = ", ".join(f"({a},{b})" for a, b in still_missing)
            raise ValueError(
                "metrics.auto_cutoff.scope='disabled' requires a complete "
                "explicit cutoff map; missing required pair(s): "
                f"{missing_text}. Set metric cutoffs or supply them through "
                "the standalone analysis/production plan."
            )
        provenance["mode"] = "disabled_explicit"
        provenance["plan_cutoffs_reused"] = any(
            pair not in fixed and pair in plan_cutoffs for pair in required_keys
        )
        provenance["notes"].append(
            "Automatic cutoff estimation is disabled; all required pairs use explicit cutoffs."
        )
        return explicit, provenance

    if scope == "per_box":
        provenance["mode"] = "per_box_auto"
        provenance["notes"].append(
            "Missing cutoffs will be estimated independently from each box; "
            "no cutoff inferred for one box is reused for another."
        )
        return fixed, provenance

    pooled_frames: list[Any] = []
    read_errors: list[str] = []
    for box in sorted(raw_boxes, key=lambda x: int(x.box)):
        frames, err = _analysis_frames_for_box(
            box,
            metrics_cfg=ctx.metrics_cfg,
            type_to_species=ctx.type_to_species,
            atom_style=str(ctx.atom_style),
            lammps_units_style=ctx.lammps_units_style,
        )
        if err is not None:
            read_errors.append(str(err))
            continue
        pooled_frames.extend(list(frames))
        provenance["n_boxes_sampled"] = int(provenance["n_boxes_sampled"]) + 1
        provenance["n_frames_sampled"] = int(provenance["n_frames_sampled"]) + len(list(frames))

    provenance["errors"] = list(read_errors)

    if pooled_frames:
        try:
            from ..analysis.structure import estimate_pair_cutoffs

            cutoffs = estimate_pair_cutoffs(
                pooled_frames,
                required_keys,
                auto=ctx.metrics_cfg.auto_cutoff,
                fixed_cutoffs=fixed,
            )
            provenance["mode"] = "pooled_ensemble_auto"
            provenance["notes"].append(
                "Estimated auto cutoffs from pooled frames read from the current analysis ensemble."
            )
            return cutoffs, provenance
        except Exception as exc:
            provenance["errors"].append(f"pooled_cutoff_estimation_failed: {exc}")

    if plan_cutoffs:
        missing_after_plan = [pair for pair in missing_pairs if pair not in plan_cutoffs]
        if not missing_after_plan:
            merged = dict(plan_cutoffs)
            merged.update(fixed)
            provenance["mode"] = "plan_fallback"
            provenance["plan_cutoffs_reused"] = True
            provenance["notes"].append(
                "Fell back to plan cutoffs because pooled auto estimation was unavailable or failed."
            )
            return merged, provenance

    provenance["mode"] = "per_box_fallback"
    provenance["fallback_policy"] = "first_successful_box_then_shared"
    provenance["notes"].append(
        "Pooled cutoff pre-resolution was unavailable. Preserving the legacy "
        "default by deriving the missing shared cutoff map from the first "
        "successfully analysed box, then reusing that map for later boxes. "
        "Set scope='per_box' to request genuinely independent per-box cutoffs."
    )
    return fixed, provenance




def _analysis_familywise_defaults(conv_cfg: ConvergenceConfig, *, m_tests: int = 0) -> dict[str, Any]:
    try:
        from statistics import NormalDist

        z = float(getattr(conv_cfg, "zscore", 1.96))
        alpha_family = 2.0 * max(0.0, 1.0 - float(NormalDist().cdf(abs(z))))
        if not (0.0 < alpha_family < 1.0):
            alpha_family = 0.05
    except Exception:
        z = 1.96
        alpha_family = 0.05
    familywise = str(getattr(conv_cfg, "familywise", "none"))
    if familywise == "bonferroni" and int(m_tests) > 1:
        alpha_test = float(alpha_family) / float(m_tests)
    else:
        alpha_test = float(alpha_family)
    if not (0.0 < alpha_test < 1.0):
        alpha_test = 0.05
    return {
        "method": familywise,
        "alpha_family": float(alpha_family),
        "m_tests": int(m_tests),
        "alpha_per_test": float(alpha_test),
        "crit": None,
        "crit_method": "not_evaluated",
        "bounded_ci_method": str(getattr(conv_cfg, "bounded_ci_method", "t")),
    }


def _descriptor_set_items_from_spec(spec: Optional[Mapping[str, Any]]) -> dict[str, list[str]]:
    spec = dict(spec or {})
    scalar_names = [str(name) for name in list(spec.get("scalar_names", []) or [])]
    return {
        "short": [
            *[name for name in scalar_names if name.startswith(("bondlen_", "angle_", "coord_", "gr_"))],
            *[f"bondlen_cdf:{x}" for x in list(spec.get("bondlen_names", []) or [])],
            *[f"angle_cdf:{x}" for x in list(spec.get("angle_names", []) or [])],
            *[f"coord_cdf:{x}" for x in list(spec.get("coord_names", []) or [])],
        ],
        "medium": [
            *[str(x) for x in list(spec.get("ring_keys", []) or [])],
            *( ["ring_mean_size"] if bool(spec.get("ring_has_mean_size", False)) else [] ),
        ],
        "long": [
            "density",
            *[f"gr_curve:{x}" for x in list(spec.get("gr_labels", []) or [])],
            *[f"sq_curve:{x}" for x in list(spec.get("sq_labels", []) or [])],
            *[f"void_cdf:{x}" for x in list(spec.get("void_names", []) or [])],
        ],
    }


def _analysis_convergence_unavailable_report(
    *,
    boxes: Sequence[Mapping[str, Any]],
    conv_spec: Optional[Mapping[str, Any]],
    conv_cfg: ConvergenceConfig,
    reason: str,
    check_enabled: bool,
    prod_cfg: ProductionEnsembleConfig,
) -> dict[str, Any]:
    items = _descriptor_set_items_from_spec(conv_spec)
    m_tests = sum(len(v) for v in items.values())
    n_boxes = int(len(boxes or []))
    min_boxes = int(getattr(prod_cfg, "min_boxes", 1) or 1)
    groups: dict[str, Any] = {}
    for name in ("short", "medium", "long"):
        groups[name] = {
            "passed": None,
            "status": "not_evaluated" if check_enabled else "skipped",
            "items": list(items.get(name, [])),
            "reason": str(reason),
        }
    return json_sanitize(
        {
            "schema": "vitriflow.analysis_descriptor_convergence.v1",
            "advisory": True,
            "status": "not_evaluated" if check_enabled else "skipped",
            "reason": str(reason),
            "mode": str(getattr(conv_cfg, "mode", "both")),
            "n_boxes": int(n_boxes),
            "familywise": _analysis_familywise_defaults(conv_cfg, m_tests=m_tests),
            "groups": groups,
            "ensemble_size": {
                "n_boxes": int(n_boxes),
                "min_boxes": int(min_boxes),
                "passed": bool(n_boxes >= min_boxes),
                "status": "ok" if n_boxes >= min_boxes else "insufficient_samples",
                "reason": "" if n_boxes >= min_boxes else "analysed_box_count_below_configured_min_boxes",
            },
            "scalars": {},
            "distributions": {},
            "stability": {"enabled": False, "status": "not_evaluated", "reason": str(reason)},
            "ci_converged": False,
            "stability_converged": False,
            "converged": False,
            "passed": False,
        }
    )


def _annotate_analysis_convergence_report(
    report: Mapping[str, Any],
    *,
    boxes: Sequence[Mapping[str, Any]],
    conv_spec: Optional[Mapping[str, Any]],
    prod_cfg: ProductionEnsembleConfig,
    advisory: bool,
) -> dict[str, Any]:
    out = dict(report or {})
    out.setdefault("schema", "vitriflow.analysis_descriptor_convergence.v1")
    out["advisory"] = bool(advisory)
    out.setdefault("status", "ok")
    n_boxes = int(len(boxes or []))
    min_boxes = int(getattr(prod_cfg, "min_boxes", 1) or 1)
    out["ensemble_size"] = {
        "n_boxes": int(n_boxes),
        "min_boxes": int(min_boxes),
        "passed": bool(n_boxes >= min_boxes),
        "status": "ok" if n_boxes >= min_boxes else "insufficient_samples",
        "reason": "" if n_boxes >= min_boxes else "analysed_box_count_below_configured_min_boxes",
    }
    groups = dict(out.get("groups", {}) or {})
    declared = _descriptor_set_items_from_spec(conv_spec)
    for name in ("short", "medium", "long"):
        g = dict(groups.get(name, {}) or {})
        g.setdefault("items", list(declared.get(name, [])))
        if "passed" in g and g.get("passed") is not None:
            g.setdefault("status", "ok" if bool(g.get("passed")) else "not_converged")
        else:
            g.setdefault("passed", None)
            g.setdefault("status", "not_evaluated" if not g.get("items") else "not_converged")
        g.setdefault("reason", "" if g.get("items") else "no_metrics_declared_for_descriptor_range")
        groups[name] = g
    out["groups"] = groups
    out.setdefault("familywise", _analysis_familywise_defaults(ConvergenceConfig(), m_tests=sum(len(v.get("items", [])) for v in groups.values() if isinstance(v, Mapping))))
    return json_sanitize(out)




def _analysis_curve_group(kind: str) -> str:
    kind = str(kind)
    if kind in {"bondlen_cdf", "angle_cdf", "coord_cdf"}:
        return "short"
    if kind in {"ring", "ring_metric", "ring_mean_size"}:
        return "medium"
    return "long"


def _analysis_curve_tolerances(kind: str, conv_cfg: ConvergenceConfig) -> tuple[float, float]:
    mapping = {
        "bondlen_cdf": ("bondlen_cdf_rel_tol", "bondlen_cdf_abs_tol", 0.0, 0.02),
        "angle_cdf": ("angle_cdf_rel_tol", "angle_cdf_abs_tol", 0.0, 0.02),
        "coord_cdf": ("coord_cdf_rel_tol", "coord_cdf_abs_tol", 0.0, 0.02),
        "void_cdf": ("void_cdf_rel_tol", "void_cdf_abs_tol", 0.0, 0.02),
        "gr_curve": ("gr_curve_rel_tol", "gr_curve_abs_tol", 0.05, 0.05),
        "sq_curve": ("sq_curve_rel_tol", "sq_curve_abs_tol", 0.05, 0.05),
        "ring": ("ring_rel_tol", "ring_abs_tol", 0.0, 0.05),
        "ring_mean_size": ("ring_size_rel_tol", "ring_size_abs_tol", 0.0, 0.5),
    }
    rel_attr, abs_attr, rel_default, abs_default = mapping.get(str(kind), ("", "", 0.0, 0.0))
    try:
        rel = float(getattr(conv_cfg, rel_attr, rel_default))
    except Exception:
        rel = float(rel_default)
    try:
        abs_tol = float(getattr(conv_cfg, abs_attr, abs_default))
    except Exception:
        abs_tol = float(abs_default)
    if not math.isfinite(rel):
        rel = float(rel_default)
    if not math.isfinite(abs_tol):
        abs_tol = float(abs_default)
    return float(rel), float(abs_tol)


def _analysis_box_label(box: Mapping[str, Any], idx: int) -> Any:
    return box.get("box", box.get("box_id", idx + 1))


def _finite_1d(values: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


_ANALYSIS_CDF_ROUNDOFF_ATOL = 1.0e-12


def _validated_analysis_curve_payload(
    payload: Mapping[str, Any],
    *,
    axis_key: str,
    value_key: str,
    is_cdf: bool,
    allow_implicit_integer_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Validate a stored analysis curve without repairing scientific input.

    The only legacy axis inference supported here is an *absent* coordination
    ``x`` field, whose CDF entries historically denoted the integer support
    ``0, 1, ...``.  A present-but-empty or malformed axis is not legacy input
    and fails validation.  All current writers store an explicit physical axis,
    which is essential for fractional coordination from soft graphs.
    """

    y = _finite_1d(payload.get(value_key, []))
    if y is None:
        raise ValueError(f"{value_key} must be a non-empty one-dimensional finite array")

    if allow_implicit_integer_axis and axis_key not in payload:
        x = np.arange(int(y.size), dtype=float)
        axis_source = "implicit_integer_index_legacy"
    else:
        x = _finite_1d(payload.get(axis_key, []))
        if x is None:
            raise ValueError(f"{axis_key} must be a non-empty one-dimensional finite array")
        axis_source = "explicit_stored_axis"

    if x.size != y.size:
        raise ValueError(
            f"{axis_key}/{value_key} length mismatch ({int(x.size)} != {int(y.size)})"
        )
    if x.size > 1 and not np.all(np.diff(x) > 0.0):
        raise ValueError(f"{axis_key} must be strictly increasing with no duplicates")
    if not is_cdf and x.size < 2:
        raise ValueError("sampled non-CDF curve requires at least two axis points")

    if is_cdf:
        atol = float(_ANALYSIS_CDF_ROUNDOFF_ATOL)
        if np.any(y < -atol) or np.any(y > 1.0 + atol):
            raise ValueError(f"{value_key} values must lie in [0, 1]")
        if y.size > 1 and np.any(np.diff(y) < -atol):
            raise ValueError(f"{value_key} must be nondecreasing")
        # Normalise only validator-accepted floating-point roundoff.  Gross
        # out-of-range values or descents have already failed above.
        y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), str(axis_source)


def _evaluate_right_continuous_cdf(x: np.ndarray, cdf: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Evaluate an empirical CDF on a common support without linear smoothing.

    CDFs are step functions.  For ensemble-level descriptor CDFs produced on
    box-specific supports, use the union support and right-continuous step
    evaluation.  This avoids inventing probability mass between support points.
    """

    x = np.asarray(x, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    if (
        x.ndim != 1
        or cdf.ndim != 1
        or x_ref.ndim != 1
        or x.size == 0
        or cdf.size != x.size
        or x_ref.size == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(cdf))
        or not np.all(np.isfinite(x_ref))
        or (x.size > 1 and not np.all(np.diff(x) > 0.0))
        or (x_ref.size > 1 and not np.all(np.diff(x_ref) > 0.0))
    ):
        raise ValueError("CDF evaluation requires finite, strictly increasing, matching one-dimensional arrays")
    atol = float(_ANALYSIS_CDF_ROUNDOFF_ATOL)
    if np.any(cdf < -atol) or np.any(cdf > 1.0 + atol) or (
        cdf.size > 1 and np.any(np.diff(cdf) < -atol)
    ):
        raise ValueError("CDF evaluation requires nondecreasing values in [0, 1]")
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    out = np.zeros_like(x_ref, dtype=float)
    idx = np.searchsorted(x, x_ref, side="right") - 1
    valid = idx >= 0
    out[valid] = cdf[idx[valid]]
    out[x_ref >= x[-1]] = float(cdf[-1])
    return out


def _empty_ensemble_curve_entry(
    *,
    name: str,
    kind: str,
    axis_key: str,
    value_key: str,
    n_boxes: int,
    reason: str,
    missing_boxes: Sequence[Any] = (),
    invalid_payloads: Sequence[Mapping[str, Any]] = (),
    conv_cfg: ConvergenceConfig,
) -> dict[str, Any]:
    rel_tol, abs_tol = _analysis_curve_tolerances(kind, conv_cfg)
    invalid_payloads = [dict(row) for row in invalid_payloads if isinstance(row, Mapping)]
    invalid_boxes = [row.get("box") for row in invalid_payloads]
    blockers = [
        {"box": box, "kind": "missing_payload", "reason": "curve payload is absent"}
        for box in missing_boxes
        if box not in invalid_boxes
    ]
    blockers.extend(invalid_payloads)
    return {
        "name": str(name),
        "group": _analysis_curve_group(kind),
        "kind": str(kind),
        "status": "unavailable",
        "metric_status": "unavailable",
        "metric_status_reason": str(reason),
        "numerical_status": "unavailable",
        "uncertainty_status": "not_applicable",
        "uncertainty_status_reason": str(reason),
        "n_boxes": int(n_boxes),
        "n_available": 0,
        "n_missing": int(len(missing_boxes)),
        "missing_boxes": list(missing_boxes),
        "n_invalid": int(len(invalid_payloads)),
        "invalid_boxes": list(invalid_boxes),
        "invalid_payloads": invalid_payloads,
        "evidence_complete": False,
        "convergence_assessed": False,
        "convergence_status": "unassessed_incomplete_evidence",
        "blocking_boxes": blockers,
        axis_key: [],
        "mean": [],
        "stderr": [],
        "ci_halfwidth": [],
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "tol": [],
        "passed": None,
        "available_subset_ci_within_tolerance": None,
        "axis": {"key": str(axis_key), "length": 0},
        "value_key": str(value_key),
        "normalization": "per_box_unweighted_mean_of_box_curves",
    }


def _analysis_ensemble_curve_entry(
    boxes: Sequence[Mapping[str, Any]],
    *,
    family: str,
    name: str,
    kind: str,
    axis_key: str,
    value_key: str,
    conv_cfg: ConvergenceConfig,
    allow_implicit_integer_axis: bool = False,
) -> dict[str, Any]:
    is_cdf = str(kind).endswith("cdf")
    raw: list[tuple[Any, np.ndarray, np.ndarray, Mapping[str, Any], str, int]] = []
    missing: list[Any] = []
    invalid_payloads: list[dict[str, Any]] = []

    def _validated_sample_count(payload: Mapping[str, Any]) -> int:
        declared = [
            (key, payload[key])
            for key in ("sample_count", "n_samples")
            if key in payload
        ]
        if not declared:
            # Older smooth-curve payloads did not record a count.  Preserve
            # that explicit legacy meaning without inventing evidence.
            return 0
        parsed: list[tuple[str, int]] = []
        for key, value in declared:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"{key} must be a finite exact nonnegative integer, got {value!r}"
                )
            try:
                numeric = Decimal(
                    str(value).strip().replace("D", "E").replace("d", "e")
                )
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{key} must be a finite exact nonnegative integer, got {value!r}"
                ) from exc
            if (
                not numeric.is_finite()
                or numeric != numeric.to_integral_value()
                or numeric < 0
            ):
                raise ValueError(
                    f"{key} must be a finite exact nonnegative integer, got {value!r}"
                )
            parsed.append((key, int(numeric)))
        if len({value for _key, value in parsed}) != 1:
            details = ", ".join(f"{key}={value}" for key, value in parsed)
            raise ValueError(f"conflicting declared sample counts: {details}")
        return int(parsed[0][1])

    for idx, box in enumerate(boxes):
        label = _analysis_box_label(box, idx)
        dist = box.get("distributions", {}) if isinstance(box, Mapping) else {}
        fam = (dist.get(family, {}) or {}) if isinstance(dist, Mapping) else {}
        payload = fam.get(name, None) if isinstance(fam, Mapping) else None
        if not isinstance(payload, Mapping):
            missing.append(label)
            continue
        try:
            x, y, axis_source = _validated_analysis_curve_payload(
                payload,
                axis_key=axis_key,
                value_key=value_key,
                is_cdf=is_cdf,
                allow_implicit_integer_axis=allow_implicit_integer_axis,
            )
            sample_count = _validated_sample_count(payload)
        except (TypeError, ValueError) as exc:
            missing.append(label)
            invalid_payloads.append(
                {
                    "box": label,
                    "kind": "malformed_payload",
                    "reason": str(exc),
                }
            )
            continue
        raw.append((label, x, y, payload, axis_source, sample_count))

    if not raw:
        return _empty_ensemble_curve_entry(
            name=name,
            kind=kind,
            axis_key=axis_key,
            value_key=value_key,
            n_boxes=len(boxes),
            reason="no valid per-box curve payloads were available",
            missing_boxes=missing,
            invalid_payloads=invalid_payloads,
            conv_cfg=conv_cfg,
        )

    ref_x = raw[0][1]
    same_grid = all(
        x.shape == ref_x.shape
        and (x.size == 0 or float(np.max(np.abs(x - ref_x))) <= 1e-10)
        for _, x, _, _, _, _ in raw
    )
    if is_cdf:
        # CDFs are step functions.  Align on the union support and use
        # right-continuous evaluation, not linear interpolation.
        xs = np.concatenate([x for _, x, _, _, _, _ in raw if x.size > 0])
        if xs.size == 0:
            return _empty_ensemble_curve_entry(
                name=name,
                kind=kind,
                axis_key=axis_key,
                value_key=value_key,
                n_boxes=len(boxes),
                reason="finite CDF payloads have empty supports",
                missing_boxes=missing,
                invalid_payloads=invalid_payloads,
                conv_cfg=conv_cfg,
            )
        x_ref = np.unique(np.sort(xs, kind="mergesort"))
        interpolation_note = "union_support_right_continuous_cdf"
    else:
        # Smooth curves such as g(r) and S(q) are aligned on the common overlap.
        interpolation_note = "none_exact_common_grid"
        if same_grid:
            x_ref = ref_x
        else:
            lo = max(float(x[0]) for _, x, _, _, _, _ in raw if x.size)
            hi = min(float(x[-1]) for _, x, _, _, _, _ in raw if x.size)
            if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                return _empty_ensemble_curve_entry(
                    name=name,
                    kind=kind,
                    axis_key=axis_key,
                    value_key=value_key,
                    n_boxes=len(boxes),
                    reason="finite per-box curves have no common axis overlap",
                    missing_boxes=missing,
                    invalid_payloads=invalid_payloads,
                    conv_cfg=conv_cfg,
                )
            n_grid = int(
                max(2, min(max(int(x.size) for _, x, _, _, _, _ in raw), int(ref_x.size)))
            )
            x_ref = np.linspace(lo, hi, n_grid, dtype=float)
            interpolation_note = "interpolated_to_common_axis_overlap"

    rows = []
    sample_counts = []
    for _label, x, y, _payload, _axis_source, sample_count in raw:
        if is_cdf:
            yi = _evaluate_right_continuous_cdf(x, y, x_ref)
        elif same_grid:
            yi = np.asarray(y, dtype=float)
        else:
            yi = np.interp(x_ref, x, y)
        rows.append(yi)
        sample_counts.append(int(sample_count))

    mat = np.vstack(rows).astype(float)
    n_available = int(mat.shape[0])
    mu = np.mean(mat, axis=0)
    rel_tol, abs_tol = _analysis_curve_tolerances(kind, conv_cfg)
    tol = np.maximum(float(abs_tol), float(rel_tol) * np.abs(mu))
    evidence_complete = bool(not missing and n_available == len(boxes))
    subset_passed: Optional[bool] = None
    if n_available >= 2:
        sd = np.std(mat, axis=0, ddof=1)
        se = sd / math.sqrt(float(n_available))
        z = float(getattr(conv_cfg, "zscore", 1.96) or 1.96)
        if not math.isfinite(z) or z <= 0.0:
            z = 1.96
        half = z * se
        subset_passed = bool(np.all(half <= tol))
        if evidence_complete:
            passed: Optional[bool] = bool(subset_passed)
            uncertainty_status = "ok"
            uncertainty_reason = ""
        else:
            passed = None
            uncertainty_status = "unassessed_incomplete_evidence"
            uncertainty_reason = "one_or_more_declared_boxes_have_missing_or_malformed_curve_payloads"
        stderr = [float(v) for v in se.tolist()]
        ci_halfwidth = [float(v) for v in half.tolist()]
    else:
        passed = None
        uncertainty_status = "not_applicable"
        uncertainty_reason = "single_structure_or_single_available_curve"
        stderr = [None for _ in range(int(mu.size))]
        ci_halfwidth = [None for _ in range(int(mu.size))]

    convergence_assessed = bool(evidence_complete and n_available >= 2)
    convergence_status = (
        "assessed"
        if convergence_assessed
        else (
            "unassessed_incomplete_evidence"
            if not evidence_complete
            else "unassessed_insufficient_samples"
        )
    )
    status = "ok" if evidence_complete else "incomplete"
    reason = "" if evidence_complete else "one_or_more_boxes_missing_or_malformed_this_curve"
    invalid_boxes = [row.get("box") for row in invalid_payloads]
    blockers = [
        {"box": box, "kind": "missing_payload", "reason": "curve payload is absent"}
        for box in missing
        if box not in invalid_boxes
    ]
    blockers.extend(invalid_payloads)
    axis_sources = [str(axis_source) for _, _, _, _, axis_source, _ in raw]
    entry = {
        "name": str(name),
        "group": _analysis_curve_group(kind),
        "kind": str(kind),
        "status": status,
        "metric_status": status,
        "metric_status_reason": reason,
        "numerical_status": "ok",
        "uncertainty_status": uncertainty_status,
        "uncertainty_status_reason": uncertainty_reason,
        "n_boxes": int(len(boxes)),
        "n_available": int(n_available),
        "n_missing": int(len(missing)),
        "missing_boxes": list(missing),
        "n_invalid": int(len(invalid_payloads)),
        "invalid_boxes": list(invalid_boxes),
        "invalid_payloads": invalid_payloads,
        "evidence_complete": bool(evidence_complete),
        "convergence_assessed": bool(convergence_assessed),
        "convergence_status": str(convergence_status),
        "blocking_boxes": blockers,
        "sample_count_total": int(sum(sample_counts)),
        "sample_counts_by_available_box": [int(x) for x in sample_counts],
        axis_key: [float(v) for v in np.asarray(x_ref, dtype=float).tolist()],
        "mean": [float(v) for v in np.asarray(mu, dtype=float).tolist()],
        "stderr": stderr,
        "ci_halfwidth": ci_halfwidth,
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "tol": [float(v) for v in np.asarray(tol, dtype=float).tolist()],
        "passed": passed,
        "available_subset_ci_within_tolerance": subset_passed,
        "axis": {
            "key": str(axis_key),
            "length": int(x_ref.size),
            "interpolation": interpolation_note,
            "source": (
                axis_sources[0]
                if len(set(axis_sources)) == 1
                else "mixed_explicit_and_implicit_legacy_axes"
            ),
        },
        "alignment": {
            "method": interpolation_note,
            "same_grid": bool(same_grid),
            "support": "union" if is_cdf else ("native" if same_grid else "common_overlap"),
        },
        "n_effective": int(n_available),
        "value_key": str(value_key),
        "normalization": "per_box_unweighted_mean_of_box_curves",
        "ensemble_cdf_semantics": "mean_over_analysed_structures_not_atom_count_weighted",
        "source": "boxes[].distributions",
    }
    if kind.endswith("cdf"):
        entry["cdf"] = entry["mean"]
        entry["ensemble_mean_cdf"] = entry["mean"]
    return entry


def _analysis_ensemble_coord_entry(
    boxes: Sequence[Mapping[str, Any]],
    *,
    name: str,
    conv_cfg: ConvergenceConfig,
) -> dict[str, Any]:
    entry = _analysis_ensemble_curve_entry(
        boxes,
        family="coord",
        name=name,
        kind="coord_cdf",
        axis_key="x",
        value_key="cdf",
        conv_cfg=conv_cfg,
        # Backward compatibility is deliberately restricted to payloads where
        # ``x`` is wholly absent.  Such legacy CDF arrays encode integer
        # coordination bins by array index.  Current explicit grids, including
        # fractional soft-graph support, are always honoured.
        allow_implicit_integer_axis=True,
    )
    x_ref = np.asarray(entry.get("x", []), dtype=float)
    integer_grid = np.arange(int(x_ref.size), dtype=float)
    integer_support = bool(
        x_ref.shape == integer_grid.shape
        and np.allclose(
            x_ref,
            integer_grid,
            rtol=0.0,
            atol=float(_ANALYSIS_CDF_ROUNDOFF_ATOL),
        )
    )
    alignment = dict(entry.get("alignment", {}) or {})
    alignment["grid_source"] = (
        "ensemble_common_integer_grid"
        if integer_support
        else "ensemble_common_explicit_coordination_grid"
    )
    alignment["coordination_support"] = (
        "integer" if integer_support else "fractional_or_nonuniform"
    )
    entry["alignment"] = alignment
    entry["normalization"] = "per_box_unweighted_mean_of_box_cdfs"
    return entry


def _analysis_distribution_names(
    boxes: Sequence[Mapping[str, Any]],
    *,
    family: str,
    spec: Optional[Mapping[str, Any]],
    spec_key: str,
) -> list[str]:
    names = {str(x) for x in list((spec or {}).get(spec_key, []) or [])}
    for box in boxes:
        dist = box.get("distributions", {}) if isinstance(box, Mapping) else {}
        fam = (dist.get(family, {}) or {}) if isinstance(dist, Mapping) else {}
        if isinstance(fam, Mapping):
            names.update(str(x) for x in fam.keys())
    return sorted(names)


def _build_analysis_ensemble_cdfs(
    boxes: Sequence[Mapping[str, Any]],
    conv_spec: Optional[Mapping[str, Any]],
    conv_cfg: ConvergenceConfig,
) -> dict[str, Any]:
    """Materialise ensemble-level CDF/curve summaries for analysis-only output.

    The production convergence checker may drop a curve from the pass/fail stage
    when an old run has missing/non-finite payloads.  Analysis output must still
    be explicit: every requested descriptor curve receives either a finite
    ensemble CDF/curve or an unavailable record with status/reason.
    """

    boxes = [b for b in list(boxes or []) if isinstance(b, Mapping)]
    spec = dict(conv_spec or {})
    distributions: dict[str, Any] = {}

    for family, spec_key, kind, axis_key, value_key in (
        ("bondlen", "bondlen_names", "bondlen_cdf", "x", "cdf"),
        ("angle", "angle_names", "angle_cdf", "x", "cdf"),
        ("void", "void_names", "void_cdf", "x", "cdf"),
        ("gr", "gr_labels", "gr_curve", "r", "g"),
        ("sq", "sq_labels", "sq_curve", "q", "s"),
    ):
        for name in _analysis_distribution_names(boxes, family=family, spec=spec, spec_key=spec_key):
            distributions[str(name)] = _analysis_ensemble_curve_entry(
                boxes,
                family=family,
                name=str(name),
                kind=kind,
                axis_key=axis_key,
                value_key=value_key,
                conv_cfg=conv_cfg,
            )

    for name in _analysis_distribution_names(boxes, family="coord", spec=spec, spec_key="coord_names"):
        distributions[str(name)] = _analysis_ensemble_coord_entry(boxes, name=str(name), conv_cfg=conv_cfg)

    # Ring fractions are stored as scalar PMF entries, not CDFs, but production
    # plotting and convergence expect the same ``distributions['ring']`` record.
    ring_keys = [str(x) for x in list(spec.get("ring_keys", []) or [])]
    if ring_keys:
        mat_rows: list[list[float]] = []
        missing: list[Any] = []
        blockers: list[dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            label = _analysis_box_label(box, idx)
            metrics = box.get("metrics", {}) if isinstance(box, Mapping) else {}
            row: list[float] = []
            invalid_keys: list[dict[str, str]] = []
            for key in ring_keys:
                if not isinstance(metrics, Mapping) or key not in metrics:
                    invalid_keys.append({"key": str(key), "reason": "missing_metric"})
                    continue
                try:
                    val = float(metrics.get(key))
                except Exception:
                    invalid_keys.append({"key": str(key), "reason": "non_numeric_metric"})
                    continue
                if not math.isfinite(val):
                    invalid_keys.append({"key": str(key), "reason": "non_finite_metric"})
                    continue
                row.append(float(val))
            if not invalid_keys and len(row) == len(ring_keys):
                mat_rows.append(row)
            else:
                missing.append(label)
                blockers.append(
                    {
                        "box": label,
                        "kind": "missing_or_malformed_ring_metrics",
                        "metrics": invalid_keys,
                        "reason": "one_or_more_declared_ring_fraction_metrics_are_missing_or_malformed",
                    }
                )
        rel_tol, abs_tol = _analysis_curve_tolerances("ring", conv_cfg)
        if mat_rows:
            mat = np.asarray(mat_rows, dtype=float)
            mu = np.mean(mat, axis=0)
            tol = np.maximum(float(abs_tol), float(rel_tol) * np.abs(mu))
            evidence_complete = bool(not missing and int(mat.shape[0]) == len(boxes))
            subset_passed: Optional[bool] = None
            if mat.shape[0] >= 2:
                se = np.std(mat, axis=0, ddof=1) / math.sqrt(float(mat.shape[0]))
                z = float(getattr(conv_cfg, "zscore", 1.96) or 1.96)
                if not math.isfinite(z) or z <= 0.0:
                    z = 1.96
                half = z * se
                stderr = [float(v) for v in se.tolist()]
                ci_halfwidth = [float(v) for v in half.tolist()]
                subset_passed = bool(np.all(half <= tol))
                if evidence_complete:
                    passed: Optional[bool] = bool(subset_passed)
                    uncertainty_status = "ok"
                    uncertainty_reason = ""
                else:
                    passed = None
                    uncertainty_status = "unassessed_incomplete_evidence"
                    uncertainty_reason = "one_or_more_declared_boxes_have_missing_or_malformed_ring_metrics"
            else:
                stderr = [None for _ in range(int(mu.size))]
                ci_halfwidth = [None for _ in range(int(mu.size))]
                passed = None
                uncertainty_status = "not_applicable"
                uncertainty_reason = "single_structure_or_single_available_ring_pmf"
            convergence_assessed = bool(evidence_complete and int(mat.shape[0]) >= 2)
            distributions["ring"] = {
                "name": "ring",
                "group": "medium",
                "kind": "pmf",
                "status": "ok" if not missing else "incomplete",
                "metric_status": "ok" if not missing else "incomplete",
                "metric_status_reason": "" if not missing else "one_or_more_boxes_missing_ring_fraction_metrics",
                "numerical_status": "ok",
                "uncertainty_status": uncertainty_status,
                "uncertainty_status_reason": uncertainty_reason,
                "n_boxes": int(len(boxes)),
                "n_available": int(mat.shape[0]),
                "n_missing": int(len(missing)),
                "missing_boxes": list(missing),
                "evidence_complete": bool(evidence_complete),
                "convergence_assessed": bool(convergence_assessed),
                "convergence_status": (
                    "assessed"
                    if convergence_assessed
                    else (
                        "unassessed_incomplete_evidence"
                        if not evidence_complete
                        else "unassessed_insufficient_samples"
                    )
                ),
                "blocking_boxes": blockers,
                "keys": list(ring_keys),
                "mean": [float(v) for v in mu.tolist()],
                "stderr": stderr,
                "ci_halfwidth": ci_halfwidth,
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "tol": [float(v) for v in tol.tolist()],
                "passed": passed,
                "available_subset_ci_within_tolerance": subset_passed,
                "normalization": "per_box_unweighted_mean_of_ring_fraction_metrics",
            }
        else:
            distributions["ring"] = _empty_ensemble_curve_entry(
                name="ring",
                kind="ring",
                axis_key="keys",
                value_key="pmf",
                n_boxes=len(boxes),
                reason="no finite ring fraction metrics were available",
                missing_boxes=missing,
                invalid_payloads=blockers,
                conv_cfg=conv_cfg,
            )
            distributions["ring"]["keys"] = list(ring_keys)

    declared_items = _descriptor_set_items_from_spec(spec)
    n_declared = sum(len(v) for v in declared_items.values())
    n_ok = sum(1 for v in distributions.values() if isinstance(v, Mapping) and str(v.get("status", "")) == "ok")
    n_incomplete = sum(1 for v in distributions.values() if isinstance(v, Mapping) and str(v.get("status", "")) == "incomplete")
    n_unavailable = sum(1 for v in distributions.values() if isinstance(v, Mapping) and str(v.get("status", "")) == "unavailable")
    blocking_distributions = [
        str(name)
        for name, entry in sorted(distributions.items(), key=lambda kv: str(kv[0]))
        if isinstance(entry, Mapping)
        and (
            str(entry.get("status", "")) != "ok"
            or str(entry.get("convergence_status", "assessed"))
            == "unassessed_incomplete_evidence"
        )
    ]
    return json_sanitize(
        {
            "schema": "vitriflow.analysis_ensemble_cdfs.v1",
            "status": "ok" if n_incomplete == 0 and n_unavailable == 0 else "incomplete",
            "evidence_complete": bool(n_incomplete == 0 and n_unavailable == 0),
            "convergence_status": (
                "complete"
                if n_incomplete == 0 and n_unavailable == 0
                else "unassessed_incomplete_evidence"
            ),
            "blocking_distributions": blocking_distributions,
            "n_boxes": int(len(boxes)),
            "normalization": "per_box_unweighted_mean_of_box_curves",
            "declared_descriptor_items": declared_items,
            "n_declared_descriptor_items": int(n_declared),
            "n_distribution_entries": int(len(distributions)),
            "n_ok": int(n_ok),
            "n_incomplete": int(n_incomplete),
            "n_unavailable": int(n_unavailable),
            "distributions": distributions,
        }
    )


def _merge_analysis_ensemble_cdfs_into_convergence(
    report: Mapping[str, Any],
    ensemble_cdfs: Mapping[str, Any],
    *,
    conv_cfg: ConvergenceConfig,
) -> dict[str, Any]:
    out = dict(report or {})
    if not isinstance(out.get("familywise", None), Mapping):
        # Keep analysis-only/no-convergence fallback explicit, but include enough
        # familywise metadata for plot-production to display CDF coverage when
        # pass/fail convergence was skipped or unavailable.
        dist_count = len((ensemble_cdfs.get("distributions", {}) or {}) if isinstance(ensemble_cdfs, Mapping) else {})
        out["familywise"] = _analysis_familywise_defaults(conv_cfg, m_tests=max(1, int(dist_count)))
    dist = dict(out.get("distributions", {}) or {}) if isinstance(out.get("distributions", {}), Mapping) else {}
    src_dist = ensemble_cdfs.get("distributions", {}) if isinstance(ensemble_cdfs, Mapping) else {}
    for name, ens_entry in sorted((src_dist or {}).items(), key=lambda kv: str(kv[0])):
        if not isinstance(ens_entry, Mapping):
            continue
        if isinstance(dist.get(str(name)), Mapping):
            merged = dict(ens_entry)
            # Production convergence pass/fail values take precedence, but the
            # ensemble CDF entry supplies status/normalization fields if absent.
            merged.update(dict(dist[str(name)]))
            merged.setdefault("ensemble_cdf_status", ens_entry.get("status"))
            merged.setdefault("normalization", ens_entry.get("normalization"))
            dist[str(name)] = merged
        else:
            dist[str(name)] = dict(ens_entry)
    out["distributions"] = dist
    out["ensemble_cdfs"] = dict(ensemble_cdfs)
    notes = list(out.get("notes", []) or [])
    if src_dist:
        notes.append(
            f"Materialized {len(src_dist)} ensemble-level CDF/curve record(s) from boxes[].distributions; "
            "missing descriptors are represented by explicit status records."
        )
    out["notes"] = notes
    return json_sanitize(out)

def _analysis_convergence_report_from_boxes(
    *,
    boxes: Sequence[Mapping[str, Any]],
    conv_spec: Optional[Mapping[str, Any]],
    conv_cfg: ConvergenceConfig,
    prod_cfg: ProductionEnsembleConfig,
    status: str = "ok",
    reason: str = "analysis ensemble CDFs materialized",
) -> dict[str, Any]:
    """Compatibility helper for analysis-only ensemble descriptor CDFs.

    This intentionally builds ensemble-level descriptor curves from
    boxes[].distributions without requiring the production convergence checker.
    It is used by plot-production fallback tests and by downstream callers that
    need CDF sidecars from analysis-only datasets.
    """

    ensemble_cdfs = _build_analysis_ensemble_cdfs(boxes, conv_spec, conv_cfg)
    distributions = dict(ensemble_cdfs.get("distributions", {}) or {})
    skipped = [
        str(name)
        for name, entry in sorted(distributions.items(), key=lambda kv: str(kv[0]))
        if isinstance(entry, Mapping) and str(entry.get("status", "")) == "unavailable"
    ]
    unassessed = [
        str(name)
        for name, entry in sorted(distributions.items(), key=lambda kv: str(kv[0]))
        if isinstance(entry, Mapping)
        and not bool(entry.get("convergence_assessed", False))
    ]
    report = _analysis_convergence_unavailable_report(
        boxes=boxes,
        conv_spec=conv_spec,
        conv_cfg=conv_cfg,
        reason=reason,
        check_enabled=True,
        prod_cfg=prod_cfg,
    )
    report = _merge_analysis_ensemble_cdfs_into_convergence(report, ensemble_cdfs, conv_cfg=conv_cfg)
    report["status"] = str(status)
    report["reason"] = str(reason)
    report["distributions"] = distributions
    report["skipped_metrics"] = skipped
    report["unassessed_metrics"] = unassessed
    return json_sanitize(report)




def _cutoff_list_to_mapping(items: Any) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    if isinstance(items, Mapping):
        for k, v in items.items():
            if isinstance(k, tuple) and len(k) == 2:
                pair = (int(k[0]), int(k[1]))
            else:
                parts = str(k).replace("-", ",").replace("_", ",").split(",")
                if len(parts) != 2:
                    continue
                pair = (int(parts[0]), int(parts[1]))
            out[(min(pair), max(pair))] = float(v)
        return out
    for item in list(items or []):
        if not isinstance(item, Mapping):
            continue
        pair_raw = item.get("pair", None)
        if pair_raw is None:
            continue
        pair_list = list(pair_raw)
        if len(pair_list) != 2:
            continue
        pair = (int(pair_list[0]), int(pair_list[1]))
        out[(min(pair), max(pair))] = float(item.get("cutoff"))
    return out


def _cutoff_mapping_to_list(cutoffs: Mapping[tuple[int, int], float] | None) -> list[dict[str, Any]]:
    return [
        {"pair": [int(a), int(b)], "cutoff": float(v)}
        for (a, b), v in sorted(dict(cutoffs or {}).items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1])))
    ]


def _analysis_worker_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Process one output-analysis box and stream heavy graph rows before returning."""
    from .production_common import analyse_production_box

    try:
        metrics_cfg = StructureMetricsConfig.model_validate(payload.get("metrics_cfg", {}))
        if bool(getattr(metrics_cfg, "collect_during_production_stages", False)):
            raise ValueError(
                "output analysis must not recompute stage metrics in source production directories"
            )
        cutoffs = _cutoff_list_to_mapping(payload.get("cutoffs", []))
        fixed_cutoffs = _cutoff_list_to_mapping(payload.get("fixed_cutoffs", []))
        required_pairs = [tuple(x) for x in list(payload.get("required_pairs", []) or [])]
        entry, box_cutoffs = analyse_production_box(
            box_id=int(payload["box_id"]),
            outdir=Path(payload["outdir"]),
            melt_stage_dir=Path(payload["melt_stage_dir"]),
            quench_stage_dir=Path(payload["quench_stage_dir"]),
            relax_stage_dir=Path(payload["relax_stage_dir"]),
            relax_data_path=Path(payload["relax_data_path"]),
            density_mean=float(payload.get("density_mean", float("nan"))),
            density_stderr=float(payload.get("density_stderr", float("nan"))),
            metrics_cfg=metrics_cfg,
            cutoffs=cutoffs,
            required_pairs=required_pairs,
            fixed_cutoffs=fixed_cutoffs,
            type_to_species=(None if payload.get("type_to_species") is None else [str(x) for x in list(payload.get("type_to_species") or [])]),
            md_timestep=float(payload.get("md_timestep", 1.0)),
            quench_window_steps_range=(None if payload.get("quench_window_steps_range") is None else tuple(payload.get("quench_window_steps_range"))),
            sampling_hint=(None if payload.get("sampling_hint") is None else dict(payload.get("sampling_hint") or {})),
            bondlen_cdf_points=int(payload.get("bondlen_cdf_points", 200)),
            angle_cdf_points=int(payload.get("angle_cdf_points", 180)),
            seeds=None,
            melt_elastic=None,
            relax_elastic=None,
            elastic_timeseries=None,
            exclude_coordination_defects=bool(payload.get("exclude_coordination_defects", False)),
            rejects_dir=(None if payload.get("rejects_dir") is None else Path(payload.get("rejects_dir"))),
            relax_dump_path=(None if payload.get("relax_dump_path") is None else Path(payload.get("relax_dump_path"))),
            relax_traj_path=(None if payload.get("relax_traj_path") is None else Path(payload.get("relax_traj_path"))),
            analysis_source_path=(None if payload.get("analysis_source_path") is None else Path(payload.get("analysis_source_path"))),
            analysis_source_role=(None if payload.get("analysis_source_role") is None else str(payload.get("analysis_source_role"))),
            atom_style=str(payload.get("atom_style", "atomic")),
            lammps_units_style=(
                None
                if payload.get("lammps_units_style") in (None, "")
                else str(payload.get("lammps_units_style"))
            ),
            engine=str(payload.get("engine", "lammps") or "lammps"),
            embed_structures=bool(payload.get("embed_structures", True)),
        )
        reused_diagnostics = payload.get("reused_task_diagnostics", None)
        if reused_diagnostics is not None:
            if not isinstance(reused_diagnostics, Mapping):
                raise ValueError("reused_task_diagnostics must be a mapping")
            entry["stage_metrics"] = reused_diagnostics.get("stage_metrics")
            entry["elastic_melt"] = reused_diagnostics.get("elastic_melt")
            entry["elastic_relax"] = reused_diagnostics.get("elastic_relax")
            entry["elastic_timeseries"] = reused_diagnostics.get("elastic_timeseries")
            entry["task_diagnostics_provenance"] = dict(
                reused_diagnostics.get("provenance", {}) or {}
            )
        elif str(payload.get("task_diagnostics_contract", "")) == "legacy_absent":
            entry["task_diagnostics_provenance"] = {
                "schema": "vitriflow.reused_task_diagnostics.v1",
                "mode": "legacy_task_result_without_diagnostics",
                "task_result": payload.get("task_result_path"),
            }
        chunk_summary = None
        if bool(payload.get("stream_graph_outputs", False)) and payload.get("stream_chunk_dir"):
            from ..analysis.graph_metrics import (
                strip_graph_analysis_payload,
                write_graph_analysis_entry_chunks,
            )

            chunk_summary = write_graph_analysis_entry_chunks(
                entry,
                Path(payload.get("stream_chunk_dir")),
                metrics=metrics_cfg,
            )
            entry = strip_graph_analysis_payload(entry, chunk_summary)
        return {
            "ok": True,
            "box": int(payload["box_id"]),
            "entry": json_sanitize(entry),
            "box_cutoffs": _cutoff_mapping_to_list(box_cutoffs),
            "chunk_summary": chunk_summary,
        }
    except Exception as exc:
        try:
            failed_box = _strict_analysis_box_id(
                payload.get("box_id", -1),
                context="analysis worker box_id",
                minimum=-1,
            )
        except ValueError:
            failed_box = -1
        return {
            "ok": False,
            "box": failed_box,
            "error": str(exc),
            "analysis_source_role": payload.get("analysis_source_role"),
            "paths": {
                "box_dir": payload.get("box_dir"),
                "relax_dir": payload.get("relax_dir_orig"),
                "analysis_artifact_dir": payload.get("relax_stage_dir"),
                "analysis_source": payload.get("analysis_source_path"),
            },
        }


def _analysis_parallel_worker_count(ctx: AnalysisContext) -> int:
    try:
        workers = int(getattr(ctx, "analysis_workers", 1) or 1)
    except Exception:
        workers = 1
    if workers < 1:
        workers = 1
    cpu = os.cpu_count() or 1
    return int(max(1, min(workers, cpu)))


def _analysis_streaming_enabled(ctx: AnalysisContext) -> bool:
    return bool(getattr(ctx, "analysis_streaming", True))


_STRUCTURE_HASH_FIELDS = ("structure_hash", "cell_hash", "positions_hash", "symbols_hash")


def _existing_structure_manifest(entry: Mapping[str, Any]) -> dict[str, Any]:
    manifest = entry.get("structure_manifest", {})
    if isinstance(manifest, Mapping) and manifest:
        return dict(manifest)
    graph_payload = entry.get("graph_analysis", {})
    if isinstance(graph_payload, Mapping):
        manifest = graph_payload.get("structure_manifest", {})
        if isinstance(manifest, Mapping) and manifest:
            return dict(manifest)
    return {}


def _structure_source_value(entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> Any:
    if manifest.get("source_path") not in (None, ""):
        return manifest.get("source_path")
    structure = entry.get("structure", {})
    if isinstance(structure, Mapping) and structure.get("source_path") not in (None, ""):
        return structure.get("source_path")
    paths = entry.get("paths", {})
    if isinstance(paths, Mapping):
        return paths.get("analysis_source") or paths.get("relax_data")
    return None


def _manifest_hash_values(entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    structure = entry.get("structure", {})
    structure_map = structure if isinstance(structure, Mapping) else {}
    return {
        key: (
            manifest.get(key)
            if manifest.get(key) not in (None, "")
            else (structure_map.get(key) if structure_map.get(key) not in (None, "") else entry.get(key))
        )
        for key in _STRUCTURE_HASH_FIELDS
    }


def _structure_source_candidates(value: Any, *, outdir: Path, source_base: Optional[Path]) -> list[Path]:
    if value in (None, ""):
        return []
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return [raw.resolve(strict=False)]
    bases: list[Path] = []
    if source_base is not None:
        base = Path(source_base)
        bases.append(base if base.is_dir() else base.parent)
    bases.append(Path(outdir))
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        candidate = (base / raw).resolve(strict=False)
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _sidecar_integrity_record(outdir: Path, relative_path: str) -> dict[str, Any]:
    """Return a content-verifiable sidecar record without escaping ``outdir``."""

    rel = str(relative_path)
    root = Path(outdir).resolve(strict=False)
    candidate = (root / rel).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except Exception:
        return {
            "path": rel,
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "schema": None,
            "status": "invalid_path_outside_output_directory",
            "valid": False,
        }
    if not candidate.is_file():
        return {
            "path": rel,
            "exists": False,
            "size_bytes": None,
            "sha256": None,
            "schema": None,
            "status": "missing",
            "valid": False,
        }
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    schema = None
    valid = True
    status = "verified"
    if candidate.suffix.lower() == ".json":
        try:
            payload = json.loads(candidate.read_text(errors="strict"))
            if isinstance(payload, Mapping):
                schema = payload.get("schema")
        except Exception:
            valid = False
            status = "invalid_json"
    return {
        "path": rel,
        "exists": True,
        "size_bytes": int(candidate.stat().st_size),
        "sha256": digest.hexdigest(),
        "schema": schema,
        "status": status,
        "valid": bool(valid),
    }


def _write_structure_provenance_sidecars(
    outdir: Path,
    *,
    boxes: Sequence[Mapping[str, Any]],
    rejected_boxes: Sequence[Mapping[str, Any]],
    type_to_species: Optional[Sequence[str]],
    atom_style: str,
    lammps_units_style: Optional[str] = None,
    source_base: Optional[Path] = None,
) -> dict[str, Any]:
    """Materialize and verify structure provenance independently of graph analysis.

    Enhanced graph descriptors remain opt-in.  The structure manifest is a
    baseline audit artifact and is therefore written for every analysis run,
    including ``embed_structures=false`` with no graph rules.
    """

    from ..analysis.graph import manifest_row_from_frame, source_file_identity

    root = Path(outdir)
    refs_dir = root / "structure_references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    reference_index: list[dict[str, Any]] = []
    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    reference_by_key: dict[tuple[str, int], str] = {}

    records: list[tuple[str, Mapping[str, Any]]] = [
        *(('accepted', entry) for entry in boxes if isinstance(entry, Mapping)),
        *(('rejected', entry) for entry in rejected_boxes if isinstance(entry, Mapping)),
    ]
    used_names: set[str] = set()
    for ordinal, (record_role, entry) in enumerate(records, start=1):
        raw_box_id = (
            entry.get("box")
            if "box" in entry
            else entry.get("box_id", ordinal)
        )
        box_id = _strict_analysis_box_id(
            raw_box_id,
            context=f"{record_role} structure-provenance box id",
            minimum=0,
        )
        key = (str(record_role), int(box_id))
        existing = _existing_structure_manifest(entry)
        expected_hashes = _manifest_hash_values(entry, existing)
        source_value = _structure_source_value(entry, existing)
        source_candidates = _structure_source_candidates(
            source_value,
            outdir=root,
            source_base=source_base,
        )
        structure_payload = entry.get("structure", {})
        nested_identity = (
            structure_payload.get("source_file_identity", {})
            if isinstance(structure_payload, Mapping)
            else {}
        )
        prior_identity = existing.get("source_file_identity", {})
        if not isinstance(prior_identity, Mapping) or prior_identity.get("sha256") in (None, ""):
            prior_identity = nested_identity if isinstance(nested_identity, Mapping) else {}
        source_path = next((candidate for candidate in source_candidates if candidate.is_file()), None)
        if prior_identity.get("sha256") not in (None, ""):
            for candidate in source_candidates:
                candidate_identity = source_file_identity(candidate)
                if (
                    str(candidate_identity.get("sha256")) == str(prior_identity.get("sha256"))
                    and int(candidate_identity.get("size_bytes", -1)) == int(prior_identity.get("size_bytes", -2))
                ):
                    source_path = candidate
                    break
        row: dict[str, Any]
        verification_error: Optional[str] = None

        if source_path is not None and source_path.is_file():
            try:
                frames = read_last_frames_auto(
                    source_path,
                    1,
                    type_to_species=type_to_species,
                    atom_style=str(atom_style),
                    units_style=lammps_units_style,
                )
                if not frames:
                    raise ValueError("source reader returned no frames")
                row = dict(
                    manifest_row_from_frame(
                        frames[-1],
                        box_id=int(box_id),
                        source_path=source_path,
                        source_role=(
                            existing.get("source_role")
                            or (entry.get("analysis_source_role") if isinstance(entry, Mapping) else None)
                            or record_role
                        ),
                        type_to_species=type_to_species,
                        density=entry.get("density"),
                    )
                )
                for hash_name, expected in expected_hashes.items():
                    if expected not in (None, "") and str(expected) != str(row.get(hash_name)):
                        raise ValueError(
                            f"stored {hash_name} does not match reloaded source for box {box_id}: "
                            f"expected={expected} actual={row.get(hash_name)}"
                        )
                if isinstance(prior_identity, Mapping) and prior_identity.get("sha256") not in (None, ""):
                    current_identity = row.get("source_file_identity", {})
                    if (
                        str(prior_identity.get("sha256")) != str(current_identity.get("sha256"))
                        or int(prior_identity.get("size_bytes", -1)) != int(current_identity.get("size_bytes", -2))
                    ):
                        raise ValueError(f"source artifact identity changed for box {box_id}: {source_path}")
                # Keep the originally recorded path spelling (normally
                # relative to the analysis directory) after verification so a
                # moved output tree remains reloadable.  The content identity
                # was computed from the resolved path above.
                if source_value not in (None, ""):
                    row["source_path"] = str(source_value)
                    current_identity = row.get("source_file_identity", {})
                    if isinstance(current_identity, Mapping):
                        row["source_file_identity"] = {
                            **dict(current_identity),
                            "path": str(source_value),
                        }
                pbc_source_verified = not str(row.get("pbc_provenance", "")).startswith(
                    "vitriflow_periodic_lammps_cell_contract"
                )
                verification_status = (
                    "verified_from_source"
                    if pbc_source_verified
                    else "verified_with_declared_pbc_assumption"
                )
                if not pbc_source_verified:
                    verification_error = (
                        "LAMMPS data files do not encode boundary flags; effective pbc=[true,true,true] "
                        "comes from VitriFlow's periodic LAMMPS workflow contract, not from the source file"
                    )
            except Exception as exc:
                # A present source that cannot reproduce its recorded identity
                # is a hard provenance failure, not an advisory condition.
                raise ValueError(
                    f"could not verify structure provenance for {record_role} box {box_id} from {source_path}: {exc}"
                ) from exc
        else:
            row = dict(existing)
            row.setdefault("schema", "vitriflow.structure_manifest.row.v2")
            row["box_id"] = int(box_id)
            row.setdefault("source_path", None if source_value in (None, "") else str(source_value))
            row.setdefault("source_role", record_role)
            for hash_name, value in expected_hashes.items():
                row.setdefault(hash_name, value)
            row.setdefault("pbc", None)
            current_source_identity = source_file_identity(
                source_path if source_path is not None else (source_candidates[0] if source_candidates else None)
            )
            if isinstance(prior_identity, Mapping) and prior_identity.get("sha256") not in (None, ""):
                # Preserve the recorded identity as historical evidence; keep
                # the failed current lookup separate so absence is not confused
                # with "this source never had an identity".
                row["source_file_identity"] = dict(prior_identity)
                row["source_file_verification"] = current_source_identity
            else:
                row["source_file_identity"] = current_source_identity
            complete_hashes = all(row.get(name) not in (None, "") for name in _STRUCTURE_HASH_FIELDS)
            if complete_hashes and row.get("pbc") is not None:
                verification_status = "hash_locked_source_unavailable"
                verification_error = "source artifact was unavailable; recorded hashes could not be re-derived"
            else:
                verification_status = "provenance_incomplete"
                verification_error = "source artifact and/or complete structure hashes were unavailable"

        row["record_role"] = str(record_role)
        row["verification"] = {
            "status": verification_status,
            "verified": bool(verification_status == "verified_from_source"),
            "error": verification_error,
            "source_artifact_verified": bool(
                verification_status in {"verified_from_source", "verified_with_declared_pbc_assumption"}
            ),
            "structure_hashes_verified": bool(
                verification_status in {"verified_from_source", "verified_with_declared_pbc_assumption"}
            ),
            "pbc_source_verified": bool(verification_status == "verified_from_source"),
        }

        base_name = f"box_{int(box_id):06d}.json"
        if base_name in used_names:
            base_name = f"box_{int(box_id):06d}_{record_role}.json"
        used_names.add(base_name)
        ref_rel = str(Path("structure_references") / base_name)
        reference_payload = {
            "schema": "vitriflow.structure_reference.v1",
            "box_id": int(box_id),
            "record_role": str(record_role),
            "status": verification_status,
            "verified": bool(verification_status == "verified_from_source"),
            "manifest_sidecar": "structure_manifest.json",
            "manifest_row": int(len(manifest_rows)),
            "hashes": {name: row.get(name) for name in _STRUCTURE_HASH_FIELDS},
            "pbc": row.get("pbc"),
            "n_atoms": row.get("n_atoms", (entry.get("structure", {}) or {}).get("n_atoms") if isinstance(entry.get("structure", {}), Mapping) else None),
            "source": {
                "path": row.get("source_path"),
                "role": row.get("source_role"),
                "file_identity": row.get("source_file_identity"),
            },
            "verification": dict(row.get("verification", {})),
        }
        atomic_write_json(root / ref_rel, json_sanitize(reference_payload))
        row["structure_reference"] = ref_rel
        manifest_rows.append(json_sanitize(row))
        manifest_by_key[key] = row
        reference_by_key[key] = ref_rel
        reference_index.append(
            {
                "box_id": int(box_id),
                "record_role": str(record_role),
                "path": ref_rel,
                "status": verification_status,
                "structure_hash": row.get("structure_hash"),
            }
        )

    manifest_payload = {
        "schema": "vitriflow.structure_manifest.v2",
        "structures": manifest_rows,
        "n_structures": int(len(manifest_rows)),
        "graph_analysis_required": False,
    }
    references_payload = {
        "schema": "vitriflow.structure_references.v1",
        "references": reference_index,
        "n_references": int(len(reference_index)),
    }
    atomic_write_json(root / "structure_manifest.json", json_sanitize(manifest_payload))
    atomic_write_json(root / "structure_references.json", json_sanitize(references_payload))

    accepted_rows = [row for row in manifest_rows if str(row.get("record_role")) == "accepted"]
    all_hash_locked = bool(accepted_rows) and all(
        all(row.get(name) not in (None, "") for name in _STRUCTURE_HASH_FIELDS) and row.get("pbc") is not None
        for row in accepted_rows
    )
    all_verified = bool(accepted_rows) and all(
        bool((row.get("verification", {}) or {}).get("verified", False)) for row in accepted_rows
    )
    all_source_artifacts_verified = bool(accepted_rows) and all(
        bool((row.get("verification", {}) or {}).get("source_artifact_verified", False))
        for row in accepted_rows
    )
    return {
        "paths": {
            "structure_manifest": "structure_manifest.json",
            "structure_references": "structure_references.json",
            **{
                f"structure_reference_{str(role)}_{int(box_id)}": path
                for (role, box_id), path in sorted(reference_by_key.items())
            },
        },
        "manifest_by_key": manifest_by_key,
        "reference_by_key": reference_by_key,
        "all_hash_locked": bool(all_hash_locked),
        "all_verified": bool(all_verified),
        "all_source_artifacts_verified": bool(all_source_artifacts_verified),
        "n_structures": int(len(manifest_rows)),
    }


def _public_analysis_box_entry(
    entry: Mapping[str, Any],
    *,
    embed_structures: bool,
    manifest: Optional[Mapping[str, Any]] = None,
    structure_reference: Optional[str] = None,
) -> dict[str, Any]:
    """Return a public JSON-safe box entry honoring structure embedding policy.

    Descriptor sidecar generation needs full structures internally.  Public
    analysis JSON may omit them to keep release outputs lightweight while
    retaining manifest locks and hashes for reproducible reload/audit.
    """

    out = json_sanitize(dict(entry))
    manifest_map = dict(manifest or _existing_structure_manifest(out))
    hashes = _manifest_hash_values(out, manifest_map)
    if bool(embed_structures):
        out["structure_embedding"] = {
            "embed_structures": True,
            "status": "embedded",
            "fields": [name for name in ("structure", "lattice") if name in out],
            **hashes,
            "pbc": manifest_map.get("pbc"),
            "manifest_sidecar": "structure_manifest.json",
            "structure_reference": structure_reference,
        }
        return out

    source_path = manifest_map.get("source_path") or out.get("analysis_source")
    if source_path is None:
        paths = out.get("paths", {}) if isinstance(out.get("paths", {}), Mapping) else {}
        source_path = paths.get("analysis_source") or paths.get("relax_data")
    for key in ("structure", "lattice", "graph_analysis"):
        out.pop(key, None)
    complete = all(hashes.get(name) not in (None, "") for name in _STRUCTURE_HASH_FIELDS)
    verification = manifest_map.get("verification", {}) if isinstance(manifest_map.get("verification", {}), Mapping) else {}
    out["structure_embedding"] = {
        "embed_structures": False,
        "status": "referenced" if complete and structure_reference else "provenance_incomplete",
        "reason": "analysis.embed_structures=false",
        **hashes,
        "pbc": manifest_map.get("pbc"),
        "source_path": source_path,
        "manifest_sidecar": "structure_manifest.json",
        "structure_reference": structure_reference,
        "verification": dict(verification),
        "note": (
            "Full coordinates were omitted; the verified manifest row and structure-reference sidecar lock the analysed source."
            if bool(verification.get("verified", False))
            else (
                "Full coordinates were omitted; source bytes and structure hashes were verified, but LAMMPS data does not encode boundary flags and the recorded fully periodic PBC follows the declared VitriFlow workflow contract."
                if str(verification.get("status", "")) == "verified_with_declared_pbc_assumption"
                else "Full coordinates were omitted; consult verification status before treating the recorded hashes as source-verified."
            )
        ),
    }
    return json_sanitize(out)

@locked_output_workflow("output analysis workflow")
def analyze_output_data(
    *,
    config: Optional[RunConfig] = None,
    input_path: Path,
    outdir: Path,
    plan: Optional[Mapping[str, Any]] = None,
    analysis_context: Optional[AnalysisContext] = None,
    progress: Optional[CondensedProgressLog] = None,
    graph_rules_override: Optional[Sequence[Mapping[str, Any]]] = None,
    embed_structures: Optional[bool] = None,
    embed_structures_override: Optional[bool] = None,
    analysis_workers_override: Optional[int] = None,
    analysis_streaming_override: Optional[bool] = None,
    analysis_max_in_flight_override: Optional[int] = None,
) -> dict[str, Any]:
    """Output data."""

    outdir = Path(outdir)
    ensure_dir(outdir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    replay_contract = _embedded_production_replay_contract(Path(input_path))
    embedded_plan = replay_contract.get("plan", {})
    custom_schedule_config_matches = _custom_schedule_replay_config_matches(
        config,
        replay_contract,
    )
    plan_was_auto_loaded = False
    if plan is None and isinstance(embedded_plan, Mapping) and embedded_plan:
        plan = dict(embedded_plan)
        plan_was_auto_loaded = True

    if plan is not None:
        ctx = _analysis_context_from_plan(config, plan)
    elif analysis_context is not None:
        ctx = analysis_context
    elif config is not None:
        ctx = _analysis_context_from_config(config)
    else:
        raise ValueError("analyze_output_data requires either a full RunConfig or a standalone analysis context")

    if embed_structures_override is not None:
        embed_structures = bool(embed_structures_override)

    if embed_structures is not None:
        ctx = AnalysisContext(
            metrics_cfg=ctx.metrics_cfg,
            type_to_species=ctx.type_to_species,
            prod_cfg=ctx.prod_cfg,
            conv_cfg=ctx.conv_cfg,
            md_timestep=ctx.md_timestep,
            atom_style=ctx.atom_style,
            cutoffs=dict(ctx.cutoffs),
            metric_warnings=list(ctx.metric_warnings),
            effective_metrics=dict(ctx.effective_metrics),
            quench_window_steps_range=ctx.quench_window_steps_range,
            sampling_hint=(None if ctx.sampling_hint is None else dict(ctx.sampling_hint)),
            source_selection=(None if ctx.source_selection is None else dict(ctx.source_selection)),
            embed_structures=bool(embed_structures),
            filter_decision_mode=str(ctx.filter_decision_mode),
            analysis_workers=int(getattr(ctx, "analysis_workers", 1) or 1),
            analysis_streaming=bool(getattr(ctx, "analysis_streaming", True)),
            analysis_max_in_flight=getattr(ctx, "analysis_max_in_flight", None),
            lammps_units_style=ctx.lammps_units_style,
            engine=str(getattr(ctx, "engine", "lammps") or "lammps"),
            exact_plan_cutoffs=bool(getattr(ctx, "exact_plan_cutoffs", False)),
        )

    if analysis_workers_override is not None or analysis_streaming_override is not None or analysis_max_in_flight_override is not None:
        ctx = AnalysisContext(
            metrics_cfg=ctx.metrics_cfg,
            type_to_species=ctx.type_to_species,
            prod_cfg=ctx.prod_cfg,
            conv_cfg=ctx.conv_cfg,
            md_timestep=ctx.md_timestep,
            atom_style=ctx.atom_style,
            cutoffs=dict(ctx.cutoffs),
            metric_warnings=list(ctx.metric_warnings),
            effective_metrics=dict(ctx.effective_metrics),
            quench_window_steps_range=ctx.quench_window_steps_range,
            sampling_hint=(None if ctx.sampling_hint is None else dict(ctx.sampling_hint)),
            source_selection=(None if ctx.source_selection is None else dict(ctx.source_selection)),
            embed_structures=bool(ctx.embed_structures),
            filter_decision_mode=str(ctx.filter_decision_mode),
            analysis_workers=int(analysis_workers_override if analysis_workers_override is not None else getattr(ctx, "analysis_workers", 1)),
            analysis_streaming=bool(analysis_streaming_override if analysis_streaming_override is not None else getattr(ctx, "analysis_streaming", True)),
            analysis_max_in_flight=(analysis_max_in_flight_override if analysis_max_in_flight_override is not None else getattr(ctx, "analysis_max_in_flight", None)),
            lammps_units_style=ctx.lammps_units_style,
            engine=str(getattr(ctx, "engine", "lammps") or "lammps"),
            exact_plan_cutoffs=bool(getattr(ctx, "exact_plan_cutoffs", False)),
        )

    raw_boxes, preset_entries, preset_rejected, dataset_meta = discover_output_dataset(
        Path(input_path),
        source_selection=ctx.source_selection,
    )
    dft_refinement_enabled = bool(
        getattr(getattr(ctx.prod_cfg, "dft_opt", None), "enabled", False)
    )
    refinement_rejected: list[dict[str, Any]] = []
    if dft_refinement_enabled:
        required_refined_ids: Optional[Sequence[int]] = None
        if replay_contract:
            required_refined_ids = list(
                replay_contract.get("accepted_dft_box_ids", []) or []
            )
        raw_boxes, refinement_rejected = _select_dft_refined_sources(
            raw_boxes,
            required_box_ids=required_refined_ids,
        )
        dataset_meta = {
            **dict(dataset_meta),
            "refinement_source": "cp2k_dft_opt_final",
            "refinement_required_box_ids": (
                None
                if required_refined_ids is None
                else [int(value) for value in required_refined_ids]
            ),
        }
    replay_fingerprint = replay_contract.get("resume_fingerprint", {})
    protected_custom_schedule = bool(
        str(replay_contract.get("workflow", "")) == "custom_stage_schedule"
        and isinstance(replay_fingerprint, Mapping)
        and _self_hashed_json_fingerprint_valid(
            replay_fingerprint,
            workflow="custom_stage_schedule",
        )
    )
    _require_unique_dataset_box_ids(
        raw_boxes,
        preset_entries,
        preset_rejected,
        minimum_box_id=(0 if protected_custom_schedule else 1),
    )
    progress.info("analysis", f"discovered {len(raw_boxes)} raw boxes and {len(preset_entries)} pre-analysed task results")

    if graph_rules_override is not None:
        metrics_cfg = _merge_analysis_graph_rules(ctx.metrics_cfg, graph_rules_override)
        effective_metrics = dict(ctx.effective_metrics)
        effective_metrics["graph_rules"] = [
            rule.model_dump(mode="json") if hasattr(rule, "model_dump") else dict(rule)
            for rule in list(metrics_cfg.graph_rules or [])
        ]
        effective_metrics["graph_rules_cli_additive"] = True
        # Cutoff resolution consumes the AnalysisContext, so install the
        # effective merged metrics there before deriving required pairs or
        # resolving any adaptive/legacy cutoff policy.
        ctx = replace(
            ctx,
            metrics_cfg=metrics_cfg,
            effective_metrics=effective_metrics,
        )

    metrics_cfg = ctx.metrics_cfg
    type_to_species = ctx.type_to_species
    prod_cfg = ctx.prod_cfg
    conv_cfg = ctx.conv_cfg
    graph_requested = graph_analysis_requested(metrics_cfg)

    required_pairs = required_pairs_from_metrics(metrics_cfg, type_to_species=type_to_species)
    fixed_cutoffs = fixed_cutoffs_from_metrics(metrics_cfg, type_to_species=type_to_species)
    prod_cutoffs, cutoff_provenance = _resolve_output_analysis_cutoffs(
        raw_boxes=raw_boxes,
        ctx=ctx,
        required_pairs=required_pairs,
        fixed_cutoffs=fixed_cutoffs,
    )
    progress.info("analysis", f"cutoff mode: {cutoff_provenance.get('mode', 'unknown')}")

    # analysis dataset discovery
    # ase module import
    from . import production_common as _production_common_module
    from .production_common import (
        analyse_production_box,
        build_production_convergence_spec,
        check_production_convergence,
        metrics_checked_from_conv_spec,
        resolve_production_time_unit_ps,
        resolve_production_warmup_duration_ps,
        resolve_production_warmup_steps,
        validate_production_entry_against_spec,
        write_graph_analysis_outputs,
    )
    fixed_count_posthoc_assessor = getattr(
        _production_common_module,
        "assess_fixed_count_convergence_posthoc",
        None,
    )
    convergence_comparator = getattr(
        _production_common_module,
        "compare_convergence_assessments",
        None,
    )

    def _build_convergence_spec(entry: Mapping[str, Any]) -> dict[str, Any]:
        parameters = inspect.signature(build_production_convergence_spec).parameters
        if "metrics_cfg" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters.values()
        ):
            return dict(build_production_convergence_spec(entry, metrics_cfg))
        return dict(build_production_convergence_spec(entry))

    def _fixed_count_posthoc_report(
        rows: Sequence[Mapping[str, Any]],
        spec: Optional[Mapping[str, Any]],
        *,
        execution_target_met: bool,
        min_boxes: int,
    ) -> dict[str, Any]:
        if callable(fixed_count_posthoc_assessor):
            return dict(
                fixed_count_posthoc_assessor(
                    rows,
                    spec,
                    conv_cfg,
                    execution_target_met=bool(execution_target_met),
                    min_boxes=int(min_boxes),
                )
            )

        # Preserve compatibility with integrations that replace
        # production_common with a narrow test/plugin shim.  Real VitriFlow
        # always takes the shared assessor above; this fallback still labels
        # the evidence honestly and never converts it into a stopping result.
        if rows and spec is not None:
            criterion_met, report = check_production_convergence(
                [dict(row) for row in rows],
                dict(spec),
                conv_cfg,
            )
            out = dict(report)
            assessment_performed = True
            status_posthoc = "fixed_n_terminal_posthoc_assessed"
        else:
            criterion_met = None
            assessment_performed = False
            status_posthoc = "fixed_n_terminal_posthoc_unassessed"
            out = _analysis_convergence_unavailable_report(
                boxes=rows,
                conv_spec=spec,
                conv_cfg=conv_cfg,
                reason="fixed-count terminal post-hoc evidence is unavailable",
                check_enabled=False,
                prod_cfg=prod_cfg,
            )
        out.update(
            {
                "status": status_posthoc,
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
                "posthoc_failed_items": [],
                "execution_target_met": bool(execution_target_met),
                "n_boxes_accepted": int(len(rows)),
                "min_boxes": int(min_boxes),
            }
        )
        contract = dict(out.get("inference_contract", {}) or {})
        contract.update(
            {
                "assessment_design": "fixed_n_terminal_posthoc",
                "sequentially_valid": False,
                "optional_stopping_coverage_guaranteed": False,
                "used_for_stopping": False,
            }
        )
        out["inference_contract"] = contract
        return out
    from ..analysis.motif_summary import summarize_production_crystal_motifs

    boxes: list[dict[str, Any]] = []
    rejected_boxes: list[dict[str, Any]] = [
        *list(preset_rejected),
        *list(refinement_rejected),
    ]
    advisory_rejected_boxes: list[dict[str, Any]] = []
    conv_spec: Optional[dict[str, Any]] = None
    source_role_counts = _analysis_source_role_counts(raw_boxes)
    def _stage_dir_for_analysis_artifacts(box: DiscoveredBox) -> Path:
        # analyze-output is a read-only consumer of its input dataset. Every
        # generated per-box artifact belongs under this analysis outdir even
        # when the source resembles an existing production relax directory.
        return outdir / "box_artifacts" / f"box_{int(box.box):03d}"

    filtering_mode = str(getattr(ctx, "filter_decision_mode", "enforcing") or "enforcing").strip().lower()
    if filtering_mode not in {"enforcing", "advisory"}:
        filtering_mode = "enforcing"

    def _consume_entry(entry: dict[str, Any]) -> None:
        nonlocal conv_spec
        if conv_spec is None:
            conv_spec = _build_convergence_spec(entry)
        else:
            validate_production_entry_against_spec(entry, conv_spec, box_label=entry.get("box", "?"))
        if bool(entry.get("reject")):
            if filtering_mode == "advisory":
                advisory = dict(entry.get("reject") or {}) if isinstance(entry.get("reject"), Mapping) else {"reason": str(entry.get("reject"))}
                advisory.setdefault(
                    "box",
                    _strict_analysis_box_id(
                        entry.get("box", entry.get("box_id", -1)),
                        context="advisory rejection box id",
                        minimum=0,
                    ),
                )
                advisory.setdefault("mode", "analysis_only_would_reject")
                advisory_rejected_boxes.append(json_sanitize(advisory))
                entry["reject_advisory"] = json_sanitize(entry.get("reject"))
                entry["reject"] = False
                boxes.append(entry)
            else:
                rejected_boxes.append(entry)
        else:
            boxes.append(entry)

    for entry in sorted(
        preset_entries,
        key=lambda x: _strict_analysis_box_id(
            x.get("box", x.get("box_id", -1)),
            context="pre-analysed task box id",
            minimum=0,
        ),
    ):
        prepared_entry = dict(entry)
        diagnostic_task_result = prepared_entry.pop(
            "_task_result_path_for_diagnostic_rebase", None
        )
        if diagnostic_task_result not in (None, ""):
            reused = _validated_task_diagnostics(
                Path(str(diagnostic_task_result)),
                result_base=outdir,
            )
            if reused is None:
                raise ValueError(
                    "A pre-analysed task entry lost its persisted diagnostic contract"
                )
            prepared_entry["stage_metrics"] = reused.get("stage_metrics")
            prepared_entry["elastic_melt"] = reused.get("elastic_melt")
            prepared_entry["elastic_relax"] = reused.get("elastic_relax")
            prepared_entry["elastic_timeseries"] = reused.get("elastic_timeseries")
            prepared_entry["task_diagnostics_provenance"] = dict(
                reused.get("provenance", {}) or {}
            )
        _consume_entry(prepared_entry)

    stream_graph_outputs = bool(graph_requested and _analysis_streaming_enabled(ctx))
    stream_chunk_dir = outdir / ".analysis_stream_chunks"
    if stream_graph_outputs:
        # Streaming chunks are deterministic scratch files, not resumable output.
        # Remove stale 0.4.29.16-era chunks up front so an interrupted run cannot
        # mix compact and oversized payloads or leave hundreds of GB on disk.
        if stream_chunk_dir.exists():
            shutil.rmtree(stream_chunk_dir, ignore_errors=True)
        ensure_dir(stream_chunk_dir)
    analysis_workers = _analysis_parallel_worker_count(ctx)
    adaptive_graph_only = str(cutoff_provenance.get("mode", "")) == "adaptive_graph_rules_only"
    per_box_auto = str(cutoff_provenance.get("mode", "")) == "per_box_auto"
    parallel_safe = bool(adaptive_graph_only or per_box_auto or prod_cutoffs or not required_pairs)
    if analysis_workers > 1 and not parallel_safe:
        progress.warn(
            "analysis",
            "analysis_workers>1 requested but legacy auto-cutoff sharing requires serial analysis; using one worker",
        )
        analysis_workers = 1
    progress.info(
        "analysis",
        f"analysis workers={int(analysis_workers)} streaming_graph_sidecars={bool(stream_graph_outputs)}",
    )

    def _prepare_task_for_box(box: DiscoveredBox, cutoffs_snapshot: Mapping[Tuple[int, int], float]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        source = box.analysis_source or box.relax_traj or box.relax_data
        relax_data = box.relax_data or source
        relax_traj = box.relax_traj or source
        if str(box.source_layout or "") == "ase_database":
            try:
                materialized_source = _materialize_ase_database_box_source(box, outdir=outdir)
            except Exception as exc:
                return None, {
                    "box": int(box.box),
                    "reason": "ase_database_materialization_failed",
                    "error": str(exc),
                    "analysis_source_role": box.analysis_source_role,
                    "source_record": (None if box.source_record is None else dict(box.source_record)),
                    "paths": {
                        "database": _relpath_or_str(source, outdir),
                        "box_dir": _relpath_or_str(box.box_dir, outdir),
                    },
                }
            if materialized_source is not None:
                source = materialized_source
                relax_data = materialized_source
                relax_traj = materialized_source
        density_mean = box.density
        density_stderr = box.density_stderr
        if density_mean is None and source is not None:
            density_mean = _estimate_density_from_source(
                source,
                type_to_species=type_to_species,
                atom_style=str(ctx.atom_style),
                lammps_units_style=ctx.lammps_units_style,
            )
            if density_mean is not None and density_stderr is None:
                density_stderr = 0.0
        if relax_data is None:
            return None, {
                "box": int(box.box),
                "reason": "missing_relax_data",
                "paths": {
                    "box_dir": _relpath_or_str(box.box_dir, outdir),
                    "analysis_source": _relpath_or_str(source, outdir),
                },
            }
        if relax_traj is None:
            return None, {
                "box": int(box.box),
                "reason": "missing_relax_trajectory",
                "paths": {
                    "box_dir": _relpath_or_str(box.box_dir, outdir),
                    "analysis_source": _relpath_or_str(source, outdir),
                },
            }
        reused_task_diagnostics = _validated_task_diagnostics(
            box.task_result,
            result_base=outdir,
        )
        analysis_stage_dir = _stage_dir_for_analysis_artifacts(box)
        ensure_dir(analysis_stage_dir)
        task = {
            "box_id": int(box.box),
            "outdir": str(outdir),
            "box_dir": str(box.box_dir),
            "relax_dir_orig": str(box.relax_dir),
            "melt_stage_dir": str(box.melt_dir),
            "quench_stage_dir": str(box.quench_dir),
            "relax_stage_dir": str(analysis_stage_dir),
            "relax_data_path": str(relax_data),
            "density_mean": float(density_mean if density_mean is not None else float("nan")),
            "density_stderr": float(density_stderr if density_stderr is not None else (0.0 if density_mean is not None else float("nan"))),
            "metrics_cfg": metrics_cfg.model_dump(mode="python") if hasattr(metrics_cfg, "model_dump") else metrics_cfg,
            "cutoffs": (
                []
                if adaptive_graph_only or per_box_auto
                else _cutoff_mapping_to_list(cutoffs_snapshot)
            ),
            "required_pairs": [[a, b] for a, b in list(required_pairs or [])],
            "fixed_cutoffs": _cutoff_mapping_to_list(fixed_cutoffs),
            "type_to_species": (None if type_to_species is None else list(type_to_species)),
            "md_timestep": float(ctx.md_timestep),
            "quench_window_steps_range": (None if ctx.quench_window_steps_range is None else list(ctx.quench_window_steps_range)),
            "sampling_hint": (None if ctx.sampling_hint is None else dict(ctx.sampling_hint)),
            "bondlen_cdf_points": int(getattr(prod_cfg, "bondlen_cdf_points", 200)),
            "angle_cdf_points": int(getattr(prod_cfg, "angle_cdf_points", 180)),
            "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
            "rejects_dir": str(outdir / str(getattr(prod_cfg, "rejects_subdir", "rejects"))),
            "relax_dump_path": (None if box.relax_dump is None else str(box.relax_dump)),
            "relax_traj_path": (None if relax_traj is None else str(relax_traj)),
            "analysis_source_path": (None if source is None else str(source)),
            "analysis_source_role": box.analysis_source_role,
            "atom_style": str(ctx.atom_style),
            "lammps_units_style": ctx.lammps_units_style,
            "engine": str(getattr(ctx, "engine", "lammps") or "lammps"),
            "embed_structures": bool(ctx.embed_structures),
            "stream_graph_outputs": bool(stream_graph_outputs),
            "stream_chunk_dir": str(stream_chunk_dir),
            "reused_task_diagnostics": reused_task_diagnostics,
            "task_diagnostics_contract": (
                "not_applicable"
                if box.task_result is None
                else (
                    "validated"
                    if reused_task_diagnostics is not None
                    else "legacy_absent"
                )
            ),
            "task_result_path": (
                None if box.task_result is None else str(box.task_result)
            ),
        }
        return task, None

    if analysis_workers > 1:
        tasks: list[dict[str, Any]] = []
        for box in sorted(raw_boxes, key=lambda x: int(x.box)):
            task, rejection = _prepare_task_for_box(box, prod_cutoffs)
            if rejection is not None:
                rejected_boxes.append(rejection)
                progress.info("analysis", f"rejected box {int(box.box)}: {rejection.get('reason', 'rejected')}")
                continue
            if task is not None:
                tasks.append(task)
        max_in_flight_cfg = getattr(ctx, "analysis_max_in_flight", None)
        max_in_flight = int(max_in_flight_cfg) if max_in_flight_cfg is not None else int(max(analysis_workers, analysis_workers * 2))
        max_in_flight = max(1, min(int(max_in_flight), int(len(tasks) or 1)))
        progress.info("analysis", f"submitting {len(tasks)} box analyses with max_in_flight={max_in_flight}")
        results_by_box: dict[int, dict[str, Any]] = {}
        task_iter = iter(tasks)
        with ProcessPoolExecutor(max_workers=int(analysis_workers)) as executor:
            pending: dict[Any, int] = {}
            for _ in range(max_in_flight):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                fut = executor.submit(_analysis_worker_task, task)
                pending[fut] = int(task["box_id"])
            while pending:
                done, _not_done = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    box_id = pending.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = {"ok": False, "box": int(box_id), "error": str(exc), "analysis_source_role": None, "paths": {}}
                    results_by_box[int(box_id)] = dict(res)
                    if bool(res.get("ok", False)):
                        progress.info("analysis", f"analysed box {int(box_id)}")
                    else:
                        progress.info("analysis", f"rejected box {int(box_id)}: analysis_failed")
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        continue
                    nfut = executor.submit(_analysis_worker_task, task)
                    pending[nfut] = int(task["box_id"])
        for box_id in sorted(results_by_box.keys()):
            res = results_by_box[box_id]
            if bool(res.get("ok", False)):
                _consume_entry(dict(res.get("entry", {}) or {}))
            else:
                rejected_boxes.append({
                    "box": int(box_id),
                    "reason": "analysis_failed",
                    "error": str(res.get("error", "unknown analysis failure")),
                    "analysis_source_role": res.get("analysis_source_role"),
                    "paths": dict(res.get("paths", {}) or {}),
                })
    else:
        for box in sorted(raw_boxes, key=lambda x: int(x.box)):
            task, rejection = _prepare_task_for_box(box, prod_cutoffs)
            if rejection is not None:
                rejected_boxes.append(rejection)
                progress.info("analysis", f"rejected box {int(box.box)}: {rejection.get('reason', 'rejected')}")
                continue
            if task is None:
                continue
            res = _analysis_worker_task(task)
            if bool(res.get("ok", False)):
                if not adaptive_graph_only and not per_box_auto:
                    prod_cutoffs = _cutoff_list_to_mapping(res.get("box_cutoffs", []))
                _consume_entry(dict(res.get("entry", {}) or {}))
                progress.info("analysis", f"analysed box {int(box.box)}")
            else:
                rejected_boxes.append({
                    "box": int(box.box),
                    "reason": "analysis_failed",
                    "error": str(res.get("error", "unknown analysis failure")),
                    "analysis_source_role": res.get("analysis_source_role", box.analysis_source_role),
                    "paths": dict(res.get("paths", {}) or {}),
                })
                progress.info("analysis", f"rejected box {int(box.box)}: analysis_failed")

    converged = False
    conv_report: dict[str, Any] = {}
    status = "ok"
    error: Optional[str] = None

    check_convergence_enabled = bool(getattr(prod_cfg, "check_convergence", True))
    convergence_advisory = True
    if not boxes:
        status = "error"
        error = "no accepted boxes available for analysis"
        if not check_convergence_enabled:
            conv_report = _fixed_count_posthoc_report(
                boxes,
                conv_spec,
                execution_target_met=False,
                min_boxes=int(getattr(prod_cfg, "min_boxes", 1) or 1),
            )
        else:
            conv_report = _analysis_convergence_unavailable_report(
                boxes=boxes,
                conv_spec=conv_spec,
                conv_cfg=conv_cfg,
                reason=str(error),
                check_enabled=check_convergence_enabled,
                prod_cfg=prod_cfg,
            )
    elif not check_convergence_enabled:
        status = "ok"
        error = None
        converged = False
        min_boxes_posthoc = int(getattr(prod_cfg, "min_boxes", 1) or 1)
        conv_report = _fixed_count_posthoc_report(
            boxes,
            conv_spec,
            execution_target_met=bool(len(boxes) >= min_boxes_posthoc),
            min_boxes=min_boxes_posthoc,
        )
        conv_report = _annotate_analysis_convergence_report(
            conv_report,
            boxes=boxes,
            conv_spec=conv_spec,
            prod_cfg=prod_cfg,
            advisory=convergence_advisory,
        )
        progress.convergence("analysis-posthoc", conv_report)
    elif conv_spec is None:
        status = "ok"
        error = None
        converged = False
        conv_report = _analysis_convergence_unavailable_report(
            boxes=boxes,
            conv_spec=conv_spec,
            conv_cfg=conv_cfg,
            reason="no convergence specification could be constructed from analysed boxes",
            check_enabled=True,
            prod_cfg=prod_cfg,
        )
        progress.warn("analysis", "convergence specification unavailable; recorded advisory descriptor-set convergence status")
    else:
        try:
            converged, conv_report = check_production_convergence(boxes, conv_spec, conv_cfg)
            conv_report = _annotate_analysis_convergence_report(
                conv_report,
                boxes=boxes,
                conv_spec=conv_spec,
                prod_cfg=prod_cfg,
                advisory=convergence_advisory,
            )
            progress.convergence("analysis", conv_report)
        except Exception as exc:
            status = "ok"
            error = None
            converged = False
            conv_report = _analysis_convergence_unavailable_report(
                boxes=boxes,
                conv_spec=conv_spec,
                conv_cfg=conv_cfg,
                reason=f"convergence analysis failed: {exc}",
                check_enabled=True,
                prod_cfg=prod_cfg,
            )
            progress.warn("analysis", f"convergence analysis failed but descriptor output remains available: {exc}")

    ensemble_cdfs = _build_analysis_ensemble_cdfs(boxes, conv_spec, conv_cfg)
    conv_report = _merge_analysis_ensemble_cdfs_into_convergence(
        conv_report,
        ensemble_cdfs,
        conv_cfg=conv_cfg,
    )

    source_convergence = replay_contract.get("convergence", None)
    embedded_plan_matches = bool(
        isinstance(embedded_plan, Mapping)
        and embedded_plan
        and isinstance(plan, Mapping)
        and json_sanitize(dict(embedded_plan)) == json_sanitize(dict(plan))
    )
    parity_comparable = bool(
        isinstance(source_convergence, Mapping)
        and (
            plan_was_auto_loaded
            or embedded_plan_matches
            or custom_schedule_config_matches
        )
        and not list(graph_rules_override or [])
        and callable(convergence_comparator)
    )
    if parity_comparable:
        convergence_parity = convergence_comparator(
            dict(source_convergence),
            conv_report,
        )
        convergence_parity.update(
            {
                "comparable": True,
                "source_results": replay_contract.get("source_results"),
                "source_role": (
                    "dft_opt_final"
                    if dft_refinement_enabled
                    else "production_relax_ensemble"
                ),
                "replay_basis": (
                    "self_hashed_custom_schedule_config"
                    if custom_schedule_config_matches
                    else "embedded_production_plan"
                ),
            }
        )
        if not bool(convergence_parity.get("equivalent", False)):
            status = "error"
            converged = False
            error = (
                "analyze-output convergence differs from the canonical source "
                "production assessment; see convergence_parity.differences"
            )
            progress.warn("analysis", str(error))
    else:
        convergence_parity = {
            "schema": "vitriflow.convergence_parity.v1",
            "comparable": False,
            "equivalent": None,
            "source_results": replay_contract.get("source_results"),
            "reason": (
                "no embedded source convergence report"
                if not isinstance(source_convergence, Mapping)
                else (
                    "graph-rule overrides change the convergence criterion"
                    if list(graph_rules_override or [])
                    else (
                        "custom-schedule YAML does not match the self-hashed producer analysis configuration"
                        if str(replay_contract.get("workflow", ""))
                        == "custom_stage_schedule"
                        else "analysis plan does not match the embedded production plan"
                    )
                )
            ),
        }

    source_selection_meta = _source_selection_for_metadata(ctx.source_selection)
    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(Path(input_path).resolve() if Path(input_path).exists() else Path(input_path)),
        "boxes": [_dataset_record_for_box(box, base_dir=outdir) for box in sorted(raw_boxes, key=lambda x: int(x.box))],
        "n_preanalysed": int(len(preset_entries)),
        "n_task_failures": int(len(preset_rejected)),
        "metadata": {
            **dict(dataset_meta),
            "analysis_source_roles": dict(source_role_counts),
            **({"source_selection": source_selection_meta} if source_selection_meta is not None else {}),
        },
    }

    warmup_duration_ps = resolve_production_warmup_duration_ps(prod_cfg=prod_cfg)
    warmup_steps = resolve_production_warmup_steps(
        prod_cfg=prod_cfg,
        md_timestep=float(ctx.md_timestep),
        time_unit_ps=resolve_production_time_unit_ps(
            config=config,
            engine=(str(plan.get("engine")) if isinstance(plan, Mapping) and plan.get("engine", None) is not None else None),
            time_unit_ps=(plan.get("time_unit_ps", None) if isinstance(plan, Mapping) else None),
        ),
    )

    # Per-box auto mode intentionally has no ensemble-wide legacy graph map.
    # Each box already carries its own resolved graph/cutoff provenance.
    legacy_common_cutoffs = {} if per_box_auto else prod_cutoffs

    graph_output_paths: dict[str, str] = {}
    if graph_requested:
        if bool(stream_graph_outputs):
            from ..analysis.graph_metrics import finalize_streamed_graph_analysis_outputs

            ensemble_frames: list[Any] = []
            ensemble_box_ids: list[int] = []
            for ent in sorted(
                boxes,
                key=lambda b: _strict_analysis_box_id(
                    (b or {}).get("box", (b or {}).get("box_id", -1)),
                    context="streamed graph-analysis box id",
                    minimum=0,
                ),
            ):
                if not isinstance(ent, Mapping) or bool(ent.get("reject", False)):
                    continue
                manifest = ent.get("structure_manifest", {}) if isinstance(ent.get("structure_manifest", {}), Mapping) else {}
                source_value = manifest.get("source_path")
                if not source_value:
                    paths_map = ent.get("paths", {}) if isinstance(ent.get("paths", {}), Mapping) else {}
                    source_value = paths_map.get("analysis_source") or paths_map.get("relax_data")
                src = _path_from_record(source_value, base_dir=outdir) if source_value else None
                if src is None or not Path(src).exists():
                    continue
                try:
                    frs = read_last_frames_auto(
                        Path(src),
                        1,
                        type_to_species=type_to_species,
                        atom_style=str(ctx.atom_style),
                        units_style=ctx.lammps_units_style,
                    )
                    if frs:
                        ensemble_frames.append(frs[-1])
                        ensemble_box_ids.append(
                            _strict_analysis_box_id(
                                ent.get("box", ent.get("box_id", -1)),
                                context="streamed graph ensemble box id",
                                minimum=0,
                            )
                        )
                except Exception as exc:
                    progress.warn("analysis", f"could not reload final frame for ensemble graph rule pass from {src}: {exc}")
            graph_output_paths = dict(
                finalize_streamed_graph_analysis_outputs(
                    outdir,
                    chunk_dir=stream_chunk_dir,
                    boxes=boxes,
                    rejected_boxes=rejected_boxes,
                    metrics=metrics_cfg,
                    type_to_species=type_to_species,
                    legacy_cutoffs=legacy_common_cutoffs,
                    ensemble_frames=ensemble_frames,
                    ensemble_box_ids=ensemble_box_ids,
                )
            )
        else:
            try:
                graph_output_paths = dict(
                    write_graph_analysis_outputs(
                        outdir,
                        boxes=boxes,
                        rejected_boxes=rejected_boxes,
                        metrics=metrics_cfg,
                        type_to_species=type_to_species,
                        legacy_cutoffs=legacy_common_cutoffs,
                    )
                )
            except TypeError as exc:
                # Preserve the historical three-argument writer hook used by
                # downstream integrations while the real writer accepts the
                # richer ensemble graph-analysis context.
                if "unexpected keyword argument" not in str(exc):
                    raise
                graph_output_paths = dict(
                    write_graph_analysis_outputs(
                        outdir,
                        boxes=boxes,
                        rejected_boxes=rejected_boxes,
                    )
                )

    structure_provenance = _write_structure_provenance_sidecars(
        outdir,
        boxes=boxes,
        rejected_boxes=rejected_boxes,
        type_to_species=type_to_species,
        atom_style=str(ctx.atom_style),
        lammps_units_style=ctx.lammps_units_style,
        source_base=Path(input_path),
    )

    ensemble_cdfs_path = outdir / "ensemble_cdfs.json"
    ensemble_cdf_sidecar = build_ensemble_cdf_sidecar(boxes, conv_spec)
    atomic_write_json(ensemble_cdfs_path, json_sanitize(ensemble_cdf_sidecar))
    general_output_paths: dict[str, str] = {
        "ensemble_cdfs": "ensemble_cdfs.json",
        **dict(structure_provenance.get("paths", {})),
    }

    all_output_paths = {**dict(graph_output_paths), **dict(general_output_paths)}
    sidecar_status = {
        str(k): _sidecar_integrity_record(outdir, str(v))
        for k, v in sorted(all_output_paths.items())
    }
    sidecar_integrity_payload = {
        "schema": "vitriflow.sidecar_integrity.v2",
        "sidecars": dict(sidecar_status),
        "missing": [str(k) for k, v in sorted(sidecar_status.items()) if not bool((v or {}).get("exists", False))],
        "all_present": all(bool((v or {}).get("exists", False)) for v in sidecar_status.values()),
        "all_content_hashed": all(bool((v or {}).get("sha256")) for v in sidecar_status.values()),
        "all_valid": all(bool((v or {}).get("valid", False)) for v in sidecar_status.values()),
        "embed_structures": bool(ctx.embed_structures),
    }
    atomic_write_json(outdir / "sidecar_integrity.json", json_sanitize(sidecar_integrity_payload))
    general_output_paths["sidecar_integrity"] = "sidecar_integrity.json"
    sidecar_status["sidecar_integrity"] = _sidecar_integrity_record(outdir, "sidecar_integrity.json")
    accepted_manifest_by_key = dict(structure_provenance.get("manifest_by_key", {}))
    material_ids = sorted(
        {
            str(row.get("material_id", "unknown"))
            for (role, _box_id), row in accepted_manifest_by_key.items()
            if str(role) == "accepted" and isinstance(row, Mapping)
        }
    )
    filtering_summary = {
        "mode": filtering_mode,
        "semantics": (
            "analysis_only_would_reject_flags_are_advisory"
            if filtering_mode == "advisory"
            else "production_accept_reject_flags_are_enforcing"
        ),
        "n_boxes_would_be_rejected": int(len(advisory_rejected_boxes)),
        "n_boxes_rejected_enforced": int(len([b for b in rejected_boxes if isinstance(b, Mapping) and str(b.get("reason", "")) != "analysis_failed"])),
    }

    manifest_by_key = dict(structure_provenance.get("manifest_by_key", {}))
    reference_by_key = dict(structure_provenance.get("reference_by_key", {}))
    public_boxes = [
        _public_analysis_box_entry(
            b,
            embed_structures=bool(ctx.embed_structures),
            manifest=manifest_by_key.get(
                (
                    "accepted",
                    _strict_analysis_box_id(
                        b.get("box", b.get("box_id", -1)),
                        context="accepted public box id",
                        minimum=0,
                    ),
                )
            ),
            structure_reference=reference_by_key.get(
                (
                    "accepted",
                    _strict_analysis_box_id(
                        b.get("box", b.get("box_id", -1)),
                        context="accepted public box id",
                        minimum=0,
                    ),
                )
            ),
        )
        for b in boxes
    ]
    public_rejected_boxes = [
        _public_analysis_box_entry(
            b,
            embed_structures=bool(ctx.embed_structures),
            manifest=manifest_by_key.get(
                (
                    "rejected",
                    _strict_analysis_box_id(
                        b.get("box", b.get("box_id", -1)),
                        context="rejected public box id",
                        minimum=0,
                    ),
                )
            ),
            structure_reference=reference_by_key.get(
                (
                    "rejected",
                    _strict_analysis_box_id(
                        b.get("box", b.get("box_id", -1)),
                        context="rejected public box id",
                        minimum=0,
                    ),
                )
            ),
        )
        if isinstance(b, Mapping)
        else b
        for b in rejected_boxes
    ]

    diagnostic_summary = {
        "source_integrity": {
            "manifest_locked": bool(structure_provenance.get("all_hash_locked", False)),
            "all_sources_verified": bool(
                structure_provenance.get("all_source_artifacts_verified", False)
            ),
            "all_source_artifacts_verified": bool(
                structure_provenance.get("all_source_artifacts_verified", False)
            ),
            "all_pbc_source_verified": bool(structure_provenance.get("all_verified", False)),
            "n_manifest_structures": int(structure_provenance.get("n_structures", 0) or 0),
            "analysis_source_roles": dict(source_role_counts),
            "embed_structures": bool(ctx.embed_structures),
        },
        "representation_rule": {
            "graph_rule_outputs": dict(graph_output_paths),
            "enhanced_graph_analysis_requested": bool(graph_requested),
        },
        "numerical": {
            "strict_json": True,
            "nonfinite_public_json_policy": "null_plus_status_fields",
        },
        "statistical_convergence": dict(conv_report or {}),
        "convergence_parity": dict(convergence_parity),
        "physical_interpretation": {
            "warnings": list(ctx.metric_warnings),
            "note": "Descriptor values are conditional on their representation rules; graph and void sidecars carry explicit descriptor-map provenance.",
        },
    }

    from .. import __version__ as vitriflow_version

    results = {
        "schema": "vitriflow.analysis_results.v2",
        "schema_version": "2.0",
        "status": str(status),
        "errors": ([] if error is None else [str(error)]),
        "warnings": list(ctx.metric_warnings),
        "error": (None if error is None else str(error)),
        "converged": bool(converged),
        "n_boxes": int(len(boxes)),
        "n_boxes_accepted": int(len(boxes)),
        "n_boxes_rejected": int(len(rejected_boxes)),
        "n_boxes_total": int(len(boxes) + len(rejected_boxes)),
        "n_boxes_would_reject": int(len(advisory_rejected_boxes)),
        "n_boxes_flagged_by_filter": int(len(advisory_rejected_boxes)),
        "check_convergence": bool(check_convergence_enabled),
        "convergence_advisory": bool(convergence_advisory),
        "embed_structures": bool(ctx.embed_structures),
        "structure_embedding": {
            "embed_structures": bool(ctx.embed_structures),
            "status": (
                "embedded"
                if bool(ctx.embed_structures)
                else "manifest_sidecar"
            ),
            "manifest_sidecar": dict(sidecar_status.get("structure_manifest", {})).get("path"),
            "structure_references_sidecar": dict(sidecar_status.get("structure_references", {})).get("path"),
            "all_hash_locked": bool(structure_provenance.get("all_hash_locked", False)),
            "all_sources_verified": bool(
                structure_provenance.get("all_source_artifacts_verified", False)
            ),
            "all_source_artifacts_verified": bool(
                structure_provenance.get("all_source_artifacts_verified", False)
            ),
            "all_pbc_source_verified": bool(structure_provenance.get("all_verified", False)),
            "note": (
                "Full final-frame structures are embedded in boxes[].structure and independently locked by the structure manifest."
                if bool(ctx.embed_structures)
                else "Full structures are omitted; the structure manifest and per-box reference sidecars carry explicit verification status and audit hashes."
            ),
        },
        "filtering": filtering_summary,
        "filtering_summary": filtering_summary,
        "analysis_filter_summary": filtering_summary,
        "n_boxes_would_be_rejected": int(len(advisory_rejected_boxes)),
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "cutoffs": _cutoffs_list_from_dict(prod_cutoffs),
        "cutoff_provenance": cutoff_provenance,
        "convergence_spec": conv_spec,
        "convergence": conv_report,
        "convergence_parity": convergence_parity,
        "crystal_motifs": summarize_production_crystal_motifs(boxes, rejected_boxes=rejected_boxes),
        "metrics_checked": metrics_checked_from_conv_spec(conv_spec),
        "effective_metrics": dict(ctx.effective_metrics),
        "metric_warnings": list(ctx.metric_warnings),
        "analysis_source_roles": dict(source_role_counts),
        "materials": material_ids,
        "ensemble_cdfs": dict(sidecar_status.get("ensemble_cdfs", {})),
        "sidecar_integrity": dict(sidecar_status.get("sidecar_integrity", {})),
        "provenance": {
            "source_space": "structure",
            "descriptor_map_policy": "representation_provenance_required_for_graph_void_ring_and_learned_metrics",
            "sidecars": sidecar_status,
        },
        "software_versions": {
            "vitriflow": str(vitriflow_version),
            "analysis_schema": "vitriflow.analysis_results.v2",
        },
        "diagnostics": diagnostic_summary,
        "graph_outputs": dict(graph_output_paths),
        "sidecars": sidecar_status,
        "boxes": public_boxes,
        "rejected_boxes": public_rejected_boxes,
        "advisory_rejected_boxes": advisory_rejected_boxes,
        "would_reject_boxes": advisory_rejected_boxes,
        "indicative_rejected_boxes": advisory_rejected_boxes,
        "paths": {
            "output_dataset": "output_dataset.json",
            "analysis_results": "analysis_results.json",
            "condensed_log": "condensed.log",
            **dict(graph_output_paths),
            **dict(general_output_paths),
        },
    }
    results.update(
        {
            "structure_manifest": dict(sidecar_status.get("structure_manifest", {})),
            "structure_references": dict(sidecar_status.get("structure_references", {})),
        }
    )
    if graph_requested:
        results.update(
            {
                "representation_rules": dict(sidecar_status.get("representation_rules", {})),
                "graph_rules": dict(sidecar_status.get("graph_rules", {})),
                "metric_results": dict(sidecar_status.get("metric_results", {})),
                "graph_metric_by_rule": dict(sidecar_status.get("graph_metric_by_rule", {})),
                "ensemble_metric_by_rule": dict(sidecar_status.get("ensemble_graph_metric_by_rule", {})),
                "coordination_stability": dict(sidecar_status.get("coordination_stability", {})),
                "shell_separability": dict(sidecar_status.get("shell_separability", {})),
                "representation_uncertainty_summary": dict(sidecar_status.get("representation_uncertainty_summary", {})),
                "graph_uncertainty_summary": dict(sidecar_status.get("graph_uncertainty_summary", {})),
                "void_scaling_summary": dict(sidecar_status.get("void_scaling_summary", {})),
                "legacy_single_rule_summary": dict(sidecar_status.get("legacy_single_cutoff_summary", {})),
            }
        )

    dataset = json_sanitize(dataset)
    results = json_sanitize(results)
    atomic_write_json(outdir / "output_dataset.json", dataset)
    atomic_write_json(outdir / "analysis_results.json", results)
    progress.info("analysis", "wrote analysis_results.json and output_dataset.json")
    return results
