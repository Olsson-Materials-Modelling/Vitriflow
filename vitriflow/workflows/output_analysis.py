from __future__ import annotations

"""Generic analysis for existing melt/quench production outputs.

This module intentionally operates on *existing* output trees. It does not run
elastic screens or stage-timeseries diagnostics; the goal is to compute the same
production-style structural metrics and convergence statistics from arbitrary
output data layouts (local runs, externally executed task batches, manually
assembled box directories, or direct ensembles of amorphous box snapshots from
LAMMPS/CP2K/VASP-style files).
"""

import json
import warnings

import numpy as np
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import re

from ..config import ConvergenceConfig, ProductionEnsembleConfig, RunConfig, StructureMetricsConfig
from ..io.thermo import parse_thermo_csv
from ..analysis.stats import window_mean_stderr
from ..analysis.trajectory import quench_window_steps, read_last_frames_auto
from ..utils import ensure_dir
from .metric_requirements import fixed_cutoffs_from_metrics, required_pairs_from_metrics
from .metrics_policy import resolve_effective_metrics_config
from .progress import CondensedProgressLog, atomic_write_json


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


_BOX_ID_RE = re.compile(r"(?:^|[^A-Za-z0-9])box[_-]*0*([0-9]+)(?=$|[^A-Za-z0-9])", re.IGNORECASE)
_TRAILING_NUMERIC_TOKEN_RE = re.compile(r"(?:^|[^A-Za-z0-9])0*([0-9]+)$")
_NATURAL_TOKEN_RE = re.compile(r"(\d+)")


def _label_for_box_id(name: str) -> str:
    """Return the filename/dirname label used for conservative box-id parsing.

    We intentionally avoid concatenating every digit in a name because chemical
    formulae such as ``Si3N4_001.data`` are common in flat final-structure
    ensembles. Only explicit ``box_###`` labels or a separated trailing numeric
    token are treated as box identifiers.
    """

    raw = str(name)
    p = Path(raw)
    suffix = p.suffix
    if suffix and any(ch.isalpha() for ch in suffix):
        return p.stem
    return p.name


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
    if isinstance(obj, Mapping):
        out: dict[Tuple[int, int], float] = {}
        for k, v in obj.items():
            if isinstance(k, tuple) and len(k) == 2:
                a = int(k[0])
                b = int(k[1])
                out[(min(a, b), max(a, b))] = float(v)
        return out
    if isinstance(obj, list):
        out: dict[Tuple[int, int], float] = {}
        for ent in obj:
            if not isinstance(ent, Mapping):
                continue
            pair = ent.get("pair", None)
            cutoff = ent.get("cutoff", None)
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and cutoff is not None:
                a = int(pair[0])
                b = int(pair[1])
                out[(min(a, b), max(a, b))] = float(cutoff)
        return out
    return {}


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


def _standalone_analysis_metric_data(root: Mapping[str, Any]) -> dict[str, Any]:
    autotune = _nested_mapping(root, "autotune")
    metrics = _nested_mapping(root, "metrics") or _nested_mapping(autotune, "metrics")
    if not metrics:
        raise ValueError(
            "Standalone output analysis requires a metrics block. Use either "
            "analysis.metrics: or top-level metrics:."
        )
    metrics = dict(metrics)
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
    prod.setdefault("enabled", True)
    prod.setdefault("min_boxes", 1)
    prod.setdefault("batch_boxes", 1)
    prod.setdefault("check_convergence", True)
    prod.setdefault("store_distributions", True)
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
    metrics_cfg = StructureMetricsConfig.model_validate(_standalone_analysis_metric_data(root))
    metrics_cfg = _analysis_metrics_config(metrics_cfg)
    prod_cfg = ProductionEnsembleConfig.model_validate(_standalone_analysis_production_data(root))
    conv_cfg = ConvergenceConfig.model_validate(_standalone_analysis_convergence_data(root))
    md_cfg = _standalone_analysis_md_data(root)
    cutoffs = _cutoffs_dict_from_any(root.get("cutoffs", None) or root.get("preferred_cutoffs", None))
    type_to_species = (
        [str(x) for x in metrics_cfg.type_to_species]
        if metrics_cfg.type_to_species is not None
        else None
    )
    return AnalysisContext(
        metrics_cfg=metrics_cfg,
        type_to_species=type_to_species,
        prod_cfg=prod_cfg,
        conv_cfg=conv_cfg,
        md_timestep=float(md_cfg.get("timestep", 1.0)),
        atom_style=str(md_cfg.get("atom_style", "atomic")),
        cutoffs=dict(cutoffs),
        metric_warnings=[],
        effective_metrics={"source": "standalone_analysis_config"},
        quench_window_steps_range=None,
        sampling_hint=None,
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
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


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


def _iter_analysis_source_candidates(directory: Path) -> list[Path]:
    d = Path(directory)
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        p = Path(path)
        if p in seen:
            return
        if _is_analysis_source_candidate(p):
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


_FINAL_NAME_RE = re.compile(r"(?:^|[-_])(?:[0-9]+)?-?1\.restart$", re.IGNORECASE)


def _source_role_score(path: Path, *, role: str) -> int:
    p = Path(path)
    name_low = p.name.lower()
    stem_low = p.stem.lower()
    suffix = p.suffix.lower()
    is_dump = _is_dump_like(p)
    is_restart = bool(suffix == ".restart" or name_low.endswith(".restart") or _FINAL_NAME_RE.search(name_low))
    is_contcar = bool(p.name.upper() == "CONTCAR" or p.name.upper().startswith("CONTCAR."))
    is_poscar = bool(p.name.upper() == "POSCAR" or p.name.upper().startswith("POSCAR."))
    is_xdatcar = bool(p.name.upper() == "XDATCAR" or p.name.upper().startswith("XDATCAR."))

    final_keywords = ("final", "relaxed", "optimized", "optimised", "converged", "last", "endpoint")
    traj_keywords = ("traj", "trajectory", "history", "movie", "path", "dump", "xdatcar")
    input_keywords = ("input", "initial", "seed", "start", "origin", "orig", "source")

    has_final_kw = any(tok in name_low for tok in final_keywords)
    has_traj_kw = any(tok in name_low for tok in traj_keywords)
    has_input_kw = any(tok in name_low for tok in input_keywords)

    if role == "final_structure":
        score = 0
        if _FINAL_NAME_RE.search(name_low):
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
    if suffix == ".restart" or p.name.lower().endswith(".restart"):
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
    if suffix == ".restart" or name_low.endswith(".restart") or _FINAL_NAME_RE.search(name_low):
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


def _resolve_box_sources(relax_dir: Path) -> ResolvedBoxSources:
    d = Path(relax_dir)
    cands = _iter_analysis_source_candidates(d)
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


def _guess_analysis_source(relax_dir: Path) -> Optional[Path]:
    return _resolve_box_sources(relax_dir).analysis_source


def _guess_analysis_source_role(relax_dir: Path) -> Optional[str]:
    return _resolve_box_sources(relax_dir).analysis_source_role


def _guess_input_structure(relax_dir: Path) -> Optional[Path]:
    return _resolve_box_sources(relax_dir).input_structure


def _guess_final_structure(relax_dir: Path) -> Optional[Path]:
    return _resolve_box_sources(relax_dir).final_structure


def _guess_relax_data(relax_dir: Path) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir)
    if resolved.final_structure is not None:
        return resolved.final_structure
    if resolved.analysis_source is not None and not _is_dump_like(resolved.analysis_source):
        return resolved.analysis_source
    d = Path(relax_dir)
    for cand in (d / "relax.data", d / "output.data"):
        if cand.exists():
            return cand
    return next((Path(c) for c in resolved.candidates if not _is_dump_like(c)), None)


def _guess_relax_dump(relax_dir: Path) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir)
    if resolved.relax_trajectory is not None and _is_dump_like(resolved.relax_trajectory):
        return resolved.relax_trajectory
    d = Path(relax_dir)
    for cand in (d / "relax.lammpstrj", d / "traj.lammpstrj"):
        if cand.exists():
            return cand
    dumps = sorted(list(d.glob("*.lammpstrj")) + list(d.glob("*.dump")) + list(d.glob("*.trj")))
    return dumps[0] if dumps else None


def _guess_relax_traj(relax_dir: Path) -> Optional[Path]:
    resolved = _resolve_box_sources(relax_dir)
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
) -> DiscoveredBox:
    bdir = Path(box_dir)
    melt_use = Path(melt_dir) if melt_dir is not None else (bdir / "melt")
    quench_use = Path(quench_dir) if quench_dir is not None else (bdir / "quench")
    relax_use = Path(relax_dir) if relax_dir is not None else (bdir / "relax")
    if relax_dir is None and not relax_use.exists():
        relax_use = bdir

    resolved = _resolve_box_sources(relax_use)
    relax_data_use = Path(relax_data) if relax_data is not None else (resolved.final_structure or _guess_relax_data(relax_use))
    relax_dump_use = Path(relax_dump) if relax_dump is not None else _guess_relax_dump(relax_use)
    relax_traj_use = Path(relax_traj) if relax_traj is not None else _guess_relax_traj(relax_use)
    analysis_source_use = Path(analysis_source) if analysis_source is not None else (resolved.analysis_source or relax_data_use or relax_traj_use)
    analysis_source_role_use = str(analysis_source_role) if analysis_source_role is not None else resolved.analysis_source_role
    input_structure_use = Path(input_structure) if input_structure is not None else resolved.input_structure
    final_structure_use = Path(final_structure) if final_structure is not None else resolved.final_structure

    dens = _optional_float(density)
    dens_se = _optional_float(density_stderr)
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


def _read_atoms_snapshot(source_path: Path, *, type_to_species: Optional[Sequence[str]], atom_style: str):
    src = Path(source_path)
    if _is_dump_like(src):
        return None
    try:
        if _looks_like_lammps_data_file(src):
            try:
                from ase.io import read as ase_read

                return ase_read(
                    str(src),
                    format="lammps-data",
                    style=str(atom_style),
                    specorder=(None if type_to_species is None else list(type_to_species)),
                )
            except Exception:
                from ..io.lammps_data_minimal import read_lammps_data_minimal

                return read_lammps_data_minimal(
                    src,
                    atom_style=str(atom_style),
                    specorder=(None if type_to_species is None else list(type_to_species)),
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
) -> Optional[float]:
    atoms = _read_atoms_snapshot(source_path, type_to_species=type_to_species, atom_style=atom_style)
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


def _load_task_result_entry(path: Path) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    data = _load_json(path)
    status = str(data.get("status", "ok")).strip().lower()
    if status in {"ok", "success"}:
        if isinstance(data.get("box_entry", None), Mapping):
            return dict(data.get("box_entry", {})), None
        if isinstance(data.get("entry", None), Mapping):
            return dict(data.get("entry", {})), None
        if {"box", "metrics", "distributions"}.issubset(set(data.keys())):
            return dict(data), None
        # task box analysis
        return None, None
    box_label = data.get("box", None)
    if box_label is None and isinstance(data.get("task", None), Mapping):
        box_label = data.get("task", {}).get("box", None)
    reject = {
        "box": int(box_label or 0),
        "reason": "task_failed",
        "error": str(data.get("error", f"task_result status={status!r}")),
        "paths": {
            "task_result": str(path),
        },
    }
    return None, reject


def _discover_from_results_file(path: Path) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Unsupported output-analysis input: {path}")

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

    raw_boxes, entries, rejected, dataset_meta = discover_output_dataset(dataset_hint)
    meta = dict(dataset_meta)
    meta["source_results_json"] = str(path)
    return raw_boxes, entries, rejected, meta


def _discover_from_dataset_file(path: Path) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(path)
    base_dir = Path(path).resolve().parent
    boxes_raw = data.get("boxes", []) if isinstance(data.get("boxes", []), list) else []
    raw_boxes: list[DiscoveredBox] = []
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
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
            raw_boxes.append(
                _box_from_source_file(
                    source,
                    task_result=task_result,
                    box=box_fallback,
                    explicit_box=True,
                    analysis_source_role=str(analysis_source_role or "single_structure"),
                    source_layout=(None if source_layout in (None, "") else str(source_layout)),
                    source_record=source_record,
                )
            )
            continue

        raw_boxes.append(
            _box_from_dirs(
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
            )
        )
    return raw_boxes, entries, rejected, data


def _discover_from_directory(path: Path) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(path).resolve()
    dataset_file = root / "output_dataset.json"
    if dataset_file.exists():
        return _discover_from_dataset_file(dataset_file)

    task_results = sorted(root.rglob("task_result.json"))
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
        raw_boxes.append(_box_from_dirs(box_dir, task_result=task_result))
        seen_box_dirs.add(box_dir)

    box_dirs = sorted([p for p in root.glob("box_*") if p.is_dir()], key=_slug_box_id)
    for box_dir in box_dirs:
        if box_dir.resolve(strict=False) in seen_box_dirs:
            continue
        raw_boxes.append(_box_from_dirs(box_dir))
        seen_box_dirs.add(box_dir.resolve(strict=False))

    # box tree wrapper
    if not box_dirs and (root / "relax").exists():
        raw_boxes.append(_box_from_dirs(root))

    if not raw_boxes and not box_dirs and not (root / "relax").exists():
        skip_names = {"melt", "quench", "relax", "results", "task", "preview", "analysis", "rejects"}
        loose_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name not in skip_names], key=lambda p: p.name)
        for idx, box_dir in enumerate(loose_dirs, start=1):
            if box_dir.resolve(strict=False) in seen_box_dirs:
                continue
            relax_dir = box_dir / "relax" if (box_dir / "relax").exists() else box_dir
            if _guess_analysis_source(relax_dir) is None:
                continue
            raw_boxes.append(_box_from_dirs(box_dir, box=_resolved_box_id(box_dir, fallback=idx)))
            seen_box_dirs.add(box_dir.resolve(strict=False))

    flat_file_sources = 0
    ase_database_files = 0
    ase_database_rows = 0
    if not raw_boxes and not box_dirs and not (root / "relax").exists():
        direct_sources_all = [p for p in _iter_analysis_source_candidates(root) if p.parent == root]
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
    return raw_boxes, entries, rejected, dataset


def discover_output_dataset(path: Path) -> tuple[list[DiscoveredBox], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    p = Path(path).expanduser()
    if p.is_dir():
        return _discover_from_directory(p)
    if _looks_like_ase_database_file(p):
        return _discover_from_ase_database_file(p)
    if _is_analysis_source_candidate(p):
        dataset = {
            "schema": "vitriflow.output_dataset.v1",
            "source_root": str(p.parent.resolve()),
            "layout": "single_file",
            "n_flat_file_sources": 1,
        }
        return [_box_from_source_file(p, box=1)], [], [], dataset
    data = _load_json(p)
    schema = str(data.get("schema", "")).strip().lower() if isinstance(data, Mapping) else ""
    if schema == "vitriflow.output_dataset.v1" or p.name == "output_dataset.json":
        return _discover_from_dataset_file(p)
    if p.name == "task_result.json":
        entry, reject = _load_task_result_entry(p)
        box_dir = p.parent
        raw: list[DiscoveredBox] = []
        entries = [entry] if entry is not None else []
        rejected = [reject] if reject is not None else []
        if entry is None and reject is None:
            raw.append(_box_from_dirs(box_dir, task_result=p))
        dataset = {"schema": "vitriflow.output_dataset.v1", "source_root": str(p.parent.resolve())}
        return raw, entries, rejected, dataset
    if p.name in {"run_results.json", "autotune_results.json"}:
        return _discover_from_results_file(p)
    if isinstance(data, Mapping) and isinstance(data.get("production", None), Mapping):
        return _discover_from_results_file(p)
    raise ValueError(f"Unsupported output-analysis input: {p}")


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

    type_to_species = _get_type_to_species(config)
    metrics_in = metrics_cfg if metrics_cfg is not None else config.autotune.metrics
    metrics_eff, _warnings, summary = resolve_effective_metrics_config(
        metrics_in,
        structure_data=None,
        type_to_species=type_to_species,
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
    if dft_enabled:
        raise ValueError("Generic output analysis does not support production.dft_opt refinement")
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
            )
    except Exception as exc:
        return [], f"failed_to_read_analysis_source {p}: {exc}"
    if not frames:
        return [], f"no_frames_from_analysis_source: {p}"
    return list(frames), None


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

    provenance: dict[str, Any] = {
        "mode": "metrics_fixed_only",
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

    pooled_frames: list[Any] = []
    read_errors: list[str] = []
    for box in sorted(raw_boxes, key=lambda x: int(x.box)):
        frames, err = _analysis_frames_for_box(
            box,
            metrics_cfg=ctx.metrics_cfg,
            type_to_species=ctx.type_to_species,
            atom_style=str(ctx.atom_style),
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
            merged = dict(fixed)
            merged.update(plan_cutoffs)
            provenance["mode"] = "plan_fallback"
            provenance["plan_cutoffs_reused"] = True
            provenance["notes"].append(
                "Fell back to plan cutoffs because pooled auto estimation was unavailable or failed."
            )
            return merged, provenance

    provenance["mode"] = "per_box_fallback"
    provenance["notes"].append(
        "Unable to build pooled ensemble cutoffs; deferring auto-cutoff estimation to per-box analysis where needed."
    )
    return fixed, provenance


def analyze_output_data(
    *,
    config: Optional[RunConfig] = None,
    input_path: Path,
    outdir: Path,
    plan: Optional[Mapping[str, Any]] = None,
    analysis_context: Optional[AnalysisContext] = None,
    progress: Optional[CondensedProgressLog] = None,
) -> dict[str, Any]:
    """Output data."""

    outdir = Path(outdir)
    ensure_dir(outdir)
    if progress is None:
        progress = CondensedProgressLog(outdir / "condensed.log")

    raw_boxes, preset_entries, preset_rejected, dataset_meta = discover_output_dataset(Path(input_path))
    progress.info("analysis", f"discovered {len(raw_boxes)} raw boxes and {len(preset_entries)} pre-analysed task results")

    if plan is not None:
        ctx = _analysis_context_from_plan(config, plan)
    elif analysis_context is not None:
        ctx = analysis_context
    elif config is not None:
        ctx = _analysis_context_from_config(config)
    else:
        raise ValueError("analyze_output_data requires either a full RunConfig or a standalone analysis context")

    metrics_cfg = ctx.metrics_cfg
    type_to_species = ctx.type_to_species
    prod_cfg = ctx.prod_cfg
    conv_cfg = ctx.conv_cfg

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
    from .production_common import (
        analyse_production_box,
        build_production_convergence_spec,
        check_production_convergence,
        metrics_checked_from_conv_spec,
        resolve_production_time_unit_ps,
        resolve_production_warmup_duration_ps,
        resolve_production_warmup_steps,
        validate_production_entry_against_spec,
    )
    from ..analysis.motif_summary import summarize_production_crystal_motifs

    boxes: list[dict[str, Any]] = []
    rejected_boxes: list[dict[str, Any]] = list(preset_rejected)
    conv_spec: Optional[dict[str, Any]] = None
    source_role_counts = _analysis_source_role_counts(raw_boxes)
    relax_dir_counts = Counter(str(Path(box.relax_dir).resolve(strict=False)) for box in raw_boxes)
    shared_direct_source_dirs = {
        str(Path(box.relax_dir).resolve(strict=False))
        for box in raw_boxes
        if box.analysis_source is not None
        and Path(box.analysis_source).parent.resolve(strict=False) == Path(box.relax_dir).resolve(strict=False)
        and int(relax_dir_counts[str(Path(box.relax_dir).resolve(strict=False))]) > 1
    }

    def _stage_dir_for_analysis_artifacts(box: DiscoveredBox) -> Path:
        key = str(Path(box.relax_dir).resolve(strict=False))
        if key in shared_direct_source_dirs:
            return outdir / "box_artifacts" / f"box_{int(box.box):03d}"
        return Path(box.relax_dir)

    def _consume_entry(entry: dict[str, Any]) -> None:
        nonlocal conv_spec
        if conv_spec is None:
            conv_spec = build_production_convergence_spec(entry)
        else:
            validate_production_entry_against_spec(entry, conv_spec, box_label=entry.get("box", "?"))
        if bool(entry.get("reject")):
            rejected_boxes.append(entry)
        else:
            boxes.append(entry)

    for entry in sorted(preset_entries, key=lambda x: int(x.get("box", 0) or 0)):
        _consume_entry(dict(entry))

    for box in sorted(raw_boxes, key=lambda x: int(x.box)):
        source = box.analysis_source or box.relax_traj or box.relax_data
        relax_data = box.relax_data or source
        relax_traj = box.relax_traj or source
        if str(box.source_layout or "") == "ase_database":
            try:
                materialized_source = _materialize_ase_database_box_source(box, outdir=outdir)
            except Exception as exc:
                rejected_boxes.append(
                    {
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
                )
                progress.info("analysis", f"rejected box {int(box.box)}: ase_database_materialization_failed")
                continue
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
            )
            if density_mean is not None and density_stderr is None:
                density_stderr = 0.0

        if relax_data is None:
            rejected_boxes.append(
                {
                    "box": int(box.box),
                    "reason": "missing_relax_data",
                    "paths": {
                        "box_dir": _relpath_or_str(box.box_dir, outdir),
                        "analysis_source": _relpath_or_str(source, outdir),
                    },
                }
            )
            continue
        if relax_traj is None:
            rejected_boxes.append(
                {
                    "box": int(box.box),
                    "reason": "missing_relax_trajectory",
                    "paths": {
                        "box_dir": _relpath_or_str(box.box_dir, outdir),
                        "analysis_source": _relpath_or_str(source, outdir),
                    },
                }
            )
            continue

        analysis_stage_dir = _stage_dir_for_analysis_artifacts(box)
        ensure_dir(analysis_stage_dir)
        try:
            entry, prod_cutoffs = analyse_production_box(
                box_id=int(box.box),
                outdir=outdir,
                melt_stage_dir=box.melt_dir,
                quench_stage_dir=box.quench_dir,
                relax_stage_dir=analysis_stage_dir,
                relax_data_path=relax_data,
                density_mean=float(density_mean if density_mean is not None else float("nan")),
                density_stderr=float(density_stderr if density_stderr is not None else (0.0 if density_mean is not None else float("nan"))),
                metrics_cfg=metrics_cfg,
                cutoffs=prod_cutoffs,
                required_pairs=required_pairs,
                fixed_cutoffs=fixed_cutoffs,
                type_to_species=type_to_species,
                md_timestep=float(ctx.md_timestep),
                quench_window_steps_range=ctx.quench_window_steps_range,
                sampling_hint=ctx.sampling_hint,
                bondlen_cdf_points=int(getattr(prod_cfg, "bondlen_cdf_points", 200)),
                angle_cdf_points=int(getattr(prod_cfg, "angle_cdf_points", 180)),
                seeds=None,
                melt_elastic=None,
                relax_elastic=None,
                elastic_timeseries=None,
                exclude_coordination_defects=bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
                rejects_dir=(outdir / str(getattr(prod_cfg, "rejects_subdir", "rejects"))),
                relax_dump_path=box.relax_dump,
                relax_traj_path=relax_traj,
                analysis_source_path=source,
                analysis_source_role=box.analysis_source_role,
                atom_style=str(ctx.atom_style),
            )
        except Exception as exc:
            rejected_boxes.append(
                {
                    "box": int(box.box),
                    "reason": "analysis_failed",
                    "error": str(exc),
                    "analysis_source_role": box.analysis_source_role,
                    "paths": {
                        "box_dir": _relpath_or_str(box.box_dir, outdir),
                        "relax_dir": _relpath_or_str(box.relax_dir, outdir),
                        "analysis_artifact_dir": _relpath_or_str(analysis_stage_dir, outdir),
                        "analysis_source": _relpath_or_str(source, outdir),
                    },
                }
            )
            progress.info("analysis", f"rejected box {int(box.box)}: analysis_failed")
            continue
        _consume_entry(entry)
        progress.info("analysis", f"analysed box {int(box.box)}")

    converged = False
    conv_report: dict[str, Any] = {}
    status = "ok"
    error: Optional[str] = None

    if not boxes:
        status = "error"
        error = "no accepted boxes available for convergence analysis"
        conv_report = {"error": str(error)}
    elif conv_spec is None:
        status = "error"
        error = "no convergence specification could be constructed from analysed boxes"
        conv_report = {"error": str(error)}
    else:
        converged, conv_report = check_production_convergence(boxes, conv_spec, conv_cfg)
        progress.convergence("analysis", conv_report)

    dataset = {
        "schema": "vitriflow.output_dataset.v1",
        "source_root": str(Path(input_path).resolve() if Path(input_path).exists() else Path(input_path)),
        "boxes": [_dataset_record_for_box(box, base_dir=outdir) for box in sorted(raw_boxes, key=lambda x: int(x.box))],
        "n_preanalysed": int(len(preset_entries)),
        "n_task_failures": int(len(preset_rejected)),
        "metadata": {
            **dict(dataset_meta),
            "analysis_source_roles": dict(source_role_counts),
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

    results = {
        "schema": "vitriflow.analysis_results.v1",
        "status": str(status),
        "error": (None if error is None else str(error)),
        "converged": bool(converged),
        "n_boxes": int(len(boxes)),
        "n_boxes_accepted": int(len(boxes)),
        "n_boxes_rejected": int(len(rejected_boxes)),
        "n_boxes_total": int(len(boxes) + len(rejected_boxes)),
        "check_convergence": True,
        "exclude_coordination_defects": bool(getattr(prod_cfg, "exclude_coordination_defects", False)),
        "rejects_subdir": str(getattr(prod_cfg, "rejects_subdir", "rejects")),
        "warmup_start_temperature": float(getattr(prod_cfg, "warmup_start_temperature", 300.0)),
        "warmup_duration_ps": float(warmup_duration_ps),
        "warmup_steps": int(warmup_steps),
        "cutoffs": _cutoffs_list_from_dict(prod_cutoffs),
        "cutoff_provenance": cutoff_provenance,
        "convergence_spec": conv_spec,
        "convergence": conv_report,
        "crystal_motifs": summarize_production_crystal_motifs(boxes, rejected_boxes=rejected_boxes),
        "metrics_checked": metrics_checked_from_conv_spec(conv_spec),
        "effective_metrics": dict(ctx.effective_metrics),
        "metric_warnings": list(ctx.metric_warnings),
        "analysis_source_roles": dict(source_role_counts),
        "boxes": boxes,
        "rejected_boxes": rejected_boxes,
        "paths": {
            "output_dataset": "output_dataset.json",
            "analysis_results": "analysis_results.json",
            "condensed_log": "condensed.log",
        },
    }

    atomic_write_json(outdir / "output_dataset.json", dataset)
    atomic_write_json(outdir / "analysis_results.json", results)
    progress.info("analysis", "wrote analysis_results.json and output_dataset.json")
    return results
