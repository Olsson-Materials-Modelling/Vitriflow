from __future__ import annotations

"""Explicit graph-induction rules and structure manifest helpers.

The graph layer deliberately sits below descriptor evaluation.  A descriptor may
consume a :class:`StructureGraph`, but it must not invent a neighbour cutoff once
that graph has been supplied.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

def _require_ase():
    try:
        from ase import Atoms as ASEAtoms  # type: ignore
        from ase.neighborlist import neighbor_list as ase_neighbor_list  # type: ignore
        return ASEAtoms, ase_neighbor_list
    except Exception as e:  # pragma: no cover
        raise ImportError("vitriflow.analysis.graph requires ASE") from e


from .common import (
    canonical_unique_mic_pairs as _canonical_unique_mic_pairs,
    resolve_selector as _resolve_selector,
    wrap_frac as _wrap_frac,
)
from .dump import frame_pbc
from .provenance import json_sanitize


_SUPPORTED_GRAPH_KINDS = {
    "hard_cutoff",
    "hard_cutoff_sweep",
    "hard_cutoff_interval",
    "soft_logistic",
    # Adaptive kinds are resolved against each analysed structure before graph
    # construction. They produce concrete hard_cutoff / soft_logistic rules
    # whose parameters contain the per-structure RDF-derived values.
    "rdf_adaptive",  # backwards-compatible generic alias
    "rdf_adaptive_hard_cutoff",
    "rdf_adaptive_hard_cutoff_sweep",
    "rdf_adaptive_hard_cutoff_interval",
    "rdf_adaptive_soft_logistic",
}

_RDF_DERIVE_VALUES = {
    "rdf",
    "rdf_minimum",
    "rdf_first_minimum",
    "pair_distribution",
    "pair_distribution_function",
    "shell_separability",
}

_RDF_DYNAMIC_KIND_MAP = {
    "rdf_adaptive": "hard_cutoff",
    "rdf_adaptive_hard_cutoff": "hard_cutoff",
    "rdf_adaptive_hard_cutoff_sweep": "hard_cutoff_sweep",
    "rdf_adaptive_hard_cutoff_interval": "hard_cutoff_interval",
    "rdf_adaptive_soft_logistic": "soft_logistic",
}


def _json_float_vector(arr: Any) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1).tolist()]


def _json_float_matrix(arr: Any) -> list[list[float]]:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        a = a.reshape((-1, 3))
    return [[float(x) for x in row] for row in a.tolist()]


def _json_safe(value: Any) -> Any:
    # Public/canonical JSON must never contain raw NaN or Inf.  The
    # provenance/status layer records why a numeric value is unavailable; the
    # serializer itself represents it as JSON null.
    return json_sanitize(value)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def source_file_identity(path: Optional[Path]) -> dict[str, Any]:
    """Return a content identity for the structure source artifact."""

    if path is None:
        return {"path": None, "exists": False, "size_bytes": None, "sha256": None}
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False, "size_bytes": None, "sha256": None}
    before = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = p.stat()
    if (
        int(before.st_size) != int(after.st_size)
        or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        or int(before.st_ino) != int(after.st_ino)
    ):
        raise ValueError(f"source artifact changed while hashing: {p}")
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def species_from_frame(frame: Any, *, type_to_species: Optional[Sequence[str]] = None) -> list[str]:
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    if type_to_species is None:
        return [f"type:{int(t)}" for t in types.tolist()]
    mapping = [str(x) for x in type_to_species]
    species: list[str] = []
    for t in types.tolist():
        ti = int(t)
        species.append(mapping[ti - 1] if 1 <= ti <= len(mapping) else f"type:{ti}")
    return species


def structure_serialized_object(frame: Any, *, type_to_species: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Return the canonical structure object used for manifest hashing.

    The structure hash is computed from exactly the analysed object
    ``(cell, species, positions, periodic boundary flags)`` in the in-memory
    order read by the trajectory/data loader.
    """

    try:
        cell = np.asarray(getattr(frame, "cell"), dtype=float)
    except Exception as exc:
        raise ValueError("structure cell must be a numeric 3x3 matrix") from exc
    if cell.shape != (3, 3):
        raise ValueError(f"structure cell must have shape (3, 3); got {cell.shape}")
    if not np.all(np.isfinite(cell)):
        raise ValueError("structure cell must contain only finite values")
    try:
        volume = abs(float(np.linalg.det(cell)))
    except Exception as exc:  # pragma: no cover - shape/type checks above are defensive
        raise ValueError("structure cell volume could not be evaluated") from exc
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("structure cell must have a finite, strictly positive volume")

    try:
        positions = np.asarray(getattr(frame, "positions"), dtype=float)
    except Exception as exc:
        raise ValueError("structure positions must be a numeric n_atoms x 3 matrix") from exc
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "structure positions must have shape (n_atoms, 3); "
            f"got {positions.shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("structure positions must contain only finite values")

    try:
        types = np.asarray(getattr(frame, "types"), dtype=int)
    except Exception as exc:
        raise ValueError("structure atom types must be a one-dimensional integer sequence") from exc
    if types.ndim != 1:
        raise ValueError(
            "structure atom types must be one-dimensional; "
            f"got shape {types.shape}"
        )
    if int(types.size) != int(positions.shape[0]):
        raise ValueError(
            "structure atom-type count must match the number of positions: "
            f"types={int(types.size)} positions={int(positions.shape[0])}"
        )

    species = species_from_frame(frame, type_to_species=type_to_species)
    if len(species) != int(positions.shape[0]):
        raise ValueError(
            "structure species count must match the number of positions: "
            f"species={len(species)} positions={int(positions.shape[0])}"
        )

    pbc_list = list(frame_pbc(frame))
    return {
        "cell": _json_float_matrix(cell),
        "species": species,
        "positions": _json_float_matrix(positions),
        "pbc": pbc_list,
    }


def structure_hash(frame: Any, *, type_to_species: Optional[Sequence[str]] = None) -> str:
    return _sha256_json(structure_serialized_object(frame, type_to_species=type_to_species))


def cell_hash(frame: Any) -> str:
    return _sha256_json({"cell": _json_float_matrix(getattr(frame, "cell"))})


def positions_hash(frame: Any) -> str:
    return _sha256_json({"positions": _json_float_matrix(getattr(frame, "positions"))})


def symbols_hash(frame: Any, *, type_to_species: Optional[Sequence[str]] = None) -> str:
    return _sha256_json({"species": species_from_frame(frame, type_to_species=type_to_species)})


def _composition_from_species(species: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for sp in species:
        key = str(sp)
        out[key] = int(out.get(key, 0)) + 1
    return {k: out[k] for k in sorted(out)}


def _reader_backend_for_path(path: Optional[Path]) -> str:
    if path is None:
        return "in_memory"
    suffix = str(Path(path).suffix).lower()
    if suffix == ".restart":
        return "vitriflow.cp2k_restart"
    if suffix in {".extxyz", ".xyz"}:
        return "ase"
    if suffix in {".data", ".lmp"}:
        return "vitriflow.lammps_data_minimal_or_ase"
    if suffix in {".dump", ".lammpstrj"}:
        return "vitriflow.lammps_dump"
    if suffix in {".db", ".sqlite"}:
        return "ase.db"
    return "ase_or_vitriflow_auto"


def _pbc_provenance_for_path(path: Optional[Path]) -> str:
    if path is None:
        return "in_memory_frame_metadata"
    suffix = str(Path(path).suffix).lower()
    if suffix in {".dump", ".lammpstrj", ".trj"}:
        return "parsed_lammps_box_bounds_flags"
    if suffix in {".extxyz", ".xyz"}:
        return "parsed_extxyz_or_ase_pbc"
    if suffix == ".restart":
        return "parsed_cp2k_cell_periodic_or_lammps_bounds"
    if suffix in {".data", ".lmp", ".dat"}:
        return "vitriflow_periodic_lammps_cell_contract_format_has_no_boundary_flags"
    return "structure_reader_pbc_metadata"


def manifest_row_from_frame(
    frame: Any,
    *,
    box_id: int,
    source_path: Optional[Path],
    source_role: Optional[str],
    type_to_species: Optional[Sequence[str]] = None,
    density: Optional[float] = None,
) -> dict[str, Any]:
    p = Path(source_path) if source_path is not None else None
    st = None
    if p is not None:
        try:
            st = p.stat()
        except Exception:
            st = None
    structure_object = structure_serialized_object(frame, type_to_species=type_to_species)
    cell = np.asarray(structure_object["cell"], dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    species = [str(x) for x in structure_object["species"]]
    composition = _composition_from_species(species)
    material_id = "-".join(f"{k}{v}" for k, v in composition.items()) if composition else "unknown"
    source_identity = source_file_identity(p)
    return {
        "schema": "vitriflow.structure_manifest.row.v2",
        "box_id": int(box_id),
        "material_id": material_id,
        "composition": composition,
        "source_path": (None if p is None else str(p)),
        "source_role": (None if source_role is None else str(source_role)),
        "structure_hash": _sha256_json(structure_object),
        "cell_hash": _sha256_json({"cell": structure_object["cell"]}),
        "positions_hash": _sha256_json({"positions": structure_object["positions"]}),
        "symbols_hash": _sha256_json({"species": structure_object["species"]}),
        "n_atoms": int(len(structure_object["positions"])),
        "volume": float(volume),
        "density": (None if density is None or not math.isfinite(float(density)) else float(density)),
        "units": {"length": "angstrom", "volume": "angstrom^3", "density": "g/cm^3"},
        "pbc": list(structure_object["pbc"]),
        "pbc_provenance": _pbc_provenance_for_path(p),
        "file_size": (None if st is None else int(st.st_size)),
        "mtime": (None if st is None else float(st.st_mtime)),
        "source_file_identity": source_identity,
        "reader_backend": _reader_backend_for_path(p),
        "reader_version": "vitriflow_auto_v2",
    }


def verify_manifest_row(
    frame: Any,
    row: Mapping[str, Any],
    *,
    type_to_species: Optional[Sequence[str]] = None,
    source_path: Optional[Path] = None,
) -> None:
    structure_object = structure_serialized_object(frame, type_to_species=type_to_species)
    actual_hashes = {
        "structure_hash": _sha256_json(structure_object),
        "cell_hash": _sha256_json({"cell": structure_object["cell"]}),
        "positions_hash": _sha256_json({"positions": structure_object["positions"]}),
        "symbols_hash": _sha256_json({"species": structure_object["species"]}),
    }
    for field_name, actual in actual_hashes.items():
        expected_raw = row.get(field_name)
        expected = "" if expected_raw is None else str(expected_raw)
        if expected != actual:
            label = (
                "structure manifest hash mismatch"
                if field_name == "structure_hash"
                else "structure manifest component hash mismatch"
            )
            raise ValueError(
                f"{label} before descriptor analysis: "
                f"box_id={row.get('box_id', '?')} component={field_name} "
                f"expected={expected or '<missing>'} actual={actual}"
            )

    expected_pbc = row.get("pbc")
    if (
        not isinstance(expected_pbc, Sequence)
        or isinstance(expected_pbc, (str, bytes, bytearray))
        or len(expected_pbc) != 3
        or not all(isinstance(value, (bool, np.bool_)) for value in expected_pbc)
    ):
        raise ValueError(
            "structure manifest pbc is missing or malformed before descriptor analysis: "
            f"box_id={row.get('box_id', '?')} pbc={expected_pbc!r}"
        )
    actual_pbc = tuple(bool(value) for value in structure_object["pbc"])
    recorded_pbc = tuple(bool(value) for value in expected_pbc)
    if recorded_pbc != actual_pbc:
        raise ValueError(
            "structure manifest pbc mismatch before descriptor analysis: "
            f"box_id={row.get('box_id', '?')} expected={list(recorded_pbc)} actual={list(actual_pbc)}"
        )

    expected_n_atoms = row.get("n_atoms")
    recorded_n_atoms = (
        int(expected_n_atoms)
        if isinstance(expected_n_atoms, (int, np.integer))
        and not isinstance(expected_n_atoms, (bool, np.bool_))
        else -1
    )
    actual_n_atoms = int(len(structure_object["positions"]))
    if recorded_n_atoms != actual_n_atoms:
        raise ValueError(
            "structure manifest n_atoms mismatch before descriptor analysis: "
            f"box_id={row.get('box_id', '?')} expected={expected_n_atoms!r} actual={actual_n_atoms}"
        )

    expected_source = row.get("source_file_identity", {})
    if isinstance(expected_source, Mapping) and expected_source.get("sha256") is not None:
        expected_sha = str(expected_source.get("sha256", ""))
        expected_size_raw = expected_source.get("size_bytes")
        expected_size = (
            int(expected_size_raw)
            if isinstance(expected_size_raw, (int, np.integer))
            and not isinstance(expected_size_raw, (bool, np.bool_))
            and int(expected_size_raw) >= 0
            else -1
        )
        try:
            valid_sha = len(expected_sha) == 64 and int(expected_sha, 16) >= 0
        except ValueError:
            valid_sha = False
        if not valid_sha or expected_size < 0:
            raise ValueError(
                "structure manifest source artifact identity is malformed: "
                f"box_id={row.get('box_id', '?')} sha256={expected_sha!r} "
                f"size_bytes={expected_size_raw!r}"
            )
        candidate = source_path if source_path is not None else row.get("source_path")
        actual_source = source_file_identity(None if candidate in (None, "") else Path(str(candidate)))
        if not bool(actual_source.get("exists", False)):
            raise ValueError(
                "structure manifest source artifact is unavailable during verification: "
                f"box_id={row.get('box_id', '?')} path={candidate}"
            )
        if (
            str(actual_source.get("sha256")) != expected_sha
            or int(actual_source.get("size_bytes", -1)) != expected_size
        ):
            raise ValueError(
                "structure manifest source artifact identity mismatch: "
                f"box_id={row.get('box_id', '?')} path={candidate}"
            )


@dataclass(frozen=True)
class GraphRule:
    name: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    provenance: Any = "runtime"

    def __post_init__(self) -> None:
        kind = str(self.kind)
        if kind not in _SUPPORTED_GRAPH_KINDS:
            raise ValueError(f"Unsupported graph rule kind: {kind}")
        if not str(self.name).strip():
            raise ValueError("graph rule name must be non-empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "kind": str(self.kind),
            "parameters": _json_safe(dict(self.parameters or {})),
            "provenance": _json_safe(self.provenance),
        }

    @classmethod
    def from_any(cls, obj: Any, *, default_name: str = "graph_rule") -> "GraphRule":
        if isinstance(obj, GraphRule):
            return obj
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")
        if not isinstance(obj, Mapping):
            raise TypeError(f"GraphRule requires a mapping-like object, got {type(obj)!r}")
        data = dict(obj)
        kind = str(data.get("kind", "hard_cutoff"))
        name = str(data.get("name", "") or default_name)
        params = dict(data.get("parameters", {}) or {})
        if kind in _RDF_DYNAMIC_KIND_MAP:
            params.setdefault("derive_from", "pair_distribution_function")
            params.setdefault("original_kind", kind)
            if kind == "rdf_adaptive_hard_cutoff":
                params.setdefault("mode", "single")
            elif kind == "rdf_adaptive_hard_cutoff_sweep":
                params.setdefault("mode", "sweep_only")
            elif kind == "rdf_adaptive_hard_cutoff_interval":
                params.setdefault("mode", "interval")
            elif kind == "rdf_adaptive_soft_logistic":
                params.setdefault("mode", "soft_only")
            kind = "rdf_adaptive"
        else:
            derive = str(params.get("derive_from", params.get("source", params.get("cutoff_source", "")))).strip().lower()
            if derive in _RDF_DERIVE_VALUES:
                params.setdefault("original_kind", kind)
                if kind == "hard_cutoff":
                    params.setdefault("mode", "single")
                    kind = "rdf_adaptive"
                elif kind == "hard_cutoff_sweep":
                    params.setdefault("mode", "sweep_only")
                    kind = "rdf_adaptive"
                elif kind == "hard_cutoff_interval":
                    params.setdefault("mode", "interval")
                    kind = "rdf_adaptive"
                elif kind == "soft_logistic":
                    params.setdefault("mode", "soft_only")
                    kind = "rdf_adaptive"
        # Ergonomic top-level aliases are folded into parameters.
        for key in (
            "cutoff",
            "cutoffs",
            "values",
            "r_values",
            "r_min",
            "r_max",
            "n",
            "points",
            "r0",
            "sigma",
            "pair_cutoffs",
            "search_radius",
            "pair",
            "pairs",
            "reference_pair",
            "selector_pair",
            "mode",
            "bin_width",
            "smooth_width",
            "connectivity_fraction",
            "connectivity_pairs",
            "apply_to",
            "sigma",
            "min_weight",
            "graph_family",
            "graph_families",
            "graph_family_strategy",
            "graph_scope",
            "rule_scope",
            "network_pairs",
            "expected_shell_pairs",
            "candidate_contact_pairs",
            "ensemble",
            "ensemble_scope",
        ):
            if key in data and key not in params:
                params[key] = data[key]
        prov = data.get("provenance", "config")
        return cls(name=name, kind=kind, parameters=params, provenance=prov)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    ai = int(a)
    bi = int(b)
    return (ai, bi) if ai <= bi else (bi, ai)


def cutoff_rows_from_dict(cutoffs: Mapping[Tuple[int, int], float]) -> list[dict[str, Any]]:
    return [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(dict(cutoffs).items())]


def pair_cutoffs_from_parameters(parameters: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    raw = parameters.get("cutoffs", parameters.get("pair_cutoffs", None))
    out: dict[tuple[int, int], float] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(k, tuple) and len(k) == 2:
                out[_pair_key(int(k[0]), int(k[1]))] = float(v)
                continue
            text = str(k).replace(",", "-").replace(":", "-")
            parts = [p for p in text.split("-") if p.strip()]
            if len(parts) == 2:
                out[_pair_key(int(parts[0]), int(parts[1]))] = float(v)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        # A numeric list belongs to hard_cutoff_sweep, not pair-specific cutoffs.
        if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in list(raw)):
            for ent in raw:
                if not isinstance(ent, Mapping):
                    continue
                pair = ent.get("pair", None)
                cutoff = ent.get("cutoff", None)
                if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)) and len(pair) == 2 and cutoff is not None:
                    out[_pair_key(int(pair[0]), int(pair[1]))] = float(cutoff)
    return out


def pair_intervals_from_parameters(parameters: Mapping[str, Any]) -> dict[tuple[int, int], tuple[float, float]]:
    """Return per-pair hard-cutoff intervals from a graph-rule parameter block.

    Adaptive RDF rules can define one interval per selector pair.  Robust
    coordination partitioning must use the interval for the coordination pair,
    not require a single global r_min/r_max.
    """

    raw = parameters.get("pair_intervals", parameters.get("intervals", None))
    out: dict[tuple[int, int], tuple[float, float]] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(k, tuple) and len(k) == 2:
                key = _pair_key(int(k[0]), int(k[1]))
            else:
                text = str(k).replace(",", "-").replace(":", "-")
                parts = [p for p in text.split("-") if p.strip()]
                if len(parts) != 2:
                    continue
                key = _pair_key(int(parts[0]), int(parts[1]))
            if isinstance(v, Mapping):
                lo = v.get("r_min", v.get("min", None))
                hi = v.get("r_max", v.get("max", None))
            elif isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)) and len(v) >= 2:
                lo, hi = v[0], v[1]
            else:
                continue
            if lo is None or hi is None:
                continue
            out[key] = (float(lo), float(hi))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for ent in raw:
            if not isinstance(ent, Mapping):
                continue
            pair = ent.get("pair", None)
            lo = ent.get("r_min", ent.get("min", None))
            hi = ent.get("r_max", ent.get("max", None))
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)) and len(pair) == 2 and lo is not None and hi is not None:
                out[_pair_key(int(pair[0]), int(pair[1]))] = (float(lo), float(hi))
    # Single-pair compatibility.
    if not out and parameters.get("r_min", None) is not None and parameters.get("r_max", None) is not None:
        try:
            cut = pair_cutoffs_from_parameters(parameters)
            if len(cut) == 1:
                key = next(iter(cut))
                out[key] = (float(parameters["r_min"]), float(parameters["r_max"]))
        except Exception:
            pass
    return out


def legacy_graph_rule_from_cutoffs(cutoffs: Mapping[Tuple[int, int], float]) -> GraphRule:
    return GraphRule(
        name="legacy_single_cutoff",
        kind="hard_cutoff",
        parameters={
            "cutoffs": cutoff_rows_from_dict(cutoffs),
            "legacy": True,
            "single_rule_output": True,
        },
        provenance={
            "source": "legacy_cutoffs",
            "note": "Backward-compatible single-rule graph induced from the existing cutoff map.",
        },
    )


def _numeric_sequence(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [float(x) for x in value]
    return [float(value)]


def expand_graph_rules(raw_rules: Sequence[Any], *, legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None) -> list[GraphRule]:
    """Expand sweep/interval rules into concrete graph-induction rules."""

    if not raw_rules:
        legacy = dict(legacy_cutoffs or {})
        return [legacy_graph_rule_from_cutoffs(legacy)] if legacy else []

    expanded: list[GraphRule] = []
    for idx, raw in enumerate(raw_rules, start=1):
        parent = GraphRule.from_any(raw, default_name=f"graph_rule_{idx}")
        params = dict(parent.parameters or {})
        if parent.kind == "hard_cutoff_sweep":
            vals = _numeric_sequence(params.get("cutoffs", params.get("values", params.get("r_values", None))))
            if not vals and all(k in params for k in ("r_min", "r_max")):
                n = max(2, int(params.get("n", params.get("points", 9)) or 9))
                vals = [float(x) for x in np.linspace(float(params["r_min"]), float(params["r_max"]), n).tolist()]
            if not vals:
                raise ValueError(f"hard_cutoff_sweep graph rule {parent.name!r} has no cutoffs")
            for r in vals:
                expanded.append(
                    GraphRule(
                        name=f"{parent.name}_r{float(r):.6g}".replace(".", "p"),
                        kind="hard_cutoff",
                        parameters={
                            "cutoff": float(r),
                            "parent_rule_name": parent.name,
                            "parent_rule_kind": parent.kind,
                        },
                        provenance=parent.provenance,
                    )
                )
        elif parent.kind == "hard_cutoff_interval":
            r_min = float(params.get("r_min", params.get("min", float("nan"))))
            r_max = float(params.get("r_max", params.get("max", float("nan"))))
            if not (math.isfinite(r_min) and math.isfinite(r_max) and r_max >= r_min > 0.0):
                raise ValueError(f"hard_cutoff_interval graph rule {parent.name!r} requires 0 < r_min <= r_max")
            n = max(2, int(params.get("n", params.get("points", 9)) or 9))
            for r in np.linspace(r_min, r_max, n).tolist():
                expanded.append(
                    GraphRule(
                        name=f"{parent.name}_r{float(r):.6g}".replace(".", "p"),
                        kind="hard_cutoff",
                        parameters={
                            "cutoff": float(r),
                            "interval": [float(r_min), float(r_max)],
                            "parent_rule_name": parent.name,
                            "parent_rule_kind": parent.kind,
                            "interval_points": int(n),
                        },
                        provenance=parent.provenance,
                    )
                )
        else:
            expanded.append(parent)
    return expanded


def interval_graph_rules(raw_rules: Sequence[Any]) -> list[GraphRule]:
    out: list[GraphRule] = []
    for idx, raw in enumerate(raw_rules or [], start=1):
        rule = GraphRule.from_any(raw, default_name=f"graph_rule_{idx}")
        if rule.kind == "hard_cutoff_interval":
            out.append(rule)
    return out


@dataclass(frozen=True)
class StructureGraph:
    nodes: list[int]
    species: list[str]
    edges: list[tuple[int, int]]
    edge_distances: list[float]
    edge_weights: list[float]
    edge_vectors: list[list[float]]
    periodic: dict[str, Any]
    graph_rule: GraphRule
    structure_hash: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "vitriflow.structure_graph.v1",
            "nodes": [int(x) for x in self.nodes],
            "species": [str(x) for x in self.species],
            "edges": [[int(a), int(b)] for a, b in self.edges],
            "edge_distances": [float(x) for x in self.edge_distances],
            "edge_weights": [float(x) for x in self.edge_weights],
            "periodic": _json_safe(self.periodic),
            "graph_rule": self.graph_rule.to_json(),
            "structure_hash": str(self.structure_hash),
        }

    @property
    def is_soft(self) -> bool:
        return str(self.graph_rule.kind) == "soft_logistic"

    @property
    def is_hard(self) -> bool:
        return not self.is_soft


def _frame_periodic_metadata(frame: Any) -> dict[str, Any]:
    pbc = structure_serialized_object(frame)["pbc"]
    return {
        "pbc": pbc,
        "cell": _json_float_matrix(getattr(frame, "cell")),
        "origin": _json_float_vector(getattr(frame, "origin", np.zeros(3, dtype=float))),
        "vectors_are_rows": True,
        "distance_units": "angstrom",
    }


def _wrapped_positions_and_fractional(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    cell = np.asarray(getattr(frame, "cell"), dtype=float)
    origin = np.asarray(getattr(frame, "origin", np.zeros(3, dtype=float)), dtype=float)
    positions = np.asarray(getattr(frame, "positions"), dtype=float)
    invH = np.linalg.inv(cell)
    frac = _wrap_frac((positions - origin) @ invH, pbc=frame_pbc(frame))
    posw = origin + frac @ cell
    return posw, frac


def _unique_pairs_with_distances(frame: Any, r_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError("graph construction requires a finite positive search radius")
    posw, frac = _wrapped_positions_and_fractional(frame)
    ASEAtoms, ase_neighbor_list = _require_ase()
    pbc = frame_pbc(frame)
    atoms = ASEAtoms(
        numbers=np.ones(int(getattr(frame, "n_atoms")), dtype=int),
        positions=posw,
        cell=np.asarray(getattr(frame, "cell"), dtype=float),
        pbc=pbc,
    )
    ii, jj = ase_neighbor_list("ij", atoms, float(r_max))
    ii, jj, vec, dist = _canonical_unique_mic_pairs(
        frac,
        np.asarray(getattr(frame, "cell"), dtype=float),
        ii,
        jj,
        cutoff=float(r_max),
        pbc=pbc,
    )
    return ii, jj, np.asarray(vec, dtype=float), np.asarray(dist, dtype=float)


def build_hard_graph(frame: Any, graph_rule: GraphRule, *, type_to_species: Optional[Sequence[str]] = None) -> StructureGraph:
    """Build a binary neighbour graph from an explicit hard-cutoff rule."""

    rule = GraphRule.from_any(graph_rule)
    params = dict(rule.parameters or {})
    pair_cutoffs = pair_cutoffs_from_parameters(params)
    global_cutoff = params.get("cutoff", None)
    global_cut = None if global_cutoff is None else float(global_cutoff)
    if pair_cutoffs:
        max_cut = max(float(x) for x in pair_cutoffs.values())
        if global_cut is not None:
            max_cut = max(max_cut, float(global_cut))
    elif global_cut is not None:
        max_cut = float(global_cut)
    else:
        raise ValueError(f"hard graph rule {rule.name!r} must define cutoff or pair cutoffs")

    ii, jj, vec, dist = _unique_pairs_with_distances(frame, max_cut)
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    edges: list[tuple[int, int]] = []
    edge_distances: list[float] = []
    edge_vectors: list[list[float]] = []
    for a, b, v, d in zip(ii.tolist(), jj.tolist(), vec.tolist(), dist.tolist()):
        key = _pair_key(int(types[int(a)]), int(types[int(b)]))
        cutoff = pair_cutoffs.get(key, global_cut)
        if cutoff is None:
            continue
        if float(d) <= float(cutoff):
            edges.append((int(a), int(b)))
            edge_distances.append(float(d))
            edge_vectors.append([float(x) for x in v])

    return StructureGraph(
        nodes=[int(i) for i in range(int(getattr(frame, "n_atoms")))],
        species=species_from_frame(frame, type_to_species=type_to_species),
        edges=edges,
        edge_distances=edge_distances,
        edge_weights=[1.0 for _ in edges],
        edge_vectors=edge_vectors,
        periodic=_frame_periodic_metadata(frame),
        graph_rule=rule,
        structure_hash=structure_hash(frame, type_to_species=type_to_species),
    )


def _logistic_weight(d: float, r0: float, sigma: float) -> float:
    z = (float(d) - float(r0)) / float(sigma)
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return 1.0
    return float(1.0 / (1.0 + math.exp(z)))


def build_soft_graph(frame: Any, graph_rule: GraphRule, *, type_to_species: Optional[Sequence[str]] = None) -> StructureGraph:
    """Build a weighted logistic neighbour graph from an explicit soft rule."""

    rule = GraphRule.from_any(graph_rule)
    params = dict(rule.parameters or {})
    r0 = float(params.get("r0", params.get("cutoff", float("nan"))))
    sigma = float(params.get("sigma", float("nan")))
    pair_r0s = pair_cutoffs_from_parameters(params)
    if pair_r0s and not math.isfinite(r0):
        r0 = float(max(pair_r0s.values()))
    if not (math.isfinite(r0) and r0 > 0.0 and math.isfinite(sigma) and sigma > 0.0):
        raise ValueError("soft_logistic graph rule requires finite positive r0 and sigma")
    max_r0 = max([float(r0)] + [float(x) for x in pair_r0s.values()]) if pair_r0s else float(r0)
    r_max = float(params.get("r_max", params.get("search_radius", max_r0 + 8.0 * sigma)))
    if not (math.isfinite(r_max) and r_max > 0.0):
        r_max = max_r0 + 8.0 * sigma

    ii, jj, vec, dist = _unique_pairs_with_distances(frame, r_max)
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    edges: list[tuple[int, int]] = []
    edge_distances: list[float] = []
    edge_vectors: list[list[float]] = []
    edge_weights: list[float] = []
    min_weight = float(params.get("min_weight", 0.0) or 0.0)
    for a, b, v, d in zip(ii.tolist(), jj.tolist(), vec.tolist(), dist.tolist()):
        key = _pair_key(int(types[int(a)]), int(types[int(b)]))
        if pair_r0s and key not in pair_r0s:
            continue
        local_r0 = float(pair_r0s.get(key, r0))
        w = _logistic_weight(float(d), local_r0, sigma)
        if w <= min_weight:
            continue
        edges.append((int(a), int(b)))
        edge_distances.append(float(d))
        edge_vectors.append([float(x) for x in v])
        edge_weights.append(float(w))

    return StructureGraph(
        nodes=[int(i) for i in range(int(getattr(frame, "n_atoms")))],
        species=species_from_frame(frame, type_to_species=type_to_species),
        edges=edges,
        edge_distances=edge_distances,
        edge_weights=edge_weights,
        edge_vectors=edge_vectors,
        periodic=_frame_periodic_metadata(frame),
        graph_rule=rule,
        structure_hash=structure_hash(frame, type_to_species=type_to_species),
    )


def build_graph(frame: Any, graph_rule: GraphRule, *, type_to_species: Optional[Sequence[str]] = None) -> StructureGraph:
    rule = GraphRule.from_any(graph_rule)
    if rule.kind == "soft_logistic":
        return build_soft_graph(frame, rule, type_to_species=type_to_species)
    if rule.kind == "hard_cutoff":
        return build_hard_graph(frame, rule, type_to_species=type_to_species)
    raise ValueError(f"Graph rule {rule.name!r} must be expanded before graph construction (kind={rule.kind})")


def directed_neighbor_lists(graph: StructureGraph, n_atoms: Optional[int] = None) -> tuple[list[list[int]], list[list[np.ndarray]], list[list[float]], list[list[float]]]:
    n = int(n_atoms if n_atoms is not None else len(graph.nodes))
    nbr_ids: list[list[int]] = [[] for _ in range(n)]
    nbr_vecs: list[list[np.ndarray]] = [[] for _ in range(n)]
    nbr_dists: list[list[float]] = [[] for _ in range(n)]
    nbr_weights: list[list[float]] = [[] for _ in range(n)]
    for (a, b), v, d, w in zip(graph.edges, graph.edge_vectors, graph.edge_distances, graph.edge_weights):
        va = np.asarray(v, dtype=float)
        nbr_ids[int(a)].append(int(b))
        nbr_vecs[int(a)].append(va)
        nbr_dists[int(a)].append(float(d))
        nbr_weights[int(a)].append(float(w))
        nbr_ids[int(b)].append(int(a))
        nbr_vecs[int(b)].append(-va)
        nbr_dists[int(b)].append(float(d))
        nbr_weights[int(b)].append(float(w))
    return nbr_ids, nbr_vecs, nbr_dists, nbr_weights


# -----------------------------------------------------------------------------
# Per-structure RDF / pair-distribution graph-rule resolution
# -----------------------------------------------------------------------------


def _species_label_for_type(t: int, type_to_species: Optional[Sequence[str]]) -> str:
    ti = int(t)
    if type_to_species is not None and 1 <= ti <= len(type_to_species):
        return str(type_to_species[ti - 1])
    return f"type:{ti}"


def _selector_pair_to_keys(raw_pair: Any, type_to_species: Optional[Sequence[str]]) -> list[tuple[int, int]]:
    if not (isinstance(raw_pair, Sequence) and not isinstance(raw_pair, (str, bytes, bytearray)) and len(raw_pair) == 2):
        return []
    a_types = [int(x) for x in _resolve_selector(raw_pair[0], type_to_species)]
    b_types = [int(x) for x in _resolve_selector(raw_pair[1], type_to_species)]
    return sorted({_pair_key(a, b) for a in a_types for b in b_types})


def _pairs_from_metrics(metrics: Any, type_to_species: Optional[Sequence[str]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cm in list(getattr(metrics, "coordinations", []) or []):
        out.extend(_selector_pair_to_keys([getattr(cm, "central"), getattr(cm, "neighbor")], type_to_species))
    ring_cfg = getattr(metrics, "rings", None)
    if ring_cfg is not None:
        for bp in list(getattr(ring_cfg, "bond_pairs", []) or []):
            out.extend(_selector_pair_to_keys(getattr(bp, "pair", None), type_to_species))
    for am in list(getattr(metrics, "angles", []) or []):
        try:
            a_sel, b_sel, c_sel = getattr(am, "triplet")
            out.extend(_selector_pair_to_keys([a_sel, b_sel], type_to_species))
            out.extend(_selector_pair_to_keys([b_sel, c_sel], type_to_species))
        except Exception:
            pass
    for pm in list(getattr(metrics, "pairs", []) or []):
        out.extend(_selector_pair_to_keys(getattr(pm, "pair", None), type_to_species))
    return sorted(set(out))




def _network_pairs_from_metrics(metrics: Any, type_to_species: Optional[Sequence[str]], *, extra: Optional[Sequence[Any]] = None) -> list[tuple[int, int]]:
    """Return pairs that define the primary network/backbone graph.

    These are expected-shell coordination pairs plus ring bond-pairs and the two
    edges implied by angle triplets.  Diagnostic pair requests (``metrics.pairs``)
    are intentionally excluded so that, for example, Si-Si/N-N close-contact
    descriptors do not become part of the backbone topology simply because they
    were requested for reporting.
    """

    out: list[tuple[int, int]] = []
    for cm in list(getattr(metrics, "coordinations", []) or []):
        if getattr(cm, "expected", None) is None and getattr(cm, "allowed", None) is None:
            continue
        out.extend(_selector_pair_to_keys([getattr(cm, "central"), getattr(cm, "neighbor")], type_to_species))
    ring_cfg = getattr(metrics, "rings", None)
    if ring_cfg is not None:
        for bp in list(getattr(ring_cfg, "bond_pairs", []) or []):
            out.extend(_selector_pair_to_keys(getattr(bp, "pair", None), type_to_species))
    for am in list(getattr(metrics, "angles", []) or []):
        try:
            a_sel, b_sel, c_sel = getattr(am, "triplet")
            out.extend(_selector_pair_to_keys([a_sel, b_sel], type_to_species))
            out.extend(_selector_pair_to_keys([b_sel, c_sel], type_to_species))
        except Exception:
            pass
    for ent in list(extra or []):
        out.extend(_selector_pair_to_keys(ent, type_to_species))
    return sorted(set(out))


def _pair_rows_filter(rows: Sequence[Mapping[str, Any]], keep: set[tuple[int, int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        pair = row.get("pair", None)
        if not (isinstance(pair, Sequence) and not isinstance(pair, (str, bytes, bytearray)) and len(pair) == 2):
            continue
        key = _pair_key(int(pair[0]), int(pair[1]))
        if key in keep:
            out.append(dict(row))
    return out


def _params_with_pair_subset(params: Mapping[str, Any], keep: set[tuple[int, int]], *, graph_family: str, graph_family_role: str) -> dict[str, Any]:
    p = dict(params or {})
    cut_rows = _pair_rows_filter(list(p.get("pair_cutoffs", p.get("cutoffs", [])) or []), keep)
    int_rows = _pair_rows_filter(list(p.get("pair_intervals", []) or []), keep)
    p["cutoffs"] = cut_rows
    p["pair_cutoffs"] = cut_rows
    p["pair_intervals"] = int_rows
    p["graph_family"] = str(graph_family)
    p["graph_family_role"] = str(graph_family_role)
    p.setdefault("graph_family_strategy", "network_and_candidate_contact_split")
    if len(cut_rows) == 1:
        p["cutoff"] = float(cut_rows[0]["cutoff"])
    else:
        p.pop("cutoff", None)
    if len(int_rows) == 1:
        p["r_min"] = float(int_rows[0]["r_min"])
        p["r_max"] = float(int_rows[0]["r_max"])
    else:
        p.pop("r_min", None)
        p.pop("r_max", None)
    return p


def _with_graph_family(rule: GraphRule, *, name_suffix: str, params: Mapping[str, Any], family: str, provenance_note: str) -> GraphRule:
    prov = rule.provenance
    if isinstance(prov, Mapping):
        prov = dict(prov)
        prov.setdefault("graph_family_note", provenance_note)
    return GraphRule(
        name=f"{rule.name}_{name_suffix}",
        kind=str(rule.kind),
        parameters=dict(params),
        provenance=prov,
    )


def split_adaptive_rules_by_graph_family(
    concrete: Sequence[GraphRule],
    intervals: Sequence[GraphRule],
    *,
    metrics: Any,
    type_to_species: Optional[Sequence[str]],
    parent_parameters: Optional[Mapping[str, Any]] = None,
) -> tuple[list[GraphRule], list[GraphRule]]:
    """Split adaptive all-pair RDF rules into graph families.

    The primary ``network_graph`` contains only expected-shell/ring/angle pairs
    and is used for backbone topology.  ``candidate_contact_graph`` retains the
    full RDF-derived candidate edge set and is used for homopolar/close-contact
    descriptors.  ``soft_ambiguity_graph`` is restricted to network pairs by
    default and is used only for soft coordination/ambiguity summaries.
    """

    pp = dict(parent_parameters or {})
    strategy = str(pp.get("graph_family_strategy", pp.get("family_strategy", "split"))).strip().lower()
    if strategy in {"legacy", "unified", "all_pair", "all_pairs", "none", "disabled"}:
        return list(concrete), list(intervals)

    raw_network = pp.get("network_pairs", pp.get("expected_shell_pairs", None))
    extra_network: list[Any] = []
    if raw_network is not None and not isinstance(raw_network, str):
        extra_network = list(raw_network or [])
    network_pairs = set(_network_pairs_from_metrics(metrics, type_to_species, extra=extra_network))
    if not network_pairs:
        return list(concrete), list(intervals)

    out_rules: list[GraphRule] = []
    out_intervals: list[GraphRule] = []
    for rule in concrete:
        params = dict(rule.parameters or {})
        if not bool(params.get("rdf_adaptive", False)) or params.get("graph_family", None) is not None:
            out_rules.append(rule)
            continue
        all_keys = set(pair_cutoffs_from_parameters(params).keys())
        keep_net = set(k for k in all_keys if k in network_pairs)
        if not keep_net:
            out_rules.append(rule)
            continue
        scope = str(params.get("graph_rule_scope", params.get("rule_scope", "per_structure")))
        params.setdefault("graph_rule_scope", scope)
        if str(rule.kind) == "soft_logistic":
            p_soft = _params_with_pair_subset(
                params,
                keep_net,
                graph_family="soft_ambiguity_graph",
                graph_family_role="soft_coordination_and_transition_shell_ambiguity",
            )
            # Keep the soft-search radius conservative after filtering.
            if p_soft.get("pair_intervals"):
                try:
                    max_hi = max(float(r["r_max"]) for r in p_soft["pair_intervals"])
                    sigma = float(p_soft.get("sigma", 0.0) or 0.0)
                    r0 = max(float(r["cutoff"]) for r in p_soft["pair_cutoffs"])
                    p_soft["r0"] = float(r0)
                    p_soft["search_radius"] = float(max(max_hi, r0 + 8.0 * sigma)) if sigma > 0 else float(max_hi)
                except Exception:
                    pass
            out_rules.append(_with_graph_family(rule, name_suffix="soft_ambiguity", params=p_soft, family="soft_ambiguity_graph", provenance_note="Soft graph restricted to network pairs for ambiguity reporting."))
            continue

        p_net = _params_with_pair_subset(
            params,
            keep_net,
            graph_family="network_graph",
            graph_family_role="primary_backbone_topology",
        )
        out_rules.append(_with_graph_family(rule, name_suffix="network", params=p_net, family="network_graph", provenance_note="Backbone graph contains expected-shell/ring/angle network pairs only."))

        # Candidate-contact graph keeps all RDF-derived pairs, including same-species
        # close contacts, but downstream metrics treat it as candidate-contact
        # evidence rather than the primary covalent topology.
        p_def = dict(params)
        p_def["graph_family"] = "candidate_contact_graph"
        p_def["graph_family_role"] = "homopolar_and_close_contact_candidates"
        p_def.setdefault("graph_family_strategy", "network_and_candidate_contact_split")
        out_rules.append(_with_graph_family(rule, name_suffix="candidate_contact", params=p_def, family="candidate_contact_graph", provenance_note="Candidate graph retains all RDF-derived pairs for homopolar/close-contact descriptors only."))

    for interval in intervals:
        params = dict(interval.parameters or {})
        if not bool(params.get("rdf_adaptive", False)) or params.get("graph_family", None) is not None:
            out_intervals.append(interval)
            continue
        all_keys = set(pair_intervals_from_parameters(params).keys())
        keep_net = set(k for k in all_keys if k in network_pairs)
        if not keep_net:
            out_intervals.append(interval)
            continue
        p_net = _params_with_pair_subset(
            params,
            keep_net,
            graph_family="network_graph",
            graph_family_role="primary_backbone_topology_interval",
        )
        out_intervals.append(_with_graph_family(interval, name_suffix="network", params=p_net, family="network_graph", provenance_note="Robust coordination interval restricted to expected-shell network pairs."))
    return out_rules, out_intervals


def graph_family_from_rule(rule: GraphRule) -> str:
    params = dict(getattr(rule, "parameters", {}) or {})
    fam = params.get("graph_family", None)
    if fam is not None and str(fam).strip():
        return str(fam)
    if bool(params.get("legacy", False)):
        return "legacy_single_cutoff_graph"
    if str(getattr(rule, "kind", "")) == "soft_logistic":
        return "soft_ambiguity_graph"
    return "unclassified_graph"

def _resolve_rdf_pairs(rule: GraphRule, frame: Any, metrics: Any, type_to_species: Optional[Sequence[str]]) -> list[tuple[int, int]]:
    params = dict(rule.parameters or {})
    out: list[tuple[int, int]] = []
    raw_pairs = params.get("pairs", None)
    raw_pair = params.get("pair", params.get("reference_pair", params.get("selector_pair", None)))
    if isinstance(raw_pairs, str) and raw_pairs.strip().lower() in {"from_metrics", "from_coordinations", "auto"}:
        out.extend(_pairs_from_metrics(metrics, type_to_species))
    elif raw_pairs is not None and isinstance(raw_pairs, Sequence) and not isinstance(raw_pairs, (str, bytes, bytearray)):
        for ent in raw_pairs:
            if isinstance(ent, Mapping):
                out.extend(_selector_pair_to_keys(ent.get("pair", None), type_to_species))
            else:
                out.extend(_selector_pair_to_keys(ent, type_to_species))
    if raw_pair is not None:
        out.extend(_selector_pair_to_keys(raw_pair, type_to_species))
    if not out:
        out.extend(_pairs_from_metrics(metrics, type_to_species))
    if not out:
        types = sorted(set(int(x) for x in np.asarray(getattr(frame, "types"), dtype=int).reshape(-1).tolist()))
        for i, a in enumerate(types):
            for b in types[i:]:
                out.append(_pair_key(a, b))
    return sorted(set(out))


def _auto_rdf_search_radius(frame: Any) -> float:
    cell = np.asarray(getattr(frame, "cell"), dtype=float)
    vol = abs(float(np.linalg.det(cell)))
    heights: list[float] = []
    for i in range(3):
        a = cell[(i + 1) % 3]
        b = cell[(i + 2) % 3]
        area = float(np.linalg.norm(np.cross(a, b)))
        if area > 0.0 and math.isfinite(area) and math.isfinite(vol):
            heights.append(vol / area)
    vals = [float(x) for x in heights if math.isfinite(float(x)) and float(x) > 0.0]
    if not vals:
        vals = [float(np.linalg.norm(v)) for v in cell if math.isfinite(float(np.linalg.norm(v))) and float(np.linalg.norm(v)) > 0.0]
    if not vals:
        raise ValueError("cannot determine automatic RDF search radius from cell")
    return max(0.25, 0.49 * min(vals))


def _pair_distances_with_indices(frame: Any, pair: tuple[int, int], r_search: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ii, jj, _vec, dist = _unique_pairs_with_distances(frame, float(r_search))
    if ii.size == 0:
        return ii, jj, np.asarray([], dtype=float)
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    a, b = int(pair[0]), int(pair[1])
    ti = types[ii]
    tj = types[jj]
    mask = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
    return ii[mask], jj[mask], np.asarray(dist[mask], dtype=float)


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    w = max(1, int(width))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def _partial_rdf_curve(frame: Any, pair: tuple[int, int], distances: np.ndarray, *, r_search: float, bin_width: float, smooth_width: float) -> dict[str, Any]:
    nbins = max(32, int(math.ceil(float(r_search) / float(bin_width))))
    edges = np.linspace(0.0, float(r_search), int(nbins) + 1)
    counts, _ = np.histogram(np.asarray(distances, dtype=float), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    n_a = int(np.sum(types == int(pair[0])))
    n_b = int(np.sum(types == int(pair[1])))
    volume = abs(float(np.linalg.det(np.asarray(getattr(frame, "cell"), dtype=float))))
    if volume <= 0.0 or not math.isfinite(volume) or n_a <= 0 or n_b <= 0:
        ideal = np.ones_like(shell_vol, dtype=float)
    elif int(pair[0]) == int(pair[1]):
        # Finite-N unordered same-type population is N(N-1)/2, not N^2/2.
        ideal = (0.5 * float(n_a) * float(n_a - 1) / volume) * shell_vol
    else:
        rho_b = float(n_b) / volume
        ideal = float(n_a) * rho_b * shell_vol
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ideal > 0.0, counts.astype(float) / ideal, 0.0)
    g[~np.isfinite(g)] = 0.0
    smooth_bins = max(1, int(round(float(smooth_width) / float(bin_width))))
    smooth = _moving_average(g, smooth_bins)
    return {"r": centers, "g": g, "smooth": smooth, "counts": counts.astype(float), "edges": edges, "nbins": int(nbins), "smooth_bins": int(smooth_bins)}




def _partial_rdf_curve_many(frames: Sequence[Any], pair: tuple[int, int], *, r_search: float, bin_width: float, smooth_width: float) -> tuple[dict[str, Any], np.ndarray]:
    """Partial RDF from an ensemble of structures using summed shell counts/ideals."""

    nbins = max(32, int(math.ceil(float(r_search) / float(bin_width))))
    edges = np.linspace(0.0, float(r_search), int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    counts_sum = np.zeros(int(nbins), dtype=float)
    ideal_sum = np.zeros(int(nbins), dtype=float)
    pooled: list[float] = []
    for frame in frames:
        _ii, _jj, dist = _pair_distances_with_indices(frame, pair, float(r_search))
        d = np.asarray(dist, dtype=float)
        d = d[np.isfinite(d) & (d > 1.0e-12) & (d <= float(r_search))]
        if d.size:
            counts, _ = np.histogram(d, bins=edges)
            counts_sum += counts.astype(float)
            pooled.extend([float(x) for x in d.tolist()])
        types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
        n_a = int(np.sum(types == int(pair[0])))
        n_b = int(np.sum(types == int(pair[1])))
        volume = abs(float(np.linalg.det(np.asarray(getattr(frame, "cell"), dtype=float))))
        if volume <= 0.0 or not math.isfinite(volume) or n_a <= 0 or n_b <= 0:
            ideal_sum += np.ones_like(shell_vol, dtype=float)
        elif int(pair[0]) == int(pair[1]):
            ideal_sum += (
                0.5 * float(n_a) * float(n_a - 1) / volume
            ) * shell_vol
        else:
            rho_b = float(n_b) / volume
            ideal_sum += float(n_a) * rho_b * shell_vol
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ideal_sum > 0.0, counts_sum / ideal_sum, 0.0)
    g[~np.isfinite(g)] = 0.0
    smooth_bins = max(1, int(round(float(smooth_width) / float(bin_width))))
    smooth = _moving_average(g, smooth_bins)
    return {"r": centers, "g": g, "smooth": smooth, "counts": counts_sum, "edges": edges, "nbins": int(nbins), "smooth_bins": int(smooth_bins)}, np.asarray(pooled, dtype=float)

def _local_extrema_indices(y: np.ndarray) -> tuple[list[int], list[int]]:
    yy = np.asarray(y, dtype=float)
    maxima: list[int] = []
    minima: list[int] = []
    for i in range(1, int(yy.size) - 1):
        if yy[i] >= yy[i - 1] and yy[i] >= yy[i + 1] and (yy[i] > yy[i - 1] or yy[i] > yy[i + 1]):
            maxima.append(int(i))
        if yy[i] <= yy[i - 1] and yy[i] <= yy[i + 1] and (yy[i] < yy[i - 1] or yy[i] < yy[i + 1]):
            minima.append(int(i))
    return maxima, minima


def _rdf_first_minimum(curve: Mapping[str, Any], distances: np.ndarray) -> dict[str, Any]:
    r = np.asarray(curve["r"], dtype=float)
    y = np.asarray(curve["smooth"], dtype=float)
    counts = np.asarray(curve["counts"], dtype=float)
    nz = np.where(counts > 0.0)[0]
    if nz.size == 0:
        raise ValueError("no pair distances found for RDF graph-rule derivation")
    first = max(1, int(nz[0]))
    maxima, minima = _local_extrema_indices(y)
    peak_candidates = [i for i in maxima if i >= first]
    if peak_candidates:
        ymax = float(np.nanmax(y[peak_candidates])) if peak_candidates else 0.0
        threshold = 0.15 * ymax
        peak_idx = next((i for i in peak_candidates if float(y[i]) >= threshold), peak_candidates[0])
    else:
        d_med = float(np.median(np.asarray(distances, dtype=float)))
        upto = np.where(r <= d_med)[0]
        peak_idx = int(upto[np.argmax(y[upto])]) if upto.size else int(np.argmax(y))
    next_peak_candidates = [i for i in maxima if i > int(peak_idx) + 1]
    if next_peak_candidates:
        next_peak_idx = int(next_peak_candidates[0])
        valley_candidates = [i for i in minima if int(peak_idx) < i < int(next_peak_idx)]
        if valley_candidates:
            min_idx = int(valley_candidates[int(np.argmin(y[valley_candidates]))])
        else:
            lo = min(int(peak_idx) + 1, int(y.size) - 1)
            hi = max(lo + 1, int(next_peak_idx))
            min_idx = int(lo + np.argmin(y[lo:hi])) if hi > lo else int(peak_idx)
    else:
        last = int(nz[-1])
        lo = min(int(peak_idx) + 1, int(y.size) - 1)
        hi = max(lo + 1, last + 1)
        min_idx = int(lo + np.argmin(y[lo:hi])) if hi > lo else int(peak_idx)
        next_peak_idx = None
    onset = float(r[min_idx])
    if next_peak_idx is not None:
        yv = float(y[min_idx])
        yp = float(y[next_peak_idx])
        rise = yv + 0.10 * max(0.0, yp - yv)
        for i in range(int(min_idx) + 1, int(next_peak_idx) + 1):
            if float(y[i]) >= rise:
                onset = float(r[i])
                break
        else:
            onset = float(r[next_peak_idx])
    return {
        "first_peak_r": float(r[peak_idx]),
        "first_peak_height": float(y[peak_idx]),
        "rdf_first_minimum": float(r[min_idx]),
        "rdf_first_minimum_height": float(y[min_idx]),
        "second_shell_onset": float(onset),
        "next_peak_r": (None if next_peak_idx is None else float(r[next_peak_idx])),
    }


def _coord_shell_objective(frame: Any, pair: tuple[int, int], metrics: Any, *, r_search: float, type_to_species: Optional[Sequence[str]]) -> dict[str, Any]:
    types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
    ii, jj, _vec, dist = _unique_pairs_with_distances(frame, float(r_search))
    nbr: list[list[tuple[int, float]]] = [[] for _ in range(int(getattr(frame, "n_atoms")))]
    for a, b, d in zip(ii.tolist(), jj.tolist(), dist.tolist()):
        if _pair_key(int(types[int(a)]), int(types[int(b)])) != _pair_key(pair[0], pair[1]):
            continue
        nbr[int(a)].append((int(b), float(d)))
        nbr[int(b)].append((int(a), float(d)))
    dz: list[float] = []
    dz1: list[float] = []
    n_central = 0
    metrics_used: list[str] = []
    for cm in list(getattr(metrics, "coordinations", []) or []):
        expected = getattr(cm, "expected", None)
        if expected is None:
            continue
        cset = set(int(x) for x in _resolve_selector(getattr(cm, "central"), type_to_species))
        nset = set(int(x) for x in _resolve_selector(getattr(cm, "neighbor"), type_to_species))
        metric_pairs = {_pair_key(a, b) for a in cset for b in nset}
        if _pair_key(pair[0], pair[1]) not in metric_pairs:
            continue
        z = int(expected)
        if z < 0:
            continue
        metrics_used.append(f"coord_{getattr(cm, 'central')}-{getattr(cm, 'neighbor')}")
        for i in range(int(getattr(frame, "n_atoms"))):
            if int(types[i]) not in cset:
                continue
            vals = sorted(float(d) for j, d in nbr[i] if int(types[int(j)]) in nset)
            n_central += 1
            if z == 0:
                dz.append(0.0)
            elif len(vals) >= z:
                dz.append(float(vals[z - 1]))
            if len(vals) >= z + 1:
                dz1.append(float(vals[z]))
    if not dz:
        return {"available": False, "reason": "no expected-coordination shell distances for this pair"}
    dz_arr = np.asarray(dz, dtype=float)
    dz_arr = dz_arr[np.isfinite(dz_arr)]
    dz1_arr = np.asarray(dz1, dtype=float) if dz1 else np.asarray([], dtype=float)
    dz1_arr = dz1_arr[np.isfinite(dz1_arr)]
    lower_all = float(np.max(dz_arr)) if dz_arr.size else float("nan")
    upper_all = float(np.min(dz1_arr)) if dz1_arr.size else float("nan")

    # Choose the graph boundary by shell-separability, not by a fixed chemistry
    # number and not by forcing every nominal z-th neighbour into the graph.  In
    # a smeared amorphous shell, max(d_z) can be driven by real defects and will
    # often pull accidental second-shell atoms inside the graph.  We therefore
    # scan observed z-th and (z+1)-th distances and minimise the explicit loss
    #
    #     loss(r) = under_fraction(d_z > r) + over_fraction(d_{z+1} <= r)
    #
    # with ties broken first by fewer accidental neighbours, then by the smallest
    # radius.  A connectivity floor, if requested, is applied after this objective
    # in _connectivity_lower_bound; its value is then recorded separately.
    candidates = sorted(set(float(x) for x in np.concatenate([dz_arr, dz1_arr]) if math.isfinite(float(x)) and float(x) > 0.0))
    if not candidates and math.isfinite(lower_all):
        candidates = [float(lower_all)]
    best_r = float(lower_all)
    best_loss = float("inf")
    best_under = float("nan")
    best_over = float("nan")
    best_accidental = float("nan")
    for r0 in candidates:
        under = float(np.sum(dz_arr > float(r0))) / float(max(1, dz_arr.size))
        over = (float(np.sum(dz1_arr <= float(r0))) / float(max(1, dz1_arr.size)) if dz1_arr.size else 0.0)
        loss = under + over
        if (
            loss < best_loss - 1.0e-12
            or (abs(loss - best_loss) <= 1.0e-12 and over < best_over - 1.0e-12)
            or (abs(loss - best_loss) <= 1.0e-12 and abs(over - best_over) <= 1.0e-12 and float(r0) < best_r)
        ):
            best_r = float(r0)
            best_loss = float(loss)
            best_under = float(under)
            best_over = float(over)
            best_accidental = float(over)

    return {
        "available": True,
        "metrics_used": metrics_used,
        "n_central": int(n_central),
        "d_z_max": lower_all,
        "d_z_p95": float(np.percentile(dz_arr, 95.0)) if dz_arr.size else None,
        "d_z_plus_1_min": (None if not math.isfinite(upper_all) else upper_all),
        "d_z_plus_1_p05": (None if dz1_arr.size == 0 else float(np.percentile(dz1_arr, 5.0))),
        "shell_objective_cutoff": float(best_r),
        "shell_objective": "minimise_under_plus_accidental_neighbour_fraction",
        "shell_objective_loss": float(best_loss),
        "shell_objective_under_fraction": float(best_under),
        "shell_objective_accidental_fraction": float(best_accidental),
        "shell_separable": bool(math.isfinite(lower_all) and (not math.isfinite(upper_all) or lower_all < upper_all)),
        "shell_gap": (None if not (math.isfinite(lower_all) and math.isfinite(upper_all)) else float(upper_all - lower_all)),
        "under_fraction_at_cutoff": float(best_under),
        "over_fraction_at_cutoff": float(best_over),
    }




def _coord_shell_objective_many(frames: Sequence[Any], pair: tuple[int, int], metrics: Any, *, r_search: float, type_to_species: Optional[Sequence[str]]) -> dict[str, Any]:
    dz: list[float] = []
    dz1: list[float] = []
    n_central = 0
    metrics_used_set: set[str] = set()
    for frame in frames:
        types = np.asarray(getattr(frame, "types"), dtype=int).reshape(-1)
        ii, jj, _vec, dist = _unique_pairs_with_distances(frame, float(r_search))
        nbr: list[list[tuple[int, float]]] = [[] for _ in range(int(getattr(frame, "n_atoms")))]
        for a, b, d in zip(ii.tolist(), jj.tolist(), dist.tolist()):
            if _pair_key(int(types[int(a)]), int(types[int(b)])) != _pair_key(pair[0], pair[1]):
                continue
            nbr[int(a)].append((int(b), float(d)))
            nbr[int(b)].append((int(a), float(d)))
        for cm in list(getattr(metrics, "coordinations", []) or []):
            expected = getattr(cm, "expected", None)
            if expected is None:
                continue
            cset = set(int(x) for x in _resolve_selector(getattr(cm, "central"), type_to_species))
            nset = set(int(x) for x in _resolve_selector(getattr(cm, "neighbor"), type_to_species))
            metric_pairs = {_pair_key(a, b) for a in cset for b in nset}
            if _pair_key(pair[0], pair[1]) not in metric_pairs:
                continue
            z = int(expected)
            if z < 0:
                continue
            metrics_used_set.add(f"coord_{getattr(cm, 'central')}-{getattr(cm, 'neighbor')}")
            for i in range(int(getattr(frame, "n_atoms"))):
                if int(types[i]) not in cset:
                    continue
                vals = sorted(float(d) for j, d in nbr[i] if int(types[int(j)]) in nset)
                n_central += 1
                if z == 0:
                    dz.append(0.0)
                elif len(vals) >= z:
                    dz.append(float(vals[z - 1]))
                if len(vals) >= z + 1:
                    dz1.append(float(vals[z]))
    if not dz:
        return {"available": False, "reason": "no expected-coordination shell distances for this pair"}
    dz_arr = np.asarray(dz, dtype=float)
    dz_arr = dz_arr[np.isfinite(dz_arr)]
    dz1_arr = np.asarray(dz1, dtype=float) if dz1 else np.asarray([], dtype=float)
    dz1_arr = dz1_arr[np.isfinite(dz1_arr)]
    lower_all = float(np.max(dz_arr)) if dz_arr.size else float("nan")
    upper_all = float(np.min(dz1_arr)) if dz1_arr.size else float("nan")
    candidates = sorted(set(float(x) for x in np.concatenate([dz_arr, dz1_arr]) if math.isfinite(float(x)) and float(x) > 0.0))
    if not candidates and math.isfinite(lower_all):
        candidates = [float(lower_all)]
    best_r = float(lower_all)
    best_loss = float("inf")
    best_under = float("nan")
    best_over = float("nan")
    best_accidental = float("nan")
    for r0 in candidates:
        under = float(np.sum(dz_arr > float(r0))) / float(max(1, dz_arr.size))
        over = (float(np.sum(dz1_arr <= float(r0))) / float(max(1, dz1_arr.size)) if dz1_arr.size else 0.0)
        loss = under + over
        if (
            loss < best_loss - 1.0e-12
            or (abs(loss - best_loss) <= 1.0e-12 and over < best_over - 1.0e-12)
            or (abs(loss - best_loss) <= 1.0e-12 and abs(over - best_over) <= 1.0e-12 and float(r0) < best_r)
        ):
            best_r = float(r0)
            best_loss = float(loss)
            best_under = float(under)
            best_over = float(over)
            best_accidental = float(over)
    return {
        "available": True,
        "scope": "ensemble",
        "metrics_used": sorted(metrics_used_set),
        "n_central": int(n_central),
        "d_z_max": lower_all,
        "d_z_p95": float(np.percentile(dz_arr, 95.0)) if dz_arr.size else None,
        "d_z_plus_1_min": (None if not math.isfinite(upper_all) else upper_all),
        "d_z_plus_1_p05": (None if dz1_arr.size == 0 else float(np.percentile(dz1_arr, 5.0))),
        "shell_objective_cutoff": float(best_r),
        "shell_objective": "ensemble_minimise_under_plus_accidental_neighbour_fraction",
        "shell_objective_loss": float(best_loss),
        "shell_objective_under_fraction": float(best_under),
        "shell_objective_accidental_fraction": float(best_accidental),
        "shell_separable": bool(math.isfinite(lower_all) and (not math.isfinite(upper_all) or lower_all < upper_all)),
        "shell_gap": (None if not (math.isfinite(lower_all) and math.isfinite(upper_all)) else float(upper_all - lower_all)),
        "under_fraction_at_cutoff": float(best_under),
        "over_fraction_at_cutoff": float(best_over),
    }

def _component_summary(n_nodes: int, edges: Sequence[tuple[int, int]]) -> tuple[int, float]:
    parent = list(range(int(n_nodes)))
    size = [1 for _ in range(int(n_nodes))]

    def find(x: int) -> int:
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in edges:
        union(int(a), int(b))
    roots: dict[int, int] = {}
    for i in range(int(n_nodes)):
        r = find(i)
        roots[r] = roots.get(r, 0) + 1
    return int(len(roots)), float(max(roots.values(), default=0)) / float(max(1, int(n_nodes)))


def _connectivity_lower_bound(frame: Any, pair: tuple[int, int], target_cutoff: float, *, r_search: float, min_lcc_fraction: Optional[float]) -> dict[str, Any]:
    ii, jj, dist = _pair_distances_with_indices(frame, pair, r_search)
    if ii.size == 0:
        return {"available": False, "reason": "no pair edges for connectivity lower bound"}
    target_edges = [(int(a), int(b)) for a, b, d in zip(ii.tolist(), jj.tolist(), dist.tolist()) if float(d) <= float(target_cutoff)]
    base_comp, base_lcc = _component_summary(int(getattr(frame, "n_atoms")), target_edges)
    required_lcc = float(min_lcc_fraction) if min_lcc_fraction is not None else max(0.0, float(base_lcc) - 1.0e-12)
    candidates = sorted(set(float(x) for x in dist.tolist() if float(x) <= float(target_cutoff)))
    selected = float(target_cutoff)
    selected_comp = int(base_comp)
    selected_lcc = float(base_lcc)
    for r0 in candidates:
        edges = [(int(a), int(b)) for a, b, d in zip(ii.tolist(), jj.tolist(), dist.tolist()) if float(d) <= float(r0)]
        comp, lcc = _component_summary(int(getattr(frame, "n_atoms")), edges)
        if int(comp) <= int(base_comp) and float(lcc) + 1.0e-12 >= float(required_lcc):
            selected = float(r0)
            selected_comp = int(comp)
            selected_lcc = float(lcc)
            break
    # If the RDF/shell-selected cutoff is below the requested connectivity
    # floor, raise it to the first distance at which that floor is met.  This is
    # the non-arbitrary lower bound: below it the induced graph no longer has the
    # same network connectivity.
    connectivity_threshold = None
    threshold_comp = None
    threshold_lcc = None
    if min_lcc_fraction is not None and float(base_lcc) + 1.0e-12 < float(required_lcc):
        for r0 in sorted(set(float(x) for x in dist.tolist() if math.isfinite(float(x)) and float(x) <= float(r_search))):
            edges = [(int(a), int(b)) for a, b, d in zip(ii.tolist(), jj.tolist(), dist.tolist()) if float(d) <= float(r0)]
            comp, lcc = _component_summary(int(getattr(frame, "n_atoms")), edges)
            if float(lcc) + 1.0e-12 >= float(required_lcc):
                connectivity_threshold = float(r0)
                threshold_comp = int(comp)
                threshold_lcc = float(lcc)
                selected = max(float(selected), float(r0))
                selected_comp = int(comp)
                selected_lcc = float(lcc)
                break
    return {
        "available": True,
        "connectivity_lower_bound": float(selected),
        "connectivity_threshold": connectivity_threshold,
        "requested_largest_component_fraction": float(required_lcc),
        "target_component_count": int(base_comp),
        "target_largest_component_fraction": float(base_lcc),
        "component_count_at_lower_bound": int(selected_comp),
        "largest_component_fraction_at_lower_bound": float(selected_lcc),
        "component_count_at_threshold": threshold_comp,
        "largest_component_fraction_at_threshold": threshold_lcc,
    }


def derive_rdf_adaptive_graph_rules(
    frame: Any,
    parent_rule: GraphRule,
    metrics: Any,
    *,
    box_id: Optional[int] = None,
    type_to_species: Optional[Sequence[str]] = None,
) -> tuple[list[GraphRule], list[GraphRule]]:
    """Expand an RDF-adaptive rule into concrete per-structure graph rules.

    The config names the induction algorithm, not an Angstrom cutoff.  The
    actual cutoff, sweep points, interval and soft midpoint are derived from the
    analysed structure's partial RDF plus expected-coordination shell ordering.
    """

    rule = GraphRule.from_any(parent_rule)
    params = dict(rule.parameters or {})
    pairs = _resolve_rdf_pairs(rule, frame, metrics, type_to_species)
    if not pairs:
        raise ValueError(f"rdf_adaptive graph rule {rule.name!r} could not resolve pair definitions")
    search_raw = params.get("search_radius", params.get("r_max", "auto"))
    if search_raw in (None, "") or (isinstance(search_raw, str) and search_raw.strip().lower() == "auto"):
        r_search = _auto_rdf_search_radius(frame)
    else:
        r_search = float(search_raw)
    if not (math.isfinite(r_search) and r_search > 0.0):
        raise ValueError(f"rdf_adaptive graph rule {rule.name!r} produced a non-finite RDF search radius")
    nbins_raw = params.get("nbins", params.get("n_bins", None))
    if nbins_raw not in (None, ""):
        try:
            nb = max(32, int(nbins_raw))
            bin_width = float(r_search) / float(nb)
        except Exception:
            bin_width = max(r_search / 1200.0, 1.0e-3)
    else:
        bin_width = float(params.get("bin_width", max(r_search / 1200.0, 1.0e-3)) or max(r_search / 1200.0, 1.0e-3))
    if not (math.isfinite(bin_width) and bin_width > 0.0):
        bin_width = max(r_search / 1200.0, 1.0e-3)
    smooth_bins_raw = params.get("smooth_bins", params.get("smooth", None))
    if smooth_bins_raw not in (None, ""):
        try:
            smooth_width = max(1.0, float(smooth_bins_raw)) * float(bin_width)
        except Exception:
            smooth_width = 7.0 * bin_width
    else:
        smooth_width = float(params.get("smooth_width", 7.0 * bin_width) or (7.0 * bin_width))
    points = max(2, int(params.get("points", params.get("n", 9)) or 9))
    mode = str(params.get("mode", "single_interval_soft")).strip().lower()
    include_sweep = mode in {"sweep", "sweep_only", "interval", "interval_only", "sweep_interval", "single_interval", "single_interval_soft", "sweep_interval_soft", "all"} or bool(params.get("include_sweep", False))
    include_soft = mode in {"soft", "soft_only", "soft_logistic", "single_interval_soft", "sweep_interval_soft", "all"} or bool(params.get("include_soft", False))
    include_single = mode not in {"sweep_only", "interval_only", "soft_only"}
    conn_raw = params.get("connectivity_fraction", params.get("target_largest_component_fraction", None))
    conn_frac = None if conn_raw in (None, "", "auto") else float(conn_raw)
    raw_conn_pairs = params.get("connectivity_pairs", params.get("network_pairs", "expected_shell_pairs"))
    explicit_conn_pairs: set[tuple[int, int]] = set()
    connectivity_shell_only = False
    if isinstance(raw_conn_pairs, str):
        token = raw_conn_pairs.strip().lower()
        connectivity_shell_only = token in {"expected_shell_pairs", "expected", "coordinated_pairs", "coordination_pairs"}
        if token in {"all", "all_pairs", "from_metrics"}:
            explicit_conn_pairs.update(pairs)
    elif raw_conn_pairs is not None and isinstance(raw_conn_pairs, Sequence) and not isinstance(raw_conn_pairs, (bytes, bytearray)):
        for ent in raw_conn_pairs:
            explicit_conn_pairs.update(_selector_pair_to_keys(ent, type_to_species))

    selected_cutoffs: dict[tuple[int, int], float] = {}
    intervals: dict[tuple[int, int], tuple[float, float]] = {}
    derivations: list[dict[str, Any]] = []
    for pair in pairs:
        ii, jj, distances = _pair_distances_with_indices(frame, pair, r_search)
        d = np.asarray(distances, dtype=float)
        d = d[np.isfinite(d) & (d > 1.0e-12) & (d <= float(r_search))]
        if d.size < 3:
            raise ValueError(f"rdf_adaptive graph rule {rule.name!r} found too few distances for pair {pair}")
        curve = _partial_rdf_curve(frame, pair, d, r_search=r_search, bin_width=bin_width, smooth_width=smooth_width)
        rdf = _rdf_first_minimum(curve, d)
        shell = _coord_shell_objective(frame, pair, metrics, r_search=r_search, type_to_species=type_to_species)
        rdf_min = float(rdf["rdf_first_minimum"])
        selected = rdf_min
        if bool(shell.get("available", False)):
            sh = float(shell.get("shell_objective_cutoff", rdf_min))
            if math.isfinite(sh) and sh > 0.0:
                # Use the explicit shell-separability objective.  This is a
                # per-structure modelling rule: minimise under-counting plus
                # accidental (z+1)-neighbour inclusion, with ties favouring fewer
                # accidental neighbours.  The RDF valley and ordered shell
                # distances remain in the provenance and interval bounds.
                selected = float(sh)
        shell_available = bool(shell.get("available", False))
        apply_conn = conn_frac is not None and (shell_available or _pair_key(pair[0], pair[1]) in explicit_conn_pairs or not connectivity_shell_only)
        conn = _connectivity_lower_bound(frame, pair, selected, r_search=r_search, min_lcc_fraction=(conn_frac if apply_conn else None))
        if conn_frac is not None and not apply_conn:
            conn = dict(conn)
            conn["connectivity_floor_skipped"] = True
            conn["skip_reason"] = "connectivity_fraction applies only to expected-shell/network pairs unless connectivity_pairs is explicit"
        if bool(conn.get("available", False)):
            c_lb = float(conn.get("connectivity_lower_bound", selected))
            if math.isfinite(c_lb) and c_lb > 0.0:
                selected = max(selected, c_lb)
        upper_candidates: list[float] = []
        if math.isfinite(float(rdf.get("second_shell_onset", float("nan")))):
            upper_candidates.append(float(rdf["second_shell_onset"]))
        if bool(shell.get("available", False)) and shell.get("d_z_plus_1_min", None) is not None:
            try:
                upper_candidates.append(float(shell["d_z_plus_1_min"]))
            except Exception:
                pass
        upper_candidates = [float(x) for x in upper_candidates if math.isfinite(float(x)) and float(x) >= selected]
        r_upper = min(upper_candidates) if upper_candidates else max(selected, rdf_min)
        r_lower = float(conn.get("connectivity_lower_bound", selected)) if bool(conn.get("available", False)) else selected
        r_lower = min(r_lower, selected)
        r_upper = max(r_upper, selected + bin_width)
        key = _pair_key(pair[0], pair[1])
        selected_cutoffs[key] = float(selected)
        intervals[key] = (float(r_lower), float(r_upper))
        derivations.append(
            {
                "pair": [int(key[0]), int(key[1])],
                "pair_species": [_species_label_for_type(key[0], type_to_species), _species_label_for_type(key[1], type_to_species)],
                "search_radius": float(r_search),
                "bin_width": float(bin_width),
                "smooth_width": float(smooth_width),
                "nbins": int(curve["nbins"]),
                **rdf,
                "shell_separability": shell,
                "connectivity": conn,
                "selected_cutoff": float(selected),
                "interval": {"r_min": float(r_lower), "r_max": float(r_upper), "points": int(points)},
            }
        )
    pair_cutoff_rows = [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(selected_cutoffs.items())]
    pair_interval_rows = [{"pair": [int(a), int(b)], "r_min": float(v[0]), "r_max": float(v[1])} for (a, b), v in sorted(intervals.items())]
    single_pair = len(pair_cutoff_rows) == 1
    base_params: dict[str, Any] = {
        "cutoffs": pair_cutoff_rows,
        "pair_cutoffs": pair_cutoff_rows,
        "pair_intervals": pair_interval_rows,
        "rdf_adaptive": True,
        "parent_rule_name": str(rule.name),
        "parent_rule_kind": str(rule.kind),
        "structure_hash": structure_hash(frame, type_to_species=type_to_species),
        "derivation_method": "partial_rdf_first_minimum_plus_minimum_shell_misassignment_and_connectivity_floor",
        "derivation": _json_safe(derivations),
    }
    if single_pair:
        base_params["cutoff"] = float(pair_cutoff_rows[0]["cutoff"])
        base_params["r_min"] = float(pair_interval_rows[0]["r_min"])
        base_params["r_max"] = float(pair_interval_rows[0]["r_max"])
    prov = {
        "source": "rdf_adaptive_graph_rule",
        "parent_rule": rule.to_json(),
        "box_id": None if box_id is None else int(box_id),
        "note": "Concrete graph rule derived for this structure from its pair distribution; no fixed YAML cutoff was used.",
    }
    concrete: list[GraphRule] = []
    if include_single:
        concrete.append(GraphRule(name=(f"{rule.name}_box{int(box_id):03d}_rdf" if box_id is not None else f"{rule.name}_rdf"), kind="hard_cutoff", parameters=base_params, provenance=prov))
    if include_sweep:
        for idx, t in enumerate(np.linspace(0.0, 1.0, int(points)).tolist(), start=1):
            rows: list[dict[str, Any]] = []
            for (a, b), (lo, hi) in sorted(intervals.items()):
                rows.append({"pair": [int(a), int(b)], "cutoff": float(lo + float(t) * (hi - lo))})
            p = dict(base_params)
            p.update({"cutoffs": rows, "pair_cutoffs": rows, "rdf_sweep_fraction": float(t), "interval_points": int(points)})
            if len(rows) == 1:
                p["cutoff"] = float(rows[0]["cutoff"])
            concrete.append(GraphRule(name=(f"{rule.name}_box{int(box_id):03d}_rdf_sweep_{idx:02d}" if box_id is not None else f"{rule.name}_rdf_sweep_{idx:02d}"), kind="hard_cutoff", parameters=p, provenance=prov))
    if include_soft and pair_cutoff_rows:
        widths = [abs(float(row["r_max"]) - float(row["r_min"])) for row in pair_interval_rows] if pair_interval_rows else [float(bin_width)]
        max_width = max(widths) if widths else float(bin_width)
        max_hi = max(float(row["r_max"]) for row in pair_interval_rows) if pair_interval_rows else max(float(row["cutoff"]) for row in pair_cutoff_rows)
        max_r0 = max(float(row["cutoff"]) for row in pair_cutoff_rows)
        if params.get("sigma", None) not in (None, ""):
            sigma = float(params.get("sigma"))
        else:
            frac_raw = params.get("soft_sigma_fraction", None)
            if frac_raw not in (None, "") and math.isfinite(float(max_width)):
                sigma = max(float(bin_width), float(max_width) * float(frac_raw))
            else:
                sigma = max(float(bin_width), float(max_width) / 4.0, float(smooth_width) / 2.0)
            sig_min = params.get("soft_sigma_min", None)
            sig_max = params.get("soft_sigma_max", None)
            if sig_min not in (None, ""):
                sigma = max(float(sigma), float(sig_min))
            if sig_max not in (None, ""):
                sigma = min(float(sigma), float(sig_max))
        if not (math.isfinite(float(sigma)) and float(sigma) > 0.0):
            sigma = max(float(bin_width), 1.0e-6)
        sp = dict(base_params)
        # build_soft_graph uses pair_cutoffs for pair-specific logistic
        # midpoints; r0 remains as a scalar fallback and CSV-friendly summary.
        sp.update({"r0": float(max_r0), "sigma": float(sigma), "search_radius": float(max(max_hi, max_r0 + 8.0 * sigma)), "min_weight": float(params.get("min_weight", 0.0) or 0.0)})
        concrete.append(GraphRule(name=(f"{rule.name}_box{int(box_id):03d}_rdf_soft" if box_id is not None else f"{rule.name}_rdf_soft"), kind="soft_logistic", parameters=sp, provenance=prov))
    interval_rule = GraphRule(name=(f"{rule.name}_box{int(box_id):03d}_rdf_interval" if box_id is not None else f"{rule.name}_rdf_interval"), kind="hard_cutoff_interval", parameters=base_params, provenance=prov)
    return concrete, [interval_rule]




def derive_rdf_adaptive_graph_rules_for_frames(
    frames: Sequence[Any],
    rule: GraphRule,
    metrics: Any,
    *,
    type_to_species: Optional[Sequence[str]] = None,
    label: str = "ensemble",
) -> tuple[list[GraphRule], list[GraphRule]]:
    """Derive one graph-rule family from the ensemble pair distribution.

    This is the ensemble counterpart of ``derive_rdf_adaptive_graph_rules``.  It
    uses pooled partial-RDF counts/ideals and pooled ordered-shell distances; the
    resulting concrete rule is then applied to every structure in the ensemble.
    """

    frames = list(frames or [])
    if not frames:
        return [], []
    rule = GraphRule.from_any(rule)
    params = dict(rule.parameters or {})
    pairs = _resolve_rdf_pairs(rule, frames[0], metrics, type_to_species)
    if not pairs:
        raise ValueError(f"rdf_adaptive ensemble graph rule {rule.name!r} could not resolve pair definitions")
    search_raw = params.get("search_radius", params.get("r_max", "auto"))
    if search_raw in (None, "", "auto"):
        r_search = min(float(_auto_rdf_search_radius(fr)) for fr in frames)
    else:
        r_search = float(search_raw)
    if not (math.isfinite(r_search) and r_search > 0.0):
        raise ValueError(f"rdf_adaptive ensemble graph rule {rule.name!r} produced a non-finite RDF search radius")
    bin_width = float(params.get("bin_width", 0.01) or 0.01)
    smooth_width = float(params.get("smooth_width", max(bin_width, 0.05)) or max(bin_width, 0.05))
    if bin_width <= 0.0:
        bin_width = 0.01
    if smooth_width <= 0.0:
        smooth_width = bin_width
    points = max(2, int(params.get("points", params.get("n", 9)) or 9))
    mode = str(params.get("mode", "all")).strip().lower()
    include_single = mode in {"all", "single", "primary", "hard", "hard_only"}
    include_sweep = mode in {"all", "sweep", "sweep_only", "interval", "hard_cutoff_sweep"}
    include_soft = mode in {"all", "soft", "soft_only", "soft_logistic"}
    if mode in {"interval", "interval_only"}:
        include_single = False
        include_sweep = True
        include_soft = False
    conn_raw = params.get("connectivity_fraction", None)
    if isinstance(conn_raw, str):
        conn_frac = None if conn_raw.strip().lower() in {"", "auto", "none", "null"} else float(conn_raw)
    else:
        conn_frac = None if conn_raw is None else float(conn_raw)
    raw_conn_pairs = params.get("connectivity_pairs", "expected_shell_pairs")
    connectivity_shell_only = True
    explicit_conn_pairs: set[tuple[int, int]] = set()
    if isinstance(raw_conn_pairs, str):
        token = raw_conn_pairs.strip().lower()
        connectivity_shell_only = token in {"expected_shell_pairs", "expected", "coordinated_pairs", "coordination_pairs"}
        if token in {"all", "all_pairs", "from_metrics"}:
            explicit_conn_pairs.update(pairs)
    elif raw_conn_pairs is not None and isinstance(raw_conn_pairs, Sequence) and not isinstance(raw_conn_pairs, (bytes, bytearray)):
        for ent in raw_conn_pairs:
            explicit_conn_pairs.update(_selector_pair_to_keys(ent, type_to_species))

    selected_cutoffs: dict[tuple[int, int], float] = {}
    intervals: dict[tuple[int, int], tuple[float, float]] = {}
    derivations: list[dict[str, Any]] = []
    for pair in pairs:
        curve, pooled = _partial_rdf_curve_many(frames, pair, r_search=r_search, bin_width=bin_width, smooth_width=smooth_width)
        d = np.asarray(pooled, dtype=float)
        d = d[np.isfinite(d) & (d > 1.0e-12) & (d <= float(r_search))]
        if d.size < 3:
            raise ValueError(f"rdf_adaptive ensemble graph rule {rule.name!r} found too few distances for pair {pair}")
        rdf = _rdf_first_minimum(curve, d)
        shell = _coord_shell_objective_many(frames, pair, metrics, r_search=r_search, type_to_species=type_to_species)
        rdf_min = float(rdf["rdf_first_minimum"])
        selected = rdf_min
        if bool(shell.get("available", False)):
            sh = float(shell.get("shell_objective_cutoff", rdf_min))
            if math.isfinite(sh) and sh > 0.0:
                selected = float(sh)
        shell_available = bool(shell.get("available", False))
        apply_conn = conn_frac is not None and (shell_available or _pair_key(pair[0], pair[1]) in explicit_conn_pairs or not connectivity_shell_only)
        conn_records: list[dict[str, Any]] = []
        conn_selected: list[float] = []
        for idx, fr in enumerate(frames, start=1):
            conn = _connectivity_lower_bound(fr, pair, selected, r_search=r_search, min_lcc_fraction=(conn_frac if apply_conn else None))
            if conn_frac is not None and not apply_conn:
                conn = dict(conn)
                conn["connectivity_floor_skipped"] = True
                conn["skip_reason"] = "connectivity_fraction applies only to expected-shell/network pairs unless connectivity_pairs is explicit"
            conn["ensemble_member_index"] = int(idx)
            conn_records.append(conn)
            if bool(conn.get("available", False)):
                c_lb = float(conn.get("connectivity_lower_bound", selected))
                if math.isfinite(c_lb) and c_lb > 0.0:
                    conn_selected.append(c_lb)
        if conn_selected:
            selected = max(float(selected), max(conn_selected))
        upper_candidates: list[float] = []
        if math.isfinite(float(rdf.get("second_shell_onset", float("nan")))):
            upper_candidates.append(float(rdf["second_shell_onset"]))
        if bool(shell.get("available", False)) and shell.get("d_z_plus_1_min", None) is not None:
            try:
                upper_candidates.append(float(shell["d_z_plus_1_min"]))
            except Exception:
                pass
        upper_candidates = [float(x) for x in upper_candidates if math.isfinite(float(x)) and float(x) >= selected]
        r_upper = min(upper_candidates) if upper_candidates else max(selected, rdf_min)
        r_lower = max(conn_selected) if conn_selected else selected
        r_lower = min(float(r_lower), float(selected))
        r_upper = max(float(r_upper), float(selected) + float(bin_width))
        key = _pair_key(pair[0], pair[1])
        selected_cutoffs[key] = float(selected)
        intervals[key] = (float(r_lower), float(r_upper))
        derivations.append(
            {
                "pair": [int(key[0]), int(key[1])],
                "pair_species": [_species_label_for_type(key[0], type_to_species), _species_label_for_type(key[1], type_to_species)],
                "scope": "ensemble",
                "ensemble_size": int(len(frames)),
                "search_radius": float(r_search),
                "bin_width": float(bin_width),
                "smooth_width": float(smooth_width),
                "nbins": int(curve["nbins"]),
                **rdf,
                "shell_separability": shell,
                "connectivity": {"per_structure": conn_records},
                "selected_cutoff": float(selected),
                "interval": {"r_min": float(r_lower), "r_max": float(r_upper), "points": int(points)},
            }
        )
    pair_cutoff_rows = [{"pair": [int(a), int(b)], "cutoff": float(c)} for (a, b), c in sorted(selected_cutoffs.items())]
    pair_interval_rows = [{"pair": [int(a), int(b)], "r_min": float(v[0]), "r_max": float(v[1])} for (a, b), v in sorted(intervals.items())]
    base_params: dict[str, Any] = {
        "cutoffs": pair_cutoff_rows,
        "pair_cutoffs": pair_cutoff_rows,
        "pair_intervals": pair_interval_rows,
        "rdf_adaptive": True,
        "parent_rule_name": str(rule.name),
        "parent_rule_kind": str(rule.kind),
        "graph_rule_scope": "ensemble",
        "ensemble_size": int(len(frames)),
        "derivation_method": "ensemble_partial_rdf_first_minimum_plus_pooled_shell_misassignment_and_connectivity_floor",
        "derivation": _json_safe(derivations),
    }
    if len(pair_cutoff_rows) == 1:
        base_params["cutoff"] = float(pair_cutoff_rows[0]["cutoff"])
        base_params["r_min"] = float(pair_interval_rows[0]["r_min"])
        base_params["r_max"] = float(pair_interval_rows[0]["r_max"])
    prov = {
        "source": "rdf_adaptive_ensemble_graph_rule",
        "parent_rule": rule.to_json(),
        "scope": "ensemble",
        "ensemble_label": str(label),
        "ensemble_size": int(len(frames)),
        "note": "Concrete graph rule derived once from the ensemble pair distribution and then applied to every structure.",
    }
    concrete: list[GraphRule] = []
    if include_single:
        concrete.append(GraphRule(name=f"{rule.name}_{label}_rdf", kind="hard_cutoff", parameters=base_params, provenance=prov))
    if include_sweep:
        for idx, t in enumerate(np.linspace(0.0, 1.0, int(points)).tolist(), start=1):
            rows: list[dict[str, Any]] = []
            for (a, b), (lo, hi) in sorted(intervals.items()):
                rows.append({"pair": [int(a), int(b)], "cutoff": float(lo + float(t) * (hi - lo))})
            p = dict(base_params)
            p.update({"cutoffs": rows, "pair_cutoffs": rows, "rdf_sweep_fraction": float(t), "interval_points": int(points)})
            if len(rows) == 1:
                p["cutoff"] = float(rows[0]["cutoff"])
            concrete.append(GraphRule(name=f"{rule.name}_{label}_rdf_sweep_{idx:02d}", kind="hard_cutoff", parameters=p, provenance=prov))
    if include_soft and pair_cutoff_rows:
        widths = [abs(float(row["r_max"]) - float(row["r_min"])) for row in pair_interval_rows] if pair_interval_rows else [float(bin_width)]
        max_width = max(widths) if widths else float(bin_width)
        max_hi = max(float(row["r_max"]) for row in pair_interval_rows) if pair_interval_rows else max(float(row["cutoff"]) for row in pair_cutoff_rows)
        max_r0 = max(float(row["cutoff"]) for row in pair_cutoff_rows)
        sigma = float(params.get("sigma")) if params.get("sigma", None) not in (None, "") else max(float(bin_width), float(max_width) / 4.0, float(smooth_width) / 2.0)
        if not (math.isfinite(float(sigma)) and float(sigma) > 0.0):
            sigma = max(float(bin_width), 1.0e-6)
        sp = dict(base_params)
        sp.update({"r0": float(max_r0), "sigma": float(sigma), "search_radius": float(max(max_hi, max_r0 + 8.0 * sigma)), "min_weight": float(params.get("min_weight", 0.0) or 0.0)})
        concrete.append(GraphRule(name=f"{rule.name}_{label}_rdf_soft", kind="soft_logistic", parameters=sp, provenance=prov))
    interval_rule = GraphRule(name=f"{rule.name}_{label}_rdf_interval", kind="hard_cutoff_interval", parameters=base_params, provenance=prov)
    concrete, intervals_out = split_adaptive_rules_by_graph_family(concrete, [interval_rule], metrics=metrics, type_to_species=type_to_species, parent_parameters=params)
    return concrete, intervals_out


def expand_graph_rules_for_frames(
    raw_rules: Sequence[Any],
    *,
    frames: Sequence[Any],
    metrics: Any,
    type_to_species: Optional[Sequence[str]] = None,
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
    label: str = "ensemble",
) -> tuple[list[GraphRule], list[GraphRule]]:
    """Expand graph rules once at ensemble scope."""

    frames = list(frames or [])
    if not frames:
        return [], []
    if not raw_rules:
        legacy = dict(legacy_cutoffs or {})
        if not legacy:
            return [], []
        legacy_rule = legacy_graph_rule_from_cutoffs(legacy)
        lp = dict(legacy_rule.parameters or {})
        lp.setdefault("graph_rule_scope", "ensemble")
        lp.setdefault("graph_family", "legacy_single_cutoff_graph")
        lp.setdefault("graph_family_role", "backward_compatible_single_cutoff_outputs")
        return [GraphRule(name=f"{legacy_rule.name}_{label}", kind=legacy_rule.kind, parameters=lp, provenance=legacy_rule.provenance)], []
    concrete: list[GraphRule] = []
    intervals: list[GraphRule] = []
    for idx, raw in enumerate(raw_rules or [], start=1):
        rule = GraphRule.from_any(raw, default_name=f"graph_rule_{idx}")
        params = dict(rule.parameters or {})
        derive = str(params.get("derive_from", "")).strip().lower()
        is_rdf = bool(rule.kind == "rdf_adaptive" or rule.kind.startswith("rdf_adaptive_") or derive in _RDF_DERIVE_VALUES)
        ensemble_enabled = params.get("ensemble", params.get("ensemble_scope", True))
        if isinstance(ensemble_enabled, str):
            ensemble_enabled = ensemble_enabled.strip().lower() not in {"false", "0", "no", "off", "disabled"}
        if is_rdf and bool(ensemble_enabled):
            adaptive_rule = GraphRule(name=rule.name, kind="rdf_adaptive", parameters=params, provenance=rule.provenance)
            rr, ii = derive_rdf_adaptive_graph_rules_for_frames(frames, adaptive_rule, metrics, type_to_species=type_to_species, label=label)
            concrete.extend(rr)
            intervals.extend(ii)
        elif not is_rdf:
            rr = expand_graph_rules([rule], legacy_cutoffs=None)
            for r in rr:
                p = dict(r.parameters or {})
                p.setdefault("graph_rule_scope", "ensemble")
                concrete.append(GraphRule(name=f"{r.name}_{label}", kind=r.kind, parameters=p, provenance=r.provenance))
            intervals.extend(interval_graph_rules([rule]))
    return concrete, intervals

def expand_graph_rules_for_frame(
    raw_rules: Sequence[Any],
    *,
    frame: Any,
    metrics: Any,
    box_id: Optional[int] = None,
    type_to_species: Optional[Sequence[str]] = None,
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
) -> tuple[list[GraphRule], list[GraphRule]]:
    """Expand possibly adaptive graph rules for one analysed structure."""

    if not raw_rules:
        legacy = dict(legacy_cutoffs or {})
        return ([legacy_graph_rule_from_cutoffs(legacy)] if legacy else []), []
    concrete: list[GraphRule] = []
    intervals: list[GraphRule] = []
    for idx, raw in enumerate(raw_rules or [], start=1):
        rule = GraphRule.from_any(raw, default_name=f"graph_rule_{idx}")
        params = dict(rule.parameters or {})
        derive = str(params.get("derive_from", "")).strip().lower()
        is_rdf = bool(rule.kind == "rdf_adaptive" or rule.kind.startswith("rdf_adaptive_") or derive in _RDF_DERIVE_VALUES)
        if is_rdf:
            # Normalise all adaptive aliases to the original rdf_adaptive
            # resolver.  The alias controls which concrete outputs are emitted
            # unless the user supplied an explicit mode.
            if "mode" not in params:
                if rule.kind in {"rdf_adaptive_hard_cutoff", "hard_cutoff"}:
                    params["mode"] = "single"
                elif rule.kind in {"rdf_adaptive_hard_cutoff_sweep", "hard_cutoff_sweep"}:
                    params["mode"] = "sweep"
                elif rule.kind in {"rdf_adaptive_hard_cutoff_interval", "hard_cutoff_interval"}:
                    params["mode"] = "interval"
                elif rule.kind in {"rdf_adaptive_soft_logistic", "soft_logistic"}:
                    params["mode"] = "soft_only"
            adaptive_rule = GraphRule(name=rule.name, kind="rdf_adaptive", parameters=params, provenance=rule.provenance)
            rr, ii = derive_rdf_adaptive_graph_rules(frame, adaptive_rule, metrics, box_id=box_id, type_to_species=type_to_species)
            rr, ii = split_adaptive_rules_by_graph_family(rr, ii, metrics=metrics, type_to_species=type_to_species, parent_parameters=params)
            concrete.extend(rr)
            intervals.extend(ii)
        else:
            concrete.extend(expand_graph_rules([rule], legacy_cutoffs=None))
            intervals.extend(interval_graph_rules([rule]))
    return concrete, intervals


def resolve_graph_rules_for_frame(
    frame: Any,
    metrics: Any,
    raw_rules: Sequence[Any],
    *,
    legacy_cutoffs: Optional[Mapping[Tuple[int, int], float]] = None,
    type_to_species: Optional[Sequence[str]] = None,
    box_id: Optional[int] = None,
) -> tuple[list[GraphRule], list[GraphRule], list[dict[str, Any]]]:
    """Resolve fixed and per-structure adaptive graph rules for one frame.

    The third return value is a compact JSON-ready audit record for adaptive
    rules.  It is redundant with each concrete graph rule's provenance but makes
    the top-level analysis JSON easier to inspect.
    """

    concrete, intervals = expand_graph_rules_for_frame(
        raw_rules,
        frame=frame,
        metrics=metrics,
        box_id=box_id,
        type_to_species=type_to_species,
        legacy_cutoffs=legacy_cutoffs,
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in list(concrete) + list(intervals):
        params = dict(rule.parameters or {})
        if not bool(params.get("rdf_adaptive", False)):
            continue
        rec = {
            "graph_rule_name": str(rule.name),
            "graph_rule_kind": str(rule.kind),
            "graph_rule_parameters": _json_safe(params),
            "graph_rule_provenance": _json_safe(rule.provenance),
        }
        key = json.dumps(rec, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            records.append(rec)
    return concrete, intervals, records
