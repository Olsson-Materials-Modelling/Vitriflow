from __future__ import annotations

"""Representation-provenance and strict numerical hygiene helpers.

The analysis layer treats descriptors as maps through an explicit
representation rule, e.g. ``x -> G_lambda(x) -> F_lambda(x)`` for graph
metrics or ``x -> void_field_phi(x) -> F_phi(x)`` for void metrics.  This
module centralises the lightweight schema helpers used by output-analysis,
auto-tune and production workflows without making graph rules the only kind of
representation.
"""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

NUMERICAL_STATUS_VALUES = {
    "ok",
    "unavailable",
    "not_applicable",
    "ill_conditioned",
    "insufficient_samples",
    "zero_variance",
    "nonfinite_input",
    "nonfinite_output",
    "failed",
    "legacy_only",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file.

    Provenance callers use content identities, never modification time alone,
    because HPC copies and archive extraction legitimately change timestamps.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provenance input is not a regular file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, recorded_path: Optional[str] = None) -> dict[str, Any]:
    """Return a portable, content-based file identity."""

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provenance input is not a regular file: {p}")
    return {
        "path": str(p if recorded_path is None else recorded_path),
        "filename": str(p.name),
        "size_bytes": int(p.stat().st_size),
        "sha256": sha256_file(p),
    }


def file_identity_matches(path: Path, identity: Mapping[str, Any]) -> bool:
    """Return whether ``path`` still matches a persisted file identity."""

    try:
        p = Path(path)
        expected_size = int(identity.get("size_bytes"))
        expected_sha = str(identity.get("sha256", "")).strip().lower()
        return (
            p.is_file()
            and int(p.stat().st_size) == expected_size
            and len(expected_sha) == 64
            and sha256_file(p).lower() == expected_sha
        )
    except (OSError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class RepresentationRule:
    """Generic intermediate-representation rule.

    GraphRule is a specialization of this schema, not the schema itself.
    """

    representation_rule_name: str
    representation_rule_kind: str
    representation_rule_parameters: Mapping[str, Any] = field(default_factory=dict)
    representation_rule_provenance: Any = "runtime"
    representation_rule_version: str = "v1"
    structure_hash: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return json_sanitize(
            {
                "representation_rule_name": str(self.representation_rule_name),
                "representation_rule_kind": str(self.representation_rule_kind),
                "representation_rule_parameters": dict(self.representation_rule_parameters or {}),
                "representation_rule_provenance": self.representation_rule_provenance,
                "representation_rule_version": str(self.representation_rule_version),
                "structure_hash": None if self.structure_hash is None else str(self.structure_hash),
            }
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, (bool, np.bool_))


def is_finite_number(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def finite_or_none(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def json_sanitize(value: Any) -> Any:
    """Return a strict JSON-compatible object with no NaN/Inf floats.

    Non-finite numeric values are represented as ``None``.  Metric rows should
    carry status/reason columns that explain why the value is unavailable; this
    function is the last safety net used before public JSON/CSV serialization.
    """

    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k in sorted(value.keys(), key=lambda kk: str(kk)):
            if isinstance(k, tuple):
                kk = "-".join(str(x) for x in k)
            else:
                kk = str(k)
            out[kk] = json_sanitize(value[k])
        return out
    if isinstance(value, (list, tuple)):
        return [json_sanitize(x) for x in value]
    return value


def assert_json_safe(value: Any, *, path: str = "$") -> None:
    """Raise if a raw non-finite value remains in a JSON payload."""

    if isinstance(value, np.ndarray):
        assert_json_safe(value.tolist(), path=path)
        return
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            assert_json_safe(v, path=f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_json_safe(v, path=f"{path}[{i}]")
        return


def json_dumps_strict(value: Any, *, indent: Optional[int] = 2, sort_keys: bool = True) -> str:
    payload = json_sanitize(value)
    assert_json_safe(payload)
    return json.dumps(payload, indent=indent, sort_keys=sort_keys, allow_nan=False)


def write_json_strict(path: Path, value: Any, *, indent: int = 2, sort_keys: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json_dumps_strict(value, indent=indent, sort_keys=sort_keys) + "\n"
    tmp_path: Path | None = None
    try:
        # A unique O_EXCL temporary file in the destination directory avoids
        # predictable-name symlink attacks and keeps os.replace on one
        # filesystem.  fsync the file and directory so an acknowledged
        # checkpoint survives a host crash as an old-or-new complete JSON file.
        fd, raw_tmp = tempfile.mkstemp(
            dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp", text=True
        )
        tmp_path = Path(raw_tmp)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, p)
        tmp_path = None
        try:
            dir_fd = os.open(str(p.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is unavailable on some supported filesystems;
            # atomic rename and file fsync still provide the core guarantee.
            pass
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def numerical_value_status(value: Any, *, unavailable_reason: str = "nonfinite_output") -> tuple[Any, str, str]:
    """Return (JSON value, status, reason) for a scalar metric value."""

    if value is None:
        return None, "unavailable", "value_not_available"
    if isinstance(value, (np.ndarray, list, tuple, Mapping)):
        # Array/dict-valued values are allowed only if finite after sanitization;
        # callers should add grid/length metadata separately where appropriate.
        cleaned = json_sanitize(value)
        try:
            assert_json_safe(cleaned)
        except Exception:
            return None, "nonfinite_output", unavailable_reason
        return cleaned, "ok", ""
    if _is_number(value):
        x = finite_or_none(value)
        if x is None:
            return None, "nonfinite_output", unavailable_reason
        return float(x), "ok", ""
    return json_sanitize(value), "ok", ""


def metric_units_for_name(metric_name: str) -> str:
    name = str(metric_name)
    if "angle" in name:
        return "degree" if not name.endswith("_count") else "count"
    if "bondlen" in name or "distance" in name or name.endswith("_r"):
        return "angstrom"
    if name.endswith("_fraction") or name.startswith("ring_frac_") or "fraction" in name:
        return "dimensionless_fraction"
    if name.endswith("_count") or name.endswith("_n_defective") or name.endswith("_cluster_count") or name.endswith("_size"):
        return "count"
    if name.startswith("coord_") or "coord" in name:
        return "neighbours"
    if "path_length" in name or "graph_path" in name:
        return "graph_edges"
    return "dimensionless"


def representation_kind_from_graph_rule_kind(kind: str) -> str:
    k = str(kind)
    if k == "hard_cutoff":
        return "hard_graph"
    if k == "hard_cutoff_sweep":
        return "hard_graph_sweep"
    if k == "hard_cutoff_interval":
        return "hard_graph_interval"
    if k == "soft_logistic":
        return "soft_graph"
    if "rdf_adaptive" in k:
        return "hard_graph"
    return f"graph_{k}" if not k.endswith("graph") else k


def representation_map_for_kind(kind: str) -> str:
    k = str(kind)
    if "graph" in k:
        return "graph_induction"
    if "void" in k:
        return "void_field"
    if "learned" in k or "embedding" in k:
        return "learned_embedding"
    return "direct_coordinate"


def graph_rule_to_representation_fields(rule: Any, *, structure_hash: str, graph_family: Optional[str] = None) -> dict[str, Any]:
    name = str(getattr(rule, "name", "graph_rule"))
    kind = str(getattr(rule, "kind", "hard_cutoff"))
    params = dict(getattr(rule, "parameters", {}) or {})
    prov = getattr(rule, "provenance", "runtime")
    rep_kind = representation_kind_from_graph_rule_kind(kind)
    rr = RepresentationRule(
        representation_rule_name=name,
        representation_rule_kind=rep_kind,
        representation_rule_parameters=params,
        representation_rule_provenance=prov,
        representation_rule_version=str(params.get("representation_rule_version", "v1")),
        structure_hash=str(structure_hash),
    ).to_json()
    rr.update(
        {
            "source_space": "structure",
            "representation_map": "graph_induction",
            "graph_family": None if graph_family is None else str(graph_family),
        }
    )
    return rr


def direct_coordinate_rule_fields(*, structure_hash: str, name: str = "coordinate_direct", parameters: Optional[Mapping[str, Any]] = None, provenance: Any = "vitriflow") -> dict[str, Any]:
    rr = RepresentationRule(
        representation_rule_name=name,
        representation_rule_kind="coordinate_direct",
        representation_rule_parameters=dict(parameters or {}),
        representation_rule_provenance=provenance,
        representation_rule_version="v1",
        structure_hash=str(structure_hash),
    ).to_json()
    rr.update({"source_space": "structure", "representation_map": "direct_coordinate"})
    return rr


def void_rule_fields(*, structure_hash: str, name: str, kind: str, parameters: Mapping[str, Any], provenance: Any = "vitriflow") -> dict[str, Any]:
    rr = RepresentationRule(
        representation_rule_name=name,
        representation_rule_kind=str(kind),
        representation_rule_parameters=dict(parameters or {}),
        representation_rule_provenance=provenance,
        representation_rule_version="v1",
        structure_hash=str(structure_hash),
    ).to_json()
    rr.update({"source_space": "structure", "representation_map": "void_field"})
    return rr


def metric_result_row(
    *,
    box_id: int,
    structure_hash: str,
    material_id: str,
    metric_family: str,
    metric_name: str,
    metric_value: Any,
    representation_fields: Mapping[str, Any],
    metric_units: Optional[str] = None,
    status: Optional[str] = None,
    status_reason: Optional[str] = None,
    n_samples: Optional[int] = None,
    normalization: Optional[str] = None,
) -> dict[str, Any]:
    value, num_status, num_reason = numerical_value_status(metric_value)
    metric_status = str(status or ("ok" if num_status == "ok" else "unavailable"))
    metric_status_reason = str(status_reason or ("" if metric_status == "ok" else num_reason))
    row = {
        "box_id": int(box_id),
        "structure_hash": str(structure_hash),
        "material_id": str(material_id or "unknown"),
        "metric_family": str(metric_family),
        "metric_name": str(metric_name),
        "metric_value": value,
        "metric_units": str(metric_units or metric_units_for_name(metric_name)),
        "metric_status": metric_status,
        "metric_status_reason": metric_status_reason,
        "numerical_status": str(num_status),
        "numerical_status_reason": str(num_reason),
        "uncertainty_status": "not_applicable",
        "uncertainty_status_reason": "per_structure_metric",
        "descriptor_map": str(metric_name),
        "target_distribution": "graph-conditioned descriptor distribution" if representation_fields.get("representation_map") == "graph_induction" else "coordinate descriptor distribution",
        "n_samples": None if n_samples is None else int(n_samples),
        "normalization": None if normalization is None else str(normalization),
    }
    row.update(json_sanitize(dict(representation_fields)))
    return row


def csv_scalar(value: Any) -> Any:
    """CSV scalar serializer: empty for null/nonfinite, JSON for containers."""

    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(json_sanitize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
