"""Versioned provenance for engine-neutral per-stage artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

from ..analysis.provenance import file_identity, file_identity_matches, write_json_strict
from ..lammps_units import (
    CANONICAL_REPORTING_CONTRACT,
    canonical_reporting_units,
    normalize_lammps_units_style,
)


STAGE_ARTIFACT_MANIFEST_NAME = "stage_artifacts.json"
STAGE_ARTIFACT_SCHEMA = "vitriflow.stage_artifacts.v1"


_LAMMPS_NATIVE_SOURCE_UNITS: dict[str, dict[str, str]] = {
    "metal": {
        "length": "angstrom",
        "volume": "angstrom^3",
        "time": "ps",
        "energy": "eV",
        "pressure": "bar",
        "density": "g/cm^3",
        "temperature": "K",
        "msd": "angstrom^2",
    },
    "real": {
        "length": "angstrom",
        "volume": "angstrom^3",
        "time": "fs",
        "energy": "kcal/mol",
        "pressure": "atm",
        "density": "g/cm^3",
        "temperature": "K",
        "msd": "angstrom^2",
    },
    "electron": {
        "length": "bohr",
        "volume": "bohr^3",
        "time": "fs",
        "energy": "hartree",
        "pressure": "Pa",
        "density": "amu/bohr^3",
        "temperature": "K",
        "msd": "bohr^2",
    },
    "nano": {
        "length": "nm",
        "volume": "nm^3",
        "time": "ns",
        "energy": "ag*nm^2/ns^2",
        "pressure": "ag/(nm*ns^2)",
        "density": "ag/nm^3",
        "temperature": "K",
        "msd": "nm^2",
    },
    "si": {
        "length": "m",
        "volume": "m^3",
        "time": "s",
        "energy": "J",
        "pressure": "Pa",
        "density": "kg/m^3",
        "temperature": "K",
        "msd": "m^2",
    },
    "cgs": {
        "length": "cm",
        "volume": "cm^3",
        "time": "s",
        "energy": "erg",
        "pressure": "dyne/cm^2",
        "density": "g/cm^3",
        "temperature": "K",
        "msd": "cm^2",
    },
    "micro": {
        "length": "um",
        "volume": "um^3",
        "time": "us",
        "energy": "pg*um^2/us^2",
        "pressure": "pg/(um*us^2)",
        "density": "pg/um^3",
        "temperature": "K",
        "msd": "um^2",
    },
}


def _native_source_units(
    *, engine: str, lammps_units_style: Optional[str]
) -> dict[str, str]:
    engine_name = str(engine or "").strip().lower()
    if engine_name == "lammps":
        units_style = normalize_lammps_units_style(str(lammps_units_style or ""))
        return {
            "system": "lammps",
            "lammps_units_style": units_style,
            **_LAMMPS_NATIVE_SOURCE_UNITS[units_style],
        }
    if engine_name == "cp2k":
        # These describe the CP2K files consumed by stage_runner.  Potential
        # energy is read from .ener in hartree and pressure from output in bar;
        # trajectory geometry is in angstrom and its derived MSD in angstrom^2.
        return {
            "system": "cp2k",
            "length": "angstrom",
            "volume": "angstrom^3",
            "time": "fs",
            "energy": "hartree",
            "pressure": "bar",
            "density": "g/cm^3",
            "temperature": "K",
            "msd": "angstrom^2",
        }
    raise ValueError(f"Unsupported stage-artifact engine {engine!r}")


def _artifact_record(path: Path, *, kind: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {
            "path": p.name,
            "available": False,
            "canonicalized": False,
            "size_bytes": None,
            "sha256": None,
            "validation_error": "file_missing",
        }
    identity = file_identity(p, recorded_path=p.name)
    available = False
    validation_error: Optional[str] = None
    if int(identity["size_bytes"]) > 0:
        try:
            from .thermo import parse_msd_csv, parse_thermo_csv

            if kind == "thermo_csv":
                parse_thermo_csv(p)
            elif kind == "msd_csv":
                parse_msd_csv(p)
            else:  # pragma: no cover
                raise ValueError(f"Unsupported stage artifact kind {kind!r}")
            available = True
        except (OSError, UnicodeError, TypeError, ValueError):
            # Keep the content identity for audit, but do not grant a malformed
            # or placeholder file the canonical-unit authority.
            available = False
            # Keep the manifest deterministic and path-independent.  Detailed
            # parser exceptions remain available to direct callers, while the
            # provenance record carries a stable machine-readable blocker.
            validation_error = f"strict_{kind}_validation_failed"
    else:
        validation_error = "empty_file"
    return {
        **identity,
        "available": bool(available),
        "canonicalized": bool(available),
        "validation_error": validation_error,
    }


def build_stage_artifact_manifest(
    *,
    engine: str,
    timestep_ps: float,
    thermo_csv: Path,
    msd_csv: Path,
    lammps_units_style: Optional[str] = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for canonical per-stage CSV files."""

    engine_name = str(engine or "").strip().lower()
    dt_ps = float(timestep_ps)
    if not (math.isfinite(dt_ps) and dt_ps > 0.0):
        raise ValueError("stage artifact timestep_ps must be finite and > 0")

    report_units = dict(canonical_reporting_units())
    contract = str(report_units.pop("reporting_contract"))
    if contract != CANONICAL_REPORTING_CONTRACT:  # pragma: no cover
        raise RuntimeError("canonical reporting contract constant is internally inconsistent")

    return {
        "schema": STAGE_ARTIFACT_SCHEMA,
        "schema_version": 1,
        "reporting_contract": CANONICAL_REPORTING_CONTRACT,
        "engine": engine_name,
        "native_source_units": _native_source_units(
            engine=engine_name,
            lammps_units_style=lammps_units_style,
        ),
        "canonical_reporting_units": report_units,
        "timestep_ps": dt_ps,
        "step_semantics": "time_ps = Step * timestep_ps",
        "step_domain": "nonnegative_strictly_increasing_integer_counts",
        "artifacts": {
            "thermo_csv": _artifact_record(Path(thermo_csv), kind="thermo_csv"),
            "msd_csv": _artifact_record(Path(msd_csv), kind="msd_csv"),
        },
    }


def write_stage_artifact_manifest(
    stage_dir: Path,
    *,
    engine: str,
    timestep_ps: float,
    thermo_csv: Path,
    msd_csv: Path,
    lammps_units_style: Optional[str] = None,
) -> Path:
    """Atomically publish a stage manifest after its neutral CSV artifacts."""

    stage_path = Path(stage_dir)
    thermo_path = Path(thermo_csv)
    msd_path = Path(msd_csv)
    for artifact_path, expected_name in (
        (thermo_path, "thermo.csv"),
        (msd_path, "msd.csv"),
    ):
        if artifact_path.name != expected_name or artifact_path.parent.resolve() != stage_path.resolve():
            raise ValueError(
                f"Stage artifact manifest requires {expected_name} directly under {stage_path}, "
                f"got {artifact_path}"
            )
    manifest_path = stage_path / STAGE_ARTIFACT_MANIFEST_NAME
    manifest = build_stage_artifact_manifest(
        engine=engine,
        timestep_ps=timestep_ps,
        thermo_csv=thermo_path,
        msd_csv=msd_path,
        lammps_units_style=lammps_units_style,
    )
    write_json_strict(manifest_path, manifest, indent=2, sort_keys=True)
    return manifest_path


def load_stage_artifact_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the versioned unit/time contract used by plotting."""

    import json

    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid stage artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid stage artifact manifest {manifest_path}: root must be an object")
    data = dict(raw)
    if data.get("schema") != STAGE_ARTIFACT_SCHEMA or data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported stage artifact manifest schema in {manifest_path}: "
            f"{data.get('schema')!r} version {data.get('schema_version')!r}"
        )
    if data.get("reporting_contract") != CANONICAL_REPORTING_CONTRACT:
        raise ValueError(f"Unsupported reporting contract in {manifest_path}")
    engine = str(data.get("engine", "")).strip().lower()
    if engine not in {"lammps", "cp2k"}:
        raise ValueError(f"Invalid stage artifact engine in {manifest_path}: {engine!r}")
    try:
        dt_ps = float(data.get("timestep_ps"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestep_ps in {manifest_path}") from exc
    if not (math.isfinite(dt_ps) and dt_ps > 0.0):
        raise ValueError(f"Invalid timestep_ps in {manifest_path}")
    expected_units = dict(canonical_reporting_units())
    expected_units.pop("reporting_contract", None)
    if data.get("canonical_reporting_units") != expected_units:
        raise ValueError(f"Invalid canonical reporting units in {manifest_path}")
    native_units = data.get("native_source_units")
    if not isinstance(native_units, Mapping) or str(native_units.get("system", "")) != engine:
        raise ValueError(f"Invalid native source units in {manifest_path}")
    try:
        expected_native_units = _native_source_units(
            engine=engine,
            lammps_units_style=(
                str(native_units.get("lammps_units_style", ""))
                if engine == "lammps"
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid native source units in {manifest_path}") from exc
    if dict(native_units) != expected_native_units:
        raise ValueError(f"Invalid native source units in {manifest_path}")
    if data.get("step_semantics") != "time_ps = Step * timestep_ps":
        raise ValueError(f"Invalid step semantics in {manifest_path}")
    if data.get("step_domain") != "nonnegative_strictly_increasing_integer_counts":
        raise ValueError(f"Invalid step domain in {manifest_path}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Invalid artifact records in {manifest_path}")
    data["engine"] = engine
    data["timestep_ps"] = dt_ps
    return data


def verify_manifest_artifact(
    *, stage_dir: Path, manifest: Mapping[str, Any], artifact_key: str
) -> bool:
    """Verify a manifest-bound artifact; return whether usable data exist."""

    records = manifest.get("artifacts", {})
    record = records.get(str(artifact_key)) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"Stage manifest has no {artifact_key!r} artifact record")
    rel = str(record.get("path", ""))
    if not rel or Path(rel).name != rel:
        raise ValueError(f"Stage manifest has an unsafe {artifact_key!r} path")
    path = Path(stage_dir) / rel
    available = bool(record.get("available", False))
    canonicalized = bool(record.get("canonicalized", False))
    if available != canonicalized:
        raise ValueError(f"Stage artifact availability/canonicalization mismatch for {path}")
    has_identity = (
        isinstance(record.get("size_bytes"), int)
        and isinstance(record.get("sha256"), str)
        and len(str(record.get("sha256"))) == 64
    )
    if not available and not has_identity:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Unbound stage artifact appeared after manifest publication: {path}")
        return False
    if not has_identity or not path.is_file() or not file_identity_matches(path, record):
        raise ValueError(f"Stage artifact identity mismatch for {path}")
    return available
