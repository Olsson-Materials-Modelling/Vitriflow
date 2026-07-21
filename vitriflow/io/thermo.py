from __future__ import annotations

"""Engine-neutral scalar time series I/O.

The analysis pipeline should not depend on a specific engine's native log format.
This module provides simple, explicit tabular files for thermodynamics and MSD.

Conventions
-----------
Thermo table:
  - file: thermo.csv
  - header row: comma-separated column names
  - numeric rows: comma-separated floats

MSD series:
  - file: msd.csv
  - columns: Step,MSD
"""

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..parse import ThermoTable


def _validate_step_axis(step: np.ndarray, *, source: str) -> np.ndarray:
    """Return a validated engine step axis.

    Engine-neutral stage artifacts use step counts, not arbitrary sample
    coordinates.  Fractional, negative, repeated, or decreasing values would
    make the manifest time contract ambiguous and therefore fail closed.
    """

    arr = np.asarray(step, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{source}: Step must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source}: Step contains non-finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{source}: Step must be nonnegative")
    if not np.all(arr == np.floor(arr)):
        raise ValueError(f"{source}: Step must contain integer counts")
    if arr.size > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{source}: Step must be strictly increasing with no duplicates")
    return arr


def _validated_thermo_table(table: ThermoTable, *, source: str) -> ThermoTable:
    cols = [str(value).strip() for value in list(table.columns)]
    if not cols or any(not value for value in cols) or len(set(cols)) != len(cols):
        raise ValueError(f"{source}: thermo columns must be non-empty and unique")
    if "Step" not in cols:
        raise ValueError(f"{source}: thermo columns must include Step")
    data = np.asarray(table.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(cols) or data.shape[0] < 1:
        raise ValueError(f"{source}: thermo data must be a non-empty rectangular 2D table")
    if np.any(np.isinf(data)):
        raise ValueError(f"{source}: thermo data contains infinite values")
    step_index = cols.index("Step")
    _validate_step_axis(data[:, step_index], source=source)
    metric_indices = [idx for idx in range(len(cols)) if idx != step_index]
    if not metric_indices:
        raise ValueError(f"{source}: thermo table has no metric columns")
    metric_data = data[:, metric_indices]
    if not np.all(np.any(np.isfinite(metric_data), axis=1)):
        raise ValueError(f"{source}: every thermo row must contain at least one finite metric")
    return ThermoTable(columns=cols, data=data)


def write_thermo_csv(path: Path, table: ThermoTable) -> None:
    """Write a strictly validated engine-neutral thermodynamic table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    validated = _validated_thermo_table(table, source=str(p))
    cols = list(validated.columns)
    data = np.asarray(validated.data, dtype=float)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in data.tolist():
            w.writerow([float(x) for x in row])


def parse_thermo_csv(path: Path, *, legacy_tolerant: bool = False) -> ThermoTable:
    """Parse an engine-neutral thermo CSV.

    Current artifacts are strict by default: headers are unique, rows are
    rectangular and numeric, and ``Step`` is a nonnegative strictly increasing
    integer axis.  ``legacy_tolerant=True`` preserves the pre-0.4.31
    pad/truncate/drop behavior only for callers intentionally importing old
    non-authoritative files.
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8", errors=("replace" if legacy_tolerant else "strict")) as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration as e:
            raise ValueError(f"Empty thermo CSV: {p}") from e
        if legacy_tolerant:
            cols = [str(x).strip() for x in header if str(x).strip() != ""]
            if not cols:
                raise ValueError(f"Invalid thermo CSV header: {p}")
        else:
            cols = [str(x).strip() for x in header]
            if not cols or any(not value for value in cols) or len(set(cols)) != len(cols):
                raise ValueError(f"Invalid blank or duplicate thermo CSV header: {p}")
        rows: list[list[float]] = []
        for row_number, row in enumerate(r, start=2):
            if not row:
                if legacy_tolerant:
                    continue
                raise ValueError(f"Blank thermo CSV row {row_number} in {p}")
            if legacy_tolerant and len(row) < len(cols):
                row = list(row) + ["nan"] * (len(cols) - len(row))
            elif not legacy_tolerant and len(row) != len(cols):
                raise ValueError(
                    f"Ragged thermo CSV row {row_number} in {p}: "
                    f"expected {len(cols)} fields, found {len(row)}"
                )
            try:
                rows.append([float(x) for x in row[: len(cols)]])
            except (TypeError, ValueError) as exc:
                if legacy_tolerant:
                    continue
                raise ValueError(f"Non-numeric thermo CSV row {row_number} in {p}") from exc
    if not rows:
        raise ValueError(f"No numeric rows parsed from thermo CSV: {p}")
    data = np.asarray(rows, dtype=float)
    table = ThermoTable(columns=list(cols), data=data)
    return table if legacy_tolerant else _validated_thermo_table(table, source=str(p))


def write_msd_csv(path: Path, *, step: Sequence[float], msd: Sequence[float]) -> None:
    """Write a strictly validated engine-neutral MSD series."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    step = np.asarray(step, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if step.ndim != 1 or msd.ndim != 1 or step.size != msd.size:
        raise ValueError("step and msd must be 1D arrays of equal length")
    if step.size < 3:
        raise ValueError("MSD series requires at least three rows")
    _validate_step_axis(step, source=str(p))
    if not np.all(np.isfinite(msd)):
        raise ValueError(f"{p}: MSD contains non-finite values")
    if np.any(msd < 0.0):
        raise ValueError(f"{p}: MSD must be nonnegative")
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Step", "MSD"])
        for s, v in zip(step.tolist(), msd.tolist()):
            w.writerow([float(s), float(v)])


def parse_msd_csv(
    path: Path,
    *,
    legacy_tolerant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse an engine-neutral MSD CSV with strict physical validation.

    ``legacy_tolerant=True`` is an explicit compatibility path for old files
    whose malformed rows were historically dropped.  It is never used for a
    manifest-bound authoritative artifact.
    """
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8", errors=("replace" if legacy_tolerant else "strict")) as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration as e:
            raise ValueError(f"Empty MSD CSV: {p}") from e
        clean_header = [str(value).strip() for value in header]
        if not legacy_tolerant and clean_header != ["Step", "MSD"]:
            raise ValueError(f"Invalid MSD CSV header in {p}; expected Step,MSD")
        rows: list[tuple[float, float]] = []
        for row_number, row in enumerate(r, start=2):
            if not legacy_tolerant and len(row) != 2:
                raise ValueError(
                    f"Malformed MSD CSV row {row_number} in {p}: expected 2 fields, found {len(row)}"
                )
            if legacy_tolerant and len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except (IndexError, TypeError, ValueError) as exc:
                if legacy_tolerant:
                    continue
                raise ValueError(f"Non-numeric MSD CSV row {row_number} in {p}") from exc
    if len(rows) < 3:
        raise ValueError(f"Too few MSD rows in {p}")
    step = np.asarray([x[0] for x in rows], dtype=float)
    msd = np.asarray([x[1] for x in rows], dtype=float)
    if not legacy_tolerant:
        _validate_step_axis(step, source=str(p))
        if not np.all(np.isfinite(msd)):
            raise ValueError(f"{p}: MSD contains non-finite values")
        if np.any(msd < 0.0):
            raise ValueError(f"{p}: MSD must be nonnegative")
    return step, msd
