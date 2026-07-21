from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_number(token: str) -> bool:
    return bool(_FLOAT_RE.match(token))


@dataclass(frozen=True)
class ThermoTable:
    columns: list[str]
    data: np.ndarray  # shape nrows ncols

    def as_dict(self) -> dict[str, np.ndarray]:
        return {c: self.data[:, i] for i, c in enumerate(self.columns)}


def _validated_thermo_table(
    columns: list[str], rows: list[list[float]], *, source: Path
) -> ThermoTable:
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Malformed thermo table dimensions in {source}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Thermo table contains non-finite values in {source}")
    if not columns or columns[0] != "Step":
        raise ValueError(f"Thermo table must begin with a Step column in {source}")
    steps = data[:, 0]
    if np.any(steps < 0.0) or np.any(steps != np.floor(steps)):
        raise ValueError(f"Thermo Step values must be nonnegative integers in {source}")
    if steps.size > 1 and np.any(np.diff(steps) <= 0.0):
        raise ValueError(f"Thermo Step values must be strictly increasing in {source}")
    return ThermoTable(columns=columns, data=data)


def parse_last_thermo_table(log_path: Path) -> ThermoTable:
    """Last thermo table."""
    lines = log_path.read_text(errors="replace").splitlines()
    header_idx = None
    header_cols: list[str] = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Step "):
            # candidate header
            cols = ln.split()
            if len(cols) >= 2:
                header_idx = i
                header_cols = cols

    if header_idx is None:
        raise ValueError(f"No thermo table found in log: {log_path}")

    rows: list[list[float]] = []
    ncols = len(header_cols)
    for ln in lines[header_idx + 1 :]:
        toks = ln.split()
        if len(toks) != ncols:
            # end table
            if rows:
                break
            else:
                continue
        if not all(_is_number(t) for t in toks):
            nonfinite_literal = any(
                t.strip().lower().lstrip("+-") in {"nan", "inf", "infinity"}
                for t in toks
            )
            if len(toks) == ncols and (nonfinite_literal or _is_number(toks[0])):
                raise ValueError(f"Malformed/non-finite thermo data row in {log_path}: {ln}")
            if rows:
                break
            else:
                continue
        rows.append([float(t) for t in toks])

    if not rows:
        raise ValueError(f"Thermo table header found but no rows parsed in log: {log_path}")

    return _validated_thermo_table(header_cols, rows, source=Path(log_path))


def parse_all_thermo_tables(log_path: Path) -> list[ThermoTable]:
    """All thermo tables."""

    lines = log_path.read_text(errors="replace").splitlines()

    tables: list[ThermoTable] = []
    i = 0
    nlines = len(lines)
    while i < nlines:
        ln = lines[i].strip()
        if not ln.startswith("Step "):
            i += 1
            continue

        header_cols = ln.split()
        ncols = len(header_cols)
        rows: list[list[float]] = []

        j = i + 1
        while j < nlines:
            ln2 = lines[j].strip()
            if not ln2:
                j += 1
                continue
            if ln2.startswith("Step "):
                # table begins
                break
            toks = ln2.split()
            if len(toks) != ncols or (not all(_is_number(t) for t in toks)):
                nonfinite_literal = any(
                    t.strip().lower().lstrip("+-") in {"nan", "inf", "infinity"}
                    for t in toks
                )
                if len(toks) == ncols and (nonfinite_literal or (toks and _is_number(toks[0]))):
                    raise ValueError(
                        f"Malformed/non-finite thermo data row in {log_path}: {ln2}"
                    )
                if rows:
                    break
                j += 1
                continue
            rows.append([float(t) for t in toks])
            j += 1

        if rows:
            tables.append(
                _validated_thermo_table(header_cols, rows, source=Path(log_path))
            )

        i = j

    if not tables:
        raise ValueError(f"No thermo tables found in log: {log_path}")
    return tables


@dataclass(frozen=True)
class MSDSeries:
    step: np.ndarray
    msd: np.ndarray


def parse_msd_file(path: Path) -> MSDSeries:
    """Parse an emitted two-column MSD series without discarding bad evidence."""
    steps: list[int] = []
    msd: list[float] = []
    for line_number, ln in enumerate(
        path.read_text(errors="replace").splitlines(), start=1
    ):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        toks = ln.split()
        if len(toks) < 2:
            raise ValueError(
                f"Malformed MSD row at {path}:{line_number}: expected step and MSD"
            )
        if not (_is_number(toks[0]) and _is_number(toks[1])):
            raise ValueError(f"Malformed numeric MSD row at {path}:{line_number}: {ln}")
        step_numeric = float(toks[0])
        value = float(toks[1])
        step = int(step_numeric) if np.isfinite(step_numeric) else 0
        if not np.isfinite(step_numeric) or step_numeric != float(step) or step < 0:
            raise ValueError(
                f"MSD step must be a nonnegative integer at {path}:{line_number}"
            )
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"MSD value must be finite and nonnegative at {path}:{line_number}"
            )
        if steps and step <= steps[-1]:
            raise ValueError(
                f"MSD steps must be strictly increasing at {path}:{line_number}"
            )
        steps.append(step)
        msd.append(value)
    if len(steps) < 3:
        raise ValueError(f"Too few MSD points in {path}")
    return MSDSeries(step=np.asarray(steps, dtype=int), msd=np.asarray(msd, dtype=float))
