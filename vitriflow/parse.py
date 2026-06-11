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
            if rows:
                break
            else:
                continue
        rows.append([float(t) for t in toks])

    if not rows:
        raise ValueError(f"Thermo table header found but no rows parsed in log: {log_path}")

    data = np.asarray(rows, dtype=float)
    return ThermoTable(columns=header_cols, data=data)


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
                if rows:
                    break
                j += 1
                continue
            rows.append([float(t) for t in toks])
            j += 1

        if rows:
            data = np.asarray(rows, dtype=float)
            tables.append(ThermoTable(columns=header_cols, data=data))

        i = j

    if not tables:
        raise ValueError(f"No thermo tables found in log: {log_path}")
    return tables


@dataclass(frozen=True)
class MSDSeries:
    step: np.ndarray
    msd: np.ndarray


def parse_msd_file(path: Path) -> MSDSeries:
    """Msd file."""
    steps: list[int] = []
    msd: list[float] = []
    for ln in path.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        toks = ln.split()
        if len(toks) < 2:
            continue
        if _is_number(toks[0]) and _is_number(toks[1]):
            steps.append(int(float(toks[0])))
            msd.append(float(toks[1]))
    if len(steps) < 3:
        raise ValueError(f"Too few MSD points in {path}")
    return MSDSeries(step=np.asarray(steps, dtype=float), msd=np.asarray(msd, dtype=float))
