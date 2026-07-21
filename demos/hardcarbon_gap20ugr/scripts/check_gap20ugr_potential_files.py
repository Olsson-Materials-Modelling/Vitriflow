#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


# Preserve the documented demo-local command while sharing the authoritative
# checksum policy with the repository-level installer and smoke test.
ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(
    str(ROOT / "scripts" / "check_gap20ugr_potential_files.py"),
    run_name="__main__",
)
