#!/usr/bin/env python3
"""Validate that the active Python environment can import and run VitriFlow."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


def main() -> int:
    try:
        vf = importlib.import_module("vitriflow")
        print(f"import vitriflow: ok")
        print(f"version: {getattr(vf, '__version__', 'unknown')}")
        print(f"module: {getattr(vf, '__file__', 'unknown')}")
        importlib.import_module("vitriflow.cli")
        print("import vitriflow.cli: ok")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"import vitriflow: failed: {exc}", file=sys.stderr)
        return 2

    exe = shutil.which("vitriflow")
    if exe is None:
        print("console script: missing", file=sys.stderr)
        return 3
    print(f"console script: {exe}")
    proc = subprocess.run([exe, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("console script execution failed", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return int(proc.returncode or 4)
    print(proc.stdout.strip() or proc.stderr.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
