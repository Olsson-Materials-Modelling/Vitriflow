from __future__ import annotations

"""Deterministic identity for the installed VitriFlow execution package."""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any


_DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONTENT_SCHEMA = "vitriflow.package_content.v1"
_RUNTIME_SCHEMA = "vitriflow.runtime.v2"

# Non-Python package data that participates in execution.  Keep this list in
# lock-step with ``tool.setuptools.package-data`` in pyproject.toml.  In
# particular, the source tree contains legacy examples which are deliberately
# not installed in a wheel; hashing every source-tree example made an editable
# install and its wheel report different runtimes despite executing identical
# packaged code.
_DECLARED_EXAMPLE_DATA = frozenset(
    {
        "examples/minimal_metal.yaml",
        "examples/al_fcc_4x4x4.data",
        "examples/si_diamond_cp2k_toy.yaml",
        "examples/sio2_bks_zbl_smoke.yaml",
        "examples/hc_C_GAP20Ugr_hc_custom_demo.yaml",
    }
)


def _is_execution_package_file(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts or "__pycache__" in parts or relative_path.suffix in {".pyc", ".pyo"}:
        return False
    if relative_path.suffix == ".py":
        return True
    relative_text = relative_path.as_posix()
    if relative_text in _DECLARED_EXAMPLE_DATA:
        return True
    # ``data/cp2k/*`` is declared package data.  Direct children are the only
    # non-Python files matched by that non-recursive setuptools glob.
    return (
        len(parts) == 3
        and parts[0] == "data"
        and parts[1] == "cp2k"
    )


def _package_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in Path(root).rglob("*")
            if path.is_file()
            and _is_execution_package_file(path.relative_to(root))
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _stat_signature(root: Path, files: list[Path]) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in files:
        stat = path.stat()
        rows.append(
            (
                path.relative_to(root).as_posix(),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        )
    return tuple(rows)


@lru_cache(maxsize=8)
def _content_identity_for_signature(
    root_text: str,
    signature: tuple[tuple[Any, ...], ...],
) -> tuple[str, tuple[tuple[str, str, int], ...]]:
    root = Path(root_text)
    digest = hashlib.sha256()
    records: list[tuple[str, str, int]] = []
    for row in signature:
        relative = str(row[0])
        data = (root / Path(relative)).read_bytes()
        file_sha = hashlib.sha256(data).hexdigest()
        records.append((relative, file_sha, len(data)))
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest(), tuple(records)


def package_content_identity(package_root: Path | None = None) -> dict[str, Any]:
    """Hash path/content pairs for every file shipped as executable package data.

    Absolute installation paths, mtimes, wheel metadata, bytecode and build
    timestamps never enter the digest.  Editable and wheel installs built from
    identical packaged source therefore have the same identity.
    """

    root = Path(package_root or _DEFAULT_PACKAGE_ROOT).resolve(strict=True)
    files = _package_files(root)
    if not files:
        raise RuntimeError(f"VitriFlow package-content identity found no files under {root}")
    # Retry if a file changes while it is being hashed.  This also makes the
    # stat-keyed cache safe against ordinary editable-source modifications.
    for _attempt in range(3):
        before = _stat_signature(root, files)
        digest, records = _content_identity_for_signature(str(root), before)
        after_files = _package_files(root)
        after = _stat_signature(root, after_files)
        if before == after:
            return {
                "schema": _CONTENT_SCHEMA,
                "algorithm": "sha256:length-prefixed-relative-path-and-content:v1",
                "sha256": digest,
                "file_count": len(records),
            }
        files = after_files
    raise RuntimeError("VitriFlow package files changed while computing runtime identity")


def runtime_identity(package_root: Path | None = None) -> dict[str, Any]:
    from . import __version__

    return {
        "schema": _RUNTIME_SCHEMA,
        "vitriflow_version": str(__version__),
        "package_content": package_content_identity(package_root),
    }


__all__ = ["package_content_identity", "runtime_identity"]
