#!/usr/bin/env python3
from __future__ import annotations

"""Safely stage only the four GAP-20U+gr files from a directory or tar."""

import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


XML_NAME = "Carbon_GAP_20U+gr.xml"
EXPECTED_NAMES = frozenset(
    {
        XML_NAME,
        *(f"{XML_NAME}.sparseX.GAP_2022_11_4_0_14_40_15_889{i}" for i in (1, 2, 3)),
    }
)
# The published sidecars are far smaller than this.  The bound prevents a
# malicious sparse/declared-size tar member from exhausting the filesystem.
MAX_MEMBER_BYTES = 4 * 1024**3


def _safe_member_basename(name: str) -> str:
    path = PurePosixPath(str(name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path.name


def _stage_tar(source: Path, destination: Path) -> None:
    selected: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(source, mode="r:*") as archive:
        for member in archive:
            basename = _safe_member_basename(member.name)
            if basename not in EXPECTED_NAMES:
                continue
            if basename in selected:
                raise ValueError(f"duplicate GAP archive member basename: {basename}")
            if not member.isreg():
                raise ValueError(
                    f"GAP archive member must be a regular file: {member.name!r}"
                )
            if int(member.size) < 1 or int(member.size) > MAX_MEMBER_BYTES:
                raise ValueError(
                    f"GAP archive member has an invalid size: {member.name!r}"
                )
            selected[basename] = member
        missing = sorted(EXPECTED_NAMES.difference(selected))
        if missing:
            raise ValueError("GAP archive is missing required files: " + ", ".join(missing))
        for basename in sorted(EXPECTED_NAMES):
            stream = archive.extractfile(selected[basename])
            if stream is None:
                raise ValueError(f"could not read GAP archive member: {basename}")
            target = destination / basename
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if target.stat().st_size != int(selected[basename].size):
                raise ValueError(f"truncated GAP archive member: {basename}")


def _stage_directory(source: Path, destination: Path) -> None:
    matches: dict[str, list[Path]] = {name: [] for name in EXPECTED_NAMES}
    for candidate in source.rglob("*"):
        if candidate.name in EXPECTED_NAMES:
            matches[candidate.name].append(candidate)
    for basename, candidates in sorted(matches.items()):
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one {basename!r} below {source}, found {len(candidates)}"
            )
        candidate = candidates[0]
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"GAP source must be a regular non-symlink file: {candidate}")
        shutil.copyfile(candidate, destination / basename)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} SOURCE STAGING_DIR", file=sys.stderr)
        return 2
    source = Path(argv[1]).expanduser().resolve(strict=True)
    destination = Path(argv[2]).expanduser().resolve(strict=True)
    if any(destination.iterdir()):
        raise ValueError(f"staging directory must be empty: {destination}")
    if source.is_dir():
        _stage_directory(source, destination)
    elif source.is_file():
        _stage_tar(source, destination)
    else:
        raise ValueError(f"source is neither a directory nor a regular file: {source}")
    print(len(EXPECTED_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
