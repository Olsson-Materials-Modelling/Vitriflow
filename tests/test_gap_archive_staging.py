from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGER = ROOT / "scripts" / "stage_gap20ugr_potential_files.py"
XML = "Carbon_GAP_20U+gr.xml"
NAMES = [
    XML,
    *(f"{XML}.sparseX.GAP_2022_11_4_0_14_40_15_889{i}" for i in (1, 2, 3)),
]


def _write_tar(path: Path, records: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content, kind in records:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                archive.addfile(info)
            else:  # pragma: no cover - fixture guard
                raise AssertionError(kind)


def _run(source: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(STAGER), str(source), str(destination)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_gap_tar_stager_extracts_only_exact_regular_files(tmp_path: Path):
    source = tmp_path / "source.tar"
    records = [(f"nested/{name}", name.encode(), "file") for name in NAMES]
    records.append(("nested/unrelated-large-model", b"ignored", "file"))
    _write_tar(source, records)
    destination = tmp_path / "staged"
    destination.mkdir()

    result = _run(source, destination)

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in destination.iterdir()) == sorted(NAMES)
    assert (destination / XML).read_bytes() == XML.encode()


def test_gap_tar_stager_rejects_traversal_duplicate_and_link_members(tmp_path: Path):
    base = [(f"data/{name}", name.encode(), "file") for name in NAMES]
    cases = {
        "traversal": [*base, (f"../{XML}", b"evil", "file")],
        "duplicate": [*base, (f"other/{XML}", b"evil", "file")],
        "symlink": [
            *[row for row in base if not row[0].endswith(NAMES[-1])],
            (f"data/{NAMES[-1]}", b"", "symlink"),
        ],
    }
    for label, records in cases.items():
        source = tmp_path / f"{label}.tar"
        _write_tar(source, records)
        destination = tmp_path / f"staged-{label}"
        destination.mkdir()
        result = _run(source, destination)
        assert result.returncode != 0, label
        assert not any(destination.iterdir()), label
