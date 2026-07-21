from __future__ import annotations

import json
import os
from pathlib import Path


def test_atomic_json_ignores_predictable_temp_symlink(tmp_path: Path):
    from vitriflow.workflows.progress import atomic_write_json

    target = tmp_path / "state.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch\n")
    predictable = tmp_path / ".state.json.tmp"
    os.symlink(victim, predictable)

    atomic_write_json(target, {"status": "ok", "value": 3})

    assert victim.read_text() == "do-not-touch\n"
    assert predictable.is_symlink()
    assert json.loads(target.read_text()) == {"status": "ok", "value": 3}
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_strict_json_replaces_complete_old_document(tmp_path: Path):
    from vitriflow.analysis.provenance import write_json_strict

    target = tmp_path / "manifest.json"
    target.write_text('{"generation": 1}\n')
    write_json_strict(target, {"generation": 2, "payload": [1, 2, 3]})

    assert json.loads(target.read_text()) == {"generation": 2, "payload": [1, 2, 3]}
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_cli_json_output_sanitizes_nonfinite_values(capsys) -> None:
    from vitriflow.cli import _print_json

    _print_json({"finite": 1.5, "missing": float("nan"), "unbounded": float("inf")})
    text = capsys.readouterr().out
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"finite": 1.5, "missing": None, "unbounded": None}


def test_input_snapshot_and_pair_coeff_strip_ignore_legacy_temp_symlinks(tmp_path: Path):
    from vitriflow.analysis.datafile import strip_lammps_data_pair_coeff_sections
    from vitriflow.workflows.hpc import _copy_input_snapshot

    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch\n")

    source = tmp_path / "source.data"
    source.write_text("source bytes\n")
    destination = tmp_path / "snapshot.data"
    os.symlink(victim, tmp_path / "snapshot.data.tmp")
    _copy_input_snapshot(source, destination)
    assert destination.read_text() == "source bytes\n"
    assert victim.read_text() == "do-not-touch\n"

    data = tmp_path / "pair.data"
    data.write_text(
        "LAMMPS data\n\nPair Coeffs\n\n1 1.0 2.0\n\nAtoms # atomic\n\n1 1 0 0 0\n"
    )
    os.symlink(victim, tmp_path / "pair.data.paircoeff_stripped.tmp")
    assert strip_lammps_data_pair_coeff_sections(data) == 1
    assert "Pair Coeffs" not in data.read_text()
    assert victim.read_text() == "do-not-touch\n"
