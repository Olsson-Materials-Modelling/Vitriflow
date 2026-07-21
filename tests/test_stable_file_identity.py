from __future__ import annotations

from pathlib import Path

import pytest


def test_stable_file_identity_hashes_one_regular_inode(tmp_path: Path) -> None:
    from vitriflow.utils import stable_file_identity

    source = tmp_path / "source.dat"
    source.write_bytes(b"stable bytes\n")
    identity = stable_file_identity(source)
    assert identity["resolved_path"] == str(source.resolve(strict=True))
    assert identity["size_bytes"] == len(b"stable bytes\n")
    assert len(str(identity["sha256"])) == 64


def test_stable_file_identity_allows_parent_alias_but_can_reject_final_symlink(
    tmp_path: Path,
) -> None:
    from vitriflow.utils import stable_file_identity

    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.dat"
    source.write_bytes(b"payload")
    alias = tmp_path / "alias"
    final_alias = real / "final-alias.dat"
    try:
        alias.symlink_to(real, target_is_directory=True)
        final_alias.symlink_to(source)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert stable_file_identity(alias / "source.dat")["sha256"] == stable_file_identity(source)["sha256"]
    with pytest.raises(RuntimeError, match="symbolic link"):
        stable_file_identity(final_alias, reject_final_symlink=True)


def test_stable_file_identity_rejects_an_inode_that_changes_while_hashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vitriflow.utils as module

    source = tmp_path / "source.dat"
    source.write_bytes(b"payload")
    real_fstat = module.os.fstat
    calls = 0

    class Changed:
        pass

    def changed_fstat(fd: int):
        nonlocal calls
        calls += 1
        value = real_fstat(fd)
        if calls != 2:
            return value
        changed = Changed()
        for name in dir(value):
            if name.startswith("st_"):
                try:
                    setattr(changed, name, getattr(value, name))
                except AttributeError:
                    pass
        changed.st_mtime_ns = int(value.st_mtime_ns) + 1
        return changed

    monkeypatch.setattr(module.os, "fstat", changed_fstat)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        module.stable_file_identity(source)


def test_resume_quarantines_only_uncommitted_box_directories(tmp_path: Path) -> None:
    from vitriflow.utils import quarantine_uncommitted_box_directories

    production = tmp_path / "production"
    committed = production / "box_001"
    interrupted = production / "box_002"
    committed.mkdir(parents=True)
    interrupted.mkdir()
    (committed / "relax.data").write_text("committed")
    (interrupted / "partial.out").write_text("partial")

    moved = quarantine_uncommitted_box_directories(
        production,
        committed_box_ids=[1],
        quarantine_root=tmp_path / "interrupted_attempts",
    )
    assert committed.is_dir()
    assert not interrupted.exists()
    assert len(moved) == 1
    assert (moved[0] / "partial.out").read_text() == "partial"

    # A second interruption of the same next box is retained under a unique
    # name rather than overwriting the first diagnostic tree.
    interrupted.mkdir()
    (interrupted / "partial.out").write_text("partial-again")
    moved_again = quarantine_uncommitted_box_directories(
        production,
        committed_box_ids=[1],
        quarantine_root=tmp_path / "interrupted_attempts",
    )
    assert moved_again[0] != moved[0]
    assert (moved_again[0] / "partial.out").read_text() == "partial-again"


def test_resume_quarantine_moves_an_orphan_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    from vitriflow.utils import quarantine_uncommitted_box_directories

    production = tmp_path / "production"
    production.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("unchanged")
    orphan = production / "box_001"
    try:
        orphan.symlink_to(victim, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    moved = quarantine_uncommitted_box_directories(
        production,
        committed_box_ids=[],
        quarantine_root=tmp_path / "interrupted_attempts",
    )
    assert len(moved) == 1 and moved[0].is_symlink()
    assert (victim / "keep.txt").read_text() == "unchanged"


def test_resume_quarantine_rejects_malformed_production_and_redirected_root(
    tmp_path: Path,
) -> None:
    from vitriflow.utils import quarantine_uncommitted_box_directories

    malformed = tmp_path / "production"
    malformed.write_text("not a directory")
    with pytest.raises(RuntimeError, match="real directory"):
        quarantine_uncommitted_box_directories(
            malformed,
            committed_box_ids=[],
            quarantine_root=tmp_path / "interrupted_attempts",
        )

    malformed.unlink()
    malformed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "interrupted_attempts"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="calculation root|symbolic links"):
        quarantine_uncommitted_box_directories(
            malformed,
            committed_box_ids=[],
            quarantine_root=redirected,
        )


def test_external_task_quarantine_preserves_manifest_and_input(tmp_path: Path) -> None:
    from vitriflow.workflows.hpc import _quarantine_incomplete_task_execution

    box = tmp_path / "production" / "box_001"
    (box / "input").mkdir(parents=True)
    (box / "input" / "structure.data").write_text("input")
    (box / "task.json").write_text("{}")
    for role in ("warmup", "melt", "continuous"):
        (box / role).mkdir()
        (box / role / "partial.out").write_text(role)

    destination = _quarantine_incomplete_task_execution(box)
    assert destination is not None
    assert (box / "task.json").is_file()
    assert (box / "input" / "structure.data").is_file()
    assert not (box / "warmup").exists()
    assert (destination / "warmup" / "partial.out").read_text() == "warmup"


def test_external_task_quarantine_rejects_redirected_ancestor_without_moving(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.hpc import _quarantine_incomplete_task_execution

    production = tmp_path / "production"
    box = production / "box_001"
    (box / "warmup").mkdir(parents=True)
    (box / "warmup" / "partial.out").write_text("partial")
    (box / "task.json").write_text("{}")
    (box / "input").mkdir()
    (box / "input" / "structure.data").write_text("input")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (production / "interrupted_task_attempts").symlink_to(
            outside, target_is_directory=True
        )
    except OSError as exc:  # pragma: no cover - host policy
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(RuntimeError, match="must not contain a symbolic link"):
        _quarantine_incomplete_task_execution(box)
    assert (box / "warmup" / "partial.out").read_text() == "partial"
    assert (box / "task.json").is_file()
    assert (box / "input" / "structure.data").is_file()
    assert not any(outside.iterdir())


def test_external_input_snapshot_rejects_source_mutation_during_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vitriflow.workflows import hpc

    source = tmp_path / "source.data"
    destination = tmp_path / "production" / "box_001" / "input" / "source.data"
    source.write_bytes(b"original scientific input\n")
    real_copy = hpc._atomic_copy_verified_regular_file

    def copy_then_mutate(*args, **kwargs):
        copied = real_copy(*args, **kwargs)
        source.write_bytes(b"changed while task was materialised\n")
        return copied

    monkeypatch.setattr(hpc, "_atomic_copy_verified_regular_file", copy_then_mutate)
    with pytest.raises(RuntimeError, match="changed while snapshotting"):
        hpc._copy_input_snapshot(source, destination)
    assert not destination.exists()
