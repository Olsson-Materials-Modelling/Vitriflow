from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("mock_engine_build_identities")


def test_workflow_lock_is_exclusive_crash_releasable_and_inside_target(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.workflow_lock import exclusive_workflow_lock

    outdir = tmp_path / "calculation"
    with exclusive_workflow_lock(outdir, purpose="first") as lock_path:
        assert lock_path.parent == outdir
        assert lock_path.name == ".vitriflow.lock"
        assert outdir.is_dir()
        metadata = json.loads(lock_path.read_text())
        assert metadata["purpose"] == "first"
        assert Path(metadata["target"]) == outdir.resolve(strict=False)
        with pytest.raises(RuntimeError, match="another VitriFlow process"):
            with exclusive_workflow_lock(outdir, purpose="duplicate"):
                pass

    # The inode remains but the kernel lock is released after normal exit.
    assert lock_path.is_file()
    with exclusive_workflow_lock(outdir, purpose="second"):
        pass


def test_workflow_lock_never_opens_a_sibling_in_read_only_parent_model(
    monkeypatch, tmp_path: Path
) -> None:
    """An existing writable calculation must not need parent write access."""

    from vitriflow.workflows import workflow_lock

    outdir = tmp_path / "shared_read_only_parent" / "owned_calculation"
    outdir.mkdir(parents=True)
    real_open = workflow_lock.os.open

    def guarded_open(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent == outdir.parent:
            raise PermissionError("simulated read-only shared parent")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(workflow_lock.os, "open", guarded_open)
    with workflow_lock.exclusive_workflow_lock(
        outdir, purpose="HPC permission regression"
    ) as lock_path:
        assert lock_path == outdir / workflow_lock.WORKFLOW_LOCK_FILENAME


def test_internal_lock_is_the_only_clean_start_and_analysis_exclusion(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows import output_analysis
    from vitriflow.workflows.workflow_lock import (
        WORKFLOW_LOCK_FILENAME,
        exclusive_workflow_lock,
        workflow_payload_entries,
    )

    outdir = tmp_path / "calculation"
    with exclusive_workflow_lock(outdir, purpose="clean-start probe") as lock_path:
        assert workflow_payload_entries(outdir) == ()
        assert lock_path.name == WORKFLOW_LOCK_FILENAME
        assert output_analysis._is_analysis_source_candidate(lock_path) is False
        other_hidden = outdir / ".untrusted-hidden-state"
        other_hidden.write_text("must remain visible to admission\n")
        assert workflow_payload_entries(outdir) == (other_hidden,)


def test_workflow_lock_releases_after_exception(tmp_path: Path) -> None:
    from vitriflow.workflows.workflow_lock import exclusive_workflow_lock

    target = tmp_path / "calculation"
    with pytest.raises(LookupError, match="interrupt"):
        with exclusive_workflow_lock(target, purpose="first"):
            raise LookupError("interrupt")
    with exclusive_workflow_lock(target, purpose="resume"):
        pass


def test_workflow_lock_rejects_symlink_lock_file(tmp_path: Path) -> None:
    from vitriflow.workflows.workflow_lock import _lock_path_for, exclusive_workflow_lock

    target = tmp_path / "calculation"
    target.mkdir()
    lock_path = _lock_path_for(target)
    victim = tmp_path / "victim"
    victim.write_text("do not alter\n")
    lock_path.symlink_to(victim)

    with pytest.raises(RuntimeError, match="Cannot open trusted workflow lock"):
        with exclusive_workflow_lock(target, purpose="unsafe"):
            pass
    assert victim.read_text() == "do not alter\n"


def test_workflow_lock_rejects_hardlink_without_modifying_victim(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.workflow_lock import _lock_path_for, exclusive_workflow_lock

    target = tmp_path / "calculation"
    target.mkdir()
    lock_path = _lock_path_for(target)
    victim = tmp_path / "victim"
    original = b"unrelated scientific data\n"
    victim.write_bytes(original)
    try:
        lock_path.hardlink_to(victim)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links unavailable on this filesystem: {exc}")

    with pytest.raises(RuntimeError, match="exactly one hard link"):
        with exclusive_workflow_lock(target, purpose="unsafe hard-link probe"):
            pass
    assert victim.read_bytes() == original


def test_duplicate_external_task_fails_before_execution(tmp_path: Path) -> None:
    from vitriflow.workflows import hpc
    from vitriflow.workflows.workflow_lock import exclusive_workflow_lock

    box_dir = tmp_path / "production" / "box_001"
    input_snapshot = box_dir / "input" / "input.data"
    input_snapshot.parent.mkdir(parents=True)
    input_snapshot.write_text("input\n")
    task_json = box_dir / "task.json"
    task = {
        "schema": "vitriflow.box_task.v1",
        "input_manifest": {
            "schema": "vitriflow.task_inputs.v2",
            "structure_snapshot": {
                "path": str(input_snapshot),
            },
        },
        "task": {
            "box": 1,
            "box_dir": str(box_dir),
            "input_snapshot": str(input_snapshot),
            "task_json": str(task_json),
            "task_result": str(box_dir / "task_result.json"),
        },
    }
    task_json.write_text(json.dumps(task))
    with exclusive_workflow_lock(box_dir, purpose="already running task"):
        with pytest.raises(RuntimeError, match="another VitriFlow process"):
            hpc.execute_production_box_task(task)
    assert not (box_dir / "task_result.json").exists()


@pytest.mark.parametrize(
    ("module_name", "function_name", "purpose"),
    [
        ("vitriflow.workflows.custom_schedule", "run_custom_schedule", "custom"),
        ("vitriflow.workflows.output_analysis", "analyze_output_data", "analysis"),
    ],
)
def test_all_public_multi_file_workflows_reject_a_duplicate_controller(
    tmp_path: Path,
    module_name: str,
    function_name: str,
    purpose: str,
) -> None:
    import importlib

    from vitriflow.workflows.workflow_lock import exclusive_workflow_lock

    function = getattr(importlib.import_module(module_name), function_name)
    outdir = tmp_path / purpose
    with exclusive_workflow_lock(outdir, purpose="already running"):
        with pytest.raises(RuntimeError, match="another VitriFlow process"):
            function(None, outdir, input_path=tmp_path) if function_name == "analyze_output_data" else function(None, outdir)
