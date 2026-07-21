from __future__ import annotations

from pathlib import Path

import pytest


def test_autotune_resume_policy_requires_explicit_checkpoint_and_clean_start(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.autotune import _resolve_autotune_resume_mode

    outdir = tmp_path / "autotune"
    outdir.mkdir()
    results = outdir / "autotune_results.json"

    with pytest.raises(RuntimeError, match="--resume.*missing"):
        _resolve_autotune_resume_mode(
            outdir=outdir, results_path=results, resume=True
        )
    assert not _resolve_autotune_resume_mode(
        outdir=outdir, results_path=results, resume=None
    )
    assert not _resolve_autotune_resume_mode(
        outdir=outdir, results_path=results, resume=False
    )

    orphan = outdir / "preflight"
    orphan.mkdir()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        _resolve_autotune_resume_mode(
            outdir=outdir, results_path=results, resume=None
        )
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        _resolve_autotune_resume_mode(
            outdir=outdir, results_path=results, resume=False
        )


def test_autotune_resume_policy_auto_detects_result_and_no_resume_rejects_it(
    tmp_path: Path,
) -> None:
    from vitriflow.workflows.autotune import _resolve_autotune_resume_mode

    results = tmp_path / "autotune_results.json"
    results.write_text("{}")
    assert _resolve_autotune_resume_mode(
        outdir=tmp_path, results_path=results, resume=None
    )
    assert _resolve_autotune_resume_mode(
        outdir=tmp_path, results_path=results, resume=True
    )
    with pytest.raises(RuntimeError, match="--no-resume"):
        _resolve_autotune_resume_mode(
            outdir=tmp_path, results_path=results, resume=False
        )


def test_autotune_resume_policy_rejects_result_symlink(tmp_path: Path) -> None:
    from vitriflow.workflows.autotune import _resolve_autotune_resume_mode

    target = tmp_path / "actual.json"
    target.write_text("{}")
    results = tmp_path / "autotune_results.json"
    results.symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic link"):
        _resolve_autotune_resume_mode(
            outdir=tmp_path, results_path=results, resume=True
        )


def test_production_resume_seed_stream_requires_exact_four_stage_record() -> None:
    from vitriflow.workflows.autotune import (
        _count_production_resume_seed_draws,
        _production_resume_seed_draw_count,
    )

    valid = {
        "seeds": {"warmup": 11, "melt": 12, "quench": 13, "relax": 14}
    }
    assert _production_resume_seed_draw_count(valid) == 4
    assert _count_production_resume_seed_draws([valid, valid]) == 8

    for invalid in (
        {"seeds": {"melt": 12, "quench": 13, "relax": 14}},
        {
            "seeds": {
                "warmup": 11,
                "melt": 12,
                "quench": 13,
                "relax": 14,
                "extra": 15,
            }
        },
        {
            "seeds": {
                "warmup": 11.0,
                "melt": 12,
                "quench": 13,
                "relax": 14,
            }
        },
        {
            "seeds": {
                "warmup": 0,
                "melt": 12,
                "quench": 13,
                "relax": 14,
            }
        },
    ):
        with pytest.raises(RuntimeError, match="Cannot safely resume"):
            _production_resume_seed_draw_count(invalid)

