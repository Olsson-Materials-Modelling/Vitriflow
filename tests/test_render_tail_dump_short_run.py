"""Tail-dump short-run path: pin the rendered output and observed frame count.

Targeted regression for ultrareview finding #3. The render adds
``dump_modify ... first yes`` when the requested tail window is longer
than the run itself. That addition is *robustness only* -- it keeps very
short runs from emitting zero frames -- it is NOT a guarantee that the
rendered dump will produce exactly ``tail_dump_frames`` frames. This file
documents and pins the actual observed frame counts so any future change
that claims "frame count = frames" cannot land silently.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest


def _make_data_file(path: Path) -> Path:
    path.write_text(
        "LAMMPS data file via vitriflow test\n"
        "\n"
        "1 atoms\n"
        "1 atom types\n"
        "\n"
        "0.0 1.0 xlo xhi\n"
        "0.0 1.0 ylo yhi\n"
        "0.0 1.0 zlo zhi\n"
        "\n"
        "Masses\n"
        "\n"
        "1 28.0855\n"
        "\n"
        "Atoms # atomic\n"
        "\n"
        "1 1 0.0 0.0 0.0\n"
    )
    return path


def _make_short_run_stage(*, run_steps: int, frames: int, stride: int, data: Path):
    from vitriflow.lammps_input import StageSpec

    return StageSpec(
        name="probe",
        input_data=data,
        output_data=data.parent / "out.data",
        temperature_start=300.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=run_steps,
        seed=1,
        velocity_mode="create",
        force_isotropic=False,
        replicate=None,
        write_dump=True,
        dump_every=1,
        tail_dump_frames=frames,
        tail_dump_stride=stride,
        msd_every=1,
        potential_lines=None,
    )


def _simulate_lammps_dump_steps(*, run_steps: int, stride: int, first_yes: bool) -> list[int]:
    """Replicate LAMMPS ``dump custom <stride>`` semantics.

    With ``dump_modify first yes`` the dump fires at step 0 and then every
    ``stride`` steps thereafter, up to and including ``run_steps`` if it lies
    on the cadence. Without ``first yes`` it only starts firing at the first
    multiple of ``stride`` strictly greater than 0.
    """

    if first_yes:
        steps = [s for s in range(0, run_steps + 1, stride)]
    else:
        steps = [s for s in range(stride, run_steps + 1, stride)]
    return steps


def _extract_short_run_dump_block(script: str) -> tuple[int, str]:
    """Return (stride, dump_modify_line) for the short-run dump rendered by
    render_stage / _render_dump_and_run. Raises AssertionError if not found.
    """

    m = re.search(r"^dump d1 all custom (\d+) [^\s]+\.lammpstrj id type xu yu zu", script, re.MULTILINE)
    assert m is not None, f"short-run dump line not found in script:\n{script}"
    dump_modify = re.search(r"^\s*dump_modify d1 .+$", script, re.MULTILINE)
    assert dump_modify is not None, f"dump_modify line not found in script:\n{script}"
    return int(m.group(1)), dump_modify.group(0).strip()


# (run_steps, frames, stride, expected_stride_eff, expected_step_set, expected_count)
# These are what the rendered script will actually produce in LAMMPS, NOT a
# claim that the count equals `frames`. Any change that breaks these numbers
# requires an explicit, deliberate update -- and probably a comment update too.
_OBSERVED_CASES = [
    # (10,4,3): 10 <= 12 triggers short-run path; stride_eff=ceil(10/4)=3.
    # frames at 0,3,6,9 -> count=4 (matches request)
    dict(run_steps=10, frames=4, stride=3, expected_stride_eff=3,
         expected_steps=[0, 3, 6, 9], expected_count=4),
    # (10,5,2): 10 <= 10 triggers short-run path; stride_eff=ceil(10/5)=2.
    # frames at 0,2,4,6,8,10 -> count=6 (overshoot by 1)
    dict(run_steps=10, frames=5, stride=2, expected_stride_eff=2,
         expected_steps=[0, 2, 4, 6, 8, 10], expected_count=6),
    # (12,4,3): 12 <= 12 triggers short-run path; stride_eff=ceil(12/4)=3.
    # frames at 0,3,6,9,12 -> count=5 (overshoot by 1)
    dict(run_steps=12, frames=4, stride=3, expected_stride_eff=3,
         expected_steps=[0, 3, 6, 9, 12], expected_count=5),
    # (10,11,1): 10 <= 11 triggers short-run path; stride_eff=ceil(10/11)=1.
    # frames at 0..10 -> count=11 (matches request)
    dict(run_steps=10, frames=11, stride=1, expected_stride_eff=1,
         expected_steps=list(range(0, 11)), expected_count=11),
    # Pathological (1,4,3): stride_eff=ceil(1/4)=1; frames at 0,1 -> count=2.
    # Without first yes this would have been zero frames.
    dict(run_steps=1, frames=4, stride=3, expected_stride_eff=1,
         expected_steps=[0, 1], expected_count=2),
]


@pytest.mark.parametrize("case", _OBSERVED_CASES)
def test_render_stage_short_run_uses_first_yes_and_documented_stride(case, tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_stage

    data = _make_data_file(tmp_path / "input.data")
    stage = _make_short_run_stage(
        run_steps=case["run_steps"],
        frames=case["frames"],
        stride=case["stride"],
        data=data,
    )

    pot = KimConfig(model="TEST_MODEL", interactions=["Si"])
    md = MDConfig(ensemble="nvt")
    script = render_stage(pot, md, stage)

    rendered_stride, dump_modify_line = _extract_short_run_dump_block(script)
    assert rendered_stride == case["expected_stride_eff"]
    # `first yes` must be present -- this is the robustness contract, not a
    # frame-count guarantee. See lammps_input.render_stage docstring.
    assert "first yes" in dump_modify_line, (
        "short-run dump_modify must include `first yes` so step 0 is always a "
        "frame; otherwise very short runs can produce zero dumps. See finding #3."
    )

    # Pin the actually observed frame count for this case. This is the truthful
    # description of what LAMMPS will do given the rendered stride and run_steps,
    # not a claim of equality with the requested `tail_dump_frames`.
    actual_steps = _simulate_lammps_dump_steps(
        run_steps=case["run_steps"], stride=rendered_stride, first_yes=True
    )
    assert actual_steps == case["expected_steps"], (
        f"observed step set {actual_steps} differs from documented {case['expected_steps']} "
        f"for run_steps={case['run_steps']}, frames={case['frames']}, stride={case['stride']}. "
        "Update the case table AND the comment in lammps_input.py if this is intentional."
    )
    assert len(actual_steps) == case["expected_count"]

    # The ONLY invariant `first yes` provides is `count >= 1` (step 0 is
    # always emitted). Anything stronger -- including count >= requested
    # frames -- is FALSE in cases like (run_steps=1, frames=4) where the
    # observed count is 2 < 4. That overclaim is exactly what finding #3
    # called out; the case table above documents truth, not aspiration.
    assert len(actual_steps) >= 1


@pytest.mark.parametrize("case", _OBSERVED_CASES)
def test_render_continuous_stages_short_run_matches_render_stage(case, tmp_path: Path):
    """The continuous-pipeline renderer takes the same short-run path via
    _render_dump_and_run; the same robustness/pinned-count contract applies.
    """
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import render_continuous_stages

    data = _make_data_file(tmp_path / "input.data")
    stage = _make_short_run_stage(
        run_steps=case["run_steps"],
        frames=case["frames"],
        stride=case["stride"],
        data=data,
    )

    pot = KimConfig(model="TEST_MODEL", interactions=["Si"])
    md = MDConfig(ensemble="nvt")
    script = render_continuous_stages(pot, md, [stage], stage_dir_prefixes={"probe": "."})

    rendered_stride, dump_modify_line = _extract_short_run_dump_block(script)
    assert rendered_stride == case["expected_stride_eff"]
    assert "first yes" in dump_modify_line

    actual_steps = _simulate_lammps_dump_steps(
        run_steps=case["run_steps"], stride=rendered_stride, first_yes=True
    )
    assert len(actual_steps) == case["expected_count"]


def test_short_run_path_does_not_promise_exact_frame_count_in_comment():
    """Source-level guard: the comment in render_stage must not claim exact
    frame count parity. The earlier patch carried a comment saying `first yes`
    "makes {frames} fit in {run_steps}" -- which is false in cases like
    (run_steps=10, frames=5) where the actual count is 6. Finding #3
    explicitly required removing that claim.
    """

    import inspect

    from vitriflow.lammps_input import render_stage

    src = inspect.getsource(render_stage)
    forbidden_claims = [
        # Phrases from the previous, inaccurate comment that should not return.
        "fits in {run_steps}",
        "frame count is consistent with the requested",
        "frame count guarantee",
        "frame count = frames",
        "guarantees the requested",
    ]
    for needle in forbidden_claims:
        assert needle not in src, (
            f"render_stage comment must not claim {needle!r}; finding #3 requires "
            "treating `first yes` as harmless robustness only."
        )


def test_first_yes_actually_prevents_zero_frame_pathological_case(tmp_path: Path):
    """Demonstrates the harm `first yes` is robustness AGAINST: without it,
    a stride_eff that exceeds run_steps would emit zero dumps. With it, we
    always get at least the step-0 frame. This is the entire claim the
    addition is allowed to make.
    """
    # Without first yes, frames would be at multiples of stride_eff strictly > 0.
    # If stride_eff > run_steps, that set is empty.
    actual_without = _simulate_lammps_dump_steps(run_steps=1, stride=5, first_yes=False)
    assert actual_without == []

    actual_with = _simulate_lammps_dump_steps(run_steps=1, stride=5, first_yes=True)
    assert actual_with == [0]
