from __future__ import annotations

import numpy as np


def _frame(step: int):
    from vitriflow.analysis.dump import DumpFrame

    return DumpFrame(
        timestep=step,
        ids=np.asarray([1]),
        types=np.asarray([1]),
        positions=np.asarray([[0.0, 0.0, 0.0]]),
        cell=np.eye(3),
        origin=np.zeros(3),
    )


def test_dense_window_selection_treats_max_frames_as_hard_cap() -> None:
    from vitriflow.analysis.trajectory import _select_dense_window_indices

    steps = np.arange(20, dtype=float)
    chosen = _select_dense_window_indices(
        steps,
        window=(4.0, 18.0),
        quench_tail_min_frames=12,
        max_frames=5,
    )
    assert len(chosen) == 5
    assert chosen[0] == 0
    assert chosen[-1] == 19
    assert all(4 <= index <= 18 for index in chosen[1:-1])


def test_dense_window_endpoint_priority_for_one_and_two_slots() -> None:
    from vitriflow.analysis.trajectory import _select_dense_window_indices

    steps = np.arange(10, dtype=float)
    assert _select_dense_window_indices(
        steps,
        window=(3.0, 7.0),
        quench_tail_min_frames=8,
        max_frames=1,
    ) == [9]
    assert _select_dense_window_indices(
        steps,
        window=(3.0, 7.0),
        quench_tail_min_frames=8,
        max_frames=2,
    ) == [0, 9]


def test_quench_selection_metadata_reports_minimum_limited_by_cap() -> None:
    from vitriflow.analysis.trajectory import select_stage_frames

    frames = [_frame(step) for step in range(20)]
    selected, metadata = select_stage_frames(
        frames,
        stage_role="quench",
        quench_window_steps_range=(3.0, 18.0),
        quench_tail_min_frames=12,
        max_frames=8,
    )
    assert len(selected) == 8
    assert selected[0].timestep == 0
    assert selected[-1].timestep == 19
    assert metadata["max_frames_hard_cap"] is True
    assert metadata["dense_window_minimum_satisfied"] is False
    assert metadata["dense_window_minimum_limited_by_cap"] is True
    assert metadata["dense_window_minimum_limited_by_availability"] is False

