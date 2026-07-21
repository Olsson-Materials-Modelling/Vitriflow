from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _minimal_datafile(path: Path) -> None:
    path.write_text(
        """LAMMPS data

2 atoms
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # atomic

1 1 1 2 3
2 1 4 5 6
"""
    )


@pytest.mark.parametrize(
    ("cp2k_version", "expects_keyword"),
    [((2023, 2, 0), False), ((2024, 1, 0), True), ((2025, 2, 0), True)],
)
def test_cp2k_md_input_binds_requested_rng_seed_restart_and_version_policy(
    cp2k_version: tuple[int, int, int], expects_keyword: bool
):
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    cfg = Cp2kConfig(
        kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")}
    )
    text = render_cp2k_md_input(
        atoms=Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        cfg=cfg,
        md_cfg=MDConfig(ensemble="nvt"),
        ensemble="nvt",
        temperature_K=300.0,
        steps=10,
        timestep_fs=1.0,
        tdamp_fs=100.0,
        project="seeded",
        energy_every=1,
        traj_every=1,
        traj_file="traj.dcd",
        ener_file="ener.dat",
        restart_file="previous.restart",
        seed=987654,
        cp2k_version=cp2k_version,
    )

    assert "  SEED 987654" in text
    assert "RESTART_RANDOMG T" in text
    assert "IGNORE_CONVERGENCE_FAILURE F" not in text
    assert ("IGNORE_CONVERGENCE_FAILURE T" in text) is expects_keyword


def test_cp2k_offline_renderer_omits_version_specific_scf_keyword():
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    text = render_cp2k_md_input(
        atoms=Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        cfg=Cp2kConfig(
            kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")}
        ),
        md_cfg=MDConfig(ensemble="nvt"),
        ensemble="nvt",
        temperature_K=300.0,
        steps=1,
        timestep_fs=1.0,
        tdamp_fs=100.0,
        project="offline",
        energy_every=1,
        traj_every=1,
        traj_file="traj.dcd",
        ener_file="ener.dat",
    )
    assert "IGNORE_CONVERGENCE_FAILURE" not in text


@pytest.mark.parametrize(
    ("nsteps", "stride", "nframes", "expected"),
    [
        (10, 2, 6, [0, 2, 4, 6, 8, 10]),
        (10, 2, 5, [2, 4, 6, 8, 10]),
        (11, 2, 7, [0, 2, 4, 6, 8, 10, 11]),
        (11, 2, 6, [2, 4, 6, 8, 10, 11]),
        (3, 10, 2, [0, 3]),
        (3, 10, 1, [3]),
    ],
)
def test_cp2k_trajectory_step_inference_accepts_only_documented_patterns(
    nsteps: int, stride: int, nframes: int, expected: list[int]
) -> None:
    from vitriflow.workflows.stage_runner import _infer_cp2k_traj_steps

    assert _infer_cp2k_traj_steps(nsteps, stride, nframes) == expected


def test_cp2k_trajectory_step_inference_rejects_unexpected_frame_count() -> None:
    from vitriflow.workflows.stage_runner import _infer_cp2k_traj_steps

    with pytest.raises(ValueError, match="Unexpected CP2K trajectory frame count"):
        _infer_cp2k_traj_steps(10, 3, 3)


def test_cp2k_segment_thermo_boundary_is_deduplicated_only_if_consistent() -> None:
    from vitriflow.workflows.stage_runner import _merge_cp2k_segment_thermo_rows

    rows = [
        (0, 300.0, float("nan"), -10.0, 100.0, 2.0),
        (10, 301.0, float("nan"), -9.5, 101.0, 1.98),
        (10, 301.0 + 1.0e-11, float("nan"), -9.5, 101.0, 1.98),
        (20, 302.0, float("nan"), -9.0, 102.0, 1.96),
    ]
    merged = _merge_cp2k_segment_thermo_rows(rows)
    assert [row[0] for row in merged] == [0, 10, 20]
    # The preceding propagated row is canonical, matching the retained DCD
    # frame; the restart-side step zero is a second observation.
    assert merged[1][1] == rows[1][1]

    pressure_boundary = [
        (0, 300.0, 5.0, -10.0, 100.0, 2.0),
        (0, 300.0, float("nan"), -10.0, 100.0, 2.0),
    ]
    assert _merge_cp2k_segment_thermo_rows(pressure_boundary)[0][2] == 5.0

    pressure_resolved_at_restart = [
        (0, 300.0, float("nan"), -10.0, 100.0, 2.0),
        (0, 300.0, 5.0, -10.0, 100.0, 2.0),
    ]
    assert (
        _merge_cp2k_segment_thermo_rows(pressure_resolved_at_restart)[0][2]
        == 5.0
    )

    inconsistent = list(rows)
    inconsistent[2] = (10, 350.0, float("nan"), -9.5, 101.0, 1.98)
    with pytest.raises(ValueError, match="Inconsistent duplicate CP2K segment-boundary"):
        _merge_cp2k_segment_thermo_rows(inconsistent)

    with pytest.raises(ValueError, match="nonmonotone"):
        _merge_cp2k_segment_thermo_rows([rows[1], rows[0]])


def test_cp2k_segment_boundary_records_independent_scf_electronic_reobservation() -> None:
    from vitriflow.workflows.stage_runner import _merge_cp2k_segment_thermo_rows

    rows = [
        (0, 300.0, float("nan"), -10.0, 100.0, 2.0),
        (2, 310.0, 42.0, -9.50, 101.0, 1.98),
        # A restarted, deliberately unconverged SCF re-evaluates electronic
        # energy and pressure at the same nuclear state.
        (2, 310.0, 47.0, -9.25, 101.0, 1.98),
        (4, 320.0, 45.0, -9.00, 102.0, 1.96),
    ]
    segment_ids = [0, 0, 1, 1]

    diagnostics: list[dict[str, object]] = []
    merged = _merge_cp2k_segment_thermo_rows(
        rows,
        segment_ids=segment_ids,
        segment_scf_failures={0: 0, 1: 0},
        segment_labels={0: "seg000.out", 1: "seg001.out"},
        boundary_diagnostics=diagnostics,
    )
    assert [row[0] for row in merged] == [0, 2, 4]
    assert merged[1][2:4] == pytest.approx((42.0, -9.50))
    assert diagnostics == [
        {
            "global_step": 2,
            "preceding_segment": 0,
            "preceding_label": "seg000.out",
            "restart_segment": 1,
            "restart_label": "seg001.out",
            "preceding_unconverged_scf_cycles": 0,
            "restart_unconverged_scf_cycles": 0,
            "electronic_mismatch_allowed": True,
            "electronic_mismatch_basis": "independent_scf_reobservation_at_restart_boundary",
            "canonical_observation": "preceding_propagated_row",
            "fields": {
                "temperature_K": {
                    "preceding": 310.0,
                    "restart": 310.0,
                    "absolute_delta": 0.0,
                    "within_output_roundoff": True,
                    "selected": 310.0,
                },
                "pressure_bar": {
                    "preceding": 42.0,
                    "restart": 47.0,
                    "absolute_delta": 5.0,
                    "within_output_roundoff": False,
                    "selected": 42.0,
                },
                "potential_eV": {
                    "preceding": -9.5,
                    "restart": -9.25,
                    "absolute_delta": 0.25,
                    "within_output_roundoff": False,
                    "selected": -9.5,
                },
                "volume_A3": {
                    "preceding": 101.0,
                    "restart": 101.0,
                    "absolute_delta": 0.0,
                    "within_output_roundoff": True,
                    "selected": 101.0,
                },
                "density_g_cm3": {
                    "preceding": 1.98,
                    "restart": 1.98,
                    "absolute_delta": 0.0,
                    "within_output_roundoff": True,
                    "selected": 1.98,
                },
            },
        }
    ]

    temperature_jump = list(rows)
    temperature_jump[2] = (2, 311.0, 47.0, -9.25, 101.0, 1.98)
    with pytest.raises(ValueError, match=r"field indices \[1\]"):
        _merge_cp2k_segment_thermo_rows(
            temperature_jump,
            segment_ids=segment_ids,
            segment_scf_failures={0: 2, 1: 3},
        )

    with pytest.raises(ValueError, match="within one segment"):
        _merge_cp2k_segment_thermo_rows(
            rows,
            segment_ids=[0, 0, 0, 1],
            segment_scf_failures={0: 3, 1: 0},
        )

    with pytest.raises(ValueError, match="non-adjacent segments"):
        _merge_cp2k_segment_thermo_rows(
            rows,
            segment_ids=[0, 0, 2, 2],
            segment_scf_failures={0: 0, 2: 0},
        )

    with pytest.raises(ValueError, match="more than one restart-side observation"):
        _merge_cp2k_segment_thermo_rows(
            [rows[0], rows[1], rows[2], rows[2]],
            segment_ids=[0, 0, 1, 2],
            segment_scf_failures={0: 0, 1: 0, 2: 0},
        )


def test_cp2k_segment_trajectory_boundary_uses_periodic_roundoff_not_md_tolerance() -> None:
    from vitriflow.workflows.stage_runner import _audit_cp2k_segment_trajectory_boundary

    cell = np.diag([10.0, 11.0, 12.0])
    preceding = np.asarray([[9.999999, 1.0, 2.0], [3.0, 4.0, 5.0]])
    # First atom is represented across the periodic boundary; both frames also
    # differ by a realistic binary32 serialization error.
    restarted = np.asarray([[0.000001, 1.0, 2.0], [3.0, 4.0, 5.0 + 2.0e-6]])
    audit = _audit_cp2k_segment_trajectory_boundary(
        global_step=20,
        preceding_positions=preceding,
        preceding_cell=cell,
        restart_positions=restarted,
        restart_cell=cell + np.diag([1.0e-6, -1.0e-6, 1.0e-6]),
    )
    assert audit["status"] == "continuous"
    assert audit["max_position_delta_A"] < audit["binary32_roundoff_bound_A"]

    discontinuous = restarted.copy()
    discontinuous[1, 0] += 0.01
    with pytest.raises(RuntimeError, match="restart trajectory is discontinuous"):
        _audit_cp2k_segment_trajectory_boundary(
            global_step=20,
            preceding_positions=preceding,
            preceding_cell=cell,
            restart_positions=discontinuous,
            restart_cell=cell,
        )


@pytest.mark.parametrize("bad_geometry", ["position", "cell"])
def test_cp2k_md_renderer_rejects_nonfinite_geometry(bad_geometry: str) -> None:
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.0, pbc=True)
    if bad_geometry == "position":
        atoms.positions[0, 0] = np.nan
    else:
        bad_cell = np.eye(3) * 5.0
        bad_cell[1, 1] = np.inf
        atoms.set_cell(bad_cell)
    cfg = Cp2kConfig(
        kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")}
    )
    with pytest.raises(ValueError, match="must all be finite"):
        render_cp2k_md_input(
            atoms=atoms,
            cfg=cfg,
            md_cfg=MDConfig(ensemble="nvt"),
            ensemble="nvt",
            temperature_K=300.0,
            steps=2,
            timestep_fs=1.0,
            tdamp_fs=100.0,
            project="bad",
            energy_every=1,
            traj_every=1,
            traj_file="traj.dcd",
            ener_file="ener.dat",
        )


def test_cp2k_md_renderer_clamps_energy_stride_to_short_segment() -> None:
    """Short NPT ramp segments must contain a propagated energy sample.

    Pressure is deliberately matched to energy by exact CP2K step number.
    Clamping the segment-local print stride preserves that physical contract
    without assigning a pressure to a different integration step.
    """

    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.config import Cp2kConfig, Cp2kKindConfig, MDConfig
    from vitriflow.cp2k_driver import render_cp2k_md_input

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.0, pbc=True)
    cfg = Cp2kConfig(
        kind_settings={"H": Cp2kKindConfig(basis_set="DZVP", potential="GTH-PBE")}
    )
    text = render_cp2k_md_input(
        atoms=atoms,
        cfg=cfg,
        md_cfg=MDConfig(ensemble="npt"),
        ensemble="npt",
        temperature_K=300.0,
        steps=2,
        timestep_fs=1.0,
        tdamp_fs=100.0,
        pdamp_fs=1000.0,
        project="short_npt",
        energy_every=5,
        traj_every=5,
        traj_file="traj.dcd",
        ener_file="ener.dat",
    )

    energy_block = text.split("&ENERGY", 1)[1].split("&END ENERGY", 1)[0]
    assert "MD 2" in energy_block
    assert "MD 5" not in energy_block


def test_cp2k_scf_and_cell_opt_convergence_are_positive_contracts(tmp_path: Path):
    from vitriflow.cp2k_driver import (
        assert_cp2k_cell_opt_converged,
        assert_cp2k_scf_converged,
        count_cp2k_scf_failures,
    )

    output = tmp_path / "cp2k.out"
    output.write_text("*** SCF run NOT converged ***\n")
    with pytest.raises(RuntimeError, match="unconverged SCF"):
        assert_cp2k_scf_converged(output)
    assert count_cp2k_scf_failures(output) == 1

    output.write_text("PROGRAM ENDED AT\nMAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED\n")
    with pytest.raises(RuntimeError, match="did not report optimisation convergence"):
        assert_cp2k_cell_opt_converged(output)

    # Workflow continuation does not silently relabel the SCF cycle as
    # converged; CELL_OPT acceptance is instead based on CP2K's positive
    # optimisation completion marker and the failure count remains auditable.
    output.write_text(
        "*** SCF run NOT converged ***\n*** GEOMETRY OPTIMIZATION COMPLETED ***\n"
    )
    assert_cp2k_cell_opt_converged(output)
    assert count_cp2k_scf_failures(output) == 1


def test_cp2k_runner_queries_exact_version_on_every_stage_entry(monkeypatch, tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    calls = []

    def fake_run_cmd(cmd, **kwargs):
        calls.append(list(cmd))
        return 0, "CP2K| version string: CP2K version 2024.1\n", ""

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run_cmd)
    runner = Cp2kRunner(
        Cp2kConfig(exec_prefix=["env-wrapper"], cp2k_cmd="cp2k")
    )
    assert runner.query_version(tmp_path) == (2024, 1)
    assert runner.query_version(tmp_path) == (2024, 1)
    assert calls == [
        ["env-wrapper", "cp2k", "--version"],
        ["env-wrapper", "cp2k", "--version"],
    ]


def test_cp2k_runner_refuses_unknown_version_instead_of_guessing(monkeypatch, tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *args, **kwargs: (0, "unparseable build banner", ""),
    )
    runner = Cp2kRunner(
        Cp2kConfig(exec_prefix=["env-wrapper"], cp2k_cmd="cp2k")
    )
    with pytest.raises(RuntimeError, match="Could not query an unambiguous CP2K version"):
        runner.query_version(tmp_path)


def test_cp2k_runner_version_probe_uses_full_prefixed_mpi_command(monkeypatch, tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    calls = []

    def fake_run_cmd(cmd, **kwargs):
        calls.append(list(cmd))
        return (
            0,
            "CP2K version 2024.3 Development Version\nCP2K version 2024.3\n",
            "",
        )

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run_cmd)
    runner = Cp2kRunner(
        Cp2kConfig(
            exec_prefix=["conda", "run", "-n", "cp2k-env"],
            mpi_cmd="mpiexec",
            cp2k_cmd=["cp2k.psmp", "--echo"],
        )
    )
    assert runner.query_version(tmp_path) == (2024, 3)
    assert calls == [[
        "conda", "run", "-n", "cp2k-env", "mpiexec", "-np", "1",
        "cp2k.psmp", "--echo", "--version",
    ]]


def test_cp2k_config_rejects_shell_mode_in_production_command():
    from vitriflow.config import Cp2kConfig

    with pytest.raises(ValueError, match="must not preselect"):
        Cp2kConfig(cp2k_cmd=["cp2k.psmp", "--shell-posix"])


def test_cp2k_runner_rejects_conflicting_version_banners(monkeypatch, tmp_path):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *args, **kwargs: (
            0,
            "wrapper: CP2K version 2023.2\nexecutable: CP2K version 2024.1\n",
            "",
        ),
    )
    runner = Cp2kRunner(
        Cp2kConfig(exec_prefix=["env-wrapper"], cp2k_cmd="cp2k")
    )
    with pytest.raises(RuntimeError, match="conflicting CP2K version strings"):
        runner.query_version(tmp_path)


def test_cp2k_energy_parser_parses_rows_atomically_and_requires_monotonicity(
    tmp_path: Path,
) -> None:
    from vitriflow.cp2k_driver import parse_cp2k_ener

    energy = tmp_path / "run.ener"
    energy.write_text(
        "# Step Time Kin Temp Pot Conserved\n"
        "0 0.0 1.0 300.0 -10.0 -9.0\n"
        "1 0.5 1.1 301.0 -9.9 -8.8\n"
    )
    parsed = parse_cp2k_ener(energy)
    assert parsed.step.tolist() == [0, 1]
    assert parsed.time_fs.tolist() == [0.0, 0.5]
    assert parsed.temperature_K.tolist() == [300.0, 301.0]

    # The old parser appended step/time before failing on temperature, leaving
    # arrays with different lengths that downstream zip() silently truncated.
    energy.write_text(
        "0 0.0 1.0 300.0 -10.0 -9.0\n"
        "1 0.5 1.1 malformed -9.9 -8.8\n"
    )
    with pytest.raises(ValueError, match="Malformed numeric CP2K energy row"):
        parse_cp2k_ener(energy)

    energy.write_text(
        "0 0.0 1.0 300.0 -10.0 -9.0\n"
        "0 0.5 1.1 301.0 -9.9 -8.8\n"
    )
    with pytest.raises(ValueError, match="steps must be strictly increasing"):
        parse_cp2k_ener(energy)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("0 0.0\n1.5 1.0\n2 2.0\n", "nonnegative integer"),
        ("0 0.0\n1 nan\n2 2.0\n", "Malformed numeric MSD row"),
        ("0 0.0\n1 -0.1\n2 2.0\n", "finite and nonnegative"),
        ("0 0.0\n1 1.0\n1 2.0\n", "strictly increasing"),
    ],
)
def test_msd_parser_rejects_fractional_nonfinite_negative_or_duplicate_evidence(
    tmp_path: Path, rows: str, message: str
) -> None:
    from vitriflow.parse import parse_msd_file

    path = tmp_path / "bad.msd.dat"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        parse_msd_file(path)


@pytest.mark.parametrize(
    ("bad_step", "message"),
    [
        ("1.5", "nonnegative integers"),
        ("nan", "Malformed/non-finite thermo data row"),
        ("0", "strictly increasing"),
    ],
)
def test_thermo_parser_rejects_invalid_step_evidence(
    tmp_path: Path, bad_step: str, message: str
) -> None:
    from vitriflow.parse import parse_last_thermo_table

    log = tmp_path / "log.lammps"
    log.write_text(
        "Step Temp Press\n"
        "0 300 0\n"
        f"{bad_step} 301 0\n"
    )
    with pytest.raises(ValueError, match=message):
        parse_last_thermo_table(log)


def test_cp2k_pressure_parser_excludes_input_target_and_preserves_steps(tmp_path: Path):
    from vitriflow.cp2k_driver import parse_cp2k_md_step_pressures

    output = tmp_path / "cp2k.out"
    output.write_text(
        """MD| Pressure [bar] 1.000000
MD| Step number 5
MD| Pressure [bar] 4.200000000000E+01
MD| Step number 7
MD| Pressure [bar] 9.964303380594E+03 7.112549739863E+02
STEP NUMBER = 10
PRESSURE [bar] = -0.611160475868D+02 -0.135378276079E+01
"""
    )
    steps, pressures = parse_cp2k_md_step_pressures(output)
    assert steps.tolist() == [5, 7, 10]
    assert pressures.tolist() == pytest.approx([42.0, 9964.303380594, -61.1160475868])


def test_cp2k_npt_pressure_alignment_fails_closed():
    from vitriflow.cp2k_driver import map_cp2k_pressures_to_energy_steps

    assert map_cp2k_pressures_to_energy_steps([0, 5], [5], [42.0]) == {5: 42.0}
    assert map_cp2k_pressures_to_energy_steps(
        [0, 5, 10], [5, 10], [42.0, 43.0]
    ) == {5: 42.0, 10: 43.0}

    with pytest.raises(RuntimeError, match=r"missing finite pressure.*\[10\]"):
        map_cp2k_pressures_to_energy_steps([0, 5, 10], [5], [42.0])

    with pytest.raises(RuntimeError, match="no finite pressure sample aligned"):
        map_cp2k_pressures_to_energy_steps([0, 5], [6], [42.0])

    with pytest.raises(RuntimeError, match="no finite pressure sample aligned"):
        map_cp2k_pressures_to_energy_steps([0, 5], [5], [float("nan")])

    with pytest.raises(RuntimeError, match="different lengths"):
        map_cp2k_pressures_to_energy_steps([0, 5], [5, 6], [42.0])

    with pytest.raises(RuntimeError, match="nonnegative integer"):
        map_cp2k_pressures_to_energy_steps([0, 5.5], [5], [42.0])

    with pytest.raises(RuntimeError, match="duplicate step"):
        map_cp2k_pressures_to_energy_steps([0, 5], [5, 5], [42.0, 42.0])


def test_cp2k_npt_stage_pressure_completeness_allows_only_initial_global_zero():
    from vitriflow.workflows.stage_runner import _validate_cp2k_npt_pressure_rows

    valid = [
        (0, 300.0, float("nan"), -10.0, 100.0, 2.0),
        (5, 301.0, 42.0, -9.9, 101.0, 1.98),
    ]
    _validate_cp2k_npt_pressure_rows(valid)

    invalid_restart_boundary = [
        (0, 300.0, 41.0, -10.0, 100.0, 2.0),
        (5, 301.0, float("nan"), -9.9, 101.0, 1.98),
    ]
    with pytest.raises(RuntimeError, match="global step 5"):
        _validate_cp2k_npt_pressure_rows(invalid_restart_boundary)


def test_cp2k_thermo_csv_pressure_is_canonical_gpa_while_raw_table_is_bar():
    from vitriflow.parse import ThermoTable
    from vitriflow.workflows.stage_runner import _canonical_cp2k_thermo_table

    raw = ThermoTable(
        columns=["Step", "Temp", "Press", "PotEng", "Volume", "Density"],
        data=np.asarray([[5.0, 1000.0, 2500.0, -123.0, 456.0, 2.5]]),
    )
    canonical = _canonical_cp2k_thermo_table(raw)

    assert raw.as_dict()["Press"].tolist() == pytest.approx([2500.0])
    assert canonical.as_dict()["Press"].tolist() == pytest.approx([0.25])
    assert canonical.as_dict()["PotEng"].tolist() == pytest.approx([-123.0])


def _write_minimal_cp2k_aligned_dcd(
    path: Path,
    *,
    cells: list[float],
    positions: list[list[float]],
) -> None:
    """Write a small standards-shaped CP2K aligned DCD regression fixture."""

    if len(cells) != len(positions):
        raise ValueError("cells and positions must contain the same number of frames")
    header_dtype = np.dtype(
        [
            ("blk0-0", "i4"),
            ("hdr", "S4"),
            ("9int", ("i4", 9)),
            ("timestep", "f4"),
            ("10int", ("i4", 10)),
            ("blk0-1", "i4"),
            ("blk1-0", "i4"),
            ("ntitle", "i4"),
            ("remark1", "S80"),
            ("remark2", "S80"),
            ("blk1-1", "i4"),
            ("blk2-0", "i4"),
            ("natoms", "i4"),
            ("blk2-1", "i4"),
        ]
    )
    frame_dtype = np.dtype(
        [
            ("x0", "i4"),
            ("x1", "f8", (6,)),
            ("x2", "i4", (2,)),
            ("x3", "f4", (1,)),
            ("x4", "i4", (2,)),
            ("x5", "f4", (1,)),
            ("x6", "i4", (2,)),
            ("x7", "f4", (1,)),
            ("x8", "i4"),
        ]
    )
    header = np.zeros(1, dtype=header_dtype)
    header["blk0-0"] = 84
    header["hdr"] = b"CORD"
    header["blk0-1"] = 84
    header["blk1-0"] = 164
    header["ntitle"] = 2
    header["remark1"] = b"CP2K VitriFlow regression trajectory"
    header["blk1-1"] = 164
    header["blk2-0"] = 4
    header["natoms"] = 1
    header["blk2-1"] = 4

    frames = np.zeros(len(cells), dtype=frame_dtype)
    for index, (length, xyz) in enumerate(zip(cells, positions)):
        # CP2K DCD ordering maps to [a, b, c, alpha, beta, gamma].
        frames["x1"][index] = [length, 90.0, length, 90.0, 90.0, length]
        frames["x3"][index, 0] = xyz[0]
        frames["x5"][index, 0] = xyz[1]
        frames["x7"][index, 0] = xyz[2]

    with path.open("wb") as handle:
        header.tofile(handle)
        frames.tofile(handle)


def test_cp2k_dcd_final_frame_reader_reads_real_last_frame(tmp_path: Path):
    pytest.importorskip("ase")
    from ase import Atoms

    from vitriflow.cp2k_driver import read_cp2k_dcd_last_aligned

    trajectory = tmp_path / "traj.dcd"
    _write_minimal_cp2k_aligned_dcd(
        trajectory,
        cells=[4.0, 5.0, 6.0],
        positions=[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
    )
    reference = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0)

    final = read_cp2k_dcd_last_aligned(trajectory, ref_atoms=reference)

    assert np.allclose(final.get_positions(), [[2.1, 2.2, 2.3]])
    assert np.asarray(final.get_cell().lengths()) == pytest.approx([6.0, 6.0, 6.0])


def test_cp2k_dcd_final_frame_reader_requests_aligned_cell(tmp_path: Path, monkeypatch):
    pytest.importorskip("ase")
    from ase import Atoms
    import ase.io.cp2k

    from vitriflow.cp2k_driver import read_cp2k_dcd_last_aligned

    calls: dict[str, object] = {}
    first = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0)
    last = Atoms("Si", positions=[[0.1, 0.2, 0.3]], cell=np.eye(3) * 5.0)

    def fake_reader(handle, *, indices, ref_atoms, aligned):
        calls.update(
            indices=indices,
            ref_atoms=ref_atoms,
            aligned=aligned,
            bytes=handle.read(),
        )
        # The iterator contract for indices=-1 yields only the final frame.
        yield last

    monkeypatch.setattr(ase.io.cp2k, "iread_cp2k_dcd", fake_reader)
    trajectory = tmp_path / "traj.dcd"
    trajectory.write_bytes(b"DCD")
    reference = first

    assert read_cp2k_dcd_last_aligned(trajectory, ref_atoms=reference) is last
    assert calls == {
        "indices": -1,
        "ref_atoms": reference,
        "aligned": True,
        "bytes": b"DCD",
    }


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([], "No frames read"),
        (["empty"], "empty"),
        (["nonfinite"], "invalid coordinates or cell"),
        (["singular"], "invalid coordinates or cell"),
        (["valid", "valid"], "more than one frame"),
    ],
)
def test_cp2k_dcd_final_frame_reader_rejects_invalid_results(
    tmp_path: Path,
    monkeypatch,
    frames: list[str],
    message: str,
):
    pytest.importorskip("ase")
    from ase import Atoms
    import ase.io.cp2k

    from vitriflow.cp2k_driver import read_cp2k_dcd_last_aligned

    class EmptyFrame:
        def __len__(self) -> int:
            return 0

    available = {
        "empty": EmptyFrame(),
        "nonfinite": Atoms(
            "Si",
            positions=[[0.0, 0.0, 0.0]],
            cell=np.asarray([[np.nan, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]),
        ),
        "singular": Atoms(
            "Si",
            positions=[[0.0, 0.0, 0.0]],
            cell=np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]),
        ),
        "valid": Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 5.0),
    }

    def fake_reader(handle, *, indices, ref_atoms, aligned):
        assert indices == -1
        assert aligned is True
        yield from (available[name] for name in frames)

    monkeypatch.setattr(ase.io.cp2k, "iread_cp2k_dcd", fake_reader)
    trajectory = tmp_path / "traj.dcd"
    trajectory.write_bytes(b"DCD")
    reference = available["valid"]

    with pytest.raises(RuntimeError, match=message):
        read_cp2k_dcd_last_aligned(trajectory, ref_atoms=reference)


def test_stage_cp2k_last_frame_reader_uses_shared_validated_path(monkeypatch, tmp_path: Path):
    from vitriflow.workflows import stage_runner

    expected = object()
    calls: dict[str, object] = {}

    def fake_reader(path, *, ref_atoms, aligned):
        calls.update(path=path, ref_atoms=ref_atoms, aligned=aligned)
        return expected

    monkeypatch.setattr(stage_runner, "_read_cp2k_dcd_last_validated", fake_reader)
    trajectory = tmp_path / "traj.dcd"
    reference = object()

    assert (
        stage_runner._read_cp2k_dcd_last(
            trajectory,
            ref_atoms=reference,
            aligned=True,
        )
        is expected
    )
    assert calls == {
        "path": trajectory,
        "ref_atoms": reference,
        "aligned": True,
    }


def test_cp2k_segmented_ramp_reaches_endpoint_at_requested_average_rate():
    from vitriflow.workflows.stage_runner import _build_cp2k_ramp_schedule

    schedule = _build_cp2k_ramp_schedule(
        T_start=2300.0,
        T_stop=300.0,
        total_steps=20,
        max_deltaT_K=100.0,
        max_segments=20,
    )
    temperatures = [temperature for temperature, _steps in schedule]
    steps = [n for _temperature, n in schedule]

    assert sum(steps) == 20
    assert temperatures[-1] == pytest.approx(300.0)
    assert temperatures[0] == pytest.approx(2200.0)
    targets_with_start = [2300.0, *temperatures]
    assert max(abs(b - a) for a, b in zip(targets_with_start, targets_with_start[1:])) <= 100.0
    assert (2300.0 - temperatures[-1]) / sum(steps) == pytest.approx(100.0)


def test_lammps_tail_split_uses_one_global_temperature_ramp(tmp_path: Path):
    from vitriflow.config import KimConfig, MDConfig
    from vitriflow.lammps_input import StageSpec, render_stage

    data = tmp_path / "input.data"
    _minimal_datafile(data)
    stage = StageSpec(
        name="quench",
        input_data=data,
        output_data=tmp_path / "output.data",
        temperature_start=1000.0,
        temperature_stop=300.0,
        pressure=0.0,
        equil_steps=0,
        run_steps=100,
        seed=123,
        write_dump=True,
        tail_dump_frames=2,
        tail_dump_stride=10,
    )
    script = render_stage(
        KimConfig(model="TEST_MODEL", interactions=["Si"]),
        MDConfig(ensemble="nvt"),
        stage,
    )

    assert "run 80 start 0 stop 100" in script
    assert "run 20 start 0 stop 100" in script


def test_cp2k_msd_removes_mass_weighted_center_of_mass_translation():
    from vitriflow.cp2k_driver import compute_msd

    positions = np.asarray(
        [
            [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    msd = compute_msd(
        positions,
        np.eye(3) * 20.0,
        unwrap=True,
        masses=np.asarray([1.0, 10.0]),
        remove_com=True,
    )
    assert msd.tolist() == pytest.approx([0.0, 0.0], abs=1.0e-14)


def test_cp2k_msd_legacy_default_keeps_center_of_mass_translation():
    from vitriflow.cp2k_driver import compute_msd

    positions = np.asarray(
        [
            [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    msd = compute_msd(positions, np.eye(3) * 20.0, unwrap=True)
    assert msd.tolist() == pytest.approx([0.0, 14.0])


@pytest.mark.parametrize(
    ("units_style", "expected"),
    [
        ("metal", 5.0),
        ("real", 5000.0),
        ("nano", 0.005),
        ("si", 5.0e-12),
        ("electron", 5000.0),
    ],
)
def test_pdamp_minimum_is_converted_from_ps_to_native_time(units_style: str, expected: float):
    from vitriflow.workflows.preflight import _duration_ps_to_lammps_time

    assert _duration_ps_to_lammps_time(5.0, units_style) == pytest.approx(expected)


def test_cp2k_npt_pressure_screen_requires_finite_bounded_samples():
    from vitriflow.workflows.preflight import _screen_cp2k_pressure_samples

    missing_ok, _ = _screen_cp2k_pressure_samples(
        [float("nan")],
        target_bar=0.0,
        tail_window=10,
        max_abs_bar=1.0e6,
        tolerance_bar=100.0,
        require_tolerance=False,
    )
    exploded_ok, _ = _screen_cp2k_pressure_samples(
        [0.0, 2.0e6],
        target_bar=0.0,
        tail_window=10,
        max_abs_bar=1.0e6,
        tolerance_bar=100.0,
        require_tolerance=False,
    )
    off_target_ok, details = _screen_cp2k_pressure_samples(
        [1000.0, 1100.0, 1200.0],
        target_bar=0.0,
        tail_window=2,
        max_abs_bar=1.0e6,
        tolerance_bar=100.0,
        require_tolerance=True,
    )

    assert missing_ok is False
    assert exploded_ok is False
    assert off_target_ok is False
    assert details["P_mean"] == pytest.approx(1150.0)


def test_quench_steps_reject_heating_or_zero_delta_temperature():
    from vitriflow.workflows.quench_rates import quench_steps_for_rate

    with pytest.raises(ValueError, match="start temperature must exceed final"):
        quench_steps_for_rate(0.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="start temperature must exceed final"):
        quench_steps_for_rate(-100.0, 10.0, 1.0)


def test_custom_schedule_rejects_ambiguous_duration_and_non_cooling_quench():
    from vitriflow.workflows.custom_schedule import _schedule_from_raw, _validate_schedule

    raw = {
        "custom_schedule": {
            "stages": [
                {"name": "melt", "temperature_K": 1000.0, "steps": 10, "role": "melt"},
                {
                    "name": "quench",
                    "temperature_start_K": 1000.0,
                    "temperature_stop_K": 300.0,
                    "steps": 10,
                    "time_ps": 1.0,
                    "role": "quench",
                },
                {"name": "relax", "temperature_K": 300.0, "steps": 10, "role": "relax"},
            ]
        }
    }
    with pytest.raises(ValueError, match="either time_ps or steps, not both"):
        _schedule_from_raw(raw)

    del raw["custom_schedule"]["stages"][1]["time_ps"]
    raw["custom_schedule"]["stages"][1]["temperature_stop_K"] = 1100.0
    raw["custom_schedule"]["stages"][2]["temperature_K"] = 1100.0
    with pytest.raises(ValueError, match="quench stage must cool"):
        _validate_schedule(_schedule_from_raw(raw))


def test_reused_directory_does_not_retain_stale_engine_neutral_trajectory(tmp_path: Path):
    from vitriflow.config import MDConfig
    from vitriflow.workflows.stage_runner import _materialize_lammps_engine_neutral_outputs

    output_data = tmp_path / "output.data"
    _minimal_datafile(output_data)
    stale = tmp_path / "traj.extxyz"
    stale.write_text("stale trajectory\n")
    (tmp_path / "final.extxyz").write_text("stale final frame\n")

    trajectory, final_frame = _materialize_lammps_engine_neutral_outputs(
        stage_dir=tmp_path,
        output_data=output_data,
        dump_path=None,
        md_cfg=MDConfig(atom_style="atomic"),
        type_to_species=["H"],
    )

    assert trajectory is None
    assert not stale.exists()
    assert final_frame.is_file()
    assert "stale final frame" not in final_frame.read_text()


def test_cp2k_dump_disabled_removes_stale_engine_neutral_trajectory(tmp_path: Path):
    from vitriflow.workflows.stage_runner import _materialize_cp2k_engine_neutral_outputs

    stale = tmp_path / "traj.extxyz"
    stale.write_text("stale trajectory\n")
    (tmp_path / "final.extxyz").write_text("stale final frame\n")

    trajectory, final_frame = _materialize_cp2k_engine_neutral_outputs(
        stage_dir=tmp_path,
        steps_all=np.asarray([10], dtype=int),
        pos_all=np.asarray([[[1.0, 2.0, 3.0]]]),
        cells_all=np.asarray([np.eye(3) * 10.0]),
        symbols=["H"],
        type_to_species=["H"],
        selected_out=[],
        write_dump=False,
    )

    assert trajectory is None
    assert not stale.exists()
    assert final_frame.is_file()
    assert "stale final frame" not in final_frame.read_text()


def test_cp2k_project_outputs_are_cleared_before_current_invocation(tmp_path: Path):
    from vitriflow.workflows.stage_runner import _clear_cp2k_project_outputs

    names = [
        "traj.dcd",
        "ener.dat",
        "cp2k.out",
        "melt_seg000-1.restart",
        "melt_seg000-RESTART.wfn",
        "melt_seg000-RESTART.wfn.bak-1",
        "melt_seg000-RESTART.wfn.bak-2",
    ]
    for name in names[:-1]:
        (tmp_path / name).write_text("stale\n")
    outside = tmp_path / "outside.wfn"
    outside.write_text("must survive\n")
    (tmp_path / names[-1]).symlink_to(outside)

    _clear_cp2k_project_outputs(
        tmp_path,
        project="melt_seg000",
        trajectory_file="traj.dcd",
        energy_file="ener.dat",
        output_file="cp2k.out",
    )

    assert all(not (tmp_path / name).exists() for name in names)
    assert all(not (tmp_path / name).is_symlink() for name in names)
    assert outside.read_text() == "must survive\n"


def test_cp2k_project_cleanup_fails_closed_if_stale_artifact_cannot_be_unlinked(
    tmp_path: Path,
):
    from vitriflow.workflows.stage_runner import _clear_cp2k_project_outputs

    stale_directory = tmp_path / "melt_seg000-RESTART.wfn.bak-1"
    stale_directory.mkdir()

    with pytest.raises(RuntimeError, match="Cannot remove stale CP2K project artifact"):
        _clear_cp2k_project_outputs(
            tmp_path,
            project="melt_seg000",
            trajectory_file="traj.dcd",
            energy_file="ener.dat",
            output_file="cp2k.out",
        )
