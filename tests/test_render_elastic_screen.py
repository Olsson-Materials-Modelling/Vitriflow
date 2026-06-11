from __future__ import annotations

from pathlib import Path


def test_render_elastic_screen_emits_born_and_stress_commands(tmp_path: Path):
    from vitriflow.config import LammpsPotentialConfig, MDConfig
    from vitriflow.lammps_input import render_elastic_screen

    pot = LammpsPotentialConfig(
        interactions=["X"],
        commands=[
            "pair_style lj/cut 2.5",
            "pair_coeff * * 1.0 1.0 2.5",
        ],
    )
    md = MDConfig(atom_style="atomic")

    script = render_elastic_screen(
        pot,
        md,
        input_data=tmp_path / "input.data",
        born_delta=1.0e-5,
        raw_output_name="born_raw.txt",
        stress_dump_name="local_stress.dump",
    )

    assert "compute born all born/matrix numdiff" in script
    assert "compute pst all stress/atom NULL virial" in script
    assert "variable S4 equal -c_vir[6]" in script
    assert "variable S5 equal -c_vir[5]" in script
    assert "variable S6 equal -c_vir[4]" in script
    assert "write_dump all custom local_stress.dump" in script
    assert "variable VOL equal vol" in script
    assert "print \"vol=${VOL}" in script
