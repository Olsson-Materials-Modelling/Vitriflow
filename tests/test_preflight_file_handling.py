from __future__ import annotations

from pathlib import Path


def _kim_cfg(input_data: Path):
    from vitriflow.config import RunConfig

    return RunConfig.model_validate(
        {
            "potential": {
                "kind": "kim",
                "model": "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
                "user_units": "metal",
                "interactions": ["Al"],
            },
            "structure": {"lammps_data": str(input_data)},
            "autotune": {"metrics": {"enabled": True, "type_to_species": ["Al"], "pairs": [{"pair": ["Al", "Al"]}] }},
        }
    )


def test_kim_extract_commands_preserves_localized_input_bytes_when_no_pair_coeffs(tmp_path: Path):
    from vitriflow.workflows.preflight import _kim_extract_commands

    input_data = tmp_path / "input.data"
    src_bytes = (
        b"LAMMPS data file via test\r\n\r\n"
        b"1 atoms\r\n"
        b"1 atom types\r\n\r\n"
        b"0.0 1.0 xlo xhi\r\n"
        b"0.0 1.0 ylo yhi\r\n"
        b"0.0 1.0 zlo zhi\r\n\r\n"
        b"Masses\r\n\r\n"
        b"1 26.9815385\r\n\r\n"
        b"Atoms # atomic\r\n\r\n"
        b"1 1 0.5 0.5 0.5\r\n"
    )
    input_data.write_bytes(src_bytes)
    cfg = _kim_cfg(input_data)

    class FakeRunner:
        def run(self, script, workdir, log_name, timeout_sec=None):
            localized = Path(workdir) / "input.data"
            assert localized.read_bytes() == src_bytes
            (Path(workdir) / log_name).write_text(
                """
LAMMPS (29 Aug 2024)
Begin KIM Interactions
pair_style kim test
pair_coeff * *
End KIM Interactions
""".lstrip()
            )

    cmds = _kim_extract_commands(FakeRunner(), cfg, input_data, tmp_path)
    assert cmds == ["pair_style kim test", "pair_coeff * *"]
