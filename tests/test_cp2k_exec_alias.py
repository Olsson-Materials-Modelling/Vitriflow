import pytest

from vitriflow.config import Cp2kConfig


def test_cp2k_exec_alias_sets_cp2k_cmd():
    cfg = Cp2kConfig.model_validate({"exec": "cp2k"})
    assert cfg.cp2k_cmd == "cp2k"


def test_cp2k_command_alias_sets_cp2k_cmd():
    cfg = Cp2kConfig.model_validate({"command": ["srun", "cp2k.psmp"]})
    assert cfg.cp2k_cmd == ["srun", "cp2k.psmp"]
