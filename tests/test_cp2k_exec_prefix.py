from vitriflow.config import Cp2kConfig


def test_cp2k_exec_prefix_sanitizes_tokens():
    cfg = Cp2kConfig(exec_prefix=["conda", "run", "", "-n", "vitriflow-cp2k", "  ", "--no-capture-output"])
    assert cfg.exec_prefix == ["conda", "run", "-n", "vitriflow-cp2k", "--no-capture-output"]
