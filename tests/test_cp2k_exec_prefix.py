import pytest

from vitriflow.config import Cp2kConfig


def test_cp2k_exec_prefix_preserves_exact_tokens():
    cfg = Cp2kConfig(
        exec_prefix=["conda", "run", "-n", "vitriflow-cp2k", "--no-capture-output"]
    )
    assert cfg.exec_prefix == ["conda", "run", "-n", "vitriflow-cp2k", "--no-capture-output"]


@pytest.mark.parametrize("bad", [["conda", ""], ["conda", "  "], [" conda"]])
def test_cp2k_exec_prefix_rejects_ambiguous_tokens(bad):
    with pytest.raises(ValueError, match="exact non-empty tokens"):
        Cp2kConfig(exec_prefix=bad)
