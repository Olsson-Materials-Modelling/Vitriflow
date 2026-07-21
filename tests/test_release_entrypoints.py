from __future__ import annotations

import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_module_entrypoint_exists():
    import vitriflow.__main__ as main_module

    assert hasattr(main_module, "main")


def test_source_tree_launcher_and_release_installer_exist_and_are_executable():
    launcher = ROOT / "bin" / "vitriflow"
    installer = ROOT / "install_release.sh"
    for path in (launcher, installer):
        assert path.is_file()
        assert path.stat().st_mode & stat.S_IXUSR
    assert "PYTHONPATH" in launcher.read_text()
    installer_text = installer.read_text()
    assert "pip install" in installer_text
    assert "TemporaryDirectory" in installer_text


def test_source_distribution_manifest_contains_release_contract_and_docs():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "include install_release.sh" in manifest
    assert "recursive-include bin *" in manifest
    assert "recursive-include docs *.md" in manifest


def test_cli_version_is_wired():
    import pytest
    from vitriflow.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
