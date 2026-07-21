from __future__ import annotations

import re
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = r"[0-9]+(?:\.[0-9]+)+"


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_project_and_runtime_versions_match():
    import vitriflow

    assert vitriflow.__version__ == _project_version()


def test_active_release_and_install_documents_match_project_version():
    # Clean layout: README.md is the single active install/release document.
    # The historical README_INSTALL.md / INSTALL_RELEASE.md duplicates were
    # dropped, so version coherence is checked against README.md alone.
    version = _project_version()
    readme = (ROOT / "README.md").read_text()
    current = re.search(
        rf"^Current packaged release:\s*`({VERSION_PATTERN})`\s*$",
        readme,
        flags=re.MULTILINE,
    )
    assert current is not None
    assert current.group(1) == version

    wheel_versions = set(
        re.findall(rf"dist/vitriflow-({VERSION_PATTERN})-py3-none-any\.whl", readme)
    )
    assert wheel_versions == {version}, "README.md contains non-current wheel references"


def test_generated_distribution_metadata_matches_project_version():
    pkg_info = ROOT / "vitriflow.egg-info" / "PKG-INFO"
    if not pkg_info.is_file():
        return
    match = re.search(r"^Version:\s*(\S+)\s*$", pkg_info.read_text(), re.MULTILINE)
    assert match is not None
    assert match.group(1) == _project_version()


def test_installer_derives_version_and_wheel_name_from_pyproject():
    script_path = ROOT / "install_release.sh"
    script = script_path.read_text()

    assert re.search(r'^VERSION="\$\(', script, flags=re.MULTILINE)
    assert re.search(
        r'^WHEEL="\$\{ROOT\}/dist/vitriflow-\$\{VERSION\}-py3-none-any\.whl"$',
        script,
        flags=re.MULTILINE,
    )
    assert not re.search(
        rf'^WHEEL=.*vitriflow-{VERSION_PATTERN}-py3-none-any\.whl',
        script,
        flags=re.MULTILINE,
    )
    assert 'INSTALL_MODE="wheel"' in script
    assert 'TemporaryDirectory(prefix="vitriflow-install-check-")' in script
    assert "Wheel validation imported the source tree" in script
    subprocess.run(["bash", "-n", str(script_path)], check=True, cwd=ROOT)

    extraction = re.search(
        r'^VERSION=.*?<<\'PY\'\n(?P<code>.*?)\nPY\n\)"$',
        script,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert extraction is not None
    completed = subprocess.run(
        [sys.executable, "-", str(ROOT / "pyproject.toml")],
        input=extraction.group("code"),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    assert completed.stdout.strip() == _project_version()
    assert completed.stderr == ""


def test_analyze_output_reports_runtime_package_version(tmp_path):
    import vitriflow
    from vitriflow.workflows.output_analysis import (
        analysis_context_from_standalone_config,
        analyze_output_data,
    )

    input_dir = tmp_path / "empty_ensemble"
    input_dir.mkdir()
    context = analysis_context_from_standalone_config(
        {
            "metrics": {"enabled": False},
            "production": {"check_convergence": False},
        }
    )
    result = analyze_output_data(
        analysis_context=context,
        input_path=input_dir,
        outdir=tmp_path / "analysis",
    )

    assert result["software_versions"]["vitriflow"] == vitriflow.__version__
