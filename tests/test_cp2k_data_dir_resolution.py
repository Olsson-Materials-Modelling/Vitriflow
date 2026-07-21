from __future__ import annotations

import pytest


def _runner(*, data_dir=None, cp2k_cmd="cp2k"):
    from vitriflow.config import Cp2kConfig
    from vitriflow.runner import Cp2kRunner

    cfg = Cp2kConfig(
        data_dir=data_dir,
        cp2k_cmd=cp2k_cmd,
        basis_set_file_name="BASIS_MOLOPT",
        potential_file_name="GTH_POTENTIALS",
    )
    return Cp2kRunner(cfg)


def test_cp2k_data_dir_precedence_is_config_then_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    configured.mkdir()
    environment.mkdir()
    monkeypatch.setenv("CP2K_DATA_DIR", str(environment))

    runner = _runner(data_dir=str(configured))
    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("executable detection must not run")),
    )

    assert runner._detect_data_dir(tmp_path) == configured


def test_cp2k_environment_precedes_executable_detection(monkeypatch, tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    monkeypatch.setenv("CP2K_DATA_DIR", str(environment))

    runner = _runner()
    monkeypatch.setattr(
        "vitriflow.runner.run_cmd",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("executable detection must not run")),
    )

    assert runner._detect_data_dir(tmp_path) == environment


def test_cp2k_data_dir_cache_invalidates_when_environment_changes(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    runner = _runner()

    monkeypatch.setenv("CP2K_DATA_DIR", str(first))
    assert runner._detect_data_dir(tmp_path) == first

    monkeypatch.setenv("CP2K_DATA_DIR", str(second))
    assert runner._detect_data_dir(tmp_path) == second


def test_cp2k_data_dir_cache_rechecks_removed_directory(monkeypatch, tmp_path):
    detected = tmp_path / "detected"
    detected.mkdir()
    monkeypatch.delenv("CP2K_DATA_DIR", raising=False)
    runner = _runner()
    calls = 0

    def fake_run_cmd(*_a, **_k):
        nonlocal calls
        calls += 1
        if detected.is_dir():
            return 0, f'__DATA_DIR="{detected}"\n', ""
        return 0, "", ""

    monkeypatch.setattr("vitriflow.runner.run_cmd", fake_run_cmd)
    assert runner._detect_data_dir(tmp_path) == detected
    detected.rmdir()
    assert runner._detect_data_dir(tmp_path) is None
    assert calls == 2


def test_cp2k_staging_uses_configured_files_not_packaged_or_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    packaged = tmp_path / "packaged"
    workdir = tmp_path / "work"
    for directory in (configured, environment, packaged, workdir):
        directory.mkdir()
    for directory, marker in (
        (configured, "configured"),
        (environment, "environment"),
        (packaged, "packaged"),
    ):
        (directory / "BASIS_MOLOPT").write_text(f"basis-{marker}\n")
        (directory / "GTH_POTENTIALS").write_text(f"potential-{marker}\n")

    monkeypatch.setenv("CP2K_DATA_DIR", str(environment))
    runner = _runner(data_dir=str(configured))
    monkeypatch.setattr(runner, "_packaged_data_dir", lambda: packaged)

    resolved = runner.resolved_data_files(workdir)
    assert resolved["basis_set"]["resolved_path"] == str((configured / "BASIS_MOLOPT").resolve())
    assert resolved["potential"]["resolved_path"] == str((configured / "GTH_POTENTIALS").resolve())

    runner._ensure_data_files_present(workdir)
    assert (workdir / "BASIS_MOLOPT").read_text() == "basis-configured\n"
    assert (workdir / "GTH_POTENTIALS").read_text() == "potential-configured\n"


def test_cp2k_staging_does_not_unlink_data_files_when_data_dir_is_workdir(tmp_path):
    (tmp_path / "BASIS_MOLOPT").write_text("basis-in-place\n")
    (tmp_path / "GTH_POTENTIALS").write_text("potential-in-place\n")
    runner = _runner(data_dir=str(tmp_path))

    runner._ensure_data_files_present(tmp_path)

    assert (tmp_path / "BASIS_MOLOPT").read_text() == "basis-in-place\n"
    assert (tmp_path / "GTH_POTENTIALS").read_text() == "potential-in-place\n"


def test_cp2k_path_qualified_data_names_must_be_absolute_for_direct_models():
    from pydantic import ValidationError

    from vitriflow.config import Cp2kConfig

    with pytest.raises(ValidationError, match="must be absolute"):
        Cp2kConfig(basis_set_file_name="relative/BASIS_MOLOPT")


def test_cp2k_yaml_resolves_relative_data_paths_against_yaml_directory(tmp_path):
    from vitriflow.config import RunConfig

    data_dir = tmp_path / "cp2k-data"
    data_dir.mkdir()
    basis = data_dir / "CUSTOM_BASIS"
    potential = data_dir / "CUSTOM_POTENTIALS"
    basis.write_text("basis\n")
    potential.write_text("potential\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "engine: cp2k\n"
        "cp2k:\n"
        "  data_dir: cp2k-data\n"
        "  basis_set_file_name: cp2k-data/CUSTOM_BASIS\n"
        "  potential_file_name: cp2k-data/CUSTOM_POTENTIALS\n"
        "  kind_settings:\n"
        "    H: {basis_set: DZVP-MOLOPT-SR-GTH, potential: GTH-PBE}\n"
        "structure:\n"
        "  generate: {method: random, formula: H, n_formula_units: 1}\n"
        "autotune:\n"
        "  metrics: {enabled: false, type_to_species: [H]}\n"
    )

    config = RunConfig.from_yaml(config_path)

    assert config.cp2k is not None
    assert config.cp2k.data_dir == str(data_dir.resolve())
    assert config.cp2k.basis_set_file_name == str(basis.resolve())
    assert config.cp2k.potential_file_name == str(potential.resolve())


def test_explicit_cp2k_data_dir_does_not_silently_fall_back(monkeypatch, tmp_path):
    import pytest

    configured = tmp_path / "configured"
    packaged = tmp_path / "packaged"
    configured.mkdir()
    packaged.mkdir()
    (packaged / "BASIS_MOLOPT").write_text("packaged\n")
    (packaged / "GTH_POTENTIALS").write_text("packaged\n")
    runner = _runner(data_dir=str(configured))
    monkeypatch.setattr(runner, "_packaged_data_dir", lambda: packaged)

    with pytest.raises(FileNotFoundError, match="configured cp2k.data_dir"):
        runner.resolved_data_files(tmp_path)


def test_disappeared_configured_data_dir_never_falls_back_to_environment(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    configured.mkdir()
    environment.mkdir()
    for name in ("BASIS_MOLOPT", "GTH_POTENTIALS"):
        (configured / name).write_text(f"configured-{name}\n")
        (environment / name).write_text(f"environment-{name}\n")
    runner = _runner(data_dir=str(configured))
    monkeypatch.setenv("CP2K_DATA_DIR", str(environment))

    assert runner._detect_data_dir(tmp_path) == configured
    for path in configured.iterdir():
        path.unlink()
    configured.rmdir()

    assert runner._detect_data_dir(tmp_path) is None
    with pytest.raises(FileNotFoundError, match="configured cp2k.data_dir"):
        runner._ensure_data_files_present(tmp_path / "work")


def test_missing_configured_source_removes_stale_staged_file(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    work = tmp_path / "work"
    environment = tmp_path / "environment"
    configured.mkdir()
    work.mkdir()
    environment.mkdir()
    (configured / "BASIS_MOLOPT").write_text("authoritative-basis\n")
    (configured / "GTH_POTENTIALS").write_text("authoritative-potential\n")
    runner = _runner(data_dir=str(configured))
    runner._ensure_data_files_present(work)

    # Replace staged links with ordinary stale files, then remove one exact
    # source.  Neither the stale workdir copy nor an environment fallback may
    # be consumed.
    for name in ("BASIS_MOLOPT", "GTH_POTENTIALS"):
        staged = work / name
        staged.unlink()
        staged.write_text(f"stale-{name}\n")
        (environment / name).write_text(f"fallback-{name}\n")
    (configured / "BASIS_MOLOPT").unlink()
    monkeypatch.setenv("CP2K_DATA_DIR", str(environment))

    with pytest.raises(FileNotFoundError, match="configured cp2k.data_dir"):
        runner._ensure_data_files_present(work)
    assert not (work / "BASIS_MOLOPT").exists()
    assert not (work / "GTH_POTENTIALS").exists()
