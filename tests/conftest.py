from __future__ import annotations

import sys
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_engine_build_identities(monkeypatch):
    """Inject authenticated deterministic builds into engine-free unit tests.

    Workflow unit tests mock LAMMPS/CP2K execution and intentionally run on CI
    hosts without either binary.  Tests of the real fail-closed probe live in
    ``test_engine_build_identity.py`` and do not use this fixture.
    """

    import vitriflow.engine_identity as engine_identity

    executable = Path(sys.executable).resolve(strict=True)
    executable_identity = engine_identity.stable_file_identity(
        executable, reject_final_symlink=True
    )

    def one(engine: str):
        return engine_identity._seal_identity(
            {
                "schema": engine_identity.ENGINE_BUILD_IDENTITY_SCHEMA,
                "status": "verified",
                "engine": str(engine),
                "configured_engine_command": [str(executable)],
                "resolved_engine_command": [str(executable)],
                "configured_execution_command": [str(executable)],
                "resolved_execution_command": [str(executable)],
                "probe": {
                    "flag": "-h" if engine == "lammps" else "--version",
                    "configured_command": [str(executable)],
                    "resolved_command": [str(executable)],
                    "release_banner": "mock" if engine == "lammps" else None,
                    "version": "2024.2" if engine == "cp2k" else None,
                    "output": {
                        "combined_sha256": "0" * 64,
                    },
                },
                "command_file_identities": [
                    {
                        "role": "engine_command",
                        "token_index": 0,
                        "configured_token": str(executable),
                        "resolved_path": str(executable),
                        "size_bytes": int(executable_identity["size_bytes"]),
                        "sha256": str(executable_identity["sha256"]),
                    }
                ],
            }
        )

    identities = {"lammps": one("lammps"), "cp2k": one("cp2k")}

    def fake_bundle(config, *, primary_engine=None, include_cp2k_refinement=False, **kwargs):
        primary = str(primary_engine or config.engine).strip().lower()
        selected = {primary: identities[primary]}
        if include_cp2k_refinement:
            selected["cp2k"] = identities["cp2k"]
        return engine_identity.engine_build_identity_bundle(
            primary_engine=primary,
            identities=selected,
        )

    monkeypatch.setattr(engine_identity, "query_engine_build_identities", fake_bundle)
    for module_name in (
        "vitriflow.workflows.run",
        "vitriflow.workflows.autotune",
        "vitriflow.workflows.hpc",
        "vitriflow.workflows.custom_schedule",
    ):
        module = importlib.import_module(module_name)
        if module is not None and hasattr(module, "query_engine_build_identities"):
            monkeypatch.setattr(module, "query_engine_build_identities", fake_bundle)
        if module is not None and module_name.endswith(".run"):
            original = module._build_run_resume_fingerprint

            def build_run(*args, __original=original, **kwargs):
                if kwargs.get("engine_build_identities") is None:
                    config = kwargs["config"]
                    plan = kwargs["production_plan"]
                    kwargs["engine_build_identities"] = fake_bundle(
                        config,
                        primary_engine=str(plan.get("engine", config.engine)),
                    )
                return __original(*args, **kwargs)

            monkeypatch.setattr(module, "_build_run_resume_fingerprint", build_run)
        if module is not None and module_name.endswith(".autotune"):
            original = module._build_autotune_resume_fingerprint

            def build_autotune(*args, __original=original, **kwargs):
                if kwargs.get("engine_build_identities") is None:
                    config = kwargs["config"]
                    kwargs["engine_build_identities"] = fake_bundle(
                        config,
                        primary_engine=str(config.engine),
                        include_cp2k_refinement=bool(
                            getattr(
                                getattr(
                                    getattr(config.autotune, "production", None),
                                    "dft_opt",
                                    None,
                                ),
                                "enabled",
                                False,
                            )
                        ),
                    )
                return __original(*args, **kwargs)

            monkeypatch.setattr(
                module, "_build_autotune_resume_fingerprint", build_autotune
            )
        if module is not None and module_name.endswith(".custom_schedule"):
            original = module._build_resume_fingerprint

            def build_custom(*args, __original=original, **kwargs):
                if kwargs.get("engine_build_identities") is None:
                    config = kwargs["config"]
                    kwargs["engine_build_identities"] = fake_bundle(
                        config,
                        primary_engine="lammps",
                    )
                return __original(*args, **kwargs)

            monkeypatch.setattr(module, "_build_resume_fingerprint", build_custom)
    return {
        "identities": identities,
        "bundle": fake_bundle,
    }
