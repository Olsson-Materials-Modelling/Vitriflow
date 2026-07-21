"""Pin the production-runner shim's status as TRANSITIONAL, not resolved.

Targeted regression for review finding #4. The earlier patch added
``production_common.run_production_ensemble`` as a thin wrapper around
``autotune._run_production_ensemble`` so ``run.py`` would stop importing
autotune internals directly. That move is **not** an architecture fix --
the cross-runner dependency still exists, just behind one indirection.

Reviewer instruction: leave the finding open, document the shim
explicitly, do NOT claim it as a fix. This file pins:

* the docstring continues to call the function a transitional shim and
  explicitly disclaims being an architecture fix;
* the implementation still lives in ``autotune.py`` (so any "we moved
  it" claim is accompanied by an actual move);
* ``run.py`` does not reach back into ``autotune`` for the ensemble runner.

When the implementation is genuinely migrated to ``production_common``,
update *this* test alongside the move -- the test failing in isolation
means the shim was relabelled without doing the work.
"""

from __future__ import annotations

import inspect


def test_shim_docstring_states_open_finding():
    """The docstring must keep an explicit transitional / open-finding banner."""
    from vitriflow.workflows import production_common

    fn = production_common.run_production_ensemble
    doc = (fn.__doc__ or "")

    # Non-negotiable phrases. If a future edit drops any of these, the test
    # fails and the editor is forced to reckon with the architectural debt.
    required = [
        "TRANSITIONAL SHIM",
        "OPEN",
        "DO NOT describe this as an architecture fix",
    ]
    for needle in required:
        assert needle in doc, (
            f"production_common.run_production_ensemble docstring must contain "
            f"{needle!r}; finding #4 requires the shim to advertise its status. "
            "If you are migrating the implementation, remove the shim instead "
            "of softening the docstring."
        )


def test_shim_docstring_does_not_overclaim_compliance():
    """The docstring must not claim the project design guide's separation rule is satisfied."""
    from vitriflow.workflows import production_common

    fn = production_common.run_production_ensemble
    doc = (fn.__doc__ or "").lower()

    # Phrases from the previous, overclaiming version of the docstring. These
    # were the wording the reviewer flagged as falsely framing the shim as a
    # rule-compliant fix.
    forbidden = [
        "no longer imports from `autotune`",
        "stable interface in `production_common` so it no longer",
        "satisfies the architecture rule",
    ]
    for needle in forbidden:
        assert needle.lower() not in doc, (
            f"docstring contains {needle!r}, which overclaims the project design guide "
            "compliance. Finding #4 explicitly forbids this framing."
        )


def test_implementation_still_lives_in_autotune():
    """The runner class is still defined in autotune; pin the open finding.

    If this test ever fails, it means someone moved the implementation. In
    that case: delete this test and the shim, and update the project design guide to mark
    the finding closed. Do NOT silently weaken this assertion.
    """
    from vitriflow.workflows import autotune

    assert hasattr(autotune, "_ProductionEnsembleRunner"), (
        "_ProductionEnsembleRunner must still be defined in autotune.py until "
        "the shim is replaced with a real implementation in production_common."
    )
    assert hasattr(autotune, "_run_production_ensemble"), (
        "_run_production_ensemble must still be defined in autotune.py until "
        "the shim is replaced with a real implementation in production_common."
    )


def test_shim_body_lazy_imports_from_autotune():
    """The shim's contract is: it forwards to autotune via a lazy import.

    A future "fix" that hides the dependency without moving the code (e.g.
    by re-exporting through some other indirection) is exactly what the
    reviewer warned against. This test ensures the call path stays
    obvious so reviewers can grep for it.
    """
    from vitriflow.workflows import production_common

    src = inspect.getsource(production_common.run_production_ensemble)
    assert "from .autotune import _run_production_ensemble" in src, (
        "The shim must continue to lazy-import directly from autotune; that "
        "explicit dependency is the honest signal that the architectural "
        "finding is unresolved. Hiding it behind another indirection only "
        "obscures the debt."
    )
    # And the call itself: not a re-export of the function object, an actual
    # delegated call so the dependency is observable in tracebacks.
    assert "return _run_production_ensemble(**kwargs)" in src


def test_run_py_does_not_import_run_production_ensemble_directly():
    """run.py must not regress to importing the autotune symbol.

    The shim's only useful property is centralising the cross-runner call
    in production_common. If run.py starts importing
    ``_run_production_ensemble`` again, even the indirection benefit is gone.

    Implemented via AST so docstrings and comments that legitimately mention
    the symbol (this very review trail does!) don't trigger the regression.
    """
    import ast

    from vitriflow.workflows import run as run_mod

    tree = ast.parse(inspect.getsource(run_mod))
    forbidden_names = {"_run_production_ensemble", "_ProductionEnsembleRunner"}

    bad_imports: list[str] = []
    bad_attr_accesses: list[str] = []

    for node in ast.walk(tree):
        # `from .autotune import _run_production_ensemble` (or absolute form)
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "")
            if mod.endswith("autotune"):
                for alias in node.names:
                    if alias.name in forbidden_names:
                        bad_imports.append(f"{mod}.{alias.name}")
        # `autotune._run_production_ensemble` (attribute access form)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            base = node.value
            base_name = getattr(base, "id", None) or getattr(base, "attr", None)
            if base_name == "autotune":
                bad_attr_accesses.append(f"autotune.{node.attr}")

    assert not bad_imports, (
        f"run.py imports {bad_imports} from autotune; the shim's only purpose "
        "was to keep this file out of autotune internals. Route through "
        "production_common.run_production_ensemble instead."
    )
    assert not bad_attr_accesses, (
        f"run.py accesses {bad_attr_accesses} as attributes; same regression."
    )
