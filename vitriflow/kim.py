from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .utils import run_cmd


_KIM_ID_RE = re.compile(r"(__MO_\d{12}_\d{3}|__SM_\d{12}_\d{3}|\bMO_\d{12}_\d{3}\b|\bSM_\d{12}_\d{3}\b)")


@dataclass(frozen=True)
class KimInstallResult:
    attempted: bool
    success: bool
    stdout: str = ""
    stderr: str = ""


def looks_like_kim_id(model: str) -> bool:
    return bool(_KIM_ID_RE.search(model))


def is_kim_potential(potential: Any) -> bool:
    """Return whether a validated or serialized potential is explicitly KIM.

    A missing ``model`` attribute is not a reliable discriminator: explicit
    LAMMPS and CP2K configurations intentionally have no KIM model.  Dispatch
    on the public ``kind`` tag so resume and external-task paths never invoke
    KIM tooling for analytic, hybrid, MG2, or CP2K potentials.
    """

    if potential is None:
        return False
    if isinstance(potential, Mapping):
        kind = potential.get("kind")
    else:
        kind = getattr(potential, "kind", None)
    return str(kind or "").strip().lower() == "kim"


def ensure_model_installed(model: str, tool: str = "kim-api-collections-management") -> KimInstallResult:
    """Model installed."""
    if not looks_like_kim_id(model):
        return KimInstallResult(attempted=False, success=True)

    # kim api collections
    # environment different installations
    # permissions sequence installed
    attempts: list[tuple[list[str], int, str, str]] = []
    for collection in ("environment", "user", "CWD"):
        cmd = [tool, "install", collection, model]
        try:
            rc, out, err = run_cmd(cmd, check=False, capture=True)
        except Exception as e:
            # executable missing unexpected
            attempts.append((cmd, 1, "", getattr(e, "stderr", str(e))))
            continue

        attempts.append((cmd, rc, out, err))
        if rc == 0:
            return KimInstallResult(attempted=True, success=True, stdout=out, stderr=err)
        msg = (out + "\n" + err).lower()
        if "already" in msg and "install" in msg:
            return KimInstallResult(attempted=True, success=True, stdout=out, stderr=err)

    # closed clusters externally
    joined_out = "\n\n".join(
        [
            "$ " + " ".join(a[0]) + "\n" + (a[2] or "")
            for a in attempts
            if (a[2] or "").strip() != ""
        ]
    )
    joined_err = "\n\n".join(
        [
            "$ " + " ".join(a[0]) + "\n" + (a[3] or "")
            for a in attempts
            if (a[3] or "").strip() != ""
        ]
    )
    return KimInstallResult(attempted=True, success=False, stdout=joined_out, stderr=joined_err)


def ensure_potential_model_installed(
    potential: Any,
    *,
    installer: Optional[Callable[[str], Optional[KimInstallResult]]] = None,
) -> Optional[KimInstallResult]:
    """Install an explicitly configured KIM model or fail closed.

    Potential dispatch is deliberately based only on :func:`is_kim_potential`.
    This prevents analytic, hybrid, MG2, and CP2K configurations from gaining
    an accidental KIM dependency merely because they expose a ``model``-like
    field.  Conversely, a failed KIM installation is never advisory: callers
    must not proceed to a potentially multi-day calculation that cannot load
    its configured Hamiltonian.

    ``None`` is accepted solely for compatibility with established workflow
    test doubles that predate :class:`KimInstallResult`.  The production
    :func:`ensure_model_installed` implementation always returns a typed
    result, and every explicit unsuccessful or malformed result fails closed.
    """

    if not is_kim_potential(potential):
        return None

    if isinstance(potential, Mapping):
        model_value = potential.get("model")
    else:
        model_value = getattr(potential, "model", None)
    model = str(model_value or "").strip()
    if not model:
        raise RuntimeError("Explicit KIM potential is missing its required model identifier")

    install_fn = ensure_model_installed if installer is None else installer
    result = install_fn(model)
    if result is None:
        # Compatibility for existing unit-test monkeypatches.  The real
        # installer has a total, typed return contract and cannot reach here.
        return None
    if not isinstance(result, KimInstallResult):
        raise RuntimeError(
            "KIM model installer returned an invalid result for "
            f"{model!r}: expected KimInstallResult, got {type(result).__name__}"
        )
    if result.success:
        return result

    diagnostics: list[str] = []
    if str(result.stderr or "").strip():
        diagnostics.append("stderr: " + str(result.stderr).strip())
    if str(result.stdout or "").strip():
        diagnostics.append("stdout: " + str(result.stdout).strip())
    detail = "; ".join(diagnostics) if diagnostics else "no installer diagnostics were produced"
    attempted = "attempted" if result.attempted else "not attempted"
    raise RuntimeError(
        f"KIM model installation failed for {model!r} ({attempted}); {detail}"
    )
