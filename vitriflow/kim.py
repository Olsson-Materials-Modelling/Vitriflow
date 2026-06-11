from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
