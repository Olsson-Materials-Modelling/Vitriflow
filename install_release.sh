#!/usr/bin/env bash
# Install the release wheel matching pyproject.toml, or fall back to this source tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
VERSION="$(${PYTHON_BIN} - "${ROOT}/pyproject.toml" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
if match is None:
    raise SystemExit("Could not read project.version from pyproject.toml")
print(match.group(1))
PY
)"
WHEEL="${ROOT}/dist/vitriflow-${VERSION}-py3-none-any.whl"

"${PYTHON_BIN}" -m pip uninstall -y vitriflow >/dev/null 2>&1 || true
if [[ -f "${WHEEL}" ]]; then
  INSTALL_MODE="wheel"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${WHEEL}"
else
  INSTALL_MODE="editable"
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps -e "${ROOT}"
fi
hash -r || true

"${PYTHON_BIN}" - "${VERSION}" "${ROOT}" "${INSTALL_MODE}" <<'PY'
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

expected = sys.argv[1]
source_root = Path(sys.argv[2]).resolve()
install_mode = sys.argv[3]
original_cwd = Path.cwd()

# Validate outside the source tree. Otherwise Python's empty sys.path entry can
# make a broken wheel appear healthy by importing ./vitriflow directly.
with tempfile.TemporaryDirectory(prefix="vitriflow-install-check-") as temp_dir:
    try:
        os.chdir(temp_dir)
        importlib.invalidate_caches()
        import vitriflow

        module_path = Path(vitriflow.__file__).resolve()
        distribution_version = metadata.version("vitriflow")
        if vitriflow.__version__ != expected or distribution_version != expected:
            raise SystemExit(
                "Version mismatch after installation: "
                f"expected={expected}, module={vitriflow.__version__}, "
                f"distribution={distribution_version}"
            )

        try:
            module_path.relative_to(source_root)
            imported_from_source = True
        except ValueError:
            imported_from_source = False
        if install_mode == "wheel" and imported_from_source:
            raise SystemExit(
                "Wheel validation imported the source tree instead of the installed wheel: "
                f"{module_path}"
            )

        scripts_dir = Path(sysconfig.get_path("scripts"))
        executable = scripts_dir / "vitriflow"
        if not executable.is_file():
            resolved = shutil.which("vitriflow")
            executable = Path(resolved) if resolved else executable
        if not executable.is_file():
            raise SystemExit(f"Installed vitriflow entrypoint was not found in {scripts_dir}")
        subprocess.run([str(executable), "--version"], check=True, cwd=temp_dir)
    finally:
        os.chdir(original_cwd)

print(f"VitriFlow {expected} import OK: {module_path}")
print(f"vitriflow executable: {executable}")
PY
