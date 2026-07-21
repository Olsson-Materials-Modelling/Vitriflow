#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LABEL = "GAP_2022_11_4_0_14_40_15_889"
EXPECTED = {
    "Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8891": "7bff09b267d7cae4a7ae76ab8085246f",
    "Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8892": "184aa8ae3958bd36ec3c98b5f667f627",
    "Carbon_GAP_20U+gr.xml.sparseX.GAP_2022_11_4_0_14_40_15_8893": "363fb89ca10a011d8ef70d3798b011bf",
}

def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main(argv: list[str]) -> int:
    xml_path = Path(argv[1] if len(argv) > 1 else "potentials/Carbon_GAP_20U+gr.xml")
    if xml_path.is_dir():
        xml_path = xml_path / "Carbon_GAP_20U+gr.xml"
    if xml_path.is_symlink():
        print(f"ERROR: XML must be a regular non-symlink file: {xml_path}", file=sys.stderr)
        return 2
    xml_path = xml_path.resolve(strict=False)
    if not xml_path.is_file():
        print(f"ERROR: missing XML: {xml_path}", file=sys.stderr)
        return 2
    text = xml_path.read_text(errors="ignore")
    if LABEL not in text:
        print(f"ERROR: expected xml_label {LABEL!r} not found in {xml_path}", file=sys.stderr)
        return 3
    try:
        ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"ERROR: XML parse failed: {exc}", file=sys.stderr)
        return 4
    # The model is a fixed published artifact.  Accepting arbitrary filenames
    # declared by a modified XML bypassed the known sidecar checksums.
    declared = sorted(set(re.findall(r'sparseX_filename\s*=\s*["\']([^"\']+)["\']', text)))
    names = sorted(EXPECTED)
    if set(declared) != set(names):
        print(
            "ERROR: XML sparseX declarations do not exactly match the published GAP-20U+gr sidecars",
            file=sys.stderr,
        )
        return 5
    ok = True
    for name in names:
        p = xml_path.parent / name
        if p.is_symlink() or not p.is_file():
            print(f"ERROR: missing sparseX sidecar: {p}", file=sys.stderr)
            ok = False
            continue
        want = EXPECTED.get(name)
        got = md5(p)
        if want is not None and got != want:
            print(f"ERROR: md5 mismatch for {p.name}: got {got}, expected {want}", file=sys.stderr)
            ok = False
        else:
            print(f"OK: {p.name} md5={got}")
    if not ok:
        return 5
    print(f"OK: {xml_path.name}; xml_label={LABEL}; sidecars={len(names)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
