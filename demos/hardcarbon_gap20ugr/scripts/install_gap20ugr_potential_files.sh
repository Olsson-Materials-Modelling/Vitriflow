#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   scripts/install_gap20ugr_potential_files.sh /path/to/complementary_data.tar potentials
#   scripts/install_gap20ugr_potential_files.sh /path/to/unpacked/complementary_data potentials
#   scripts/install_gap20ugr_potential_files.sh potentials   # download tar into potentials/
if [[ $# -eq 1 ]]; then
  SRC=""
  OUTDIR="$1"
elif [[ $# -eq 2 ]]; then
  SRC="$1"
  OUTDIR="$2"
else
  echo "Usage: $0 [complementary_data.tar|unpacked_dir] OUTDIR" >&2
  exit 2
fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
ZENODO_URL='https://zenodo.org/records/7463706/files/complementary_data.tar?download=1'
TARFILE="${OUTDIR}/complementary_data.tar"
if [[ -z "$SRC" ]]; then
  if [[ ! -s "$TARFILE" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --retry 3 -o "$TARFILE" "$ZENODO_URL"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$TARFILE" "$ZENODO_URL"
    else
      echo "ERROR: neither curl nor wget is available; download complementary_data.tar manually." >&2
      exit 3
    fi
  fi
  SRC="$TARFILE"
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
copy_found() {
  local root="$1"
  local n=0
  while IFS= read -r -d '' f; do
    cp -f "$f" "$OUTDIR/$(basename "$f")"
    n=$((n+1))
  done < <(find "$root" -type f \( -name 'Carbon_GAP_20U+gr.xml' -o -name 'Carbon_GAP_20U+gr.xml.sparseX.*' \) -print0)
  echo "$n"
}
if [[ -d "$SRC" ]]; then
  N=$(copy_found "$SRC")
elif [[ -f "$SRC" ]]; then
  tar -xf "$SRC" -C "$TMP"
  N=$(copy_found "$TMP")
else
  echo "ERROR: source not found: $SRC" >&2
  exit 4
fi
if [[ "$N" -lt 4 ]]; then
  echo "ERROR: found/copied only $N GAP-20U+gr files; expected XML + 3 sparseX sidecars." >&2
  echo "Check that the source is Zenodo record 7463706 complementary_data.tar." >&2
  exit 5
fi
python "$(dirname "$0")/check_gap20ugr_potential_files.py" "$OUTDIR/Carbon_GAP_20U+gr.xml"
echo "Installed GAP-20U+gr files to $OUTDIR"
