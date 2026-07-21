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
STAGE="$TMP/staged"
mkdir -p "$STAGE"
if [[ ! -d "$SRC" && ! -f "$SRC" ]]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 4
fi
python "$(dirname "$0")/stage_gap20ugr_potential_files.py" "$SRC" "$STAGE" >/dev/null
# Validate the complete staged set before replacing any working potential.
python "$(dirname "$0")/check_gap20ugr_potential_files.py" "$STAGE/Carbon_GAP_20U+gr.xml"
for f in "$STAGE"/Carbon_GAP_20U+gr.xml*; do
  cp -f -- "$f" "$OUTDIR/$(basename "$f").new"
  mv -f -- "$OUTDIR/$(basename "$f").new" "$OUTDIR/$(basename "$f")"
done
echo "Installed GAP-20U+gr files to $OUTDIR"
