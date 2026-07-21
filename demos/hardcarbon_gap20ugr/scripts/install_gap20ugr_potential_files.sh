#!/usr/bin/env bash
set -euo pipefail

# Keep the demonstrator entry point, but delegate to the single hardened
# installer so archive validation cannot drift between two copies.
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec "$ROOT/scripts/install_gap20ugr_potential_files.sh" "$@"
