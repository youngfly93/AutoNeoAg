#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export LANG=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DTU_DIR="$ROOT/.local_tools/dtu"
NETMHCPAN_HOME="$DTU_DIR/netMHCpan-4.2"
NETMHCSTABPAN_HOME="$DTU_DIR/netMHCstabpan"
NETMHCSTABPAN_DATA_TAR="$DTU_DIR/netMHCstabpan-data.tar.gz"
NETMHCSTABPAN_DATA_URL="https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/data.tar.gz"

pick_archive() {
  local pattern="$1"
  find "$ROOT" -maxdepth 1 -type f -name "$pattern" | sort | head -n 1
}

NETMHCPAN_TAR="$(pick_archive 'netMHCpan-*.tar')"
NETMHCSTABPAN_TAR="$(pick_archive 'netMHCstabpan-*.tar')"

mkdir -p "$DTU_DIR"

cleanup_tree() {
  local target="$1"
  if [[ -d "$target" ]]; then
    find "$target" \( -name '._*' -o -name '.DS_Store' \) -delete
  fi
}

extract_tar() {
  local archive="$1"
  COPYFILE_DISABLE=1 tar -xf "$archive" \
    --exclude '._*' \
    --exclude '*/._*' \
    --exclude '.DS_Store' \
    --exclude '*/.DS_Store' \
    -C "$DTU_DIR"
}

replace_in_file() {
  local path="$1"
  local from="$2"
  local to="$3"
  /Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python - <<PY
from pathlib import Path
path = Path(r"""$path""")
text = path.read_text()
updated = text.replace(r"""$from""", r"""$to""")
if text == updated:
    raise SystemExit(f"Patch target not found in {path}")
path.write_text(updated)
PY
}

if [[ -z "$NETMHCPAN_TAR" || ! -f "$NETMHCPAN_TAR" ]]; then
  echo "Missing netMHCpan archive in $ROOT" >&2
  exit 1
fi

if [[ -z "$NETMHCSTABPAN_TAR" || ! -f "$NETMHCSTABPAN_TAR" ]]; then
  echo "Missing netMHCstabpan archive in $ROOT" >&2
  exit 1
fi

rm -rf "$NETMHCPAN_HOME"
extract_tar "$NETMHCPAN_TAR"
cleanup_tree "$NETMHCPAN_HOME"
replace_in_file \
  "$NETMHCPAN_HOME/netMHCpan" \
  $'setenv\tNMHOME\t/tools/src/netMHCpan-4.2' \
  $'setenv\tNMHOME\t'"$NETMHCPAN_HOME"

rm -rf "$NETMHCSTABPAN_HOME"
extract_tar "$NETMHCSTABPAN_TAR"
cleanup_tree "$DTU_DIR"
STAB_EXTRACTED="$(find "$DTU_DIR" -maxdepth 1 -type d -name 'netMHCstabpan-*' | sort | head -n 1)"
if [[ -z "$STAB_EXTRACTED" ]]; then
  echo "Failed to locate extracted netMHCstabpan directory" >&2
  exit 1
fi
mv "$STAB_EXTRACTED" "$NETMHCSTABPAN_HOME"
cleanup_tree "$NETMHCSTABPAN_HOME"
replace_in_file \
  "$NETMHCSTABPAN_HOME/netMHCstabpan" \
  $'setenv\tNMHOME\t/usr/cbs/packages/netMHCstabpan/1.0/netMHCstabpan-1.0' \
  $'setenv\tNMHOME\t'"$NETMHCSTABPAN_HOME"
replace_in_file \
  "$NETMHCSTABPAN_HOME/netMHCstabpan" \
  'set	NetMHCpan = /usr/cbs/bio/src/netMHCpan-2.8/netMHCpan' \
  "set	NetMHCpan = $NETMHCPAN_HOME/netMHCpan"
replace_in_file \
  "$NETMHCSTABPAN_HOME/netMHCstabpan" \
  'setenv	AR	`uname -m`' \
  'setenv	AR	x86_64'

if [[ ! -d "$NETMHCSTABPAN_HOME/data" ]]; then
  curl -fL "$NETMHCSTABPAN_DATA_URL" -o "$NETMHCSTABPAN_DATA_TAR"
  COPYFILE_DISABLE=1 tar -xf "$NETMHCSTABPAN_DATA_TAR" \
    --exclude '._*' \
    --exclude '*/._*' \
    --exclude '.DS_Store' \
    --exclude '*/.DS_Store' \
    -C "$NETMHCSTABPAN_HOME"
  cleanup_tree "$NETMHCSTABPAN_HOME"
fi

echo "Installed NetMHCpan to $NETMHCPAN_HOME"
echo "Installed NetMHCstabpan to $NETMHCSTABPAN_HOME"
