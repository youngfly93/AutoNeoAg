#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DTU_DIR="$ROOT/.local_tools/dtu"
NETMHCPAN_TAR="$ROOT/netMHCpan-4.2c.Darwin_arm64.tar"
NETMHCPAN_HOME="$DTU_DIR/netMHCpan-4.2"
NETMHCSTABPAN_TAR="$ROOT/netMHCstabpan.Darwin_arm64.tar"
NETMHCSTABPAN_HOME="$DTU_DIR/netMHCstabpan"

mkdir -p "$DTU_DIR"

cleanup_tree() {
  local target="$1"
  if [[ -d "$target" ]]; then
    find "$target" \( -name '._*' -o -name '.DS_Store' \) -delete
  fi
}

extract_tar() {
  local archive="$1"
  COPYFILE_DISABLE=1 tar -xf "$archive" -C "$DTU_DIR"
}

if [[ ! -f "$NETMHCPAN_TAR" ]]; then
  echo "Missing $NETMHCPAN_TAR" >&2
  exit 1
fi

if [[ ! -f "$NETMHCSTABPAN_TAR" ]]; then
  echo "Missing $NETMHCSTABPAN_TAR" >&2
  exit 1
fi

rm -rf "$NETMHCPAN_HOME"
extract_tar "$NETMHCPAN_TAR"
cleanup_tree "$NETMHCPAN_HOME"
python <<'PY'
from pathlib import Path
root = Path("/Volumes/KINGSTON/work/research/AutoNeoAg/.local_tools/dtu/netMHCpan-4.2/netMHCpan")
text = root.read_text()
patched = "setenv\tNMHOME\t/Volumes/KINGSTON/work/research/AutoNeoAg/.local_tools/dtu/netMHCpan-4.2"
text = text.replace("setenv\tNMHOME\t/tools/src/netMHCpan-4.2", patched)
root.write_text(text)
PY

rm -rf "$NETMHCSTABPAN_HOME"
extract_tar "$NETMHCSTABPAN_TAR"
cleanup_tree "$NETMHCSTABPAN_HOME"

echo "Installed NetMHCpan to $NETMHCPAN_HOME"
echo "Installed NetMHCstabpan to $NETMHCSTABPAN_HOME"
