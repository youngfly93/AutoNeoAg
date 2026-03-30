#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH="$ROOT/.conda/autoneoag-py312.sparsebundle"
MOUNTPOINT="/Volumes/AutoNeoAgEnv"
ENV_PATH="$MOUNTPOINT/autoneoag-py312"
export COPYFILE_DISABLE=1
export COPY_EXTENDED_ATTRIBUTES_DISABLE=1

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found" >&2
  exit 1
fi

if ! command -v hdiutil >/dev/null 2>&1; then
  echo "hdiutil is required on macOS but not found" >&2
  exit 1
fi

rm -rf "$ROOT/.conda/autoneoag-py312"
rm -rf "$ENV_PATH"
mkdir -p "$ROOT/.conda"
if [[ ! -d "$IMAGE_PATH" ]]; then
  hdiutil create -size 24g -type SPARSEBUNDLE -fs APFS -volname AutoNeoAgEnv "$IMAGE_PATH"
fi
if ! mount | grep -Fq "on $MOUNTPOINT "; then
  hdiutil attach "$IMAGE_PATH" -nobrowse
fi
conda create -y -p "$ENV_PATH" python=3.12
conda install -y -p "$ENV_PATH" -c conda-forge -c pytorch \
  numpy pandas pyarrow scikit-learn xgboost lightgbm scipy pytorch

conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install \
  pydantic typer rich pyyaml requests biopython synapseclient pytest

conda run -p "$ENV_PATH" python -m pip install -e "$ROOT"

echo "Environment ready at $ENV_PATH"
