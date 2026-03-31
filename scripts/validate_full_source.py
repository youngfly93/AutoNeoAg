#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

from autoneoag.config import load_settings
from autoneoag.ingest.full import run_source_adapter
from autoneoag.manifests import load_manifest_bundle
from autoneoag.tasks import list_task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list_task_ids(), required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    settings = load_settings(ROOT)
    bundle = load_manifest_bundle(settings, args.task)
    matches = bundle.source_manifest.loc[bundle.source_manifest["source_id"] == args.source]
    if matches.empty:
        raise RuntimeError(f"Unknown source_id {args.source!r} for task {args.task!r}")
    source_row = matches.iloc[0].to_dict()
    standardized = run_source_adapter(settings, source_row)
    print(f"task_id: {args.task}")
    print(f"source_id: {args.source}")
    print(f"rows: {len(standardized)}")
    print(f"columns: {list(standardized.columns)}")
    print(f"labels: {standardized['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
