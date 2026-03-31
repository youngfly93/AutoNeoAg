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
from autoneoag.ingest.full import write_source_template
from autoneoag.manifests import load_manifest_bundle
from autoneoag.tasks import list_task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list_task_ids(), required=True)
    parser.add_argument("--sources", nargs="*", default=[])
    args = parser.parse_args()

    settings = load_settings(ROOT)
    bundle = load_manifest_bundle(settings, args.task)
    if args.sources:
        target_sources = args.sources
    else:
        target_sources = bundle.source_manifest.loc[bundle.source_manifest["ingest_status"] == "implemented", "source_id"].tolist()

    created = []
    skipped = []
    for source_id in target_sources:
        try:
            path = write_source_template(settings, args.task, source_id)
        except RuntimeError as exc:
            skipped.append((source_id, str(exc)))
            continue
        created.append((source_id, str(path)))

    print(f"task_id: {args.task}")
    print(f"created_templates: {created}")
    print(f"skipped_templates: {skipped}")


if __name__ == "__main__":
    main()
