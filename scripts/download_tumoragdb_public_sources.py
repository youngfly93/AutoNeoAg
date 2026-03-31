#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)


DOWNLOADS = {
    "neo_tumoragdb2_core": [
        (
            "https://tumoragdb.com.cn/files/immunogenic%20Neo-peptide%20Dataset.xlsx",
            "tumoragdb_immunogenic_neo_peptide_dataset.xlsx",
        ),
        (
            "https://tumoragdb.com.cn/files/Non-immunogenic%20Neo-peptide%20Dataset.xlsx",
            "tumoragdb_non_immunogenic_neo_peptide_dataset.xlsx",
        ),
    ],
    "neo_2024plus": [
        (
            "https://tumoragdb.com.cn/files/NeoAntigen-PubData%202024-2025.xlsx",
            "tumoragdb_neoantigen_pubdata_2024_2025.xlsx",
        ),
    ],
}

TARGET_DIRS = {
    "neo_tumoragdb2_core": ROOT / "data" / "raw" / "neoantigen" / "tumoragdb2",
    "neo_2024plus": ROOT / "data" / "raw" / "neoantigen" / "2024plus_holdout",
}


def download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", choices=sorted(DOWNLOADS), default=["neo_tumoragdb2_core"])
    args = parser.parse_args()

    created = []
    for target in args.targets:
        target_dir = TARGET_DIRS[target]
        for url, filename in DOWNLOADS[target]:
            output_path = target_dir / filename
            download_file(url, output_path)
            created.append((target, str(output_path)))

    print(f"targets: {args.targets}")
    print(f"downloaded: {created}")


if __name__ == "__main__":
    main()
