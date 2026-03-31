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


def normalize_allele(raw: str) -> str:
    raw = raw.strip().upper()
    if raw.startswith("HLA-") and "*" in raw and ":" in raw:
        return raw
    raw = raw.replace("HLA-", "").replace("*", "").replace(":", "")
    if len(raw) >= 5 and raw[0] in {"A", "B", "C"}:
        return f"HLA-{raw[0]}*{raw[1:3]}:{raw[3:5]}"
    raise RuntimeError(f"Unsupported allele format in training.pseudo: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "src" / "autoneoag" / "resources" / "hla_pseudosequences.tsv"),
    )
    args = parser.parse_args()

    settings = load_settings(ROOT)
    source_path = settings.netmhcpan_home / "data" / "training.pseudo"
    if not source_path.exists():
        raise RuntimeError(f"Missing training.pseudo at {source_path}")

    rows = []
    with source_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            allele, pseudosequence = line.split("\t")
            compact = allele.strip().upper().replace("HLA-", "").replace("*", "").replace(":", "")
            if not compact or compact[0] not in {"A", "B", "C"}:
                continue
            rows.append((normalize_allele(allele), pseudosequence))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("hla\tpseudosequence\n")
        for allele, pseudosequence in sorted(rows):
            handle.write(f"{allele}\t{pseudosequence}\n")

    print(f"source_path: {source_path}")
    print(f"output_path: {output_path}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
