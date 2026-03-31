#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)


QUERY_URL = "https://query-api.iedb.org/tcell_search"
SELECT_FIELDS = ",".join(
    [
        "tcell_id",
        "linear_sequence",
        "mhc_allele_name",
        "qualitative_measure",
        "assay_names",
        "reference_id",
        "reference_dates",
        "parent_source_antigen_name",
        "curated_source_antigen",
    ]
)


def query_page(start: int, end: int) -> list[dict[str, object]]:
    params = {
        "select": SELECT_FIELDS,
        "host_organism_name": "eq.Homo sapiens (human)",
        "structure_type": "eq.Linear peptide",
        "mhc_class": "eq.I",
        "qualitative_measure": "in.(Positive,Negative)",
        "mhc_allele_name": "like.HLA-%",
    }
    headers = {"Range-Unit": "items", "Range": f"{start}-{end}"}
    response = requests.get(QUERY_URL, params=params, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def year_from_reference_dates(value: object) -> int:
    if isinstance(value, list) and value:
        text = str(value[0]).strip()
        if text.isdigit():
            return int(text)
    return 2025


def curated_name(value: object) -> str:
    if isinstance(value, dict):
        name = value.get("name")
        if name:
            return str(name)
    return ""


def canonicalize(records: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for record in records:
        epitope = str(record.get("linear_sequence") or "").strip().upper()
        allele = str(record.get("mhc_allele_name") or "").strip()
        qualitative = str(record.get("qualitative_measure") or "").strip()
        if not epitope or not allele or qualitative not in {"Positive", "Negative"}:
            continue
        antigen = str(record.get("parent_source_antigen_name") or "").strip()
        if not antigen:
            antigen = curated_name(record.get("curated_source_antigen"))
        if not antigen:
            antigen = "IEDB_ANTIGEN"
        rows.append(
            {
                "epitope": epitope,
                "reference_peptide": epitope,
                "allele_name": allele,
                "antigen_gene": antigen,
                "variant_name": epitope,
                "study_accession": str(record.get("reference_id") or ""),
                "subject_id": f"IEDB-TCELL-{record.get('tcell_id')}",
                "assay_group": str(record.get("assay_names") or ""),
                "qualitative_measure": qualitative,
                "year": year_from_reference_dates(record.get("reference_dates")),
                "tier": "A",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        subset=["epitope", "allele_name", "qualitative_measure", "study_accession", "subject_id"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "raw" / "hla_immunogenicity" / "iedb_functional" / "iedb_hla_i_tcell.csv"),
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for page in range(args.max_pages):
        start = page * args.page_size
        end = start + args.page_size - 1
        chunk = query_page(start, end)
        records.extend(chunk)
        if len(chunk) < args.page_size:
            break

    frame = canonicalize(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    (output_path.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "query_url": QUERY_URL,
                "page_size": args.page_size,
                "max_pages": args.max_pages,
                "downloaded_records": len(records),
                "canonical_rows": int(len(frame)),
                "output_path": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"output_path: {output_path}")
    print(f"downloaded_records: {len(records)}")
    print(f"canonical_rows: {len(frame)}")


if __name__ == "__main__":
    main()
