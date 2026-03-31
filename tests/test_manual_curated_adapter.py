from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import load_settings
from autoneoag.ingest.full import run_source_adapter


def test_neo_literature_manual_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "literature_curated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "manual_rows.tsv"
    frame = pd.DataFrame(
        [
            {
                "mut_peptide": "YLQPRTFLL",
                "wt_peptide": "YLQPRTFVL",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "KRAS",
                "protein_change": "G12D",
                "study": "LIT-001",
                "patient": "P001",
                "readout": "ELISpot",
                "immunogenic": "positive",
                "year": 2024,
            },
            {
                "mut_peptide": "GLCTLVAML",
                "wt_peptide": "GLCTLVAMM",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "EGFR",
                "protein_change": "L858R",
                "study": "LIT-002",
                "patient": "P002",
                "readout": "FACS",
                "immunogenic": "negative",
                "year": 2023,
            },
        ]
    )
    frame.to_csv(raw_path, sep="\t", index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "neo_literature_curated",
        "source_name": "Manual curated human neoantigen literature set",
        "adapter_id": "neo_literature_manual_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["source_id"].tolist() == ["neo_literature_curated", "neo_literature_curated"]
    assert standardized["peptide_length"].tolist() == [9, 9]
    assert standardized["mutation_event"].tolist() == ["KRAS:G12D", "EGFR:L858R"]
    assert set(standardized.columns) >= {
        "peptide_mut",
        "peptide_wt",
        "hla",
        "gene",
        "aa_change",
        "study_id",
        "patient_id",
        "assay_type",
        "label",
        "label_tier",
        "source_name",
        "source_year",
        "is_tesla",
        "is_simulated",
        "is_mouse",
        "mutation_event",
        "peptide_length",
        "source_id",
    }
