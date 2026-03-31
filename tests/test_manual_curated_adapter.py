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


def test_iedb_functional_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "iedb_functional"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "iedb_rows.csv"
    frame = pd.DataFrame(
        [
            {
                "epitope": "YLQPRTFLL",
                "reference_peptide": "YLQPRTFVL",
                "allele_name": "HLA-A*02:01",
                "antigen_gene": "KRAS",
                "variant_name": "G12D",
                "study_accession": "IEDB-001",
                "subject_id": "S001",
                "assay_group": "ELISpot",
                "qualitative_measure": "Positive",
                "year": 2022,
            },
            {
                "epitope": "GLCTLVAML",
                "reference_peptide": "GLCTLVAMM",
                "allele_name": "HLA-A*02:01",
                "antigen_gene": "EGFR",
                "variant_name": "L858R",
                "study_accession": "IEDB-002",
                "subject_id": "S002",
                "assay_group": "FACS",
                "qualitative_measure": "Negative",
                "year": 2021,
            },
        ]
    )
    frame.to_csv(raw_path, index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "neo_iedb_functional",
        "source_name": "IEDB human HLA-I neoantigen functional subset",
        "adapter_id": "iedb_neo_functional_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["study_id"].tolist() == ["IEDB-001", "IEDB-002"]
    assert standardized["patient_id"].tolist() == ["S001", "S002"]
    assert standardized["assay_type"].tolist() == ["ELISpot", "FACS"]
    assert standardized["mutation_event"].tolist() == ["KRAS:G12D", "EGFR:L858R"]
