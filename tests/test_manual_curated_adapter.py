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


def test_immuno_literature_manual_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "immuno_literature"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "immuno_rows.tsv"
    frame = pd.DataFrame(
        [
            {
                "mut_peptide": "GILGFVFTL",
                "wt_peptide": "GILGFVYTL",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "TP53",
                "protein_change": "R175H",
                "study": "IMM-LIT-001",
                "patient": "IMM-L001",
                "readout": "ELISpot",
                "immunogenic": "positive",
                "year": 2024,
            },
            {
                "mut_peptide": "LLGATCMFV",
                "wt_peptide": "LLGATCMFI",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "IDH1",
                "protein_change": "R132H",
                "study": "IMM-LIT-002",
                "patient": "IMM-L002",
                "readout": "Tetramer",
                "immunogenic": "negative",
                "year": 2023,
            },
        ]
    )
    frame.to_csv(raw_path, sep="\t", index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "immuno_curated_literature",
        "source_name": "Manual curated human peptide immunogenicity literature set",
        "adapter_id": "immuno_literature_manual_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["source_id"].tolist() == ["immuno_curated_literature", "immuno_curated_literature"]
    assert standardized["assay_type"].tolist() == ["ELISpot", "Tetramer"]


def test_tumoragdb2_curated_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "tumoragdb2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "tumoragdb2_rows.tsv"
    frame = pd.DataFrame(
        [
            {
                "mutated_peptide": "SPRWYFYYL",
                "wildtype_sequence": "SPRWYFYFL",
                "hla_type": "HLA-B*07:02",
                "gene_name": "PIK3CA",
                "amino_acid_change": "E545K",
                "study_name": "TADB-001",
                "sample_identifier": "TADB-P001",
                "assay_readout": "ELISpot",
                "functional_result": "positive",
                "pub_year": 2022,
                "record_tier": "A",
            },
            {
                "mutated_peptide": "KLVVVGAGG",
                "wildtype_sequence": "KLVVVGAGD",
                "hla_type": "HLA-A*11:01",
                "gene_name": "KRAS",
                "amino_acid_change": "G13D",
                "study_name": "TADB-002",
                "sample_identifier": "TADB-P002",
                "assay_readout": "Multimer",
                "functional_result": "negative",
                "pub_year": 2021,
                "record_tier": "B",
            },
        ]
    )
    frame.to_csv(raw_path, sep="\t", index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "neo_tumoragdb2_core",
        "source_name": "TumorAgDB2.0 curated human HLA-I subset",
        "adapter_id": "tumoragdb2_curated_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["study_id"].tolist() == ["TADB-001", "TADB-002"]
    assert standardized["patient_id"].tolist() == ["TADB-P001", "TADB-P002"]
    assert standardized["mutation_event"].tolist() == ["PIK3CA:E545K", "KRAS:G13D"]


def test_neo_timesplit_holdout_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "neo_2024plus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "neo_2024plus_rows.tsv"
    frame = pd.DataFrame(
        [
            {
                "mut_peptide": "LLDFVRFMGV",
                "wt_peptide": "LLDFVRFMAV",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "BRAF",
                "protein_change": "V600E",
                "study": "OOT-NEO-001",
                "patient": "OOTN001",
                "readout": "ELISpot",
                "immunogenic": "positive",
                "year": 2025,
            },
            {
                "mut_peptide": "SIINFEKQL",
                "wt_peptide": "SIINFEKEL",
                "hla_allele": "HLA-B*08:01",
                "gene_symbol": "NRAS",
                "protein_change": "Q61R",
                "study": "OOT-NEO-002",
                "patient": "OOTN002",
                "readout": "Multimer",
                "immunogenic": "negative",
                "year": 2024,
            },
        ]
    )
    frame.to_csv(raw_path, sep="\t", index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "neo_2024plus",
        "source_name": "2024-2025 out-of-time literature holdout",
        "adapter_id": "neo_timesplit_holdout_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["source_id"].tolist() == ["neo_2024plus", "neo_2024plus"]
    assert standardized["study_id"].tolist() == ["OOT-NEO-001", "OOT-NEO-002"]


def test_immuno_external_lockbox_adapter_standardizes_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "immuno_external"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "immuno_external_rows.tsv"
    frame = pd.DataFrame(
        [
            {
                "mut_peptide": "YLQPRTFLL",
                "wt_peptide": "YLQPRTFVL",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "KRAS",
                "protein_change": "G12D",
                "study": "IMM-EXT-001",
                "patient": "IMME001",
                "readout": "ELISpot",
                "immunogenic": "positive",
                "year": 2022,
            },
            {
                "mut_peptide": "GLLGTLVAML",
                "wt_peptide": "GLLGTLVAMM",
                "hla_allele": "HLA-A*02:01",
                "gene_symbol": "EGFR",
                "protein_change": "L858R",
                "study": "IMM-EXT-002",
                "patient": "IMME002",
                "readout": "FACS",
                "immunogenic": "negative",
                "year": 2021,
            },
        ]
    )
    frame.to_csv(raw_path, sep="\t", index=False)

    settings = load_settings(Path.cwd())
    source_row = {
        "source_id": "immuno_external_study",
        "source_name": "Study-held-out external human HLA-I cohort",
        "adapter_id": "immuno_external_lockbox_adapter",
        "raw_file_path": str(raw_dir),
        "year_end": 2025,
    }
    standardized = run_source_adapter(settings, source_row)

    assert len(standardized) == 2
    assert standardized["label"].tolist() == [1, 0]
    assert standardized["source_id"].tolist() == ["immuno_external_study", "immuno_external_study"]
    assert standardized["study_id"].tolist() == ["IMM-EXT-001", "IMM-EXT-002"]
