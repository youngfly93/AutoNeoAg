from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import Settings
from autoneoag.manifests import resolve_manifest_path


CANONICAL_COLUMNS = [
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
]

ALIAS_GROUPS = {
    "peptide_mut": ["peptide_mut", "mut_peptide", "mutant_peptide", "neo_peptide", "epitope", "peptide"],
    "peptide_wt": ["peptide_wt", "wt_peptide", "wildtype_peptide", "reference_peptide", "wt_epitope", "reference_sequence"],
    "hla": ["hla", "hla_allele", "mhc", "allele", "allele_name", "mhc_allele"],
    "gene": ["gene", "gene_symbol", "antigen_gene"],
    "aa_change": ["aa_change", "protein_change", "mutation", "variant", "variant_name"],
    "study_id": ["study_id", "study", "cohort_id", "study_accession", "reference_id"],
    "patient_id": ["patient_id", "patient", "sample_id", "subject_id"],
    "assay_type": ["assay_type", "assay", "readout", "assay_group", "method"],
    "label": ["label", "is_positive", "immunogenic", "response", "assay_result", "qualitative_measure"],
    "label_tier": ["label_tier", "tier"],
    "source_name": ["source_name", "source"],
    "source_year": ["source_year", "year"],
    "is_tesla": ["is_tesla"],
    "is_simulated": ["is_simulated"],
    "is_mouse": ["is_mouse"],
}


def _list_tabular_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files = sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
    )
    return files


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise RuntimeError(f"Unsupported tabular file type: {path}")


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    output = pd.DataFrame(index=renamed.index)
    for canonical, aliases in ALIAS_GROUPS.items():
        for alias in aliases:
            if alias in renamed.columns:
                output[canonical] = renamed[alias]
                break
    return output


def _normalize_bool(value: object, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return default


def _normalize_label(value: object) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise RuntimeError("Missing label value in manual curated adapter input.")
    text = str(value).strip().lower()
    if text in {"1", "positive", "pos", "yes", "true", "immunogenic"}:
        return 1
    if text in {"0", "negative", "neg", "no", "false", "non_immunogenic", "non-immunogenic"}:
        return 0
    raise RuntimeError(f"Unsupported label value: {value!r}")


def _require_columns(frame: pd.DataFrame, columns: list[str], source_id: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Adapter input for {source_id} is missing required columns: {missing}")


def _normalize_functional_immunology(
    standardized: pd.DataFrame,
    source_row: dict[str, object],
    *,
    default_assay: str,
    default_tier: str,
) -> pd.DataFrame:
    standardized["patient_id"] = standardized.get("patient_id", pd.Series([""] * len(standardized)))
    standardized["assay_type"] = standardized.get("assay_type", pd.Series([default_assay] * len(standardized)))
    standardized["label_tier"] = standardized.get("label_tier", pd.Series([default_tier] * len(standardized)))
    standardized["source_name"] = standardized.get("source_name", pd.Series([source_row["source_name"]] * len(standardized)))
    standardized["source_year"] = standardized.get("source_year", pd.Series([source_row["year_end"]] * len(standardized)))
    standardized["is_tesla"] = standardized.get("is_tesla", pd.Series([0] * len(standardized)))
    standardized["is_simulated"] = standardized.get("is_simulated", pd.Series([0] * len(standardized)))
    standardized["is_mouse"] = standardized.get("is_mouse", pd.Series([0] * len(standardized)))

    standardized["label"] = standardized["label"].map(_normalize_label)
    for bool_column in ("is_tesla", "is_simulated", "is_mouse"):
        standardized[bool_column] = standardized[bool_column].map(lambda value: _normalize_bool(value, default=0))

    for column in CANONICAL_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = ""
    standardized = standardized[CANONICAL_COLUMNS].copy()
    standardized["source_year"] = standardized["source_year"].astype(int)
    standardized["patient_id"] = standardized["patient_id"].fillna("").astype(str)
    standardized["assay_type"] = standardized["assay_type"].fillna(default_assay).astype(str)
    standardized["label_tier"] = standardized["label_tier"].fillna(default_tier).astype(str)
    standardized["mutation_event"] = standardized["gene"].astype(str) + ":" + standardized["aa_change"].astype(str)
    standardized["peptide_length"] = standardized["peptide_mut"].astype(str).str.len()
    standardized["source_id"] = str(source_row["source_id"])
    return standardized


def standardize_manual_curated_immunology(frame: pd.DataFrame, source_row: dict[str, object]) -> pd.DataFrame:
    standardized = _canonicalize_columns(frame)
    required = [
        "peptide_mut",
        "peptide_wt",
        "hla",
        "gene",
        "aa_change",
        "study_id",
        "label",
    ]
    _require_columns(standardized, required, str(source_row["source_id"]))
    return _normalize_functional_immunology(standardized, source_row, default_assay="manual_curated", default_tier="A")


def standardize_iedb_functional_immunology(frame: pd.DataFrame, source_row: dict[str, object]) -> pd.DataFrame:
    standardized = _canonicalize_columns(frame)
    required = [
        "peptide_mut",
        "peptide_wt",
        "hla",
        "gene",
        "aa_change",
        "study_id",
        "label",
    ]
    _require_columns(standardized, required, str(source_row["source_id"]))
    return _normalize_functional_immunology(standardized, source_row, default_assay="iedb_functional", default_tier="A")


def run_manual_curated_adapter(settings: Settings, source_row: dict[str, object]) -> pd.DataFrame:
    raw_path = resolve_manifest_path(settings, str(source_row["raw_file_path"]))
    files = _list_tabular_files(raw_path)
    if not files:
        raise RuntimeError(f"No supported raw files found for source {source_row['source_id']} at {raw_path}")
    frames = []
    for file_path in files:
        frame = _read_tabular_file(file_path)
        frame["__source_file__"] = str(file_path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    return standardize_manual_curated_immunology(merged, source_row)


def run_iedb_functional_adapter(settings: Settings, source_row: dict[str, object]) -> pd.DataFrame:
    raw_path = resolve_manifest_path(settings, str(source_row["raw_file_path"]))
    files = _list_tabular_files(raw_path)
    if not files:
        raise RuntimeError(f"No supported raw files found for source {source_row['source_id']} at {raw_path}")
    frames = []
    for file_path in files:
        frame = _read_tabular_file(file_path)
        frame["__source_file__"] = str(file_path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    return standardize_iedb_functional_immunology(merged, source_row)


ADAPTER_REGISTRY = {
    "iedb_neo_functional_adapter": run_iedb_functional_adapter,
    "iedb_immunogenicity_adapter": run_iedb_functional_adapter,
    "neo_literature_manual_adapter": run_manual_curated_adapter,
}


def run_source_adapter(settings: Settings, source_row: dict[str, object]) -> pd.DataFrame:
    adapter_id = str(source_row["adapter_id"])
    try:
        adapter = ADAPTER_REGISTRY[adapter_id]
    except KeyError as exc:
        raise RuntimeError(f"No adapter implementation is registered for {adapter_id}") from exc
    return adapter(settings, source_row)
