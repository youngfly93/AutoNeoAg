from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from autoneoag.config import Settings
from autoneoag.manifests import load_manifest_bundle, resolve_manifest_path
from autoneoag.splits.pipeline import assign_split_by_source_role


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

SOURCE_META_COLUMNS = [
    "source_id",
    "source_name",
    "split_role",
    "source_priority",
    "download_method",
    "label_strength",
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
        if file_path.is_file()
        and not file_path.name.startswith("._")
        and not file_path.name.startswith(".")
        and file_path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
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


TEMPLATE_REGISTRY: dict[str, tuple[str, pd.DataFrame]] = {
    "neo_literature_manual_adapter": (
        "template_manual_curated.tsv",
        pd.DataFrame(
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
                    "tier": "A",
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
                    "tier": "A",
                },
            ]
        ),
    ),
    "iedb_neo_functional_adapter": (
        "template_iedb_functional.csv",
        pd.DataFrame(
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
                    "tier": "A",
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
                    "tier": "A",
                },
            ]
        ),
    ),
    "iedb_immunogenicity_adapter": (
        "template_iedb_functional.csv",
        pd.DataFrame(
            [
                {
                    "epitope": "KLVALGINAV",
                    "reference_peptide": "KLVALGINAI",
                    "allele_name": "HLA-A*02:01",
                    "antigen_gene": "PIK3CA",
                    "variant_name": "H1047R",
                    "study_accession": "IEDB-IMM-001",
                    "subject_id": "IMM001",
                    "assay_group": "ELISpot",
                    "qualitative_measure": "Positive",
                    "year": 2022,
                    "tier": "A",
                },
                {
                    "epitope": "SLYNTVATL",
                    "reference_peptide": "SLYNTVAAL",
                    "allele_name": "HLA-A*02:01",
                    "antigen_gene": "BRAF",
                    "variant_name": "V600E",
                    "study_accession": "IEDB-IMM-002",
                    "subject_id": "IMM002",
                    "assay_group": "FACS",
                    "qualitative_measure": "Negative",
                    "year": 2021,
                    "tier": "A",
                },
            ]
        ),
    ),
}


def run_source_adapter(settings: Settings, source_row: dict[str, object]) -> pd.DataFrame:
    adapter_id = str(source_row["adapter_id"])
    try:
        adapter = ADAPTER_REGISTRY[adapter_id]
    except KeyError as exc:
        raise RuntimeError(f"No adapter implementation is registered for {adapter_id}") from exc
    return adapter(settings, source_row)


def _sample_uid(row: pd.Series) -> str:
    key = "|".join(
        [
            str(row.get("source_id", "")),
            str(row.get("study_id", "")),
            str(row.get("patient_id", "")),
            str(row.get("gene", "")),
            str(row.get("aa_change", "")),
            str(row.get("peptide_mut", "")),
            str(row.get("hla", "")),
            str(row.get("label", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_task_level_dataset(
    standardized_frames: list[pd.DataFrame],
    source_manifest: pd.DataFrame,
    *,
    num_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not standardized_frames:
        raise RuntimeError("Cannot build task-level full dataset without standardized source frames.")

    combined = pd.concat(standardized_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    source_meta = source_manifest[SOURCE_META_COLUMNS].rename(columns={"split_role": "source_split_role"}).copy()
    combined = combined.merge(source_meta, on=["source_id", "source_name"], how="left")
    combined["source_split_role"] = combined["source_split_role"].fillna("train_candidate")
    combined["source_priority"] = combined["source_priority"].fillna(999).astype(int)
    combined["label_strength"] = combined["label_strength"].fillna("unknown").astype(str)
    combined["download_method"] = combined["download_method"].fillna("unknown").astype(str)
    combined["canonical_event_key"] = (
        combined["gene"].astype(str)
        + "|"
        + combined["aa_change"].astype(str)
        + "|"
        + combined["peptide_mut"].astype(str)
        + "|"
        + combined["hla"].astype(str)
    )
    combined["sample_uid"] = combined.apply(_sample_uid, axis=1)

    source_roles = {
        str(row["source_id"]): str(row["split_role"])
        for row in source_manifest[["source_id", "split_role"]].to_dict(orient="records")
    }
    combined = assign_split_by_source_role(combined, source_roles, num_folds=num_folds)
    source_index = (
        combined.groupby(["source_id", "source_split_role", "split"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["source_split_role", "source_id", "split"])
        .reset_index(drop=True)
    )
    return combined, source_index


def write_source_template(settings: Settings, task_id: str, source_id: str) -> Path:
    bundle = load_manifest_bundle(settings, task_id)
    matches = bundle.source_manifest.loc[bundle.source_manifest["source_id"] == source_id]
    if matches.empty:
        raise RuntimeError(f"Unknown source_id {source_id!r} for task {task_id!r}")
    source_row = matches.iloc[0].to_dict()
    adapter_id = str(source_row["adapter_id"])
    try:
        filename, template = TEMPLATE_REGISTRY[adapter_id]
    except KeyError as exc:
        raise RuntimeError(f"No template is registered for adapter {adapter_id}") from exc
    raw_root = resolve_manifest_path(settings, str(source_row["raw_file_path"]))
    raw_root.mkdir(parents=True, exist_ok=True)
    output_path = raw_root / filename
    if output_path.suffix.lower() == ".tsv":
        template.to_csv(output_path, sep="\t", index=False)
    else:
        template.to_csv(output_path, index=False)
    return output_path
