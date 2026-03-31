from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from autoneoag.config import Settings


SOURCE_MANIFEST_COLUMNS = [
    "source_id",
    "source_name",
    "task_id",
    "source_type",
    "download_method",
    "adapter_id",
    "ingest_status",
    "license_or_access",
    "raw_file_path",
    "source_priority",
    "expected_format",
    "assay_scope",
    "species_scope",
    "hla_scope",
    "split_role",
    "label_strength",
    "is_train_eligible",
    "is_confirm_eligible",
    "is_blind_only",
    "year_start",
    "year_end",
    "normalization_profile",
    "notes",
]

LOCKBOX_MANIFEST_COLUMNS = [
    "selector_id",
    "task_id",
    "lockbox_name",
    "selector_type",
    "selector_value",
    "reason",
    "allowed_for_training",
    "allowed_for_confirm",
    "allowed_for_blind",
    "notes",
]

ALLOWED_BOOL_COLUMNS = [
    "is_train_eligible",
    "is_confirm_eligible",
    "is_blind_only",
    "allowed_for_training",
    "allowed_for_confirm",
    "allowed_for_blind",
]

ALLOWED_SOURCE_SPLIT_ROLES = {
    "train_candidate",
    "confirm_candidate",
    "blind_only",
    "excluded_aux_only",
}

ALLOWED_INGEST_STATUS = {
    "planned",
    "manual_required",
    "implemented",
    "external_lockbox",
}


@dataclass(frozen=True)
class ManifestBundle:
    task_id: str
    root: Path
    data_card_path: Path
    source_manifest_path: Path
    lockbox_manifest_path: Path
    split_manifest_path: Path
    task_policy_path: Path
    source_manifest: pd.DataFrame
    lockbox_manifest: pd.DataFrame
    split_manifest: dict[str, object]


def manifest_root(settings: Settings, task_id: str) -> Path:
    return settings.root / "manifests" / task_id


def resolve_manifest_path(settings: Settings, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (settings.root / path).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing manifest file: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(f"Missing manifest file: {path}")
    return json.loads(path.read_text())


def _assert_columns(frame: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Manifest {path} is missing required columns: {missing}")


def _assert_task_id(frame: pd.DataFrame, task_id: str, path: Path) -> None:
    if "task_id" not in frame.columns:
        return
    values = sorted({str(value) for value in frame["task_id"].dropna().tolist()})
    if values != [task_id]:
        raise RuntimeError(f"Manifest {path} has mismatched task_id values: {values}, expected {task_id}")


def _assert_bool_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = {str(value) for value in frame[column].dropna().tolist()}
        if not values.issubset({"0", "1", "0.0", "1.0"}):
            raise RuntimeError(f"Manifest {path} column {column} must contain only 0/1 values, found {sorted(values)}")


def _assert_source_constraints(frame: pd.DataFrame, path: Path) -> None:
    split_roles = {str(value) for value in frame["split_role"].dropna().tolist()}
    if not split_roles.issubset(ALLOWED_SOURCE_SPLIT_ROLES):
        raise RuntimeError(f"Manifest {path} has unsupported split_role values: {sorted(split_roles - ALLOWED_SOURCE_SPLIT_ROLES)}")

    ingest_status = {str(value) for value in frame["ingest_status"].dropna().tolist()}
    if not ingest_status.issubset(ALLOWED_INGEST_STATUS):
        raise RuntimeError(f"Manifest {path} has unsupported ingest_status values: {sorted(ingest_status - ALLOWED_INGEST_STATUS)}")

    duplicates = frame[frame.duplicated(subset=["source_id"], keep=False)]["source_id"].tolist()
    if duplicates:
        raise RuntimeError(f"Manifest {path} contains duplicate source_id values: {sorted(set(duplicates))}")


def _assert_lockbox_constraints(source_frame: pd.DataFrame, lockbox_frame: pd.DataFrame, path: Path) -> None:
    source_ids = set(source_frame["source_id"].tolist())
    invalid = []
    for _, row in lockbox_frame.iterrows():
        if row["selector_type"] == "source_id" and row["selector_value"] not in source_ids:
            invalid.append(str(row["selector_value"]))
    if invalid:
        raise RuntimeError(f"Manifest {path} references unknown source_id values: {sorted(set(invalid))}")


def load_manifest_bundle(settings: Settings, task_id: str) -> ManifestBundle:
    root = manifest_root(settings, task_id)
    bundle = ManifestBundle(
        task_id=task_id,
        root=root,
        data_card_path=root / "data_card.md",
        source_manifest_path=root / "source_manifest.csv",
        lockbox_manifest_path=root / "lockbox_manifest.csv",
        split_manifest_path=root / "split_manifest.json",
        task_policy_path=root / "task_policy.md",
        source_manifest=_read_csv(root / "source_manifest.csv"),
        lockbox_manifest=_read_csv(root / "lockbox_manifest.csv"),
        split_manifest=_read_json(root / "split_manifest.json"),
    )
    validate_manifest_bundle(bundle)
    return bundle


def validate_manifest_bundle(bundle: ManifestBundle) -> None:
    for path in (bundle.data_card_path, bundle.task_policy_path):
        if not path.exists():
            raise RuntimeError(f"Missing manifest file: {path}")

    _assert_columns(bundle.source_manifest, SOURCE_MANIFEST_COLUMNS, bundle.source_manifest_path)
    _assert_columns(bundle.lockbox_manifest, LOCKBOX_MANIFEST_COLUMNS, bundle.lockbox_manifest_path)
    _assert_task_id(bundle.source_manifest, bundle.task_id, bundle.source_manifest_path)
    _assert_task_id(bundle.lockbox_manifest, bundle.task_id, bundle.lockbox_manifest_path)
    _assert_bool_columns(bundle.source_manifest, ALLOWED_BOOL_COLUMNS, bundle.source_manifest_path)
    _assert_bool_columns(bundle.lockbox_manifest, ALLOWED_BOOL_COLUMNS, bundle.lockbox_manifest_path)
    _assert_source_constraints(bundle.source_manifest, bundle.source_manifest_path)
    _assert_lockbox_constraints(bundle.source_manifest, bundle.lockbox_manifest, bundle.lockbox_manifest_path)

    split_task_id = str(bundle.split_manifest.get("task_id", ""))
    if split_task_id != bundle.task_id:
        raise RuntimeError(
            f"Manifest {bundle.split_manifest_path} has task_id={split_task_id!r}, expected {bundle.task_id!r}"
        )


def manifest_summary(bundle: ManifestBundle) -> dict[str, object]:
    source_manifest = bundle.source_manifest.copy()
    lockbox_manifest = bundle.lockbox_manifest.copy()
    return {
        "task_id": bundle.task_id,
        "num_sources": int(len(source_manifest)),
        "source_ids": source_manifest["source_id"].tolist(),
        "train_eligible_sources": source_manifest.loc[source_manifest["is_train_eligible"] == 1, "source_id"].tolist(),
        "confirm_eligible_sources": source_manifest.loc[source_manifest["is_confirm_eligible"] == 1, "source_id"].tolist(),
        "blind_only_sources": source_manifest.loc[source_manifest["is_blind_only"] == 1, "source_id"].tolist(),
        "source_split_roles": source_manifest["split_role"].value_counts().to_dict(),
        "ingest_status_counts": source_manifest["ingest_status"].value_counts().to_dict(),
        "lockbox_names": lockbox_manifest["lockbox_name"].value_counts().to_dict(),
        "dev_num_folds": bundle.split_manifest.get("dev_num_folds"),
        "challenge_splits": bundle.split_manifest.get("challenge_splits", []),
    }


def write_manifest_summary(bundle: ManifestBundle, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest_summary(bundle), indent=2, sort_keys=True))
    return output_path
