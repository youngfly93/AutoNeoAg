#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

from autoneoag.config import ensure_directories, load_settings
from autoneoag.features.pseudoseq import load_pseudosequences
from autoneoag.ingest.full import standardize_iedb_functional_immunology, standardize_tumoragdb2_curated
from autoneoag.manifests import load_manifest_bundle
from autoneoag.tasks import get_task_spec


def _supported_hla_subset(settings, task_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    task = get_task_spec(task_id)
    supported = set(load_pseudosequences(settings, task).keys())
    return frame.loc[frame["hla"].astype(str).isin(supported)].reset_index(drop=True)


def _write_bundle(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_official_tables(raw_dir: Path) -> pd.DataFrame:
    files = sorted(
        file_path
        for file_path in raw_dir.iterdir()
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and not file_path.name.startswith("template_")
        and not file_path.name.startswith("real_")
        and file_path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
    )
    if not files:
        raise RuntimeError(f"No official raw files found at {raw_dir}")
    frames = []
    for file_path in files:
        if file_path.suffix.lower() == ".tsv":
            frame = pd.read_csv(file_path, sep="\t")
        elif file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
        else:
            frame = pd.read_excel(file_path)
        frame["__source_file__"] = str(file_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def materialize_neoantigen_real_sources(settings) -> dict[str, object]:
    bundle = load_manifest_bundle(settings, "neoantigen")
    source_row = bundle.source_manifest.loc[bundle.source_manifest["source_id"] == "neo_tumoragdb2_core"].iloc[0].to_dict()
    raw_dir = ROOT / "data" / "raw" / "neoantigen" / "tumoragdb2"
    standardized = _supported_hla_subset(
        settings,
        "neoantigen",
        standardize_tumoragdb2_curated(_load_official_tables(raw_dir), source_row),
    )

    core = standardized.loc[standardized["study_id"] == "NCI"].reset_index(drop=True)
    confirm = standardized.loc[standardized["study_id"] == "HiTIDE"].reset_index(drop=True)
    blind = standardized.loc[standardized["study_id"] == "TESLA"].reset_index(drop=True)

    outputs = {
        "neo_tumoragdb2_core": ROOT / "data" / "raw" / "neoantigen" / "tumoragdb2" / "real_neo_tumoragdb2_core.tsv",
        "neo_literature_curated": ROOT
        / "data"
        / "raw"
        / "neoantigen"
        / "literature_curated"
        / "real_neo_literature_curated.tsv",
        "neo_tesla": ROOT / "data" / "raw" / "neoantigen" / "tesla" / "real_neo_tesla_lockbox.tsv",
    }
    _write_bundle(outputs["neo_tumoragdb2_core"], core)
    _write_bundle(outputs["neo_literature_curated"], confirm)
    _write_bundle(outputs["neo_tesla"], blind)

    payload = {
        "task_id": "neoantigen",
        "total_supported_rows": int(len(standardized)),
        "core_rows": int(len(core)),
        "confirm_rows": int(len(confirm)),
        "blind_rows": int(len(blind)),
        "core_label_counts": {str(key): int(value) for key, value in core["label"].value_counts().to_dict().items()},
        "confirm_label_counts": {str(key): int(value) for key, value in confirm["label"].value_counts().to_dict().items()},
        "blind_label_counts": {str(key): int(value) for key, value in blind["label"].value_counts().to_dict().items()},
    }
    _write_metadata(ROOT / "data" / "raw" / "neoantigen" / "real_source_split_meta.json", payload)
    return payload


def _choose_study_holdouts(frame: pd.DataFrame, *, blind_target: int, confirm_target: int) -> tuple[list[str], list[str]]:
    counts = frame["study_id"].astype(str).value_counts()
    max_share = max(int(len(frame) * 0.20), 1)
    blind_ids: list[str] = []
    blind_rows = 0
    for study_id, count in counts.items():
        if count > max_share:
            continue
        blind_ids.append(str(study_id))
        blind_rows += int(count)
        if blind_rows >= blind_target:
            break

    confirm_ids: list[str] = []
    confirm_rows = 0
    for study_id, count in counts.items():
        if str(study_id) in blind_ids or count > max_share:
            continue
        confirm_ids.append(str(study_id))
        confirm_rows += int(count)
        if confirm_rows >= confirm_target:
            break
    return blind_ids, confirm_ids


def materialize_immunogenicity_real_sources(settings) -> dict[str, object]:
    bundle = load_manifest_bundle(settings, "hla_immunogenicity")
    source_row = bundle.source_manifest.loc[bundle.source_manifest["source_id"] == "immuno_iedb_functional"].iloc[0].to_dict()
    raw_dir = ROOT / "data" / "raw" / "hla_immunogenicity" / "iedb_functional"
    standardized = _supported_hla_subset(
        settings,
        "hla_immunogenicity",
        standardize_iedb_functional_immunology(_load_official_tables(raw_dir), source_row),
    )
    time_holdout = standardized.loc[standardized["source_year"].astype(int) >= 2024].reset_index(drop=True)
    candidate_pool = standardized.loc[standardized["source_year"].astype(int) < 2024].reset_index(drop=True)

    blind_ids, confirm_ids = _choose_study_holdouts(
        candidate_pool,
        blind_target=max(int(len(standardized) * 0.15), 400),
        confirm_target=max(int(len(standardized) * 0.10), 300),
    )
    blind = candidate_pool.loc[candidate_pool["study_id"].astype(str).isin(blind_ids)].reset_index(drop=True)
    confirm = candidate_pool.loc[candidate_pool["study_id"].astype(str).isin(confirm_ids)].reset_index(drop=True)
    train = candidate_pool.loc[
        ~candidate_pool["study_id"].astype(str).isin(set(blind_ids) | set(confirm_ids))
    ].reset_index(drop=True)

    outputs = {
        "immuno_iedb_functional": ROOT
        / "data"
        / "raw"
        / "hla_immunogenicity"
        / "iedb_functional"
        / "real_immuno_iedb_train.tsv",
        "immuno_curated_literature": ROOT
        / "data"
        / "raw"
        / "hla_immunogenicity"
        / "literature_curated"
        / "real_immuno_confirm.tsv",
        "immuno_external_study": ROOT
        / "data"
        / "raw"
        / "hla_immunogenicity"
        / "external_study_holdout"
        / "real_immuno_external_lockbox.tsv",
    }
    _write_bundle(outputs["immuno_iedb_functional"], train)
    _write_bundle(outputs["immuno_curated_literature"], confirm)
    _write_bundle(outputs["immuno_external_study"], blind)
    if not time_holdout.empty:
        time_path = ROOT / "data" / "raw" / "hla_immunogenicity" / "2024plus_holdout" / "real_immuno_2024plus.tsv"
        _write_bundle(time_path, time_holdout)

    payload = {
        "task_id": "hla_immunogenicity",
        "total_supported_rows": int(len(standardized)),
        "time_holdout_rows": int(len(time_holdout)),
        "train_rows": int(len(train)),
        "confirm_rows": int(len(confirm)),
        "blind_rows": int(len(blind)),
        "blind_studies": blind_ids,
        "confirm_studies": confirm_ids,
        "train_label_counts": {str(key): int(value) for key, value in train["label"].value_counts().to_dict().items()},
        "confirm_label_counts": {str(key): int(value) for key, value in confirm["label"].value_counts().to_dict().items()},
        "blind_label_counts": {str(key): int(value) for key, value in blind["label"].value_counts().to_dict().items()},
    }
    _write_metadata(ROOT / "data" / "raw" / "hla_immunogenicity" / "real_source_split_meta.json", payload)
    return payload


def main() -> None:
    settings = load_settings(ROOT)
    ensure_directories(settings)
    neo_payload = materialize_neoantigen_real_sources(settings)
    immuno_payload = materialize_immunogenicity_real_sources(settings)
    print(json.dumps({"neoantigen": neo_payload, "hla_immunogenicity": immuno_payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
