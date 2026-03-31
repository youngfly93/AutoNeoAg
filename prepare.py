#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

import pandas as pd

from autoneoag.config import ensure_directories, load_settings, require_env
from autoneoag.features.biochem import aromaticity, delta_residue_fraction, gravy, log_safe_ratio, non_polar_ratio
from autoneoag.features.dtu import netmhcpan_predict, netmhcstabpan_predict
from autoneoag.features.foreignness import blast_foreignness
from autoneoag.ingest.full import build_task_level_dataset, run_source_adapter
from autoneoag.manifests import load_manifest_bundle, resolve_manifest_path, write_manifest_summary
from autoneoag.features.pseudoseq import load_pseudosequences
from autoneoag.ingest.public import load_smoke_seed, write_raw_snapshot
from autoneoag.splits.pipeline import assign_splits, exact_dedup, write_manifest
from autoneoag.tasks import get_task_spec, list_task_ids, processed_dataset_path, split_manifest_path, task_interim_dir


def _verify_hard_requirements(settings, task, mode: str) -> None:
    if task.require_dtu and not settings.netmhcpan_home.exists():
        raise RuntimeError(f"Missing NetMHCpan install at {settings.netmhcpan_home}")
    if task.require_dtu and not settings.netmhcstabpan_home.exists():
        raise RuntimeError(f"Missing NetMHCstabpan install at {settings.netmhcstabpan_home}")


def _verify_manifest_credentials(bundle) -> None:
    download_methods = set(bundle.source_manifest["download_method"].tolist())
    if "synapse_download" in download_methods:
        require_env("SYNAPSE_USERNAME")
        if not (Path.home() / ".synapseConfig").exists() and not ("SYNAPSE_API_TOKEN" in __import__("os").environ):
            raise RuntimeError("Synapse-backed full ingest requires ~/.synapseConfig or SYNAPSE_API_TOKEN")


def _attach_common_features(df: pd.DataFrame) -> pd.DataFrame:
    attached = df.copy()
    attached["gravy"] = attached["peptide_mut"].map(gravy)
    attached["aromaticity"] = attached["peptide_mut"].map(aromaticity)
    attached["non_polar_ratio"] = attached["peptide_mut"].map(non_polar_ratio)
    attached["delta_fraction"] = [
        delta_residue_fraction(m, w) for m, w in zip(attached["peptide_mut"], attached["peptide_wt"], strict=True)
    ]
    attached["expression_tpm"] = attached.get("expression_tpm", 0.0)
    return attached


def materialize_immunology_features(settings, task, base: pd.DataFrame) -> pd.DataFrame:
    pseudoseqs = load_pseudosequences(settings, task)
    attached = base.copy().reset_index(drop=True)
    attached["hla_pseudosequence"] = attached["hla"].map(pseudoseqs)
    if attached["hla_pseudosequence"].isna().any():
        missing = sorted(attached.loc[attached["hla_pseudosequence"].isna(), "hla"].unique().tolist())
        raise RuntimeError(f"Missing HLA pseudosequences for task {task.task_id}: {missing}")

    mut_queries = attached[["peptide_mut", "hla"]].drop_duplicates().reset_index(drop=True)
    mut_aff = netmhcpan_predict(settings, mut_queries["peptide_mut"].tolist(), mut_queries["hla"].tolist()).drop_duplicates(
        subset=["peptide_mut", "hla"]
    )
    wt_queries = attached[["peptide_wt", "hla"]].drop_duplicates().reset_index(drop=True)
    wt_aff = netmhcpan_predict(settings, wt_queries["peptide_wt"].tolist(), wt_queries["hla"].tolist()).rename(
        columns={
            "peptide_mut": "peptide_wt",
            "ba_score": "wt_ba_score",
            "el_score": "wt_el_score",
            "ba_rank": "wt_ba_rank",
            "el_rank": "wt_el_rank",
        }
    ).drop_duplicates(subset=["peptide_wt", "hla"])
    stab = netmhcstabpan_predict(settings, mut_queries["peptide_mut"].tolist(), mut_queries["hla"].tolist()).drop_duplicates(
        subset=["peptide_mut", "hla"]
    )
    foreignness = blast_foreignness(settings, task, attached["peptide_mut"].tolist())

    df = attached.merge(mut_aff, on=["peptide_mut", "hla"]).merge(wt_aff, on=["peptide_wt", "hla"]).merge(
        stab, on=["peptide_mut", "hla"]
    )
    df = pd.concat([df.reset_index(drop=True), foreignness.reset_index(drop=True)], axis=1)
    df["agretopicity"] = [log_safe_ratio(wt, mt) for wt, mt in zip(df["wt_ba_score"], df["ba_score"], strict=True)]
    return _attach_common_features(df)


def build_immunology_smoke_dataset(settings, task) -> pd.DataFrame:
    base = exact_dedup(load_smoke_seed(settings, task))
    write_raw_snapshot(base, settings, task, "smoke")
    return assign_splits(materialize_immunology_features(settings, task, base), num_folds=task.dev_num_folds)


def build_generic_pairwise_smoke_dataset(settings, task) -> pd.DataFrame:
    df = exact_dedup(load_smoke_seed(settings, task))
    write_raw_snapshot(df, settings, task, "smoke")
    pseudoseqs = load_pseudosequences(settings, task)
    df["hla_pseudosequence"] = df["hla"].map(pseudoseqs)
    if df["hla_pseudosequence"].isna().any():
        missing = sorted(df.loc[df["hla_pseudosequence"].isna(), "hla"].unique().tolist())
        raise RuntimeError(f"Missing context pseudosequences for task {task.task_id}: {missing}")
    if "agretopicity" not in df.columns:
        df["agretopicity"] = [log_safe_ratio(wt, mt) for wt, mt in zip(df["wt_ba_score"], df["ba_score"], strict=True)]
    required_scalar_columns = [
        "ba_score",
        "el_score",
        "ba_rank",
        "el_rank",
        "wt_ba_score",
        "wt_el_score",
        "wt_ba_rank",
        "wt_el_rank",
        "stab_score",
        "stab_rank",
        "foreignness_score",
        "blast_bitscore",
        "blast_pident",
    ]
    missing = [column for column in required_scalar_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Task {task.task_id} smoke seed is missing required proxy columns: {missing}")
    return assign_splits(_attach_common_features(df), num_folds=task.dev_num_folds)


def build_dataset(settings, task, mode: str) -> pd.DataFrame:
    if mode != "smoke":
        raise RuntimeError(f"Task {task.task_id} full ingest is not enabled yet.")
    if task.family == "immunology":
        return build_immunology_smoke_dataset(settings, task)
    if task.family == "generic_pairwise":
        return build_generic_pairwise_smoke_dataset(settings, task)
    raise RuntimeError(f"Unsupported task family: {task.family}")


def stage_full_preparation_plan(settings, task) -> dict[str, object]:
    bundle = load_manifest_bundle(settings, task.task_id)
    _verify_manifest_credentials(bundle)

    interim_dir = task_interim_dir(settings, task.task_id, "full")
    interim_dir.mkdir(parents=True, exist_ok=True)

    summary_path = write_manifest_summary(bundle, interim_dir / "manifest_summary.json")

    staged_sources = bundle.source_manifest.copy()
    staged_sources["resolved_raw_path"] = [str(resolve_manifest_path(settings, value)) for value in staged_sources["raw_file_path"]]
    staged_sources["raw_exists"] = [Path(path).exists() for path in staged_sources["resolved_raw_path"]]
    staged_sources["adapter_ready"] = staged_sources["ingest_status"] == "implemented"
    staged_sources_path = interim_dir / "source_manifest_staged.tsv"
    staged_sources.to_csv(staged_sources_path, sep="\t", index=False)

    blocked_sources = staged_sources.loc[staged_sources["ingest_status"] != "implemented", "source_id"].tolist()
    missing_raw_sources = staged_sources.loc[~staged_sources["raw_exists"], "source_id"].tolist()
    standardized_dir = interim_dir / "sources"
    standardized_dir.mkdir(parents=True, exist_ok=True)
    standardized_outputs: list[str] = []
    adapter_failures: dict[str, str] = {}
    standardized_frames: list[pd.DataFrame] = []
    for row in staged_sources.to_dict(orient="records"):
        if row["ingest_status"] != "implemented":
            continue
        if not row["raw_exists"]:
            continue
        try:
            standardized = run_source_adapter(settings, row)
        except Exception as exc:
            adapter_failures[str(row["source_id"])] = str(exc)
            continue
        output_path = standardized_dir / f"{row['source_id']}.parquet"
        standardized.to_parquet(output_path, index=False)
        standardized_outputs.append(str(output_path))
        standardized_frames.append(standardized)

    combined_path = None
    full_dataset_path = None
    full_split_manifest_path = None
    source_index_path = None
    full_rows = 0
    full_splits: dict[str, int] = {}
    if standardized_frames:
        combined = pd.concat(standardized_frames, ignore_index=True)
        combined_path = standardized_dir / "standardized_sources.parquet"
        combined.to_parquet(combined_path, index=False)
        full_dataset, source_index = build_task_level_dataset(
            standardized_frames,
            staged_sources,
            num_folds=task.dev_num_folds,
        )
        if task.family == "immunology":
            full_dataset = materialize_immunology_features(settings, task, full_dataset)
        full_dataset_path = processed_dataset_path(settings, task.task_id, "full")
        full_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        full_dataset.to_parquet(full_dataset_path, index=False)
        full_split_manifest_path = write_manifest(full_dataset, split_manifest_path(settings, task.task_id, "full"))
        source_index_path = interim_dir / "source_index.tsv"
        source_index.to_csv(source_index_path, sep="\t", index=False)
        full_rows = int(len(full_dataset))
        full_splits = {str(key): int(value) for key, value in full_dataset["split"].value_counts().to_dict().items()}

    plan_payload = {
        "task_id": task.task_id,
        "status": "phase2_sources_staged" if full_dataset_path is None else "phase2_dataset_materialized",
        "summary_path": str(summary_path),
        "staged_source_manifest_path": str(staged_sources_path),
        "num_sources": int(len(staged_sources)),
        "implemented_sources": staged_sources.loc[staged_sources["ingest_status"] == "implemented", "source_id"].tolist(),
        "blocked_sources": blocked_sources,
        "missing_raw_sources": missing_raw_sources,
        "standardized_outputs": standardized_outputs,
        "adapter_failures": adapter_failures,
        "combined_standardized_path": str(combined_path) if combined_path is not None else "",
        "full_dataset_path": str(full_dataset_path) if full_dataset_path is not None else "",
        "full_split_manifest_path": str(full_split_manifest_path) if full_split_manifest_path is not None else "",
        "source_index_path": str(source_index_path) if source_index_path is not None else "",
        "full_rows": full_rows,
        "full_splits": full_splits,
        "train_candidate_sources": staged_sources.loc[staged_sources["split_role"] == "train_candidate", "source_id"].tolist(),
        "blind_only_sources": staged_sources.loc[staged_sources["split_role"] == "blind_only", "source_id"].tolist(),
    }
    plan_path = interim_dir / "full_prepare_plan.json"
    plan_path.write_text(json.dumps(plan_payload, indent=2, sort_keys=True))
    return {
        "task_id": task.task_id,
        "summary_path": summary_path,
        "staged_source_manifest_path": staged_sources_path,
        "plan_path": plan_path,
        "blocked_sources": blocked_sources,
        "missing_raw_sources": missing_raw_sources,
        "standardized_outputs": standardized_outputs,
        "adapter_failures": adapter_failures,
        "full_dataset_path": full_dataset_path,
        "full_split_manifest_path": full_split_manifest_path,
        "source_index_path": source_index_path,
        "full_rows": full_rows,
        "full_splits": full_splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list_task_ids(), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    settings = load_settings(ROOT)
    ensure_directories(settings)
    task = get_task_spec(args.task)
    _verify_hard_requirements(settings, task, args.mode)
    if args.mode == "full" and not task.full_enabled:
        raise RuntimeError(f"Task {task.task_id} full ingest is intentionally blocked until source manifests are provided.")
    if args.mode == "full":
        plan = stage_full_preparation_plan(settings, task)
        print(f"task_id: {plan['task_id']}")
        print(f"manifest_summary_path: {plan['summary_path']}")
        print(f"staged_source_manifest_path: {plan['staged_source_manifest_path']}")
        print(f"full_prepare_plan_path: {plan['plan_path']}")
        print(f"blocked_sources: {plan['blocked_sources']}")
        print(f"missing_raw_sources: {plan['missing_raw_sources']}")
        print(f"standardized_outputs: {plan['standardized_outputs']}")
        print(f"adapter_failures: {plan['adapter_failures']}")
        print(f"full_dataset_path: {plan['full_dataset_path']}")
        print(f"full_split_manifest_path: {plan['full_split_manifest_path']}")
        print(f"source_index_path: {plan['source_index_path']}")
        print(f"full_rows: {plan['full_rows']}")
        print(f"full_splits: {plan['full_splits']}")
        return
    df = build_dataset(settings, task, args.mode)
    output_path = processed_dataset_path(settings, task.task_id, args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    manifest_path = write_manifest(df, split_manifest_path(settings, task.task_id, args.mode))
    print(f"dataset_path: {output_path}")
    print(f"manifest_path: {manifest_path}")
    print(f"task_id: {task.task_id}")
    print(f"rows: {len(df)}")
    print(f"splits: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
