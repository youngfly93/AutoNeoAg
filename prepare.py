#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from autoneoag.config import ensure_directories, load_settings, require_env
from autoneoag.features.biochem import aromaticity, delta_residue_fraction, gravy, log_safe_ratio, non_polar_ratio
from autoneoag.features.dtu import netmhcpan_predict, netmhcstabpan_predict
from autoneoag.features.foreignness import blast_foreignness
from autoneoag.features.pseudoseq import load_pseudosequences
from autoneoag.ingest.public import load_smoke_seed, write_raw_snapshot
from autoneoag.splits.pipeline import assign_splits, exact_dedup, write_manifest


def _verify_hard_requirements(settings, mode: str) -> None:
    if not settings.netmhcpan_home.exists():
        raise RuntimeError(f"Missing NetMHCpan install at {settings.netmhcpan_home}")
    if not settings.netmhcstabpan_home.exists():
        raise RuntimeError(f"Missing NetMHCstabpan install at {settings.netmhcstabpan_home}")
    if mode == "full":
        require_env("SYNAPSE_USERNAME")
        if not (Path.home() / ".synapseConfig").exists() and not ("SYNAPSE_API_TOKEN" in __import__("os").environ):
            raise RuntimeError("Full mode requires Synapse credentials via ~/.synapseConfig or SYNAPSE_API_TOKEN")


def build_smoke_dataset(settings) -> pd.DataFrame:
    base = exact_dedup(load_smoke_seed(settings))
    write_raw_snapshot(base, settings.data_raw, "smoke")
    pseudoseqs = load_pseudosequences(settings)
    mut_aff = netmhcpan_predict(settings, base["peptide_mut"].tolist(), base["hla"].tolist()).drop_duplicates(
        subset=["peptide_mut", "hla"]
    )
    wt_aff = netmhcpan_predict(settings, base["peptide_wt"].tolist(), base["hla"].tolist()).rename(
        columns={
            "peptide_mut": "peptide_wt",
            "ba_score": "wt_ba_score",
            "el_score": "wt_el_score",
            "ba_rank": "wt_ba_rank",
            "el_rank": "wt_el_rank",
        }
    ).drop_duplicates(subset=["peptide_wt", "hla"])
    stab = netmhcstabpan_predict(settings, base["peptide_mut"].tolist(), base["hla"].tolist()).drop_duplicates(
        subset=["peptide_mut", "hla"]
    )
    foreignness = blast_foreignness(settings, base["peptide_mut"].tolist())
    df = base.merge(mut_aff, on=["peptide_mut", "hla"]).merge(wt_aff, on=["peptide_wt", "hla"]).merge(
        stab, on=["peptide_mut", "hla"]
    )
    df = pd.concat([df.reset_index(drop=True), foreignness.reset_index(drop=True)], axis=1)
    df["hla_pseudosequence"] = df["hla"].map(pseudoseqs)
    df["gravy"] = df["peptide_mut"].map(gravy)
    df["aromaticity"] = df["peptide_mut"].map(aromaticity)
    df["non_polar_ratio"] = df["peptide_mut"].map(non_polar_ratio)
    df["delta_fraction"] = [delta_residue_fraction(m, w) for m, w in zip(df["peptide_mut"], df["peptide_wt"], strict=True)]
    df["agretopicity"] = [log_safe_ratio(wt, mt) for wt, mt in zip(df["wt_ba_score"], df["ba_score"], strict=True)]
    df["expression_tpm"] = 0.0
    df = assign_splits(df, num_folds=settings.smoke_dev_num_folds)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    settings = load_settings(ROOT)
    ensure_directories(settings)
    _verify_hard_requirements(settings, args.mode)
    if args.mode == "full":
        raise RuntimeError("Full mode is wired but intentionally blocked until source manifests and credentials are provided.")
    df = build_smoke_dataset(settings)
    output_dir = settings.data_processed / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.parquet"
    df.to_parquet(output_path, index=False)
    manifest_path = write_manifest(df, settings.split_manifest)
    print(f"dataset_path: {output_path}")
    print(f"manifest_path: {manifest_path}")
    print(f"rows: {len(df)}")
    print(f"splits: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
