from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import Settings


def load_smoke_seed(settings: Settings) -> pd.DataFrame:
    resource = settings.root / "src" / "autoneoag" / "resources"
    path = resource / settings.smoke_dataset_resource
    df = pd.read_csv(path, sep="\t")
    df["peptide_length"] = df["peptide_mut"].str.len()
    df["mutation_event"] = df["gene"] + ":" + df["aa_change"]
    return df


def write_raw_snapshot(df: pd.DataFrame, output_dir: Path, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{mode}_raw.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path
