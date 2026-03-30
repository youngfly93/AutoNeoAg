from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_processed_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_frame(df: pd.DataFrame, fold: int, target_split: str = "dev") -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = df[df["split"] == target_split].reset_index(drop=True)
    train_df = dev[dev["fold"] != fold].reset_index(drop=True)
    val_df = dev[dev["fold"] == fold].reset_index(drop=True)
    return train_df, val_df


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())

