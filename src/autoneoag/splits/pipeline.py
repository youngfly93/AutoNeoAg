from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["peptide_mut", "hla", "label"]).reset_index(drop=True)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    split = []
    fold = []
    for _, row in df.iterrows():
        if row["study_id"] == "STUDY-BLIND":
            split.append("blind")
            fold.append(-1)
        elif row["study_id"] == "STUDY-CONFIRM":
            split.append("confirm")
            fold.append(-1)
        else:
            split.append("dev")
            fold.append(hash((row["mutation_event"], row["hla"])) % 3)
    assigned = df.copy()
    assigned["split"] = split
    assigned["fold"] = fold
    return assigned


def write_manifest(df: pd.DataFrame, path: Path) -> Path:
    payload = {
        "rows": len(df),
        "splits": df["split"].value_counts().to_dict(),
        "folds": df[df["split"] == "dev"]["fold"].value_counts().sort_index().to_dict(),
        "challenge_splits": ["leave-study-out", "leave-HLA-supertype-out"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path

