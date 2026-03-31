from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["peptide_mut", "hla", "label"]).reset_index(drop=True)


def _stable_fold(value: str, num_folds: int = 3) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_folds


def stable_fold(value: str, num_folds: int = 3) -> int:
    return _stable_fold(value, num_folds=num_folds)


def assign_splits(df: pd.DataFrame, num_folds: int = 3) -> pd.DataFrame:
    split = []
    fold = []
    for _, row in df.iterrows():
        study_id = str(row["study_id"])
        if study_id.startswith("STUDY-BLIND"):
            split.append("blind")
            fold.append(-1)
        elif study_id.startswith("STUDY-CONFIRM"):
            split.append("confirm")
            fold.append(-1)
        else:
            split.append("dev")
            fold.append(_stable_fold(f"{row['mutation_event']}|{row['hla']}", num_folds=num_folds))
    assigned = df.copy()
    assigned["split"] = split
    assigned["fold"] = fold
    return assigned


def assign_split_by_source_role(
    df: pd.DataFrame,
    source_roles: dict[str, str],
    *,
    num_folds: int = 3,
) -> pd.DataFrame:
    split = []
    fold = []
    for _, row in df.iterrows():
        role = str(source_roles.get(str(row["source_id"]), "train_candidate"))
        if role == "blind_only":
            split.append("blind")
            fold.append(-1)
        elif role == "confirm_candidate":
            split.append("confirm")
            fold.append(-1)
        elif role == "excluded_aux_only":
            split.append("excluded_aux")
            fold.append(-1)
        else:
            split.append("dev")
            fold_key = "|".join(
                [
                    str(row.get("mutation_event", "")),
                    str(row.get("hla", "")),
                    str(row.get("study_id", "")),
                    str(row.get("source_id", "")),
                ]
            )
            fold.append(stable_fold(fold_key, num_folds=num_folds))
    assigned = df.copy()
    assigned["split"] = split
    assigned["fold"] = fold
    return assigned


def write_manifest(df: pd.DataFrame, path: Path) -> Path:
    dev_folds = df[df["split"] == "dev"]["fold"]
    payload = {
        "rows": len(df),
        "splits": df["split"].value_counts().to_dict(),
        "folds": dev_folds.value_counts().sort_index().to_dict(),
        "num_dev_folds": int(dev_folds.nunique()),
        "challenge_splits": ["leave-study-out", "leave-HLA-supertype-out"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
