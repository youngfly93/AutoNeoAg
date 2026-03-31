from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import Settings
from autoneoag.tasks import TaskSpec, raw_snapshot_path, resource_path


def load_smoke_seed(settings: Settings, task: TaskSpec) -> pd.DataFrame:
    path = resource_path(settings, task.smoke_dataset_resource)
    df = pd.read_csv(path, sep="\t")
    df["peptide_length"] = df["peptide_mut"].str.len()
    df["mutation_event"] = df["gene"] + ":" + df["aa_change"]
    return df


def write_raw_snapshot(df: pd.DataFrame, settings: Settings, task: TaskSpec, mode: str) -> Path:
    path = raw_snapshot_path(settings, task.task_id, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path
