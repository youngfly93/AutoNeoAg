from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import Settings
from autoneoag.tasks import TaskSpec, resource_path


def load_pseudosequences(settings: Settings, task: TaskSpec) -> dict[str, str]:
    path = resource_path(settings, task.context_resource)
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["hla"], df["pseudosequence"], strict=True))
