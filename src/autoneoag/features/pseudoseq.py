from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoneoag.config import Settings


def load_pseudosequences(settings: Settings) -> dict[str, str]:
    path = settings.root / "src" / "autoneoag" / "resources" / settings.smoke_hla_resource
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["hla"], df["pseudosequence"], strict=True))

