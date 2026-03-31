from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from autoneoag.config import Settings
from autoneoag.tasks import TaskSpec, resource_path


def ensure_blast_db(settings: Settings, task: TaskSpec) -> Path:
    if task.reference_resource is None:
        raise RuntimeError(f"Task {task.task_id} does not define a foreignness reference resource.")
    ref_path = resource_path(settings, task.reference_resource)
    db_dir = settings.artifacts_cache / "blast"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / f"{task.task_id}_reference"
    if not (db_prefix.with_suffix(".pin").exists() or db_prefix.with_suffix(".psq").exists()):
        subprocess.run(
            [
                "makeblastdb",
                "-in",
                str(ref_path),
                "-dbtype",
                "prot",
                "-out",
                str(db_prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return db_prefix


def blast_foreignness(settings: Settings, task: TaskSpec, peptides: list[str]) -> pd.DataFrame:
    db_prefix = ensure_blast_db(settings, task)
    with tempfile.TemporaryDirectory() as tmpdir:
        query = Path(tmpdir) / "peptides.faa"
        with query.open("w") as handle:
            for idx, peptide in enumerate(peptides):
                handle.write(f">pep_{idx}\n{peptide}\n")
        result = Path(tmpdir) / "blast.tsv"
        subprocess.run(
            [
                "blastp",
                "-task",
                "blastp-short",
                "-query",
                str(query),
                "-db",
                str(db_prefix),
                "-outfmt",
                "6 qseqid pident bitscore evalue sseqid",
                "-out",
                str(result),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        best: dict[str, tuple[float, float]] = {}
        if result.exists():
            for line in result.read_text().splitlines():
                qseqid, pident, bitscore, _evalue, _sseqid = line.split("\t")
                current = best.get(qseqid)
                candidate = (float(bitscore), float(pident))
                if current is None or candidate > current:
                    best[qseqid] = candidate
        rows = []
        for idx, _peptide in enumerate(peptides):
            bitscore, pident = best.get(f"pep_{idx}", (0.0, 0.0))
            rows.append(
                {
                    "blast_bitscore": bitscore,
                    "blast_pident": pident,
                    "foreignness_score": 1.0 - (pident / 100.0),
                }
            )
        return pd.DataFrame(rows)
