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


def _foreignness_cache_path(settings: Settings, task: TaskSpec) -> Path:
    cache_dir = settings.artifacts_cache / "blast"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{task.task_id}_foreignness_cache.parquet"


def _run_blast_chunk(db_prefix: Path, peptides: list[str]) -> pd.DataFrame:
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
                    "peptide_mut": peptides[idx],
                    "blast_bitscore": bitscore,
                    "blast_pident": pident,
                    "foreignness_score": 1.0 - (pident / 100.0),
                }
            )
        return pd.DataFrame(rows)


def blast_foreignness(settings: Settings, task: TaskSpec, peptides: list[str], batch_size: int = 512) -> pd.DataFrame:
    db_prefix = ensure_blast_db(settings, task)
    request = pd.DataFrame({"peptide_mut": peptides}).reset_index(names="request_idx")
    unique_request = request[["peptide_mut"]].drop_duplicates().reset_index(drop=True)
    cache_path = _foreignness_cache_path(settings, task)
    if cache_path.exists():
        cached = pd.read_parquet(cache_path).drop_duplicates(subset=["peptide_mut"]).reset_index(drop=True)
    else:
        cached = pd.DataFrame(columns=["peptide_mut", "blast_bitscore", "blast_pident", "foreignness_score"])

    missing = unique_request.loc[~unique_request["peptide_mut"].isin(cached["peptide_mut"]), "peptide_mut"].tolist()
    if missing:
        fresh_frames = []
        for start in range(0, len(missing), batch_size):
            fresh_frames.append(_run_blast_chunk(db_prefix, missing[start : start + batch_size]))
        fresh = pd.concat(fresh_frames, ignore_index=True)
        cached = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(subset=["peptide_mut"], keep="last")
        cached.to_parquet(cache_path, index=False)

    merged = request.merge(cached, on="peptide_mut", how="left").sort_values("request_idx")
    return merged[["blast_bitscore", "blast_pident", "foreignness_score"]].reset_index(drop=True)
