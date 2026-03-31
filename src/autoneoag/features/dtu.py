from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from autoneoag.config import Settings


def _require_executable(path: Path, script_name: str) -> Path:
    script = path / script_name
    if not script.exists():
        raise RuntimeError(f"Required DTU script is missing: {script}")
    return script


def _normalize_allele(allele: str) -> str:
    return allele.replace("*", "")


def _run_tool(script: Path, allele: str, peptides: list[str], extra_args: list[str]) -> str:
    if not peptides:
        return ""
    with tempfile.TemporaryDirectory() as tmpdir:
        peptide_file = Path(tmpdir) / "peptides.txt"
        with peptide_file.open("w") as handle:
            handle.write("\n".join(peptides))
            handle.write("\n")
        cmd = [str(script), "-p", "-f", str(peptide_file), "-a", _normalize_allele(allele), *extra_args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _parse_affinity_output(stdout: str, peptides: list[str]) -> pd.DataFrame:
    rows = {peptide: {"ba_score": None, "el_score": None, "ba_rank": None, "el_rank": None} for peptide in peptides}
    peptide_set = set(peptides)
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        tokens = re.split(r"\s+", line)
        peptide = next((token for token in tokens if token in peptide_set), None)
        if peptide is None:
            continue
        floats = []
        for token in tokens:
            try:
                floats.append(float(token))
            except ValueError:
                continue
        if len(floats) >= 5:
            rows[peptide] = {
                "ba_score": floats[-1],
                "el_score": floats[-5],
                "ba_rank": floats[-2],
                "el_rank": floats[-4],
            }
    return pd.DataFrame([rows[p] for p in peptides])


def _parse_stability_output(stdout: str, peptides: list[str]) -> pd.DataFrame:
    rows = {peptide: {"stab_score": None, "stab_rank": None} for peptide in peptides}
    peptide_set = set(peptides)
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        tokens = re.split(r"\s+", line)
        peptide = next((token for token in tokens if token in peptide_set), None)
        if peptide is None:
            continue
        floats = []
        for token in tokens:
            try:
                floats.append(float(token))
            except ValueError:
                continue
        if len(floats) >= 3:
            rows[peptide] = {"stab_score": floats[-2], "stab_rank": floats[-1]}
    return pd.DataFrame([rows[p] for p in peptides])


def _dtu_cache_dir(settings: Settings) -> Path:
    cache_dir = settings.artifacts_cache / "dtu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_cache(path: Path, columns: list[str], subset: list[str]) -> pd.DataFrame:
    if path.exists():
        cached = pd.read_parquet(path)
        for column in columns:
            if column not in cached.columns:
                cached[column] = None
        return cached[columns].drop_duplicates(subset=subset).reset_index(drop=True)
    return pd.DataFrame(columns=columns)


def _write_cache(path: Path, cached: pd.DataFrame, fresh: pd.DataFrame, *, subset: list[str]) -> None:
    merged = pd.concat([cached, fresh], ignore_index=True)
    merged = merged.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    merged.to_parquet(path, index=False)


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def netmhcpan_predict(settings: Settings, peptides: list[str], alleles: list[str], batch_size: int = 256) -> pd.DataFrame:
    script = _require_executable(settings.netmhcpan_home, "netMHCpan")
    cache_path = _dtu_cache_dir(settings) / "netmhcpan_cache.parquet"
    columns = ["peptide_mut", "hla", "ba_score", "el_score", "ba_rank", "el_rank"]
    subset = ["peptide_mut", "hla"]
    request = pd.DataFrame({"peptide_mut": peptides, "hla": alleles}).reset_index(names="request_idx")
    unique_request = request[["peptide_mut", "hla"]].drop_duplicates().reset_index(drop=True)
    cached = _load_cache(cache_path, columns, subset)
    resolved = unique_request.merge(cached, on=["peptide_mut", "hla"], how="left")
    missing = resolved[resolved["ba_score"].isna()][["peptide_mut", "hla"]].drop_duplicates().reset_index(drop=True)

    fresh_frames = []
    for allele in sorted(missing["hla"].unique().tolist()):
        allele_peptides = missing.loc[missing["hla"] == allele, "peptide_mut"].tolist()
        for peptide_batch in _batched(allele_peptides, batch_size):
            stdout = _run_tool(script, allele, peptide_batch, ["-BA"])
            frame = _parse_affinity_output(stdout, peptide_batch)
            frame["hla"] = allele
            frame["peptide_mut"] = peptide_batch
            fresh_frames.append(frame[columns])
    if fresh_frames:
        fresh = pd.concat(fresh_frames, ignore_index=True)
        _write_cache(cache_path, cached, fresh, subset=subset)
        cached = _load_cache(cache_path, columns, subset)

    merged = request.merge(cached, on=["peptide_mut", "hla"], how="left").sort_values("request_idx")
    return merged[columns].reset_index(drop=True)


def netmhcstabpan_predict(settings: Settings, peptides: list[str], alleles: list[str], batch_size: int = 256) -> pd.DataFrame:
    script = _require_executable(settings.netmhcstabpan_home, "netMHCstabpan")
    cache_path = _dtu_cache_dir(settings) / "netmhcstabpan_cache.parquet"
    columns = ["peptide_mut", "hla", "stab_score", "stab_rank"]
    subset = ["peptide_mut", "hla"]
    request = pd.DataFrame({"peptide_mut": peptides, "hla": alleles}).reset_index(names="request_idx")
    unique_request = request[["peptide_mut", "hla"]].drop_duplicates().reset_index(drop=True)
    cached = _load_cache(cache_path, columns, subset)
    resolved = unique_request.merge(cached, on=["peptide_mut", "hla"], how="left")
    missing = resolved[resolved["stab_score"].isna()][["peptide_mut", "hla"]].drop_duplicates().reset_index(drop=True)

    fresh_frames = []
    groups: dict[tuple[str, int], list[str]] = {}
    for peptide, allele in zip(missing["peptide_mut"], missing["hla"], strict=True):
        groups.setdefault((allele, len(peptide)), []).append(peptide)
    for (allele, _length), allele_peptides in sorted(groups.items()):
        for peptide_batch in _batched(allele_peptides, batch_size):
            stdout = _run_tool(script, allele, peptide_batch, [])
            frame = _parse_stability_output(stdout, peptide_batch)
            frame["hla"] = allele
            frame["peptide_mut"] = peptide_batch
            fresh_frames.append(frame[columns])
    if fresh_frames:
        fresh = pd.concat(fresh_frames, ignore_index=True)
        _write_cache(cache_path, cached, fresh, subset=subset)
        cached = _load_cache(cache_path, columns, subset)

    merged = request.merge(cached, on=["peptide_mut", "hla"], how="left").sort_values("request_idx")
    return merged[columns].reset_index(drop=True)
