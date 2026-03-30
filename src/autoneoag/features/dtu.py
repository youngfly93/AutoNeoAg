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


def _run_tool(script: Path, allele: str, peptides: list[str], extra_args: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = Path(tmpdir) / "peptides.faa"
        with fasta.open("w") as handle:
            for idx, peptide in enumerate(peptides):
                handle.write(f">p{idx}\n{peptide}\n")
        cmd = [str(script), "-p", str(fasta), "-a", allele, *extra_args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _parse_affinity_output(stdout: str, peptides: list[str]) -> pd.DataFrame:
    rows = {peptide: {"ba_score": None, "el_score": None, "ba_rank": None, "el_rank": None} for peptide in peptides}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if all(peptide not in line for peptide in peptides):
            continue
        tokens = re.split(r"\s+", line)
        peptide = next((item for item in peptides if item in line), None)
        if peptide is None or len(tokens) < 6:
            continue
        floats = []
        for token in tokens:
            try:
                floats.append(float(token))
            except ValueError:
                continue
        if len(floats) >= 4:
            rows[peptide] = {
                "ba_score": floats[-4],
                "el_score": floats[-3],
                "ba_rank": floats[-2],
                "el_rank": floats[-1],
            }
    return pd.DataFrame([rows[p] for p in peptides])


def _parse_stability_output(stdout: str, peptides: list[str]) -> pd.DataFrame:
    rows = {peptide: {"stab_score": None, "stab_rank": None} for peptide in peptides}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if all(peptide not in line for peptide in peptides):
            continue
        tokens = re.split(r"\s+", line)
        peptide = next((item for item in peptides if item in line), None)
        if peptide is None:
            continue
        floats = []
        for token in tokens:
            try:
                floats.append(float(token))
            except ValueError:
                continue
        if len(floats) >= 2:
            rows[peptide] = {"stab_score": floats[-2], "stab_rank": floats[-1]}
    return pd.DataFrame([rows[p] for p in peptides])


def netmhcpan_predict(settings: Settings, peptides: list[str], alleles: list[str]) -> pd.DataFrame:
    script = _require_executable(settings.netmhcpan_home, "netMHCpan")
    frames = []
    for allele in sorted(set(alleles)):
        allele_peptides = [peptide for peptide, current in zip(peptides, alleles, strict=True) if current == allele]
        stdout = _run_tool(script, allele, allele_peptides, ["-BA"])
        frame = _parse_affinity_output(stdout, allele_peptides)
        frame["hla"] = allele
        frame["peptide_mut"] = allele_peptides
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    return merged[["peptide_mut", "hla", "ba_score", "el_score", "ba_rank", "el_rank"]]


def netmhcstabpan_predict(settings: Settings, peptides: list[str], alleles: list[str]) -> pd.DataFrame:
    script = _require_executable(settings.netmhcstabpan_home, "netMHCstabpan")
    frames = []
    for allele in sorted(set(alleles)):
        allele_peptides = [peptide for peptide, current in zip(peptides, alleles, strict=True) if current == allele]
        stdout = _run_tool(script, allele, allele_peptides, [])
        frame = _parse_stability_output(stdout, allele_peptides)
        frame["hla"] = allele
        frame["peptide_mut"] = allele_peptides
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    return merged[["peptide_mut", "hla", "stab_score", "stab_rank"]]

