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
    groups: dict[tuple[str, int], list[str]] = {}
    for peptide, allele in zip(peptides, alleles, strict=True):
        groups.setdefault((allele, len(peptide)), []).append(peptide)
    for (allele, _length), allele_peptides in sorted(groups.items()):
        stdout = _run_tool(script, allele, allele_peptides, [])
        frame = _parse_stability_output(stdout, allele_peptides)
        frame["hla"] = allele
        frame["peptide_mut"] = allele_peptides
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    return merged[["peptide_mut", "hla", "stab_score", "stab_rank"]]
