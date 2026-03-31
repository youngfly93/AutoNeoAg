from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    artifacts_runs: Path
    artifacts_cache: Path
    artifacts_logs: Path
    results_tsv: Path
    smoke_report: Path
    split_manifest: Path
    conda_env_name: str
    conda_env_path: Path
    netmhcpan_tar: Path
    netmhcpan_home: Path
    netmhcstabpan_tar: Path
    netmhcstabpan_home: Path
    reasoning_effort: str
    smoke_dataset_resource: str
    smoke_reference_resource: str
    smoke_hla_resource: str
    smoke_confirm_split: str
    smoke_blind_split: str
    smoke_val_fold: int
    smoke_dev_num_folds: int
    smoke_rounds: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(root: Path | None = None) -> Settings:
    root = (root or repo_root()).resolve()
    raw = tomllib.loads((root / "project.toml").read_text())
    paths = raw["paths"]
    env_cfg = raw["environment"]
    dtu = raw["dtu"]
    smoke = raw["smoke"]
    project = raw["project"]
    default_netmhcpan = (root / dtu["netmhcpan_home"]).resolve()
    default_netmhcstabpan = (root / dtu["netmhcstabpan_home"]).resolve()
    return Settings(
        root=root,
        data_raw=root / paths["data_raw"],
        data_interim=root / paths["data_interim"],
        data_processed=root / paths["data_processed"],
        artifacts_runs=root / paths["artifacts_runs"],
        artifacts_cache=root / paths["artifacts_cache"],
        artifacts_logs=root / paths["artifacts_logs"],
        results_tsv=root / paths["results_tsv"],
        smoke_report=root / paths["smoke_report"],
        split_manifest=root / paths["split_manifest"],
        conda_env_name=env_cfg["conda_env_name"],
        conda_env_path=root / env_cfg["conda_env_path"],
        netmhcpan_tar=root / dtu["netmhcpan_tar"],
        netmhcpan_home=Path(os.environ.get("NETMHCPAN_HOME", str(default_netmhcpan))).resolve(),
        netmhcstabpan_tar=root / dtu["netmhcstabpan_tar"],
        netmhcstabpan_home=Path(os.environ.get("NETMHCSTABPAN_HOME", str(default_netmhcstabpan))).resolve(),
        reasoning_effort=project["reasoning_effort"],
        smoke_dataset_resource=smoke["dataset_resource"],
        smoke_reference_resource=smoke["human_reference_resource"],
        smoke_hla_resource=smoke["hla_pseudoseq_resource"],
        smoke_confirm_split=smoke["default_confirm_split"],
        smoke_blind_split=smoke["default_blind_split"],
        smoke_val_fold=int(smoke["default_val_fold"]),
        smoke_dev_num_folds=int(smoke["dev_num_folds"]),
        smoke_rounds=int(project["smoke_rounds"]),
    )


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.data_raw,
        settings.data_interim,
        settings.data_processed,
        settings.artifacts_runs,
        settings.artifacts_cache,
        settings.artifacts_logs,
    ):
        path.mkdir(parents=True, exist_ok=True)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value
