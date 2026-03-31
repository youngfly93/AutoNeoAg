from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoneoag.config import Settings


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    family: str
    smoke_dataset_resource: str
    context_resource: str
    reference_resource: str | None = None
    require_dtu: bool = True
    require_foreignness: bool = True
    confirm_split: str = "confirm"
    blind_split: str = "blind"
    dev_num_folds: int = 4
    full_enabled: bool = False


TASK_SPECS: dict[str, TaskSpec] = {
    "neoantigen": TaskSpec(
        task_id="neoantigen",
        title="Neoantigen Ranking",
        family="immunology",
        smoke_dataset_resource="smoke_seed_v3.tsv",
        context_resource="hla_pseudosequences.tsv",
        reference_resource="smoke_human_reference.fasta",
        require_dtu=True,
        require_foreignness=True,
    ),
    "hla_immunogenicity": TaskSpec(
        task_id="hla_immunogenicity",
        title="Human HLA-I Immunogenicity Ranking",
        family="immunology",
        smoke_dataset_resource="immunogenicity_smoke_seed_v1.tsv",
        context_resource="hla_pseudosequences.tsv",
        reference_resource="smoke_human_reference.fasta",
        require_dtu=True,
        require_foreignness=True,
    ),
    "variant_prioritization": TaskSpec(
        task_id="variant_prioritization",
        title="Variant Prioritization",
        family="generic_pairwise",
        smoke_dataset_resource="variant_smoke_seed_v1.tsv",
        context_resource="variant_context_pseudosequences.tsv",
        reference_resource=None,
        require_dtu=False,
        require_foreignness=False,
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task_id: {task_id}") from exc


def list_task_ids() -> list[str]:
    return sorted(TASK_SPECS)


def resource_path(settings: Settings, resource_name: str) -> Path:
    return settings.root / "src" / "autoneoag" / "resources" / resource_name


def processed_dataset_path(settings: Settings, task_id: str, mode: str) -> Path:
    return settings.data_processed / task_id / mode / "dataset.parquet"


def split_manifest_path(settings: Settings, task_id: str, mode: str) -> Path:
    return settings.data_processed / task_id / mode / "splits_grouped_v1.json"


def raw_snapshot_path(settings: Settings, task_id: str, mode: str) -> Path:
    return settings.data_raw / task_id / f"{mode}_raw.tsv"


def run_dir(settings: Settings, task_id: str, mode: str, strategy: str, run_id: int, round_id: int) -> Path:
    return settings.artifacts_runs / task_id / mode / strategy / f"run_{run_id:02d}" / f"round_{round_id:02d}"


def log_dir(settings: Settings, task_id: str, strategy: str, run_id: int) -> Path:
    return settings.artifacts_logs / task_id / strategy / f"run_{run_id:02d}"


def report_path(settings: Settings, task_id: str, strategy: str, run_id: int) -> Path:
    return log_dir(settings, task_id, strategy, run_id) / "report.md"
