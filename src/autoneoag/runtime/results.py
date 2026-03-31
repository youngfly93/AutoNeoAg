from __future__ import annotations

from pathlib import Path


HEADER = (
    "task_id\tstrategy\trun_id\tround_id\tcommit\tdev_score\tconfirm_score\tblind_score\tstatus\tfailure_type\t"
    "training_seconds\tlines_changed\tdescription\n"
)


def ensure_results_file(path: Path) -> None:
    if not path.exists():
        path.write_text(HEADER)


def reset_results_file(path: Path) -> None:
    path.write_text(HEADER)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def append_result(
    path: Path,
    *,
    task_id: str,
    strategy: str,
    run_id: int,
    round_id: int,
    commit: str,
    dev_score: float | None,
    confirm_score: float | None,
    blind_score: float | None,
    status: str,
    failure_type: str | None,
    training_seconds: float | None,
    lines_changed: int | None,
    description: str,
) -> None:
    ensure_results_file(path)
    with path.open("a") as handle:
        fields = [
            task_id,
            strategy,
            run_id,
            round_id,
            commit,
            dev_score,
            confirm_score,
            blind_score,
            status,
            failure_type,
            training_seconds,
            lines_changed,
            description,
        ]
        handle.write("\t".join(_format_value(field) for field in fields) + "\n")
