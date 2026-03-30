from __future__ import annotations

from pathlib import Path


HEADER = "round\tcommit\tval_score\tstatus\tdescription\n"


def ensure_results_file(path: Path) -> None:
    if not path.exists():
        path.write_text(HEADER)


def append_result(path: Path, round_id: int, commit: str, val_score: float, status: str, description: str) -> None:
    ensure_results_file(path)
    with path.open("a") as handle:
        handle.write(f"{round_id}\t{commit}\t{val_score:.6f}\t{status}\t{description}\n")

