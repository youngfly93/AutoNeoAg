from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "task_id",
    "strategy",
    "run_id",
    "run_policy",
    "evidence_bundle_id",
    "baseline_bundle_id",
    "round_id",
    "commit",
    "worker_declared_family",
    "worker_declared_subfamily",
    "controller_inferred_family",
    "controller_inferred_subfamily",
    "proposal_family",
    "proposal_subfamily",
    "family_consensus",
    "parent_round_id",
    "parent_commit",
    "search_mode",
    "gate_stage",
    "dev_passed_gate",
    "confirm_gate_required",
    "confirm_gate_passed",
    "strict_keep_eligible",
    "strict_reject_reason",
    "dev_score",
    "confirm_score",
    "blind_score",
    "confirm_checked",
    "confirm_round_score",
    "confirm_survival",
    "delta_vs_best",
    "delta_vs_parent",
    "status",
    "decision_reason",
    "failure_type",
    "failure_mode",
    "training_seconds",
    "lines_changed",
    "novelty_level",
    "description",
]

LEGACY_COLUMNS = [
    "task_id",
    "strategy",
    "run_id",
    "round_id",
    "commit",
    "dev_score",
    "confirm_score",
    "blind_score",
    "status",
    "failure_type",
    "training_seconds",
    "lines_changed",
    "description",
]

HEADER = "\t".join(RESULT_COLUMNS) + "\n"


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {column: row.get(column, "") for column in RESULT_COLUMNS}
    normalized["task_id"] = normalized["task_id"] or row.get("task_id", "")
    normalized["strategy"] = normalized["strategy"] or row.get("strategy", "")
    normalized["run_id"] = normalized["run_id"] or row.get("run_id", "")
    normalized["run_policy"] = normalized["run_policy"] or row.get("run_policy", "fast-dev")
    normalized["evidence_bundle_id"] = normalized["evidence_bundle_id"] or row.get("evidence_bundle_id", "")
    normalized["baseline_bundle_id"] = normalized["baseline_bundle_id"] or row.get("baseline_bundle_id", "")
    normalized["round_id"] = normalized["round_id"] or row.get("round_id", "")
    normalized["commit"] = normalized["commit"] or row.get("commit", "")
    normalized["dev_score"] = normalized["dev_score"] or row.get("dev_score", "")
    normalized["confirm_score"] = normalized["confirm_score"] or row.get("confirm_score", "")
    normalized["blind_score"] = normalized["blind_score"] or row.get("blind_score", "")
    normalized["status"] = normalized["status"] or row.get("status", "")
    normalized["failure_type"] = normalized["failure_type"] or row.get("failure_type", "")
    normalized["training_seconds"] = normalized["training_seconds"] or row.get("training_seconds", "")
    normalized["lines_changed"] = normalized["lines_changed"] or row.get("lines_changed", "")
    normalized["description"] = normalized["description"] or row.get("description", "")
    normalized["gate_stage"] = normalized["gate_stage"] or row.get("gate_stage", "")
    normalized["dev_passed_gate"] = normalized["dev_passed_gate"] or row.get("dev_passed_gate", "")
    normalized["confirm_gate_required"] = normalized["confirm_gate_required"] or row.get("confirm_gate_required", "")
    normalized["confirm_gate_passed"] = normalized["confirm_gate_passed"] or row.get("confirm_gate_passed", "")
    normalized["strict_keep_eligible"] = normalized["strict_keep_eligible"] or row.get("strict_keep_eligible", "")
    normalized["strict_reject_reason"] = normalized["strict_reject_reason"] or row.get("strict_reject_reason", "")
    if not normalized["decision_reason"] and normalized["failure_type"]:
        normalized["decision_reason"] = normalized["failure_type"]
    if not normalized["confirm_checked"]:
        normalized["confirm_checked"] = "0"
    return normalized


def migrate_results_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames == RESULT_COLUMNS:
            return
        rows = [_normalize_row(dict(row)) for row in reader]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_results_file(path: Path) -> None:
    if not path.exists():
        path.write_text(HEADER)
        return
    migrate_results_file(path)


def reset_results_file(path: Path) -> None:
    path.write_text(HEADER)


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    migrate_results_file(path)
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [_normalize_row(dict(row)) for row in reader]


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def append_result(path: Path, **values: Any) -> None:
    ensure_results_file(path)
    row = {column: _format_value(values.get(column)) for column in RESULT_COLUMNS}
    row["run_policy"] = row["run_policy"] or "fast-dev"
    row["confirm_checked"] = row["confirm_checked"] or "0"
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writerow(row)
