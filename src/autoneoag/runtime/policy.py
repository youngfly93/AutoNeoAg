from __future__ import annotations


def should_run_round_confirm(run_policy: str, status: str, keep_count_after_round: int, dev_passed_gate: bool) -> bool:
    if run_policy == "strict-confirm":
        return dev_passed_gate
    return status == "keep" and (keep_count_after_round == 1 or keep_count_after_round % 3 == 0)


def strict_dev_gate_passes(val_score: float, best_score_before: float) -> bool:
    return best_score_before == float("-inf") or val_score > best_score_before + 1e-4


def strict_confirm_gate_passes(confirm_round_score: float, best_confirm_score: float | None) -> bool:
    return best_confirm_score is None or confirm_round_score >= best_confirm_score - 1e-4


def current_gate_stage(
    *,
    failure_type: str | None,
    confirm_gate_required: bool,
    confirm_checked: bool,
    dev_passed_gate: bool,
) -> str:
    if failure_type == "worker_failed":
        return "worker"
    if failure_type == "train_failed":
        return "train"
    if confirm_checked or confirm_gate_required:
        return "confirm"
    if dev_passed_gate:
        return "dev"
    return "proposal"
