#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

from autoneoag.config import ensure_directories, load_settings
from autoneoag.runtime.codex_worker import allowed_edit_scope, run_codex_worker
from autoneoag.runtime.evidence import (
    create_evidence_bundle,
    evidence_bundle_path,
    load_evidence_bundle,
    update_baseline_record,
    validate_evidence_bundle,
)
from autoneoag.runtime.frontier import (
    build_frontier_state,
    canonical_family,
    infer_family_from_text,
    infer_subfamily_from_text,
    write_frontier_artifacts,
)
from autoneoag.runtime.git_ops import (
    changed_files,
    changed_line_count,
    checkout_branch,
    commit_paths,
    current_branch,
    current_commit,
    ensure_branch,
    has_commits,
    reset_hard_preserving,
)
from autoneoag.runtime.policy import (
    current_gate_stage,
    should_run_round_confirm,
    strict_confirm_gate_passes,
    strict_dev_gate_passes,
)
from autoneoag.runtime.random_worker import run_random_worker
from autoneoag.runtime.results import append_result, load_results, reset_results_file
from autoneoag.tasks import get_task_spec, list_task_ids, log_dir, report_path, run_dir as task_run_dir


RUN_POLICIES = ("fast-dev", "strict-confirm")


def run_python(module: str, *args: str) -> str:
    completed = subprocess.run([sys.executable, module, *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout


def format_subprocess_failure(exc: subprocess.CalledProcessError) -> str:
    parts = []
    if exc.stdout:
        parts.append(exc.stdout.strip())
    if exc.stderr:
        parts.append(exc.stderr.strip())
    return "\n\n".join(part for part in parts if part).strip()


def parse_metric(stdout: str, key: str = "val_score") -> float:
    for line in stdout.splitlines():
        if line.startswith(f"{key}:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"{key} not found in output")


def parse_json_output(stdout: str) -> dict[str, object]:
    return json.loads(stdout)


def proposal_for_strategy(
    settings,
    task_id: str,
    strategy: str,
    round_id: int,
    summary: str,
    frontier_state: dict[str, object] | None = None,
    frontier_hint: str = "",
) -> dict[str, object]:
    if strategy == "random":
        return run_random_worker(round_id, ROOT, frontier_state=frontier_state)
    return run_codex_worker(
        settings,
        task_id=task_id,
        strategy=strategy,
        round_id=round_id,
        root=ROOT,
        summary=summary,
        frontier_hint=frontier_hint,
        frontier_state=frontier_state,
    )


def snapshot_files(root: Path, paths: set[str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for path in paths:
        file_path = root / path
        snapshot[path] = file_path.read_text() if file_path.exists() else None
    return snapshot


def diff_text(root: Path, paths: list[str]) -> str:
    if not paths:
        return ""
    completed = subprocess.run(
        ["git", "diff", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def boolish(value: object) -> str:
    return "1" if bool(value) else "0"


def count_keeps(settings, task_id: str, strategy: str, run_id: int) -> int:
    rows = [
        row
        for row in load_results(settings.results_tsv)
        if row["task_id"] == task_id and row["strategy"] == strategy and int(row["run_id"]) == run_id and row["status"] == "keep"
    ]
    return len(rows)


def decision_reason(
    status: str,
    val_score: float | None,
    best_score_before: float,
    failure_type: str | None,
    strict_reject_reason: str | None = None,
) -> str:
    if failure_type:
        return failure_type
    if strict_reject_reason:
        return strict_reject_reason
    if status == "keep":
        return "new_best"
    if val_score is None:
        return "unknown"
    if best_score_before == float("-inf"):
        return "baseline"
    if best_score_before - val_score <= 0.01:
        return "near_tie_but_worse"
    return "clear_regression"


def infer_failure_mode(failure_type: str | None, family: str, description: str) -> str | None:
    if failure_type in {"train_failed", "worker_failed", "no_op"}:
        return failure_type
    text = description.lower()
    if family == "gating":
        return "family_repeat_regression"
    if "direct" in text and "context" in text:
        return "redundant_context_injection"
    if "unstable" in text:
        return "unstable_training"
    return None


def verify_evidence_bundle_or_raise(bundle: dict[str, object] | None) -> None:
    if not bundle:
        return
    mismatches = validate_evidence_bundle(bundle)
    if mismatches:
        raise RuntimeError("Evidence drift detected: " + "; ".join(mismatches))


@dataclass
class ResumeState:
    start_round: int
    best_commit: str
    best_score: float
    best_checkpoint: str
    best_round: int
    summary_lines: list[str]


def _result_float(value: str) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def load_resume_state(settings, task_id: str, mode: str, strategy: str, run_id: int) -> ResumeState | None:
    rows = [
        row
        for row in load_results(settings.results_tsv)
        if row["task_id"] == task_id and row["strategy"] == strategy and int(row["run_id"]) == run_id
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: int(row["round_id"]))
    keep_rows = [row for row in rows if row["status"] == "keep" and row["commit"]]
    if not keep_rows:
        return None
    best_row = keep_rows[-1]
    best_round = int(best_row["round_id"])
    best_checkpoint = str(task_run_dir(settings, task_id, mode, strategy, run_id, best_round) / f"round_{best_round:02d}.pt")
    summary_lines = [
        json.dumps(
            {
                "round": int(row["round_id"]),
                "val_score": _result_float(row["dev_score"]),
                "status": row["status"],
                "failure": row["failure_type"] or None,
                "summary": row["description"],
            },
            ensure_ascii=False,
        )
        for row in rows
    ]
    return ResumeState(
        start_round=int(rows[-1]["round_id"]) + 1,
        best_commit=best_row["commit"],
        best_score=float(best_row["dev_score"]),
        best_checkpoint=best_checkpoint,
        best_round=best_round,
        summary_lines=summary_lines,
    )


def run_experiment(
    task_id: str,
    mode: str,
    strategy: str,
    run_id: int,
    rounds: int,
    run_policy: str = "fast-dev",
    reset_results: bool = False,
    resume: bool = False,
) -> None:
    if run_policy not in RUN_POLICIES:
        raise RuntimeError(f"Unsupported run_policy: {run_policy}")
    task = get_task_spec(task_id)
    settings = load_settings(ROOT)
    ensure_directories(settings)
    results_path = settings.results_tsv
    if not has_commits(ROOT):
        raise RuntimeError("Create an initial scaffold commit before running controller.")
    if reset_results:
        reset_results_file(settings.results_tsv)

    logs = log_dir(settings, task.task_id, strategy, run_id)
    logs.mkdir(parents=True, exist_ok=True)
    prepare_stdout = run_python("prepare.py", "--task", task.task_id, "--mode", mode)
    (logs / "prepare.log").write_text(prepare_stdout)
    strict_confirm = run_policy == "strict-confirm"
    evidence_bundle = None
    if strict_confirm:
        evidence_bundle = load_evidence_bundle(logs)
        if evidence_bundle is None and not resume:
            evidence_bundle = create_evidence_bundle(
                settings=settings,
                task_id=task.task_id,
                mode=mode,
                strategy=strategy,
                run_policy=run_policy,
                run_id=run_id,
                logs_dir=logs,
            )
        if evidence_bundle is not None:
            verify_evidence_bundle_or_raise(evidence_bundle)
    bundle_id = evidence_bundle["bundle_id"] if evidence_bundle is not None else ""

    base_branch = current_branch(ROOT) or "main"
    branch_name = f"autoneoag/{task.task_id}/{strategy}/run_{run_id:02d}"
    ensure_branch(ROOT, branch_name)
    try:
        resume_state = load_resume_state(settings, task.task_id, mode, strategy, run_id) if resume else None
        best_commit = resume_state.best_commit if resume_state is not None else current_commit(ROOT)
        best_score = resume_state.best_score if resume_state is not None else float("-inf")
        best_checkpoint = resume_state.best_checkpoint if resume_state is not None else ""
        best_round = resume_state.best_round if resume_state is not None else 0
        summary_lines = [prepare_stdout.strip(), *(resume_state.summary_lines if resume_state is not None else [])]
        start_round = resume_state.start_round if resume_state is not None else 1
        allowed_files = set(allowed_edit_scope(strategy)) if strategy != "random" else {"train.py"}
        prior_rows = [
            row
            for row in load_results(settings.results_tsv)
            if row["task_id"] == task.task_id and row["strategy"] == strategy and int(row["run_id"]) == run_id
        ]
        prior_policies = {row.get("run_policy") or "fast-dev" for row in prior_rows}
        if prior_policies and prior_policies != {run_policy}:
            raise RuntimeError(
                f"Run {task.task_id}/{strategy}/run_{run_id} mixes run_policy values {sorted(prior_policies)} with requested {run_policy}."
            )
        if strict_confirm and prior_rows and evidence_bundle is None:
            raise RuntimeError(
                f"Strict-confirm resume for {task.task_id}/{strategy}/run_{run_id} requires an existing evidence bundle at "
                f"{evidence_bundle_path(logs)}."
            )
        if strict_confirm and evidence_bundle is None:
            evidence_bundle = create_evidence_bundle(
                settings=settings,
                task_id=task.task_id,
                mode=mode,
                strategy=strategy,
                run_policy=run_policy,
                run_id=run_id,
                logs_dir=logs,
            )
            bundle_id = evidence_bundle["bundle_id"]
        keep_count = sum(1 for row in prior_rows if row["status"] == "keep")
        confirmed_scores = [
            _result_float(row.get("confirm_round_score") or row.get("confirm_score"))
            for row in prior_rows
            if row.get("status") == "keep" and row.get("confirm_checked") in {"1", "true", "True"}
        ]
        best_confirm_score = max((score for score in confirmed_scores if score is not None), default=None)

        for round_id in range(start_round, rounds + 1):
            if strict_confirm:
                verify_evidence_bundle_or_raise(evidence_bundle)
            description = "baseline"
            proposal = {
                "hypothesis": "baseline",
                "expected_change": "baseline measurement",
                "risk": "none",
                "edit_scope": ["train.py"],
                "summary": "baseline",
                "worker_declared_family": "other",
                "worker_declared_subfamily": "baseline",
                "proposal_family": "other",
                "proposal_subfamily": "baseline",
                "parent_round_id": None,
                "search_mode": "exploit",
                "novelty_level": "low",
            }
            candidate_commit = best_commit
            line_count = 0
            frontier_state: dict[str, object] | None = None
            frontier_hint_text = ""
            parent_commit = best_commit
            worker_declared_family = "other"
            worker_declared_subfamily = "baseline"
            controller_family = "other"
            controller_subfamily = "baseline"
            proposal_family = "other"
            proposal_subfamily = "baseline"
            family_consensus = "controller_only"
            parent_round_id: int | None = None
            search_mode = "exploit"
            novelty_level = "low"

            if round_id > 1:
                frontier_state = build_frontier_state(
                    task_id=task.task_id,
                    strategy=strategy,
                    run_id=run_id,
                    current_round=round_id,
                    rows=load_results(settings.results_tsv),
                )
                _, hint_path, _ = write_frontier_artifacts(logs, frontier_state)
                frontier_hint_text = hint_path.read_text()
                champion = frontier_state.get("champion", {})
                parent_round_id = champion.get("round_id")
                parent_commit = champion.get("commit", best_commit) or best_commit
                search_mode = str(frontier_state.get("search_mode", "exploit"))
                preexisting_dirty = set(changed_files(ROOT))
                before_snapshot = snapshot_files(ROOT, allowed_files)
                try:
                    proposal = proposal_for_strategy(
                        settings,
                        task.task_id,
                        strategy,
                        round_id,
                        "\n".join(summary_lines[-10:]),
                        frontier_state=frontier_state,
                        frontier_hint=frontier_hint_text,
                    )
                except RuntimeError as exc:
                    failure_message = str(exc).strip().replace("\t", " ").replace("\n", " | ")
                    append_result(
                        settings.results_tsv,
                        task_id=task.task_id,
                        strategy=strategy,
                        run_id=run_id,
                        run_policy=run_policy,
                        evidence_bundle_id=bundle_id,
                        baseline_bundle_id=bundle_id,
                        round_id=round_id,
                        commit=best_commit,
                        gate_stage=current_gate_stage(
                            failure_type="worker_failed",
                            confirm_gate_required=False,
                            confirm_checked=False,
                            dev_passed_gate=False,
                        ),
                        dev_passed_gate=False,
                        confirm_gate_required=False,
                        confirm_gate_passed=False,
                        strict_keep_eligible=False,
                        strict_reject_reason="worker_failed",
                        dev_score=best_score if best_score > float("-inf") else None,
                        confirm_score=None,
                        blind_score=None,
                        status="discard",
                        decision_reason="worker_failed",
                        failure_type="worker_failed",
                        failure_mode="worker_failed",
                        training_seconds=None,
                        lines_changed=0,
                        worker_declared_family="",
                        worker_declared_subfamily="",
                        controller_inferred_family="",
                        controller_inferred_subfamily="",
                        proposal_family="uncertain",
                        proposal_subfamily="uncertain",
                        family_consensus="uncertain",
                        parent_round_id=parent_round_id,
                        parent_commit=parent_commit,
                        search_mode=search_mode,
                        novelty_level="medium",
                        confirm_checked=False,
                        confirm_round_score=None,
                        confirm_survival=None,
                        delta_vs_best=None,
                        delta_vs_parent=None,
                        description=failure_message,
                    )
                    summary_lines.append(
                        json.dumps(
                            {
                                "round": round_id,
                                "status": "discard",
                                "failure": "worker_failed",
                                "message": failure_message,
                                "search_mode": search_mode,
                            },
                            ensure_ascii=False,
                        )
                    )
                    break
                description = proposal["summary"]
                worker_declared_family = str(proposal.get("worker_declared_family") or proposal.get("proposal_family") or "other")
                worker_declared_subfamily = str(proposal.get("worker_declared_subfamily") or proposal.get("proposal_subfamily") or worker_declared_family)
                parent_round_id = int(proposal.get("parent_round_id")) if proposal.get("parent_round_id") not in {None, ""} else parent_round_id
                search_mode = str(proposal.get("search_mode") or search_mode)
                novelty_level = str(proposal.get("novelty_level") or novelty_level)
                current_dirty = set(changed_files(ROOT))
                changed_allowed = sorted(
                    path
                    for path in allowed_files
                    if snapshot_files(ROOT, {path})[path] != before_snapshot[path]
                )
                unexpected = [path for path in sorted(current_dirty - preexisting_dirty) if path not in allowed_files]
                if not changed_allowed:
                    append_result(
                        settings.results_tsv,
                        task_id=task.task_id,
                        strategy=strategy,
                        run_id=run_id,
                        run_policy=run_policy,
                        evidence_bundle_id=bundle_id,
                        baseline_bundle_id=bundle_id,
                        round_id=round_id,
                        commit=best_commit,
                        gate_stage=current_gate_stage(
                            failure_type="no_op",
                            confirm_gate_required=False,
                            confirm_checked=False,
                            dev_passed_gate=False,
                        ),
                        dev_passed_gate=False,
                        confirm_gate_required=False,
                        confirm_gate_passed=False,
                        strict_keep_eligible=False,
                        strict_reject_reason="no_op",
                        dev_score=best_score if best_score > float("-inf") else None,
                        confirm_score=None,
                        blind_score=None,
                        status="discard",
                        decision_reason="no_op",
                        failure_type="no_op",
                        failure_mode="no_op",
                        training_seconds=None,
                        lines_changed=0,
                        worker_declared_family=worker_declared_family,
                        worker_declared_subfamily=worker_declared_subfamily,
                        controller_inferred_family="uncertain",
                        controller_inferred_subfamily="uncertain",
                        proposal_family=worker_declared_family,
                        proposal_subfamily=worker_declared_subfamily,
                        family_consensus="worker_only",
                        parent_round_id=parent_round_id,
                        parent_commit=parent_commit,
                        search_mode=search_mode,
                        novelty_level=novelty_level,
                        confirm_checked=False,
                        confirm_round_score=None,
                        confirm_survival=None,
                        delta_vs_best=None,
                        delta_vs_parent=None,
                        description="no-op proposal",
                    )
                    summary_lines.append(
                        json.dumps(
                            {
                                "round": round_id,
                                "status": "discard",
                                "failure": "no_op",
                                "proposal_family": worker_declared_family,
                                "search_mode": search_mode,
                                **proposal,
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                if unexpected:
                    raise RuntimeError(f"{strategy} worker changed unexpected files: {unexpected}")
                line_count = changed_line_count(ROOT, changed_allowed)
                diff = diff_text(ROOT, changed_allowed)
                controller_family = infer_family_from_text(
                    description,
                    str(proposal.get("hypothesis", "")),
                    str(proposal.get("expected_change", "")),
                    diff,
                )
                controller_subfamily = infer_subfamily_from_text(
                    controller_family,
                    description,
                    str(proposal.get("hypothesis", "")),
                    str(proposal.get("expected_change", "")),
                    diff,
                )
                proposal_family, family_consensus = canonical_family(worker_declared_family, controller_family)
                proposal_subfamily = (
                    worker_declared_subfamily
                    if family_consensus == "agreed"
                    else controller_subfamily or worker_declared_subfamily or proposal_family
                )
                candidate_commit = commit_paths(
                    ROOT,
                    changed_allowed,
                    f"{task.task_id} {strategy} run {run_id} round {round_id}: {description}",
                )

            candidate_checkpoint = str(
                task_run_dir(settings, task.task_id, mode, strategy, run_id, round_id) / f"round_{round_id:02d}.pt"
            )
            try:
                stdout = run_python(
                    "train.py",
                    "--task",
                    task.task_id,
                    "--mode",
                    mode,
                    "--strategy",
                    strategy,
                    "--run-id",
                    str(run_id),
                    "--round-id",
                    str(round_id),
                )
            except subprocess.CalledProcessError as exc:
                failure_log = format_subprocess_failure(exc) or f"train.py failed with exit code {exc.returncode}"
                (logs / f"round_{round_id:03d}.log").write_text(failure_log)
                if round_id == 1:
                    raise RuntimeError(f"Baseline training failed for round {round_id}\n{failure_log}") from exc
                reset_hard_preserving(ROOT, best_commit, [results_path])
                append_result(
                    settings.results_tsv,
                    task_id=task.task_id,
                    strategy=strategy,
                    run_id=run_id,
                    run_policy=run_policy,
                    evidence_bundle_id=bundle_id,
                    baseline_bundle_id=bundle_id,
                    round_id=round_id,
                    commit=candidate_commit,
                    gate_stage=current_gate_stage(
                        failure_type="train_failed",
                        confirm_gate_required=False,
                        confirm_checked=False,
                        dev_passed_gate=False,
                    ),
                    dev_passed_gate=False,
                    confirm_gate_required=False,
                    confirm_gate_passed=False,
                    strict_keep_eligible=False,
                    strict_reject_reason="train_failed",
                    dev_score=None,
                    confirm_score=None,
                    blind_score=None,
                    status="discard",
                    decision_reason="train_failed",
                    failure_type="train_failed",
                    failure_mode="train_failed",
                    training_seconds=None,
                    lines_changed=line_count,
                    worker_declared_family=worker_declared_family,
                    worker_declared_subfamily=worker_declared_subfamily,
                    controller_inferred_family=controller_family,
                    controller_inferred_subfamily=controller_subfamily,
                    proposal_family=proposal_family,
                    proposal_subfamily=proposal_subfamily,
                    family_consensus=family_consensus,
                    parent_round_id=parent_round_id,
                    parent_commit=parent_commit,
                    search_mode=search_mode,
                    confirm_checked=False,
                    confirm_round_score=None,
                    confirm_survival=None,
                    delta_vs_best=None,
                    delta_vs_parent=None,
                    novelty_level=novelty_level,
                    description=f"{description} [train failed]",
                )
                summary_lines.append(
                    json.dumps(
                        {"round": round_id, "status": "discard", "failure": "train_failed", **proposal},
                        ensure_ascii=False,
                    )
                )
                continue

            (logs / f"round_{round_id:03d}.log").write_text(stdout)
            val_score = parse_metric(stdout)
            training_seconds = parse_metric(stdout, "training_seconds")
            best_score_before = best_score
            delta_vs_best = 0.0 if best_score_before == float("-inf") else val_score - best_score_before
            parent_score = None
            if parent_round_id is not None:
                parent_rows = [
                    row
                    for row in load_results(settings.results_tsv)
                    if row["task_id"] == task.task_id
                    and row["strategy"] == strategy
                    and int(row["run_id"]) == run_id
                    and int(row["round_id"]) == parent_round_id
                ]
                if parent_rows:
                    parent_score = _result_float(parent_rows[-1].get("dev_score"))
            delta_vs_parent = (val_score - parent_score) if parent_score is not None else None
            status = "discard"
            dev_passed_gate = False
            confirm_gate_required = False
            confirm_gate_passed = False
            strict_keep_eligible = False
            strict_reject_reason = None
            confirm_round_score = None
            confirm_survival = None
            confirm_checked = False

            if strict_confirm:
                dev_passed_gate = strict_dev_gate_passes(val_score, best_score_before)
                confirm_gate_required = dev_passed_gate
                confirm_checked = confirm_gate_required
                if confirm_checked:
                    confirm_metrics = parse_json_output(
                        run_python("confirm.py", "--task", task.task_id, "--mode", mode, "--checkpoint", candidate_checkpoint)
                    )
                    confirm_round_score = float(confirm_metrics["val_score"])
                    confirm_gate_passed = strict_confirm_gate_passes(confirm_round_score, best_confirm_score)
                    confirm_survival = confirm_gate_passed
                strict_keep_eligible = dev_passed_gate and (not confirm_gate_required or confirm_gate_passed)
                status = "keep" if strict_keep_eligible else "discard"
                if not dev_passed_gate:
                    strict_reject_reason = "dev_not_improved"
                elif confirm_gate_required and not confirm_gate_passed:
                    strict_reject_reason = "confirm_failed_gate"
                if status == "keep":
                    keep_count += 1
                    best_score = val_score
                    best_commit = current_commit(ROOT) if round_id > 1 else best_commit
                    best_round = round_id
                    best_checkpoint = candidate_checkpoint
                    if confirm_round_score is not None:
                        best_confirm_score = confirm_round_score
                elif round_id > 1:
                    reset_hard_preserving(ROOT, best_commit, [results_path])
            else:
                status = "keep" if val_score > best_score + 1e-4 else "discard"
                dev_passed_gate = status == "keep" or best_score_before == float("-inf")
                if status == "keep":
                    keep_count += 1
                confirm_checked = should_run_round_confirm(run_policy, status, keep_count, dev_passed_gate)
                confirm_gate_required = confirm_checked
                if round_id == 1 or status == "keep":
                    best_score = val_score
                    best_commit = current_commit(ROOT) if round_id > 1 else best_commit
                    best_round = round_id
                    best_checkpoint = candidate_checkpoint
                elif round_id > 1:
                    reset_hard_preserving(ROOT, best_commit, [results_path])
                if confirm_checked:
                    confirm_metrics = parse_json_output(
                        run_python("confirm.py", "--task", task.task_id, "--mode", mode, "--checkpoint", best_checkpoint)
                    )
                    confirm_round_score = float(confirm_metrics["val_score"])
                    confirm_survival = best_confirm_score is None or confirm_round_score >= best_confirm_score - 1e-4
                    confirm_gate_passed = bool(confirm_survival)
                    if confirm_survival:
                        best_confirm_score = confirm_round_score
                strict_keep_eligible = status == "keep"

            gate_stage = current_gate_stage(
                failure_type=None,
                confirm_gate_required=confirm_gate_required,
                confirm_checked=confirm_checked,
                dev_passed_gate=dev_passed_gate,
            )
            if strict_confirm and round_id == 1 and status == "keep":
                baseline_metrics_path = logs / "baseline_metrics.json"
                baseline_metrics_path.write_text(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "strategy": strategy,
                            "run_id": run_id,
                            "run_policy": run_policy,
                            "round_id": round_id,
                            "commit": best_commit,
                            "dev_score": val_score,
                            "confirm_round_score": confirm_round_score,
                            "checkpoint_path": best_checkpoint,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                evidence_bundle = update_baseline_record(
                    logs,
                    evidence_bundle,
                    round_id=round_id,
                    commit=best_commit,
                    checkpoint_path=best_checkpoint,
                    metrics_path=str(baseline_metrics_path),
                    dev_score=val_score,
                    confirm_score=confirm_round_score,
                )

            append_result(
                settings.results_tsv,
                task_id=task.task_id,
                strategy=strategy,
                run_id=run_id,
                run_policy=run_policy,
                evidence_bundle_id=bundle_id,
                baseline_bundle_id=bundle_id,
                round_id=round_id,
                commit=candidate_commit,
                gate_stage=gate_stage,
                dev_passed_gate=dev_passed_gate,
                confirm_gate_required=confirm_gate_required,
                confirm_gate_passed=confirm_gate_passed,
                strict_keep_eligible=strict_keep_eligible,
                strict_reject_reason=strict_reject_reason,
                dev_score=val_score,
                confirm_score=confirm_round_score,
                blind_score=None,
                status=status,
                decision_reason=decision_reason(status, val_score, best_score_before, None, strict_reject_reason),
                failure_type=None,
                training_seconds=training_seconds,
                lines_changed=line_count,
                worker_declared_family=worker_declared_family,
                worker_declared_subfamily=worker_declared_subfamily,
                controller_inferred_family=controller_family,
                controller_inferred_subfamily=controller_subfamily,
                proposal_family=proposal_family,
                proposal_subfamily=proposal_subfamily,
                family_consensus=family_consensus,
                parent_round_id=parent_round_id,
                parent_commit=parent_commit,
                search_mode=search_mode,
                confirm_checked=confirm_checked,
                confirm_round_score=confirm_round_score,
                confirm_survival=confirm_survival,
                delta_vs_best=delta_vs_best,
                delta_vs_parent=delta_vs_parent,
                novelty_level=novelty_level,
                description=description,
                failure_mode=(
                    None
                    if status == "keep"
                    else infer_failure_mode(None, proposal_family, description)
                ),
            )
            summary_lines.append(
                json.dumps(
                    {
                        "round": round_id,
                        "val_score": val_score,
                        "status": status,
                        "proposal_family": proposal_family,
                        "search_mode": search_mode,
                        "delta_vs_best": delta_vs_best,
                        "confirm_round_score": confirm_round_score,
                        **proposal,
                    },
                    ensure_ascii=False,
                )
            )
            latest_state = build_frontier_state(
                task_id=task.task_id,
                strategy=strategy,
                run_id=run_id,
                current_round=round_id + 1,
                rows=load_results(settings.results_tsv),
            )
            write_frontier_artifacts(logs, latest_state)

        if strict_confirm:
            verify_evidence_bundle_or_raise(evidence_bundle)
        confirm_metrics = parse_json_output(
            run_python("confirm.py", "--task", task.task_id, "--mode", mode, "--checkpoint", best_checkpoint)
        )
        blind_metrics = parse_json_output(
            run_python("blind_eval.py", "--task", task.task_id, "--mode", mode, "--checkpoint", best_checkpoint)
        )
        report_path(settings, task.task_id, strategy, run_id).write_text(
            "\n".join(
                [
                    f"# {task.title} Report",
                    "",
                    f"task_id: {task.task_id}",
                    f"strategy: {strategy}",
                    f"run_id: {run_id}",
                    f"run_policy: {run_policy}",
                    f"evidence_bundle_path: {evidence_bundle_path(logs) if strict_confirm else ''}",
                    f"best_round: {best_round}",
                    f"best_commit: {best_commit}",
                    f"best_val_score: {best_score:.6f}",
                    "",
                    "## Confirm",
                    json.dumps(confirm_metrics, indent=2, sort_keys=True),
                    "",
                    "## Blind",
                    json.dumps(blind_metrics, indent=2, sort_keys=True),
                ]
            )
        )
        final_state = build_frontier_state(
            task_id=task.task_id,
            strategy=strategy,
            run_id=run_id,
            current_round=rounds + 1,
            rows=load_results(settings.results_tsv),
        )
        write_frontier_artifacts(logs, final_state)
    finally:
        checkout_branch(ROOT, base_branch)


def run_matrix(
    tasks: list[str],
    mode: str,
    strategies: list[str],
    runs: int,
    rounds: int,
    run_policy: str,
    reset_results: bool,
    resume: bool = False,
) -> None:
    first = True
    for task_id in tasks:
        for strategy in strategies:
            for run_id in range(1, runs + 1):
                run_experiment(
                    task_id=task_id,
                    mode=mode,
                    strategy=strategy,
                    run_id=run_id,
                    rounds=rounds,
                    run_policy=run_policy,
                    reset_results=reset_results and first,
                    resume=resume,
                )
                first = False


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", choices=list_task_ids(), required=True)
    run_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    run_parser.add_argument("--strategy", choices=["constrained", "random", "unconstrained"], default="constrained")
    run_parser.add_argument("--run-policy", choices=list(RUN_POLICIES), default="fast-dev")
    run_parser.add_argument("--run-id", type=int, default=1)
    run_parser.add_argument("--rounds", type=int, default=10)
    run_parser.add_argument("--reset-results", action="store_true")
    run_parser.add_argument("--resume", action="store_true")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--tasks", nargs="+", choices=list_task_ids(), required=True)
    matrix_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    matrix_parser.add_argument("--strategies", nargs="+", choices=["constrained", "random", "unconstrained"], required=True)
    matrix_parser.add_argument("--run-policy", choices=list(RUN_POLICIES), default="fast-dev")
    matrix_parser.add_argument("--runs", type=int, default=1)
    matrix_parser.add_argument("--rounds", type=int, default=10)
    matrix_parser.add_argument("--reset-results", action="store_true")
    matrix_parser.add_argument("--resume", action="store_true")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--rounds", type=int, default=10)

    args = parser.parse_args()
    if args.command == "run":
        run_experiment(args.task, args.mode, args.strategy, args.run_id, args.rounds, args.run_policy, args.reset_results, args.resume)
    elif args.command == "matrix":
        run_matrix(args.tasks, args.mode, args.strategies, args.runs, args.rounds, args.run_policy, args.reset_results, args.resume)
    elif args.command == "smoke":
        run_experiment("neoantigen", "smoke", "constrained", 1, args.rounds, "fast-dev", True)


if __name__ == "__main__":
    main()
