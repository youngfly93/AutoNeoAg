#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

from autoneoag.config import ensure_directories, load_settings
from autoneoag.runtime.codex_worker import allowed_edit_scope, run_codex_worker
from autoneoag.runtime.git_ops import (
    changed_files,
    changed_line_count,
    checkout_branch,
    commit_paths,
    current_branch,
    current_commit,
    ensure_branch,
    has_commits,
    reset_hard,
)
from autoneoag.runtime.random_worker import run_random_worker
from autoneoag.runtime.results import append_result, reset_results_file
from autoneoag.tasks import get_task_spec, list_task_ids, log_dir, report_path, run_dir as task_run_dir


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


def proposal_for_strategy(settings, task_id: str, strategy: str, round_id: int, summary: str) -> dict[str, object]:
    if strategy == "random":
        return run_random_worker(round_id, ROOT)
    return run_codex_worker(
        settings,
        task_id=task_id,
        strategy=strategy,
        round_id=round_id,
        root=ROOT,
        summary=summary,
    )


def snapshot_files(root: Path, paths: set[str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for path in paths:
        file_path = root / path
        snapshot[path] = file_path.read_text() if file_path.exists() else None
    return snapshot


def run_experiment(task_id: str, mode: str, strategy: str, run_id: int, rounds: int, reset_results: bool = False) -> None:
    task = get_task_spec(task_id)
    settings = load_settings(ROOT)
    ensure_directories(settings)
    if not has_commits(ROOT):
        raise RuntimeError("Create an initial scaffold commit before running controller.")
    if reset_results:
        reset_results_file(settings.results_tsv)

    logs = log_dir(settings, task.task_id, strategy, run_id)
    logs.mkdir(parents=True, exist_ok=True)
    prepare_stdout = run_python("prepare.py", "--task", task.task_id, "--mode", mode)
    (logs / "prepare.log").write_text(prepare_stdout)

    base_branch = current_branch(ROOT) or "main"
    branch_name = f"autoneoag/{task.task_id}/{strategy}/run_{run_id:02d}"
    ensure_branch(ROOT, branch_name)
    try:
        best_commit = current_commit(ROOT)
        best_score = float("-inf")
        best_checkpoint = ""
        best_round = 0
        summary_lines = [prepare_stdout.strip()]
        allowed_files = set(allowed_edit_scope(strategy)) if strategy != "random" else {"train.py"}

        for round_id in range(1, rounds + 1):
            description = "baseline"
            proposal = {
                "hypothesis": "baseline",
                "expected_change": "baseline measurement",
                "risk": "none",
                "edit_scope": ["train.py"],
                "summary": "baseline",
            }
            candidate_commit = best_commit
            line_count = 0

            if round_id > 1:
                preexisting_dirty = set(changed_files(ROOT))
                before_snapshot = snapshot_files(ROOT, allowed_files)
                try:
                    proposal = proposal_for_strategy(settings, task.task_id, strategy, round_id, "\n".join(summary_lines[-10:]))
                except RuntimeError as exc:
                    failure_message = str(exc).strip().replace("\t", " ").replace("\n", " | ")
                    append_result(
                        settings.results_tsv,
                        task_id=task.task_id,
                        strategy=strategy,
                        run_id=run_id,
                        round_id=round_id,
                        commit=best_commit,
                        dev_score=best_score if best_score > float("-inf") else None,
                        confirm_score=None,
                        blind_score=None,
                        status="discard",
                        failure_type="worker_failed",
                        training_seconds=None,
                        lines_changed=0,
                        description=failure_message,
                    )
                    summary_lines.append(
                        json.dumps(
                            {"round": round_id, "status": "discard", "failure": "worker_failed", "message": failure_message},
                            ensure_ascii=False,
                        )
                    )
                    break
                description = proposal["summary"]
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
                        round_id=round_id,
                        commit=best_commit,
                        dev_score=best_score if best_score > float("-inf") else None,
                        confirm_score=None,
                        blind_score=None,
                        status="discard",
                        failure_type="no_op",
                        training_seconds=None,
                        lines_changed=0,
                        description="no-op proposal",
                    )
                    summary_lines.append(
                        json.dumps({"round": round_id, "status": "discard", "failure": "no_op", **proposal}, ensure_ascii=False)
                    )
                    continue
                if unexpected:
                    raise RuntimeError(f"{strategy} worker changed unexpected files: {unexpected}")
                line_count = changed_line_count(ROOT, changed_allowed)
                candidate_commit = commit_paths(
                    ROOT,
                    changed_allowed,
                    f"{task.task_id} {strategy} run {run_id} round {round_id}: {description}",
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
                reset_hard(ROOT, best_commit)
                append_result(
                    settings.results_tsv,
                    task_id=task.task_id,
                    strategy=strategy,
                    run_id=run_id,
                    round_id=round_id,
                    commit=candidate_commit,
                    dev_score=None,
                    confirm_score=None,
                    blind_score=None,
                    status="discard",
                    failure_type="train_failed",
                    training_seconds=None,
                    lines_changed=line_count,
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
            status = "keep" if val_score > best_score + 1e-4 else "discard"
            if round_id == 1 or status == "keep":
                best_score = val_score
                best_commit = current_commit(ROOT) if round_id > 1 else best_commit
                best_round = round_id
                best_checkpoint = str(
                    task_run_dir(settings, task.task_id, mode, strategy, run_id, round_id) / f"round_{round_id:02d}.pt"
                )
            elif round_id > 1:
                reset_hard(ROOT, best_commit)

            append_result(
                settings.results_tsv,
                task_id=task.task_id,
                strategy=strategy,
                run_id=run_id,
                round_id=round_id,
                commit=candidate_commit,
                dev_score=val_score,
                confirm_score=None,
                blind_score=None,
                status=status,
                failure_type=None,
                training_seconds=training_seconds,
                lines_changed=line_count,
                description=description,
            )
            summary_lines.append(json.dumps({"round": round_id, "val_score": val_score, "status": status, **proposal}, ensure_ascii=False))

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
    finally:
        checkout_branch(ROOT, base_branch)


def run_matrix(tasks: list[str], mode: str, strategies: list[str], runs: int, rounds: int, reset_results: bool) -> None:
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
                    reset_results=reset_results and first,
                )
                first = False


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", choices=list_task_ids(), required=True)
    run_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    run_parser.add_argument("--strategy", choices=["constrained", "random", "unconstrained"], default="constrained")
    run_parser.add_argument("--run-id", type=int, default=1)
    run_parser.add_argument("--rounds", type=int, default=10)
    run_parser.add_argument("--reset-results", action="store_true")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--tasks", nargs="+", choices=list_task_ids(), required=True)
    matrix_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    matrix_parser.add_argument("--strategies", nargs="+", choices=["constrained", "random", "unconstrained"], required=True)
    matrix_parser.add_argument("--runs", type=int, default=1)
    matrix_parser.add_argument("--rounds", type=int, default=10)
    matrix_parser.add_argument("--reset-results", action="store_true")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--rounds", type=int, default=10)

    args = parser.parse_args()
    if args.command == "run":
        run_experiment(args.task, args.mode, args.strategy, args.run_id, args.rounds, args.reset_results)
    elif args.command == "matrix":
        run_matrix(args.tasks, args.mode, args.strategies, args.runs, args.rounds, args.reset_results)
    elif args.command == "smoke":
        run_experiment("neoantigen", "smoke", "constrained", 1, args.rounds, True)


if __name__ == "__main__":
    main()
