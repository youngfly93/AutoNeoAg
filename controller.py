#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.config import ensure_directories, load_settings
from autoneoag.runtime.codex_worker import run_codex_worker
from autoneoag.runtime.git_ops import (
    changed_files,
    checkout_branch,
    commit_all,
    current_branch,
    current_commit,
    ensure_branch,
    has_commits,
    reset_hard,
)
from autoneoag.runtime.results import append_result, ensure_results_file


def run_python(module: str, *args: str) -> str:
    completed = subprocess.run([sys.executable, module, *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout


def parse_metric(stdout: str, key: str = "val_score") -> float:
    for line in stdout.splitlines():
        if line.startswith(f"{key}:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"{key} not found in output")


def smoke(rounds: int) -> None:
    settings = load_settings(ROOT)
    ensure_directories(settings)
    ensure_results_file(settings.results_tsv)
    if not has_commits(ROOT):
        raise RuntimeError("Create an initial scaffold commit before running controller smoke.")
    prepare_stdout = run_python("prepare.py", "--mode", "smoke")
    log_dir = settings.artifacts_logs / "smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prepare.log").write_text(prepare_stdout)
    base_branch = current_branch(ROOT) or "main"
    branch_name = "autoneoag/smoke"
    ensure_branch(ROOT, branch_name)
    try:
        best_commit = current_commit(ROOT)
        best_score = float("-inf")
        best_checkpoint = ""
        summary_lines = [prepare_stdout.strip()]
        for round_id in range(1, rounds + 1):
            description = "baseline"
            proposal = {
                "hypothesis": "baseline",
                "expected_change": "baseline measurement",
                "risk": "none",
                "edit_scope": ["train.py"],
                "summary": "baseline",
            }
            if round_id > 1:
                proposal = run_codex_worker(settings, round_id, ROOT, "\n".join(summary_lines[-8:]))
                description = proposal["summary"]
                files = changed_files(ROOT)
                if not files:
                    append_result(settings.results_tsv, round_id, best_commit, best_score, "discard", "no-op proposal")
                    summary_lines.append(json.dumps({"round": round_id, "val_score": best_score, "status": "discard", **proposal}, ensure_ascii=False))
                    continue
                unexpected = [path for path in files if path != "train.py"]
                if unexpected:
                    raise RuntimeError(f"Codex worker changed unexpected files: {unexpected}")
                candidate_commit = commit_all(ROOT, f"smoke round {round_id}: {description}")
            else:
                candidate_commit = best_commit
            stdout = run_python("train.py", "--mode", "smoke", "--round-id", str(round_id))
            (log_dir / f"round_{round_id:02d}.log").write_text(stdout)
            val_score = parse_metric(stdout)
            status = "keep" if val_score > best_score + 1e-4 else "discard"
            if round_id == 1 or status == "keep":
                best_score = val_score
                best_commit = current_commit(ROOT) if round_id > 1 else best_commit
                best_checkpoint = str(settings.artifacts_runs / "smoke" / f"round_{round_id:02d}" / f"round_{round_id:02d}.pt")
            elif round_id > 1:
                reset_hard(ROOT, best_commit)
            append_result(settings.results_tsv, round_id, candidate_commit, val_score, status, description)
            summary_lines.append(json.dumps({"round": round_id, "val_score": val_score, "status": status, **proposal}, ensure_ascii=False))
        confirm_out = run_python("confirm.py", "--mode", "smoke", "--checkpoint", best_checkpoint)
        blind_out = run_python("blind_eval.py", "--mode", "smoke", "--checkpoint", best_checkpoint)
        settings.smoke_report.write_text(
            "\n".join(
                [
                    "# Smoke Report",
                    "",
                    f"Best commit: {best_commit}",
                    f"Best val_score: {best_score:.6f}",
                    "",
                    "## Confirm",
                    confirm_out.strip(),
                    "",
                    "## Blind",
                    blind_out.strip(),
                ]
            )
        )
    finally:
        checkout_branch(ROOT, base_branch)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.rounds)


if __name__ == "__main__":
    main()
