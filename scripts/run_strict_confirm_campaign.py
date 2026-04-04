#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_one(
    *,
    task_id: str,
    strategy: str,
    run_id: int,
    mode: str,
    rounds: int,
    run_policy: str,
    resume: bool,
    reset_results: bool,
) -> None:
    cmd = [
        sys.executable,
        "controller.py",
        "run",
        "--task",
        task_id,
        "--mode",
        mode,
        "--strategy",
        strategy,
        "--run-policy",
        run_policy,
        "--run-id",
        str(run_id),
        "--rounds",
        str(rounds),
    ]
    if resume:
        cmd.append("--resume")
    if reset_results:
        cmd.append("--reset-results")
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--strategies", nargs="+", default=["constrained", "random"])
    parser.add_argument("--run-ids", nargs="+", type=int, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--run-policy", choices=["fast-dev", "strict-confirm"], default="strict-confirm")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-results-first", action="store_true")
    args = parser.parse_args()

    reset_next = args.reset_results_first
    for task_id in args.tasks:
        for strategy in args.strategies:
            for run_id in args.run_ids:
                run_one(
                    task_id=task_id,
                    strategy=strategy,
                    run_id=run_id,
                    mode=args.mode,
                    rounds=args.rounds,
                    run_policy=args.run_policy,
                    resume=args.resume,
                    reset_results=reset_next,
                )
                reset_next = False


if __name__ == "__main__":
    main()
