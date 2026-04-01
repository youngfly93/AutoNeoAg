from __future__ import annotations

import subprocess
from pathlib import Path

from autoneoag.runtime.git_ops import reset_hard_preserving


def run(root: Path, *args: str) -> str:
    completed = subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def test_reset_hard_preserving_keeps_results_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run(root, "git", "init")
    run(root, "git", "config", "user.email", "test@example.com")
    run(root, "git", "config", "user.name", "Test User")
    (root / "train.py").write_text("print('v1')\n")
    (root / "results.tsv").write_text("header\nrow1\n")
    run(root, "git", "add", "train.py", "results.tsv")
    run(root, "git", "commit", "-m", "init")
    commit = run(root, "git", "rev-parse", "--short", "HEAD")

    (root / "train.py").write_text("print('v2')\n")
    (root / "results.tsv").write_text("header\nrow1\nrow2\n")

    reset_hard_preserving(root, commit, [root / "results.tsv"])

    assert (root / "train.py").read_text() == "print('v1')\n"
    assert (root / "results.tsv").read_text() == "header\nrow1\nrow2\n"
