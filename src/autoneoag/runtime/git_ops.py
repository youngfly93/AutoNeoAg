from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str], root: Path) -> str:
    return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def has_commits(root: Path) -> bool:
    try:
        _run(["git", "rev-parse", "HEAD"], root)
        return True
    except subprocess.CalledProcessError:
        return False


def current_commit(root: Path) -> str:
    return _run(["git", "rev-parse", "--short", "HEAD"], root)


def current_branch(root: Path) -> str:
    return _run(["git", "branch", "--show-current"], root)


def ensure_branch(root: Path, name: str) -> None:
    _run(["git", "checkout", "-B", name], root)


def checkout_branch(root: Path, name: str) -> None:
    _run(["git", "checkout", name], root)


def commit_all(root: Path, message: str) -> str:
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-m", message], root)
    return current_commit(root)


def reset_hard(root: Path, commit: str) -> None:
    _run(["git", "reset", "--hard", commit], root)


def changed_files(root: Path) -> list[str]:
    out = _run(["git", "diff", "--name-only"], root)
    return [line for line in out.splitlines() if line]
