from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str], root: Path) -> str:
    return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _check(args: list[str], root: Path) -> bool:
    return subprocess.run(args, cwd=root, capture_output=True, text=True).returncode == 0


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
    if _check(["git", "rev-parse", "--verify", name], root):
        _run(["git", "checkout", name], root)
    else:
        _run(["git", "checkout", "-b", name], root)


def checkout_branch(root: Path, name: str) -> None:
    _run(["git", "checkout", name], root)


def commit_all(root: Path, message: str) -> str:
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-m", message], root)
    return current_commit(root)


def commit_paths(root: Path, paths: list[str], message: str) -> str:
    if not paths:
        raise RuntimeError("commit_paths requires at least one path.")
    _run(["git", "add", "--", *paths], root)
    _run(["git", "commit", "-m", message], root)
    return current_commit(root)


def reset_hard(root: Path, commit: str) -> None:
    _run(["git", "reset", "--hard", commit], root)


def changed_files(root: Path, paths: list[str] | None = None) -> list[str]:
    args = ["git", "diff", "--name-only"]
    if paths:
        args.extend(["--", *paths])
    out = _run(args, root)
    return [line for line in out.splitlines() if line]


def changed_line_count(root: Path, paths: list[str] | None = None) -> int:
    args = ["git", "diff", "--numstat"]
    if paths:
        args.extend(["--", *paths])
    out = _run(args, root)
    total = 0
    for line in out.splitlines():
        if not line.strip():
            continue
        added, deleted, _path = line.split("\t", 2)
        if added.isdigit():
            total += int(added)
        if deleted.isdigit():
            total += int(deleted)
    return total
