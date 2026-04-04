#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED_LINKS = [
    (Path(".local_tools"), ROOT / ".local_tools"),
    (Path("data/raw"), ROOT / "data/raw"),
    (Path("data/interim"), ROOT / "data/interim"),
    (Path("data/processed"), ROOT / "data/processed"),
    (Path("artifacts/cache"), ROOT / "artifacts/cache"),
]


def ensure_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"Refusing to replace existing path: {dst}")
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-ref", default="main")
    args = parser.parse_args()

    worktree = args.worktree.expanduser().resolve()
    if not (worktree / ".git").exists():
        subprocess.run(
            ["git", "worktree", "add", "-B", args.branch, str(worktree), args.base_ref],
            cwd=ROOT,
            check=True,
        )
    for rel_dst, src in SHARED_LINKS:
        ensure_symlink(worktree / rel_dst, src)
    print(worktree)


if __name__ == "__main__":
    main()
