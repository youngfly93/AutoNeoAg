from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


def ensure_project_python(root: Path) -> None:
    if os.environ.get("AUTONEOAG_BOOTSTRAPPED") == "1":
        return
    project_toml = root / "project.toml"
    if not project_toml.exists():
        return
    raw = tomllib.loads(project_toml.read_text())
    env_root = Path(raw["environment"]["conda_env_path"]).expanduser()
    env_python = env_root / "bin" / "python"
    if not env_python.exists():
        return
    if Path(sys.executable).resolve() == env_python.resolve():
        return
    os.environ["AUTONEOAG_BOOTSTRAPPED"] = "1"
    os.execv(str(env_python), [str(env_python), *sys.argv])
