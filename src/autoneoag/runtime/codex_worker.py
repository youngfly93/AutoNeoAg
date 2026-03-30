from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from autoneoag.config import Settings


def run_codex_worker(settings: Settings, round_id: int, root: Path, summary: str) -> dict[str, object]:
    schema = root / "schemas" / "codex_worker_output.schema.json"
    prompt = f"""
You are proposing experiment round {round_id} for AutoNeoAg.
Read program.md and train.py, then modify only train.py.
Do not touch any other file.
Keep the change small and compatible with MPS/CPU.

Recent summary:
{summary}

Return JSON matching the provided schema.
""".strip()
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)
    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--full-auto",
        "-C",
        str(root),
        "-c",
        f'model_reasoning_effort="{settings.reasoning_effort}"',
        "--output-schema",
        str(schema),
        "-o",
        str(output_path),
        prompt,
    ]
    try:
        completed = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "codex exec failed\n"
            f"exit_code: {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    _ = completed.stdout
    return json.loads(output_path.read_text())
