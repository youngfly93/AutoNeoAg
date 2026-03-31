from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from autoneoag.config import Settings


def allowed_edit_scope(strategy: str) -> list[str]:
    if strategy == "constrained":
        return ["train.py"]
    if strategy == "unconstrained":
        return ["train.py", "program.md"]
    raise KeyError(f"Unsupported codex strategy: {strategy}")


def run_codex_worker(
    settings: Settings,
    *,
    task_id: str,
    strategy: str,
    round_id: int,
    root: Path,
    summary: str,
) -> dict[str, object]:
    schema = root / "schemas" / "codex_worker_output.schema.json"
    scope = allowed_edit_scope(strategy)
    scope_text = ", ".join(scope)
    prompt = f"""
You are proposing experiment round {round_id} for AutoNeoAg task {task_id}.
Read program.md and train.py, then modify only these files if necessary: {scope_text}.
Do not touch any other file.
Keep the change small and compatible with MPS/CPU.
Prefer higher-level edits in scalar feature blocks, WT-vs-Mut contrast heads, pair/group ranking objectives, and HLA-conditioned interaction structure.
Only use local pooling or fusion tweaks when they support one of those higher-level changes.
Avoid isolated tweaks to loss functions, label smoothing, class weights, sampling, or output-bias initialization unless paired with a clear representation change.

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
