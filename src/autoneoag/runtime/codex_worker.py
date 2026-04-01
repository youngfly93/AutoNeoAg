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
    frontier_hint: str = "",
    frontier_state: dict[str, object] | None = None,
) -> dict[str, object]:
    schema = root / "schemas" / "codex_worker_output.schema.json"
    scope = allowed_edit_scope(strategy)
    scope_text = ", ".join(scope)
    champion = (frontier_state or {}).get("champion", {})
    parent_round_id = champion.get("round_id")
    search_mode = (frontier_state or {}).get("search_mode", "exploit")
    prompt = f"""
You are proposing experiment round {round_id} for AutoNeoAg task {task_id}.
Read program.md and train.py, then modify only these files if necessary: {scope_text}.
Do not touch any other file.
Keep the change small and compatible with MPS/CPU.
Only make one small structural change.
Prefer refining the current winning family over introducing a new branch.
Avoid repeated failure patterns listed below.
If you choose a new family, justify why the current frontier warrants exploration.
Do not disguise a cross-family change as a local refinement.

Current parent_round_id: {parent_round_id}
Expected search_mode: {search_mode}

Frontier hint:
{frontier_hint or "(no frontier hint available)"}

Recent summary:
{summary}

Return JSON matching the provided schema.
Set worker_declared_family/subfamily to your own classification.
Set proposal_family/subfamily equal to your declared family/subfamily.
Set parent_round_id to the champion round shown above when applicable.
Set search_mode to the expected search_mode unless you have a strong reason to switch.
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
