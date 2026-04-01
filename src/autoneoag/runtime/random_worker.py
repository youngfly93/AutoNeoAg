from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from autoneoag.runtime.frontier import infer_family_from_text


@dataclass(frozen=True)
class MutationSpec:
    field_name: str
    pattern: str
    choices: tuple[str, ...]
    summary_template: str
    hypothesis: str


MUTATIONS = (
    MutationSpec(
        field_name="feature_blocks",
        pattern=r'(feature_blocks: tuple\[str, \.\.\.\] = )([^\n]+)',
        choices=('("base", "comparison")', '("base", "context")', '("base", "comparison", "context")'),
        summary_template="randomize feature block composition -> {value}",
        hypothesis="Different scalar block mixes may change how much pairwise signal reaches the contrast path.",
    ),
    MutationSpec(
        field_name="objective_mode",
        pattern=r'(objective_mode: str = )([^\n]+)',
        choices=('"bce"', '"hybrid_pairwise"', '"pairwise_only"'),
        summary_template="randomize objective mode -> {value}",
        hypothesis="Alternative ranking objectives may outperform plain BCE under grouped evaluation.",
    ),
    MutationSpec(
        field_name="pairwise_weight",
        pattern=r'(pairwise_weight: float = )([^\n]+)',
        choices=("0.15", "0.25", "0.35"),
        summary_template="randomize pairwise loss weight -> {value}",
        hypothesis="Ranking pressure may need a different pairwise mixing weight to help OOF selection.",
    ),
    MutationSpec(
        field_name="contrast_logit_weight",
        pattern=r'(contrast_logit_weight: float = )([^\n]+)',
        choices=("0.20", "0.30", "0.40"),
        summary_template="randomize contrast logit weight -> {value}",
        hypothesis="The direct contrast branch may be under- or over-weighted relative to the fusion head.",
    ),
    MutationSpec(
        field_name="preference_logit_weight",
        pattern=r'(preference_logit_weight: float = )([^\n]+)',
        choices=("0.05", "0.15", "0.25"),
        summary_template="randomize allele preference weight -> {value}",
        hypothesis="The preference delta branch may need stronger or weaker influence on the final score.",
    ),
)


def _apply_mutation(text: str, spec: MutationSpec, rng: random.Random) -> tuple[str, str] | None:
    match = re.search(spec.pattern, text)
    if match is None:
        return None
    current = match.group(2).strip()
    candidates = [choice for choice in spec.choices if choice != current]
    if not candidates:
        return None
    choice = rng.choice(candidates)
    updated, count = re.subn(spec.pattern, lambda match: f"{match.group(1)}{choice}", text, count=1)
    if count != 1:
        return None
    return updated, choice


def run_random_worker(round_id: int, root: Path, frontier_state: dict[str, object] | None = None) -> dict[str, object]:
    rng = random.Random(101 + round_id)
    train_path = root / "train.py"
    original = train_path.read_text()
    specs = list(MUTATIONS)
    rng.shuffle(specs)
    for spec in specs:
        applied = _apply_mutation(original, spec, rng)
        if applied is None:
            continue
        updated, value = applied
        train_path.write_text(updated)
        worker_family = infer_family_from_text(spec.field_name, spec.hypothesis, spec.summary_template)
        parent_round_id = None
        if frontier_state is not None:
            champion = frontier_state.get("champion", {})
            parent_round_id = champion.get("round_id")
        return {
            "hypothesis": spec.hypothesis,
            "expected_change": f"Sample a random mutation on {spec.field_name}.",
            "risk": "Random search may degrade grouped-CV performance and produce non-improving proposals.",
            "edit_scope": ["train.py"],
            "summary": spec.summary_template.format(value=value),
            "worker_declared_family": worker_family,
            "worker_declared_subfamily": spec.field_name,
            "proposal_family": worker_family,
            "proposal_subfamily": spec.field_name,
            "parent_round_id": parent_round_id,
            "search_mode": frontier_state.get("search_mode", "explore") if frontier_state is not None else "explore",
            "novelty_level": "medium",
        }
    raise RuntimeError("Random worker could not apply any supported mutation to train.py.")
