from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ranking_objective", ("pairwise", "hybrid_pairwise", "pairwise_only", "ranking objective", "rank_context")),
    ("gating", (" gate", "gating", "sigmoid gate", "tanh gate", "conditioner")),
    (
        "preference_contrast",
        ("preference_contrast", "joint preference", "preference delta", "preference_pair", "preference bridge"),
    ),
    (
        "preference_context",
        ("shared competition embedding", "preference inputs", "mut and wt preference", "preference context"),
    ),
    ("scalar_contrast", ("scalar_contrast", "contrast scalars", "comparison/context scalar", "scalar block", "scalar tower")),
    ("interaction_balance", ("mut_hla", "wt_hla", "average_peptide * hla", "interaction", "competition term", "support delta")),
    ("auxiliary_head", ("auxiliary head", "auxiliary logit", "logit_weight")),
    ("fusion_path", ("final conditioning", "residual into contrast_hidden", "fusion", "affine-modulate", "residual path")),
)


def _as_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _as_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _slugify(text: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    text = "_".join(part for part in text.split("_") if part)
    return text or "other"


def shorten(text: str, limit: int = 96) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def infer_family_from_text(*parts: str) -> str:
    haystack = " ".join(part for part in parts if part).lower()
    for family, patterns in FAMILY_RULES:
        if any(pattern in haystack for pattern in patterns):
            return family
    return "other"


def infer_subfamily_from_text(family: str, *parts: str) -> str:
    haystack = " ".join(part for part in parts if part).lower()
    markers = [
        "preference_contrast_head",
        "preference_context",
        "scalar_contrast_head",
        "contrast_scalar",
        "pair_scalar",
        "support_delta",
        "interaction_balance",
        "contrast_agreement",
        "final_conditioning",
        "auxiliary_logit",
        "gate",
    ]
    for marker in markers:
        if marker in haystack:
            return marker
    return family


def canonical_family(worker_family: str, controller_family: str) -> tuple[str, str]:
    worker = worker_family or ""
    controller = controller_family or ""
    if worker and controller:
        if worker == controller:
            return worker, "agreed"
        if controller != "other":
            return controller, "controller_override"
        return worker, "worker_only"
    if controller:
        return controller, "controller_only"
    if worker:
        return worker, "worker_only"
    return "uncertain", "uncertain"


def filter_run_rows(rows: list[dict[str, str]], task_id: str, strategy: str, run_id: int) -> list[dict[str, str]]:
    subset = [
        dict(row)
        for row in rows
        if row.get("task_id") == task_id and row.get("strategy") == strategy and _as_int(row.get("run_id")) == run_id
    ]
    subset.sort(key=lambda row: _as_int(row.get("round_id")) or 0)
    return subset


def annotate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    best_so_far = float("-inf")
    for row in rows:
        annotated_row: dict[str, object] = dict(row)
        round_id = _as_int(row.get("round_id")) or 0
        dev_score = _as_float(row.get("dev_score"))
        worker_family = row.get("worker_declared_family", "")
        controller_family = row.get("controller_inferred_family", "") or infer_family_from_text(
            row.get("description", ""),
            row.get("hypothesis", ""),
            row.get("expected_change", ""),
        )
        proposal_family = row.get("proposal_family", "")
        if not proposal_family:
            proposal_family, family_consensus = canonical_family(worker_family, controller_family)
        else:
            _, family_consensus = canonical_family(worker_family, controller_family)
        proposal_subfamily = row.get("proposal_subfamily", "") or infer_subfamily_from_text(
            proposal_family,
            row.get("description", ""),
            row.get("hypothesis", ""),
            row.get("expected_change", ""),
        )
        delta_vs_best = _as_float(row.get("delta_vs_best"))
        if delta_vs_best is None:
            if dev_score is None:
                delta_vs_best = None
            elif best_so_far == float("-inf"):
                delta_vs_best = 0.0
            else:
                delta_vs_best = dev_score - best_so_far
        if dev_score is not None:
            best_so_far = max(best_so_far, dev_score)
        annotated_row.update(
            {
                "round_id": round_id,
                "dev_score": dev_score,
                "proposal_family": proposal_family,
                "proposal_subfamily": proposal_subfamily,
                "worker_declared_family": worker_family,
                "controller_inferred_family": controller_family,
                "family_consensus": row.get("family_consensus", "") or family_consensus,
                "delta_vs_best": delta_vs_best,
                "status": row.get("status", ""),
                "failure_type": row.get("failure_type", ""),
                "confirm_survival": row.get("confirm_survival", ""),
            }
        )
        annotated.append(annotated_row)
    return annotated


def build_family_stats(annotated_rows: list[dict[str, object]], current_round: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in annotated_rows:
        grouped[str(row.get("proposal_family") or "uncertain")].append(row)

    family_stats: list[dict[str, object]] = []
    for family, rows in grouped.items():
        scored = [row for row in rows if row.get("dev_score") is not None]
        deltas = [float(row["delta_vs_best"]) for row in scored if row.get("delta_vs_best") is not None]
        keeps = [row for row in rows if row.get("status") == "keep"]
        confirm_checked = [row for row in rows if str(row.get("confirm_checked", "")) in {"1", "true", "True"}]
        confirm_survived = [row for row in confirm_checked if str(row.get("confirm_survival", "")) in {"1", "true", "True"}]
        consecutive_regressions = 0
        for row in reversed(scored):
            delta = row.get("delta_vs_best")
            if delta is None or float(delta) > -0.03:
                break
            consecutive_regressions += 1
        frozen_until_round = ""
        if consecutive_regressions >= 3:
            frozen_until_round = str(current_round + 5)
        recent = scored[-5:]
        recent_mean = sum(float(row["delta_vs_best"]) for row in recent if row.get("delta_vs_best") is not None) / max(len(recent), 1)
        family_stats.append(
            {
                "proposal_family": family,
                "attempts": len(rows),
                "keeps": len(keeps),
                "mean_delta_vs_best": (sum(deltas) / len(deltas)) if deltas else None,
                "best_gain": max(deltas) if deltas else None,
                "last_gain_round": max((int(row["round_id"]) for row in keeps), default=None),
                "recent_5_mean": recent_mean if recent else None,
                "consecutive_regressions": consecutive_regressions,
                "confirm_checks": len(confirm_checked),
                "confirm_promotions": len(confirm_survived),
                "confirm_survival_rate": (len(confirm_survived) / len(confirm_checked)) if confirm_checked else None,
                "frozen_until_round": frozen_until_round,
            }
        )
    family_stats.sort(key=lambda row: (row["attempts"], row.get("best_gain") or -999.0), reverse=True)
    return family_stats


def choose_search_mode(annotated_rows: list[dict[str, object]], champion_round: int, champion_family: str, family_stats: list[dict[str, object]], current_round: int) -> str:
    recent_failures = annotated_rows[-2:]
    if recent_failures and all(row.get("failure_type") in {"train_failed", "worker_failed"} for row in recent_failures):
        return "exploit"
    rounds_since_best = current_round - champion_round
    champion_stats = next((row for row in family_stats if row["proposal_family"] == champion_family), None)
    if champion_stats and champion_stats["frozen_until_round"]:
        frozen_until = int(champion_stats["frozen_until_round"])
        if current_round <= frozen_until:
            return "explore"
    if rounds_since_best >= 10:
        return "recovery"
    return "exploit"


def build_frontier_state(task_id: str, strategy: str, run_id: int, current_round: int, rows: list[dict[str, str]]) -> dict[str, object]:
    annotated = annotate_rows(filter_run_rows(rows, task_id, strategy, run_id))
    if not annotated:
        champion = {
            "round_id": 1,
            "commit": "",
            "dev_score": None,
            "proposal_family": "uncertain",
            "proposal_subfamily": "uncertain",
        }
        family_stats: list[dict[str, object]] = []
        search_mode = "exploit"
    else:
        scored = [row for row in annotated if row.get("dev_score") is not None]
        best_row = max(scored, key=lambda row: float(row["dev_score"])) if scored else annotated[0]
        champion = {
            "round_id": int(best_row["round_id"]),
            "commit": best_row.get("commit", ""),
            "dev_score": best_row.get("dev_score"),
            "proposal_family": best_row.get("proposal_family", "uncertain"),
            "proposal_subfamily": best_row.get("proposal_subfamily", "uncertain"),
        }
        family_stats = build_family_stats(annotated, current_round)
        search_mode = choose_search_mode(
            annotated_rows=annotated,
            champion_round=int(champion["round_id"]),
            champion_family=str(champion["proposal_family"]),
            family_stats=family_stats,
            current_round=current_round,
        )

    avoid = [
        row["proposal_family"]
        for row in family_stats
        if row["frozen_until_round"]
        or (row["keeps"] == 0 and row["attempts"] >= 3 and (row["mean_delta_vs_best"] or 0.0) < -0.05)
    ]
    scored_families = []
    for row in family_stats:
        if row["proposal_family"] in avoid:
            continue
        score = (row["keeps"] * 4) + ((row["best_gain"] or 0.0) * 10) + ((row["confirm_survival_rate"] or 0.0) * 2) - row["consecutive_regressions"]
        scored_families.append((score, row["proposal_family"]))
    scored_families.sort(reverse=True)
    prioritize = [family for _, family in scored_families[:3]] or ["preference_context", "preference_contrast", "fusion_path"]
    recent_fail_patterns = [
        f"{row.get('proposal_family','uncertain')}: {shorten(str(row.get('description','')))}"
        for row in annotated[-5:]
        if row.get("status") == "discard"
    ]

    return {
        "task_id": task_id,
        "strategy": strategy,
        "run_id": run_id,
        "current_round": current_round,
        "champion": champion,
        "search_mode": search_mode,
        "confirm_feedback": {
            "enabled": True,
            "policy": "every_3_keeps_or_new_best",
            "last_checked_round": max(
                (
                    int(row["round_id"])
                    for row in annotated
                    if str(row.get("confirm_checked", "")) in {"1", "true", "True"}
                ),
                default=None,
            ),
        },
        "family_stats": family_stats,
        "prioritize": prioritize,
        "avoid": avoid,
        "recent_fail_patterns": recent_fail_patterns,
    }


def render_frontier_hint(state: dict[str, object]) -> str:
    champion = state["champion"]
    priority_lines = "\n".join(f"- {item}" for item in state.get("prioritize", []))
    avoid_lines = "\n".join(f"- {item}" for item in state.get("avoid", [])) or "- none"
    fail_lines = "\n".join(f"- {item}" for item in state.get("recent_fail_patterns", [])[:5]) or "- none"
    return (
        "# Frontier Hint\n\n"
        f"Current champion: round {champion['round_id']}, family {champion['proposal_family']}, "
        f"dev_score {champion['dev_score']}.\n\n"
        "Successful direction:\n"
        f"{priority_lines}\n\n"
        "Recent failure patterns:\n"
        f"{fail_lines}\n\n"
        f"Expected search_mode: {state['search_mode']}\n\n"
        "Avoid:\n"
        f"{avoid_lines}\n"
    )


def write_frontier_artifacts(logs_dir: Path, state: dict[str, object]) -> tuple[Path, Path, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / "frontier_state.json"
    hint_path = logs_dir / "frontier_hint.md"
    stats_path = logs_dir / "family_stats.tsv"

    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    hint_path.write_text(render_frontier_hint(state))
    with stats_path.open("w", newline="") as handle:
        fieldnames = [
            "proposal_family",
            "attempts",
            "keeps",
            "mean_delta_vs_best",
            "best_gain",
            "last_gain_round",
            "recent_5_mean",
            "consecutive_regressions",
            "confirm_checks",
            "confirm_promotions",
            "confirm_survival_rate",
            "frozen_until_round",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in state.get("family_stats", []):
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return state_path, hint_path, stats_path
