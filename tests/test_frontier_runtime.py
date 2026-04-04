from __future__ import annotations

from pathlib import Path

from autoneoag.runtime.frontier import build_frontier_state, canonical_family
from autoneoag.runtime.results import load_results


def test_results_load_migrates_legacy_header(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    path.write_text(
        "task_id\tstrategy\trun_id\tround_id\tcommit\tdev_score\tconfirm_score\tblind_score\tstatus\tfailure_type\ttraining_seconds\tlines_changed\tdescription\n"
        "neoantigen\tconstrained\t1\t1\tabc123\t0.500000\t\t\tkeep\t\t300.000000\t0\tbaseline\n"
    )
    rows = load_results(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "neoantigen"
    assert row["dev_score"] == "0.500000"
    assert row["run_policy"] == "fast-dev"
    assert row["worker_declared_family"] == ""
    assert row["controller_inferred_family"] == ""
    assert row["confirm_checked"] == "0"


def test_canonical_family_prefers_controller_on_conflict() -> None:
    proposal_family, family_consensus = canonical_family("gating", "preference_contrast")
    assert proposal_family == "preference_contrast"
    assert family_consensus == "controller_override"


def test_frontier_state_enters_recovery_after_long_plateau() -> None:
    rows: list[dict[str, str]] = [
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "1",
            "round_id": "1",
            "commit": "c1",
            "dev_score": "0.500000",
            "status": "keep",
            "description": "baseline",
        },
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "1",
            "round_id": "2",
            "commit": "c2",
            "dev_score": "0.700000",
            "status": "keep",
            "description": "Added preference_contrast_head with WT-vs-Mut preference delta",
            "worker_declared_family": "preference_contrast",
            "proposal_family": "preference_contrast",
        },
    ]
    for round_id in range(3, 13):
        rows.append(
            {
                "task_id": "neoantigen",
                "strategy": "constrained",
                "run_id": "1",
                "round_id": str(round_id),
                "commit": f"c{round_id}",
                "dev_score": "0.620000",
                "status": "discard",
                "description": "Added gating block to modulate preference hidden states",
                "worker_declared_family": "gating",
                "proposal_family": "gating",
            }
        )

    state = build_frontier_state("neoantigen", "constrained", 1, 13, rows)
    assert state["champion"]["round_id"] == 2
    assert state["search_mode"] == "recovery"
    assert "preference_contrast" in state["prioritize"]


def test_frontier_prefers_keep_row_as_champion() -> None:
    rows = [
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "1",
            "round_id": "1",
            "commit": "c1",
            "dev_score": "0.600000",
            "status": "keep",
            "description": "baseline",
        },
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "1",
            "round_id": "2",
            "commit": "c2",
            "dev_score": "0.900000",
            "status": "discard",
            "description": "discarded dev spike",
            "proposal_family": "gating",
        },
    ]

    state = build_frontier_state("neoantigen", "constrained", 1, 3, rows)
    assert state["champion"]["round_id"] == 1


def test_strict_confirm_frontier_tracks_shadow_champion() -> None:
    rows = [
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "11",
            "run_policy": "strict-confirm",
            "round_id": "1",
            "commit": "c1",
            "dev_score": "0.590922",
            "confirm_round_score": "0.387634",
            "confirm_checked": "1",
            "confirm_survival": "1",
            "status": "keep",
            "proposal_family": "other",
            "proposal_subfamily": "baseline",
            "description": "baseline",
        },
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "11",
            "run_policy": "strict-confirm",
            "round_id": "3",
            "commit": "c3",
            "dev_score": "0.688472",
            "confirm_round_score": "0.377265",
            "confirm_checked": "1",
            "confirm_survival": "0",
            "status": "discard",
            "decision_reason": "confirm_failed_gate",
            "proposal_family": "gating",
            "proposal_subfamily": "contrast_scalar",
            "description": "near-miss confirm failure",
        },
        {
            "task_id": "neoantigen",
            "strategy": "constrained",
            "run_id": "11",
            "run_policy": "strict-confirm",
            "round_id": "4",
            "commit": "c4",
            "dev_score": "0.739307",
            "confirm_round_score": "0.315994",
            "confirm_checked": "1",
            "confirm_survival": "0",
            "status": "discard",
            "decision_reason": "confirm_failed_gate",
            "proposal_family": "interaction_balance",
            "proposal_subfamily": "contrast_scalar",
            "description": "weaker confirm miss",
        },
    ]

    state = build_frontier_state("neoantigen", "constrained", 11, 5, rows)
    assert state["champion"]["round_id"] == 1
    assert state["shadow_champion"]["round_id"] == 3
    assert "gating" in state["near_miss_families"]

    family_stats = {row["proposal_family"]: row for row in state["family_stats"]}
    assert family_stats["gating"]["best_confirm_score"] == 0.377265
    assert family_stats["gating"]["near_miss_count"] == 1
