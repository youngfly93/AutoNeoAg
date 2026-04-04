#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _as_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _extract_json_block(text: str, heading: str) -> dict[str, float] | None:
    match = re.search(rf"## {re.escape(heading)}\s*(\{{.*?\}})", text, flags=re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def _load_report_metrics(report_path: Path) -> tuple[float | None, float | None]:
    if not report_path.exists():
        return None, None
    text = report_path.read_text()
    confirm = _extract_json_block(text, "Confirm")
    blind = _extract_json_block(text, "Blind")
    return (
        float(confirm["val_score"]) if confirm and "val_score" in confirm else None,
        float(blind["val_score"]) if blind and "val_score" in blind else None,
    )


def summarize_results(
    *,
    results_path: Path,
    tasks: list[str] | None,
    strategies: list[str] | None,
    run_ids: list[int] | None,
) -> list[dict[str, object]]:
    with results_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    selected = [row for row in rows if row.get("run_policy", "fast-dev") == "strict-confirm"]
    if tasks:
        selected = [row for row in selected if row.get("task_id") in set(tasks)]
    if strategies:
        selected = [row for row in selected if row.get("strategy") in set(strategies)]
    if run_ids:
        selected = [row for row in selected if int(row.get("run_id", "0")) in set(run_ids)]

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in selected:
        key = (row["task_id"], row["strategy"], int(row["run_id"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (task_id, strategy, run_id), sub in sorted(grouped.items()):
        sub.sort(key=lambda row: int(row["round_id"]))
        rounds = len(sub)
        keep_rows = [row for row in sub if row.get("status") == "keep"]
        best_keep = max(keep_rows, key=lambda row: float(row.get("dev_score") or "-inf")) if keep_rows else None
        best_any = max(sub, key=lambda row: float(row.get("dev_score") or "-inf"))
        confirm_required_count = sum(1 for row in sub if row.get("confirm_gate_required") in {"1", "true", "True"})
        confirm_pass_count = sum(1 for row in sub if row.get("confirm_gate_passed") in {"1", "true", "True"})
        report_path = results_path.parent / "artifacts" / "logs" / task_id / strategy / f"run_{run_id:02d}" / "report.md"
        report_confirm, report_blind = _load_report_metrics(report_path)
        best_keep_dev = _as_float(best_keep.get("dev_score")) if best_keep else None
        best_keep_confirm = _as_float(best_keep.get("confirm_round_score") or best_keep.get("confirm_score")) if best_keep else None
        best_any_dev = _as_float(best_any.get("dev_score"))

        summary_rows.append(
            {
                "task_id": task_id,
                "strategy": strategy,
                "run_id": run_id,
                "rounds": rounds,
                "keep_count": len(keep_rows),
                "keep_rate": (len(keep_rows) / rounds) if rounds else None,
                "confirm_gate_required_count": confirm_required_count,
                "confirm_gate_pass_count": confirm_pass_count,
                "confirm_gate_pass_rate": (confirm_pass_count / confirm_required_count) if confirm_required_count else None,
                "best_any_round": int(best_any["round_id"]),
                "best_any_dev": best_any_dev,
                "best_keep_round": int(best_keep["round_id"]) if best_keep else None,
                "best_keep_dev": best_keep_dev,
                "best_keep_confirm_round": best_keep_confirm,
                "report_confirm": report_confirm,
                "report_blind": report_blind,
                "dev_confirm_gap": (best_keep_dev - report_confirm) if best_keep_dev is not None and report_confirm is not None else None,
                "dev_blind_gap": (best_keep_dev - report_blind) if best_keep_dev is not None and report_blind is not None else None,
                "best_any_dev_confirm_gap": (best_any_dev - report_confirm) if best_any_dev is not None and report_confirm is not None else None,
                "best_any_dev_blind_gap": (best_any_dev - report_blind) if best_any_dev is not None and report_blind is not None else None,
                "winner_changed_by_strict_confirm": int(bool(best_keep and int(best_any["round_id"]) != int(best_keep["round_id"]))),
                "results_path": str(results_path),
                "report_path": str(report_path),
            }
        )
    return summary_rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No strict-confirm rows matched the requested filters.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Strict-Confirm Campaign Summary",
        "",
        "| Task | Strategy | Run | Keep Rate | Confirm Pass Rate | Best Keep dev | Confirm | Blind | Winner Changed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {strategy} | {run_id} | {keep_rate:.1%} | {confirm_gate_pass_rate:.1%} | {best_keep_dev:.6f} | {report_confirm:.6f} | {report_blind:.6f} | {winner_changed_by_strict_confirm} |".format(
                task_id=row["task_id"],
                strategy=row["strategy"],
                run_id=row["run_id"],
                keep_rate=row["keep_rate"] or 0.0,
                confirm_gate_pass_rate=row["confirm_gate_pass_rate"] or 0.0,
                best_keep_dev=row["best_keep_dev"] or 0.0,
                report_confirm=row["report_confirm"] or 0.0,
                report_blind=row["report_blind"] or 0.0,
                winner_changed_by_strict_confirm=row["winner_changed_by_strict_confirm"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--run-ids", nargs="*", type=int)
    parser.add_argument("--out-tsv", type=Path, default=ROOT / "artifacts" / "analysis" / "strict_confirm_summary.tsv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "artifacts" / "analysis" / "strict_confirm_summary.md")
    args = parser.parse_args()

    rows = summarize_results(
        results_path=args.results.expanduser().resolve(),
        tasks=args.tasks,
        strategies=args.strategies,
        run_ids=args.run_ids,
    )
    write_tsv(args.out_tsv.resolve(), rows)
    write_markdown(args.out_md.resolve(), rows)
    print(args.out_tsv.resolve())
    print(args.out_md.resolve())


if __name__ == "__main__":
    main()
