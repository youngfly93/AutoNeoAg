#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "figures" / "neoantigen_run_comparison.png"


@dataclass
class RunSpec:
    label: str
    results_path: Path
    report_path: Path | None
    task_id: str
    strategy: str
    run_id: int | None


def _parse_run_arg(raw: str) -> RunSpec:
    parts = raw.split("|")
    if len(parts) < 2:
        raise ValueError(
            "Each --run must be 'label|results_path|report_path|task_id|strategy|run_id'. "
            "report_path, task_id, strategy, run_id are optional."
        )
    label = parts[0].strip()
    results_path = Path(parts[1]).expanduser().resolve()
    report_path = Path(parts[2]).expanduser().resolve() if len(parts) > 2 and parts[2].strip() else None
    task_id = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "neoantigen"
    strategy = parts[4].strip() if len(parts) > 4 and parts[4].strip() else "constrained"
    run_id = int(parts[5]) if len(parts) > 5 and parts[5].strip() else None
    return RunSpec(
        label=label,
        results_path=results_path,
        report_path=report_path,
        task_id=task_id,
        strategy=strategy,
        run_id=run_id,
    )


def _load_results(spec: RunSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.results_path, sep="\t")
    if "task_id" in df.columns:
        df = df[df["task_id"] == spec.task_id]
    if "strategy" in df.columns:
        df = df[df["strategy"] == spec.strategy]
    if spec.run_id is not None and "run_id" in df.columns:
        df = df[pd.to_numeric(df["run_id"], errors="coerce") == spec.run_id]
    if df.empty:
        raise RuntimeError(f"No matching rows found for {spec.label} in {spec.results_path}")
    df = df.copy()
    df["round_id"] = pd.to_numeric(df["round_id"], errors="coerce")
    df["dev_score"] = pd.to_numeric(df["dev_score"], errors="coerce")
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
    else:
        df["status"] = ""
    df = df[df["round_id"].notna()].sort_values("round_id").reset_index(drop=True)
    df["exp_idx"] = range(1, len(df) + 1)
    df["running_best"] = df["dev_score"].cummax()
    return df


def _extract_json_block(text: str, heading: str) -> dict[str, float] | None:
    pattern = rf"## {re.escape(heading)}\s*(\{{.*?\}})"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def _load_report_metrics(spec: RunSpec) -> tuple[float | None, float | None]:
    if not spec.report_path or not spec.report_path.exists():
        return None, None
    text = spec.report_path.read_text()
    confirm = _extract_json_block(text, "Confirm")
    blind = _extract_json_block(text, "Blind")
    confirm_score = float(confirm["val_score"]) if confirm and "val_score" in confirm else None
    blind_score = float(blind["val_score"]) if blind and "val_score" in blind else None
    return confirm_score, blind_score


def _build_summary(specs: list[RunSpec]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for spec in specs:
        df = _load_results(spec)
        best_row = df.loc[df["dev_score"].idxmax()]
        confirm_score, blind_score = _load_report_metrics(spec)
        keep_count = int((df["status"] == "keep").sum())
        summary.append(
            {
                "label": spec.label,
                "rounds": int(len(df)),
                "keep_rate": keep_count / len(df),
                "best_round": int(best_row["round_id"]),
                "best_dev": float(best_row["dev_score"]),
                "confirm": confirm_score,
                "blind": blind_score,
                "baseline": float(df.iloc[0]["dev_score"]),
            }
        )
    return summary


def plot_comparison(specs: list[RunSpec], outpath: Path) -> Path:
    run_dfs = [(spec, _load_results(spec)) for spec in specs]
    summary = _build_summary(specs)
    colors = ["#2f80ed", "#27ae60", "#d35400", "#8e44ad", "#c0392b"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.8], hspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    for idx, (spec, df) in enumerate(run_dfs):
        color = colors[idx % len(colors)]
        discard = df[df["status"] == "discard"]
        keep = df[df["status"] == "keep"]
        ax.scatter(
            discard["exp_idx"],
            discard["dev_score"],
            s=14,
            alpha=0.18,
            color=color,
            zorder=1,
        )
        ax.scatter(
            keep["exp_idx"],
            keep["dev_score"],
            s=42,
            alpha=0.88,
            edgecolors="black",
            linewidths=0.4,
            color=color,
            zorder=3,
        )
        ax.step(
            df["exp_idx"],
            df["running_best"],
            where="post",
            linewidth=2.5,
            color=color,
            zorder=2,
            label=spec.label,
        )
        best_row = df.loc[df["dev_score"].idxmax()]
        ax.annotate(
            f"{spec.label}: r{int(best_row['round_id'])} {float(best_row['dev_score']):.4f}",
            (float(best_row["exp_idx"]), float(best_row["dev_score"])),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.8},
        )

    ax.set_title("Neoantigen Full Run Comparison: dev-score trajectories", fontsize=15)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("dev_score")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    col_labels = ["Run", "Rounds", "Keep Rate", "Best Round", "Best dev", "Confirm", "Blind", "dev-blind gap"]
    cell_text = []
    for row in summary:
        blind = row["blind"]
        cell_text.append(
            [
                str(row["label"]),
                f"{int(row['rounds'])}",
                f"{float(row['keep_rate']):.1%}",
                f"{int(row['best_round'])}",
                f"{float(row['best_dev']):.6f}",
                f"{float(row['confirm']):.6f}" if row["confirm"] is not None else "-",
                f"{float(blind):.6f}" if blind is not None else "-",
                f"{float(row['best_dev']) - float(blind):.6f}" if blind is not None else "-",
            ]
        )
    table = ax2.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    fig.text(
        0.01,
        0.01,
        "Interpretation: run_01 reaches the highest dev peak; run_02 has the best blind score; "
        "run_03 sits in between with near-run_01 dev and much stronger blind than run_01.",
        fontsize=10,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label|results_path|report_path|task_id|strategy|run_id",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = [_parse_run_arg(raw) for raw in args.run]
    outpath = plot_comparison(specs, args.out.resolve())
    print(outpath)


if __name__ == "__main__":
    main()
